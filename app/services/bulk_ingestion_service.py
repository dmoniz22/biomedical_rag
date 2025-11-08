"""
Bulk Ingestion Service for biomedical literature
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import hashlib

from app.services.pubmed_service import pubmed_service
from app.services.rag_service import rag_service
from app.models.database import Paper, Author, IngestionJob
from app.core.database import get_database_session
from app.models.schemas import BulkIngestionRequest, IngestionJobCreate, IngestionJob
from app.core.config import settings

logger = logging.getLogger(__name__)


class BulkIngestionService:
    """Service for bulk ingestion of biomedical literature"""
    
    def __init__(self):
        self.pubmed_service = pubmed_service
        self.rag_service = rag_service
        self.active_jobs = {}
    
    async def initialize(self):
        """Initialize bulk ingestion service"""
        await self.pubmed_service.initialize()
        await self.rag_service.initialize()
        logger.info("Bulk ingestion service initialized")
    
    async def start_bulk_ingestion(self, request: BulkIngestionRequest, 
                                 job_name: Optional[str] = None) -> str:
        """Start a new bulk ingestion job"""
        try:
            # Generate job ID and name
            job_id = str(uuid.uuid4())
            job_name = job_name or f"Bulk Ingestion {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Create job record
            async with get_database_session() as session:
                job = IngestionJob(
                    id=job_id,
                    job_name=job_name,
                    job_type="bulk_import",
                    source=request.source_database,
                    target_subject_areas=request.subject_areas,
                    status="pending",
                    created_at=datetime.now()
                )
                
                # Add parameters
                job.parameters = {
                    "max_documents": request.max_documents,
                    "date_range_start": request.date_range_start.isoformat() if request.date_range_start else None,
                    "date_range_end": request.date_range_end.isoformat() if request.date_range_end else None,
                    "include_full_text": request.include_full_text,
                    "quality_threshold": request.quality_threshold
                }
                
                session.add(job)
                await session.commit()
            
            # Store job in active jobs
            self.active_jobs[job_id] = job
            
            # Start ingestion in background
            asyncio.create_task(self._execute_bulk_ingestion(job_id, request))
            
            logger.info(f"Started bulk ingestion job {job_id}: {job_name}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to start bulk ingestion: {e}")
            raise
    
    async def _execute_bulk_ingestion(self, job_id: str, request: BulkIngestionRequest):
        """Execute the bulk ingestion process"""
        start_time = datetime.now()
        job = self.active_jobs.get(job_id)
        
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        try:
            # Update job status to running
            await self._update_job_status(job_id, "running", started_at=start_time)
            
            # Initialize paper counters
            total_processed = 0
            successful_papers = 0
            failed_papers = 0
            duplicate_papers = 0
            errors = []
            
            # Fetch papers from source
            if request.source_database == "pubmed":
                papers_data = await self.pubmed_service.bulk_search_and_fetch(request)
            else:
                raise ValueError(f"Unsupported source database: {request.source_database}")
            
            # Update total documents count
            job.total_documents = len(papers_data)
            await self._update_job_progress(job_id, 0, total_processed, 
                                          successful_papers, failed_papers, duplicate_papers)
            
            # Process papers in batches
            batch_size = settings.BATCH_SIZE
            for i in range(0, len(papers_data), batch_size):
                batch = papers_data[i:i + batch_size]
                
                # Process batch
                batch_results = await self._process_paper_batch(batch, request.quality_threshold)
                
                # Update counters
                total_processed += len(batch)
                successful_papers += batch_results['successful']
                failed_papers += batch_results['failed']
                duplicate_papers += batch_results['duplicates']
                errors.extend(batch_results['errors'])
                
                # Update progress
                progress_percentage = (total_processed / len(papers_data)) * 100
                await self._update_job_progress(
                    job_id, progress_percentage, total_processed,
                    successful_papers, failed_papers, duplicate_papers, errors
                )
                
                # Add delay to respect rate limits
                await asyncio.sleep(0.5)
            
            # Update job to completed
            completion_time = datetime.now()
            job_summary = {
                "total_processed": total_processed,
                "successful": successful_papers,
                "failed": failed_papers,
                "duplicates": duplicate_papers,
                "success_rate": (successful_papers / total_processed) * 100 if total_processed > 0 else 0,
                "processing_time_minutes": (completion_time - start_time).total_seconds() / 60,
                "average_papers_per_minute": total_processed / ((completion_time - start_time).total_seconds() / 60) if (completion_time - start_time).total_seconds() > 0 else 0
            }
            
            await self._update_job_completion(job_id, "completed", completion_time, job_summary, errors)
            
            logger.info(f"Bulk ingestion job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Bulk ingestion job {job_id} failed: {e}")
            await self._update_job_completion(job_id, "failed", datetime.now(), errors=[str(e)])
        
        finally:
            # Remove from active jobs
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
    
    async def _process_paper_batch(self, batch: List[Dict[str, Any]], 
                                 quality_threshold: float) -> Dict[str, Any]:
        """Process a batch of papers"""
        results = {
            'successful': 0,
            'failed': 0,
            'duplicates': 0,
            'errors': []
        }
        
        async with get_database_session() as session:
            for paper_data in batch:
                try:
                    # Check for duplicates
                    is_duplicate = await self._check_duplicate(session, paper_data)
                    
                    if is_duplicate:
                        results['duplicates'] += 1
                        continue
                    
                    # Create paper record
                    paper = await self._create_paper_record(session, paper_data)
                    
                    if paper.quality_score >= quality_threshold:
                        # Process for RAG
                        await self.rag_service.process_paper_for_rag(paper.id, paper_data)
                        results['successful'] += 1
                    else:
                        # Mark as failed due to low quality
                        paper.processing_status = "failed"
                        results['failed'] += 1
                    
                    await session.commit()
                    
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Paper {paper_data.get('pmid', 'unknown')}: {str(e)}")
                    logger.error(f"Error processing paper: {e}")
        
        return results
    
    async def _check_duplicate(self, session, paper_data: Dict[str, Any]) -> bool:
        """Check if paper already exists (duplicate detection)"""
        try:
            # Check by PMID
            if paper_data.get('pmid'):
                existing = await session.execute(
                    "SELECT id FROM papers WHERE pmid = :pmid",
                    {"pmid": paper_data['pmid']}
                )
                if existing.fetchone():
                    return True
            
            # Check by DOI
            if paper_data.get('doi'):
                existing = await session.execute(
                    "SELECT id FROM papers WHERE doi = :doi",
                    {"doi": paper_data['doi']}
                )
                if existing.fetchone():
                    return True
            
            # Check by title hash (fallback)
            title_hash = hashlib.md5(paper_data.get('title', '').lower().encode()).hexdigest()
            existing = await session.execute(
                "SELECT id FROM papers WHERE SUBSTRING(MD5(LOWER(title)), 1, 8) = :title_hash",
                {"title_hash": title_hash[:8]}
            )
            return bool(existing.fetchone())
            
        except Exception as e:
            logger.error(f"Error checking for duplicates: {e}")
            return False
    
    async def _create_paper_record(self, session, paper_data: Dict[str, Any]) -> Paper:
        """Create paper record in database"""
        try:
            # Create paper
            paper = Paper(
                pmid=paper_data.get('pmid'),
                title=paper_data.get('title', ''),
                abstract=paper_data.get('abstract'),
                journal=paper_data.get('journal'),
                publication_date=paper_data.get('publication_date'),
                doi=paper_data.get('doi'),
                publication_type=paper_data.get('publication_type'),
                keywords=paper_data.get('keywords', []),
                mesh_terms=paper_data.get('mesh_terms', []),
                subject_areas=paper_data.get('subject_areas', []),
                source_database=paper_data.get('source_database', 'pubmed'),
                processing_status="processing",
                quality_score=0.7  # Default quality score
            )
            
            session.add(paper)
            await session.flush()  # Get paper ID
            
            # Create authors
            if paper_data.get('authors'):
                await self._create_authors(session, paper.id, paper_data['authors'])
            
            return paper
            
        except Exception as e:
            logger.error(f"Error creating paper record: {e}")
            raise
    
    async def _create_authors(self, session, paper_id: str, authors: List[str]):
        """Create author records and link to paper"""
        try:
            for i, author_name in enumerate(authors):
                # Parse author name
                name_parts = author_name.strip().split()
                if len(name_parts) >= 2:
                    last_name = name_parts[-1]
                    first_name = " ".join(name_parts[:-1])
                else:
                    last_name = author_name
                    first_name = ""
                
                # Check if author exists
                result = await session.execute(
                    "SELECT id FROM authors WHERE last_name = :last_name AND first_name = :first_name",
                    {"last_name": last_name, "first_name": first_name}
                )
                
                author = result.fetchone()
                if not author:
                    # Create new author
                    author = Author(
                        first_name=first_name,
                        last_name=last_name
                    )
                    session.add(author)
                    await session.flush()
                    author_id = author.id
                else:
                    author_id = author[0]
                
                # Link author to paper
                await session.execute(
                    "INSERT INTO paper_authors (paper_id, author_id, author_order) VALUES (:paper_id, :author_id, :author_order)",
                    {"paper_id": paper_id, "author_id": author_id, "author_order": i + 1}
                )
                
        except Exception as e:
            logger.error(f"Error creating authors: {e}")
    
    async def _update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status"""
        try:
            async with get_database_session() as session:
                await session.execute(
                    "UPDATE ingestion_jobs SET status = :status WHERE id = :job_id",
                    {"status": status, "job_id": job_id}
                )
                
                # Update additional fields if provided
                for field, value in kwargs.items():
                    if field == "started_at":
                        await session.execute(
                            "UPDATE ingestion_jobs SET started_at = :value WHERE id = :job_id",
                            {"value": value, "job_id": job_id}
                        )
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
    
    async def _update_job_progress(self, job_id: str, progress_percentage: float,
                                 total_processed: int, successful: int, 
                                 failed: int, duplicates: int, errors: List[str] = None):
        """Update job progress"""
        try:
            async with get_database_session() as session:
                await session.execute(
                    """UPDATE ingestion_jobs 
                       SET progress_percentage = :progress,
                           processed_documents = :processed,
                           successful_documents = :successful,
                           failed_documents = :failed,
                           duplicate_documents = :duplicates
                       WHERE id = :job_id""",
                    {
                        "progress": progress_percentage,
                        "processed": total_processed,
                        "successful": successful,
                        "failed": failed,
                        "duplicates": duplicates,
                        "job_id": job_id
                    }
                )
                
                if errors:
                    await session.execute(
                        "UPDATE ingestion_jobs SET errors = :errors WHERE id = :job_id",
                        {"errors": errors, "job_id": job_id}
                    )
                
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error updating job progress: {e}")
    
    async def _update_job_completion(self, job_id: str, status: str, 
                                   completed_at: datetime, 
                                   job_summary: Dict[str, Any] = None,
                                   errors: List[str] = None):
        """Update job completion status"""
        try:
            async with get_database_session() as session:
                await session.execute(
                    """UPDATE ingestion_jobs 
                       SET status = :status,
                           completed_at = :completed_at,
                           job_summary = :summary,
                           errors = :errors
                       WHERE id = :job_id""",
                    {
                        "status": status,
                        "completed_at": completed_at,
                        "summary": job_summary,
                        "errors": errors,
                        "job_id": job_id
                    }
                )
                await session.commit()
                
        except Exception as e:
            logger.error(f"Error updating job completion: {e}")
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        try:
            async with get_database_session() as session:
                result = await session.execute(
                    "SELECT * FROM ingestion_jobs WHERE id = :job_id",
                    {"job_id": job_id}
                )
                row = result.fetchone()
                
                if row:
                    return {
                        "id": row[0],
                        "job_name": row[1],
                        "status": row[4],
                        "progress_percentage": row[6],
                        "total_documents": row[7],
                        "processed_documents": row[8],
                        "successful_documents": row[9],
                        "failed_documents": row[10],
                        "duplicate_documents": row[11],
                        "created_at": row[15],
                        "started_at": row[16],
                        "completed_at": row[17]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    async def pause_job(self, job_id: str) -> bool:
        """Pause a running job"""
        try:
            if job_id in self.active_jobs:
                self.active_jobs[job_id].status = "paused"
                await self._update_job_status(job_id, "paused")
                return True
            return False
        except Exception as e:
            logger.error(f"Error pausing job: {e}")
            return False
    
    async def resume_job(self, job_id: str) -> bool:
        """Resume a paused job"""
        try:
            # This would require storing the job state to resume properly
            # For now, just update status
            await self._update_job_status(job_id, "pending")
            return True
        except Exception as e:
            logger.error(f"Error resuming job: {e}")
            return False
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        try:
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]
            await self._update_job_status(job_id, "failed")
            return True
        except Exception as e:
            logger.error(f"Error canceling job: {e}")
            return False


# Global bulk ingestion service instance
bulk_ingestion_service = BulkIngestionService()