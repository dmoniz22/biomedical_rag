"""
FastAPI router for biomedical RAG system API v1
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
import logging

from app.core.config import settings
from app.services.rag_service import rag_service
from app.services.bulk_ingestion_service import bulk_ingestion_service
from app.services.pubmed_service import pubmed_service
from app.models.schemas import (
    SearchRequest, SearchResponse, BulkIngestionRequest, DatabaseStats
)

logger = logging.getLogger(__name__)

# Create API router
api_router = APIRouter()


# Search endpoints
@api_router.post("/search", response_model=SearchResponse)
async def search_papers(search_request: SearchRequest):
    """Search biomedical literature using RAG"""
    try:
        result = await rag_service.search_papers(search_request)
        return result
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@api_router.get("/search/suggestions")
async def get_search_suggestions(query: str):
    """Get semantic search suggestions"""
    try:
        suggestions = await rag_service.get_semantic_suggestions(query)
        return {"query": query, "suggestions": suggestions}
    except Exception as e:
        logger.error(f"Search suggestions error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get suggestions: {str(e)}"
        )


@api_router.post("/search/enhance")
async def enhance_search_query(query: str):
    """Enhance search query with MeSH terms and related concepts"""
    try:
        enhancement = await rag_service.enhance_query(query)
        return enhancement
    except Exception as e:
        logger.error(f"Query enhancement error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query enhancement failed: {str(e)}"
        )


# Bulk ingestion endpoints
@api_router.post("/ingestion/bulk/start")
async def start_bulk_ingestion(request: BulkIngestionRequest):
    """Start a new bulk ingestion job"""
    try:
        job_id = await bulk_ingestion_service.start_bulk_ingestion(request)
        return {"job_id": job_id, "message": "Bulk ingestion started successfully"}
    except Exception as e:
        logger.error(f"Bulk ingestion start error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start bulk ingestion: {str(e)}"
        )


@api_router.get("/ingestion/bulk/status/{job_id}")
async def get_bulk_ingestion_status(job_id: str):
    """Get bulk ingestion job status"""
    try:
        status = await bulk_ingestion_service.get_job_status(job_id)
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk ingestion status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )


# Database management endpoints
@api_router.get("/database/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics"""
    try:
        from app.core.database import get_database_session
        from app.models.database import Paper, Author, IngestionJob
        
        async with get_database_session() as session:
            # Get total papers count
            total_papers_result = await session.execute("SELECT COUNT(*) FROM papers")
            total_papers = total_papers_result.scalar()
            
            # Get papers with embeddings
            papers_with_embeddings_result = await session.execute(
                "SELECT COUNT(*) FROM papers WHERE embedding_generated = true"
            )
            papers_with_embeddings = papers_with_embeddings_result.scalar()
            
            # Get total authors count
            total_authors_result = await session.execute("SELECT COUNT(*) FROM authors")
            total_authors = total_authors_result.scalar()
            
            # Get pending jobs count
            pending_jobs_result = await session.execute(
                "SELECT COUNT(*) FROM ingestion_jobs WHERE status = 'pending'"
            )
            pending_ingestion_jobs = pending_jobs_result.scalar()
            
            # Get failed papers count
            failed_papers_result = await session.execute(
                "SELECT COUNT(*) FROM papers WHERE processing_status = 'failed'"
            )
            failed_papers = failed_papers_result.scalar()
            
            # Get average quality score
            avg_quality_result = await session.execute(
                "SELECT AVG(quality_score) FROM papers WHERE quality_score > 0"
            )
            average_quality_score = avg_quality_result.scalar() or 0.0
            
            return DatabaseStats(
                total_papers=total_papers,
                papers_with_embeddings=papers_with_embeddings,
                total_authors=total_authors,
                total_subject_areas=len(settings.SUBJECT_AREAS),
                pending_ingestion_jobs=pending_ingestion_jobs,
                failed_papers=failed_papers,
                average_quality_score=float(average_quality_score)
            )
    except Exception as e:
        logger.error(f"Database stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get database stats: {str(e)}"
        )


# Subject areas endpoints
@api_router.get("/subject-areas")
async def get_subject_areas():
    """Get available subject areas"""
    return {
        "subject_areas": settings.SUBJECT_AREAS,
        "total_count": len(settings.SUBJECT_AREAS)
    }


# System information endpoints
@api_router.get("/system/info")
async def get_system_info():
    """Get system information"""
    import sys
    import platform
    
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "python_version": sys.version,
        "platform": platform.platform(),
        "database_url_configured": bool(settings.DATABASE_URL != "postgresql+asyncpg://user:password@localhost/biomed_rag"),
        "pubmed_api_key_configured": bool(settings.PUBMED_API_KEY),
        "vector_db_type": settings.VECTOR_DB_TYPE,
        "max_workers": settings.MAX_WORKERS,
        "chunk_size": settings.CHUNK_SIZE
    }