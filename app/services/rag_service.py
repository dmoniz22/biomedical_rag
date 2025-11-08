"""
RAG (Retrieval-Augmented Generation) service for biomedical literature search
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime

from app.services.embedding_service import embedding_service
from app.services.vector_db_service import vector_db_service
from app.models.database import Paper, PaperEmbedding
from app.core.database import get_database_session
from app.models.schemas import SearchRequest, SearchResponse, SearchResult, SearchType

logger = logging.getLogger(__name__)


class RAGService:
    """Main RAG service for biomedical literature search"""
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.vector_db_service = vector_db_service
    
    async def initialize(self):
        """Initialize RAG service and its dependencies"""
        await self.embedding_service.initialize()
        await self.vector_db_service.initialize()
        logger.info("RAG service initialized successfully")
    
    async def process_paper_for_rag(self, paper_id: str, paper_data: Dict[str, Any]):
        """Process a paper and generate embeddings for RAG"""
        try:
            async with get_database_session() as session:
                # Generate embeddings for the paper
                embeddings_data = await self.embedding_service.generate_paper_embeddings(paper_data)
                
                # Store embeddings in vector database
                for content_type, embedding_info in embeddings_data.items():
                    if content_type == 'full_text_chunks':
                        # Store chunked embeddings
                        await self.vector_db_service.store_embeddings(
                            paper_id=paper_id,
                            content_type="chunk",
                            embeddings_data=embedding_info,
                            model_name="sentence-transformer"
                        )
                    else:
                        # Store single embedding
                        await self.vector_db_service.store_embeddings(
                            paper_id=paper_id,
                            content_type=content_type,
                            embeddings_data={
                                'text': paper_data.get(content_type, ''),
                                'embedding': embedding_info
                            },
                            model_name="sentence-transformer"
                        )
                
                # Update paper status
                paper = await session.get(Paper, paper_id)
                if paper:
                    paper.embedding_generated = True
                    paper.processing_status = "processed"
                    await session.commit()
                
                logger.info(f"Successfully processed paper {paper_id} for RAG")
                
        except Exception as e:
            logger.error(f"Failed to process paper {paper_id} for RAG: {e}")
            # Update paper status to failed
            async with get_database_session() as session:
                paper = await session.get(Paper, paper_id)
                if paper:
                    paper.processing_status = "failed"
                    await session.commit()
            raise
    
    async def search_papers(self, search_request: SearchRequest) -> SearchResponse:
        """Perform RAG search on biomedical literature"""
        start_time = time.time()
        
        try:
            # Embed the search query
            if search_request.search_type == SearchType.NATURAL_LANGUAGE:
                query_embedding = await self.embedding_service.embed_query(search_request.query)
            elif search_request.search_type == SearchType.KEYWORD:
                # For keyword search, embed each keyword and combine
                keywords = search_request.query.split()
                keyword_embeddings = await self.embedding_service.embed_keywords(keywords)
                # Average the embeddings
                query_embedding = [
                    sum(emb[i] for emb in keyword_embeddings) / len(keyword_embeddings)
                    for i in range(len(keyword_embeddings[0]))
                ]
            else:  # MESH_TERM
                query_embedding = await self.embedding_service.embed_query(search_request.query)
            
            # Perform vector search
            search_results = []
            
            # Search in different content types
            content_types = ["title", "abstract"]
            if search_request.include_full_text:
                content_types.append("chunk")
            
            for content_type in content_types:
                results = await self.vector_db_service.search_similar_papers(
                    query_embedding=query_embedding,
                    content_type=content_type,
                    top_k=search_request.max_results,
                    threshold=search_request.min_confidence_score
                )
                
                # Add results to our search results
                for result in results:
                    # Avoid duplicates (same paper from different content types)
                    paper_id = result['paper_id']
                    if not any(r['paper']['id'] == paper_id for r in search_results):
                        search_results.append({
                            'paper_id': paper_id,
                            'content_type': content_type,
                            'content': result['content'],
                            'similarity_score': result['similarity_score'],
                            'metadata': result['metadata']
                        })
            
            # Sort by similarity score and limit results
            search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            search_results = search_results[:search_request.max_results]
            
            # Get paper details from database
            paper_results = []
            async with get_database_session() as session:
                for result in search_results:
                    paper = await session.get(Paper, result['paper_id'])
                    if paper:
                        # Apply additional filters
                        if self._apply_filters(paper, search_request.filters):
                            search_result = SearchResult(
                                paper=paper,
                                relevance_score=result['similarity_score'],
                                highlight=self._generate_highlight(
                                    result['content'], search_request.query
                                )
                            )
                            paper_results.append(search_result)
            
            execution_time = int((time.time() - start_time) * 1000)
            
            return SearchResponse(
                query=search_request.query,
                results=paper_results,
                total_results=len(paper_results),
                execution_time_ms=execution_time,
                query_type=search_request.search_type
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            execution_time = int((time.time() - start_time) * 1000)
            return SearchResponse(
                query=search_request.query,
                results=[],
                total_results=0,
                execution_time_ms=execution_time,
                query_type=search_request.search_type
            )
    
    def _apply_filters(self, paper: Paper, filters: Optional[Dict[str, Any]]) -> bool:
        """Apply search filters to a paper"""
        if not filters:
            return True
        
        try:
            # Date range filter
            if filters.get('date_range_start') and paper.publication_date:
                if paper.publication_date < filters['date_range_start']:
                    return False
            
            if filters.get('date_range_end') and paper.publication_date:
                if paper.publication_date > filters['date_range_end']:
                    return False
            
            # Quality score filter
            if filters.get('min_quality_score'):
                if paper.quality_score < filters['min_quality_score']:
                    return False
            
            # Subject area filter
            if filters.get('subject_area'):
                if filters['subject_area'] not in paper.subject_areas:
                    return False
            
            # Journal filter
            if filters.get('journal'):
                if not paper.journal or filters['journal'].lower() not in paper.journal.lower():
                    return False
            
            # MeSH terms filter
            if filters.get('mesh_terms'):
                if not paper.mesh_terms or not any(
                    mesh_term in paper.mesh_terms for mesh_term in filters['mesh_terms']
                ):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return True  # If filter application fails, don't exclude the paper
    
    def _generate_highlight(self, content: str, query: str) -> Optional[str]:
        """Generate a highlighted snippet of content"""
        try:
            query_words = query.lower().split()
            content_lower = content.lower()
            
            # Find the first occurrence of any query word
            for word in query_words:
                if word in content_lower:
                    start_pos = content_lower.find(word)
                    # Get context around the match
                    snippet_start = max(0, start_pos - 50)
                    snippet_end = min(len(content), start_pos + 200)
                    snippet = content[snippet_start:snippet_end]
                    
                    # Add ellipsis if we trimmed content
                    if snippet_start > 0:
                        snippet = "..." + snippet
                    if snippet_end < len(content):
                        snippet = snippet + "..."
                    
                    return snippet
            
            # If no match found, return first 200 characters
            return content[:200] + "..." if len(content) > 200 else content
            
        except Exception as e:
            logger.error(f"Error generating highlight: {e}")
            return content[:200] + "..." if len(content) > 200 else content
    
    async def get_semantic_suggestions(self, partial_query: str) -> List[str]:
        """Get semantic suggestions for query completion"""
        try:
            # This would typically use the vector database to find similar terms
            # For now, return some basic medical terms as suggestions
            medical_terms = [
                "cardiology", "oncology", "neurology", "immunology", "diabetes",
                "hypertension", "cancer", "Alzheimer", "Parkinson", "heart disease",
                "diabetes mellitus", "blood pressure", "mental health", "vaccine",
                "clinical trial", "randomized controlled trial", "systematic review"
            ]
            
            # Filter terms that start with the partial query
            suggestions = [
                term for term in medical_terms
                if term.lower().startswith(partial_query.lower())
            ]
            
            return suggestions[:10]  # Return top 10 suggestions
            
        except Exception as e:
            logger.error(f"Error getting semantic suggestions: {e}")
            return []
    
    async def enhance_query(self, query: str) -> Dict[str, Any]:
        """Enhance a query with MeSH terms and related concepts"""
        try:
            # This would use medical ontologies and MeSH database
            # For now, return basic enhancement
            enhancement = {
                "original_query": query,
                "suggested_mesh_terms": [],
                "related_concepts": [],
                "query_expansions": []
            }
            
            # Simple keyword matching for demo
            query_lower = query.lower()
            if "diabetes" in query_lower:
                enhancement["suggested_mesh_terms"] = ["Diabetes Mellitus", "Blood Glucose", "Insulin"]
                enhancement["related_concepts"] = ["HbA1c", "glucose tolerance", "insulin resistance"]
            
            if "heart" in query_lower or "cardiac" in query_lower:
                enhancement["suggested_mesh_terms"] = ["Heart Diseases", "Cardiovascular System", "Electrocardiography"]
                enhancement["related_concepts"] = ["myocardial infarction", "arrhythmia", "hypertension"]
            
            return enhancement
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return {"original_query": query, "suggested_mesh_terms": [], "related_concepts": []}


# Global RAG service instance
rag_service = RAGService()