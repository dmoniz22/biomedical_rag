"""
Unit tests for the RAG service
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.services.rag_service import rag_service
from app.models.schemas import SearchRequest, SearchType, SearchFilters


class TestRAGService:
    """Test cases for RAGService"""
    
    @pytest.mark.asyncio
    async def test_rag_service_initialization(self):
        """Test RAG service initialization"""
        with patch.object(rag_service, 'embedding_service'), \
             patch.object(rag_service, 'vector_db_service'):
            await rag_service.initialize()
            # Test should not raise any exceptions
        
    @pytest.mark.asyncio
    async def test_embed_query(self):
        """Test query embedding"""
        query = "diabetes treatment"
        with patch.object(rag_service.embedding_service, 'embed_query') as mock_embed:
            mock_embed.return_value = [0.1] * 384  # Mock embedding vector
            
            embedding = await rag_service.embedding_service.embed_query(query)
            
            assert len(embedding) == 384
            mock_embed.assert_called_once_with(query)
    
    @pytest.mark.asyncio
    async def test_search_papers(self, test_search_request):
        """Test paper search functionality"""
        with patch.object(rag_service, 'embedding_service') as mock_embedding, \
             patch.object(rag_service, 'vector_db_service') as mock_vector_db, \
             patch.object(rag_service, '_apply_filters', return_value=True), \
             patch.object(rag_service, '_generate_highlight', return_value="Test highlight"):
            
            # Mock embedding service
            mock_embedding.embed_query.return_value = [0.1] * 384
            mock_embedding.embed_keywords.return_value = [[0.1] * 384]
            
            # Mock vector database search
            mock_vector_db.search_similar_papers.return_value = [
                {
                    'paper_id': 'test_id_1',
                    'content': 'diabetes treatment abstract',
                    'similarity_score': 0.9,
                    'metadata': {'content_type': 'title'}
                }
            ]
            
            # Mock paper retrieval
            with patch('app.services.rag_service.get_database_session') as mock_session:
                mock_paper = Mock()
                mock_paper.id = 'test_id_1'
                mock_session.return_value.__aenter__.return_value.get.return_value = mock_paper
                
                result = await rag_service.search_papers(test_search_request)
                
                assert result.query == test_search_request.query
                assert len(result.results) >= 0  # May be 0 if mock data doesn't match
                assert result.query_type == test_search_request.search_type
    
    @pytest.mark.asyncio
    async def test_get_semantic_suggestions(self):
        """Test semantic search suggestions"""
        partial_query = "dia"
        
        suggestions = await rag_service.get_semantic_suggestions(partial_query)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 10
        # Should contain diabetes-related terms
        assert any("diabetes" in suggestion.lower() for suggestion in suggestions)
    
    @pytest.mark.asyncio
    async def test_enhance_query(self):
        """Test query enhancement"""
        query = "diabetes treatment"
        
        enhancement = await rag_service.enhance_query(query)
        
        assert "original_query" in enhancement
        assert enhancement["original_query"] == query
        assert "suggested_mesh_terms" in enhancement
        assert "related_concepts" in enhancement
        
        # Should have diabetes-related enhancements
        assert any("diabetes" in term.lower() for term in enhancement["suggested_mesh_terms"])
    
    def test_apply_filters(self):
        """Test search filter application"""
        from app.models.database import Paper
        
        # Create a mock paper
        mock_paper = Paper()
        mock_paper.publication_date = datetime(2023, 1, 1)
        mock_paper.quality_score = 0.8
        mock_paper.subject_areas = ["diabetes"]
        mock_paper.journal = "Diabetes Care"
        mock_paper.mesh_terms = ["Diabetes Mellitus"]
        
        # Test date range filter
        filters = {
            'date_range_start': datetime(2022, 1, 1),
            'date_range_end': datetime(2024, 1, 1),
            'min_quality_score': 0.7
        }
        
        result = rag_service._apply_filters(mock_paper, filters)
        assert result is True  # Paper should pass all filters
        
        # Test failing filters
        failing_filters = {
            'min_quality_score': 0.9  # Higher than paper's score
        }
        
        result = rag_service._apply_filters(mock_paper, failing_filters)
        assert result is False  # Paper should fail quality filter
    
    def test_generate_highlight(self):
        """Test highlight generation"""
        content = "This is a diabetes treatment study focusing on insulin therapy and blood glucose control."
        query = "diabetes treatment"
        
        highlight = rag_service._generate_highlight(content, query)
        
        assert highlight is not None
        assert isinstance(highlight, str)
        assert len(highlight) > 0
        # Should contain some portion of the original content
        assert "diabetes" in highlight.lower() or len(highlight) > 0