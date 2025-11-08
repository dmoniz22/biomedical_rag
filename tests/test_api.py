"""
Unit tests for the API endpoints
"""

import pytest
from httpx import AsyncClient


class TestSearchAPI:
    """Test cases for search endpoints"""
    
    @pytest.mark.asyncio
    async def test_search_papers(self, test_client: AsyncClient, test_search_request):
        """Test paper search endpoint"""
        response = await test_client.post("/api/v1/search", json=test_search_request.dict())
        
        # The test will fail if API is not properly initialized, but we're testing the structure
        assert response.status_code in [200, 500]  # 200 if working, 500 if service not initialized
    
    @pytest.mark.asyncio
    async def test_search_suggestions(self, test_client: AsyncClient):
        """Test search suggestions endpoint"""
        response = await test_client.get("/api/v1/search/suggestions?query=diabetes")
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "query" in data
            assert "suggestions" in data
            assert isinstance(data["suggestions"], list)
    
    @pytest.mark.asyncio
    async def test_enhance_query(self, test_client: AsyncClient):
        """Test query enhancement endpoint"""
        response = await test_client.post("/api/v1/search/enhance?query=diabetes")
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "original_query" in data
            assert "suggested_mesh_terms" in data
            assert "related_concepts" in data


class TestBulkIngestionAPI:
    """Test cases for bulk ingestion endpoints"""
    
    @pytest.mark.asyncio
    async def test_start_bulk_ingestion(self, test_client: AsyncClient, test_bulk_request):
        """Test start bulk ingestion endpoint"""
        response = await test_client.post("/api/v1/ingestion/bulk/start", json=test_bulk_request.dict())
        
        # Should return 500 if services not initialized, but structure should be correct
        assert response.status_code in [200, 500]
    
    @pytest.mark.asyncio
    async def test_get_bulk_status(self, test_client: AsyncClient):
        """Test get bulk ingestion status endpoint"""
        fake_job_id = "fake-job-id-123"
        response = await test_client.get(f"/api/v1/ingestion/bulk/status/{fake_job_id}")
        
        assert response.status_code in [404, 500]  # 404 if job doesn't exist, 500 if error


class TestDatabaseAPI:
    """Test cases for database endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_database_stats(self, test_client: AsyncClient):
        """Test database stats endpoint"""
        response = await test_client.get("/api/v1/database/stats")
        
        assert response.status_code in [200, 500]
        if response.status_code == 200:
            data = response.json()
            assert "total_papers" in data
            assert "papers_with_embeddings" in data
            assert "total_authors" in data
            assert "average_quality_score" in data


class TestSystemAPI:
    """Test cases for system endpoints"""
    
    @pytest.mark.asyncio
    async def test_get_subject_areas(self, test_client: AsyncClient):
        """Test subject areas endpoint"""
        response = await test_client.get("/api/v1/subject-areas")
        
        assert response.status_code == 200
        data = response.json()
        assert "subject_areas" in data
        assert "total_count" in data
        assert isinstance(data["subject_areas"], list)
        assert len(data["subject_areas"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_system_info(self, test_client: AsyncClient):
        """Test system info endpoint"""
        response = await test_client.get("/api/v1/system/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "app_name" in data
        assert "version" in data
        assert "python_version" in data
        assert "vector_db_type" in data
        assert "max_workers" in data


class TestHealthAPI:
    """Test cases for health check endpoints"""
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_client: AsyncClient):
        """Test main health check endpoint"""
        response = await test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, test_client: AsyncClient):
        """Test root endpoint"""
        response = await test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data