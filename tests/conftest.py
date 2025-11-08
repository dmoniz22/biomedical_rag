"""
Test configuration and fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Import the app and dependencies
from app.main import app
from app.core.database import get_database_session
from app.core.config import settings

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://biomed_user:biomed_password@localhost:5432/biomed_rag_test"

# Test engine
test_engine = create_async_engine(TEST_DATABASE_URL, echo=True)
TestAsyncSession = sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db():
    """Set up test database"""
    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(import_db_models)
    
    yield test_engine
    
    # Clean up
    async with test_engine.begin() as conn:
        await conn.run_sync(drop_db_models)


def import_db_models(Base):
    """Import all database models for table creation"""
    from app.models.database import Base


def drop_db_models(Base):
    """Drop all database tables"""
    Base.metadata.drop_all()


@pytest.fixture
async def test_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session"""
    async with TestAsyncSession() as session:
        yield session


@pytest.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Create a test client for the FastAPI app"""
    # Override database dependency
    app.dependency_overrides[get_database_session] = test_session
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    
    # Clean up dependency overrides
    app.dependency_overrides.clear()


@pytest.fixture
async def test_papers(test_session: AsyncSession):
    """Create test papers in the database"""
    from app.models.database import Paper
    from datetime import datetime
    import uuid
    
    papers = []
    for i in range(5):
        paper = Paper(
            id=str(uuid.uuid4()),
            title=f"Test Paper {i+1}",
            abstract=f"This is a test abstract for paper {i+1}",
            journal="Test Journal",
            publication_date=datetime.now(),
            source_database="test",
            keywords=[f"keyword{i}"],
            mesh_terms=[f"mesh_term{i}"],
            subject_areas=["test_area"],
            processing_status="processed",
            quality_score=0.8,
            embedding_generated=True
        )
        test_session.add(paper)
        papers.append(paper)
    
    await test_session.commit()
    return papers


@pytest.fixture
async def test_search_request():
    """Create a test search request"""
    from app.models.schemas import SearchRequest, SearchType, SearchFilters
    
    return SearchRequest(
        query="diabetes treatment",
        search_type=SearchType.NATURAL_LANGUAGE,
        max_results=10,
        min_confidence_score=0.7
    )


@pytest.fixture
async def test_bulk_request():
    """Create a test bulk ingestion request"""
    from app.models.schemas import BulkIngestionRequest
    from datetime import datetime
    
    return BulkIngestionRequest(
        source_database="pubmed",
        subject_areas=["diabetes", "endocrinology"],
        max_documents=100,
        date_range_start=datetime(2020, 1, 1),
        date_range_end=datetime(2023, 12, 31),
        include_full_text=True,
        quality_threshold=0.7
    )