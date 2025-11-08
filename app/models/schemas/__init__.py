"""
Pydantic schemas for the Biomedical RAG System
"""

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# Search Types
class SearchType(str, Enum):
    """Types of search queries"""
    NATURAL_LANGUAGE = "natural_language"
    KEYWORD = "keyword"
    MESH_TERM = "mesh_term"


# Search Filters
class SearchFilters(BaseModel):
    """Filters for search results"""
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    min_quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    subject_area: Optional[str] = None
    journal: Optional[str] = None
    mesh_terms: Optional[List[str]] = None
    authors: Optional[List[str]] = None
    publication_type: Optional[List[str]] = None
    language: Optional[str] = None
    has_abstract: Optional[bool] = None
    has_full_text: Optional[bool] = None


# Search Request
class SearchRequest(BaseModel):
    """Request for searching papers"""
    query: str = Field(..., min_length=1, max_length=1000)
    search_type: SearchType = SearchType.NATURAL_LANGUAGE
    max_results: int = Field(10, ge=1, le=100)
    min_confidence_score: float = Field(0.7, ge=0.0, le=1.0)
    include_full_text: bool = False
    filters: Optional[SearchFilters] = None
    timeout_seconds: int = Field(30, ge=1, le=300)


# Paper Schema (referenced by SearchResult)
class PaperSchema(BaseModel):
    """Paper information schema"""
    id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    journal: Optional[str] = None
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    quality_score: float = 0.0
    has_embeddings: bool = False
    processing_status: str = "pending"


# Search Result
class SearchResult(BaseModel):
    """Individual search result"""
    paper: PaperSchema
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    highlight: Optional[str] = None
    matched_content_type: Optional[str] = None
    matched_keywords: Optional[List[str]] = None


# Search Response
class SearchResponse(BaseModel):
    """Response for search queries"""
    query: str
    results: List[SearchResult] = []
    total_results: int = 0
    execution_time_ms: int = 0
    query_type: SearchType
    search_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Bulk Ingestion Request
class BulkIngestionRequest(BaseModel):
    """Request for bulk ingestion of papers"""
    search_queries: List[str] = Field(..., min_items=1, max_items=100)
    max_papers_per_query: int = Field(100, ge=1, le=10000)
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    paper_types: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    max_concurrent_requests: int = Field(5, ge=1, le=20)
    generate_embeddings: bool = True
    priority: str = Field("normal", regex="^(low|normal|high)$")
    notify_on_completion: bool = False
    contact_email: Optional[str] = None


# Ingestion Job Status
class IngestionJobStatus(str, Enum):
    """Status of ingestion jobs"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Ingestion Job Create
class IngestionJobCreate(BaseModel):
    """Schema for creating ingestion jobs"""
    request: BulkIngestionRequest
    user_id: Optional[str] = None
    priority: str = Field("normal", regex="^(low|normal|high)$")


# Ingestion Job
class IngestionJob(BaseModel):
    """Ingestion job information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: IngestionJobStatus = IngestionJobStatus.PENDING
    total_papers: int = 0
    processed_papers: int = 0
    successful_papers: int = 0
    failed_papers: int = 0
    estimated_completion: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    request: BulkIngestionRequest
    user_id: Optional[str] = None
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)


# Database Statistics
class DatabaseStats(BaseModel):
    """Database statistics information"""
    total_papers: int = 0
    papers_with_embeddings: int = 0
    total_authors: int = 0
    total_subject_areas: int = 0
    pending_ingestion_jobs: int = 0
    failed_papers: int = 0
    average_quality_score: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# Export all schemas
__all__ = [
    "SearchType",
    "SearchFilters", 
    "SearchRequest",
    "PaperSchema",
    "SearchResult",
    "SearchResponse",
    "BulkIngestionRequest",
    "IngestionJobStatus",
    "IngestionJobCreate",
    "IngestionJob",
    "DatabaseStats"
]
