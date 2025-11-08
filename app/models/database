"""
Database models for the Biomedical RAG System
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON, Index, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from typing import Optional, Dict, Any
import uuid
from datetime import datetime

Base = declarative_base()


class Paper(Base):
    """Model for storing biomedical papers and articles"""
    
    __tablename__ = "papers"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    pmid = Column(String, unique=True, index=True, nullable=True)  # PubMed ID
    title = Column(String(2000), nullable=False)
    abstract = Column(Text, nullable=True)
    full_text = Column(Text, nullable=True)
    
    # Publication information
    journal = Column(String(500), nullable=True)
    publication_date = Column(DateTime, nullable=True)
    doi = Column(String(255), unique=True, index=True, nullable=True)
    publication_type = Column(String(100), nullable=True)
    
    # Content metadata
    keywords = Column(JSON, default=list)  # List of MeSH terms and keywords
    mesh_terms = Column(JSON, default=list)  # MeSH headings
    subject_areas = Column(JSON, default=list)  # Categorized subject areas
    
    # Quality and validation
    quality_score = Column(Float, default=0.0, index=True)  # Automated quality assessment
    validation_status = Column(String(50), default="pending")  # pending, validated, rejected
    validation_date = Column(DateTime, nullable=True)
    validation_notes = Column(Text, nullable=True)
    
    # Processing status
    processing_status = Column(String(50), default="pending")  # pending, processing, processed, failed
    embedding_generated = Column(Boolean, default=False)
    full_text_extracted = Column(Boolean, default=False)
    duplicate_checked = Column(Boolean, default=False)
    is_duplicate = Column(Boolean, default=False)
    
    # Source and ingestion
    source_database = Column(String(100), nullable=False)  # pubmed, cochrane, clinicaltrials, etc.
    source_url = Column(String(1000), nullable=True)
    ingestion_date = Column(DateTime, default=func.now())
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    authors = relationship("Author", secondary="paper_authors", back_populates="papers")
    citations_received = relationship("Citation", back_populates="cited_paper")
    citations_given = relationship("Citation", back_populates="citing_paper")
    embeddings = relationship("PaperEmbedding", back_populates="paper", cascade="all, delete-orphan")
    searches = relationship("SearchQuery", back_populates="paper")
    
    # Indexes
    __table_args__ = (
        Index('idx_paper_quality_score', 'quality_score'),
        Index('idx_paper_processing_status', 'processing_status'),
        Index('idx_paper_ingestion_date', 'ingestion_date'),
        Index('idx_paper_publication_date', 'publication_date'),
        Index('idx_paper_subject_areas', 'subject_areas', postgresql_using='gin'),
        Index('idx_paper_mesh_terms', 'mesh_terms', postgresql_using='gin'),
    )


class Author(Base):
    """Model for paper authors"""
    
    __tablename__ = "authors"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    first_name = Column(String(200), nullable=True)
    last_name = Column(String(200), nullable=False)
    middle_initial = Column(String(10), nullable=True)
    affiliation = Column(String(500), nullable=True)
    email = Column(String(255), nullable=True)
    orcid = Column(String(50), nullable=True)  # ORCID identifier
    
    # Validation
    name_variants = Column(JSON, default=list)  # Different name spellings
    validated = Column(Boolean, default=False)
    
    # Relationships
    papers = relationship("Paper", secondary="paper_authors", back_populates="authors")
    
    __table_args__ = (
        Index('idx_author_name', 'last_name', 'first_name'),
        Index('idx_author_orcid', 'orcid'),
    )


class PaperAuthor(Base):
    """Many-to-many relationship between papers and authors"""
    
    __tablename__ = "paper_authors"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    author_id = Column(String, ForeignKey("authors.id"), nullable=False)
    author_order = Column(Integer, nullable=False)  # Order of authorship
    is_corresponding = Column(Boolean, default=False)
    
    paper = relationship("Paper", back_populates="authors")
    author = relationship("Author", back_populates="papers")
    
    __table_args__ = (
        UniqueConstraint('paper_id', 'author_id', name='uq_paper_author'),
    )


class Citation(Base):
    """Model for citation relationships between papers"""
    
    __tablename__ = "citations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    citing_paper_id = Column(String, ForeignKey("papers.id"), nullable=False)  # Paper that cites
    cited_paper_id = Column(String, ForeignKey("papers.id"), nullable=False)   # Paper being cited
    citation_context = Column(Text, nullable=True)  # Text where citation appears
    confidence_score = Column(Float, default=0.0)  # Confidence in citation detection
    
    citing_paper = relationship("Paper", foreign_keys=[citing_paper_id], back_populates="citations_given")
    cited_paper = relationship("Paper", foreign_keys=[cited_paper_id], back_populates="citations_received")
    
    __table_args__ = (
        Index('idx_citation_citing', 'citing_paper_id'),
        Index('idx_citation_cited', 'cited_paper_id'),
        Index('idx_citation_confidence', 'confidence_score'),
    )


class PaperEmbedding(Base):
    """Model for storing vector embeddings of papers"""
    
    __tablename__ = "paper_embeddings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    content_type = Column(String(50), nullable=False)  # title, abstract, full_text, chunk
    content_hash = Column(String(64), nullable=False)  # Hash of content to detect changes
    
    # Vector embedding
    embedding_vector = Column(JSON, nullable=False)  # Store as JSON array
    embedding_model = Column(String(200), nullable=False)  # Model used for embedding
    embedding_dimension = Column(Integer, nullable=False)
    
    # Metadata
    chunk_index = Column(Integer, nullable=True)  # For chunked documents
    chunk_text = Column(Text, nullable=True)  # Store chunk text for reference
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    paper = relationship("Paper", back_populates="embeddings")
    
    __table_args__ = (
        Index('idx_embedding_paper', 'paper_id'),
        Index('idx_embedding_content_type', 'content_type'),
        Index('idx_embedding_model', 'embedding_model'),
    )


class IngestionJob(Base):
    """Model for tracking bulk ingestion jobs"""
    
    __tablename__ = "ingestion_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_name = Column(String(200), nullable=False)
    job_type = Column(String(100), nullable=False)  # bulk_import, real_time_update, manual
    source = Column(String(100), nullable=False)  # pubmed, cochrane, manual_file, etc.
    
    # Job parameters
    parameters = Column(JSON, default=dict)  # Job-specific parameters
    target_subject_areas = Column(JSON, default=list)  # Subject areas to target
    
    # Job status
    status = Column(String(50), default="pending")  # pending, running, completed, failed, paused
    progress_percentage = Column(Float, default=0.0)
    total_documents = Column(Integer, default=0)
    processed_documents = Column(Integer, default=0)
    successful_documents = Column(Integer, default=0)
    failed_documents = Column(Integer, default=0)
    duplicate_documents = Column(Integer, default=0)
    
    # Results
    errors = Column(JSON, default=list)  # List of error messages
    warnings = Column(JSON, default=list)  # List of warnings
    job_summary = Column(JSON, default=dict)  # Summary statistics
    
    # Timing
    created_at = Column(DateTime, default=func.now())
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    estimated_completion = Column(DateTime, nullable=True)
    
    # Control
    can_resume = Column(Boolean, default=True)
    resume_from_position = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_job_status', 'status'),
        Index('idx_job_type', 'job_type'),
        Index('idx_job_created', 'created_at'),
    )


class SearchQuery(Base):
    """Model for storing user search queries and results"""
    
    __tablename__ = "search_queries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), default="natural_language")  # natural_language, keyword, mesh_term
    
    # Search parameters
    search_filters = Column(JSON, default=dict)  # Applied filters
    max_results = Column(Integer, default=10)
    min_quality_score = Column(Float, default=0.0)
    subject_area_filter = Column(String(200), nullable=True)
    date_range_start = Column(DateTime, nullable=True)
    date_range_end = Column(DateTime, nullable=True)
    
    # Search results
    results_count = Column(Integer, default=0)
    results_papers = Column(JSON, default=list)  # List of paper IDs and relevance scores
    execution_time_ms = Column(Integer, nullable=True)
    
    # User info (optional)
    user_id = Column(String, nullable=True)
    user_ip = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Relationships
    paper = relationship("Paper", back_populates="searches")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_search_created', 'created_at'),
        Index('idx_search_type', 'query_type'),
        Index('idx_search_user', 'user_id'),
    )


class QualityAssessment(Base):
    """Model for storing automated quality assessments"""
    
    __tablename__ = "quality_assessments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String, ForeignKey("papers.id"), nullable=False)
    
    # Quality scores (0-1 scale)
    overall_score = Column(Float, nullable=False)
    content_quality_score = Column(Float, nullable=False)
    methodology_score = Column(Float, nullable=False)
    writing_quality_score = Column(Float, nullable=False)
    citation_quality_score = Column(Float, nullable=False)
    
    # Assessment details
    assessment_criteria = Column(JSON, default=dict)  # Detailed criteria scores
    identified_issues = Column(JSON, default=list)  # List of issues found
    recommendations = Column(JSON, default=list)  # Improvement recommendations
    
    # Metadata
    assessment_model = Column(String(200), nullable=False)
    assessment_version = Column(String(50), nullable=False)
    confidence_level = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    
    paper = relationship("Paper")
    
    __table_args__ = (
        Index('idx_quality_paper', 'paper_id'),
        Index('idx_quality_score', 'overall_score'),
        Index('idx_quality_model', 'assessment_model'),
    )


class SubjectArea(Base):
    """Model for managing dynamic subject area classifications"""
    
    __tablename__ = "subject_areas"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), unique=True, nullable=False)
    display_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Classification rules
    mesh_keywords = Column(JSON, default=list)  # MeSH terms that trigger this area
    keyword_patterns = Column(JSON, default=list)  # Regex patterns for classification
    
    # Hierarchy
    parent_area = Column(String, ForeignKey("subject_areas.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    priority = Column(Integer, default=0)  # Higher priority areas are checked first
    
    # Statistics
    paper_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=func.now())
    
    # Relationships
    children = relationship("SubjectArea", backref="parent", remote_side=[id])
    
    __table_args__ = (
        Index('idx_subject_area_name', 'name'),
        Index('idx_subject_area_active', 'is_active'),
        Index('idx_subject_area_priority', 'priority'),
    )