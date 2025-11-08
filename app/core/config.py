"""
Configuration settings for the Biomedical RAG System
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "Biomedical Literature Search System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/biomed_rag"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Vector Database
    VECTOR_DB_PATH: str = "./data/vector_db"
    VECTOR_DB_TYPE: str = "chroma"  # chroma, faiss, elasticsearch
    
    # ML Models
    SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CLINICAL_BERT_MODEL: str = "dmis-lab/biobert-base-cased-v1.1"
    EMBEDDING_DIMENSION: int = 384
    
    # RAG Configuration
    TOP_K_RESULTS: int = 10
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MIN_CONFIDENCE_SCORE: float = 0.7
    
    # Medical Database APIs
    PUBMED_API_KEY: Optional[str] = None
    PUBMED_BASE_URL: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    COCHRANE_API_KEY: Optional[str] = None
    CLINICALTRIALS_BASE_URL: str = "https://clinicaltrials.gov/api"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    
    # Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 100
    CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_PORT: int = 8001
    LOG_LEVEL: str = "INFO"
    
    # Bulk Ingestion
    MAX_DOCUMENTS_PER_BATCH: int = 1000
    INGESTION_RATE_LIMIT: int = 100  # requests per hour
    BULK_PROCESS_TIMEOUT: int = 3600  # seconds
    
    # Cache
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 10000
    
    # File Processing
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_FILE_EXTENSIONS: List[str] = [".pdf", ".txt", ".xml", ".json"]
    
    # Subject Areas
    SUBJECT_AREAS: List[str] = [
        "cardiology", "oncology", "neurology", "immunology", "endocrinology",
        "gastroenterology", "nephrology", "pulmonology", "rheumatology",
        "dermatology", "ophthalmology", "psychiatry", "pediatrics", "geriatrics"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()