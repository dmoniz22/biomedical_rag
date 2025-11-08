# Project Context

## Purpose

**Biomedical Literature Search and Analysis System**

The primary goal is to build a Retrieval-Augmented Generation (RAG) system that enables clinicians and researchers to efficiently discover and analyze relevant medical literature through natural language queries. The system will automatically ingest new research papers from medical databases, dynamically organize them by subject areas, and provide intelligent search capabilities to support evidence-based medical decision making.

**Key Objectives:**
- **Initial Setup**: Bulk ingestion of existing medical literature from historical databases
- **Ongoing Operations**: Automated scanning and ingestion of new medical literature
- Dynamic subject area classification and table creation
- Natural language query processing for medical research
- Real-time updates when new papers become available
- Scalable architecture to handle large volumes of medical literature
- Support for evidence-based clinical decision making

## Tech Stack

- **Python 3.11+** - Primary development language
- **LangChain** - RAG framework and document processing
- **Sentence Transformers** - Medical text embedding and similarity search
- **Vector Database**: ChromaDB or Pinecone for document storage
- **FastAPI** - REST API framework for query interface
- **PostgreSQL** - Metadata storage and subject area table management
- **Docker** - Containerization for deployment consistency
- **React/TypeScript** - Frontend interface for search and results
- **spaCy** - Medical text processing and NER
- **scikit-learn** - Classification and clustering algorithms
- **Celery** - Background task processing for document ingestion
- **Redis** - Caching and message broker
- **GitHub Actions** - CI/CD pipeline
- **pytest** - Testing framework

## Project Conventions

### Code Style

- **PEP 8** compliance with 88 character line limit
- **Type hints** required for all function parameters and return values
- **Docstrings** in Google format for all public functions and classes
- **Variable naming**: snake_case for variables and functions, PascalCase for classes
- **Configuration**: Use Pydantic BaseModel for configuration classes
- **Error handling**: Specific exception classes, proper logging with context
- **Code organization**: Modular architecture with clear separation of concerns
- **Import organization**: Standard library, third-party, local imports (alphabetized)
- **Maximum function length**: 50 lines, maximum class length: 200 lines

### Architecture Patterns

- **RAG Architecture**: Document ingestion → Vector storage → Query processing → Response generation
- **Bulk Ingestion Pipeline**: Batch processing for initial data seeding with progress tracking and resume capability
- **Microservices**: Separate services for ingestion, search, and API layers
- **Event-driven**: Pub/sub pattern for document processing workflows
- **Repository pattern**: Abstraction layer for data access
- **Factory pattern**: Dynamic subject area table creation
- **Observer pattern**: Real-time updates for new paper ingestion
- **Circuit breaker**: For external API calls to medical databases
- **Strategy pattern**: Multiple embedding and search strategies
- **Data Seeding Architecture**: Efficient handling of large-scale historical data import with parallel processing

### Testing Strategy

- **Unit tests**: Minimum 90% coverage for business logic
- **Integration tests**: End-to-end RAG pipeline testing
- **Test data**: Use anonymized medical abstracts and synthetic data
- **Performance tests**: Query response time < 2 seconds for 95th percentile
- **Accuracy tests**: RAG retrieval accuracy validation against known results
- **Load testing**: Concurrent user simulation for system capacity
- **Security tests**: Input validation and data leakage prevention
- **Test fixtures**: Reusable medical document samples for consistent testing
- **Mock services**: External medical database APIs for testing

### Git Workflow

- **Branching strategy**: GitFlow with main, develop, feature/ branches
- **Branch naming**: feature/description, hotfix/issue-number, release/version
- **Commit conventions**: Conventional Commits with semantic prefixes
  - feat: new feature
  - fix: bug fix
  - docs: documentation changes
  - test: test additions/modifications
  - refactor: code refactoring
  - chore: maintenance tasks
- **Pull requests**: Required for all changes, peer review mandatory
- **Merge strategy**: Squash and merge for feature branches, merge for hotfixes
- **Release process**: Semantic versioning, tagged releases with changelogs

**Initial Setup Workflow**:
- **Bulk Import Process**: Historical medical literature ingestion from PubMed/MEDLINE, Cochrane Library, and other sources
- **Data Seeding**: Initial population of subject area tables with existing research
- **Batch Processing**: Efficient handling of large datasets (100K+ documents initially)
- **Quality Assessment**: Automated quality scoring and validation of historical content
- **Progress Tracking**: Monitoring and resuming interrupted bulk import processes
- **Archive Management**: Organizing and indexing pre-existing literature by subject areas

## Domain Context

**Medical Literature Analysis**

- **Subject areas**: Cardiology, Oncology, Neurology, Immunology, Pharmacology, etc.
- **Document types**: Research papers, clinical trials, case studies, reviews
- **Medical terminology**: MeSH terms, medical abbreviations, drug names
- **Quality indicators**: Peer review status, impact factor, citation count
- **Temporal relevance**: Publication date, research recency importance
- **Evidence levels**: RCTs, systematic reviews, case reports hierarchy
- **Regulatory considerations**: FDA, EMA guidelines for medical information
- **Data privacy**: Patient data anonymization, HIPAA compliance if applicable

**Key Medical Database Integrations**:
- PubMed/MEDLINE
- Cochrane Library
- ClinicalTrials.gov
- arXiv (medical physics/preprints)
- Web of Science
- Scopus

## Important Constraints

**Medical Accuracy Requirements**:
- High precision in medical information retrieval
- Source attribution for all information
- Clear indication of confidence levels
- No hallucinations in medical advice generation
- Evidence-based response prioritization

**Performance Constraints**:
- Query response time < 2 seconds (95th percentile)
- Support for 100+ concurrent users
- **Bulk Ingestion Performance**: Process 1000+ documents/hour during initial setup
- **Parallel Processing**: Multi-threaded ingestion for large historical datasets
- Real-time ingestion of new papers
- Scalable to millions of documents
- 99.9% uptime for critical clinical usage
- **Resume Capability**: Interruption and resumption of bulk import processes

**Data Quality Constraints**:
- Automatic quality scoring for ingested papers
- Duplicate detection and removal
- Proper medical terminology normalization
- Consistent subject area classification
- Version control for document updates

**Compliance Constraints**:
- GDPR/CCPA compliance for user data
- Medical device software regulations if applicable
- Data retention policies for research papers
- Audit trails for all system decisions
- Secure handling of sensitive medical information

## External Dependencies

**Medical Database APIs**:
- NCBI E-utilities (PubMed access)
- Cochrane Library API
- ClinicalTrials.gov API
- CrossRef DOI resolution service

**AI/ML Services**:
- OpenAI API (optional, for enhanced text generation)
- Hugging Face Transformers models
- Sentence Transformers pre-trained models
- Medical NER models (e.g., ClinicalBERT)

**Infrastructure Services**:
- Cloud storage for document repositories
- Vector database service (Pinecone/Weaviate cloud)
- CDN for document delivery
- Monitoring and logging services
- Backup and disaster recovery systems

**Research and Development**:
- Medical subject matter expert consultation
- Medical librarian expertise for taxonomy
- Clinical validation testing environments
- Peer review process for system accuracy
- Academic partnership for evaluation studies
