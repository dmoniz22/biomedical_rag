# Biomedical Literature Search System

A comprehensive RAG-powered system for searching and analyzing biomedical literature using state-of-the-art AI technologies.

## 🌟 Features

### Core Capabilities
- **RAG-Powered Search**: Advanced retrieval-augmented generation for semantic search of medical literature
- **Bulk Ingestion**: Automated bulk ingestion of medical papers from PubMed and other databases
- **Real-time Processing**: Dynamic subject area classification and automated table creation
- **Multi-modal Search**: Support for natural language, keyword, and MeSH term queries
- **Quality Assessment**: Automated quality scoring and validation of medical papers
- **Vector Embeddings**: Sentence transformers and ClinicalBERT for medical-specific embeddings

### Architecture
- **Microservices Design**: Modular, scalable architecture
- **Event-driven**: Pub/sub patterns for real-time updates
- **Vector Database**: ChromaDB for efficient similarity search
- **FastAPI**: High-performance async API layer
- **PostgreSQL**: Robust relational database for metadata
- **Redis**: Caching and session management

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Bulk Ingestion │    │   RAG Service   │
│   Frontend      │    │   Service        │    │                 │
│   (Port 8000)   │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌────────────────────────────┼────────────────────────────┐
    │                                                    │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │    ChromaDB      │    │     Redis       │
│   Metadata DB   │    │  Vector Store    │    │   Cache/Queue   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+

### Option 1: Docker Deployment (Recommended)

1. **Clone and setup**:
```bash
git clone <repository-url>
cd projects/Biomed_rag
```

2. **Start with Docker Compose**:
```bash
docker-compose up -d
```

3. **Setup initial data**:
```bash
docker-compose exec biomed_rag_api python setup_initial_data.py
```

4. **Access the system**:
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- API Base: http://localhost:8000/api/v1

### Option 2: Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Setup database**:
```bash
# Create PostgreSQL database
createdb biomed_rag

# Run setup script
python setup_initial_data.py
```

3. **Start the application**:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📋 API Endpoints

### Search Endpoints
- `POST /api/v1/search` - RAG-powered search
- `GET /api/v1/search/suggestions` - Query suggestions
- `POST /api/v1/search/enhance` - Query enhancement

### Bulk Ingestion
- `POST /api/v1/ingestion/bulk/start` - Start bulk ingestion
- `GET /api/v1/ingestion/bulk/status/{job_id}` - Check job status
- `POST /api/v1/ingestion/bulk/pause/{job_id}` - Pause job
- `POST /api/v1/ingestion/bulk/resume/{job_id}` - Resume job
- `DELETE /api/v1/ingestion/bulk/cancel/{job_id}` - Cancel job

### System Management
- `GET /api/v1/database/stats` - Database statistics
- `GET /api/v1/subject-areas` - Available subject areas
- `GET /api/v1/system/info` - System information

### Monitoring
- `GET /api/v1/monitoring/health` - Health check
- `GET /api/v1/monitoring/metrics` - Performance metrics
- `GET /metrics` - Prometheus metrics

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/biomed_rag
REDIS_URL=redis://localhost:6379/0

# API Keys
PUBMED_API_KEY=your_pubmed_api_key
COCHRANE_API_KEY=your_cochrane_api_key

# Vector Database
VECTOR_DB_PATH=./data/vector_db
VECTOR_DB_TYPE=chroma

# ML Models
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
CLINICAL_BERT_MODEL=dmis-lab/biobert-base-cased-v1.1

# Performance
MAX_WORKERS=4
BATCH_SIZE=100
CHUNK_SIZE=1000
```

### Subject Areas Configuration
The system supports dynamic subject area classification:

```python
SUBJECT_AREAS = [
    "cardiology", "oncology", "neurology", "immunology", "endocrinology",
    "gastroenterology", "nephrology", "pulmonology", "rheumatology",
    "dermatology", "ophthalmology", "psychiatry", "pediatrics", "geriatrics"
]
```

## 📊 Initial Data Seeding

The system includes a comprehensive setup script that:

1. **Database Setup**: Creates all tables and indexes
2. **Subject Areas**: Initializes medical subject classifications
3. **Sample Data**: Creates test papers for development
4. **Bulk Ingestion**: Starts initial PubMed data fetching

Run the setup:
```bash
python setup_initial_data.py
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_rag_service.py

# Run with verbose output
pytest -v
```

### Test Structure
- `tests/conftest.py` - Test fixtures and configuration
- `tests/test_rag_service.py` - RAG service tests
- `tests/test_api.py` - API endpoint tests

## 📈 Monitoring & Logging

### Health Monitoring
- Real-time system health checks
- Component status monitoring (database, vector DB, API)
- Performance metrics collection
- Prometheus metrics for observability

### Logging
- Structured logging with JSON format
- Error tracking and alerting
- Request/response logging
- Performance profiling

### Access Monitoring Data
```bash
# System health
curl http://localhost:8000/api/v1/monitoring/health

# Performance metrics
curl http://localhost:8000/api/v1/monitoring/metrics

# Prometheus metrics
curl http://localhost:8000/metrics
```

## 🏭 Production Deployment

### Docker Production
```bash
# Build production image
docker build -t biomed-rag:latest .

# Run with production settings
docker run -d \
  --name biomed-rag \
  -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DATABASE_URL=postgresql://... \
  biomed-rag:latest
```

### Environment Setup
1. **Configure production database**
2. **Set up SSL certificates**
3. **Configure monitoring and alerting**
4. **Set up backup procedures**
5. **Configure auto-scaling**

## 🔄 Data Lifecycle

### Initial Setup (One-time)
1. Database schema creation
2. Subject area initialization
3. Historical data bulk ingestion
4. Vector embedding generation

### Ongoing Operations
1. **Real-time Updates**: Automatic ingestion of new papers
2. **Quality Assessment**: Automated paper validation
3. **Index Updates**: Continuous vector database updates
4. **System Maintenance**: Health monitoring and optimization

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   ```bash
   # Check database status
   docker-compose exec postgres psql -U biomed_user -d biomed_rag -c "SELECT 1;"
   ```

2. **Vector Database Issues**
   ```bash
   # Reset vector database (development only)
   curl -X DELETE http://localhost:8000/api/v1/vector/reset
   ```

3. **Memory Issues During Bulk Ingestion**
   - Reduce batch size in configuration
   - Enable progress monitoring
   - Use pause/resume functionality

4. **Performance Issues**
   - Check system metrics endpoint
   - Monitor active ingestion jobs
   - Review error rates in logs

### Logs
```bash
# View application logs
docker-compose logs -f biomed_rag_api

# View all service logs
docker-compose logs -f

# Follow specific service
docker-compose logs -f biomed_rag_worker
```

## 📚 Documentation

### OpenAPI Documentation
Access interactive API documentation at http://localhost:8000/docs

### System Architecture
- OpenSpec workflow documentation in `openspec/`
- Component documentation in respective service files
- Database schema in `app/models/database.py`

## 🤝 Contributing

1. Follow the OpenSpec workflow
2. Maintain 90%+ test coverage
3. Document all public APIs
4. Use structured logging
5. Follow PEP 8 style guidelines

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review system logs
3. Consult API documentation
4. Create an issue in the repository

---

**Built with ❤️ for the biomedical research community**