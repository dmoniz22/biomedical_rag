"""
Celery configuration for the Biomedical RAG System
"""

import os
from celery import Celery
from app.core.config import settings

# Initialize Celery
celery_app = Celery(
    "biomed_rag",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.core.celery",
        "app.services.bulk_ingestion_service",
    ]
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,  # 1 hour
    # Task routing
    task_routes={
        'app.services.bulk_ingestion_service.*': {'queue': 'bulk_ingestion'},
        'app.services.monitoring_service.*': {'queue': 'monitoring'},
    },
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-old-embeddings': {
            'task': 'app.services.monitoring_service.cleanup_old_embeddings',
            'schedule': 24 * 60 * 60,  # Daily
        },
        'update-system-metrics': {
            'task': 'app.services.monitoring_service.update_system_metrics',
            'schedule': 5 * 60,  # Every 5 minutes
        },
    },
)

# Health check task
@celery_app.task(bind=True)
def health_check(self):
    """Health check task for monitoring Celery worker status"""
    return {
        'status': 'healthy',
        'worker_id': self.request.hostname,
        'timestamp': self.request.timestamp,
    }

# Graceful shutdown
def shutdown_handler(signum, frame):
    """Handle graceful shutdown of Celery worker"""
    celery_app.control.shutdown()
    raise SystemExit(0)

# Make sure celery_app is accessible from other modules
if __name__ == '__main__':
    celery_app.start()
