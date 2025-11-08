"""
Monitoring service for biomedical RAG system
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from dataclasses import dataclass
from collections import defaultdict, deque

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    response_time_avg: float
    requests_per_minute: int
    error_rate: float
    timestamp: datetime


class MonitoringService:
    """Service for monitoring system performance and health"""
    
    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter(
            'biomed_rag_requests_total',
            'Total requests received',
            ['endpoint', 'method', 'status_code']
        )
        
        self.request_duration = Histogram(
            'biomed_rag_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint']
        )
        
        self.active_ingestion_jobs = Gauge(
            'biomed_rag_active_ingestion_jobs',
            'Number of active ingestion jobs'
        )
        
        self.database_connections = Gauge(
            'biomed_rag_database_connections',
            'Active database connections'
        )
        
        self.vector_db_documents = Gauge(
            'biomed_rag_vector_db_documents',
            'Number of documents in vector database',
            ['content_type']
        )
        
        # Internal metrics storage
        self.request_times = deque(maxlen=1000)  # Keep last 1000 request times
        self.endpoint_stats = defaultdict(lambda: {
            'count': 0,
            'errors': 0,
            'total_duration': 0.0
        })
        
    def record_request(self, endpoint: str, method: str, status_code: int, 
                      duration: float):
        """Record API request metrics"""
        self.request_count.labels(
            endpoint=endpoint,
            method=method,
            status_code=status_code
        ).inc()
        
        self.request_duration.labels(endpoint=endpoint).observe(duration)
        
        # Store for internal monitoring
        self.request_times.append(duration)
        
        stats = self.endpoint_stats[endpoint]
        stats['count'] += 1
        stats['total_duration'] += duration
        
        if status_code >= 400:
            stats['errors'] += 1
    
    def record_ingestion_job(self, job_id: str, status: str, documents_processed: int = 0):
        """Record ingestion job metrics"""
        if status == 'started':
            self.active_ingestion_jobs.inc()
        elif status in ['completed', 'failed', 'cancelled']:
            self.active_ingestion_jobs.dec()
        
        # Log job events
        logger.info(f"Ingestion job {job_id}: {status} (docs: {documents_processed})")
    
    def record_vector_operation(self, operation: str, document_count: int = 1):
        """Record vector database operations"""
        # This would be called by the vector database service
        logger.debug(f"Vector DB operation: {operation} ({document_count} documents)")
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics"""
        try:
            import psutil
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate request metrics
            now = time.time()
            recent_requests = [
                rt for rt in self.request_times 
                if now - rt < 300  # Last 5 minutes
            ]
            avg_response_time = (
                sum(recent_requests) / len(recent_requests) 
                if recent_requests else 0.0
            )
            requests_per_minute = len(recent_requests) / 5.0
            
            # Calculate error rate
            total_requests = sum(stats['count'] for stats in self.endpoint_stats.values())
            total_errors = sum(stats['errors'] for stats in self.endpoint_stats.values())
            error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0
            
            return SystemMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                active_connections=0,  # Would get from database
                response_time_avg=avg_response_time,
                requests_per_minute=int(requests_per_minute),
                error_rate=error_rate,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0, datetime.now())
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
            "metrics": {}
        }
        
        try:
            # Check database health
            db_healthy = await self._check_database_health()
            health_status["components"]["database"] = {
                "status": "healthy" if db_healthy else "unhealthy",
                "message": "Database connection is working" if db_healthy else "Database connection failed"
            }
            
            # Check vector database health
            vector_healthy = await self._check_vector_db_health()
            health_status["components"]["vector_db"] = {
                "status": "healthy" if vector_healthy else "unhealthy",
                "message": "Vector database is working" if vector_healthy else "Vector database connection failed"
            }
            
            # Check system resources
            metrics = await self.collect_system_metrics()
            health_status["metrics"] = {
                "cpu_usage": f"{metrics.cpu_usage:.1f}%",
                "memory_usage": f"{metrics.memory_usage:.1f}%",
                "disk_usage": f"{metrics.disk_usage:.1f}%",
                "response_time_avg": f"{metrics.response_time_avg:.3f}s",
                "requests_per_minute": metrics.requests_per_minute,
                "error_rate": f"{metrics.error_rate:.2%}"
            }
            
            # Determine overall health
            component_statuses = [
                comp["status"] for comp in health_status["components"].values()
            ]
            
            if "unhealthy" in component_statuses:
                health_status["status"] = "unhealthy"
            elif metrics.memory_usage > 90 or metrics.cpu_usage > 90:
                health_status["status"] = "degraded"
            elif metrics.error_rate > 0.1:  # > 10% error rate
                health_status["status"] = "degraded"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status.update({
                "status": "error",
                "error": str(e)
            })
        
        return health_status
    
    async def _check_database_health(self) -> bool:
        """Check database connection health"""
        try:
            from app.core.database import db_manager
            return await db_manager.health_check()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def _check_vector_db_health(self) -> bool:
        """Check vector database health"""
        try:
            from app.services.vector_db_service import vector_db_service
            # Try to get collection stats as a health check
            stats = await vector_db_service.get_collection_stats()
            return isinstance(stats, dict)
        except Exception as e:
            logger.error(f"Vector DB health check failed: {e}")
            return False
    
    def get_endpoint_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all endpoints"""
        stats = {}
        for endpoint, data in self.endpoint_stats.items():
            if data['count'] > 0:
                stats[endpoint] = {
                    'total_requests': data['count'],
                    'errors': data['errors'],
                    'error_rate': data['errors'] / data['count'],
                    'avg_duration': data['total_duration'] / data['count']
                }
        return stats
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        metrics = await self.collect_system_metrics()
        endpoint_stats = self.get_endpoint_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "cpu_usage": f"{metrics.cpu_usage:.1f}%",
                "memory_usage": f"{metrics.memory_usage:.1f}%",
                "disk_usage": f"{metrics.disk_usage:.1f}%",
                "response_time_avg": f"{metrics.response_time_avg:.3f}s",
                "requests_per_minute": metrics.requests_per_minute,
                "error_rate": f"{metrics.error_rate:.2%}"
            },
            "endpoint_statistics": endpoint_stats,
            "active_ingestion_jobs": int(self.active_ingestion_jobs._value._value),
            "recommendations": self._generate_recommendations(metrics, endpoint_stats)
        }
    
    def _generate_recommendations(self, metrics: SystemMetrics, 
                                 endpoint_stats: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if metrics.memory_usage > 80:
            recommendations.append("High memory usage detected. Consider scaling up or optimizing memory usage.")
        
        if metrics.cpu_usage > 80:
            recommendations.append("High CPU usage detected. Consider scaling up or optimizing CPU-intensive operations.")
        
        if metrics.error_rate > 0.05:
            recommendations.append(f"High error rate ({metrics.error_rate:.2%}). Check application logs for issues.")
        
        # Check for slow endpoints
        for endpoint, stats in endpoint_stats.items():
            if stats['avg_duration'] > 2.0:  # > 2 seconds
                recommendations.append(f"Endpoint {endpoint} is slow (avg: {stats['avg_duration']:.2f}s). Consider optimization.")
        
        if not recommendations:
            recommendations.append("System performance is within normal parameters.")
        
        return recommendations
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest().decode('utf-8')


# Global monitoring service instance
monitoring_service = MonitoringService()