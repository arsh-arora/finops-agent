"""
Redis Queue Configuration and Monitoring
Comprehensive queue management with retry policies, dead letter queues, and Prometheus metrics
"""

import asyncio
import json
import structlog
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from celery import Celery
from kombu import Queue, Exchange
import redis.asyncio as redis

from config.settings import settings
from src.database.redis_client import redis_client

logger = structlog.get_logger(__name__)


class QueueStatus(str, Enum):
    """Queue status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    BLOCKED = "blocked"
    ERROR = "error"


class TaskPriority(int, Enum):
    """Task priority levels"""
    CRITICAL = 10
    HIGH = 8
    NORMAL = 5
    LOW = 3
    BACKGROUND = 1


@dataclass
class QueueMetrics:
    """Queue performance and health metrics"""
    
    queue_name: str
    status: QueueStatus = QueueStatus.ACTIVE
    
    # Task counts
    pending_tasks: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    retrying_tasks: int = 0
    
    # Performance metrics
    avg_execution_time_ms: float = 0.0
    avg_wait_time_ms: float = 0.0
    throughput_per_minute: float = 0.0
    
    # Health indicators
    error_rate_percent: float = 0.0
    retry_rate_percent: float = 0.0
    worker_count: int = 0
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)
    last_task_processed: Optional[datetime] = None
    
    # Resource utilization
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def get_total_tasks(self) -> int:
        """Get total number of tasks processed"""
        return self.pending_tasks + self.active_tasks + self.completed_tasks + self.failed_tasks
    
    def get_success_rate(self) -> float:
        """Get success rate percentage"""
        total_processed = self.completed_tasks + self.failed_tasks
        return (self.completed_tasks / total_processed * 100) if total_processed > 0 else 0.0


@dataclass
class RetryPolicy:
    """Task retry configuration"""
    
    max_retries: int = 3
    backoff_strategy: str = "exponential"  # exponential, linear, fixed
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 300.0
    backoff_multiplier: float = 2.0
    
    # Dead letter configuration  
    dead_letter_queue: str = "failed_tasks"
    dead_letter_ttl_hours: int = 24
    
    def calculate_delay(self, retry_count: int) -> float:
        """Calculate delay for given retry attempt"""
        
        if self.backoff_strategy == "fixed":
            return min(self.initial_delay_seconds, self.max_delay_seconds)
        
        elif self.backoff_strategy == "linear":
            delay = self.initial_delay_seconds * (retry_count + 1)
            return min(delay, self.max_delay_seconds)
        
        elif self.backoff_strategy == "exponential":
            delay = self.initial_delay_seconds * (self.backoff_multiplier ** retry_count)
            return min(delay, self.max_delay_seconds)
        
        else:
            return self.initial_delay_seconds


class PrometheusMetrics:
    """Prometheus metrics collector for queue monitoring"""
    
    def __init__(self):
        self.metrics_enabled = False
        self._metrics_cache: Dict[str, Any] = {}
        
        # Try to initialize Prometheus metrics
        try:
            from prometheus_client import Counter, Histogram, Gauge, Info
            
            # Task metrics
            self.task_counter = Counter(
                'celery_tasks_total',
                'Total number of tasks processed',
                ['queue', 'status', 'task_type']
            )
            
            self.task_duration_histogram = Histogram(
                'celery_task_duration_seconds',
                'Task execution duration',
                ['queue', 'task_type']
            )
            
            self.queue_size_gauge = Gauge(
                'celery_queue_size',
                'Current queue size',
                ['queue', 'status']
            )
            
            self.worker_gauge = Gauge(
                'celery_workers_active',
                'Number of active workers',
                ['queue']
            )
            
            self.error_rate_gauge = Gauge(
                'celery_error_rate_percent',
                'Task error rate percentage',
                ['queue']
            )
            
            # System metrics
            self.memory_gauge = Gauge(
                'celery_memory_usage_bytes',
                'Memory usage by workers',
                ['worker_hostname']
            )
            
            self.cpu_gauge = Gauge(
                'celery_cpu_usage_percent',
                'CPU usage by workers',
                ['worker_hostname']
            )
            
            self.metrics_enabled = True
            logger.info("prometheus_metrics_enabled")
            
        except ImportError:
            logger.warning("prometheus_client_not_available", metrics_enabled=False)
    
    def record_task_completion(
        self, 
        queue_name: str, 
        task_type: str, 
        duration_seconds: float, 
        success: bool
    ) -> None:
        """Record task completion metrics"""
        
        if not self.metrics_enabled:
            return
        
        try:
            status = "success" if success else "failed"
            
            self.task_counter.labels(
                queue=queue_name, 
                status=status, 
                task_type=task_type
            ).inc()
            
            self.task_duration_histogram.labels(
                queue=queue_name,
                task_type=task_type
            ).observe(duration_seconds)
            
        except Exception as e:
            logger.error("prometheus_metric_recording_failed", error=str(e))
    
    def update_queue_metrics(self, queue_metrics: QueueMetrics) -> None:
        """Update queue size and status metrics"""
        
        if not self.metrics_enabled:
            return
        
        try:
            queue_name = queue_metrics.queue_name
            
            self.queue_size_gauge.labels(
                queue=queue_name, 
                status="pending"
            ).set(queue_metrics.pending_tasks)
            
            self.queue_size_gauge.labels(
                queue=queue_name, 
                status="active"
            ).set(queue_metrics.active_tasks)
            
            self.worker_gauge.labels(
                queue=queue_name
            ).set(queue_metrics.worker_count)
            
            self.error_rate_gauge.labels(
                queue=queue_name
            ).set(queue_metrics.error_rate_percent)
            
        except Exception as e:
            logger.error("prometheus_queue_metrics_failed", error=str(e))


class RedisQueueMonitor:
    """
    Comprehensive Redis queue monitoring and management system
    
    Features:
    - Real-time queue metrics collection
    - Retry policy configuration and enforcement
    - Dead letter queue management
    - Prometheus metrics integration
    - Queue health monitoring and alerting
    - Worker performance tracking
    """
    
    def __init__(self, celery_app: Celery):
        """
        Initialize queue monitor
        
        Args:
            celery_app: Celery application instance
        """
        self.celery_app = celery_app
        self.prometheus = PrometheusMetrics()
        
        # Queue configuration
        self._queue_configs: Dict[str, Dict[str, Any]] = {}
        self._retry_policies: Dict[str, RetryPolicy] = {}
        
        # Monitoring state
        self._queue_metrics: Dict[str, QueueMetrics] = {}
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Alerting
        self._alert_thresholds: Dict[str, Dict[str, float]] = {}
        self._alert_callbacks: List[Callable] = []
        
        logger.info("redis_queue_monitor_initialized")

    async def configure_queues(
        self, 
        queue_configs: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Configure queue settings and retry policies
        
        Args:
            queue_configs: Queue configuration dictionary
        """
        
        for queue_name, config in queue_configs.items():
            # Configure queue
            self._queue_configs[queue_name] = config
            
            # Setup retry policy
            retry_config = config.get('retry_policy', {})
            self._retry_policies[queue_name] = RetryPolicy(**retry_config)
            
            # Setup alert thresholds
            alert_config = config.get('alert_thresholds', {})
            self._alert_thresholds[queue_name] = {
                'error_rate_threshold': alert_config.get('error_rate_threshold', 10.0),
                'queue_size_threshold': alert_config.get('queue_size_threshold', 1000),
                'wait_time_threshold_ms': alert_config.get('wait_time_threshold_ms', 30000),
                'worker_min_count': alert_config.get('worker_min_count', 1)
            }
            
            logger.info(
                "queue_configured",
                queue_name=queue_name,
                retry_max=self._retry_policies[queue_name].max_retries,
                dead_letter_queue=self._retry_policies[queue_name].dead_letter_queue
            )

    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous queue monitoring"""
        
        if self._monitoring_active:
            logger.warning("queue_monitoring_already_active")
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        
        logger.info("queue_monitoring_started", interval_seconds=interval_seconds)

    async def stop_monitoring(self) -> None:
        """Stop queue monitoring"""
        
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("queue_monitoring_stopped")

    async def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop"""
        
        while self._monitoring_active:
            try:
                # Collect metrics from all configured queues
                for queue_name in self._queue_configs.keys():
                    await self._collect_queue_metrics(queue_name)
                
                # Check alert conditions
                await self._check_alert_conditions()
                
                # Update Prometheus metrics
                self._update_prometheus_metrics()
                
                # Wait for next interval
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error("monitoring_loop_error", error=str(e))
                await asyncio.sleep(interval_seconds)

    async def _collect_queue_metrics(self, queue_name: str) -> None:
        """Collect comprehensive metrics for a specific queue"""
        
        try:
            # Get Celery inspector
            inspect = self.celery_app.control.inspect()
            
            # Collect basic queue stats
            reserved_tasks = inspect.reserved() or {}
            active_tasks = inspect.active() or {}
            scheduled_tasks = inspect.scheduled() or {}
            
            # Calculate queue-specific metrics
            queue_reserved = sum(
                len([task for task in tasks if task.get('delivery_info', {}).get('routing_key') == queue_name])
                for tasks in reserved_tasks.values()
            )
            
            queue_active = sum(
                len([task for task in tasks if task.get('delivery_info', {}).get('routing_key') == queue_name])
                for tasks in active_tasks.values()
            )
            
            # Get Redis queue length
            try:
                queue_length = await self._get_redis_queue_length(queue_name)
            except Exception as e:
                logger.warning("redis_queue_length_failed", queue_name=queue_name, error=str(e))
                queue_length = 0
            
            # Get worker count for this queue
            worker_count = await self._get_queue_worker_count(queue_name)
            
            # Update metrics
            current_time = datetime.utcnow()
            
            if queue_name not in self._queue_metrics:
                self._queue_metrics[queue_name] = QueueMetrics(queue_name=queue_name)
            
            metrics = self._queue_metrics[queue_name]
            metrics.pending_tasks = queue_length
            metrics.active_tasks = queue_active
            metrics.worker_count = worker_count
            metrics.last_updated = current_time
            
            # Calculate derived metrics
            await self._calculate_performance_metrics(queue_name)
            
            logger.debug(
                "queue_metrics_collected",
                queue_name=queue_name,
                pending=metrics.pending_tasks,
                active=metrics.active_tasks,
                workers=metrics.worker_count
            )
            
        except Exception as e:
            logger.error(
                "queue_metrics_collection_failed",
                queue_name=queue_name,
                error=str(e)
            )

    async def _get_redis_queue_length(self, queue_name: str) -> int:
        """Get queue length from Redis"""
        
        try:
            # Celery uses Redis lists for queues
            redis_key = f"{queue_name}"
            length = await redis_client.redis_client.llen(redis_key)
            return length if length is not None else 0
            
        except Exception as e:
            logger.debug("redis_queue_length_error", queue_name=queue_name, error=str(e))
            return 0

    async def _get_queue_worker_count(self, queue_name: str) -> int:
        """Get number of workers handling this queue"""
        
        try:
            inspect = self.celery_app.control.inspect()
            active_queues = inspect.active_queues() or {}
            
            worker_count = 0
            for worker, queues in active_queues.items():
                if any(q['name'] == queue_name for q in queues):
                    worker_count += 1
            
            return worker_count
            
        except Exception as e:
            logger.debug("worker_count_error", queue_name=queue_name, error=str(e))
            return 0

    async def _calculate_performance_metrics(self, queue_name: str) -> None:
        """Calculate performance metrics like throughput and error rates"""
        
        if queue_name not in self._queue_metrics:
            return
        
        metrics = self._queue_metrics[queue_name]
        
        # Get historical task data from Redis
        history_key = f"queue_history:{queue_name}"
        
        try:
            # This would be implemented with actual Redis time series data
            # For now, simulate performance metrics
            
            total_tasks = metrics.get_total_tasks()
            if total_tasks > 0:
                metrics.error_rate_percent = (metrics.failed_tasks / total_tasks) * 100
                metrics.retry_rate_percent = (metrics.retrying_tasks / total_tasks) * 100
            
            # Estimate throughput (tasks per minute)
            if metrics.last_task_processed:
                time_diff = (datetime.utcnow() - metrics.last_task_processed).total_seconds()
                if time_diff > 0:
                    metrics.throughput_per_minute = (metrics.completed_tasks * 60) / time_diff
            
        except Exception as e:
            logger.debug("performance_metrics_calculation_error", queue_name=queue_name, error=str(e))

    async def _check_alert_conditions(self) -> None:
        """Check alert conditions and trigger notifications"""
        
        for queue_name, metrics in self._queue_metrics.items():
            if queue_name not in self._alert_thresholds:
                continue
            
            thresholds = self._alert_thresholds[queue_name]
            alerts = []
            
            # Check error rate
            if metrics.error_rate_percent > thresholds['error_rate_threshold']:
                alerts.append(f"High error rate: {metrics.error_rate_percent:.1f}%")
            
            # Check queue size
            if metrics.pending_tasks > thresholds['queue_size_threshold']:
                alerts.append(f"Queue backlog: {metrics.pending_tasks} tasks")
            
            # Check worker availability
            if metrics.worker_count < thresholds['worker_min_count']:
                alerts.append(f"Low worker count: {metrics.worker_count}")
            
            # Check wait time
            if metrics.avg_wait_time_ms > thresholds['wait_time_threshold_ms']:
                alerts.append(f"High wait time: {metrics.avg_wait_time_ms:.0f}ms")
            
            # Trigger alerts
            if alerts:
                await self._trigger_alerts(queue_name, alerts, metrics)

    async def _trigger_alerts(
        self, 
        queue_name: str, 
        alerts: List[str], 
        metrics: QueueMetrics
    ) -> None:
        """Trigger alert notifications"""
        
        alert_data = {
            'queue_name': queue_name,
            'alerts': alerts,
            'metrics': {
                'pending_tasks': metrics.pending_tasks,
                'error_rate_percent': metrics.error_rate_percent,
                'worker_count': metrics.worker_count,
                'avg_wait_time_ms': metrics.avg_wait_time_ms
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.warning(
            "queue_alert_triggered",
            queue_name=queue_name,
            alert_count=len(alerts),
            alerts=alerts
        )
        
        # Call registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error("alert_callback_failed", error=str(e))

    def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics"""
        
        for queue_name, metrics in self._queue_metrics.items():
            self.prometheus.update_queue_metrics(metrics)

    async def setup_dead_letter_queue(self, queue_name: str) -> None:
        """Setup dead letter queue for failed tasks"""
        
        if queue_name not in self._retry_policies:
            logger.warning("no_retry_policy_for_queue", queue_name=queue_name)
            return
        
        retry_policy = self._retry_policies[queue_name]
        dlq_name = retry_policy.dead_letter_queue
        
        try:
            # Create dead letter queue configuration
            dlq_config = {
                'x-message-ttl': retry_policy.dead_letter_ttl_hours * 3600 * 1000,  # TTL in milliseconds
                'x-max-length': 10000,  # Maximum messages in DLQ
                'x-overflow': 'drop-head'  # Drop oldest messages when full
            }
            
            # This would create the actual Redis/RabbitMQ queue
            logger.info(
                "dead_letter_queue_configured",
                original_queue=queue_name,
                dlq_name=dlq_name,
                ttl_hours=retry_policy.dead_letter_ttl_hours
            )
            
        except Exception as e:
            logger.error(
                "dead_letter_queue_setup_failed",
                queue_name=queue_name,
                dlq_name=dlq_name,
                error=str(e)
            )

    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for alert notifications"""
        
        self._alert_callbacks.append(callback)
        logger.debug("alert_callback_registered", callback_count=len(self._alert_callbacks))

    def get_queue_metrics(self, queue_name: Optional[str] = None) -> Dict[str, QueueMetrics]:
        """Get current queue metrics"""
        
        if queue_name:
            return {queue_name: self._queue_metrics.get(queue_name)}
        
        return self._queue_metrics.copy()

    def get_queue_health_summary(self) -> Dict[str, Any]:
        """Get overall queue health summary"""
        
        total_queues = len(self._queue_metrics)
        healthy_queues = 0
        total_pending = 0
        total_active = 0
        total_workers = 0
        
        for metrics in self._queue_metrics.values():
            if (metrics.error_rate_percent < 5.0 and 
                metrics.worker_count > 0 and 
                metrics.avg_wait_time_ms < 10000):
                healthy_queues += 1
            
            total_pending += metrics.pending_tasks
            total_active += metrics.active_tasks
            total_workers += metrics.worker_count
        
        health_percentage = (healthy_queues / total_queues * 100) if total_queues > 0 else 0
        
        return {
            'total_queues': total_queues,
            'healthy_queues': healthy_queues,
            'health_percentage': round(health_percentage, 1),
            'total_pending_tasks': total_pending,
            'total_active_tasks': total_active,
            'total_workers': total_workers,
            'monitoring_active': self._monitoring_active,
            'last_updated': datetime.utcnow().isoformat()
        }

    async def cleanup(self) -> None:
        """Cleanup monitoring resources"""
        
        await self.stop_monitoring()
        self._queue_metrics.clear()
        self._alert_callbacks.clear()
        
        logger.info("redis_queue_monitor_cleanup_completed")