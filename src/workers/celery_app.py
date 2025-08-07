"""
Celery Application Configuration
Redis broker with JSON serialization and task registration system
"""

import json
import structlog
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
from celery import Celery, Task
from celery.signals import (
    worker_ready, worker_shutdown, task_prerun, task_postrun,
    task_failure, task_success, worker_process_init
)
from kombu import Queue

from config.settings import settings

logger = structlog.get_logger(__name__)


# Celery application configuration
celery_app = Celery(
    'finops-agent-chat',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'src.workers.tasks'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Task routing
    task_routes={
        'src.workers.tasks.execute_heavy_agent_task': {'queue': 'heavy_tasks'},
        'src.workers.tasks.process_financial_analysis': {'queue': 'finops_tasks'},
        'src.workers.tasks.process_document_analysis': {'queue': 'document_tasks'},
        'src.workers.tasks.process_security_analysis': {'queue': 'security_tasks'},
        'src.workers.tasks.process_research_task': {'queue': 'research_tasks'},
    },
    
    # Queue configuration
    task_create_missing_queues=True,
    task_queues=[
        Queue('heavy_tasks', routing_key='heavy_tasks', queue_arguments={'x-max-priority': 10}),
        Queue('finops_tasks', routing_key='finops_tasks', queue_arguments={'x-max-priority': 8}),
        Queue('document_tasks', routing_key='document_tasks', queue_arguments={'x-max-priority': 6}),
        Queue('security_tasks', routing_key='security_tasks', queue_arguments={'x-max-priority': 7}),
        Queue('research_tasks', routing_key='research_tasks', queue_arguments={'x-max-priority': 5}),
        Queue('default', routing_key='default', queue_arguments={'x-max-priority': 3}),
    ],
    
    # Retry configuration
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    
    # Result backend configuration
    result_expires=3600,  # 1 hour
    result_persistent=True,
    result_compression='gzip',
    
    # Worker configuration
    worker_disable_rate_limits=True,
    worker_max_tasks_per_child=1000,
    worker_max_memory_per_child=512000,  # 512MB
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Error handling
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,       # 10 minutes hard limit
    
    # Beat schedule (for periodic tasks)
    beat_schedule={
        'cleanup-expired-results': {
            'task': 'src.workers.tasks.cleanup_expired_results',
            'schedule': timedelta(hours=1),
        },
        'worker-health-check': {
            'task': 'src.workers.tasks.worker_health_check',
            'schedule': timedelta(minutes=5),
        },
    },
    timezone='UTC',
)


# Heavy task decorator registry
_heavy_task_registry: Dict[str, Dict[str, Any]] = {}


def heavy_task(
    name: Optional[str] = None,
    queue: str = 'heavy_tasks',
    priority: int = 5,
    retry_limit: int = 3,
    soft_time_limit: int = 300,
    time_limit: int = 600,
    rate_limit: Optional[str] = None,
    **task_kwargs
) -> Callable:
    """
    Decorator to mark agent tool methods as heavy tasks for async execution
    
    Args:
        name: Task name (defaults to function name)
        queue: Celery queue name
        priority: Task priority (1-10, higher = more important)
        retry_limit: Maximum retry attempts
        soft_time_limit: Soft timeout in seconds
        time_limit: Hard timeout in seconds
        rate_limit: Rate limiting (e.g., '10/m' for 10 per minute)
        **task_kwargs: Additional Celery task arguments
        
    Returns:
        Decorated function with heavy task metadata
    """
    def decorator(func: Callable) -> Callable:
        # Generate task name
        task_name = name or f"{func.__module__}.{func.__qualname__}"
        
        # Register heavy task metadata
        _heavy_task_registry[task_name] = {
            'function': func,
            'queue': queue,
            'priority': priority,
            'retry_limit': retry_limit,
            'soft_time_limit': soft_time_limit,
            'time_limit': time_limit,
            'rate_limit': rate_limit,
            'task_kwargs': task_kwargs,
            'registered_at': datetime.utcnow().isoformat()
        }
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check if we should execute locally or queue for Celery
            execution_context = kwargs.pop('execution_context', {})
            force_local = execution_context.get('force_local_execution', False)
            
            if force_local:
                # Execute locally for testing/development
                logger.debug(
                    "heavy_task_executing_locally",
                    task_name=task_name,
                    force_local=True
                )
                return await func(*args, **kwargs)
            
            # Queue for Celery execution
            from .tasks import execute_heavy_agent_task
            
            # Put execution_context back in kwargs for payload
            kwargs['execution_context'] = execution_context
            task_payload = {
                'task_name': task_name,
                'function_module': func.__module__,
                'function_name': func.__name__,
                'args': args[1:] if args else [],  # Skip 'self' argument
                'kwargs': kwargs,
                'execution_context': execution_context
            }
            
            logger.info(
                "heavy_task_queued",
                task_name=task_name,
                queue=queue,
                priority=priority
            )
            
            # Queue task with specified configuration
            async_result = execute_heavy_agent_task.apply_async(
                args=[task_payload],
                queue=queue,
                priority=priority,
                retry=retry_limit > 0,
                retry_policy={
                    'max_retries': retry_limit,
                    'interval_start': 0,
                    'interval_step': 0.2,
                    'interval_max': 0.2,
                } if retry_limit > 0 else None,
                soft_time_limit=soft_time_limit,
                time_limit=time_limit,
                rate_limit=rate_limit,
                **task_kwargs
            )
            
            return async_result
        
        # Attach metadata to wrapper
        wrapper._is_heavy_task = True
        wrapper._tool_name = task_name  # For backward compatibility
        wrapper._heavy_task_name = task_name
        wrapper._heavy_task_config = _heavy_task_registry[task_name]
        wrapper._original_func = func
        
        logger.debug(
            "heavy_task_registered",
            task_name=task_name,
            queue=queue,
            priority=priority
        )
        
        return wrapper
    
    return decorator


class MetricsCollectorTask(Task):
    """Custom Celery Task class with metrics collection"""
    
    def __init__(self):
        self.start_time = None
        self.task_metrics = {}
    
    def __call__(self, *args, **kwargs):
        """Execute task with metrics collection"""
        self.start_time = datetime.utcnow()
        
        try:
            result = super().__call__(*args, **kwargs)
            
            # Collect success metrics
            execution_time = (datetime.utcnow() - self.start_time).total_seconds() * 1000
            self.task_metrics.update({
                'execution_time_ms': execution_time,
                'status': 'success',
                'completed_at': datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            # Collect failure metrics
            execution_time = (datetime.utcnow() - self.start_time).total_seconds() * 1000
            self.task_metrics.update({
                'execution_time_ms': execution_time,
                'status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'completed_at': datetime.utcnow().isoformat()
            })
            raise


# Set custom task class
celery_app.Task = MetricsCollectorTask


# Celery signal handlers for monitoring and metrics
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready event"""
    logger.info(
        "celery_worker_ready",
        worker_hostname=sender.hostname if sender else 'unknown',
        registered_tasks=len(_heavy_task_registry),
        queues=list(celery_app.conf.task_routes.keys())
    )
    
    # Log registered heavy tasks
    for task_name, config in _heavy_task_registry.items():
        logger.debug(
            "heavy_task_available",
            task_name=task_name,
            queue=config['queue'],
            priority=config['priority']
        )


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown event"""
    logger.info(
        "celery_worker_shutdown",
        worker_hostname=sender.hostname if sender else 'unknown'
    )


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task pre-execution"""
    logger.info(
        "celery_task_started",
        task_id=task_id,
        task_name=task.name if task else 'unknown',
        queue=getattr(task, 'queue', 'default')
    )


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task post-execution"""
    logger.info(
        "celery_task_completed",
        task_id=task_id,
        task_name=task.name if task else 'unknown',
        state=state,
        has_result=retval is not None
    )


@task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    """Handle successful task completion"""
    logger.debug(
        "celery_task_success",
        task_name=sender.name if sender else 'unknown',
        result_type=type(result).__name__ if result else 'none'
    )


@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwargs):
    """Handle task failure"""
    logger.error(
        "celery_task_failed",
        task_id=task_id,
        task_name=sender.name if sender else 'unknown',
        exception=str(exception),
        exception_type=type(exception).__name__ if exception else 'unknown'
    )


@worker_process_init.connect
def worker_process_init_handler(**kwargs):
    """Handle worker process initialization"""
    logger.info("celery_worker_process_initialized")


# Utility functions for task management
def get_heavy_task_registry() -> Dict[str, Dict[str, Any]]:
    """Get registry of all heavy tasks"""
    return _heavy_task_registry.copy()


def is_heavy_task(func: Callable) -> bool:
    """Check if a function is marked as a heavy task"""
    return hasattr(func, '_is_heavy_task') and func._is_heavy_task


def get_task_config(func: Callable) -> Optional[Dict[str, Any]]:
    """Get heavy task configuration for a function"""
    if not is_heavy_task(func):
        return None
    return getattr(func, '_heavy_task_config', None)


def get_queue_stats() -> Dict[str, Any]:
    """Get statistics about Celery queues"""
    try:
        inspect = celery_app.control.inspect()
        active_queues = inspect.active_queues()
        reserved_tasks = inspect.reserved()
        active_tasks = inspect.active()
        
        return {
            'active_queues': active_queues,
            'reserved_tasks': reserved_tasks,
            'active_tasks': active_tasks,
            'registered_heavy_tasks': len(_heavy_task_registry)
        }
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {'error': str(e)}


def get_worker_stats() -> Dict[str, Any]:
    """Get Celery worker statistics"""
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        ping = inspect.ping()
        
        return {
            'worker_stats': stats,
            'worker_ping': ping,
            'broker_url': settings.CELERY_BROKER_URL,
            'result_backend': settings.CELERY_RESULT_BACKEND
        }
    except Exception as e:
        logger.error(f"Failed to get worker stats: {e}")
        return {'error': str(e)}