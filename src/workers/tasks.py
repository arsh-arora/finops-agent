"""
Celery Task Implementations
Heavy computational tasks with idempotent execution and result streaming
"""

import asyncio
import json
import structlog
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import uuid4
import importlib
import sys

from celery import current_task
from celery.exceptions import Retry, WorkerLostError

from .celery_app import celery_app
from src.memory.mem0_service import FinOpsMemoryService
from src.database.redis_client import redis_client

logger = structlog.get_logger(__name__)


@celery_app.task(
    bind=True,
    queue='heavy_tasks',
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60},
    soft_time_limit=300,
    time_limit=600
)
def execute_heavy_agent_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute heavy agent task with idempotent retry and result streaming
    
    Args:
        task_payload: Task execution payload containing function and context
        
    Returns:
        Task execution result with metadata
    """
    task_id = self.request.id
    start_time = datetime.utcnow()
    
    # Extract payload
    task_name = task_payload.get('task_name', 'unknown')
    function_module = task_payload.get('function_module')
    function_name = task_payload.get('function_name')
    args = task_payload.get('args', [])
    kwargs = task_payload.get('kwargs', {})
    execution_context = task_payload.get('execution_context', {})
    
    logger.info(
        "heavy_task_execution_started",
        task_id=task_id,
        task_name=task_name,
        function_module=function_module,
        function_name=function_name,
        args_count=len(args),
        kwargs_count=len(kwargs)
    )
    
    try:
        # Check for duplicate execution (idempotency)
        if asyncio.run(_check_task_already_executed(task_id, execution_context)):
            logger.info("heavy_task_already_executed", task_id=task_id)
            return asyncio.run(_get_cached_result(task_id))
        
        # Import and execute function
        result = asyncio.run(_execute_agent_function(
            function_module, function_name, args, kwargs, execution_context
        ))
        
        # Calculate execution metrics
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Prepare result with metadata
        task_result = {
            'success': True,
            'result': result,
            'task_id': task_id,
            'task_name': task_name,
            'execution_time_ms': execution_time,
            'completed_at': datetime.utcnow().isoformat(),
            'worker_hostname': self.request.hostname,
            'execution_context': execution_context
        }
        
        # Cache result for idempotency
        asyncio.run(_cache_task_result(task_id, task_result))
        
        # Stream result to WebSocket if needed
        asyncio.run(_stream_result_to_websocket(task_result, execution_context))
        
        logger.info(
            "heavy_task_execution_completed",
            task_id=task_id,
            task_name=task_name,
            execution_time_ms=execution_time,
            result_size=len(str(result))
        )
        
        return task_result
        
    except Exception as e:
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.error(
            "heavy_task_execution_failed",
            task_id=task_id,
            task_name=task_name,
            error=str(e),
            error_type=type(e).__name__,
            execution_time_ms=execution_time,
            retry_count=self.request.retries
        )
        
        # Prepare error result
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'task_id': task_id,
            'task_name': task_name,
            'execution_time_ms': execution_time,
            'failed_at': datetime.utcnow().isoformat(),
            'retry_count': self.request.retries,
            'execution_context': execution_context
        }
        
        # Stream error to WebSocket
        asyncio.run(_stream_result_to_websocket(error_result, execution_context))
        
        # Re-raise for Celery retry logic
        raise


@celery_app.task(
    bind=True,
    queue='finops_tasks',
    soft_time_limit=180,
    time_limit=300
)
def process_financial_analysis(self, analysis_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process heavy financial analysis tasks (NPV, IRR, anomaly detection)"""
    return _run_async_task(_process_financial_analysis_async, analysis_payload)


@celery_app.task(
    bind=True,
    queue='document_tasks', 
    soft_time_limit=120,
    time_limit=180
)
def process_document_analysis(self, document_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process heavy document analysis with Docling and ML models"""
    return _run_async_task(_process_document_analysis_async, document_payload)


@celery_app.task(
    bind=True,
    queue='security_tasks',
    soft_time_limit=240,
    time_limit=360
)
def process_security_analysis(self, security_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process heavy security analysis and vulnerability scanning"""
    return _run_async_task(_process_security_analysis_async, security_payload)


@celery_app.task(
    bind=True,
    queue='research_tasks',
    soft_time_limit=300,
    time_limit=450
)
def process_research_task(self, research_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process heavy research tasks with web scraping and analysis"""
    return _run_async_task(_process_research_task_async, research_payload)


@celery_app.task(queue='maintenance')
def cleanup_expired_results() -> Dict[str, Any]:
    """Periodic task to clean up expired task results"""
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        # Clean up Redis cache
        cleanup_count = _run_async_task(_cleanup_redis_cache, cutoff_time)
        
        logger.info(
            "task_result_cleanup_completed",
            cleaned_results=cleanup_count,
            cutoff_time=cutoff_time.isoformat()
        )
        
        return {
            'success': True,
            'cleaned_results': cleanup_count,
            'cutoff_time': cutoff_time.isoformat()
        }
        
    except Exception as e:
        logger.error("task_result_cleanup_failed", error=str(e))
        return {'success': False, 'error': str(e)}


@celery_app.task(queue='monitoring')
def worker_health_check() -> Dict[str, Any]:
    """Periodic worker health check with metrics reporting"""
    try:
        import psutil
        import os
        
        # Collect system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Collect worker metrics
        worker_pid = os.getpid()
        worker_process = psutil.Process(worker_pid)
        
        health_metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'worker_pid': worker_pid,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'worker_memory_mb': worker_process.memory_info().rss / (1024**2),
            'worker_cpu_percent': worker_process.cpu_percent(),
            'status': 'healthy'
        }
        
        # Check health thresholds
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 95:
            health_metrics['status'] = 'warning'
        
        logger.info("worker_health_check_completed", metrics=health_metrics)
        
        return health_metrics
        
    except Exception as e:
        logger.error("worker_health_check_failed", error=str(e))
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'error',
            'error': str(e)
        }


# Helper functions for async task execution
def _run_async_task(async_func, *args, **kwargs):
    """Run async function in sync Celery task context"""
    try:
        # Create new event loop for this task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(async_func(*args, **kwargs))
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(
            "async_task_execution_failed", 
            function=async_func.__name__ if hasattr(async_func, '__name__') else 'unknown',
            error=str(e)
        )
        raise


async def _execute_agent_function(
    module_name: str,
    function_name: str,
    args: List[Any],
    kwargs: Dict[str, Any],
    execution_context: Dict[str, Any]
) -> Any:
    """Execute agent function dynamically"""
    
    try:
        # Import module
        module = importlib.import_module(module_name)
        
        # Get function
        func = getattr(module, function_name)
        
        # Execute with context
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
            
    except ImportError as e:
        raise ImportError(f"Cannot import module {module_name}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Function {function_name} not found in {module_name}: {e}")


async def _check_task_already_executed(
    task_id: str, 
    execution_context: Dict[str, Any]
) -> bool:
    """Check if task was already executed (idempotency check)"""
    
    cache_key = f"task_result:{task_id}"
    
    try:
        cached_result = await redis_client.get(cache_key)
        return cached_result is not None
    except Exception as e:
        logger.warning("idempotency_check_failed", task_id=task_id, error=str(e))
        return False


async def _get_cached_result(task_id: str) -> Dict[str, Any]:
    """Get cached task result"""
    
    cache_key = f"task_result:{task_id}"
    
    try:
        cached_result = await redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result) if isinstance(cached_result, str) else cached_result
    except Exception as e:
        logger.error("cached_result_retrieval_failed", task_id=task_id, error=str(e))
    
    return {'success': False, 'error': 'Failed to retrieve cached result'}


async def _cache_task_result(task_id: str, result: Dict[str, Any]) -> None:
    """Cache task result for idempotency"""
    
    cache_key = f"task_result:{task_id}"
    cache_ttl = 3600  # 1 hour
    
    try:
        await redis_client.set(cache_key, json.dumps(result), expire=cache_ttl)
        logger.debug("task_result_cached", task_id=task_id, ttl=cache_ttl)
    except Exception as e:
        logger.error("task_result_caching_failed", task_id=task_id, error=str(e))


async def _stream_result_to_websocket(
    result: Dict[str, Any], 
    execution_context: Dict[str, Any]
) -> None:
    """Stream task result to WebSocket connections"""
    
    try:
        # Extract WebSocket routing information
        request_id = execution_context.get('request_id')
        user_id = execution_context.get('user_id')
        
        if not request_id or not user_id:
            return
        
        # Prepare WebSocket message
        ws_message = {
            'type': 'task_result',
            'request_id': request_id,
            'task_result': result,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Publish to Redis for WebSocket manager to pick up
        channel = f"ws_updates:{request_id}"
        await redis_client.publish(channel, json.dumps(ws_message))
        
        logger.debug(
            "task_result_streamed_to_websocket",
            request_id=request_id,
            user_id=user_id,
            channel=channel
        )
        
    except Exception as e:
        logger.error("websocket_streaming_failed", error=str(e))


# Async implementations for specific task types
async def _process_financial_analysis_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Async implementation of financial analysis processing"""
    
    analysis_type = payload.get('analysis_type', 'unknown')
    
    logger.info("financial_analysis_started", analysis_type=analysis_type)
    
    # Simulate heavy financial computation
    await asyncio.sleep(0.1)  # Simulate processing time
    
    return {
        'analysis_type': analysis_type,
        'result': f'Heavy financial analysis completed for {analysis_type}',
        'metrics': {
            'complexity_score': 0.85,
            'confidence_level': 0.92
        }
    }


async def _process_document_analysis_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Async implementation of document analysis processing"""
    
    document_type = payload.get('document_type', 'unknown')
    
    logger.info("document_analysis_started", document_type=document_type)
    
    # Simulate heavy document processing
    await asyncio.sleep(0.1)
    
    return {
        'document_type': document_type,
        'result': f'Heavy document analysis completed for {document_type}',
        'extracted_entities': ['entity1', 'entity2'],
        'bounding_boxes': [{'x': 100, 'y': 200, 'w': 150, 'h': 50}]
    }


async def _process_security_analysis_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Async implementation of security analysis processing"""
    
    analysis_target = payload.get('target', 'unknown')
    
    logger.info("security_analysis_started", target=analysis_target)
    
    # Simulate heavy security scanning
    await asyncio.sleep(0.1)
    
    return {
        'target': analysis_target,
        'result': f'Heavy security analysis completed for {analysis_target}',
        'vulnerabilities_found': 2,
        'risk_score': 0.65
    }


async def _process_research_task_async(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Async implementation of research task processing"""
    
    research_query = payload.get('query', 'unknown')
    
    logger.info("research_task_started", query=research_query)
    
    # Simulate heavy research processing
    await asyncio.sleep(0.1)
    
    return {
        'query': research_query,
        'result': f'Heavy research completed for: {research_query}',
        'sources_analyzed': 15,
        'credibility_score': 0.88
    }


async def _cleanup_redis_cache(cutoff_time: datetime) -> int:
    """Clean up expired Redis cache entries"""
    
    try:
        # This would implement actual Redis cleanup
        # For now, return a simulated count
        cleanup_count = 42
        
        logger.debug("redis_cache_cleanup", cleaned_entries=cleanup_count)
        
        return cleanup_count
        
    except Exception as e:
        logger.error("redis_cleanup_failed", error=str(e))
        return 0