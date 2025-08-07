"""
Heavy Task Identification and Offloading System
Runtime detection and Celery queue management for computationally intensive tasks
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum

from src.workers.celery_app import heavy_task, is_heavy_task, get_task_config, celery_app
from src.agents.base.registry import ToolRegistry
from src.agents.base.agent import HardenedAgent
from src.orchestration.dataflow import NodeOutput
from celery.result import AsyncResult

logger = structlog.get_logger(__name__)


class TaskExecutionMode(str, Enum):
    """Task execution mode classification"""
    LOCAL_SYNC = "local_sync"          # Execute immediately in current thread
    LOCAL_ASYNC = "local_async"        # Execute async in current process
    CELERY_HEAVY = "celery_heavy"      # Offload to Celery worker
    CELERY_PRIORITY = "celery_priority"  # High-priority Celery execution


@dataclass
class TaskOffloadingDecision:
    """Decision result for task execution routing"""
    
    execution_mode: TaskExecutionMode
    reasoning: str
    queue_name: str = "default"
    priority: int = 5
    estimated_execution_time_s: float = 0.0
    
    # Celery-specific configuration
    soft_time_limit: int = 300
    time_limit: int = 600
    retry_limit: int = 3
    
    # Context for tracking
    decision_timestamp: datetime = field(default_factory=datetime.utcnow)
    factors_considered: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskOffloadingMetrics:
    """Metrics for task offloading decisions and performance"""
    
    total_tasks_analyzed: int = 0
    local_executions: int = 0
    celery_offloads: int = 0
    
    # Performance tracking
    avg_local_execution_time_ms: float = 0.0
    avg_celery_execution_time_ms: float = 0.0
    
    # Decision accuracy
    correct_predictions: int = 0
    prediction_errors: int = 0
    
    # Queue utilization
    queue_utilization: Dict[str, int] = field(default_factory=dict)
    
    def get_offload_ratio(self) -> float:
        """Get ratio of tasks offloaded to Celery"""
        total = self.local_executions + self.celery_offloads
        return self.celery_offloads / total if total > 0 else 0.0
    
    def get_prediction_accuracy(self) -> float:
        """Get prediction accuracy percentage"""
        total_predictions = self.correct_predictions + self.prediction_errors
        return self.correct_predictions / total_predictions if total_predictions > 0 else 0.0


class TaskOffloadingEngine:
    """
    Intelligent task offloading engine for heavy computational tasks
    
    Features:
    - Runtime analysis of tool execution characteristics
    - Dynamic offloading decisions based on system load
    - Performance tracking and optimization
    - Integration with Celery worker infrastructure
    - Fallback execution for worker unavailability
    """
    
    def __init__(self, enable_offloading: bool = True):
        """
        Initialize task offloading engine
        
        Args:
            enable_offloading: Whether to enable Celery offloading
        """
        self.enable_offloading = enable_offloading
        
        # Decision tracking
        self._decision_history: List[TaskOffloadingDecision] = []
        self._execution_history: Dict[str, List[float]] = {}
        self._metrics = TaskOffloadingMetrics()
        
        # Performance optimization
        self._tool_characteristics: Dict[str, Dict[str, Any]] = {}
        self._system_load_cache: Dict[str, Any] = {}
        self._cache_ttl = 30  # 30 seconds
        
        logger.info(
            "task_offloading_engine_initialized",
            enable_offloading=enable_offloading
        )

    async def should_offload_task(
        self,
        tool_func: Callable,
        tool_inputs: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> TaskOffloadingDecision:
        """
        Determine if a task should be offloaded to Celery workers
        
        Args:
            tool_func: Tool function to analyze
            tool_inputs: Input parameters for the tool
            execution_context: Execution context and constraints
            
        Returns:
            TaskOffloadingDecision with routing recommendation
        """
        start_time = datetime.utcnow()
        
        # Check if offloading is disabled
        if not self.enable_offloading:
            return TaskOffloadingDecision(
                execution_mode=TaskExecutionMode.LOCAL_ASYNC,
                reasoning="Offloading disabled by configuration"
            )
        
        # Check if tool is marked as heavy task
        if is_heavy_task(tool_func):
            decision = await self._analyze_heavy_task(
                tool_func, tool_inputs, execution_context
            )
        else:
            decision = await self._analyze_regular_task(
                tool_func, tool_inputs, execution_context
            )
        
        # Record decision metrics
        decision.decision_timestamp = start_time
        decision.factors_considered = await self._collect_decision_factors(
            tool_func, tool_inputs, execution_context
        )
        
        self._decision_history.append(decision)
        self._metrics.total_tasks_analyzed += 1
        
        if decision.execution_mode == TaskExecutionMode.CELERY_HEAVY:
            self._metrics.celery_offloads += 1
            self._metrics.queue_utilization[decision.queue_name] = \
                self._metrics.queue_utilization.get(decision.queue_name, 0) + 1
        else:
            self._metrics.local_executions += 1
        
        logger.debug(
            "task_offloading_decision",
            tool_name=getattr(tool_func, '__name__', 'unknown'),
            execution_mode=decision.execution_mode.value,
            reasoning=decision.reasoning[:100],
            queue=decision.queue_name,
            priority=decision.priority
        )
        
        return decision

    async def execute_with_offloading(
        self,
        tool_func: Callable,
        tool_inputs: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Union[Any, AsyncResult]:
        """
        Execute task with intelligent offloading
        
        Args:
            tool_func: Tool function to execute
            tool_inputs: Tool input parameters
            execution_context: Execution context
            
        Returns:
            Direct result for local execution or AsyncResult for Celery
        """
        execution_start = datetime.utcnow()
        
        # Make offloading decision
        decision = await self.should_offload_task(
            tool_func, tool_inputs, execution_context
        )
        
        logger.info(
            "task_execution_started_with_offloading",
            tool_name=getattr(tool_func, '__name__', 'unknown'),
            execution_mode=decision.execution_mode.value,
            estimated_time_s=decision.estimated_execution_time_s
        )
        
        try:
            if decision.execution_mode == TaskExecutionMode.CELERY_HEAVY:
                # Execute via Celery
                result = await self._execute_via_celery(
                    tool_func, tool_inputs, execution_context, decision
                )
            else:
                # Execute locally
                result = await self._execute_locally(
                    tool_func, tool_inputs, execution_context, decision
                )
            
            # Record execution metrics
            execution_time = (datetime.utcnow() - execution_start).total_seconds() * 1000
            tool_name = getattr(tool_func, '__name__', 'unknown')
            
            if tool_name not in self._execution_history:
                self._execution_history[tool_name] = []
            self._execution_history[tool_name].append(execution_time)
            
            # Update metrics
            if decision.execution_mode == TaskExecutionMode.CELERY_HEAVY:
                self._update_celery_metrics(execution_time)
            else:
                self._update_local_metrics(execution_time)
            
            logger.info(
                "task_execution_completed_with_offloading",
                tool_name=tool_name,
                execution_mode=decision.execution_mode.value,
                actual_time_ms=execution_time
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "task_execution_failed_with_offloading",
                tool_name=getattr(tool_func, '__name__', 'unknown'),
                execution_mode=decision.execution_mode.value,
                error=str(e)
            )
            raise

    async def _analyze_heavy_task(
        self,
        tool_func: Callable,
        tool_inputs: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> TaskOffloadingDecision:
        """Analyze tool marked with @heavy_task decorator"""
        
        # Get heavy task configuration
        task_config = get_task_config(tool_func)
        if not task_config:
            return TaskOffloadingDecision(
                execution_mode=TaskExecutionMode.LOCAL_ASYNC,
                reasoning="Heavy task marker present but no configuration found"
            )
        
        # Check system constraints
        system_load = await self._get_system_load()
        worker_availability = await self._check_worker_availability(task_config['queue'])
        
        # Force local execution if requested
        if execution_context.get('force_local_execution', False):
            return TaskOffloadingDecision(
                execution_mode=TaskExecutionMode.LOCAL_ASYNC,
                reasoning="Local execution forced by context",
                factors_considered={
                    'force_local': True,
                    'heavy_task_config': task_config
                }
            )
        
        # Check worker availability
        if not worker_availability['available']:
            return TaskOffloadingDecision(
                execution_mode=TaskExecutionMode.LOCAL_ASYNC,
                reasoning=f"No workers available for queue: {task_config['queue']}",
                factors_considered={
                    'worker_availability': worker_availability,
                    'fallback_to_local': True
                }
            )
        
        # Decide to offload heavy task
        return TaskOffloadingDecision(
            execution_mode=TaskExecutionMode.CELERY_HEAVY,
            reasoning="Heavy task with available workers",
            queue_name=task_config['queue'],
            priority=task_config['priority'],
            soft_time_limit=task_config['soft_time_limit'],
            time_limit=task_config['time_limit'],
            retry_limit=task_config['retry_limit'],
            factors_considered={
                'heavy_task_config': task_config,
                'system_load': system_load,
                'worker_availability': worker_availability
            }
        )

    async def _analyze_regular_task(
        self,
        tool_func: Callable,
        tool_inputs: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> TaskOffloadingDecision:
        """Analyze regular tool for potential offloading"""
        
        tool_name = getattr(tool_func, '__name__', 'unknown')
        
        # Get historical performance data
        historical_times = self._execution_history.get(tool_name, [])
        
        # Estimate execution time based on inputs
        estimated_time = await self._estimate_execution_time(
            tool_func, tool_inputs
        )
        
        # Decision factors
        factors = {
            'estimated_time_s': estimated_time,
            'historical_executions': len(historical_times),
            'avg_historical_time_ms': sum(historical_times) / len(historical_times) if historical_times else 0
        }
        
        # Simple heuristic: offload if estimated time > 30 seconds
        if estimated_time > 30.0:
            # Check worker availability
            worker_availability = await self._check_worker_availability('heavy_tasks')
            
            if worker_availability['available']:
                return TaskOffloadingDecision(
                    execution_mode=TaskExecutionMode.CELERY_HEAVY,
                    reasoning=f"Estimated execution time {estimated_time}s exceeds threshold",
                    queue_name='heavy_tasks',
                    estimated_execution_time_s=estimated_time,
                    factors_considered=factors
                )
        
        # Default to local execution
        return TaskOffloadingDecision(
            execution_mode=TaskExecutionMode.LOCAL_ASYNC,
            reasoning=f"Estimated execution time {estimated_time}s within local limits",
            estimated_execution_time_s=estimated_time,
            factors_considered=factors
        )

    async def _execute_via_celery(
        self,
        tool_func: Callable,
        tool_inputs: Dict[str, Any],
        execution_context: Dict[str, Any],
        decision: TaskOffloadingDecision
    ) -> AsyncResult:
        """Execute task via Celery worker"""
        
        from src.workers.tasks import execute_heavy_agent_task
        
        # Prepare task payload
        task_payload = {
            'task_name': getattr(tool_func, '_heavy_task_name', tool_func.__name__),
            'function_module': tool_func.__module__,
            'function_name': tool_func.__name__,
            'args': [],  # Tool functions are methods, no positional args
            'kwargs': tool_inputs,
            'execution_context': execution_context
        }
        
        # Submit to Celery
        async_result = execute_heavy_agent_task.apply_async(
            args=[task_payload],
            queue=decision.queue_name,
            priority=decision.priority,
            retry=decision.retry_limit > 0,
            soft_time_limit=decision.soft_time_limit,
            time_limit=decision.time_limit
        )
        
        logger.info(
            "task_submitted_to_celery",
            task_id=async_result.id,
            queue=decision.queue_name,
            priority=decision.priority
        )
        
        return async_result

    async def _execute_locally(
        self,
        tool_func: Callable,
        tool_inputs: Dict[str, Any],
        execution_context: Dict[str, Any],
        decision: TaskOffloadingDecision
    ) -> Any:
        """Execute task locally"""
        
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(**tool_inputs)
        else:
            # Run sync function in executor to avoid blocking
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: tool_func(**tool_inputs)
            )

    async def _estimate_execution_time(
        self,
        tool_func: Callable,
        tool_inputs: Dict[str, Any]
    ) -> float:
        """Estimate execution time based on function and inputs"""
        
        tool_name = getattr(tool_func, '__name__', 'unknown')
        
        # Get cached characteristics
        if tool_name in self._tool_characteristics:
            characteristics = self._tool_characteristics[tool_name]
            
            # Simple estimation based on input size
            input_size = len(str(tool_inputs))
            base_time = characteristics.get('base_execution_time_s', 1.0)
            size_factor = characteristics.get('input_size_factor', 0.001)
            
            return base_time + (input_size * size_factor)
        
        # Default estimation
        input_complexity = len(tool_inputs) + len(str(tool_inputs)) / 1000
        return max(1.0, input_complexity * 0.1)  # Minimum 1 second

    async def _get_system_load(self) -> Dict[str, Any]:
        """Get current system load metrics"""
        
        # Check cache
        cache_key = "system_load"
        now = datetime.utcnow()
        
        if (cache_key in self._system_load_cache and 
            (now - self._system_load_cache[cache_key]['timestamp']).seconds < self._cache_ttl):
            return self._system_load_cache[cache_key]['data']
        
        try:
            import psutil
            
            load_data = {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'timestamp': now
            }
            
            # Cache the result
            self._system_load_cache[cache_key] = {
                'data': load_data,
                'timestamp': now
            }
            
            return load_data
            
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_percent': 50,  # Assume moderate load
                'memory_percent': 60,
                'load_average': 1.0,
                'timestamp': now
            }

    async def _check_worker_availability(self, queue_name: str) -> Dict[str, Any]:
        """Check if Celery workers are available for the queue"""
        
        try:
            inspect = celery_app.control.inspect()
            
            # Get active queues from all workers
            active_queues = inspect.active_queues()
            if not active_queues:
                return {
                    'available': False,
                    'reason': 'No active workers found'
                }
            
            # Check if any worker handles this queue
            queue_available = False
            worker_count = 0
            
            for worker, queues in active_queues.items():
                worker_count += 1
                if any(q['name'] == queue_name for q in queues):
                    queue_available = True
            
            return {
                'available': queue_available,
                'worker_count': worker_count,
                'queue_name': queue_name,
                'reason': 'Queue available' if queue_available else f'Queue {queue_name} not handled by workers'
            }
            
        except Exception as e:
            logger.warning("worker_availability_check_failed", error=str(e))
            return {
                'available': False,
                'reason': f'Worker check failed: {str(e)}'
            }

    async def _collect_decision_factors(
        self,
        tool_func: Callable,
        tool_inputs: Dict[str, Any],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect factors that influenced the offloading decision"""
        
        return {
            'tool_name': getattr(tool_func, '__name__', 'unknown'),
            'is_heavy_task': is_heavy_task(tool_func),
            'input_size_bytes': len(str(tool_inputs)),
            'input_parameter_count': len(tool_inputs),
            'force_local': execution_context.get('force_local_execution', False),
            'enable_offloading': self.enable_offloading
        }

    def _update_local_metrics(self, execution_time_ms: float) -> None:
        """Update metrics for local execution"""
        
        current_avg = self._metrics.avg_local_execution_time_ms
        local_count = self._metrics.local_executions
        
        # Update running average
        if local_count > 1:
            self._metrics.avg_local_execution_time_ms = (
                (current_avg * (local_count - 1) + execution_time_ms) / local_count
            )
        else:
            self._metrics.avg_local_execution_time_ms = execution_time_ms

    def _update_celery_metrics(self, execution_time_ms: float) -> None:
        """Update metrics for Celery execution"""
        
        current_avg = self._metrics.avg_celery_execution_time_ms
        celery_count = self._metrics.celery_offloads
        
        # Update running average
        if celery_count > 1:
            self._metrics.avg_celery_execution_time_ms = (
                (current_avg * (celery_count - 1) + execution_time_ms) / celery_count
            )
        else:
            self._metrics.avg_celery_execution_time_ms = execution_time_ms

    def get_offloading_metrics(self) -> TaskOffloadingMetrics:
        """Get current offloading metrics"""
        return self._metrics

    def get_decision_history(self) -> List[TaskOffloadingDecision]:
        """Get history of offloading decisions"""
        return self._decision_history.copy()

    def optimize_decision_thresholds(self) -> Dict[str, Any]:
        """Analyze performance and optimize decision thresholds"""
        
        if len(self._decision_history) < 10:
            return {'message': 'Insufficient data for optimization'}
        
        # Analyze decision accuracy
        # This would implement ML-based optimization in production
        
        return {
            'total_decisions': len(self._decision_history),
            'offload_ratio': self._metrics.get_offload_ratio(),
            'prediction_accuracy': self._metrics.get_prediction_accuracy(),
            'recommendation': 'Continue current thresholds'
        }

    async def cleanup(self) -> None:
        """Cleanup resources and clear caches"""
        
        self._decision_history.clear()
        self._execution_history.clear()
        self._tool_characteristics.clear()
        self._system_load_cache.clear()
        
        logger.info("task_offloading_engine_cleanup_completed")