"""
Worker Scaling and Load Balancing System
Dynamic worker scaling with autoscaling hooks and idempotent task execution
"""

import asyncio
import json
import structlog
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import subprocess
import os
import signal
from uuid import uuid4

from celery import Celery
from .queue_monitor import QueueMetrics, RedisQueueMonitor

logger = structlog.get_logger(__name__)


class ScalingAction(str, Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    NO_ACTION = "no_action"
    EMERGENCY_SCALE = "emergency_scale"


class WorkerStatus(str, Enum):
    """Worker status enumeration"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ScalingDecision:
    """Scaling decision with reasoning and parameters"""
    
    action: ScalingAction
    target_worker_count: int
    current_worker_count: int
    reasoning: str
    
    # Scaling parameters
    queue_name: str = "default"
    priority: int = 5
    
    # Decision context
    decision_timestamp: datetime = field(default_factory=datetime.utcnow)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    
    # Execution tracking
    executed: bool = False
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None


@dataclass
class WorkerInstance:
    """Worker instance metadata"""
    
    worker_id: str
    process_id: Optional[int] = None
    hostname: str = "localhost"
    status: WorkerStatus = WorkerStatus.STARTING
    
    # Configuration
    queue_names: List[str] = field(default_factory=list)
    concurrency: int = 4
    command_args: List[str] = field(default_factory=list)
    
    # Resource allocation
    max_memory_mb: int = 512
    max_cpu_percent: float = 80.0
    
    # Lifecycle tracking
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: Optional[datetime] = None
    restart_count: int = 0
    
    # Performance metrics
    tasks_processed: int = 0
    avg_task_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def is_healthy(self) -> bool:
        """Check if worker is healthy"""
        
        if self.status != WorkerStatus.RUNNING:
            return False
        
        # Check heartbeat (should be within last 2 minutes)
        if self.last_heartbeat:
            heartbeat_age = datetime.utcnow() - self.last_heartbeat
            if heartbeat_age > timedelta(minutes=2):
                return False
        
        # Check resource usage
        if (self.memory_usage_mb > self.max_memory_mb or 
            self.cpu_usage_percent > self.max_cpu_percent):
            return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get worker summary"""
        return {
            'worker_id': self.worker_id,
            'status': self.status.value,
            'hostname': self.hostname,
            'concurrency': self.concurrency,
            'queues': self.queue_names,
            'tasks_processed': self.tasks_processed,
            'is_healthy': self.is_healthy(),
            'uptime_hours': (datetime.utcnow() - self.started_at).total_seconds() / 3600,
            'restart_count': self.restart_count
        }


@dataclass
class ScalingPolicy:
    """Autoscaling policy configuration"""
    
    # Scaling triggers
    queue_length_threshold: int = 100
    avg_wait_time_threshold_ms: float = 30000
    cpu_threshold_percent: float = 75.0
    memory_threshold_percent: float = 80.0
    
    # Scaling limits
    min_workers: int = 1
    max_workers: int = 10
    scale_up_cooldown_minutes: int = 5
    scale_down_cooldown_minutes: int = 10
    
    # Scaling increment
    scale_up_count: int = 1
    scale_down_count: int = 1
    emergency_scale_multiplier: float = 2.0
    
    # Health checks
    health_check_interval_seconds: int = 60
    unhealthy_worker_threshold_minutes: int = 5
    
    def should_scale_up(self, metrics: QueueMetrics, worker_count: int) -> bool:
        """Check if scale up is needed"""
        
        if worker_count >= self.max_workers:
            return False
        
        # Check queue backlog
        if metrics.pending_tasks > self.queue_length_threshold:
            return True
        
        # Check wait time
        if metrics.avg_wait_time_ms > self.avg_wait_time_threshold_ms:
            return True
        
        # Check resource utilization
        if (metrics.cpu_usage_percent > self.cpu_threshold_percent or 
            metrics.memory_usage_mb > self.memory_threshold_percent):
            return True
        
        return False
    
    def should_scale_down(self, metrics: QueueMetrics, worker_count: int) -> bool:
        """Check if scale down is appropriate"""
        
        if worker_count <= self.min_workers:
            return False
        
        # Only scale down if queue is empty or small
        if metrics.pending_tasks > 10:
            return False
        
        # Check if resources are underutilized
        if (metrics.cpu_usage_percent < 30.0 and 
            metrics.memory_usage_mb < 50.0 and 
            metrics.avg_wait_time_ms < 5000):
            return True
        
        return False


class WorkerScalingManager:
    """
    Manages dynamic worker scaling with load balancing
    
    Features:
    - Automatic scaling based on queue metrics
    - Health monitoring and worker restart
    - Load balancing across worker instances
    - Graceful worker shutdown
    - Idempotent task execution guarantees
    """
    
    def __init__(
        self, 
        celery_app: Celery,
        queue_monitor: RedisQueueMonitor
    ):
        """
        Initialize worker scaling manager
        
        Args:
            celery_app: Celery application instance
            queue_monitor: Queue monitoring system
        """
        self.celery_app = celery_app
        self.queue_monitor = queue_monitor
        
        # Worker management
        self._workers: Dict[str, WorkerInstance] = {}
        self._scaling_policies: Dict[str, ScalingPolicy] = {}
        
        # Scaling control
        self._scaling_active = False
        self._scaling_task: Optional[asyncio.Task] = None
        self._last_scale_actions: Dict[str, datetime] = {}
        
        # Decision tracking
        self._scaling_decisions: List[ScalingDecision] = []
        self._scaling_metrics: Dict[str, Any] = {
            'total_scale_ups': 0,
            'total_scale_downs': 0,
            'emergency_scales': 0,
            'failed_scales': 0
        }
        
        # Autoscaling hooks
        self._autoscaling_hooks: List[Callable] = []
        
        logger.info("worker_scaling_manager_initialized")

    def configure_scaling_policy(
        self, 
        queue_name: str, 
        policy: ScalingPolicy
    ) -> None:
        """Configure scaling policy for a queue"""
        
        self._scaling_policies[queue_name] = policy
        
        logger.info(
            "scaling_policy_configured",
            queue_name=queue_name,
            min_workers=policy.min_workers,
            max_workers=policy.max_workers,
            scale_up_threshold=policy.queue_length_threshold
        )

    async def start_scaling_system(self, check_interval_seconds: int = 60) -> None:
        """Start the automatic scaling system"""
        
        if self._scaling_active:
            logger.warning("scaling_system_already_active")
            return
        
        self._scaling_active = True
        self._scaling_task = asyncio.create_task(
            self._scaling_loop(check_interval_seconds)
        )
        
        logger.info("scaling_system_started", interval_seconds=check_interval_seconds)

    async def stop_scaling_system(self) -> None:
        """Stop the automatic scaling system"""
        
        self._scaling_active = False
        
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("scaling_system_stopped")

    async def _scaling_loop(self, check_interval: int) -> None:
        """Main scaling decision and execution loop"""
        
        while self._scaling_active:
            try:
                # Evaluate scaling decisions for each queue
                for queue_name, policy in self._scaling_policies.items():
                    await self._evaluate_scaling_decision(queue_name, policy)
                
                # Perform health checks
                await self._perform_health_checks()
                
                # Clean up old decisions
                self._cleanup_old_decisions()
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error("scaling_loop_error", error=str(e))
                await asyncio.sleep(check_interval)

    async def _evaluate_scaling_decision(
        self, 
        queue_name: str, 
        policy: ScalingPolicy
    ) -> None:
        """Evaluate and execute scaling decision for a queue"""
        
        # Get current queue metrics
        queue_metrics = self.queue_monitor.get_queue_metrics(queue_name)
        if not queue_metrics or queue_name not in queue_metrics:
            logger.debug("no_metrics_for_queue", queue_name=queue_name)
            return
        
        metrics = queue_metrics[queue_name]
        current_workers = len(self._get_workers_for_queue(queue_name))
        
        # Make scaling decision
        decision = await self._make_scaling_decision(
            queue_name, metrics, policy, current_workers
        )
        
        if decision.action == ScalingAction.NO_ACTION:
            return
        
        # Check cooldown period
        if not self._can_perform_scaling_action(queue_name, decision.action, policy):
            logger.debug(
                "scaling_action_in_cooldown",
                queue_name=queue_name,
                action=decision.action.value
            )
            return
        
        # Execute scaling action
        await self._execute_scaling_decision(decision)

    async def _make_scaling_decision(
        self,
        queue_name: str,
        metrics: QueueMetrics,
        policy: ScalingPolicy,
        current_workers: int
    ) -> ScalingDecision:
        """Make intelligent scaling decision based on metrics"""
        
        # Check for emergency conditions
        emergency_conditions = (
            metrics.pending_tasks > policy.queue_length_threshold * 3 or
            metrics.avg_wait_time_ms > policy.avg_wait_time_threshold_ms * 2 or
            metrics.error_rate_percent > 50.0
        )
        
        if emergency_conditions and current_workers < policy.max_workers:
            target_workers = min(
                policy.max_workers,
                int(current_workers * policy.emergency_scale_multiplier)
            )
            
            return ScalingDecision(
                action=ScalingAction.EMERGENCY_SCALE,
                target_worker_count=target_workers,
                current_worker_count=current_workers,
                reasoning="Emergency scaling due to severe queue backlog or high error rate",
                queue_name=queue_name,
                priority=10,
                metrics_snapshot=self._create_metrics_snapshot(metrics),
                thresholds_used=self._extract_policy_thresholds(policy)
            )
        
        # Check for normal scale up
        if policy.should_scale_up(metrics, current_workers):
            target_workers = min(
                policy.max_workers,
                current_workers + policy.scale_up_count
            )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                target_worker_count=target_workers,
                current_worker_count=current_workers,
                reasoning=f"Scale up due to queue backlog ({metrics.pending_tasks} tasks) or high resource usage",
                queue_name=queue_name,
                metrics_snapshot=self._create_metrics_snapshot(metrics),
                thresholds_used=self._extract_policy_thresholds(policy)
            )
        
        # Check for scale down
        if policy.should_scale_down(metrics, current_workers):
            target_workers = max(
                policy.min_workers,
                current_workers - policy.scale_down_count
            )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                target_worker_count=target_workers,
                current_worker_count=current_workers,
                reasoning="Scale down due to low queue utilization and resource usage",
                queue_name=queue_name,
                metrics_snapshot=self._create_metrics_snapshot(metrics),
                thresholds_used=self._extract_policy_thresholds(policy)
            )
        
        # No action needed
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            target_worker_count=current_workers,
            current_worker_count=current_workers,
            reasoning="Current worker count is appropriate for current load",
            queue_name=queue_name
        )

    async def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute a scaling decision"""
        
        logger.info(
            "executing_scaling_decision",
            queue_name=decision.queue_name,
            action=decision.action.value,
            current_workers=decision.current_worker_count,
            target_workers=decision.target_worker_count,
            reasoning=decision.reasoning[:100]
        )
        
        try:
            if decision.action == ScalingAction.SCALE_UP or decision.action == ScalingAction.EMERGENCY_SCALE:
                workers_to_add = decision.target_worker_count - decision.current_worker_count
                await self._scale_up_workers(decision.queue_name, workers_to_add)
                
                if decision.action == ScalingAction.EMERGENCY_SCALE:
                    self._scaling_metrics['emergency_scales'] += 1
                else:
                    self._scaling_metrics['total_scale_ups'] += 1
            
            elif decision.action == ScalingAction.SCALE_DOWN:
                workers_to_remove = decision.current_worker_count - decision.target_worker_count
                await self._scale_down_workers(decision.queue_name, workers_to_remove)
                self._scaling_metrics['total_scale_downs'] += 1
            
            # Mark decision as executed
            decision.executed = True
            decision.execution_result = "success"
            
            # Update last action time
            self._last_scale_actions[decision.queue_name] = decision.decision_timestamp
            
            # Execute autoscaling hooks
            await self._execute_autoscaling_hooks(decision)
            
        except Exception as e:
            decision.execution_error = str(e)
            self._scaling_metrics['failed_scales'] += 1
            
            logger.error(
                "scaling_decision_execution_failed",
                queue_name=decision.queue_name,
                action=decision.action.value,
                error=str(e)
            )
        
        finally:
            self._scaling_decisions.append(decision)

    async def _scale_up_workers(self, queue_name: str, count: int) -> None:
        """Scale up workers for a queue"""
        
        for i in range(count):
            worker_id = f"worker_{queue_name}_{uuid4().hex[:8]}"
            
            # Create worker instance
            worker = WorkerInstance(
                worker_id=worker_id,
                queue_names=[queue_name],
                concurrency=4,  # Default concurrency
                command_args=self._build_worker_command(worker_id, [queue_name])
            )
            
            # Start worker process
            success = await self._start_worker_process(worker)
            
            if success:
                self._workers[worker_id] = worker
                logger.info("worker_started", worker_id=worker_id, queue_name=queue_name)
            else:
                logger.error("worker_start_failed", worker_id=worker_id, queue_name=queue_name)

    async def _scale_down_workers(self, queue_name: str, count: int) -> None:
        """Scale down workers for a queue"""
        
        queue_workers = self._get_workers_for_queue(queue_name)
        workers_to_stop = queue_workers[:count]
        
        for worker in workers_to_stop:
            await self._stop_worker_gracefully(worker)

    async def _start_worker_process(self, worker: WorkerInstance) -> bool:
        """Start a worker process"""
        
        try:
            # Build Celery worker command
            cmd = [
                "celery", "worker",
                f"--app=src.workers.celery_app:celery_app",
                f"--hostname={worker.worker_id}@{worker.hostname}",
                f"--concurrency={worker.concurrency}",
                f"--queues={','.join(worker.queue_names)}",
                "--loglevel=info"
            ]
            
            # Add additional command arguments
            cmd.extend(worker.command_args)
            
            # Start process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            worker.process_id = process.pid
            worker.status = WorkerStatus.RUNNING
            worker.started_at = datetime.utcnow()
            
            logger.info(
                "worker_process_started",
                worker_id=worker.worker_id,
                process_id=worker.process_id,
                command=' '.join(cmd)
            )
            
            return True
            
        except Exception as e:
            worker.status = WorkerStatus.ERROR
            logger.error(
                "worker_process_start_failed",
                worker_id=worker.worker_id,
                error=str(e)
            )
            return False

    async def _stop_worker_gracefully(self, worker: WorkerInstance) -> None:
        """Stop a worker gracefully"""
        
        logger.info("stopping_worker_gracefully", worker_id=worker.worker_id)
        
        try:
            worker.status = WorkerStatus.STOPPING
            
            # Send graceful shutdown signal
            if worker.process_id:
                try:
                    # Send TERM signal for graceful shutdown
                    os.kill(worker.process_id, signal.SIGTERM)
                    
                    # Wait for graceful shutdown
                    await asyncio.sleep(10)
                    
                    # Force kill if still running
                    try:
                        os.kill(worker.process_id, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Process already terminated
                        
                except ProcessLookupError:
                    pass  # Process already terminated
            
            worker.status = WorkerStatus.STOPPED
            self._workers.pop(worker.worker_id, None)
            
            logger.info("worker_stopped", worker_id=worker.worker_id)
            
        except Exception as e:
            logger.error(
                "worker_stop_failed",
                worker_id=worker.worker_id,
                error=str(e)
            )

    def _build_worker_command(
        self, 
        worker_id: str, 
        queue_names: List[str]
    ) -> List[str]:
        """Build command arguments for worker"""
        
        return [
            f"--max-tasks-per-child=1000",
            f"--max-memory-per-child=512000",  # 512MB
        ]

    def _get_workers_for_queue(self, queue_name: str) -> List[WorkerInstance]:
        """Get workers handling a specific queue"""
        
        return [
            worker for worker in self._workers.values()
            if queue_name in worker.queue_names and worker.status == WorkerStatus.RUNNING
        ]

    def _can_perform_scaling_action(
        self,
        queue_name: str,
        action: ScalingAction,
        policy: ScalingPolicy
    ) -> bool:
        """Check if scaling action can be performed (cooldown check)"""
        
        if queue_name not in self._last_scale_actions:
            return True
        
        last_action_time = self._last_scale_actions[queue_name]
        now = datetime.utcnow()
        
        if action in [ScalingAction.SCALE_UP, ScalingAction.EMERGENCY_SCALE]:
            cooldown = timedelta(minutes=policy.scale_up_cooldown_minutes)
        else:
            cooldown = timedelta(minutes=policy.scale_down_cooldown_minutes)
        
        return (now - last_action_time) >= cooldown

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all workers"""
        
        unhealthy_workers = []
        
        for worker_id, worker in self._workers.items():
            if not worker.is_healthy():
                unhealthy_workers.append(worker)
        
        # Restart unhealthy workers
        for worker in unhealthy_workers:
            await self._restart_unhealthy_worker(worker)

    async def _restart_unhealthy_worker(self, worker: WorkerInstance) -> None:
        """Restart an unhealthy worker"""
        
        logger.warning("restarting_unhealthy_worker", worker_id=worker.worker_id)
        
        # Stop the worker
        await self._stop_worker_gracefully(worker)
        
        # Update restart count
        worker.restart_count += 1
        
        # Start new worker with same configuration
        new_worker = WorkerInstance(
            worker_id=f"{worker.worker_id}_r{worker.restart_count}",
            queue_names=worker.queue_names,
            concurrency=worker.concurrency,
            command_args=worker.command_args
        )
        
        success = await self._start_worker_process(new_worker)
        
        if success:
            self._workers[new_worker.worker_id] = new_worker
        
        logger.info(
            "worker_restart_completed",
            original_worker_id=worker.worker_id,
            new_worker_id=new_worker.worker_id,
            success=success
        )

    async def _execute_autoscaling_hooks(self, decision: ScalingDecision) -> None:
        """Execute registered autoscaling hooks"""
        
        for hook in self._autoscaling_hooks:
            try:
                await hook(decision)
            except Exception as e:
                logger.error("autoscaling_hook_failed", error=str(e))

    def _create_metrics_snapshot(self, metrics: QueueMetrics) -> Dict[str, Any]:
        """Create snapshot of queue metrics"""
        
        return {
            'pending_tasks': metrics.pending_tasks,
            'active_tasks': metrics.active_tasks,
            'error_rate_percent': metrics.error_rate_percent,
            'avg_wait_time_ms': metrics.avg_wait_time_ms,
            'worker_count': metrics.worker_count,
            'snapshot_time': datetime.utcnow().isoformat()
        }

    def _extract_policy_thresholds(self, policy: ScalingPolicy) -> Dict[str, float]:
        """Extract threshold values from scaling policy"""
        
        return {
            'queue_length_threshold': policy.queue_length_threshold,
            'wait_time_threshold_ms': policy.avg_wait_time_threshold_ms,
            'cpu_threshold_percent': policy.cpu_threshold_percent,
            'memory_threshold_percent': policy.memory_threshold_percent
        }

    def _cleanup_old_decisions(self, max_age_hours: int = 24) -> None:
        """Clean up old scaling decisions"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        self._scaling_decisions = [
            decision for decision in self._scaling_decisions
            if decision.decision_timestamp > cutoff_time
        ]

    def register_autoscaling_hook(self, hook: Callable) -> None:
        """Register hook for autoscaling events"""
        
        self._autoscaling_hooks.append(hook)
        logger.debug("autoscaling_hook_registered", hook_count=len(self._autoscaling_hooks))

    def get_worker_summary(self) -> Dict[str, Any]:
        """Get summary of all workers"""
        
        workers_by_status = {}
        for status in WorkerStatus:
            workers_by_status[status.value] = len([
                w for w in self._workers.values() if w.status == status
            ])
        
        return {
            'total_workers': len(self._workers),
            'workers_by_status': workers_by_status,
            'workers_by_queue': {
                queue: len(self._get_workers_for_queue(queue))
                for queue in self._scaling_policies.keys()
            },
            'scaling_metrics': self._scaling_metrics,
            'recent_decisions': len([
                d for d in self._scaling_decisions
                if (datetime.utcnow() - d.decision_timestamp).hours < 1
            ])
        }

    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get scaling performance metrics"""
        
        return {
            **self._scaling_metrics,
            'total_scaling_decisions': len(self._scaling_decisions),
            'successful_scales': len([d for d in self._scaling_decisions if d.executed]),
            'failed_scales': len([d for d in self._scaling_decisions if d.execution_error]),
            'scaling_active': self._scaling_active
        }

    async def cleanup(self) -> None:
        """Cleanup scaling system"""
        
        await self.stop_scaling_system()
        
        # Stop all workers
        for worker in list(self._workers.values()):
            await self._stop_worker_gracefully(worker)
        
        self._workers.clear()
        self._scaling_decisions.clear()
        self._autoscaling_hooks.clear()
        
        logger.info("worker_scaling_manager_cleanup_completed")