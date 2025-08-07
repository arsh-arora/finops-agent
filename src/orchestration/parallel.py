"""
Parallel Execution Group Handling
Manages concurrent execution of independent nodes with deterministic output merging
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum
import json

logger = structlog.get_logger(__name__)


class ExecutionStatus(str, Enum):
    """Execution status for parallel groups"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionGroup:
    """Parallel execution group with metadata and tracking"""
    
    group_id: str
    node_ids: List[str]
    max_concurrency: int = 4
    
    # Execution tracking
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    node_results: Dict[str, Any] = field(default_factory=dict)
    node_errors: Dict[str, str] = field(default_factory=dict)
    node_execution_times: Dict[str, float] = field(default_factory=dict)
    
    # Dependencies
    dependency_matrix: Dict[str, Set[str]] = field(default_factory=dict)
    ready_nodes: Set[str] = field(default_factory=set)
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    
    # Performance metrics
    total_execution_time_ms: float = 0.0
    max_node_time_ms: float = 0.0
    min_node_time_ms: float = float('inf')
    parallelism_efficiency: float = 0.0
    
    def calculate_efficiency(self) -> float:
        """Calculate parallelism efficiency (0.0 to 1.0)"""
        if not self.node_execution_times or self.total_execution_time_ms == 0:
            return 0.0
        
        # Sum of all node execution times
        total_node_time = sum(self.node_execution_times.values())
        
        # Theoretical minimum time (if all nodes ran in perfect parallel)
        theoretical_min_time = max(self.node_execution_times.values()) if self.node_execution_times else 0
        
        if theoretical_min_time == 0:
            return 0.0
        
        # Efficiency = theoretical_min / actual_total
        efficiency = theoretical_min_time / self.total_execution_time_ms
        return min(1.0, efficiency)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get execution group summary"""
        return {
            "group_id": self.group_id,
            "node_count": len(self.node_ids),
            "status": self.status.value,
            "completed_nodes": len(self.completed_nodes),
            "failed_nodes": len(self.failed_nodes),
            "total_execution_time_ms": self.total_execution_time_ms,
            "parallelism_efficiency": round(self.parallelism_efficiency, 3),
            "max_concurrency": self.max_concurrency
        }


@dataclass
class ParallelExecutionResult:
    """Result of parallel execution with merged outputs"""
    
    success: bool
    group_results: Dict[str, ExecutionGroup]
    merged_output: Dict[str, Any]
    
    # Metrics
    total_groups: int = 0
    successful_groups: int = 0
    total_execution_time_ms: float = 0.0
    overall_efficiency: float = 0.0
    
    # Error information
    failed_groups: List[str] = field(default_factory=list)
    error_summary: Dict[str, List[str]] = field(default_factory=dict)


class ParallelExecutor:
    """
    High-performance parallel execution coordinator for StateGraph nodes
    
    Features:
    - Configurable concurrency limits per group
    - Dependency-aware execution ordering
    - Deterministic output merging
    - Comprehensive performance metrics
    - Fault tolerance and error isolation
    - Real-time execution monitoring
    """
    
    def __init__(self, default_concurrency: int = 4):
        """
        Initialize parallel executor
        
        Args:
            default_concurrency: Default max concurrent executions per group
        """
        self.default_concurrency = default_concurrency
        
        # Execution tracking
        self._active_groups: Dict[str, ExecutionGroup] = {}
        self._execution_locks: Dict[str, asyncio.Lock] = {}
        
        # Performance optimization
        self._execution_history: List[ExecutionGroup] = []
        self._optimization_cache: Dict[str, int] = {}  # group_signature -> optimal_concurrency
        
        logger.info(
            "parallel_executor_initialized",
            default_concurrency=default_concurrency
        )

    async def execute_parallel_groups(
        self,
        groups: Dict[str, List[str]],
        node_executors: Dict[str, Callable],
        execution_context: Dict[str, Any],
        dependency_matrix: Optional[Dict[str, Set[str]]] = None
    ) -> ParallelExecutionResult:
        """
        Execute multiple parallel groups with optimal scheduling
        
        Args:
            groups: Dictionary mapping group_id -> list of node_ids
            node_executors: Dictionary mapping node_id -> async callable
            execution_context: Shared execution context
            dependency_matrix: Optional node dependency information
            
        Returns:
            ParallelExecutionResult with merged outputs and metrics
        """
        start_time = datetime.utcnow()
        execution_id = execution_context.get('execution_id', uuid4().hex)
        
        logger.info(
            "parallel_execution_started",
            execution_id=execution_id,
            groups_count=len(groups),
            total_nodes=sum(len(nodes) for nodes in groups.values())
        )
        
        try:
            # Create execution groups
            execution_groups = await self._create_execution_groups(
                groups, dependency_matrix
            )
            
            # Execute groups in parallel
            group_tasks = []
            for group in execution_groups.values():
                task = self._execute_group(
                    group, node_executors, execution_context
                )
                group_tasks.append(task)
            
            # Wait for all groups to complete
            await asyncio.gather(*group_tasks, return_exceptions=True)
            
            # Merge outputs deterministically
            merged_output = await self._merge_group_outputs(
                execution_groups, execution_context
            )
            
            # Calculate final metrics
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = ParallelExecutionResult(
                success=all(g.status == ExecutionStatus.COMPLETED for g in execution_groups.values()),
                group_results=execution_groups,
                merged_output=merged_output,
                total_groups=len(execution_groups),
                successful_groups=len([g for g in execution_groups.values() if g.status == ExecutionStatus.COMPLETED]),
                total_execution_time_ms=total_time,
                overall_efficiency=self._calculate_overall_efficiency(execution_groups),
                failed_groups=[g.group_id for g in execution_groups.values() if g.status == ExecutionStatus.FAILED],
                error_summary=self._collect_error_summary(execution_groups)
            )
            
            # Store execution history for optimization
            self._execution_history.extend(execution_groups.values())
            
            logger.info(
                "parallel_execution_completed",
                execution_id=execution_id,
                total_time_ms=total_time,
                successful_groups=result.successful_groups,
                failed_groups=len(result.failed_groups),
                overall_efficiency=result.overall_efficiency
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "parallel_execution_failed",
                execution_id=execution_id,
                error=str(e),
                error_type=type(e).__name__
            )
            
            return ParallelExecutionResult(
                success=False,
                group_results={},
                merged_output={},
                error_summary={"execution_error": [str(e)]}
            )

    async def _create_execution_groups(
        self,
        groups: Dict[str, List[str]],
        dependency_matrix: Optional[Dict[str, Set[str]]] = None
    ) -> Dict[str, ExecutionGroup]:
        """Create ExecutionGroup objects with dependency analysis"""
        
        execution_groups = {}
        
        for group_id, node_ids in groups.items():
            # Determine optimal concurrency for this group
            optimal_concurrency = self._get_optimal_concurrency(group_id, node_ids)
            
            # Create execution group
            group = ExecutionGroup(
                group_id=group_id,
                node_ids=node_ids,
                max_concurrency=optimal_concurrency,
                dependency_matrix=dependency_matrix or {}
            )
            
            # Identify ready nodes (no dependencies within group)
            group.ready_nodes = self._identify_ready_nodes(
                node_ids, dependency_matrix or {}
            )
            
            execution_groups[group_id] = group
            
            logger.debug(
                "execution_group_created",
                group_id=group_id,
                node_count=len(node_ids),
                ready_nodes=len(group.ready_nodes),
                max_concurrency=optimal_concurrency
            )
        
        return execution_groups

    async def _execute_group(
        self,
        group: ExecutionGroup,
        node_executors: Dict[str, Callable],
        execution_context: Dict[str, Any]
    ) -> None:
        """Execute a single parallel group with concurrency control"""
        
        group.status = ExecutionStatus.RUNNING
        group.started_at = datetime.utcnow()
        
        # Create execution lock for this group
        lock = asyncio.Lock()
        self._execution_locks[group.group_id] = lock
        self._active_groups[group.group_id] = group
        
        logger.info(
            "group_execution_started",
            group_id=group.group_id,
            node_count=len(group.node_ids),
            ready_nodes=len(group.ready_nodes)
        )
        
        try:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(group.max_concurrency)
            
            # Execute nodes with dependency respect
            await self._execute_nodes_with_dependencies(
                group, node_executors, execution_context, semaphore
            )
            
            # Mark group as completed
            group.status = ExecutionStatus.COMPLETED
            group.completed_at = datetime.utcnow()
            group.total_execution_time_ms = (
                group.completed_at - group.started_at
            ).total_seconds() * 1000
            
            # Calculate efficiency
            group.parallelism_efficiency = group.calculate_efficiency()
            
            logger.info(
                "group_execution_completed",
                group_id=group.group_id,
                execution_time_ms=group.total_execution_time_ms,
                efficiency=group.parallelism_efficiency,
                completed_nodes=len(group.completed_nodes),
                failed_nodes=len(group.failed_nodes)
            )
            
        except Exception as e:
            group.status = ExecutionStatus.FAILED
            group.completed_at = datetime.utcnow()
            
            logger.error(
                "group_execution_failed",
                group_id=group.group_id,
                error=str(e)
            )
            
        finally:
            self._active_groups.pop(group.group_id, None)
            self._execution_locks.pop(group.group_id, None)

    async def _execute_nodes_with_dependencies(
        self,
        group: ExecutionGroup,
        node_executors: Dict[str, Callable],
        execution_context: Dict[str, Any],
        semaphore: asyncio.Semaphore
    ) -> None:
        """Execute nodes respecting dependencies and concurrency limits"""
        
        pending_nodes = set(group.node_ids)
        
        while pending_nodes:
            # Identify nodes that are ready to execute
            ready_to_execute = []
            for node_id in pending_nodes:
                if self._node_dependencies_satisfied(node_id, group):
                    ready_to_execute.append(node_id)
            
            if not ready_to_execute:
                # Check for deadlock
                if group.completed_nodes.union(group.failed_nodes) == set():
                    logger.error(
                        "dependency_deadlock_detected",
                        group_id=group.group_id,
                        pending_nodes=list(pending_nodes)
                    )
                    break
                
                # Wait briefly for dependencies to complete
                await asyncio.sleep(0.01)
                continue
            
            # Execute ready nodes concurrently
            node_tasks = []
            for node_id in ready_to_execute:
                task = self._execute_single_node(
                    node_id, group, node_executors, execution_context, semaphore
                )
                node_tasks.append(task)
                pending_nodes.remove(node_id)
            
            # Wait for this batch to complete before checking dependencies again
            await asyncio.gather(*node_tasks, return_exceptions=True)

    async def _execute_single_node(
        self,
        node_id: str,
        group: ExecutionGroup,
        node_executors: Dict[str, Callable],
        execution_context: Dict[str, Any],
        semaphore: asyncio.Semaphore
    ) -> None:
        """Execute a single node with metrics tracking"""
        
        if node_id not in node_executors:
            error_msg = f"No executor found for node: {node_id}"
            group.node_errors[node_id] = error_msg
            group.failed_nodes.add(node_id)
            logger.error("node_executor_missing", node_id=node_id, group_id=group.group_id)
            return
        
        async with semaphore:
            start_time = datetime.utcnow()
            
            logger.debug(
                "node_execution_started",
                node_id=node_id,
                group_id=group.group_id
            )
            
            try:
                # Execute node
                executor = node_executors[node_id]
                result = await executor(execution_context)
                
                # Record successful execution
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                group.node_results[node_id] = result
                group.node_execution_times[node_id] = execution_time
                group.completed_nodes.add(node_id)
                
                # Update timing stats
                group.max_node_time_ms = max(group.max_node_time_ms, execution_time)
                group.min_node_time_ms = min(group.min_node_time_ms, execution_time)
                
                logger.debug(
                    "node_execution_completed",
                    node_id=node_id,
                    group_id=group.group_id,
                    execution_time_ms=execution_time
                )
                
            except Exception as e:
                # Record failed execution
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                group.node_errors[node_id] = str(e)
                group.node_execution_times[node_id] = execution_time
                group.failed_nodes.add(node_id)
                
                logger.error(
                    "node_execution_failed",
                    node_id=node_id,
                    group_id=group.group_id,
                    error=str(e),
                    execution_time_ms=execution_time
                )

    def _node_dependencies_satisfied(
        self, 
        node_id: str, 
        group: ExecutionGroup
    ) -> bool:
        """Check if all dependencies for a node are satisfied"""
        
        if node_id not in group.dependency_matrix:
            return True
        
        dependencies = group.dependency_matrix[node_id]
        return dependencies.issubset(group.completed_nodes)

    async def _merge_group_outputs(
        self,
        execution_groups: Dict[str, ExecutionGroup],
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge outputs from all parallel groups deterministically"""
        
        merged_output = {
            "parallel_execution_results": {},
            "group_summaries": {},
            "execution_metadata": {
                "merged_at": datetime.utcnow().isoformat(),
                "total_groups": len(execution_groups),
                "execution_id": execution_context.get('execution_id', '')
            }
        }
        
        # Merge results from each group
        for group_id, group in execution_groups.items():
            merged_output["parallel_execution_results"][group_id] = group.node_results
            merged_output["group_summaries"][group_id] = group.get_summary()
        
        # Create flattened results for downstream consumption
        merged_output["all_node_results"] = {}
        for group in execution_groups.values():
            merged_output["all_node_results"].update(group.node_results)
        
        logger.debug(
            "group_outputs_merged",
            total_groups=len(execution_groups),
            total_node_results=len(merged_output["all_node_results"])
        )
        
        return merged_output

    def _identify_ready_nodes(
        self, 
        node_ids: List[str], 
        dependency_matrix: Dict[str, Set[str]]
    ) -> Set[str]:
        """Identify nodes with no dependencies that can start immediately"""
        
        ready_nodes = set()
        
        for node_id in node_ids:
            if node_id not in dependency_matrix or not dependency_matrix[node_id]:
                ready_nodes.add(node_id)
        
        return ready_nodes

    def _get_optimal_concurrency(
        self, 
        group_id: str, 
        node_ids: List[str]
    ) -> int:
        """Determine optimal concurrency for a group based on historical performance"""
        
        # Create signature for this group type
        group_signature = f"{len(node_ids)}nodes"
        
        # Check optimization cache
        if group_signature in self._optimization_cache:
            cached_concurrency = self._optimization_cache[group_signature]
            return min(cached_concurrency, len(node_ids))
        
        # Use default concurrency
        return min(self.default_concurrency, len(node_ids))

    def _calculate_overall_efficiency(
        self, 
        execution_groups: Dict[str, ExecutionGroup]
    ) -> float:
        """Calculate overall parallelism efficiency across all groups"""
        
        if not execution_groups:
            return 0.0
        
        total_efficiency = sum(
            group.parallelism_efficiency 
            for group in execution_groups.values()
        )
        
        return total_efficiency / len(execution_groups)

    def _collect_error_summary(
        self, 
        execution_groups: Dict[str, ExecutionGroup]
    ) -> Dict[str, List[str]]:
        """Collect all errors from execution groups"""
        
        error_summary = {}
        
        for group_id, group in execution_groups.items():
            if group.node_errors:
                error_summary[group_id] = [
                    f"{node_id}: {error}" 
                    for node_id, error in group.node_errors.items()
                ]
        
        return error_summary

    def get_active_groups(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active execution groups"""
        
        active_info = {}
        
        for group_id, group in self._active_groups.items():
            active_info[group_id] = {
                "group_id": group_id,
                "status": group.status.value,
                "node_count": len(group.node_ids),
                "completed_nodes": len(group.completed_nodes),
                "failed_nodes": len(group.failed_nodes),
                "started_at": group.started_at.isoformat() if group.started_at else None
            }
        
        return active_info

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics for parallel execution"""
        
        if not self._execution_history:
            return {"no_execution_history": True}
        
        # Calculate aggregate metrics
        total_groups = len(self._execution_history)
        successful_groups = len([g for g in self._execution_history if g.status == ExecutionStatus.COMPLETED])
        
        avg_efficiency = sum(g.parallelism_efficiency for g in self._execution_history) / total_groups
        avg_execution_time = sum(g.total_execution_time_ms for g in self._execution_history) / total_groups
        
        return {
            "total_groups_executed": total_groups,
            "successful_groups": successful_groups,
            "success_rate": round(successful_groups / total_groups, 3),
            "average_efficiency": round(avg_efficiency, 3),
            "average_execution_time_ms": round(avg_execution_time, 2),
            "optimization_cache_size": len(self._optimization_cache)
        }

    async def cancel_group(self, group_id: str) -> bool:
        """Cancel execution of a specific group"""
        
        if group_id not in self._active_groups:
            return False
        
        group = self._active_groups[group_id]
        group.status = ExecutionStatus.CANCELLED
        
        logger.info(
            "group_execution_cancelled",
            group_id=group_id
        )
        
        return True