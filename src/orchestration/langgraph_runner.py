"""
LangGraph StateGraph Execution Runtime
Executes compiled StateGraphs with memory integration, metrics tracking, and error handling
"""

import asyncio
import time
import structlog
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from uuid import uuid4
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

try:
    from langgraph import StateGraph
    from langgraph.graph import START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    START = "start"
    END = "end"

from src.agents.base.registry import ToolRegistry
from src.agents.base.agent import HardenedAgent
from src.memory.mem0_service import FinOpsMemoryService
from src.planner.compiler import GraphState
from src.agents.models import ExecutionPlan, TaskStatus
from .dataflow import DataFlowManager, NodeOutput
from .parallel import ParallelExecutor
from .task_offloader import TaskOffloadingEngine, TaskExecutionMode

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionMetrics:
    """Comprehensive execution metrics and performance data"""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_execution_time_ms: float = 0.0
    
    # Node execution metrics
    nodes_executed: int = 0
    nodes_failed: int = 0
    nodes_skipped: int = 0
    node_execution_times: Dict[str, float] = field(default_factory=dict)
    
    # Memory metrics
    memory_queries: int = 0
    memory_writes: int = 0
    memory_context_size: int = 0
    
    # Token and cost tracking
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    token_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Graph traversal metrics
    graph_hops: int = 0
    parallel_groups_executed: int = 0
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass  
class ExecutionResult:
    """Complete execution result with outputs, metrics, and context"""
    success: bool
    final_state: Dict[str, Any]
    metrics: ExecutionMetrics
    execution_id: str = field(default_factory=lambda: uuid4().hex)
    
    # Execution context
    plan_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Output data
    task_results: Dict[str, Any] = field(default_factory=dict)
    memory_updates: List[str] = field(default_factory=list)
    
    # Error information
    error_message: str = ""
    failed_nodes: List[str] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate execution summary for logging/reporting"""
        return {
            "execution_id": self.execution_id,
            "success": self.success,
            "duration_ms": self.metrics.total_execution_time_ms,
            "nodes_executed": self.metrics.nodes_executed,
            "nodes_failed": self.metrics.nodes_failed,
            "total_tokens": self.metrics.total_tokens,
            "estimated_cost": self.metrics.estimated_cost_usd,
            "memory_operations": self.metrics.memory_queries + self.metrics.memory_writes,
            "parallel_groups": self.metrics.parallel_groups_executed,
            "error_count": len(self.metrics.errors)
        }


class LangGraphRunner:
    """
    Production-grade LangGraph StateGraph execution engine
    
    Features:
    - Async execution with cancellation support
    - Comprehensive metrics collection
    - Memory session management across agents
    - Error handling and recovery
    - Parallel execution coordination
    - Real-time progress tracking
    """
    
    def __init__(self, memory_service: FinOpsMemoryService):
        """
        Initialize LangGraph execution runtime
        
        Args:
            memory_service: Memory service for context management
        """
        self.memory_service = memory_service
        self.dataflow_manager = DataFlowManager()
        self.parallel_executor = ParallelExecutor()
        self.task_offloader = TaskOffloadingEngine()
        
        # Execution tracking
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._execution_locks: Dict[str, asyncio.Lock] = {}
        
        # Performance optimization
        self._agent_cache: Dict[str, HardenedAgent] = {}
        self._tool_registry_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            "langgraph_runner_initialized",
            has_langgraph=LANGGRAPH_AVAILABLE,
            memory_service_available=memory_service is not None
        )

    async def run_stategraph(
        self,
        graph: StateGraph,
        context: Dict[str, Any],
        timeout_seconds: int = 300,
        execution_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute StateGraph with comprehensive monitoring and error handling
        
        Args:
            graph: Compiled StateGraph to execute
            context: Initial execution context
            timeout_seconds: Maximum execution time
            execution_id: Optional execution identifier
            
        Returns:
            ExecutionResult with complete execution data
        """
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph not available - install langgraph>=0.2.0")
        
        execution_id = execution_id or uuid4().hex
        metrics = ExecutionMetrics()
        
        logger.info(
            "stategraph_execution_started",
            execution_id=execution_id,
            context_keys=list(context.keys()),
            timeout_seconds=timeout_seconds
        )
        
        try:
            # Initialize execution state
            initial_state = GraphState(context)
            
            # Register tools from all agents involved
            await self._register_graph_tools(graph, initial_state)
            
            # Execute with timeout and cancellation support
            async with self._execution_context(execution_id, timeout_seconds):
                final_state = await self._execute_with_monitoring(
                    graph, initial_state, metrics, execution_id
                )
            
            # Calculate final metrics
            metrics.end_time = datetime.utcnow()
            metrics.total_execution_time_ms = (
                metrics.end_time - metrics.start_time
            ).total_seconds() * 1000
            
            result = ExecutionResult(
                success=True,
                final_state=dict(final_state),
                metrics=metrics,
                execution_id=execution_id,
                plan_id=context.get('plan_id'),
                user_id=context.get('user_id'),
                request_id=context.get('request_id'),
                task_results=final_state.get('task_results', {}),
                memory_updates=final_state.get('execution_log', [])
            )
            
            logger.info(
                "stategraph_execution_completed",
                execution_id=execution_id,
                summary=result.get_summary()
            )
            
            return result
            
        except asyncio.TimeoutError:
            metrics.errors.append({
                "type": "timeout",
                "message": f"Execution exceeded {timeout_seconds}s timeout",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.error(
                "stategraph_execution_timeout",
                execution_id=execution_id,
                timeout_seconds=timeout_seconds
            )
            
            return self._create_error_result(
                execution_id, metrics, f"Execution timeout after {timeout_seconds}s",
                context
            )
            
        except Exception as e:
            metrics.errors.append({
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.error(
                "stategraph_execution_failed",
                execution_id=execution_id,
                error=str(e),
                error_type=type(e).__name__
            )
            
            return self._create_error_result(
                execution_id, metrics, str(e), context
            )

    async def _execute_with_monitoring(
        self,
        graph: StateGraph,
        initial_state: GraphState,
        metrics: ExecutionMetrics,
        execution_id: str
    ) -> GraphState:
        """Execute graph with comprehensive monitoring and metrics collection"""
        
        # Track memory context size
        metrics.memory_context_size = len(initial_state.get('memory_context', []))
        
        # Create execution wrapper for node monitoring
        wrapped_graph = self._wrap_graph_for_monitoring(graph, metrics, execution_id)
        
        # Execute the wrapped graph
        if hasattr(wrapped_graph, 'ainvoke'):
            # LangGraph 0.2+ async execution
            final_state = await wrapped_graph.ainvoke(initial_state)
        elif hasattr(wrapped_graph, 'invoke'):
            # Fallback to sync execution (run in executor)
            final_state = await asyncio.get_event_loop().run_in_executor(
                None, wrapped_graph.invoke, initial_state
            )
        else:
            # Manual graph execution for development/testing
            final_state = await self._manual_graph_execution(
                wrapped_graph, initial_state, metrics
            )
        
        return final_state

    def _wrap_graph_for_monitoring(
        self, 
        graph: StateGraph, 
        metrics: ExecutionMetrics,
        execution_id: str
    ) -> StateGraph:
        """Wrap graph nodes with monitoring and metrics collection"""
        
        # For production LangGraph, we would use middleware/interceptors
        # For now, we'll track at the execution level
        
        if hasattr(graph, 'nodes'):
            # Track node information
            for node_name, node_func in graph.nodes.items():
                # Wrap node functions with monitoring
                original_func = node_func
                
                async def monitored_node(state: GraphState, 
                                       node_name=node_name, 
                                       original=original_func) -> GraphState:
                    node_start = time.time()
                    
                    logger.debug(
                        "node_execution_started",
                        execution_id=execution_id,
                        node_name=node_name,
                        state_keys=list(state.keys())
                    )
                    
                    try:
                        result_state = await original(state)
                        execution_time = (time.time() - node_start) * 1000
                        
                        metrics.nodes_executed += 1
                        metrics.node_execution_times[node_name] = execution_time
                        metrics.graph_hops += 1
                        
                        logger.debug(
                            "node_execution_completed",
                            execution_id=execution_id,
                            node_name=node_name,
                            execution_time_ms=execution_time
                        )
                        
                        return result_state
                        
                    except Exception as e:
                        metrics.nodes_failed += 1
                        metrics.errors.append({
                            "node": node_name,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        logger.error(
                            "node_execution_failed",
                            execution_id=execution_id,
                            node_name=node_name,
                            error=str(e)
                        )
                        
                        # Continue execution with error in state
                        state["error_context"][f"{node_name}_error"] = str(e)
                        return state
                
                graph.nodes[node_name] = monitored_node
        
        return graph

    async def _manual_graph_execution(
        self,
        graph: StateGraph,
        initial_state: GraphState,
        metrics: ExecutionMetrics
    ) -> GraphState:
        """Manual graph execution for development/testing when LangGraph unavailable"""
        
        if not hasattr(graph, 'nodes') or not hasattr(graph, 'edges'):
            raise ValueError("Graph missing required nodes/edges for manual execution")
        
        current_state = initial_state
        executed_nodes = set()
        
        # Simple sequential execution for testing
        for node_name, node_func in graph.nodes.items():
            if node_name in [START, END]:
                continue
                
            try:
                logger.debug(f"Executing node: {node_name}")
                current_state = await node_func(current_state)
                executed_nodes.add(node_name)
                metrics.nodes_executed += 1
                metrics.graph_hops += 1
                
            except Exception as e:
                metrics.nodes_failed += 1
                logger.error(f"Node {node_name} failed: {e}")
                current_state["error_context"][f"{node_name}_error"] = str(e)
        
        return current_state

    async def _register_graph_tools(
        self, 
        graph: StateGraph, 
        state: GraphState
    ) -> None:
        """Register all agent tools referenced in the graph"""
        
        # Extract agent information from state
        agent_id = state.get('agent_id')
        if not agent_id:
            return
        
        # Cache tool registry for performance
        if agent_id not in self._tool_registry_cache:
            # This would typically load tools from the agent registry
            # For now, we'll use a placeholder
            self._tool_registry_cache[agent_id] = {}
        
        logger.debug(
            "graph_tools_registered",
            agent_id=agent_id,
            tools_count=len(self._tool_registry_cache[agent_id])
        )

    @asynccontextmanager
    async def _execution_context(self, execution_id: str, timeout_seconds: int):
        """Manage execution context with timeout and cleanup"""
        
        # Create execution lock
        lock = asyncio.Lock()
        self._execution_locks[execution_id] = lock
        
        try:
            async with lock:
                # Track active execution
                current_task = asyncio.current_task()
                self._active_executions[execution_id] = current_task
                
                # Apply timeout
                async with asyncio.timeout(timeout_seconds):
                    yield
                    
        finally:
            # Cleanup
            self._active_executions.pop(execution_id, None)
            self._execution_locks.pop(execution_id, None)

    def _create_error_result(
        self,
        execution_id: str,
        metrics: ExecutionMetrics,
        error_message: str,
        context: Dict[str, Any]
    ) -> ExecutionResult:
        """Create ExecutionResult for failed executions"""
        
        metrics.end_time = datetime.utcnow()
        metrics.total_execution_time_ms = (
            metrics.end_time - metrics.start_time
        ).total_seconds() * 1000
        
        return ExecutionResult(
            success=False,
            final_state={},
            metrics=metrics,
            execution_id=execution_id,
            plan_id=context.get('plan_id'),
            user_id=context.get('user_id'),
            request_id=context.get('request_id'),
            error_message=error_message
        )

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        
        if execution_id not in self._active_executions:
            logger.warning(f"Execution {execution_id} not found for cancellation")
            return False
        
        task = self._active_executions[execution_id]
        task.cancel()
        
        logger.info(
            "execution_cancelled",
            execution_id=execution_id
        )
        
        return True

    def get_active_executions(self) -> List[str]:
        """Get list of currently active execution IDs"""
        return list(self._active_executions.keys())

    def get_execution_metrics(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get real-time metrics for an active execution"""
        
        if execution_id not in self._active_executions:
            return None
        
        # This would return real-time metrics
        # Implementation depends on how we track metrics during execution
        return {
            "execution_id": execution_id,
            "status": "running",
            "start_time": datetime.utcnow().isoformat()
        }

    async def cleanup(self) -> None:
        """Cleanup resources and cancel active executions"""
        
        # Cancel all active executions
        for execution_id, task in list(self._active_executions.items()):
            task.cancel()
            
        # Clear caches
        self._agent_cache.clear()
        self._tool_registry_cache.clear()
        
        logger.info("langgraph_runner_cleanup_completed")