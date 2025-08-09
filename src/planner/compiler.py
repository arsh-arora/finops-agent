"""
DynamicGraphCompiler - ExecutionPlan to LangGraph StateGraph conversion
Auto-inserts mem0 nodes and handles parallel execution
"""

import time
import networkx as nx
import structlog
from typing import Dict, Any, List, Callable, Optional, Set, Tuple
from datetime import datetime, timezone

try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    # Fallback for development/testing
    LANGGRAPH_AVAILABLE = False
    class StateGraph:
        def __init__(self, schema=None): 
            self.nodes = {}
            self.edges = []
            self.schema = schema
        def add_node(self, name, func): 
            self.nodes[name] = func
        def add_edge(self, from_node, to_node): 
            self.edges.append((from_node, to_node))
        def compile(self): 
            return self
    START = "start"
    END = "end"

from src.memory.mem0_service import FinOpsMemoryService
from src.agents.models import ExecutionPlan, Task, CompilationResult, TaskStatus
from src.agents.base.agent import HardenedAgent
from .exceptions import PlanningError

logger = structlog.get_logger(__name__)


class GraphState(Dict[str, Any]):
    """State object for LangGraph execution"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize required state fields
        self.setdefault("request_id", "unknown")
        self.setdefault("user_id", "unknown")  
        self.setdefault("current_task", None)
        self.setdefault("task_results", {})
        self.setdefault("memory_context", [])
        self.setdefault("error_context", {})
        self.setdefault("execution_log", [])


class DynamicGraphCompiler:
    """
    Compiles ExecutionPlan into runnable LangGraph StateGraph
    
    Features:
    - Auto-inserts mem0.nodes.Retrieval (first) and mem0.nodes.Persistence (last)
    - Handles parallel execution groups
    - Detects cycles and validates dependencies
    - Optimizes task ordering by cost
    """
    
    def __init__(self, memory_service: FinOpsMemoryService):
        """
        Initialize compiler with memory service
        
        Args:
            memory_service: Memory service for retrieval/persistence nodes
        """
        self.memory_service = memory_service
        
        logger.info("dynamic_graph_compiler_initialized")

    async def compile(
        self, 
        plan: ExecutionPlan, 
        agent: HardenedAgent
    ) -> CompilationResult:
        """
        Compile ExecutionPlan into StateGraph
        
        Args:
            plan: Execution plan to compile
            agent: Agent that will execute the tasks
            
        Returns:
            CompilationResult with StateGraph or error details
        """
        start_time = time.time()
        
        logger.info(
            "graph_compilation_started",
            plan_id=str(plan.id),
            tasks_count=len(plan.tasks),
            dependencies_count=len(plan.dependencies),
            agent_domain=agent.get_domain()
        )
        
        try:
            # Validate plan structure
            self._validate_plan(plan)
            
            # Detect cycles in dependency graph
            if self._has_cycles(plan):
                raise PlanningError("Plan contains circular dependencies")
            
            # Create StateGraph
            if LANGGRAPH_AVAILABLE:
                state_graph = StateGraph(GraphState)
            else:
                state_graph = StateGraph(schema=GraphState)
            
            # Insert memory retrieval node (first)
            retrieval_node = self._create_retrieval_node(plan, agent)
            state_graph.add_node("memory_retrieval", retrieval_node)
            state_graph.add_edge(START, "memory_retrieval")
            
            # Add task nodes
            task_nodes = {}
            for task in plan.tasks:
                node_name = f"task_{task.id.hex[:8]}"
                task_node = self._create_task_node(task, agent)
                state_graph.add_node(node_name, task_node)
                task_nodes[task.id] = node_name
            
            # Add dependency edges
            self._add_dependency_edges(state_graph, plan, task_nodes)
            
            # Handle parallel groups
            self._handle_parallel_execution(state_graph, plan, task_nodes)
            
            # Insert memory persistence node (last)
            persistence_node = self._create_persistence_node(plan, agent)
            state_graph.add_node("memory_persistence", persistence_node)
            
            # Connect final tasks to persistence
            self._connect_final_tasks(state_graph, plan, task_nodes)
            state_graph.add_edge("memory_persistence", END)
            
            # Compile graph
            if LANGGRAPH_AVAILABLE:
                compiled_graph = state_graph.compile()
            else:
                compiled_graph = state_graph  # Mock for development
            
            compilation_time = (time.time() - start_time) * 1000
            
            # Analyze compiled graph
            nodes_created = len(state_graph.nodes) + 2  # +2 for START/END
            edges_created = len(getattr(state_graph, 'edges', []))
            parallel_groups = len(plan.get_parallel_groups())
            
            logger.info(
                "graph_compilation_completed",
                plan_id=str(plan.id),
                compilation_time_ms=compilation_time,
                nodes_created=nodes_created,
                edges_created=edges_created,
                parallel_groups=parallel_groups
            )
            
            return CompilationResult(
                success=True,
                graph=compiled_graph,
                graph_json=self._serialize_graph(compiled_graph),
                compilation_time_ms=compilation_time,
                nodes_created=nodes_created,
                edges_created=edges_created,
                has_cycles=False,  # We already checked
                parallel_groups_count=parallel_groups
            )
            
        except Exception as e:
            compilation_time = (time.time() - start_time) * 1000
            
            logger.error(
                "graph_compilation_failed",
                plan_id=str(plan.id),
                error=str(e),
                error_type=type(e).__name__,
                compilation_time_ms=compilation_time
            )
            
            return CompilationResult(
                success=False,
                error_message=str(e),
                compilation_time_ms=compilation_time,
                nodes_created=0,
                edges_created=0
            )

    def _validate_plan(self, plan: ExecutionPlan) -> None:
        """Validate plan structure before compilation"""
        if not plan.tasks:
            raise PlanningError("Plan must contain at least one task")
        
        # Validate task tools exist (basic check)
        for task in plan.tasks:
            if not task.tool_name or not task.tool_name.strip():
                raise PlanningError(f"Task {task.id} has invalid tool_name")
        
        # Validate dependencies reference existing tasks
        task_ids = {task.id for task in plan.tasks}
        for dep in plan.dependencies:
            if dep.parent_id not in task_ids:
                raise PlanningError(f"Dependency references unknown parent task: {dep.parent_id}")
            if dep.child_id not in task_ids:
                raise PlanningError(f"Dependency references unknown child task: {dep.child_id}")

    def _has_cycles(self, plan: ExecutionPlan) -> bool:
        """Detect cycles in task dependency graph"""
        if not plan.dependencies:
            return False
        
        # Build networkx graph for cycle detection
        G = nx.DiGraph()
        
        # Add nodes
        for task in plan.tasks:
            G.add_node(str(task.id))
        
        # Add edges  
        for dep in plan.dependencies:
            G.add_edge(str(dep.parent_id), str(dep.child_id))
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(G))
            return len(cycles) > 0
        except nx.NetworkXError:
            return False

    def _create_retrieval_node(self, plan: ExecutionPlan, agent: HardenedAgent) -> Callable:
        """Create memory retrieval node (inserted first)"""
        
        async def memory_retrieval_node(state: GraphState) -> GraphState:
            """Retrieve relevant memories for plan execution context"""
            try:
                logger.info(
                    "memory_retrieval_started",
                    plan_id=str(plan.id),
                    user_id=plan.user_id,
                    request_id=plan.request_id
                )
                
                # Retrieve relevant memories using the original message
                memories = await self.memory_service.retrieve_relevant_memories(
                    query=plan.original_message,
                    user_id=plan.user_id,
                    limit=5
                )
                
                # Extract memory content
                memory_context = []
                if memories:
                    memory_context = [memory.content for memory in memories]
                
                # Update state with memory context
                state["memory_context"] = memory_context
                state["request_id"] = plan.request_id
                state["user_id"] = plan.user_id
                if "execution_log" not in state:
                    state["execution_log"] = []
                state["execution_log"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "memory_retrieval",
                    "memories_retrieved": len(memory_context)
                })
                
                logger.info(
                    "memory_retrieval_completed",
                    plan_id=str(plan.id),
                    memories_retrieved=len(memory_context)
                )
                
                return state
                
            except Exception as e:
                logger.error(
                    "memory_retrieval_failed",
                    plan_id=str(plan.id),
                    error=str(e)
                )
                
                # Continue with empty context on failure
                state["memory_context"] = []
                if "error_context" not in state:
                    state["error_context"] = {}
                state["error_context"]["memory_retrieval_error"] = str(e)
                return state
        
        return memory_retrieval_node

    def _create_persistence_node(self, plan: ExecutionPlan, agent: HardenedAgent) -> Callable:
        """Create memory persistence node (inserted last)"""
        
        async def memory_persistence_node(state: GraphState) -> GraphState:
            """Persist execution results to memory"""
            try:
                logger.info(
                    "memory_persistence_started",
                    plan_id=str(plan.id),
                    user_id=state.get("user_id", plan.user_id)
                )
                
                # Aggregate task results
                task_results = state.get("task_results", {})
                execution_summary = self._create_execution_summary(task_results, plan)
                
                # Store execution results in memory
                await self.memory_service.store_conversation_memory(
                    user_id=state.get("user_id", plan.user_id),
                    message=execution_summary,
                    conversation_id=plan.request_id,
                    context={
                        "plan_id": str(plan.id),
                        "agent_domain": agent.get_domain(),
                        "agent_id": agent.agent_id,
                        "tasks_executed": len(task_results),
                        "execution_log": state.get("execution_log", [])
                    }
                )
                
                # Update execution log
                if "execution_log" not in state:
                    state["execution_log"] = []
                state["execution_log"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "memory_persistence", 
                    "results_persisted": len(task_results)
                })
                
                logger.info(
                    "memory_persistence_completed",
                    plan_id=str(plan.id),
                    results_persisted=len(task_results)
                )
                
                return state
                
            except Exception as e:
                logger.error(
                    "memory_persistence_failed",
                    plan_id=str(plan.id),
                    error=str(e)
                )
                
                if "error_context" not in state:
                    state["error_context"] = {}
                state["error_context"]["memory_persistence_error"] = str(e)
                return state
        
        return memory_persistence_node

    def _create_task_node(self, task: Task, agent: HardenedAgent) -> Callable:
        """Create execution node for a specific task"""
        
        async def task_execution_node(state: GraphState) -> GraphState:
            """Execute specific task using agent tools"""
            try:
                logger.info(
                    "task_execution_started",
                    task_id=str(task.id),
                    tool_name=task.tool_name,
                    agent_id=agent.agent_id
                )
                
                # Mark task as in progress
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now(timezone.utc)
                
                # Get agent tools
                agent_tools = agent.get_tools()
                
                if task.tool_name not in agent_tools:
                    raise PlanningError(f"Tool '{task.tool_name}' not available in agent")
                
                # Execute tool
                tool_func = agent_tools[task.tool_name]
                result = await tool_func(**task.inputs)
                
                # Mark task as completed
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc)
                
                # Store result in state
                state["task_results"][str(task.id)] = {
                    "task_name": task.name,
                    "tool_name": task.tool_name,
                    "result": result,
                    "completed_at": task.completed_at.isoformat()
                }
                
                # Update execution log
                if "execution_log" not in state:
                    state["execution_log"] = []
                state["execution_log"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "task_execution",
                    "task_id": str(task.id),
                    "tool_name": task.tool_name,
                    "status": "completed"
                })
                
                logger.info(
                    "task_execution_completed",
                    task_id=str(task.id),
                    tool_name=task.tool_name,
                    result_length=len(str(result)) if result else 0
                )
                
                return state
                
            except Exception as e:
                logger.error(
                    "task_execution_failed",
                    task_id=str(task.id),
                    tool_name=task.tool_name,
                    error=str(e)
                )
                
                # Mark task as failed
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now(timezone.utc)
                
                # Store error in state
                if "error_context" not in state:
                    state["error_context"] = {}
                state["error_context"][str(task.id)] = str(e)
                if "execution_log" not in state:
                    state["execution_log"] = []
                state["execution_log"].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": "task_execution",
                    "task_id": str(task.id),
                    "tool_name": task.tool_name,
                    "status": "failed",
                    "error": str(e)
                })
                
                return state
        
        return task_execution_node

    def _add_dependency_edges(
        self, 
        state_graph: StateGraph, 
        plan: ExecutionPlan, 
        task_nodes: Dict[str, str]
    ) -> None:
        """Add edges for task dependencies"""
        
        # Connect memory retrieval to first tasks (tasks with no dependencies)
        first_tasks = self._get_first_tasks(plan)
        for task in first_tasks:
            node_name = task_nodes[task.id]
            state_graph.add_edge("memory_retrieval", node_name)
        
        # Add dependency edges
        for dep in plan.dependencies:
            parent_node = task_nodes[dep.parent_id]
            child_node = task_nodes[dep.child_id]
            state_graph.add_edge(parent_node, child_node)

    def _handle_parallel_execution(
        self,
        state_graph: StateGraph,
        plan: ExecutionPlan, 
        task_nodes: Dict[str, str]
    ) -> None:
        """Handle parallel execution groups"""
        
        parallel_groups = plan.get_parallel_groups()
        
        for group_id, tasks in parallel_groups.items():
            # Tasks in same parallel group can execute concurrently
            # LangGraph handles this automatically when tasks don't depend on each other
            logger.debug(
                "parallel_group_configured",
                group_id=group_id,
                tasks_count=len(tasks),
                task_names=[task.name for task in tasks]
            )

    def _connect_final_tasks(
        self,
        state_graph: StateGraph,
        plan: ExecutionPlan,
        task_nodes: Dict[str, str]
    ) -> None:
        """Connect final tasks (no children) to memory persistence"""
        
        final_tasks = self._get_final_tasks(plan)
        for task in final_tasks:
            node_name = task_nodes[task.id]
            state_graph.add_edge(node_name, "memory_persistence")

    def _get_first_tasks(self, plan: ExecutionPlan) -> List[Task]:
        """Get tasks that have no dependencies (can start immediately)"""
        if not plan.dependencies:
            return plan.tasks
        
        dependent_task_ids = {dep.child_id for dep in plan.dependencies}
        return [task for task in plan.tasks if task.id not in dependent_task_ids]

    def _get_final_tasks(self, plan: ExecutionPlan) -> List[Task]:
        """Get tasks that have no dependents (final tasks)"""
        if not plan.dependencies:
            return plan.tasks
        
        parent_task_ids = {dep.parent_id for dep in plan.dependencies}
        return [task for task in plan.tasks if task.id not in parent_task_ids]

    def _create_execution_summary(self, task_results: Dict[str, Any], plan: ExecutionPlan) -> str:
        """Create summary of execution results for memory storage"""
        
        completed_tasks = len(task_results)
        total_tasks = len(plan.tasks)
        
        summary = f"Executed {completed_tasks}/{total_tasks} tasks from plan {plan.id.hex[:8]}.\n"
        summary += f"Original request: {plan.original_message}\n"
        
        if task_results:
            summary += "Results:\n"
            for task_id, result_data in task_results.items():
                task_name = result_data.get("task_name", "Unknown task")
                result = str(result_data.get("result", ""))[:200]  # Truncate long results
                summary += f"- {task_name}: {result}\n"
        
        return summary

    def _serialize_graph(self, compiled_graph) -> Dict[str, Any]:
        """Serialize compiled graph for storage/transmission"""
        try:
            # For now, return basic metadata
            # In production, would serialize full graph structure
            return {
                "type": "StateGraph",
                "nodes": list(getattr(compiled_graph, 'nodes', {}).keys()),
                "edges": getattr(compiled_graph, 'edges', []),
                "serialized_at": datetime.now(timezone.utc).isoformat(),
                "langgraph_available": LANGGRAPH_AVAILABLE
            }
        except Exception as e:
            logger.warning(
                "graph_serialization_failed",
                error=str(e)
            )
            return {"error": f"Serialization failed: {str(e)}"}

    def optimize_task_order(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Re-order independent tasks by lowest estimated cost first"""
        
        if not plan.dependencies:
            # All tasks independent - sort by cost
            def task_cost(task: Task) -> float:
                # Simple cost estimate based on token usage
                return task.estimate_tokens * 0.0001  # Rough cost per token
            
            plan.tasks.sort(key=task_cost)
        
        return plan