"""
Cross-Agent Workflow Management
Handles graphs containing nodes from multiple agents with isolated namespaces
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional, Set, Type, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum

from src.agents.base.agent import HardenedAgent
from src.agents.base.registry import ToolRegistry
from src.memory.mem0_service import FinOpsMemoryService
from src.agents.routing.selector import AgentRouter
from src.agents.models import ExecutionPlan, Task
from src.planner.compiler import GraphState

logger = structlog.get_logger(__name__)


class AgentNamespaceStatus(str, Enum):
    """Status of agent namespace in cross-agent execution"""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentNamespace:
    """Isolated agent namespace within cross-agent workflow"""
    
    agent_id: str
    domain: str
    agent_class: Type[HardenedAgent]
    agent_instance: Optional[HardenedAgent] = None
    
    # Tool isolation
    tool_registry: Dict[str, Any] = field(default_factory=dict)
    tool_schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    namespace_prefix: str = ""
    
    # Execution context
    status: AgentNamespaceStatus = AgentNamespaceStatus.INITIALIZING
    tasks_assigned: List[str] = field(default_factory=list)
    tasks_completed: List[str] = field(default_factory=list)
    tasks_failed: List[str] = field(default_factory=list)
    
    # Memory context isolation
    memory_session_id: str = field(default_factory=lambda: uuid4().hex)
    memory_context: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    initialization_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    task_execution_times: Dict[str, float] = field(default_factory=dict)
    
    def get_namespaced_tool_name(self, tool_name: str) -> str:
        """Get tool name with namespace prefix"""
        return f"{self.namespace_prefix}{tool_name}" if self.namespace_prefix else tool_name
    
    def get_summary(self) -> Dict[str, Any]:
        """Get namespace execution summary"""
        return {
            "agent_id": self.agent_id,
            "domain": self.domain,
            "status": self.status.value,
            "tasks_assigned": len(self.tasks_assigned),
            "tasks_completed": len(self.tasks_completed),
            "tasks_failed": len(self.tasks_failed),
            "tools_available": len(self.tool_registry),
            "total_execution_time_ms": self.total_execution_time_ms,
            "memory_session_id": self.memory_session_id
        }


@dataclass
class CrossAgentWorkflowResult:
    """Result of cross-agent workflow execution"""
    
    success: bool
    workflow_id: str
    
    # Agent namespace results
    agent_results: Dict[str, AgentNamespace]
    
    # Execution metrics
    total_agents: int = 0
    successful_agents: int = 0
    total_execution_time_ms: float = 0.0
    
    # Memory consistency
    memory_operations: List[Dict[str, Any]] = field(default_factory=list)
    memory_conflicts: List[str] = field(default_factory=list)
    
    # Context propagation
    shared_context: Dict[str, Any] = field(default_factory=dict)
    context_transfers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Error tracking
    failed_agents: List[str] = field(default_factory=list)
    error_summary: Dict[str, str] = field(default_factory=dict)


class CrossAgentWorkflowManager:
    """
    Manages execution of workflows spanning multiple specialized agents
    
    Features:
    - Isolated tool namespaces per agent
    - Shared memory session with conflict resolution
    - Context propagation between agents
    - Deterministic execution ordering
    - Performance optimization across agents
    """
    
    def __init__(
        self, 
        memory_service: FinOpsMemoryService,
        agent_router: AgentRouter
    ):
        """
        Initialize cross-agent workflow manager
        
        Args:
            memory_service: Shared memory service
            agent_router: Agent routing service
        """
        self.memory_service = memory_service
        self.agent_router = agent_router
        
        # Namespace management
        self._active_namespaces: Dict[str, AgentNamespace] = {}
        self._namespace_locks: Dict[str, asyncio.Lock] = {}
        
        # Memory coordination
        self._shared_memory_session: Optional[str] = None
        self._memory_operation_log: List[Dict[str, Any]] = []
        
        # Performance optimization
        self._agent_startup_cache: Dict[str, HardenedAgent] = {}
        self._cross_agent_metrics: Dict[str, Any] = {}
        
        logger.info(
            "cross_agent_workflow_manager_initialized",
            memory_service_available=memory_service is not None,
            agent_router_available=agent_router is not None
        )

    async def execute_cross_agent_workflow(
        self,
        execution_plan: ExecutionPlan,
        workflow_context: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> CrossAgentWorkflowResult:
        """
        Execute workflow spanning multiple agents
        
        Args:
            execution_plan: Plan containing tasks for multiple agents
            workflow_context: Shared workflow context
            workflow_id: Optional workflow identifier
            
        Returns:
            CrossAgentWorkflowResult with execution details
        """
        workflow_id = workflow_id or uuid4().hex
        start_time = datetime.utcnow()
        
        logger.info(
            "cross_agent_workflow_started",
            workflow_id=workflow_id,
            plan_id=str(execution_plan.id),
            total_tasks=len(execution_plan.tasks)
        )
        
        try:
            # Initialize shared memory session
            await self._initialize_shared_memory_session(workflow_context)
            
            # Analyze plan for cross-agent requirements
            agent_task_mapping = await self._analyze_cross_agent_requirements(
                execution_plan
            )
            
            # Create isolated namespaces for each agent
            agent_namespaces = await self._create_agent_namespaces(
                agent_task_mapping, workflow_context
            )
            
            # Execute workflow with context propagation
            execution_results = await self._execute_with_context_propagation(
                execution_plan, agent_namespaces, workflow_context
            )
            
            # Finalize memory operations
            await self._finalize_memory_operations(
                agent_namespaces, workflow_context
            )
            
            # Calculate final metrics
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = CrossAgentWorkflowResult(
                success=all(ns.status == AgentNamespaceStatus.COMPLETED for ns in agent_namespaces.values()),
                workflow_id=workflow_id,
                agent_results=agent_namespaces,
                total_agents=len(agent_namespaces),
                successful_agents=len([ns for ns in agent_namespaces.values() if ns.status == AgentNamespaceStatus.COMPLETED]),
                total_execution_time_ms=total_time,
                memory_operations=self._memory_operation_log.copy(),
                shared_context=workflow_context.copy(),
                failed_agents=[ns.agent_id for ns in agent_namespaces.values() if ns.status == AgentNamespaceStatus.FAILED],
                error_summary=self._collect_namespace_errors(agent_namespaces)
            )
            
            logger.info(
                "cross_agent_workflow_completed",
                workflow_id=workflow_id,
                total_time_ms=total_time,
                successful_agents=result.successful_agents,
                failed_agents=len(result.failed_agents)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "cross_agent_workflow_failed",
                workflow_id=workflow_id,
                error=str(e),
                error_type=type(e).__name__
            )
            
            return CrossAgentWorkflowResult(
                success=False,
                workflow_id=workflow_id,
                agent_results={},
                error_summary={"workflow_error": str(e)}
            )

    async def _analyze_cross_agent_requirements(
        self,
        execution_plan: ExecutionPlan
    ) -> Dict[str, List[Task]]:
        """Analyze plan to determine which agents are needed for which tasks"""
        
        agent_task_mapping = {}
        
        for task in execution_plan.tasks:
            # Determine which agent should handle this task
            # This could be based on tool_name prefix, explicit agent_id, or routing
            
            # Extract agent domain from tool name (e.g., "finops_compute_npv" -> "finops")
            tool_parts = task.tool_name.split('_')
            potential_domain = tool_parts[0] if tool_parts else 'default'
            
            # Validate domain through router
            try:
                agent_class = await self.agent_router.select(
                    domain=potential_domain,
                    user_ctx={'message': task.description or task.name}
                )
                
                # Get domain from agent class
                if hasattr(agent_class, 'get_domain'):
                    domain = agent_class.get_domain()
                else:
                    domain = potential_domain
                
                if domain not in agent_task_mapping:
                    agent_task_mapping[domain] = []
                
                agent_task_mapping[domain].append(task)
                
            except Exception as e:
                logger.warning(
                    "task_agent_mapping_failed",
                    task_id=str(task.id),
                    tool_name=task.tool_name,
                    error=str(e)
                )
                
                # Fallback to default domain
                if 'default' not in agent_task_mapping:
                    agent_task_mapping['default'] = []
                agent_task_mapping['default'].append(task)
        
        logger.info(
            "cross_agent_requirements_analyzed",
            total_domains=len(agent_task_mapping),
            domain_task_counts={domain: len(tasks) for domain, tasks in agent_task_mapping.items()}
        )
        
        return agent_task_mapping

    async def _create_agent_namespaces(
        self,
        agent_task_mapping: Dict[str, List[Task]],
        workflow_context: Dict[str, Any]
    ) -> Dict[str, AgentNamespace]:
        """Create isolated namespaces for each agent in the workflow"""
        
        agent_namespaces = {}
        
        for domain, tasks in agent_task_mapping.items():
            try:
                start_time = datetime.utcnow()
                
                # Get agent class from router
                agent_class = await self.agent_router.select(
                    domain=domain,
                    user_ctx={'message': f"Initialize {domain} agent"}
                )
                
                # Create namespace
                namespace = AgentNamespace(
                    agent_id=f"{domain}_{uuid4().hex[:8]}",
                    domain=domain,
                    agent_class=agent_class,
                    namespace_prefix=f"{domain}_",
                    tasks_assigned=[str(task.id) for task in tasks]
                )
                
                # Initialize agent instance
                await self._initialize_agent_instance(namespace, workflow_context)
                
                # Extract and namespace tools
                await self._setup_tool_namespace(namespace)
                
                # Calculate initialization time
                namespace.initialization_time_ms = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000
                
                namespace.status = AgentNamespaceStatus.READY
                
                agent_namespaces[domain] = namespace
                
                logger.info(
                    "agent_namespace_created",
                    domain=domain,
                    agent_id=namespace.agent_id,
                    tasks_assigned=len(namespace.tasks_assigned),
                    tools_available=len(namespace.tool_registry),
                    initialization_time_ms=namespace.initialization_time_ms
                )
                
            except Exception as e:
                logger.error(
                    "agent_namespace_creation_failed",
                    domain=domain,
                    error=str(e)
                )
                
                # Create failed namespace for tracking
                failed_namespace = AgentNamespace(
                    agent_id=f"{domain}_failed",
                    domain=domain,
                    agent_class=None,
                    status=AgentNamespaceStatus.FAILED
                )
                agent_namespaces[domain] = failed_namespace
        
        return agent_namespaces

    async def _initialize_agent_instance(
        self,
        namespace: AgentNamespace,
        workflow_context: Dict[str, Any]
    ) -> None:
        """Initialize agent instance for namespace"""
        
        if not namespace.agent_class:
            raise ValueError(f"No agent class available for domain: {namespace.domain}")
        
        # Check cache for existing instance
        cache_key = namespace.domain
        if cache_key in self._agent_startup_cache:
            namespace.agent_instance = self._agent_startup_cache[cache_key]
            logger.debug("agent_instance_cached", domain=namespace.domain)
            return
        
        # Create new agent instance
        namespace.agent_instance = namespace.agent_class(
            memory_service=self.memory_service,
            agent_id=namespace.agent_id
        )
        
        # Cache for reuse
        self._agent_startup_cache[cache_key] = namespace.agent_instance
        
        logger.debug(
            "agent_instance_initialized",
            domain=namespace.domain,
            agent_id=namespace.agent_id
        )

    async def _setup_tool_namespace(self, namespace: AgentNamespace) -> None:
        """Setup isolated tool namespace for agent"""
        
        if not namespace.agent_instance:
            raise ValueError("Agent instance not initialized")
        
        # Extract tools using registry
        namespace.tool_registry = ToolRegistry.extract_tools(namespace.agent_instance)
        namespace.tool_schemas = ToolRegistry.generate_schema(namespace.tool_registry)
        
        # Create namespaced tool mapping
        namespaced_tools = {}
        for tool_name, tool_func in namespace.tool_registry.items():
            namespaced_name = namespace.get_namespaced_tool_name(tool_name)
            namespaced_tools[namespaced_name] = tool_func
        
        # Update registry with namespaced names
        namespace.tool_registry.update(namespaced_tools)
        
        logger.debug(
            "tool_namespace_setup",
            domain=namespace.domain,
            original_tools=len(namespace.tool_registry) // 2,  # Divide by 2 since we doubled them
            namespace_prefix=namespace.namespace_prefix
        )

    async def _execute_with_context_propagation(
        self,
        execution_plan: ExecutionPlan,
        agent_namespaces: Dict[str, AgentNamespace],
        workflow_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute plan with context propagation between agents"""
        
        execution_results = {}
        
        # Process tasks in dependency order
        ready_tasks = execution_plan.get_ready_tasks()
        
        while ready_tasks:
            # Group tasks by agent domain
            agent_task_groups = {}
            for task in ready_tasks:
                agent_domain = self._get_task_agent_domain(task, agent_namespaces)
                if agent_domain not in agent_task_groups:
                    agent_task_groups[agent_domain] = []
                agent_task_groups[agent_domain].append(task)
            
            # Execute tasks for each agent
            for domain, tasks in agent_task_groups.items():
                if domain not in agent_namespaces:
                    logger.error("agent_namespace_not_found", domain=domain)
                    continue
                
                namespace = agent_namespaces[domain]
                namespace.status = AgentNamespaceStatus.ACTIVE
                
                # Execute tasks in this namespace
                for task in tasks:
                    await self._execute_task_in_namespace(
                        task, namespace, workflow_context, execution_results
                    )
            
            # Update plan progress and get next ready tasks
            execution_plan.update_progress()
            ready_tasks = execution_plan.get_ready_tasks()
        
        return execution_results

    async def _execute_task_in_namespace(
        self,
        task: Task,
        namespace: AgentNamespace,
        workflow_context: Dict[str, Any],
        execution_results: Dict[str, Any]
    ) -> None:
        """Execute a single task within an agent namespace"""
        
        start_time = datetime.utcnow()
        task_id = str(task.id)
        
        logger.info(
            "task_execution_started_in_namespace",
            task_id=task_id,
            tool_name=task.tool_name,
            domain=namespace.domain,
            agent_id=namespace.agent_id
        )
        
        try:
            # Get namespaced tool name
            namespaced_tool = namespace.get_namespaced_tool_name(task.tool_name)
            
            # Check if tool exists in namespace
            if namespaced_tool not in namespace.tool_registry:
                # Try original tool name
                if task.tool_name not in namespace.tool_registry:
                    raise ValueError(f"Tool '{task.tool_name}' not found in namespace '{namespace.domain}'")
                tool_func = namespace.tool_registry[task.tool_name]
            else:
                tool_func = namespace.tool_registry[namespaced_tool]
            
            # Prepare execution context with namespace isolation
            task_context = {
                **workflow_context,
                'namespace_id': namespace.agent_id,
                'memory_session_id': namespace.memory_session_id,
                'cross_agent_context': self._get_cross_agent_context(execution_results)
            }
            
            # Execute tool
            result = await tool_func(**task.inputs)
            
            # Record execution
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            namespace.task_execution_times[task_id] = execution_time
            namespace.tasks_completed.append(task_id)
            namespace.total_execution_time_ms += execution_time
            
            # Store result
            execution_results[task_id] = {
                'task': task,
                'result': result,
                'namespace': namespace.domain,
                'execution_time_ms': execution_time
            }
            
            # Update task status
            task.status = "completed"
            task.completed_at = datetime.utcnow()
            
            logger.info(
                "task_execution_completed_in_namespace",
                task_id=task_id,
                domain=namespace.domain,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            namespace.task_execution_times[task_id] = execution_time
            namespace.tasks_failed.append(task_id)
            
            # Update task status
            task.status = "failed"
            task.completed_at = datetime.utcnow()
            
            logger.error(
                "task_execution_failed_in_namespace",
                task_id=task_id,
                domain=namespace.domain,
                error=str(e)
            )
            
            # Store error result
            execution_results[task_id] = {
                'task': task,
                'error': str(e),
                'namespace': namespace.domain,
                'execution_time_ms': execution_time
            }

    def _get_task_agent_domain(
        self,
        task: Task,
        agent_namespaces: Dict[str, AgentNamespace]
    ) -> str:
        """Determine which agent domain should handle a task"""
        
        task_id = str(task.id)
        
        # Check which namespace has this task assigned
        for domain, namespace in agent_namespaces.items():
            if task_id in namespace.tasks_assigned:
                return domain
        
        # Fallback: analyze tool name
        tool_parts = task.tool_name.split('_')
        potential_domain = tool_parts[0] if tool_parts else 'default'
        
        if potential_domain in agent_namespaces:
            return potential_domain
        
        # Last resort: default domain
        return 'default' if 'default' in agent_namespaces else list(agent_namespaces.keys())[0]

    def _get_cross_agent_context(
        self,
        execution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build cross-agent context from previous execution results"""
        
        context = {
            'completed_tasks': [],
            'results_by_domain': {},
            'shared_outputs': {}
        }
        
        for task_id, result_data in execution_results.items():
            if 'error' not in result_data:
                context['completed_tasks'].append(task_id)
                
                domain = result_data.get('namespace', 'unknown')
                if domain not in context['results_by_domain']:
                    context['results_by_domain'][domain] = []
                
                context['results_by_domain'][domain].append({
                    'task_id': task_id,
                    'result': result_data.get('result')
                })
                
                # Add to shared outputs if result is serializable
                try:
                    json.dumps(result_data.get('result'))
                    context['shared_outputs'][task_id] = result_data.get('result')
                except:
                    pass
        
        return context

    async def _initialize_shared_memory_session(
        self,
        workflow_context: Dict[str, Any]
    ) -> None:
        """Initialize shared memory session for cross-agent workflow"""
        
        self._shared_memory_session = workflow_context.get('memory_session_id') or uuid4().hex
        
        logger.info(
            "shared_memory_session_initialized",
            session_id=self._shared_memory_session
        )

    async def _finalize_memory_operations(
        self,
        agent_namespaces: Dict[str, AgentNamespace],
        workflow_context: Dict[str, Any]
    ) -> None:
        """Finalize memory operations ensuring single persistence"""
        
        # Collect all memory operations from namespaces
        all_memory_data = []
        
        for namespace in agent_namespaces.values():
            if namespace.memory_context:
                all_memory_data.append({
                    'domain': namespace.domain,
                    'agent_id': namespace.agent_id,
                    'context': namespace.memory_context
                })
        
        # Store consolidated memory data once
        if all_memory_data and self.memory_service:
            try:
                await self.memory_service.store_conversation_memory(
                    user_id=workflow_context.get('user_id', 'system'),
                    message=f"Cross-agent workflow completed with {len(agent_namespaces)} agents",
                    conversation_id=workflow_context.get('request_id', self._shared_memory_session),
                    context={
                        'workflow_type': 'cross_agent',
                        'agents_involved': list(agent_namespaces.keys()),
                        'memory_session_id': self._shared_memory_session,
                        'namespace_data': all_memory_data
                    }
                )
                
                self._memory_operation_log.append({
                    'operation': 'workflow_memory_persistence',
                    'session_id': self._shared_memory_session,
                    'agents_count': len(agent_namespaces),
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                logger.info(
                    "cross_agent_memory_finalized",
                    session_id=self._shared_memory_session,
                    agents_count=len(agent_namespaces)
                )
                
            except Exception as e:
                logger.error(
                    "cross_agent_memory_finalization_failed",
                    error=str(e)
                )

    def _collect_namespace_errors(
        self,
        agent_namespaces: Dict[str, AgentNamespace]
    ) -> Dict[str, str]:
        """Collect error summary from all namespaces"""
        
        error_summary = {}
        
        for domain, namespace in agent_namespaces.items():
            if namespace.status == AgentNamespaceStatus.FAILED:
                failed_tasks = len(namespace.tasks_failed)
                if failed_tasks > 0:
                    error_summary[domain] = f"Agent failed with {failed_tasks} task failures"
                else:
                    error_summary[domain] = "Agent initialization or execution failed"
        
        return error_summary

    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active cross-agent workflows"""
        
        active_workflows = {}
        
        for domain, namespace in self._active_namespaces.items():
            if namespace.status in [AgentNamespaceStatus.ACTIVE, AgentNamespaceStatus.READY]:
                active_workflows[domain] = namespace.get_summary()
        
        return active_workflows

    async def cleanup(self) -> None:
        """Cleanup resources and agent instances"""
        
        # Clear active namespaces
        self._active_namespaces.clear()
        self._namespace_locks.clear()
        
        # Clear agent cache
        self._agent_startup_cache.clear()
        
        # Clear memory logs
        self._memory_operation_log.clear()
        
        logger.info("cross_agent_workflow_manager_cleanup_completed")