"""
HardenedAgent base class - Phase 3 core implementation
Production-ready agent framework with mem0 integration and lifecycle hooks
"""

import logging
import structlog
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
from uuid import uuid4

from src.memory.mem0_service import FinOpsMemoryService
from .exceptions import AgentError, ToolError, ExecutionError
from .registry import ToolRegistry

logger = structlog.get_logger(__name__)


class HardenedAgent(ABC):
    """
    Base class for all agents in the FinOps framework.
    
    Provides:
    - Common interface (send, plan, execute, persist)  
    - Lifecycle hooks (before_plan, after_execute, on_error)
    - Tool registry integration
    - Memory service integration
    - Error handling and logging
    """
    
    def __init__(self, memory_service: FinOpsMemoryService, agent_id: Optional[str] = None):
        """
        Initialize HardenedAgent with required memory service
        
        Args:
            memory_service: FinOps memory service instance from Phase 2
            agent_id: Optional agent identifier (auto-generated if not provided)
        """
        self.memory_service = memory_service
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid4().hex[:8]}"
        self._tools: Dict[str, Callable] = {}
        self._initialized = False
        
        # Extract tools from agent instance using registry
        self._tools = ToolRegistry.extract_tools(self)
        self._initialized = True
        
        logger.info(
            "agent_initialized",
            agent_id=self.agent_id,
            agent_type=self.__class__.__name__,
            tools_count=len(self._tools)
        )

    async def send(
        self, 
        message: str, 
        user_id: str, 
        conversation_id: Optional[str] = None,
        **context
    ) -> str:
        """
        Main entry point for agent communication
        
        Args:
            message: User message to process
            user_id: User identifier for memory context
            conversation_id: Optional conversation context
            **context: Additional context data
            
        Returns:
            Agent response string
            
        Raises:
            AgentError: If processing fails
        """
        request_id = context.get('request_id', uuid4().hex)
        
        try:
            logger.info(
                "agent_message_received",
                agent_id=self.agent_id,
                user_id=user_id,
                request_id=request_id,
                message_length=len(message)
            )
            
            # Store incoming message in memory
            await self.memory_service.store_conversation_memory(
                user_id=user_id,
                message=message,
                conversation_id=conversation_id,
                context={
                    "agent_id": self.agent_id,
                    "request_id": request_id,
                    **context
                }
            )
            
            # Create execution plan
            plan_request = {
                "message": message,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "request_id": request_id,
                **context
            }
            
            execution_plan = await self.plan(plan_request)
            
            # Execute the plan
            result = await self.execute(execution_plan)
            
            # Persist results
            await self.persist(result, user_id)
            
            # Extract response from result
            response = result.get("response", "Task completed successfully")
            
            logger.info(
                "agent_message_processed",
                agent_id=self.agent_id,
                user_id=user_id,
                request_id=request_id,
                response_length=len(response)
            )
            
            return response
            
        except Exception as e:
            await self.on_error(e, {
                "user_id": user_id,
                "request_id": request_id,
                "message": message,
                **context
            })
            raise

    async def plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create execution plan from request
        
        Args:
            request: Request data including message, user context
            
        Returns:
            Execution plan dictionary
            
        Raises:
            PlanningError: If plan creation fails
        """
        request_id = request.get('request_id', 'unknown')
        
        try:
            # Lifecycle hook
            modified_request = await self.before_plan(request)
            
            logger.info(
                "agent_planning_started",
                agent_id=self.agent_id,
                request_id=request_id,
                capabilities=self.get_capabilities()
            )
            
            # Basic plan structure - to be enhanced with AgentPlanner in Deliverable 4
            plan = {
                "id": uuid4().hex,
                "request_id": request_id,
                "agent_id": self.agent_id,
                "user_id": modified_request.get("user_id"),
                "message": modified_request.get("message"),
                "capabilities_required": self.get_capabilities(),
                "tools_available": list(self._tools.keys()),
                "context": modified_request
            }
            
            logger.info(
                "agent_plan_created", 
                agent_id=self.agent_id,
                request_id=request_id,
                plan_id=plan["id"]
            )
            
            return plan
            
        except Exception as e:
            logger.error(
                "agent_planning_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise AgentError(
                f"Planning failed: {str(e)}", 
                agent_id=self.agent_id,
                request_id=request_id
            )

    async def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the given plan
        
        Args:
            plan: Execution plan from plan() method
            
        Returns:
            Execution result dictionary
            
        Raises:
            ExecutionError: If execution fails
        """
        plan_id = plan.get("id", "unknown")
        request_id = plan.get("request_id", "unknown")
        
        try:
            logger.info(
                "agent_execution_started",
                agent_id=self.agent_id,
                request_id=request_id,
                plan_id=plan_id
            )
            
            # Basic execution - to be enhanced with DynamicGraphCompiler in Deliverable 5
            message = plan.get("message", "")
            user_id = plan.get("user_id")
            
            # Retrieve relevant memories for context
            if user_id:
                memories = await self.memory_service.retrieve_relevant_memories(
                    query=message,
                    user_id=user_id,
                    limit=5
                )
                memory_context = [mem.content for mem in memories] if memories else []
            else:
                memory_context = []
            
            # Process with agent-specific logic
            response = await self._process_message(message, memory_context, plan)
            
            result = {
                "plan_id": plan_id,
                "request_id": request_id,
                "agent_id": self.agent_id,
                "response": response,
                "memory_context_used": len(memory_context),
                "execution_metadata": {
                    "capabilities_used": self.get_capabilities(),
                    "tools_available": list(self._tools.keys())
                }
            }
            
            # Lifecycle hook
            result = await self.after_execute(result)
            
            logger.info(
                "agent_execution_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                plan_id=plan_id
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "agent_execution_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                plan_id=plan_id,
                error=str(e)
            )
            raise ExecutionError(
                f"Execution failed: {str(e)}",
                agent_id=self.agent_id,
                request_id=request_id
            )

    async def persist(self, result: Dict[str, Any], user_id: str) -> None:
        """
        Persist execution results to memory
        
        Args:
            result: Execution result from execute() method
            user_id: User identifier for memory context
        """
        request_id = result.get("request_id", "unknown")
        
        try:
            logger.info(
                "agent_persistence_started",
                agent_id=self.agent_id,
                request_id=request_id,
                user_id=user_id
            )
            
            # Store agent response in memory
            await self.memory_service.store_conversation_memory(
                user_id=user_id,
                message=result.get("response", ""),
                conversation_id=result.get("conversation_id"),
                context={
                    "agent_id": self.agent_id,
                    "request_id": request_id,
                    "plan_id": result.get("plan_id"),
                    "execution_metadata": result.get("execution_metadata", {})
                }
            )
            
            logger.info(
                "agent_persistence_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(
                "agent_persistence_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                user_id=user_id,
                error=str(e)
            )
            # Don't raise - persistence failure shouldn't break the response

    # Lifecycle hooks - override in subclasses
    async def before_plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called before plan creation
        Override in subclasses for custom preprocessing
        
        Args:
            request: Original request data
            
        Returns:
            Modified request data
        """
        return request

    async def after_execute(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called after execution completion
        Override in subclasses for custom postprocessing
        
        Args:
            result: Original execution result
            
        Returns:
            Modified execution result
        """
        return result

    async def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Hook called when errors occur
        Override in subclasses for custom error handling
        
        Args:
            error: Exception that occurred
            context: Context data when error occurred
        """
        logger.error(
            "agent_error_occurred",
            agent_id=self.agent_id,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context
        )

    # Tool management methods
    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool function with this agent"""
        self._tools[name] = func
        logger.debug(
            "tool_registered",
            agent_id=self.agent_id,
            tool_name=name
        )

    def get_tool_schema(self) -> Dict[str, Any]:
        """Get JSON schema for all registered tools"""
        return ToolRegistry.generate_schema(self._tools)

    def get_tools(self) -> Dict[str, Callable]:
        """Get all registered tools"""
        return self._tools.copy()

    # Abstract methods - implement in subclasses
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of capabilities this agent provides
        
        Returns:
            List of capability strings
        """
        pass

    @abstractmethod
    def get_domain(self) -> str:
        """
        Return the domain this agent specializes in
        
        Returns:
            Domain string (e.g., 'finops', 'github', 'research')
        """
        pass

    @abstractmethod
    async def _process_message(
        self, 
        message: str, 
        memory_context: List[str], 
        plan: Dict[str, Any]
    ) -> str:
        """
        Agent-specific message processing logic
        
        Args:
            message: User message to process
            memory_context: Relevant memories from memory service
            plan: Execution plan context
            
        Returns:
            Agent response string
        """
        pass