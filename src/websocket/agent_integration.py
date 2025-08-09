"""
WebSocket-Agent Integration Bridge

Connects WebSocket message handling to the agent execution pipeline,
enabling real-time streaming of agent execution results back to clients.
"""

import asyncio
import structlog
import traceback
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession

from ..agents import get_agent_registry
from ..agents.models import ChatRequest
from ..planner.planner import AgentPlanner
from ..planner.compiler import DynamicGraphCompiler
from ..orchestration.langgraph_runner import LangGraphRunner
from ..memory.mem0_service import finops_memory_service
from .connection_manager import ConnectionManager

logger = structlog.get_logger(__name__)


class WebSocketAgentBridge:
    """
    Bridges WebSocket connections to agent execution pipeline.
    
    Handles the complete flow:
    WebSocket Message → Agent Selection → Plan Creation → Graph Compilation → 
    → LangGraph Execution → Real-time Result Streaming
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize WebSocket-Agent bridge.
        
        Args:
            connection_manager: WebSocket connection manager instance
        """
        self.connection_manager = connection_manager
        self.agent_registry = None
        self.planner = None
        self.compiler = None
        self.runner = None
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> None:
        """Initialize agent system components."""
        try:
            logger.info("Initializing WebSocket-Agent bridge")
            
            # Initialize memory service
            await finops_memory_service.initialize()
            
            # Initialize agent components
            self.agent_registry = get_agent_registry()
            # Agent registry initializes in constructor, no async initialize needed
            
            # Get LLM client from agent registry
            from ..llm.openrouter_client import openrouter_client
            self.planner = AgentPlanner(
                memory_service=finops_memory_service,
                llm_client=openrouter_client
            )
            self.compiler = DynamicGraphCompiler(memory_service=finops_memory_service)
            self.runner = LangGraphRunner(memory_service=finops_memory_service)
            
            logger.info("WebSocket-Agent bridge initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize WebSocket-Agent bridge", error=str(e))
            raise
    
    async def cleanup(self) -> None:
        """Cleanup bridge resources."""
        try:
            # Cancel active executions
            for execution_id, execution_info in self.active_executions.items():
                if 'task' in execution_info and not execution_info['task'].done():
                    execution_info['task'].cancel()
            
            self.active_executions.clear()
            
            # Cleanup components
            if self.runner:
                await self.runner.cleanup()
            # Agent registry doesn't need async cleanup
                
            logger.info("WebSocket-Agent bridge cleanup completed")
            
        except Exception as e:
            logger.error("Error during WebSocket-Agent bridge cleanup", error=str(e))
    
    async def process_user_message(
        self,
        connection_id: str,
        message: Dict[str, Any],
        db_session: AsyncSession
    ) -> None:
        """
        Process user message through complete agent pipeline.
        
        Args:
            connection_id: WebSocket connection ID
            message: User message data
            db_session: Database session
        """
        if not self.agent_registry:
            await self.send_error(connection_id, "Agent system not initialized")
            return
        
        execution_id = f"exec_{connection_id}_{datetime.now().timestamp()}"
        
        try:
            # Get connection info
            connection_info = self.connection_manager.get_connection_info(connection_id)
            if not connection_info:
                logger.error("Connection info not found", connection_id=connection_id)
                return
            
            user_id = connection_info['user_id']
            conversation_id = connection_info.get('conversation_id', connection_id)
            
            # Create chat request
            chat_request = ChatRequest(
                message=message.get('text', ''),
                user_id=user_id,
                context_id=conversation_id,  # Use context_id instead of conversation_id
                source="websocket"
            )
            
            # Start async execution
            execution_task = asyncio.create_task(
                self._execute_agent_pipeline(
                    connection_id=connection_id,
                    chat_request=chat_request,
                    execution_id=execution_id,
                    db_session=db_session
                )
            )
            
            # Track execution
            self.active_executions[execution_id] = {
                'connection_id': connection_id,
                'task': execution_task,
                'start_time': datetime.now(timezone.utc),
                'user_id': user_id,
                'conversation_id': conversation_id
            }
            
            logger.info(
                "Started agent execution",
                execution_id=execution_id,
                connection_id=connection_id,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(
                "Failed to process user message",
                connection_id=connection_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
            await self.send_error(connection_id, f"Failed to process message: {str(e)}")
    
    async def _execute_agent_pipeline(
        self,
        connection_id: str,
        chat_request: ChatRequest,
        execution_id: str,
        db_session: AsyncSession
    ) -> None:
        """Execute the complete agent pipeline with streaming."""
        try:
            # Send execution started event
            await self.send_execution_event(connection_id, {
                'event': 'execution_started',
                'execution_id': execution_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            # Step 1: Agent Selection
            await self.send_execution_event(connection_id, {
                'event': 'agent_selection_started',
                'execution_id': execution_id
            })
            
            user_context = {
                'user_id': chat_request.user_id,
                'conversation_id': chat_request.context_id,  # Use context_id
                'message': chat_request.message,
                'metadata': {}
            }
            
            agent = await self.agent_registry.get_agent(
                domain=None,  # Let router decide
                user_context=user_context
            )
            
            await self.send_execution_event(connection_id, {
                'event': 'agent_selected',
                'execution_id': execution_id,
                'agent_domain': agent.get_domain(),
                'agent_capabilities': agent.get_capabilities()
            })
            
            # Step 2: Plan Creation
            await self.send_execution_event(connection_id, {
                'event': 'planning_started',
                'execution_id': execution_id
            })
            
            planning_result = await self.planner.create_plan(
                request=chat_request,
                agent=agent
            )
            
            if not planning_result.success or not planning_result.plan:
                raise RuntimeError(f"Planning failed: {planning_result.error_message}")
            
            execution_plan = planning_result.plan
            
            await self.send_execution_event(connection_id, {
                'event': 'plan_created',
                'execution_id': execution_id,
                'plan_id': str(execution_plan.id),
                'task_count': len(execution_plan.tasks),
                'estimated_cost': execution_plan.cost.model_dump() if execution_plan.cost else None
            })
            
            # Step 3: Graph Compilation
            await self.send_execution_event(connection_id, {
                'event': 'compilation_started', 
                'execution_id': execution_id
            })
            
            compilation_result = await self.compiler.compile(execution_plan, agent)
            
            if not compilation_result.success:
                raise RuntimeError(f"Graph compilation failed: {compilation_result.error_message}")
            
            await self.send_execution_event(connection_id, {
                'event': 'graph_compiled',
                'execution_id': execution_id,
                'node_count': compilation_result.nodes_created,
                'edge_count': compilation_result.edges_created
            })
            
            # Step 4: Graph Execution with Streaming
            await self.send_execution_event(connection_id, {
                'event': 'graph_execution_started',
                'execution_id': execution_id
            })
            
            # Execute graph with streaming updates
            execution_context = {
                'request_id': execution_plan.request_id,
                'user_id': chat_request.user_id,
                'conversation_id': chat_request.context_id,  # Use context_id
                'execution_id': execution_id,
                'connection_id': connection_id
            }
            
            async for execution_event in self._stream_graph_execution(
                connection_id=connection_id,
                graph=compilation_result.graph,
                context=execution_context
            ):
                await self.send_execution_event(connection_id, execution_event)
            
            # Execution completed
            await self.send_execution_event(connection_id, {
                'event': 'execution_completed',
                'execution_id': execution_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(
                "Agent pipeline execution failed",
                execution_id=execution_id,
                connection_id=connection_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
            
            await self.send_execution_event(connection_id, {
                'event': 'execution_failed',
                'execution_id': execution_id,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        finally:
            # Cleanup execution tracking
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _stream_graph_execution(
        self,
        connection_id: str,
        graph: Any,  # LangGraph StateGraph
        context: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream graph execution events."""
        try:
            # Execute graph with event streaming
            result = await self.runner.run_stategraph(
                graph=graph,
                context=context,
                timeout_seconds=300
            )
            
            if result.success:
                # Stream node execution events
                for node_id in result.final_state.get('executed_nodes', []):
                    yield {
                        'event': 'node_completed',
                        'execution_id': context['execution_id'],
                        'node_id': node_id,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                
                # Stream final result
                yield {
                    'event': 'graph_completed',
                    'execution_id': context['execution_id'],
                    'final_result': result.final_state,
                    'metrics': {
                        'execution_time_ms': result.metrics.total_execution_time_ms,
                        'nodes_executed': result.metrics.nodes_executed,
                        'graph_hops': result.metrics.graph_hops
                    },
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            else:
                yield {
                    'event': 'graph_failed',
                    'execution_id': context['execution_id'],
                    'error': result.error_message,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            yield {
                'event': 'graph_execution_error',
                'execution_id': context['execution_id'],
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def send_execution_event(
        self,
        connection_id: str,
        event_data: Dict[str, Any]
    ) -> None:
        """Send execution event to WebSocket client."""
        try:
            await self.connection_manager.send_personal_message(
                connection_id=connection_id,
                message=event_data
            )
        except Exception as e:
            logger.error(
                "Failed to send execution event",
                connection_id=connection_id,
                event=event_data.get('event', 'unknown'),
                error=str(e)
            )
    
    async def send_error(
        self,
        connection_id: str,
        error_message: str
    ) -> None:
        """Send error message to WebSocket client."""
        error_event = {
            'event': 'error',
            'message': error_message,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self.send_execution_event(connection_id, error_event)
    
    def get_active_executions(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active executions."""
        return {
            execution_id: {
                'connection_id': info['connection_id'],
                'user_id': info['user_id'],
                'conversation_id': info['conversation_id'],
                'start_time': info['start_time'].isoformat(),
                'duration_seconds': (datetime.now(timezone.utc) - info['start_time']).total_seconds(),
                'status': 'running' if not info['task'].done() else 'completed'
            }
            for execution_id, info in self.active_executions.items()
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        if execution_id not in self.active_executions:
            return False
        
        execution_info = self.active_executions[execution_id]
        if not execution_info['task'].done():
            execution_info['task'].cancel()
            
            # Send cancellation event
            await self.send_execution_event(execution_info['connection_id'], {
                'event': 'execution_cancelled',
                'execution_id': execution_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
        return True


# Global bridge instance
websocket_agent_bridge = WebSocketAgentBridge(None)  # Will be initialized in startup