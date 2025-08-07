from typing import Dict, Any, Optional
import json
import logging
from datetime import datetime
import uuid
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db_session, ConversationEvent
from ..database.models import ConversationEventType, AgentType
from .connection_manager import connection_manager

logger = logging.getLogger(__name__)


class WebSocketHandler:
    """
    Handles WebSocket message processing and conversation event creation.
    """
    
    def __init__(self):
        pass
    
    async def handle_message(
        self,
        connection_id: str,
        message: Dict[str, Any],
        db_session: AsyncSession
    ):
        """
        Process incoming WebSocket message and create conversation event.
        
        Args:
            connection_id: WebSocket connection ID
            message: Parsed message from client
            db_session: Database session
        """
        try:
            # Get connection info
            connection_info = connection_manager.get_connection_info(connection_id)
            if not connection_info:
                logger.error(f"Connection info not found for {connection_id}")
                return
            
            user_id = connection_info["user_id"]
            conversation_id = connection_info.get("conversation_id")
            
            # Extract message data
            message_type = message.get("type", "user_message")
            message_text = message.get("text", "")
            metadata = message.get("metadata", {})
            agent_type = message.get("agent_type")
            
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                # Update connection metadata
                connection_manager.connection_metadata[connection_id]["conversation_id"] = conversation_id
            
            # Determine event type
            if message_type == "user_message":
                event_type = ConversationEventType.USER_MESSAGE
                agent_type_enum = None
            else:
                event_type = ConversationEventType.SYSTEM_EVENT
                agent_type_enum = None
                if agent_type:
                    try:
                        agent_type_enum = AgentType(agent_type)
                    except ValueError:
                        logger.warning(f"Invalid agent type: {agent_type}")
            
            # Create conversation event
            conversation_event = ConversationEvent(
                conversation_id=conversation_id,
                user_id=user_id,
                agent_type=agent_type_enum,
                event_type=event_type,
                message_text=message_text,
                event_metadata={
                    **metadata,
                    "connection_id": connection_id,
                    "client_timestamp": message.get("timestamp"),
                    "message_id": message.get("message_id", str(uuid.uuid4()))
                }
            )
            
            # Save to database
            db_session.add(conversation_event)
            await db_session.commit()
            await db_session.refresh(conversation_event)
            
            logger.info(f"Created conversation event {conversation_event.id} for user {user_id}")
            
            # Send acknowledgment to client
            ack_message = {
                "type": "message_received",
                "event_id": conversation_event.event_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "processed"
            }
            
            await connection_manager.send_personal_message(connection_id, ack_message)
            
            # Process message based on type
            if event_type == ConversationEventType.USER_MESSAGE:
                await self.process_user_message(
                    conversation_event,
                    connection_id,
                    db_session
                )
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.send_error_message(connection_id, str(e))
    
    async def process_user_message(
        self,
        conversation_event: ConversationEvent,
        connection_id: str,
        db_session: AsyncSession
    ):
        """
        Process a user message and determine agent routing.
        
        Args:
            conversation_event: The conversation event record
            connection_id: WebSocket connection ID
            db_session: Database session
        """
        # For now, send a simple response
        # In Phase 2, this will route to the appropriate agent
        
        response_message = {
            "type": "agent_thinking",
            "conversation_id": conversation_event.conversation_id,
            "message": "Processing your request...",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await connection_manager.send_personal_message(connection_id, response_message)
        
        # TODO: Route to agent planner and graph compiler
        # This will be implemented in Phase 2 with the agent framework
        
        # Create a mock response event for now
        response_event = ConversationEvent(
            conversation_id=conversation_event.conversation_id,
            user_id=conversation_event.user_id,
            agent_type=AgentType.FINOPS,  # Default to FinOps for now
            event_type=ConversationEventType.AGENT_RESPONSE,
            message_text="I understand you need help with FinOps. Let me analyze your request...",
            event_metadata={
                "original_event_id": conversation_event.event_id,
                "response_type": "initial_acknowledgment"
            }
        )
        
        db_session.add(response_event)
        await db_session.commit()
        await db_session.refresh(response_event)
        
        # Send response to client
        response = {
            "type": "agent_response",
            "event_id": response_event.event_id,
            "conversation_id": response_event.conversation_id,
            "agent_type": response_event.agent_type.value,
            "message": response_event.message_text,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await connection_manager.send_personal_message(connection_id, response)
    
    async def send_error_message(self, connection_id: str, error_message: str):
        """
        Send an error message to the client.
        
        Args:
            connection_id: WebSocket connection ID
            error_message: Error message to send
        """
        error_response = {
            "type": "error",
            "message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await connection_manager.send_personal_message(connection_id, error_response)
    
    async def handle_ping(self, connection_id: str):
        """
        Handle ping message and send pong response.
        
        Args:
            connection_id: WebSocket connection ID
        """
        pong_message = {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await connection_manager.send_personal_message(connection_id, pong_message)
    
    async def handle_disconnect(self, connection_id: str):
        """
        Handle WebSocket disconnection cleanup.
        
        Args:
            connection_id: WebSocket connection ID
        """
        await connection_manager.disconnect(connection_id)


# Global message handler instance
websocket_handler = WebSocketHandler()