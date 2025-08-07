from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
import json
import logging
from typing import Optional

from ..auth.jwt_auth import verify_token
from ..database import get_db_session
from .connection_manager import connection_manager
from .handler import websocket_handler

logger = logging.getLogger(__name__)

# Create WebSocket router
websocket_router = APIRouter(prefix="/api/v1/ws", tags=["websocket"])


async def authenticate_websocket(token: str) -> Optional[str]:
    """
    Authenticate WebSocket connection using JWT token.
    
    Args:
        token: JWT token from query parameters
        
    Returns:
        str: User ID if authentication successful, None otherwise
    """
    if not token:
        return None
    
    token_data = verify_token(token)
    if not token_data:
        return None
    
    return token_data.user_id


@websocket_router.websocket("/chat")
async def websocket_chat_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT authentication token"),
    conversation_id: Optional[str] = Query(None, description="Optional conversation ID")
):
    """
    WebSocket endpoint for real-time chat with FinOps agents.
    
    Authentication is performed using JWT token passed as query parameter.
    Messages are processed and routed to appropriate agents based on content analysis.
    
    Query Parameters:
        token: JWT authentication token (required)
        conversation_id: Optional conversation session ID for context
    
    Message Format (Client -> Server):
    {
        "type": "user_message",
        "text": "Your message here", 
        "metadata": {"key": "value"},
        "timestamp": "ISO timestamp",
        "message_id": "unique_id"
    }
    
    Response Format (Server -> Client):
    {
        "type": "agent_response|system_event|error",
        "event_id": "unique_event_id",
        "conversation_id": "conversation_id", 
        "message": "Response text",
        "agent_type": "finops|github|document|research",
        "timestamp": "ISO timestamp"
    }
    """
    # Authenticate user
    user_id = await authenticate_websocket(token)
    if not user_id:
        await websocket.close(code=4001, reason="Authentication failed")
        return
    
    # Establish connection
    connection_id = await connection_manager.connect(
        websocket=websocket,
        user_id=user_id,
        conversation_id=conversation_id
    )
    
    logger.info(f"WebSocket connection established: {connection_id}")
    
    try:
        # Message processing loop
        while True:
            # Wait for message from client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON message
                message = json.loads(data)
                
                # Handle different message types
                message_type = message.get("type")
                
                if message_type == "ping":
                    # Handle ping/pong for connection health
                    await websocket_handler.handle_ping(connection_id)
                    continue
                
                # Get database session for message processing
                async for db_session in get_db_session():
                    await websocket_handler.handle_message(
                        connection_id=connection_id,
                        message=message,
                        db_session=db_session
                    )
                    break  # Exit the async generator
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from {connection_id}: {e}")
                await websocket_handler.send_error_message(
                    connection_id, 
                    "Invalid JSON format"
                )
            except Exception as e:
                logger.error(f"Error processing message from {connection_id}: {e}")
                await websocket_handler.send_error_message(
                    connection_id,
                    "Error processing message"
                )
                
    except WebSocketDisconnect:
        logger.info(f"Client {connection_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
    finally:
        # Clean up connection
        await websocket_handler.handle_disconnect(connection_id)


@websocket_router.get("/stats")
async def get_websocket_stats():
    """
    Get WebSocket connection statistics.
    
    Returns connection counts and active user information.
    """
    return connection_manager.get_stats()