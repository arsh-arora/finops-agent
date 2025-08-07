"""
Tests for WebSocket message handler.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.websocket.handler import WebSocketHandler
from src.database.models import ConversationEvent, ConversationEventType, AgentType


@pytest.mark.unit
@pytest.mark.websocket
class TestWebSocketHandler:
    """Test WebSocket message handler functionality."""
    
    @pytest.fixture
    def handler(self):
        """Create WebSocket handler instance."""
        return WebSocketHandler()
    
    @pytest.fixture
    def mock_connection_info(self):
        """Mock connection info."""
        return {
            "user_id": "test_user_123",
            "conversation_id": str(uuid.uuid4()),
            "connected_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
    
    async def test_handle_user_message_success(
        self, 
        handler, 
        db_session, 
        mock_connection_info,
        sample_websocket_message
    ):
        """Test handling user message successfully."""
        connection_id = "test_connection_123"
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.get_connection_info.return_value = mock_connection_info
            mock_manager.send_personal_message = AsyncMock()
            
            await handler.handle_message(
                connection_id=connection_id,
                message=sample_websocket_message,
                db_session=db_session
            )
            
            # Verify conversation event was created
            from sqlalchemy import select
            result = await db_session.execute(
                select(ConversationEvent).where(
                    ConversationEvent.user_id == mock_connection_info["user_id"]
                )
            )
            events = result.scalars().all()
            
            assert len(events) >= 1
            event = events[0]
            assert event.user_id == mock_connection_info["user_id"]
            assert event.message_text == sample_websocket_message["text"]
            assert event.event_type == ConversationEventType.USER_MESSAGE
            
            # Verify acknowledgment was sent
            mock_manager.send_personal_message.assert_called()
    
    async def test_handle_message_no_connection_info(
        self, 
        handler, 
        db_session, 
        sample_websocket_message
    ):
        """Test handling message when connection info is not found."""
        connection_id = "nonexistent_connection"
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.get_connection_info.return_value = None
            
            # Should handle gracefully without raising exception
            await handler.handle_message(
                connection_id=connection_id,
                message=sample_websocket_message,
                db_session=db_session
            )
            
            # No events should be created
            from sqlalchemy import select
            result = await db_session.execute(select(ConversationEvent))
            events = result.scalars().all()
            assert len(events) == 0
    
    async def test_handle_message_generates_conversation_id(
        self,
        handler,
        db_session,
        sample_websocket_message
    ):
        """Test message handling generates conversation ID when missing."""
        connection_id = "test_connection_456"
        connection_info = {
            "user_id": "test_user_456",
            "conversation_id": None  # No conversation ID
        }
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.get_connection_info.return_value = connection_info
            mock_manager.send_personal_message = AsyncMock()
            mock_manager.connection_metadata = {connection_id: connection_info}
            
            await handler.handle_message(
                connection_id=connection_id,
                message=sample_websocket_message,
                db_session=db_session
            )
            
            # Verify conversation ID was generated and stored
            updated_info = mock_manager.connection_metadata[connection_id]
            assert updated_info["conversation_id"] is not None
            assert isinstance(updated_info["conversation_id"], str)
    
    async def test_handle_system_event_message(
        self,
        handler,
        db_session,
        mock_connection_info
    ):
        """Test handling system event message."""
        connection_id = "test_connection_system"
        system_message = {
            "type": "system_event",
            "text": "User joined the conversation",
            "metadata": {"event": "user_join"}
        }
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.get_connection_info.return_value = mock_connection_info
            mock_manager.send_personal_message = AsyncMock()
            
            await handler.handle_message(
                connection_id=connection_id,
                message=system_message,
                db_session=db_session
            )
            
            # Verify system event was created
            from sqlalchemy import select
            result = await db_session.execute(
                select(ConversationEvent).where(
                    ConversationEvent.event_type == ConversationEventType.SYSTEM_EVENT
                )
            )
            event = result.scalar_one()
            
            assert event.event_type == ConversationEventType.SYSTEM_EVENT
            assert event.message_text == system_message["text"]
    
    async def test_handle_message_with_agent_type(
        self,
        handler,
        db_session,
        mock_connection_info
    ):
        """Test handling message with agent type specified."""
        connection_id = "test_connection_agent"
        agent_message = {
            "type": "system_event",
            "text": "Agent response",
            "agent_type": "finops",
            "metadata": {"source": "agent"}
        }
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.get_connection_info.return_value = mock_connection_info
            mock_manager.send_personal_message = AsyncMock()
            
            await handler.handle_message(
                connection_id=connection_id,
                message=agent_message,
                db_session=db_session
            )
            
            # Verify agent type was not set for system events (should be None)
            from sqlalchemy import select
            result = await db_session.execute(
                select(ConversationEvent).where(
                    ConversationEvent.message_text == "Agent response"
                )
            )
            event = result.scalar_one()
            
            assert event.agent_type is None  # Agent type only for agent responses
    
    async def test_process_user_message(
        self,
        handler,
        db_session,
        create_conversation_event
    ):
        """Test processing user message and generating response."""
        # Create a test conversation event
        conversation_event = await create_conversation_event(
            db_session,
            user_id="test_user",
            message_text="Help me with AWS costs",
            event_type=ConversationEventType.USER_MESSAGE
        )
        
        connection_id = "test_connection_process"
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.send_personal_message = AsyncMock()
            
            await handler.process_user_message(
                conversation_event=conversation_event,
                connection_id=connection_id,
                db_session=db_session
            )
            
            # Verify thinking message was sent
            thinking_call = mock_manager.send_personal_message.call_args_list[0]
            thinking_message = thinking_call[0][1]
            assert thinking_message["type"] == "agent_thinking"
            
            # Verify response message was sent
            response_call = mock_manager.send_personal_message.call_args_list[1]
            response_message = response_call[0][1]
            assert response_message["type"] == "agent_response"
            assert response_message["agent_type"] == "finops"
            
            # Verify response event was created in database
            from sqlalchemy import select
            result = await db_session.execute(
                select(ConversationEvent).where(
                    ConversationEvent.event_type == ConversationEventType.AGENT_RESPONSE
                )
            )
            response_event = result.scalar_one()
            
            assert response_event.agent_type == AgentType.FINOPS
            assert response_event.conversation_id == conversation_event.conversation_id
            assert "original_event_id" in response_event.event_metadata
    
    async def test_send_error_message(self, handler):
        """Test sending error message."""
        connection_id = "test_connection_error"
        error_message = "Something went wrong"
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.send_personal_message = AsyncMock()
            
            await handler.send_error_message(connection_id, error_message)
            
            # Verify error message was sent
            mock_manager.send_personal_message.assert_called_once()
            call_args = mock_manager.send_personal_message.call_args
            sent_message = call_args[0][1]
            
            assert sent_message["type"] == "error"
            assert sent_message["message"] == error_message
            assert "timestamp" in sent_message
    
    async def test_handle_ping(self, handler):
        """Test handling ping message."""
        connection_id = "test_connection_ping"
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.send_personal_message = AsyncMock()
            
            await handler.handle_ping(connection_id)
            
            # Verify pong response was sent
            mock_manager.send_personal_message.assert_called_once()
            call_args = mock_manager.send_personal_message.call_args
            sent_message = call_args[0][1]
            
            assert sent_message["type"] == "pong"
            assert "timestamp" in sent_message
    
    async def test_handle_disconnect(self, handler):
        """Test handling disconnect event."""
        connection_id = "test_connection_disconnect"
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.disconnect = AsyncMock()
            
            await handler.handle_disconnect(connection_id)
            
            # Verify connection manager disconnect was called
            mock_manager.disconnect.assert_called_once_with(connection_id)
    
    async def test_handle_message_database_error(
        self,
        handler,
        mock_connection_info,
        sample_websocket_message
    ):
        """Test handling message when database error occurs."""
        connection_id = "test_connection_db_error"
        
        # Mock database session that raises exception
        mock_db_session = AsyncMock()
        mock_db_session.add.side_effect = Exception("Database error")
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.get_connection_info.return_value = mock_connection_info
            mock_manager.send_personal_message = AsyncMock()
            
            # Should handle exception gracefully
            await handler.handle_message(
                connection_id=connection_id,
                message=sample_websocket_message,
                db_session=mock_db_session
            )
            
            # Error message should be sent to client
            mock_manager.send_personal_message.assert_called()
    
    async def test_handle_message_metadata_preservation(
        self,
        handler,
        db_session,
        mock_connection_info
    ):
        """Test that message metadata is preserved correctly."""
        connection_id = "test_connection_metadata"
        message_with_metadata = {
            "type": "user_message",
            "text": "Test message",
            "metadata": {
                "source": "mobile_app",
                "version": "1.2.3",
                "custom_field": "custom_value"
            },
            "timestamp": "2024-01-01T10:00:00Z",
            "message_id": "msg_123"
        }
        
        with patch('src.websocket.handler.connection_manager') as mock_manager:
            mock_manager.get_connection_info.return_value = mock_connection_info
            mock_manager.send_personal_message = AsyncMock()
            
            await handler.handle_message(
                connection_id=connection_id,
                message=message_with_metadata,
                db_session=db_session
            )
            
            # Verify metadata was stored correctly
            from sqlalchemy import select
            result = await db_session.execute(
                select(ConversationEvent).where(
                    ConversationEvent.user_id == mock_connection_info["user_id"]
                )
            )
            event = result.scalar_one()
            
            # Check original metadata is preserved
            assert event.event_metadata["source"] == "mobile_app"
            assert event.event_metadata["version"] == "1.2.3"
            assert event.event_metadata["custom_field"] == "custom_value"
            
            # Check additional metadata was added
            assert event.event_metadata["connection_id"] == connection_id
            assert event.event_metadata["client_timestamp"] == "2024-01-01T10:00:00Z"
            assert event.event_metadata["message_id"] == "msg_123"