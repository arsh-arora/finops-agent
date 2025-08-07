"""
Integration tests for complete WebSocket flow.
"""

import pytest
import asyncio
import json
import uuid
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timezone

from src.websocket.connection_manager import ConnectionManager
from src.websocket.handler import WebSocketHandler
from src.auth.jwt_auth import create_access_token, verify_token


@pytest.mark.integration
@pytest.mark.websocket
class TestWebSocketFlow:
    """Test complete WebSocket flow integration."""
    
    @pytest.fixture
    async def connection_manager(self):
        """Create connection manager for testing."""
        manager = ConnectionManager()
        yield manager
        
        # Cleanup after test
        manager.active_connections.clear()
        manager.user_connections.clear()
        manager.connection_metadata.clear()
        manager.conversation_sessions.clear()
    
    @pytest.fixture
    def handler(self):
        """Create WebSocket handler for testing."""
        return WebSocketHandler()
    
    @pytest.fixture
    def auth_token(self):
        """Create authentication token."""
        return create_access_token({
            "sub": "integration_test_user",
            "username": "integration_user"
        })
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        websocket = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.close = AsyncMock()
        return websocket
    
    async def test_complete_connection_flow(
        self,
        connection_manager,
        auth_token,
        mock_websocket
    ):
        """Test complete connection establishment flow."""
        # Verify token first
        token_data = verify_token(auth_token)
        assert token_data is not None
        user_id = token_data.user_id
        
        # Connect user
        conversation_id = str(uuid.uuid4())
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Verify connection was established
        assert connection_id in connection_manager.active_connections
        assert user_id in connection_manager.user_connections
        assert connection_id in connection_manager.user_connections[user_id]
        
        # Verify WebSocket interactions
        mock_websocket.accept.assert_called_once()
        mock_websocket.send_text.assert_called_once()
        
        # Verify welcome message was sent
        welcome_call_args = mock_websocket.send_text.call_args
        welcome_message_json = welcome_call_args[0][0]
        welcome_message = json.loads(welcome_message_json)
        
        assert welcome_message["type"] == "connection_established"
        assert welcome_message["connection_id"] == connection_id
        
        # Test disconnect
        await connection_manager.disconnect(connection_id)
        
        # Verify cleanup
        assert connection_id not in connection_manager.active_connections
        assert user_id not in connection_manager.user_connections
    
    async def test_message_handling_flow(
        self,
        connection_manager,
        handler,
        db_session,
        auth_token,
        mock_websocket
    ):
        """Test complete message handling flow."""
        # Setup connection
        token_data = verify_token(auth_token)
        user_id = token_data.user_id
        conversation_id = str(uuid.uuid4())
        
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Create test message
        test_message = {
            "type": "user_message",
            "text": "Help me optimize my AWS costs",
            "metadata": {"source": "integration_test"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": str(uuid.uuid4())
        }
        
        # Handle message
        await handler.handle_message(
            connection_id=connection_id,
            message=test_message,
            db_session=db_session
        )
        
        # Verify database event was created
        from sqlalchemy import select
        from src.database.models import ConversationEvent, ConversationEventType
        
        result = await db_session.execute(
            select(ConversationEvent).where(
                ConversationEvent.user_id == user_id
            )
        )
        events = result.scalars().all()
        
        # Should have user message and agent response
        assert len(events) >= 1
        
        user_event = next(
            (e for e in events if e.event_type == ConversationEventType.USER_MESSAGE),
            None
        )
        assert user_event is not None
        assert user_event.message_text == test_message["text"]
        assert user_event.conversation_id == conversation_id
        
        # Verify acknowledgment and response messages were sent
        assert mock_websocket.send_text.call_count >= 2  # Welcome + ACK + thinking + response
        
        # Check the sent messages
        sent_calls = mock_websocket.send_text.call_args_list
        sent_messages = [json.loads(call[0][0]) for call in sent_calls[1:]]  # Skip welcome
        
        # Find acknowledgment message
        ack_message = next(
            (msg for msg in sent_messages if msg.get("type") == "message_received"),
            None
        )
        assert ack_message is not None
        assert ack_message["status"] == "processed"
        
        # Find thinking message
        thinking_message = next(
            (msg for msg in sent_messages if msg.get("type") == "agent_thinking"),
            None
        )
        assert thinking_message is not None
        
        # Find response message
        response_message = next(
            (msg for msg in sent_messages if msg.get("type") == "agent_response"),
            None
        )
        assert response_message is not None
        assert response_message["agent_type"] == "finops"
    
    async def test_ping_pong_flow(
        self,
        connection_manager,
        handler,
        auth_token,
        mock_websocket
    ):
        """Test ping/pong message flow."""
        # Setup connection
        token_data = verify_token(auth_token)
        user_id = token_data.user_id
        
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            user_id=user_id
        )
        
        # Reset mock to ignore welcome message
        mock_websocket.reset_mock()
        
        # Handle ping
        await handler.handle_ping(connection_id)
        
        # Verify pong response
        mock_websocket.send_text.assert_called_once()
        pong_call_args = mock_websocket.send_text.call_args
        pong_message = json.loads(pong_call_args[0][0])
        
        assert pong_message["type"] == "pong"
        assert "timestamp" in pong_message
    
    async def test_error_handling_flow(
        self,
        connection_manager,
        handler,
        auth_token,
        mock_websocket
    ):
        """Test error handling in WebSocket flow."""
        # Setup connection
        token_data = verify_token(auth_token)
        user_id = token_data.user_id
        
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            user_id=user_id
        )
        
        # Reset mock to ignore welcome message
        mock_websocket.reset_mock()
        
        # Send error message
        error_text = "Test error occurred"
        await handler.send_error_message(connection_id, error_text)
        
        # Verify error message was sent
        mock_websocket.send_text.assert_called_once()
        error_call_args = mock_websocket.send_text.call_args
        error_message = json.loads(error_call_args[0][0])
        
        assert error_message["type"] == "error"
        assert error_message["message"] == error_text
        assert "timestamp" in error_message
    
    async def test_multi_user_conversation_flow(
        self,
        connection_manager,
        auth_token,
        mock_websocket
    ):
        """Test multi-user conversation flow."""
        # Create tokens for multiple users
        user1_token = create_access_token({
            "sub": "user1",
            "username": "user_one"
        })
        user2_token = create_access_token({
            "sub": "user2", 
            "username": "user_two"
        })
        
        # Setup mock WebSockets
        websocket1 = AsyncMock()
        websocket1.accept = AsyncMock()
        websocket1.send_text = AsyncMock()
        
        websocket2 = AsyncMock()
        websocket2.accept = AsyncMock()
        websocket2.send_text = AsyncMock()
        
        # Connect both users to same conversation
        conversation_id = str(uuid.uuid4())
        
        user1_data = verify_token(user1_token)
        user2_data = verify_token(user2_token)
        
        conn1_id = await connection_manager.connect(
            websocket=websocket1,
            user_id=user1_data.user_id,
            conversation_id=conversation_id
        )
        
        conn2_id = await connection_manager.connect(
            websocket=websocket2,
            user_id=user2_data.user_id,
            conversation_id=conversation_id
        )
        
        # Verify both connections are tracked
        assert len(connection_manager.active_connections) == 2
        assert len(connection_manager.user_connections) == 2
        assert len(connection_manager.conversation_sessions[conversation_id]) == 2
        
        # Send message to conversation
        conversation_message = {
            "type": "conversation_update",
            "message": "New user joined",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await connection_manager.send_to_conversation(
            conversation_id,
            conversation_message
        )
        
        # Verify both users received the message
        websocket1.send_text.assert_called()
        websocket2.send_text.assert_called()
        
        # Verify message content
        sent_message1 = json.loads(websocket1.send_text.call_args_list[-1][0][0])
        sent_message2 = json.loads(websocket2.send_text.call_args_list[-1][0][0])
        
        assert sent_message1["type"] == "conversation_update"
        assert sent_message2["type"] == "conversation_update"
        assert sent_message1["message"] == "New user joined"
        assert sent_message2["message"] == "New user joined"
    
    async def test_connection_cleanup_flow(
        self,
        connection_manager,
        auth_token
    ):
        """Test connection cleanup flow when WebSocket fails."""
        token_data = verify_token(auth_token)
        user_id = token_data.user_id
        
        # Create WebSocket that will fail on send
        failing_websocket = AsyncMock()
        failing_websocket.accept = AsyncMock()
        failing_websocket.send_text = AsyncMock(side_effect=Exception("WebSocket error"))
        
        # Connect user
        connection_id = await connection_manager.connect(
            websocket=failing_websocket,
            user_id=user_id
        )
        
        # Verify connection was established
        assert connection_id in connection_manager.active_connections
        
        # Try to send message (will fail and trigger cleanup)
        test_message = {"type": "test", "content": "test"}
        await connection_manager.send_personal_message(connection_id, test_message)
        
        # Verify connection was cleaned up due to error
        assert connection_id not in connection_manager.active_connections
        assert user_id not in connection_manager.user_connections
    
    async def test_broadcast_flow(
        self,
        connection_manager,
        auth_token
    ):
        """Test broadcast message flow."""
        # Create multiple connections
        websockets = []
        connection_ids = []
        
        for i in range(3):
            websocket = AsyncMock()
            websocket.accept = AsyncMock()
            websocket.send_text = AsyncMock()
            websockets.append(websocket)
            
            token = create_access_token({
                "sub": f"user_{i}",
                "username": f"user_{i}"
            })
            
            token_data = verify_token(token)
            conn_id = await connection_manager.connect(
                websocket=websocket,
                user_id=token_data.user_id
            )
            connection_ids.append(conn_id)
        
        # Broadcast message
        broadcast_message = {
            "type": "system_announcement",
            "message": "Server maintenance in 5 minutes",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await connection_manager.broadcast(broadcast_message)
        
        # Verify all connections received the message
        for websocket in websockets:
            websocket.send_text.assert_called()
            
            # Check message content
            sent_calls = websocket.send_text.call_args_list
            broadcast_call = sent_calls[-1]  # Last call should be broadcast
            sent_message = json.loads(broadcast_call[0][0])
            
            assert sent_message["type"] == "system_announcement"
            assert sent_message["message"] == "Server maintenance in 5 minutes"
    
    async def test_stats_tracking_flow(
        self,
        connection_manager,
        auth_token
    ):
        """Test statistics tracking through connection flow."""
        # Initially empty
        stats = connection_manager.get_stats()
        assert stats["total_connections"] == 0
        assert stats["unique_users"] == 0
        
        # Add connections
        websocket1 = AsyncMock()
        websocket1.accept = AsyncMock()
        websocket1.send_text = AsyncMock()
        
        websocket2 = AsyncMock()
        websocket2.accept = AsyncMock()
        websocket2.send_text = AsyncMock()
        
        # Same user, multiple connections
        token_data = verify_token(auth_token)
        user_id = token_data.user_id
        
        conn1 = await connection_manager.connect(websocket1, user_id)
        conn2 = await connection_manager.connect(websocket2, user_id)
        
        # Check stats after connections
        stats = connection_manager.get_stats()
        assert stats["total_connections"] == 2
        assert stats["unique_users"] == 1
        assert stats["connections_per_user"][user_id] == 2
        
        # Disconnect one
        await connection_manager.disconnect(conn1)
        
        # Check stats after disconnect
        stats = connection_manager.get_stats()
        assert stats["total_connections"] == 1
        assert stats["unique_users"] == 1
        assert stats["connections_per_user"][user_id] == 1