"""
Tests for WebSocket connection manager.
"""

import pytest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.websocket.connection_manager import ConnectionManager


@pytest.mark.unit
@pytest.mark.websocket
class TestConnectionManager:
    """Test WebSocket connection manager functionality."""
    
    def test_connection_manager_initialization(self):
        """Test ConnectionManager initialization."""
        manager = ConnectionManager()
        
        assert manager.active_connections == {}
        assert manager.user_connections == {}
        assert manager.connection_metadata == {}
        assert manager.conversation_sessions == {}
    
    async def test_connect_new_user(self, connection_manager, mock_websocket):
        """Test connecting a new user."""
        user_id = "test_user_123"
        conversation_id = str(uuid.uuid4())
        
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = "connection_id_123"
            mock_uuid.return_value.__str__ = lambda self: "connection_id_123"
            
            connection_id = await connection_manager.connect(
                websocket=mock_websocket,
                user_id=user_id,
                conversation_id=conversation_id
            )
        
        # Verify connection was established
        mock_websocket.accept.assert_called_once()
        mock_websocket.send_text.assert_called_once()
        
        # Verify connection tracking
        assert connection_id in connection_manager.active_connections
        assert connection_manager.active_connections[connection_id] == mock_websocket
        
        # Verify user tracking
        assert user_id in connection_manager.user_connections
        assert connection_id in connection_manager.user_connections[user_id]
        
        # Verify metadata tracking
        assert connection_id in connection_manager.connection_metadata
        metadata = connection_manager.connection_metadata[connection_id]
        assert metadata["user_id"] == user_id
        assert metadata["conversation_id"] == conversation_id
        
        # Verify conversation session tracking
        assert conversation_id in connection_manager.conversation_sessions
        assert connection_id in connection_manager.conversation_sessions[conversation_id]
    
    async def test_connect_without_conversation(self, connection_manager, mock_websocket):
        """Test connecting without conversation ID."""
        user_id = "test_user_456"
        
        connection_id = await connection_manager.connect(
            websocket=mock_websocket,
            user_id=user_id
        )
        
        # Verify connection was established
        assert connection_id in connection_manager.active_connections
        
        # Verify metadata without conversation
        metadata = connection_manager.connection_metadata[connection_id]
        assert metadata["user_id"] == user_id
        assert metadata["conversation_id"] is None
    
    async def test_connect_multiple_users(self, connection_manager):
        """Test connecting multiple users."""
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        user1 = "user_1"
        user2 = "user_2"
        
        connection_id1 = await connection_manager.connect(websocket1, user1)
        connection_id2 = await connection_manager.connect(websocket2, user2)
        
        # Verify both connections are tracked
        assert len(connection_manager.active_connections) == 2
        assert connection_id1 in connection_manager.active_connections
        assert connection_id2 in connection_manager.active_connections
        
        # Verify user separation
        assert len(connection_manager.user_connections) == 2
        assert user1 in connection_manager.user_connections
        assert user2 in connection_manager.user_connections
    
    async def test_disconnect_existing_connection(self, connection_manager, mock_websocket):
        """Test disconnecting existing connection."""
        user_id = "test_user_disconnect"
        conversation_id = str(uuid.uuid4())
        
        # First connect
        connection_id = await connection_manager.connect(
            mock_websocket, user_id, conversation_id
        )
        
        # Verify connection exists
        assert connection_id in connection_manager.active_connections
        
        # Disconnect
        await connection_manager.disconnect(connection_id)
        
        # Verify cleanup
        assert connection_id not in connection_manager.active_connections
        assert user_id not in connection_manager.user_connections
        assert connection_id not in connection_manager.connection_metadata
        assert conversation_id not in connection_manager.conversation_sessions
    
    async def test_disconnect_nonexistent_connection(self, connection_manager):
        """Test disconnecting non-existent connection."""
        # Should not raise exception
        await connection_manager.disconnect("nonexistent_connection")
        
        # Manager should remain empty
        assert len(connection_manager.active_connections) == 0
    
    async def test_disconnect_partial_cleanup(self, connection_manager):
        """Test disconnect with partial cleanup (user has multiple connections)."""
        user_id = "multi_connection_user"
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        
        # Connect same user twice
        conn1 = await connection_manager.connect(websocket1, user_id)
        conn2 = await connection_manager.connect(websocket2, user_id)
        
        # Verify user has multiple connections
        assert len(connection_manager.user_connections[user_id]) == 2
        
        # Disconnect one connection
        await connection_manager.disconnect(conn1)
        
        # Verify partial cleanup
        assert conn1 not in connection_manager.active_connections
        assert conn2 in connection_manager.active_connections
        assert user_id in connection_manager.user_connections
        assert len(connection_manager.user_connections[user_id]) == 1
    
    async def test_send_personal_message_success(self, connection_manager, mock_websocket):
        """Test sending personal message successfully."""
        user_id = "test_user_message"
        connection_id = await connection_manager.connect(mock_websocket, user_id)
        
        message = {"type": "test", "content": "Hello World"}
        
        await connection_manager.send_personal_message(connection_id, message)
        
        # Verify message was sent
        mock_websocket.send_text.assert_called()
        sent_message = mock_websocket.send_text.call_args[0][0]
        assert '"type": "test"' in sent_message
        assert '"content": "Hello World"' in sent_message
    
    async def test_send_personal_message_nonexistent_connection(self, connection_manager):
        """Test sending message to non-existent connection."""
        message = {"type": "test", "content": "Hello"}
        
        # Should not raise exception
        await connection_manager.send_personal_message("nonexistent", message)
    
    async def test_send_personal_message_websocket_error(self, connection_manager):
        """Test handling WebSocket send error."""
        mock_websocket = AsyncMock()
        mock_websocket.send_text.side_effect = Exception("Send failed")
        
        user_id = "error_user"
        connection_id = await connection_manager.connect(mock_websocket, user_id)
        
        message = {"type": "test", "content": "Hello"}
        
        # Should handle exception and disconnect
        await connection_manager.send_personal_message(connection_id, message)
        
        # Connection should be cleaned up after error
        assert connection_id not in connection_manager.active_connections
    
    async def test_send_to_user_single_connection(self, connection_manager, mock_websocket):
        """Test sending message to user with single connection."""
        user_id = "single_user"
        connection_id = await connection_manager.connect(mock_websocket, user_id)
        
        message = {"type": "broadcast", "content": "User message"}
        
        await connection_manager.send_to_user(user_id, message)
        
        # Verify message was sent to user's connection
        mock_websocket.send_text.assert_called()
    
    async def test_send_to_user_multiple_connections(self, connection_manager):
        """Test sending message to user with multiple connections."""
        user_id = "multi_user"
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        
        await connection_manager.connect(websocket1, user_id)
        await connection_manager.connect(websocket2, user_id)
        
        message = {"type": "broadcast", "content": "Multi message"}
        
        await connection_manager.send_to_user(user_id, message)
        
        # Verify message was sent to both connections
        websocket1.send_text.assert_called()
        websocket2.send_text.assert_called()
    
    async def test_send_to_user_nonexistent(self, connection_manager):
        """Test sending message to non-existent user."""
        message = {"type": "test", "content": "Hello"}
        
        # Should not raise exception
        await connection_manager.send_to_user("nonexistent_user", message)
    
    async def test_send_to_conversation(self, connection_manager):
        """Test sending message to conversation."""
        conversation_id = str(uuid.uuid4())
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        
        # Connect multiple users to same conversation
        await connection_manager.connect(websocket1, "user1", conversation_id)
        await connection_manager.connect(websocket2, "user2", conversation_id)
        
        message = {"type": "conversation", "content": "Group message"}
        
        await connection_manager.send_to_conversation(conversation_id, message)
        
        # Verify message was sent to all participants
        websocket1.send_text.assert_called()
        websocket2.send_text.assert_called()
    
    async def test_broadcast_message(self, connection_manager):
        """Test broadcasting message to all connections."""
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        websocket3 = AsyncMock()
        
        # Connect multiple users
        await connection_manager.connect(websocket1, "user1")
        await connection_manager.connect(websocket2, "user2") 
        await connection_manager.connect(websocket3, "user3")
        
        message = {"type": "broadcast", "content": "System announcement"}
        
        await connection_manager.broadcast(message)
        
        # Verify message was sent to all connections
        websocket1.send_text.assert_called()
        websocket2.send_text.assert_called()
        websocket3.send_text.assert_called()
    
    async def test_get_connection_info(self, connection_manager, mock_websocket):
        """Test getting connection information."""
        user_id = "info_user"
        conversation_id = str(uuid.uuid4())
        
        connection_id = await connection_manager.connect(
            mock_websocket, user_id, conversation_id
        )
        
        info = connection_manager.get_connection_info(connection_id)
        
        assert info is not None
        assert info["user_id"] == user_id
        assert info["conversation_id"] == conversation_id
        assert "connected_at" in info
        assert "last_activity" in info
    
    def test_get_connection_info_nonexistent(self, connection_manager):
        """Test getting info for non-existent connection."""
        info = connection_manager.get_connection_info("nonexistent")
        
        assert info is None
    
    async def test_get_user_connections(self, connection_manager):
        """Test getting user connections."""
        user_id = "connection_user"
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        
        conn1 = await connection_manager.connect(websocket1, user_id)
        conn2 = await connection_manager.connect(websocket2, user_id)
        
        connections = connection_manager.get_user_connections(user_id)
        
        assert len(connections) == 2
        assert conn1 in connections
        assert conn2 in connections
    
    def test_get_user_connections_nonexistent(self, connection_manager):
        """Test getting connections for non-existent user."""
        connections = connection_manager.get_user_connections("nonexistent")
        
        assert connections == set()
    
    async def test_get_stats(self, connection_manager):
        """Test getting connection statistics."""
        # Initially empty
        stats = connection_manager.get_stats()
        assert stats["total_connections"] == 0
        assert stats["unique_users"] == 0
        assert stats["active_conversations"] == 0
        
        # Add some connections
        websocket1 = AsyncMock()
        websocket2 = AsyncMock()
        
        conversation_id = str(uuid.uuid4())
        await connection_manager.connect(websocket1, "user1", conversation_id)
        await connection_manager.connect(websocket2, "user1")  # Same user, different conversation
        await connection_manager.connect(websocket2, "user2", conversation_id)
        
        stats = connection_manager.get_stats()
        
        assert stats["total_connections"] == 3
        assert stats["unique_users"] == 2
        assert stats["active_conversations"] == 1
        assert "user1" in stats["connections_per_user"]
        assert stats["connections_per_user"]["user1"] == 2