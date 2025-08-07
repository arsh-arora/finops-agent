from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import json
import logging
import asyncio
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for the FinOps Agent Chat system.
    
    Handles connection lifecycle, message routing, and user session management.
    """
    
    def __init__(self):
        # Active connections: {connection_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # User connections: {user_id: set of connection_ids}
        self.user_connections: Dict[str, Set[str]] = {}
        
        # Connection metadata: {connection_id: metadata}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Conversation sessions: {conversation_id: set of connection_ids}
        self.conversation_sessions: Dict[str, Set[str]] = {}
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Accept a WebSocket connection and register it.
        
        Args:
            websocket: WebSocket connection
            user_id: Authenticated user ID
            conversation_id: Optional conversation session ID
            
        Returns:
            str: Connection ID
        """
        await websocket.accept()
        
        # Generate unique connection ID
        connection_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[connection_id] = websocket
        
        # Track user connections
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        # Store connection metadata
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "connected_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        
        # Add to conversation session if provided
        if conversation_id:
            if conversation_id not in self.conversation_sessions:
                self.conversation_sessions[conversation_id] = set()
            self.conversation_sessions[conversation_id].add(connection_id)
        
        logger.info(f"User {user_id} connected with connection {connection_id}")
        
        # Send welcome message
        await self.send_personal_message(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "message": "Connected to FinOps Agent Chat",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """
        Disconnect a WebSocket connection and clean up.
        
        Args:
            connection_id: Connection ID to disconnect
        """
        if connection_id not in self.active_connections:
            return
        
        # Get connection metadata
        metadata = self.connection_metadata.get(connection_id, {})
        user_id = metadata.get("user_id")
        conversation_id = metadata.get("conversation_id")
        
        # Remove from active connections
        del self.active_connections[connection_id]
        
        # Clean up user connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Clean up conversation sessions
        if conversation_id and conversation_id in self.conversation_sessions:
            self.conversation_sessions[conversation_id].discard(connection_id)
            if not self.conversation_sessions[conversation_id]:
                del self.conversation_sessions[conversation_id]
        
        # Remove metadata
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"Connection {connection_id} disconnected")
    
    async def send_personal_message(self, connection_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: Target connection ID
            message: Message to send
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Connection {connection_id} not found")
            return
        
        try:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(json.dumps(message))
            
            # Update last activity
            if connection_id in self.connection_metadata:
                self.connection_metadata[connection_id]["last_activity"] = datetime.utcnow().isoformat()
                
        except Exception as e:
            logger.error(f"Error sending message to {connection_id}: {e}")
            await self.disconnect(connection_id)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """
        Send a message to all connections of a specific user.
        
        Args:
            user_id: Target user ID
            message: Message to send
        """
        if user_id not in self.user_connections:
            logger.warning(f"No active connections for user {user_id}")
            return
        
        connection_ids = list(self.user_connections[user_id])
        for connection_id in connection_ids:
            await self.send_personal_message(connection_id, message)
    
    async def send_to_conversation(self, conversation_id: str, message: Dict[str, Any]):
        """
        Send a message to all connections in a conversation session.
        
        Args:
            conversation_id: Target conversation ID
            message: Message to send
        """
        if conversation_id not in self.conversation_sessions:
            logger.warning(f"No active connections for conversation {conversation_id}")
            return
        
        connection_ids = list(self.conversation_sessions[conversation_id])
        for connection_id in connection_ids:
            await self.send_personal_message(connection_id, message)
    
    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all active connections.
        
        Args:
            message: Message to broadcast
        """
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            await self.send_personal_message(connection_id, message)
    
    def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a connection.
        
        Args:
            connection_id: Connection ID
            
        Returns:
            Dict: Connection metadata or None if not found
        """
        return self.connection_metadata.get(connection_id)
    
    def get_user_connections(self, user_id: str) -> Set[str]:
        """
        Get all connection IDs for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Set[str]: Set of connection IDs
        """
        return self.user_connections.get(user_id, set())
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.
        
        Returns:
            Dict: Connection statistics
        """
        return {
            "total_connections": len(self.active_connections),
            "unique_users": len(self.user_connections),
            "active_conversations": len(self.conversation_sessions),
            "connections_per_user": {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            }
        }


# Global connection manager instance
connection_manager = ConnectionManager()