"""
Tests for Neo4j graph database client.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from src.database.neo4j_client import Neo4jGraphStore


@pytest.mark.unit
@pytest.mark.database  
class TestNeo4jGraphStore:
    """Test Neo4j graph store functionality."""
    
    async def test_connect_success(self):
        """Test successful Neo4j connection."""
        with patch('src.database.neo4j_client.AsyncGraphDatabase') as mock_db:
            mock_driver = AsyncMock()
            mock_session = AsyncMock()
            mock_result = AsyncMock()
            mock_record = MagicMock()
            mock_record.__getitem__.return_value = 1
            
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value.__aenter__.return_value = mock_session
            mock_session.run.return_value = mock_result
            mock_result.single.return_value = mock_record
            
            neo4j_store = Neo4jGraphStore()
            await neo4j_store.connect()
            
            assert neo4j_store.driver == mock_driver
            mock_session.run.assert_called()
    
    async def test_connect_failure(self):
        """Test Neo4j connection failure."""
        with patch('src.database.neo4j_client.AsyncGraphDatabase') as mock_db:
            mock_db.driver.side_effect = Exception("Connection failed")
            
            neo4j_store = Neo4jGraphStore()
            
            with pytest.raises(Exception, match="Connection failed"):
                await neo4j_store.connect()
    
    async def test_disconnect(self, mock_neo4j):
        """Test Neo4j disconnection."""
        await mock_neo4j.disconnect()
        
        mock_neo4j.driver.close.assert_called_once()
    
    async def test_create_indexes(self, mock_neo4j):
        """Test creating Neo4j indexes."""
        await mock_neo4j.create_indexes()
        
        # Should have called run multiple times for each index
        assert mock_neo4j.driver.session.return_value.__aenter__.return_value.run.call_count >= 5
    
    async def test_create_user_node_success(self, mock_neo4j):
        """Test creating user node successfully."""
        user_id = "test_user"
        properties = {"name": "Test User", "email": "test@example.com"}
        
        # Setup mock to return a record
        mock_record = MagicMock()
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = mock_record
        
        result = await mock_neo4j.create_user_node(user_id, properties)
        
        assert result is True
    
    async def test_create_user_node_failure(self, mock_neo4j):
        """Test user node creation failure."""
        # Setup mock to return None (no record)
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = None
        
        result = await mock_neo4j.create_user_node("test_user")
        
        assert result is False
    
    async def test_create_memory_node_success(self, mock_neo4j):
        """Test creating memory node successfully."""
        memory_id = str(uuid.uuid4())
        user_id = "test_user"
        content = "Test memory content"
        metadata = {"type": "conversation", "topic": "finops"}
        
        # Setup mock to return a record
        mock_record = MagicMock()
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = mock_record
        
        result = await mock_neo4j.create_memory_node(memory_id, user_id, content, metadata)
        
        assert result is True
    
    async def test_create_memory_node_failure(self, mock_neo4j):
        """Test memory node creation failure."""
        # Setup mock to return None
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = None
        
        result = await mock_neo4j.create_memory_node(
            str(uuid.uuid4()), "test_user", "content"
        )
        
        assert result is False
    
    async def test_create_entity_relationship_success(self, mock_neo4j):
        """Test creating entity relationship successfully."""
        entity_name = "AWS"
        entity_type = "cloud_provider"
        memory_id = str(uuid.uuid4())
        
        # Setup mock to return a record
        mock_record = MagicMock()
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = mock_record
        
        result = await mock_neo4j.create_entity_relationship(
            entity_name, entity_type, memory_id
        )
        
        assert result is True
    
    async def test_create_entity_relationship_custom_type(self, mock_neo4j):
        """Test creating entity relationship with custom relationship type."""
        # Setup mock to return a record
        mock_record = MagicMock()
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = mock_record
        
        result = await mock_neo4j.create_entity_relationship(
            "GitHub", "platform", str(uuid.uuid4()), "USES"
        )
        
        assert result is True
    
    async def test_find_related_memories_with_entities(self, mock_neo4j):
        """Test finding related memories with entity filter."""
        user_id = "test_user"
        entity_names = ["AWS", "S3"]
        
        # Setup mock data
        mock_data = [
            {
                "m": {
                    "memory_id": str(uuid.uuid4()),
                    "content": "Test memory about AWS S3",
                    "timestamp": "2024-01-01T10:00:00Z",
                    "topic": "storage"
                },
                "entities": ["AWS", "S3"]
            }
        ]
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.data.return_value = mock_data
        
        memories = await mock_neo4j.find_related_memories(user_id, entity_names)
        
        assert len(memories) == 1
        assert "AWS" in memories[0]["entities"]
        assert "S3" in memories[0]["entities"]
        assert "Test memory about AWS S3" in memories[0]["content"]
    
    async def test_find_related_memories_no_entities(self, mock_neo4j):
        """Test finding related memories without entity filter."""
        user_id = "test_user"
        
        # Setup mock data
        mock_data = [
            {
                "m": {
                    "memory_id": str(uuid.uuid4()),
                    "content": "General memory",
                    "timestamp": "2024-01-01T10:00:00Z"
                },
                "entities": [None]  # No entities
            }
        ]
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.data.return_value = mock_data
        
        memories = await mock_neo4j.find_related_memories(user_id)
        
        assert len(memories) == 1
        assert memories[0]["entities"] == []  # None values filtered out
    
    async def test_find_related_memories_empty(self, mock_neo4j):
        """Test finding related memories with no results."""
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.data.return_value = []
        
        memories = await mock_neo4j.find_related_memories("nonexistent_user")
        
        assert memories == []
    
    async def test_get_user_context_existing_user(self, mock_neo4j):
        """Test getting context for existing user."""
        user_id = "test_user"
        
        # Setup mock user context data
        mock_user_node = {
            "user_id": user_id,
            "name": "Test User",
            "email": "test@example.com"
        }
        mock_record = {
            "u": mock_user_node,
            "memory_count": 5,
            "entities": ["AWS", "S3", "EC2", None],
            "entity_types": ["cloud_provider", "service", None]
        }
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = mock_record
        
        context = await mock_neo4j.get_user_context(user_id)
        
        assert context["user_id"] == user_id
        assert context["exists"] is True
        assert context["memory_count"] == 5
        assert "AWS" in context["entities"]
        assert "S3" in context["entities"]
        assert None not in context["entities"]  # None values filtered out
        assert "cloud_provider" in context["entity_types"]
        assert context["user_properties"] == mock_user_node
    
    async def test_get_user_context_nonexistent_user(self, mock_neo4j):
        """Test getting context for non-existent user."""
        user_id = "nonexistent_user"
        
        # Setup mock to return None (user not found)
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = None
        
        context = await mock_neo4j.get_user_context(user_id)
        
        assert context["user_id"] == user_id
        assert context["exists"] is False
    
    async def test_error_handling(self, mock_neo4j):
        """Test error handling in Neo4j operations."""
        # Setup session to raise exceptions
        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.side_effect = Exception("Neo4j error")
        
        # Test all methods handle exceptions gracefully
        assert await mock_neo4j.create_user_node("test") is False
        assert await mock_neo4j.create_memory_node("mem_id", "user_id", "content") is False
        assert await mock_neo4j.create_entity_relationship("entity", "type", "mem_id") is False
        
        memories = await mock_neo4j.find_related_memories("test_user")
        assert memories == []
        
        context = await mock_neo4j.get_user_context("test_user")
        assert context["exists"] is False
        assert "error" in context