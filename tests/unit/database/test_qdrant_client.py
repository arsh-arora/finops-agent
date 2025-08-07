"""
Tests for Qdrant vector database client.
"""

import pytest
from unittest.mock import MagicMock, patch
import uuid

from src.database.qdrant_client import QdrantVectorStore


@pytest.mark.unit
@pytest.mark.database
class TestQdrantVectorStore:
    """Test Qdrant vector store functionality."""
    
    async def test_connect_success(self):
        """Test successful Qdrant connection."""
        with patch('src.database.qdrant_client.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_collections = MagicMock()
            mock_collections.collections = []
            mock_client.get_collections.return_value = mock_collections
            mock_client_class.return_value = mock_client
            
            qdrant_store = QdrantVectorStore()
            await qdrant_store.connect()
            
            assert qdrant_store.client == mock_client
            mock_client.get_collections.assert_called_once()
    
    async def test_connect_with_api_key(self):
        """Test connection with API key."""
        with patch('src.database.qdrant_client.QdrantClient') as mock_client_class:
            with patch('src.database.qdrant_client.settings') as mock_settings:
                mock_settings.QDRANT_API_KEY = "test-api-key"
                mock_settings.QDRANT_HOST = "test-host"
                mock_settings.QDRANT_PORT = 6333
                
                mock_client = MagicMock()
                mock_collections = MagicMock()
                mock_collections.collections = []
                mock_client.get_collections.return_value = mock_collections
                mock_client_class.return_value = mock_client
                
                qdrant_store = QdrantVectorStore()
                await qdrant_store.connect()
                
                # Verify client was created with API key and HTTPS
                mock_client_class.assert_called_once_with(
                    host="test-host",
                    port=6333,
                    api_key="test-api-key",
                    https=True
                )
    
    async def test_create_collection_new(self, mock_qdrant):
        """Test creating new collection."""
        # Setup mock to show collection doesn't exist
        mock_collections = MagicMock()
        mock_collections.collections = []  # Empty list means collection doesn't exist
        mock_qdrant.client.get_collections.return_value = mock_collections
        
        await mock_qdrant.create_collection()
        
        mock_qdrant.client.create_collection.assert_called_once()
    
    async def test_create_collection_existing(self, mock_qdrant):
        """Test handling existing collection."""
        # Setup mock to show collection exists
        mock_collection = MagicMock()
        mock_collection.name = "conversation_embeddings"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_qdrant.client.get_collections.return_value = mock_collections
        
        await mock_qdrant.create_collection()
        
        # Should not call create_collection since it already exists
        mock_qdrant.client.create_collection.assert_not_called()
    
    async def test_store_embedding_success(self, mock_qdrant):
        """Test storing embedding successfully."""
        vector = [0.1, 0.2, 0.3]
        payload = {"user_id": "test", "content": "test content"}
        
        point_id = await mock_qdrant.store_embedding(vector, payload)
        
        assert isinstance(point_id, str)
        mock_qdrant.client.upsert.assert_called_once()
    
    async def test_store_embedding_with_custom_id(self, mock_qdrant):
        """Test storing embedding with custom point ID."""
        vector = [0.1, 0.2, 0.3]
        payload = {"user_id": "test", "content": "test content"}
        custom_id = str(uuid.uuid4())
        
        point_id = await mock_qdrant.store_embedding(vector, payload, custom_id)
        
        assert point_id == custom_id
        mock_qdrant.client.upsert.assert_called_once()
    
    async def test_search_similar_success(self, mock_qdrant):
        """Test searching for similar vectors."""
        query_vector = [0.1, 0.2, 0.3]
        
        # Setup mock search results
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.score = 0.95
        mock_point.payload = {"user_id": "test", "content": "test content"}
        mock_qdrant.client.search.return_value = [mock_point]
        
        results = await mock_qdrant.search_similar(query_vector, limit=5)
        
        assert len(results) == 1
        assert results[0]["id"] == "test_id"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"] == {"user_id": "test", "content": "test content"}
        mock_qdrant.client.search.assert_called_once()
    
    async def test_search_similar_with_filter(self, mock_qdrant):
        """Test searching with filter conditions."""
        query_vector = [0.1, 0.2, 0.3]
        filter_conditions = {"user_id": "test_user"}
        
        mock_qdrant.client.search.return_value = []
        
        results = await mock_qdrant.search_similar(
            query_vector,
            limit=10,
            score_threshold=0.8,
            filter_conditions=filter_conditions
        )
        
        assert results == []
        # Verify search was called with filter
        call_args = mock_qdrant.client.search.call_args
        assert call_args[1]["query_filter"] is not None
    
    async def test_search_similar_empty_results(self, mock_qdrant):
        """Test search with no results."""
        query_vector = [0.1, 0.2, 0.3]
        mock_qdrant.client.search.return_value = []
        
        results = await mock_qdrant.search_similar(query_vector)
        
        assert results == []
    
    async def test_delete_vectors_success(self, mock_qdrant):
        """Test deleting vectors successfully."""
        point_ids = ["id1", "id2", "id3"]
        
        result = await mock_qdrant.delete_vectors(point_ids)
        
        assert result is True
        mock_qdrant.client.delete.assert_called_once()
    
    async def test_get_collection_info_success(self, mock_qdrant):
        """Test getting collection information."""
        # Setup mock collection info
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.indexed_vectors_count = 95
        mock_info.points_count = 100
        mock_info.segments_count = 1
        mock_info.status = "green"
        mock_qdrant.client.get_collection.return_value = mock_info
        
        info = await mock_qdrant.get_collection_info()
        
        assert info["vectors_count"] == 100
        assert info["indexed_vectors_count"] == 95
        assert info["points_count"] == 100
        assert info["segments_count"] == 1
        assert info["status"] == "green"
    
    async def test_error_handling(self, mock_qdrant):
        """Test error handling in Qdrant operations."""
        # Setup client to raise exceptions
        mock_qdrant.client.upsert.side_effect = Exception("Qdrant error")
        mock_qdrant.client.search.side_effect = Exception("Search error")
        mock_qdrant.client.delete.side_effect = Exception("Delete error")
        mock_qdrant.client.get_collection.side_effect = Exception("Info error")
        
        # Test all methods handle exceptions gracefully
        with pytest.raises(Exception):
            await mock_qdrant.store_embedding([0.1, 0.2], {"test": "data"})
        
        results = await mock_qdrant.search_similar([0.1, 0.2])
        assert results == []
        
        result = await mock_qdrant.delete_vectors(["id1"])
        assert result is False
        
        info = await mock_qdrant.get_collection_info()
        assert info == {}