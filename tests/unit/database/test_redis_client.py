"""
Tests for Redis client functionality.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from src.database.redis_client import RedisClient


@pytest.mark.unit
@pytest.mark.database
class TestRedisClient:
    """Test Redis client functionality."""
    
    @pytest.fixture
    def redis_client(self):
        """Create Redis client for testing."""
        return RedisClient()
    
    async def test_connect_success(self, redis_client):
        """Test successful Redis connection."""
        with pytest.importorskip("fakeredis.aioredis"):
            import fakeredis.aioredis
            
            # Mock the connection pool and client
            mock_pool = MagicMock()
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            
            with pytest.MonkeyPatch.context() as m:
                m.setattr("redis.asyncio.ConnectionPool.from_url", lambda *args, **kwargs: mock_pool)
                m.setattr("redis.asyncio.Redis", lambda **kwargs: mock_client)
                
                await redis_client.connect()
                
                assert redis_client.redis_pool == mock_pool
                assert redis_client.redis_client == mock_client
                mock_client.ping.assert_called_once()
    
    async def test_connect_failure(self, redis_client):
        """Test Redis connection failure."""
        with pytest.MonkeyPatch.context() as m:
            m.setattr(
                "redis.asyncio.ConnectionPool.from_url",
                lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Connection failed"))
            )
            
            with pytest.raises(Exception, match="Connection failed"):
                await redis_client.connect()
    
    async def test_disconnect(self, redis_client):
        """Test Redis disconnection."""
        mock_client = AsyncMock()
        redis_client.redis_client = mock_client
        
        await redis_client.disconnect()
        
        mock_client.close.assert_called_once()
    
    async def test_set_string_value(self, mock_redis):
        """Test setting string value."""
        result = await mock_redis.set("test_key", "test_value")
        
        assert result is True
        stored_value = await mock_redis.get("test_key")
        assert stored_value == "test_value"
    
    async def test_set_json_value(self, mock_redis):
        """Test setting JSON value."""
        test_data = {"key": "value", "number": 123}
        
        result = await mock_redis.set("test_json", test_data)
        
        assert result is True
        stored_value = await mock_redis.get("test_json")
        assert stored_value == test_data
    
    async def test_set_with_expiration(self, mock_redis):
        """Test setting value with expiration."""
        result = await mock_redis.set("test_expire", "value", expire=60)
        
        assert result is True
        # Note: fakeredis doesn't fully support TTL, so we just test the call succeeds
    
    async def test_get_nonexistent_key(self, mock_redis):
        """Test getting non-existent key."""
        result = await mock_redis.get("nonexistent")
        
        assert result is None
    
    async def test_get_json_value(self, mock_redis):
        """Test getting JSON value."""
        test_data = {"key": "value", "list": [1, 2, 3]}
        
        await mock_redis.set("json_key", test_data)
        result = await mock_redis.get("json_key")
        
        assert result == test_data
    
    async def test_delete_existing_key(self, mock_redis):
        """Test deleting existing key."""
        await mock_redis.set("delete_me", "value")
        
        result = await mock_redis.delete("delete_me")
        
        assert result is True
        assert await mock_redis.get("delete_me") is None
    
    async def test_delete_nonexistent_key(self, mock_redis):
        """Test deleting non-existent key."""
        result = await mock_redis.delete("nonexistent")
        
        # fakeredis returns 0 for non-existent keys, which our client converts to False
        assert result is False
    
    async def test_exists_existing_key(self, mock_redis):
        """Test checking existence of existing key."""
        await mock_redis.set("exists_key", "value")
        
        result = await mock_redis.exists("exists_key")
        
        assert result is True
    
    async def test_exists_nonexistent_key(self, mock_redis):
        """Test checking existence of non-existent key."""
        result = await mock_redis.exists("nonexistent")
        
        assert result is False
    
    async def test_publish_string_message(self, mock_redis):
        """Test publishing string message."""
        result = await mock_redis.publish("test_channel", "test message")
        
        # fakeredis returns 0 for publish (no subscribers)
        assert isinstance(result, int)
    
    async def test_publish_json_message(self, mock_redis):
        """Test publishing JSON message."""
        test_data = {"type": "test", "data": "value"}
        
        result = await mock_redis.publish("test_channel", test_data)
        
        assert isinstance(result, int)
    
    async def test_redis_error_handling(self, redis_client):
        """Test Redis error handling."""
        # Mock client that raises exceptions
        mock_client = AsyncMock()
        mock_client.set.side_effect = Exception("Redis error")
        mock_client.get.side_effect = Exception("Redis error")
        mock_client.delete.side_effect = Exception("Redis error")
        mock_client.exists.side_effect = Exception("Redis error")
        mock_client.publish.side_effect = Exception("Redis error")
        
        redis_client.redis_client = mock_client
        
        # Test all methods handle exceptions gracefully
        assert await redis_client.set("key", "value") is False
        assert await redis_client.get("key") is None
        assert await redis_client.delete("key") is False
        assert await redis_client.exists("key") is False
        assert await redis_client.publish("channel", "message") == 0