import redis.asyncio as redis
from typing import Any, Optional, Union
import json
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Async Redis client for caching and pub/sub operations.
    """
    
    def __init__(self):
        self.redis_pool = None
        self.redis_client = None
    
    async def connect(self):
        """Initialize Redis connection pool."""
        try:
            self.redis_pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                max_connections=20,
                retry_on_timeout=True
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[int] = None
    ) -> bool:
        """
        Set a key-value pair in Redis.
        
        Args:
            key: Redis key
            value: Value to store (will be JSON serialized)
            expire: Optional expiration time in seconds
            
        Returns:
            bool: True if successful
        """
        try:
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            result = await self.redis_client.set(key, serialized_value, ex=expire)
            return result is True
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from Redis.
        
        Args:
            key: Redis key
            
        Returns:
            Any: Deserialized value or None if not found
        """
        try:
            value = await self.redis_client.get(key)
            if value is None:
                return None
            
            # Try to deserialize as JSON, fallback to string
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value.decode('utf-8')
                
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Redis key to delete
            
        Returns:
            bool: True if key was deleted
        """
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis.
        
        Args:
            key: Redis key to check
            
        Returns:
            bool: True if key exists
        """
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def publish(self, channel: str, message: Union[str, dict]) -> int:
        """
        Publish a message to a Redis channel.
        
        Args:
            channel: Redis channel name
            message: Message to publish
            
        Returns:
            int: Number of subscribers that received the message
        """
        try:
            if isinstance(message, dict):
                message = json.dumps(message)
            
            result = await self.redis_client.publish(channel, message)
            return result
        except Exception as e:
            logger.error(f"Redis PUBLISH error for channel {channel}: {e}")
            return 0


# Global Redis client instance
redis_client = RedisClient()