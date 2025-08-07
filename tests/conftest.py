"""
Test configuration and fixtures for FinOps Agent Chat.
"""

import pytest
import asyncio
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock
import uuid
from datetime import datetime, timezone

# Database imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
import fakeredis.aioredis

# Application imports
from src.database.models import Base, ConversationEvent, ConversationEventType, AgentType
from src.database.redis_client import RedisClient
from src.websocket.connection_manager import ConnectionManager
from src.auth.jwt_auth import JWTAuth
from config.settings import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        DEBUG=True,
        POSTGRES_URL="sqlite+aiosqlite:///:memory:",
        REDIS_URL="redis://localhost:6379/0",
        QDRANT_HOST="localhost",
        QDRANT_PORT=6333,
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="test",
        NEO4J_PASSWORD="test",
        JWT_SECRET_KEY="test-secret-key",
        JWT_ALGORITHM="HS256",
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
    )


@pytest.fixture
async def db_engine():
    """Create test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    fake_redis = fakeredis.aioredis.FakeRedis()
    
    # Create mock Redis client
    redis_client = RedisClient()
    redis_client.redis_client = fake_redis
    redis_client.redis_pool = MagicMock()
    
    return redis_client


@pytest.fixture
def mock_qdrant():
    """Create mock Qdrant client."""
    from unittest.mock import MagicMock
    
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.create_collection.return_value = None
    mock_client.upsert.return_value = None
    mock_client.search.return_value = []
    mock_client.delete.return_value = None
    mock_client.get_collection.return_value = MagicMock(
        vectors_count=0,
        indexed_vectors_count=0,
        points_count=0,
        segments_count=0,
        status="green"
    )
    
    from src.database.qdrant_client import QdrantVectorStore
    qdrant_store = QdrantVectorStore()
    qdrant_store.client = mock_client
    
    return qdrant_store


@pytest.fixture
def mock_neo4j():
    """Create mock Neo4j client."""
    mock_driver = AsyncMock()
    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_record = MagicMock()
    
    # Setup mock chain
    mock_driver.session.return_value.__aenter__.return_value = mock_session
    mock_session.run.return_value = mock_result
    mock_result.single.return_value = mock_record
    mock_result.data.return_value = []
    mock_record.__getitem__.return_value = 1
    
    from src.database.neo4j_client import Neo4jGraphStore
    neo4j_store = Neo4jGraphStore()
    neo4j_store.driver = mock_driver
    
    return neo4j_store


@pytest.fixture
def jwt_auth():
    """Create JWT auth instance for testing."""
    return JWTAuth()


@pytest.fixture
def connection_manager():
    """Create WebSocket connection manager."""
    return ConnectionManager()


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "user_id": "test_user_123",
        "username": "testuser",
        "email": "test@example.com"
    }


@pytest.fixture
def sample_conversation_event_data():
    """Sample conversation event data."""
    return {
        "conversation_id": str(uuid.uuid4()),
        "user_id": "test_user_123",
        "agent_type": AgentType.FINOPS,
        "event_type": ConversationEventType.USER_MESSAGE,
        "message_text": "Help me analyze my AWS costs",
        "event_metadata": {
            "source": "web",
            "user_agent": "Mozilla/5.0"
        }
    }


@pytest.fixture
def sample_websocket_message():
    """Sample WebSocket message."""
    return {
        "type": "user_message",
        "text": "What are my cloud costs this month?",
        "metadata": {"source": "web"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": str(uuid.uuid4())
    }


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


# Factory fixtures for creating test data
class ConversationEventFactory:
    """Factory for creating ConversationEvent instances."""
    
    @staticmethod
    def create(**kwargs) -> ConversationEvent:
        defaults = {
            "conversation_id": str(uuid.uuid4()),
            "user_id": "test_user",
            "event_type": ConversationEventType.USER_MESSAGE,
            "message_text": "Test message",
            "event_metadata": {}
        }
        defaults.update(kwargs)
        return ConversationEvent(**defaults)


@pytest.fixture
def conversation_event_factory():
    """Provide ConversationEvent factory."""
    return ConversationEventFactory


# Async utility functions
async def create_test_conversation_event(
    db_session: AsyncSession,
    **kwargs
) -> ConversationEvent:
    """Create and save a test conversation event."""
    event = ConversationEventFactory.create(**kwargs)
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)
    return event


@pytest.fixture
def create_conversation_event():
    """Provide function to create conversation events."""
    return create_test_conversation_event