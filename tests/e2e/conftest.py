"""
End-to-End Test Configuration and Fixtures
Comprehensive E2E test harness with real database integration and agent system setup
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
import os
import tempfile
import logging
from typing import AsyncGenerator, Dict, Any, Optional
from unittest.mock import AsyncMock
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Database imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
import redis.asyncio as redis

# Application imports  
from src.database.models import Base
from src.database.connection import get_db_session
from src.database.redis_client import RedisClient
from src.database.neo4j_client import Neo4jGraphStore
from src.database.qdrant_client import QdrantVectorStore
from src.auth.jwt_auth import create_access_token
from src.memory.mem0_service import FinOpsMemoryService
from src.memory.config import get_mem0_config
from src.agents import get_agent_registry, Phase4AgentRegistry
from src.websocket.connection_manager import ConnectionManager
from config.settings import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the E2E test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")  
def e2e_settings():
    """E2E test settings with real database connections."""
    return Settings(
        DEBUG=True,
        # Use real PostgreSQL for E2E tests
        POSTGRES_URL=os.getenv("E2E_POSTGRES_URL", "postgresql+asyncpg://postgres:password@localhost:5432/finops_e2e_test"),
        # Use real Redis for E2E tests  
        REDIS_URL=os.getenv("E2E_REDIS_URL", "redis://localhost:6379/10"),
        # Use real Qdrant for E2E tests
        QDRANT_HOST=os.getenv("E2E_QDRANT_HOST", "localhost"),
        QDRANT_PORT=int(os.getenv("E2E_QDRANT_PORT", "6333")),
        # Use real Neo4j for E2E tests
        NEO4J_URI=os.getenv("E2E_NEO4J_URI", "bolt://localhost:7687"),
        NEO4J_USERNAME=os.getenv("E2E_NEO4J_USERNAME", "neo4j"),
        NEO4J_PASSWORD=os.getenv("E2E_NEO4J_PASSWORD", "password"),
        # Celery configuration for E2E tests
        CELERY_BROKER_URL=os.getenv("E2E_CELERY_BROKER", "redis://localhost:6379/11"),
        CELERY_RESULT_BACKEND=os.getenv("E2E_CELERY_BACKEND", "redis://localhost:6379/11"),
        # JWT settings
        JWT_SECRET_KEY="e2e-test-secret-key-do-not-use-in-production",
        JWT_ALGORITHM="HS256",
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
    )


@pytest_asyncio.fixture
async def e2e_db_engine(e2e_settings):
    """Create E2E test database engine with real PostgreSQL."""
    engine = create_async_engine(
        e2e_settings.POSTGRES_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=300
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup: Drop all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest_asyncio.fixture
async def e2e_db_session(e2e_db_engine):
    """Create E2E test database session."""
    async_session = async_sessionmaker(
        e2e_db_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        # Start a transaction
        transaction = await session.begin()
        yield session
        # Rollback transaction for test isolation
        await transaction.rollback()


@pytest_asyncio.fixture  
async def e2e_redis_client(e2e_settings):
    """Create real Redis client for E2E tests."""
    redis_client = redis.from_url(e2e_settings.REDIS_URL, decode_responses=True)
    
    # Test connection
    await redis_client.ping()
    
    yield redis_client
    
    # Cleanup: Flush E2E test database
    await redis_client.flushdb()
    await redis_client.close()


@pytest_asyncio.fixture
async def e2e_neo4j_client(e2e_settings):
    """Create real Neo4j client for E2E tests."""
    # Temporarily override settings for E2E tests
    import config.settings
    original_uri = config.settings.settings.NEO4J_URI
    original_username = config.settings.settings.NEO4J_USERNAME
    original_password = config.settings.settings.NEO4J_PASSWORD
    
    config.settings.settings.NEO4J_URI = e2e_settings.NEO4J_URI
    config.settings.settings.NEO4J_USERNAME = e2e_settings.NEO4J_USERNAME
    config.settings.settings.NEO4J_PASSWORD = e2e_settings.NEO4J_PASSWORD
    
    try:
        neo4j_store = Neo4jGraphStore()
        await neo4j_store.connect()
        
        # Create E2E test namespace
        e2e_namespace = f"e2e_test_{uuid.uuid4().hex[:8]}"
        
        yield neo4j_store, e2e_namespace
        
        # Cleanup: Delete E2E test data
        try:
            if neo4j_store.driver:
                async with neo4j_store.driver.session() as session:
                    await session.run(
                        "MATCH (n) WHERE n.e2e_namespace = $namespace DETACH DELETE n",
                        namespace=e2e_namespace
                    )
        except Exception as cleanup_error:
            logger.warning(f"Neo4j cleanup error: {cleanup_error}")
        
        await neo4j_store.disconnect()
        
    finally:
        # Restore original settings
        config.settings.settings.NEO4J_URI = original_uri
        config.settings.settings.NEO4J_USERNAME = original_username
        config.settings.settings.NEO4J_PASSWORD = original_password


@pytest_asyncio.fixture
async def e2e_qdrant_client(e2e_settings):
    """Create real Qdrant client for E2E tests."""
    # Temporarily override settings for E2E tests
    import config.settings
    original_host = config.settings.settings.QDRANT_HOST
    original_port = config.settings.settings.QDRANT_PORT
    
    config.settings.settings.QDRANT_HOST = e2e_settings.QDRANT_HOST
    config.settings.settings.QDRANT_PORT = e2e_settings.QDRANT_PORT
    
    try:
        qdrant_store = QdrantVectorStore()
        await qdrant_store.connect()
        
        # Create E2E test collection
        e2e_collection = f"e2e_test_{uuid.uuid4().hex[:8]}"
        
        # Set the collection name for this test
        qdrant_store.collection_name = e2e_collection
        
        yield qdrant_store, e2e_collection
        
        # Cleanup: Delete E2E test collection
        try:
            if hasattr(qdrant_store, 'delete_collection'):
                await qdrant_store.delete_collection(e2e_collection)
        except Exception as cleanup_error:
            logger.warning(f"Qdrant cleanup error: {cleanup_error}")
        
    finally:
        # Restore original settings
        config.settings.settings.QDRANT_HOST = original_host
        config.settings.settings.QDRANT_PORT = original_port


@pytest.fixture
def celery_config(e2e_settings):
    """Default Celery configuration for E2E tests (eager mode)."""
    return {
        'task_always_eager': True,
        'task_eager_propagates': True,
        'task_store_eager_result': True,
        'broker_url': e2e_settings.CELERY_BROKER_URL,
        'result_backend': e2e_settings.CELERY_RESULT_BACKEND,
        'task_serializer': 'json',
        'result_serializer': 'json',
        'accept_content': ['json'],
        'timezone': 'UTC',
        'enable_utc': True
    }


@pytest.fixture
def live_celery_config(e2e_settings):
    """Celery configuration for live worker tests."""
    return {
        'task_always_eager': False,
        'task_eager_propagates': False,
        'worker_prefetch_multiplier': 1,
        'broker_url': e2e_settings.CELERY_BROKER_URL,
        'result_backend': e2e_settings.CELERY_RESULT_BACKEND,
        'task_serializer': 'json',
        'result_serializer': 'json',
        'accept_content': ['json'],
        'task_routes': {
            'src.workers.tasks.*': {'queue': 'e2e_test_queue'}
        },
        'task_create_missing_queues': True,
        'timezone': 'UTC',
        'enable_utc': True
    }


@pytest_asyncio.fixture
async def isolated_memory_service(e2e_neo4j_client, e2e_qdrant_client, e2e_settings):
    """Create isolated Mem0 service with unique namespace per test."""
    neo4j_store, neo4j_namespace = e2e_neo4j_client
    qdrant_store, qdrant_collection = e2e_qdrant_client
    
    # Create isolated memory configuration
    memory_config = get_mem0_config()
    
    # Configure Neo4j with namespace isolation
    memory_config['graph_store'] = {
        'provider': 'neo4j',
        'config': {
            'url': e2e_settings.NEO4J_URI,
            'username': e2e_settings.NEO4J_USERNAME, 
            'password': e2e_settings.NEO4J_PASSWORD,
            'database': 'neo4j'
        }
    }
    
    # Configure Qdrant with isolated collection
    memory_config['vector_store'] = {
        'provider': 'qdrant',
        'config': {
            'host': e2e_settings.QDRANT_HOST,
            'port': e2e_settings.QDRANT_PORT,
            'collection_name': qdrant_collection
        }
    }
    
    # Initialize memory service
    memory_service = FinOpsMemoryService(config=memory_config)
    await memory_service.initialize()
    
    # Store namespace information for cleanup
    memory_service._e2e_namespace = neo4j_namespace
    memory_service._e2e_collection = qdrant_collection
    
    yield memory_service
    
    # Cleanup is handled by individual database fixtures


class MockLLMClient:
    """Mock LLM client for E2E tests to avoid API costs"""
    
    def __init__(self):
        self.call_count = 0
    
    async def complete(self, messages, **kwargs):
        """Mock LLM completion for E2E tests"""
        self.call_count += 1
        
        # Check message content to determine response type
        content = messages[0]["content"].lower()
        
        # Routing requests
        if "agent router" in content or "select" in content or "domain" in content:
            return '{"selected_domain": "finops", "confidence_score": 0.9, "reasoning": "Financial analysis request"}'
        
        # Planning requests  
        return '''
        {
            "tasks": [
                {
                    "id": "task_1", 
                    "tool_name": "finops_cost_analysis",
                    "inputs": {"query": "analyze costs"},
                    "description": "Analyze financial costs"
                }
            ],
            "dependencies": [],
            "estimated_cost": {"tokens": 1000, "usd_cost": 0.02}
        }
        '''


@pytest_asyncio.fixture
async def e2e_agent_registry(isolated_memory_service):
    """Create agent registry with real memory service for E2E tests."""
    
    # Create mock LLM client
    mock_llm_client = MockLLMClient()
    
    # Initialize registry with memory service and mock LLM client
    registry = Phase4AgentRegistry(
        memory_service=isolated_memory_service,
        llm_client=mock_llm_client
    )
    
    yield registry
    
    # Clean up any agent instances
    if hasattr(registry, '_agent_instances'):
        for agent in registry._agent_instances.values():
            if hasattr(agent, 'cleanup'):
                try:
                    await agent.cleanup()
                except Exception as e:
                    logger.warning(f"Agent cleanup error: {e}")


@pytest.fixture
def e2e_auth_token():
    """Create JWT token for E2E test authentication."""
    token_data = {
        "sub": f"e2e_test_user_{uuid.uuid4().hex[:8]}",
        "username": "e2e_test_user",
        "email": "e2e.test@example.com"
    }
    
    return create_access_token(data=token_data)


@pytest_asyncio.fixture
async def e2e_connection_manager():
    """Create WebSocket connection manager for E2E tests."""
    manager = ConnectionManager()
    yield manager
    
    # Cleanup all connections
    for connection_id in list(manager.active_connections.keys()):
        await manager.disconnect(connection_id)


@pytest.fixture
def e2e_test_context():
    """Provide test context with unique identifiers."""
    test_id = uuid.uuid4().hex[:8]
    return {
        'test_id': test_id,
        'user_id': f"e2e_user_{test_id}",
        'conversation_id': f"e2e_conv_{test_id}",
        'request_id': f"e2e_req_{test_id}",
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'namespace': f"e2e_test_{test_id}"
    }


# Performance monitoring fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during E2E tests."""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
            
        def start_timer(self, operation: str):
            """Start timing an operation."""
            import time
            self.start_times[operation] = time.time()
            
        def end_timer(self, operation: str):
            """End timing and record duration."""
            import time
            if operation in self.start_times:
                duration = (time.time() - self.start_times[operation]) * 1000
                self.metrics[operation] = duration
                del self.start_times[operation]
                return duration
            return None
            
        def get_metrics(self) -> Dict[str, float]:
            """Get all recorded metrics."""
            return self.metrics.copy()
            
        def assert_performance(self, operation: str, max_duration_ms: float):
            """Assert operation completed within time limit."""
            if operation in self.metrics:
                actual_duration = self.metrics[operation]
                assert actual_duration <= max_duration_ms, \
                    f"{operation} took {actual_duration:.1f}ms, expected <= {max_duration_ms}ms"
    
    return PerformanceMonitor()


# Test data fixtures
@pytest.fixture
def sample_finops_message():
    """Sample FinOps message for testing."""
    return {
        "type": "user_message",
        "text": "What are my current AWS costs and optimization opportunities?",
        "metadata": {
            "source": "e2e_test",
            "cloud_provider": "aws",
            "account_id": "123456789012"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": str(uuid.uuid4())
    }


@pytest.fixture
def sample_github_message():
    """Sample GitHub analysis message for testing."""
    return {
        "type": "user_message", 
        "text": "Analyze the security vulnerabilities in my repository",
        "metadata": {
            "source": "e2e_test",
            "repository_url": "https://github.com/example/test-repo"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": str(uuid.uuid4())
    }


@pytest.fixture
def sample_document_message():
    """Sample document analysis message for testing."""
    return {
        "type": "user_message",
        "text": "Extract tables and financial data from this document",
        "metadata": {
            "source": "e2e_test",
            "document_type": "pdf"
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": str(uuid.uuid4())
    }


# Database helpers for E2E tests
async def create_test_conversation_event(
    db_session: AsyncSession,
    user_id: str,
    conversation_id: str,
    message_text: str,
    **kwargs
):
    """Create conversation event for E2E testing."""
    from src.database.models import ConversationEvent, ConversationEventType, AgentType
    
    event = ConversationEvent(
        conversation_id=conversation_id,
        user_id=user_id, 
        event_type=ConversationEventType.USER_MESSAGE,
        message_text=message_text,
        event_metadata={
            'e2e_test': True,
            'test_timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
    )
    
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)
    return event


@pytest.fixture
def create_e2e_conversation_event():
    """Provide helper to create conversation events."""
    return create_test_conversation_event


# Cleanup utilities
@pytest_asyncio.fixture(autouse=True)
async def e2e_test_cleanup():
    """Automatic cleanup after each E2E test."""
    yield
    
    # Additional cleanup can be added here
    # Most cleanup is handled by individual fixtures