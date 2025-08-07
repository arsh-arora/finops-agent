"""
E2E Infrastructure Test

Validates that the E2E test infrastructure is properly set up and functioning.
This test ensures all components can be initialized and basic operations work.
"""

import pytest
import asyncio
from tests.e2e.utils import E2EWebSocketClient, wait_for_condition


@pytest.mark.e2e
@pytest.mark.asyncio
class TestE2EInfrastructure:
    """Test E2E infrastructure components."""
    
    async def test_database_connections(
        self, 
        e2e_db_session,
        e2e_redis_client,
        e2e_neo4j_client,
        e2e_qdrant_client
    ):
        """Test that all database connections work."""
        # Test PostgreSQL
        result = await e2e_db_session.execute("SELECT 1 as test")
        assert result.scalar() == 1
        
        # Test Redis
        await e2e_redis_client.set("test_key", "test_value")
        value = await e2e_redis_client.get("test_key")
        assert value == "test_value"
        
        # Test Neo4j
        neo4j_store, namespace = e2e_neo4j_client
        if neo4j_store.driver:
            async with neo4j_store.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                assert record['test'] == 1
        
        # Test Qdrant
        qdrant_store, collection = e2e_qdrant_client
        if hasattr(qdrant_store, 'client') and qdrant_store.client:
            collections = qdrant_store.client.get_collections()
            assert collections is not None
    
    async def test_memory_service_isolation(self, isolated_memory_service):
        """Test that memory service has proper namespace isolation."""
        # Verify service is initialized
        assert isolated_memory_service.is_initialized
        
        # Verify health status
        health = isolated_memory_service.get_health_status()
        assert health['initialized'] is True
        assert health['memory_client_available'] is True
    
    async def test_agent_registry_initialization(self, e2e_agent_registry):
        """Test that agent registry initializes properly."""
        # Verify registry is available
        assert e2e_agent_registry is not None
        
        # Test basic agent selection (should work with mocked LLM)
        user_context = {
            'user_id': 'test_user',
            'message': 'What are my AWS costs?',
            'conversation_id': 'test_conv'
        }
        
        agent = await e2e_agent_registry.get_agent(user_context=user_context)
        assert agent is not None
        assert hasattr(agent, 'get_domain')
    
    async def test_websocket_client_creation(self, e2e_auth_token):
        """Test that WebSocket client can be created and configured."""
        client = E2EWebSocketClient()
        
        # Test basic configuration
        assert client.base_url == "ws://localhost:8000/api/v1/ws/chat"
        assert client.timeout == 30
        assert not client.is_connected
        
        # Test event handling setup
        client.add_event_filter("test_event", lambda data: True)
        assert "test_event" in client.event_filters
    
    def test_celery_configuration(self, celery_config, live_celery_config):
        """Test Celery configuration fixtures."""
        # Test eager mode config
        assert celery_config['task_always_eager'] is True
        assert 'broker_url' in celery_config
        
        # Test live worker config  
        assert live_celery_config['task_always_eager'] is False
        assert 'broker_url' in live_celery_config
    
    def test_test_context_generation(self, e2e_test_context):
        """Test that test context provides unique identifiers."""
        assert 'test_id' in e2e_test_context
        assert 'user_id' in e2e_test_context
        assert 'conversation_id' in e2e_test_context
        assert 'request_id' in e2e_test_context
        assert 'namespace' in e2e_test_context
        
        # Verify uniqueness
        assert e2e_test_context['user_id'].startswith('e2e_user_')
        assert e2e_test_context['conversation_id'].startswith('e2e_conv_')
    
    def test_sample_messages(
        self, 
        sample_finops_message,
        sample_github_message,
        sample_document_message
    ):
        """Test that sample message fixtures are properly formatted."""
        for message in [sample_finops_message, sample_github_message, sample_document_message]:
            # Verify required fields
            assert 'type' in message
            assert 'text' in message
            assert 'metadata' in message
            assert 'timestamp' in message
            assert 'message_id' in message
            
            # Verify proper format
            assert message['type'] == 'user_message'
            assert len(message['text']) > 0
            assert 'source' in message['metadata']
            assert message['metadata']['source'] == 'e2e_test'
    
    async def test_performance_monitor(self, performance_monitor):
        """Test performance monitoring utilities."""
        # Test timer functionality
        performance_monitor.start_timer("test_operation")
        await asyncio.sleep(0.01)  # 10ms
        duration = performance_monitor.end_timer("test_operation")
        
        assert duration is not None
        assert duration >= 10  # Should be at least 10ms
        
        # Test metrics retrieval
        metrics = performance_monitor.get_metrics()
        assert "test_operation" in metrics
        
        # Test performance assertion
        performance_monitor.assert_performance("test_operation", 100)  # Should pass
    
    def test_database_helper_functions(self, create_e2e_conversation_event):
        """Test database helper functions are available."""
        assert callable(create_e2e_conversation_event)
    
    async def test_wait_for_condition_utility(self):
        """Test wait_for_condition helper function."""
        # Test condition that becomes true
        counter = [0]
        
        def increment_condition():
            counter[0] += 1
            return counter[0] >= 3
        
        result = await wait_for_condition(increment_condition, timeout=1)
        assert result is True
        assert counter[0] >= 3
        
        # Test timeout
        def never_true():
            return False
        
        with pytest.raises(TimeoutError):
            await wait_for_condition(never_true, timeout=0.1)