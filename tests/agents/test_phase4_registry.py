"""
Test suite for Phase 4 Agent Registry and Routing
Comprehensive testing of Phase4AgentRegistry and intelligent agent routing
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents import Phase4AgentRegistry, initialize_agent_registry, get_agent_registry
from src.agents.routing.selector import AgentRouter, RoutingDecision
from src.agents.models import ChatRequest


@pytest.fixture
def mock_memory_service():
    """Mock memory service for testing"""
    mock = Mock()
    mock.search_memories = AsyncMock(return_value=[])
    mock.add_memories = AsyncMock(return_value=None)
    mock.delete_memories = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for routing testing"""
    mock = Mock()
    mock.complete = AsyncMock(return_value='{"selected_domain": "finops", "confidence_score": 0.92, "reasoning": "Financial analysis request"}')
    return mock


@pytest.fixture
def phase4_registry(mock_memory_service, mock_llm_client):
    """Create Phase4AgentRegistry instance for testing"""
    return Phase4AgentRegistry(
        memory_service=mock_memory_service,
        llm_client=mock_llm_client,
        config={'test_mode': True}
    )


@pytest.fixture
def sample_user_contexts():
    """Sample user contexts for routing testing"""
    return {
        'finops_request': {
            'message': 'Calculate NPV for this investment project with 10% discount rate',
            'user_id': 'test_user',
            'session_id': 'test_session',
            'request_id': 'req_001'
        },
        'github_request': {
            'message': 'Analyze the security vulnerabilities in this GitHub repository',
            'user_id': 'test_user',
            'session_id': 'test_session',
            'request_id': 'req_002'
        },
        'document_request': {
            'message': 'Extract bounding boxes from this PDF document with high precision',
            'user_id': 'test_user',
            'session_id': 'test_session',
            'request_id': 'req_003'
        },
        'research_request': {
            'message': 'Research the latest trends in cloud computing for 2024',
            'user_id': 'test_user',
            'session_id': 'test_session',
            'request_id': 'req_004'
        },
        'deep_research_request': {
            'message': 'Conduct comprehensive analysis combining financial, security, and market research',
            'user_id': 'test_user',
            'session_id': 'test_session',
            'request_id': 'req_005'
        },
        'ambiguous_request': {
            'message': 'Help me understand this',
            'user_id': 'test_user',
            'session_id': 'test_session',
            'request_id': 'req_006'
        }
    }


class TestPhase4AgentRegistry:
    """Test cases for Phase4AgentRegistry"""
    
    def test_registry_initialization(self, phase4_registry):
        """Test registry initialization and agent registration"""
        assert phase4_registry.memory_service is not None
        assert phase4_registry.router is not None
        assert isinstance(phase4_registry.router, AgentRouter)
        
        # Verify all Phase 4 agents are registered
        available_domains = phase4_registry.get_available_domains()
        expected_domains = ['finops', 'github', 'document', 'research', 'deep_research']
        
        for domain in expected_domains:
            assert domain in available_domains
    
    @pytest.mark.asyncio
    async def test_get_agent_routing(self, phase4_registry, sample_user_contexts):
        """Test agent retrieval with intelligent routing"""
        # Test explicit domain specification
        finops_agent = await phase4_registry.get_agent(
            domain='finops',
            user_context=sample_user_contexts['finops_request']
        )
        assert finops_agent is not None
        assert finops_agent.get_domain() == 'finops'
        
        # Test intelligent routing without explicit domain
        with patch.object(phase4_registry.router, 'select') as mock_select:
            from src.agents.finops import AdvancedFinOpsAgent
            mock_select.return_value = AdvancedFinOpsAgent
            
            routed_agent = await phase4_registry.get_agent(
                user_context=sample_user_contexts['finops_request']
            )
            
            assert routed_agent is not None
            mock_select.assert_called_once()
    
    def test_agent_capabilities_retrieval(self, phase4_registry):
        """Test retrieval of agent capabilities"""
        # Test FinOps agent capabilities
        finops_capabilities = phase4_registry.get_agent_capabilities('finops')
        assert 'capabilities' in finops_capabilities
        assert 'advanced_financial_modeling' in finops_capabilities['capabilities']
        assert 'intelligent_anomaly_detection' in finops_capabilities['capabilities']
        
        # Test GitHub agent capabilities
        github_capabilities = phase4_registry.get_agent_capabilities('github')
        assert 'capabilities' in github_capabilities
        assert 'repository_security_analysis' in github_capabilities['capabilities']
        assert 'vulnerability_assessment' in github_capabilities['capabilities']
        
        # Test Deep Research agent capabilities
        deep_research_capabilities = phase4_registry.get_agent_capabilities('deep_research')
        assert 'capabilities' in deep_research_capabilities
        assert 'multi_hop_orchestration' in deep_research_capabilities['capabilities']
        assert 'cross_domain_synthesis' in deep_research_capabilities['capabilities']
    
    @pytest.mark.asyncio
    async def test_health_check(self, phase4_registry):
        """Test registry health check functionality"""
        # Mock some agents to be instantiated
        with patch.object(phase4_registry, '_agent_instances') as mock_instances:
            mock_agent = Mock()
            mock_agent.get_capabilities.return_value = ['test_capability']
            mock_agent.get_domain.return_value = 'test_domain'
            mock_agent.memory_service = phase4_registry.memory_service
            
            mock_instances.__iter__.return_value = iter([('test_domain', mock_agent)])
            mock_instances.items.return_value = [('test_domain', mock_agent)]
            
            health_status = await phase4_registry.health_check()
            
            assert health_status['registry_status'] in ['healthy', 'degraded']
            assert 'agent_health' in health_status
            assert 'routing_stats' in health_status
            assert health_status['memory_service_status'] == 'connected'
    
    def test_routing_stats(self, phase4_registry):
        """Test routing statistics retrieval"""
        stats = phase4_registry.get_routing_stats()
        
        assert 'total_requests' in stats
        assert 'successful_routes' in stats
        assert 'fallback_used' in stats
        assert 'routing_misses' in stats
        assert 'miss_rate_percent' in stats
        assert 'success_rate_percent' in stats


class TestAgentRouter:
    """Test cases for AgentRouter"""
    
    def test_router_initialization(self, mock_llm_client):
        """Test router initialization"""
        router = AgentRouter(llm_client=mock_llm_client)
        
        assert router._llm_client == mock_llm_client
        assert len(router._registry) == 0
        assert router._routing_stats['total_requests'] == 0
    
    def test_agent_registration(self, mock_llm_client):
        """Test agent registration in router"""
        router = AgentRouter(llm_client=mock_llm_client)
        
        # Mock agent class
        mock_agent_class = Mock()
        mock_agent_class.__name__ = 'TestAgent'
        mock_agent_class.get_domain = Mock(return_value='test_domain')
        mock_agent_class.get_capabilities = Mock(return_value=['test_capability'])
        mock_agent_class.__doc__ = 'Test agent for testing'
        
        capabilities = {'test_capability': 'Test capability description'}
        
        router.register_agent(
            domain='test_domain',
            agent_class=mock_agent_class,
            capabilities=capabilities
        )
        
        assert 'test_domain' in router._registry
        assert router._registry['test_domain'] == mock_agent_class
        assert 'test_domain' in router._agent_capabilities
    
    @pytest.mark.asyncio
    async def test_llm_based_routing(self, mock_llm_client, sample_user_contexts):
        """Test LLM-based intelligent routing"""
        router = AgentRouter(llm_client=mock_llm_client)
        
        # Register a test agent
        mock_agent_class = Mock()
        mock_agent_class.__name__ = 'FinOpsAgent'
        mock_agent_class.get_domain = Mock(return_value='finops')
        mock_agent_class.get_capabilities = Mock(return_value=['financial_modeling'])
        
        router.register_agent('finops', mock_agent_class)
        
        # Mock LLM response for routing
        mock_llm_client.complete.return_value = '{"selected_domain": "finops", "confidence_score": 0.92, "reasoning": "Financial calculation request"}'
        
        selected_agent = await router.select(
            domain=None,
            user_ctx=sample_user_contexts['finops_request']
        )
        
        assert selected_agent == mock_agent_class
        mock_llm_client.complete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_heuristic_fallback_routing(self, sample_user_contexts):
        """Test heuristic fallback routing when LLM is unavailable"""
        router = AgentRouter(llm_client=None)  # No LLM client
        
        # Register test agents
        mock_finops_agent = Mock()
        mock_finops_agent.__name__ = 'FinOpsAgent'
        mock_finops_agent.get_domain = Mock(return_value='finops')
        mock_finops_agent.get_capabilities = Mock(return_value=['financial_modeling'])
        
        router.register_agent('finops', mock_finops_agent)
        
        selected_agent = await router.select(
            domain=None,
            user_ctx=sample_user_contexts['finops_request']
        )
        
        # Should fallback to heuristic routing and select finops based on keywords
        assert selected_agent == mock_finops_agent
    
    def test_heuristic_keyword_mapping(self, sample_user_contexts):
        """Test heuristic keyword-based routing"""
        router = AgentRouter(llm_client=None)
        
        # Test different types of requests
        test_cases = [
            (sample_user_contexts['finops_request'], 'finops'),
            (sample_user_contexts['github_request'], 'github'),
            (sample_user_contexts['document_request'], 'document'),
            (sample_user_contexts['research_request'], 'research'),
            (sample_user_contexts['deep_research_request'], 'deep_research')
        ]
        
        for user_ctx, expected_domain in test_cases:
            routing_decision = router._heuristic_fallback_routing(user_ctx)
            
            # Should correctly identify domain based on keywords
            if expected_domain == 'deep_research':
                # Deep research has specific keywords
                assert routing_decision.selected_domain in ['deep_research', 'research']
            else:
                assert routing_decision.selected_domain == expected_domain
            assert routing_decision.fallback_used is True
    
    def test_routing_stats_tracking(self, mock_llm_client):
        """Test routing statistics tracking"""
        router = AgentRouter(llm_client=mock_llm_client)
        
        # Simulate some routing operations
        router._routing_stats['total_requests'] = 10
        router._routing_stats['successful_routes'] = 8
        router._routing_stats['fallback_used'] = 2
        router._routing_stats['routing_misses'] = 0
        
        stats = router.get_routing_stats()
        
        assert stats['total_requests'] == 10
        assert stats['successful_routes'] == 8
        assert stats['success_rate_percent'] == 80.0
        assert stats['miss_rate_percent'] == 0.0


class TestGlobalRegistryFunctions:
    """Test global registry initialization functions"""
    
    def test_initialize_agent_registry(self, mock_memory_service, mock_llm_client):
        """Test global registry initialization"""
        config = {'test_mode': True}
        
        registry = initialize_agent_registry(
            memory_service=mock_memory_service,
            llm_client=mock_llm_client,
            config=config
        )
        
        assert isinstance(registry, Phase4AgentRegistry)
        assert registry.memory_service == mock_memory_service
        assert registry.config == config
    
    def test_get_agent_registry(self, mock_memory_service, mock_llm_client):
        """Test global registry retrieval"""
        # Initialize registry first
        initialize_agent_registry(
            memory_service=mock_memory_service,
            llm_client=mock_llm_client
        )
        
        # Retrieve registry
        registry = get_agent_registry()
        assert isinstance(registry, Phase4AgentRegistry)
    
    def test_get_registry_without_initialization(self):
        """Test error when getting registry without initialization"""
        # Clear any existing registry
        import src.agents
        src.agents._registry_instance = None
        
        with pytest.raises(RuntimeError, match="Agent registry not initialized"):
            get_agent_registry()


class TestIntegrationScenarios:
    """Integration test scenarios for registry and routing"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_agent_selection(self, phase4_registry, sample_user_contexts):
        """Test complete agent selection workflow"""
        # Test various request types
        test_scenarios = [
            ('finops_request', 'finops'),
            ('github_request', 'github'),
            ('document_request', 'document'),
            ('research_request', 'research'),
            ('deep_research_request', 'deep_research')
        ]
        
        for request_key, expected_domain in test_scenarios:
            user_ctx = sample_user_contexts[request_key]
            
            # Mock the router selection
            with patch.object(phase4_registry.router, 'select') as mock_select:
                # Import the expected agent class dynamically
                if expected_domain == 'finops':
                    from src.agents.finops import AdvancedFinOpsAgent
                    mock_select.return_value = AdvancedFinOpsAgent
                elif expected_domain == 'github':
                    from src.agents.github import AdvancedGitHubAgent
                    mock_select.return_value = AdvancedGitHubAgent
                elif expected_domain == 'document':
                    from src.agents.document import AdvancedDocumentAgent
                    mock_select.return_value = AdvancedDocumentAgent
                elif expected_domain == 'research':
                    from src.agents.research import AdvancedResearchAgent
                    mock_select.return_value = AdvancedResearchAgent
                elif expected_domain == 'deep_research':
                    from src.agents.deep_research import AdvancedDeepResearchAgent
                    mock_select.return_value = AdvancedDeepResearchAgent
                
                agent = await phase4_registry.get_agent(user_context=user_ctx)
                assert agent is not None
                assert agent.get_domain() == expected_domain
    
    @pytest.mark.asyncio
    async def test_agent_caching_behavior(self, phase4_registry, sample_user_contexts):
        """Test agent instance caching"""
        user_ctx = sample_user_contexts['finops_request']
        
        with patch.object(phase4_registry.router, 'select') as mock_select:
            from src.agents.finops import AdvancedFinOpsAgent
            mock_select.return_value = AdvancedFinOpsAgent
            
            # First request - should create new instance
            agent1 = await phase4_registry.get_agent(
                domain='finops',
                user_context=user_ctx
            )
            
            # Second request - should return cached instance
            agent2 = await phase4_registry.get_agent(
                domain='finops',
                user_context=user_ctx
            )
            
            # Should be the same instance
            assert agent1 is agent2
    
    @pytest.mark.asyncio
    async def test_routing_performance(self, phase4_registry, sample_user_contexts):
        """Test routing performance with multiple requests"""
        requests = [
            sample_user_contexts['finops_request'],
            sample_user_contexts['github_request'],
            sample_user_contexts['document_request']
        ]
        
        start_time = datetime.now()
        
        # Process multiple requests
        for user_ctx in requests:
            with patch.object(phase4_registry.router, 'select') as mock_select:
                from src.agents.finops import AdvancedFinOpsAgent
                mock_select.return_value = AdvancedFinOpsAgent
                
                await phase4_registry.get_agent(user_context=user_ctx)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete quickly even with multiple requests
        assert processing_time < 5.0
    
    def test_error_recovery(self, phase4_registry):
        """Test error recovery in routing"""
        # Test with invalid domain
        with pytest.raises(ValueError):
            phase4_registry.router.register_agent(
                domain='invalid',
                agent_class=object,  # Invalid agent class
                capabilities={}
            )