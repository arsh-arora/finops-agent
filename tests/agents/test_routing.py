"""
Unit tests for Agent Routing & Selection
"""

import pytest
from unittest.mock import AsyncMock, Mock
from typing import List

from agents.routing.selector import AgentRouter, RoutingDecision
from agents.specialized.finops import FinOpsAgent
from agents.specialized.default import DefaultAgent


class MockLLMClient:
    """Mock LLM client for testing"""
    
    def __init__(self, response: str = None):
        self.response = response or '{"selected_domain": "finops", "confidence_score": 0.8, "reasoning": "Test routing"}'
    
    async def complete(self, **kwargs):
        return self.response


@pytest.fixture
def mock_memory_service():
    """Mock memory service"""
    mock = AsyncMock()
    mock.store_conversation_memory = AsyncMock()
    mock.retrieve_relevant_memories = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def agent_router():
    """Create agent router with mock LLM"""
    llm_client = MockLLMClient()
    return AgentRouter(llm_client=llm_client)


@pytest.fixture
def populated_router(agent_router, mock_memory_service):
    """Router with registered agents"""
    agent_router.register_agent("finops", FinOpsAgent)
    agent_router.register_agent("default", DefaultAgent)
    return agent_router


@pytest.mark.unit
class TestAgentRouter:
    """Test AgentRouter functionality"""
    
    def test_router_initialization(self):
        """Test router initializes correctly"""
        llm_client = MockLLMClient()
        router = AgentRouter(llm_client=llm_client)
        
        assert router._llm_client == llm_client
        assert len(router._registry) == 0
        assert router._routing_stats["total_requests"] == 0
    
    def test_agent_registration(self, agent_router, mock_memory_service):
        """Test agent registration"""
        agent_router.register_agent("finops", FinOpsAgent, capabilities={"cost_analysis": True})
        
        assert "finops" in agent_router._registry
        assert agent_router._registry["finops"] == FinOpsAgent
        assert "finops" in agent_router._agent_capabilities
        
        capabilities = agent_router._agent_capabilities["finops"]
        assert capabilities["class_name"] == "FinOpsAgent"
        assert "cost_analysis" in capabilities["capabilities"]
    
    def test_agent_registration_validation(self, agent_router):
        """Test agent registration validates interface"""
        class InvalidAgent:
            pass  # Missing required methods
        
        with pytest.raises(ValueError, match="must implement get_domain"):
            agent_router.register_agent("invalid", InvalidAgent)
    
    @pytest.mark.asyncio
    async def test_explicit_domain_selection(self, populated_router):
        """Test explicit domain selection"""
        user_ctx = {
            "message": "Analyze my costs",
            "user_id": "user123",
            "request_id": "req123"
        }
        
        selected_class = await populated_router.select("finops", user_ctx)
        
        assert selected_class == FinOpsAgent
        assert populated_router._routing_stats["successful_routes"] == 1
    
    @pytest.mark.asyncio
    async def test_llm_based_routing(self, populated_router):
        """Test LLM-based intelligent routing"""
        user_ctx = {
            "message": "Help me optimize my AWS spending",
            "user_id": "user123",
            "request_id": "req123",
            "conversation_history": []
        }
        
        selected_class = await populated_router.select(None, user_ctx)
        
        assert selected_class == FinOpsAgent
        assert populated_router._routing_stats["successful_routes"] == 1
    
    @pytest.mark.asyncio
    async def test_fallback_to_default(self, agent_router, mock_memory_service):
        """Test fallback to default agent"""
        # Register only default agent
        agent_router.register_agent("default", DefaultAgent)
        
        user_ctx = {
            "message": "Random question",
            "user_id": "user123",
            "request_id": "req123"
        }
        
        selected_class = await agent_router.select(None, user_ctx)
        
        assert selected_class == DefaultAgent
        assert agent_router._routing_stats["fallback_used"] == 1
    
    @pytest.mark.asyncio
    async def test_heuristic_fallback_routing(self, mock_memory_service):
        """Test heuristic routing when LLM unavailable"""
        router = AgentRouter(llm_client=None)  # No LLM client
        router.register_agent("finops", FinOpsAgent)
        router.register_agent("default", DefaultAgent)
        
        user_ctx = {
            "message": "What are my cloud costs this month?",
            "user_id": "user123"
        }
        
        selected_class = await router.select(None, user_ctx)
        
        assert selected_class == FinOpsAgent
    
    @pytest.mark.asyncio
    async def test_llm_parsing_fallback(self, mock_memory_service):
        """Test fallback when LLM response parsing fails"""
        bad_llm_client = MockLLMClient(response="Invalid JSON response")
        router = AgentRouter(llm_client=bad_llm_client)
        router.register_agent("finops", FinOpsAgent)
        router.register_agent("default", DefaultAgent)
        
        user_ctx = {
            "message": "Analyze costs",
            "user_id": "user123"
        }
        
        # Should fallback to heuristic routing
        selected_class = await router.select(None, user_ctx)
        
        # Should still select FinOps based on keyword "costs"
        assert selected_class == FinOpsAgent
    
    @pytest.mark.asyncio
    async def test_no_suitable_agent_error(self, agent_router):
        """Test error when no suitable agent found"""
        user_ctx = {
            "message": "Help me",
            "user_id": "user123"
        }
        
        with pytest.raises(ValueError, match="No suitable agent found"):
            await agent_router.select(None, user_ctx)
    
    def test_routing_stats(self, populated_router):
        """Test routing statistics tracking"""
        stats = populated_router.get_routing_stats()
        
        assert "total_requests" in stats
        assert "successful_routes" in stats
        assert "fallback_used" in stats
        assert "routing_misses" in stats
        assert "miss_rate_percent" in stats
        assert "success_rate_percent" in stats
    
    def test_clear_registry(self, populated_router):
        """Test registry clearing"""
        assert len(populated_router._registry) > 0
        
        populated_router.clear_registry()
        
        assert len(populated_router._registry) == 0
        assert len(populated_router._agent_capabilities) == 0


@pytest.mark.unit
class TestRoutingDecision:
    """Test RoutingDecision data class"""
    
    def test_routing_decision_creation(self):
        """Test creating RoutingDecision"""
        decision = RoutingDecision(
            selected_domain="finops",
            confidence_score=0.85,
            reasoning="Cost analysis keywords detected",
            fallback_used=False,
            analysis_tokens=50
        )
        
        assert decision.selected_domain == "finops"
        assert decision.confidence_score == 0.85
        assert not decision.fallback_used
        assert decision.analysis_tokens == 50


@pytest.mark.integration
class TestAgentRoutingIntegration:
    """Integration tests for agent routing system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_routing_flow(self, mock_memory_service):
        """Test complete routing flow from request to agent selection"""
        # Setup router with real LLM response
        llm_response = '''
        {
            "selected_domain": "finops",
            "confidence_score": 0.9,
            "reasoning": "User is asking about AWS costs which is clearly a FinOps task"
        }
        '''
        
        llm_client = MockLLMClient(response=llm_response)
        router = AgentRouter(llm_client=llm_client)
        
        # Register agents
        router.register_agent("finops", FinOpsAgent)
        router.register_agent("default", DefaultAgent)
        
        # Test routing
        user_ctx = {
            "message": "Can you analyze my AWS costs for the last quarter and suggest optimizations?",
            "user_id": "user123",
            "request_id": "req123",
            "conversation_history": [
                {"role": "user", "content": "I'm concerned about my cloud spending"},
                {"role": "assistant", "content": "I can help you analyze your costs"}
            ]
        }
        
        selected_class = await router.select(None, user_ctx)
        
        assert selected_class == FinOpsAgent
        
        # Verify stats
        stats = router.get_routing_stats()
        assert stats["total_requests"] == 1
        assert stats["successful_routes"] == 1
        assert stats["miss_rate_percent"] == 0.0