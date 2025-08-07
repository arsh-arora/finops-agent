"""
Unit tests for AgentPlanner
"""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime

from agents.models import ChatRequest, ExecutionPlan, Task, CostEstimate, PlanningResult
from planner.planner import AgentPlanner
from planner.models import CostModel, PlanningConfig
from planner.exceptions import BudgetExceededError, InvalidRequestError
from agents.specialized.finops import FinOpsAgent


class MockLLMClient:
    """Mock LLM client for testing"""
    
    def __init__(self, response: str = None):
        self.response = response or '''
        {
            "tasks": [
                {
                    "tool_name": "analyze_costs",
                    "inputs": {"time_period": "last_30_days"},
                    "estimate_tokens": 200,
                    "name": "Cost analysis",
                    "description": "Analyze cloud costs"
                }
            ],
            "dependencies": [],
            "reasoning": "User wants cost analysis"
        }
        '''
    
    async def complete(self, **kwargs):
        return self.response


@pytest.fixture
def mock_memory_service():
    """Mock memory service"""
    mock = AsyncMock()
    mock.retrieve_relevant_memories = AsyncMock(return_value=[
        Mock(content="Previous cost analysis showed $5000/month spend"),
        Mock(content="User is concerned about EC2 costs")
    ])
    return mock


@pytest.fixture
def mock_agent(mock_memory_service):
    """Mock FinOps agent"""
    agent = FinOpsAgent(memory_service=mock_memory_service, agent_id="test_agent")
    return agent


@pytest.fixture
def chat_request():
    """Sample chat request"""
    return ChatRequest(
        message="Analyze my AWS costs for last month",
        user_id="user123",
        budget_usd=1.0,
        request_id="req123"
    )


@pytest.fixture
def planner(mock_memory_service):
    """Agent planner with mock dependencies"""
    llm_client = MockLLMClient()
    config = PlanningConfig()
    cost_model = CostModel()
    
    return AgentPlanner(
        memory_service=mock_memory_service,
        llm_client=llm_client,
        config=config,
        cost_model=cost_model
    )


@pytest.mark.unit
class TestAgentPlanner:
    """Test AgentPlanner functionality"""
    
    def test_planner_initialization(self, mock_memory_service):
        """Test planner initializes correctly"""
        llm_client = MockLLMClient()
        planner = AgentPlanner(memory_service=mock_memory_service, llm_client=llm_client)
        
        assert planner.memory_service == mock_memory_service
        assert planner.llm_client == llm_client
        assert planner.config is not None
        assert planner.cost_model is not None
    
    @pytest.mark.asyncio
    async def test_create_plan_success(self, planner, chat_request, mock_agent):
        """Test successful plan creation"""
        result = await planner.create_plan(chat_request, mock_agent)
        
        assert result.success == True
        assert result.plan is not None
        assert len(result.plan.tasks) == 1
        assert result.plan.tasks[0].tool_name == "analyze_costs"
        assert result.plan.request_id == "req123"
        assert result.plan.user_id == "user123"
        assert result.planning_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_create_plan_with_dependencies(self, mock_memory_service, mock_agent):
        """Test plan creation with task dependencies"""
        llm_response = '''
        {
            "tasks": [
                {
                    "tool_name": "analyze_costs",
                    "inputs": {"time_period": "last_30_days"},
                    "estimate_tokens": 150,
                    "name": "Analyze costs"
                },
                {
                    "tool_name": "track_budget",
                    "inputs": {"budget_name": "monthly"},
                    "estimate_tokens": 100,
                    "name": "Track budget"
                }
            ],
            "dependencies": [
                {
                    "parent_task_index": 0,
                    "child_task_index": 1,
                    "dependency_type": "sequential"
                }
            ],
            "reasoning": "Need to analyze costs before tracking budget"
        }
        '''
        
        llm_client = MockLLMClient(response=llm_response)
        planner = AgentPlanner(memory_service=mock_memory_service, llm_client=llm_client)
        
        request = ChatRequest(
            message="Analyze costs and track budget",
            user_id="user123",
            budget_usd=2.0
        )
        
        result = await planner.create_plan(request, mock_agent)
        
        assert result.success == True
        assert len(result.plan.tasks) == 2
        assert len(result.plan.dependencies) == 1
        
        dep = result.plan.dependencies[0]
        assert dep.parent_id == result.plan.tasks[0].id
        assert dep.child_id == result.plan.tasks[1].id
    
    @pytest.mark.asyncio
    async def test_budget_exceeded_error(self, planner, mock_agent):
        """Test budget exceeded validation"""
        low_budget_request = ChatRequest(
            message="Analyze costs",
            user_id="user123",
            budget_usd=0.001  # Very low budget
        )
        
        result = await planner.create_plan(low_budget_request, mock_agent)
        
        assert result.success == False
        assert "budget" in result.error_message.lower() or "cost" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_request_validation(self, planner, mock_agent):
        """Test request validation"""
        invalid_request = ChatRequest(
            message="",  # Empty message
            user_id="user123",
            budget_usd=1.0
        )
        
        result = await planner.create_plan(invalid_request, mock_agent)
        
        assert result.success == False
        assert "validation" in result.error_message.lower() or "empty" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_fallback_plan_generation(self, mock_memory_service, mock_agent):
        """Test fallback when LLM is unavailable"""
        planner = AgentPlanner(memory_service=mock_memory_service, llm_client=None)
        
        request = ChatRequest(
            message="Analyze my costs",
            user_id="user123",
            budget_usd=1.0
        )
        
        result = await planner.create_plan(request, mock_agent)
        
        assert result.success == True
        assert result.plan is not None
        assert len(result.plan.tasks) == 1
        assert "fallback" in result.plan.planning_reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_memory_context_integration(self, planner, chat_request, mock_agent):
        """Test memory context is retrieved and used"""
        result = await planner.create_plan(chat_request, mock_agent)
        
        # Verify memory service was called
        planner.memory_service.retrieve_relevant_memories.assert_called_once()
        
        # Should have memory context in result
        assert result.memory_queries > 0
    
    @pytest.mark.asyncio
    async def test_cost_calculation(self, planner, chat_request, mock_agent):
        """Test cost estimation"""
        result = await planner.create_plan(chat_request, mock_agent)
        
        assert result.success == True
        assert result.plan.cost.usd > 0
        assert result.plan.cost.tokens > 0
        assert result.plan.cost.confidence > 0
        assert "breakdown" in result.plan.cost.breakdown
    
    @pytest.mark.asyncio
    async def test_llm_parsing_error_handling(self, mock_memory_service, mock_agent):
        """Test handling of LLM parsing errors"""
        bad_llm_client = MockLLMClient(response="Invalid JSON response")
        planner = AgentPlanner(memory_service=mock_memory_service, llm_client=bad_llm_client)
        
        request = ChatRequest(
            message="Analyze costs",
            user_id="user123",
            budget_usd=1.0
        )
        
        result = await planner.create_plan(request, mock_agent)
        
        # Should fallback to simple plan
        assert result.success == True
        assert result.plan is not None


@pytest.mark.unit
class TestCostModel:
    """Test cost calculation model"""
    
    def test_llm_cost_calculation(self):
        """Test LLM cost calculation"""
        cost_model = CostModel()
        
        cost = cost_model.calculate_llm_cost(input_tokens=1000, output_tokens=500)
        
        assert cost > 0
        # Should be input_cost + output_cost
        expected = (1000 / 1000) * 0.00015 + (500 / 1000) * 0.0006
        assert abs(cost - expected) < 0.0001
    
    def test_memory_cost_calculation(self):
        """Test memory system cost calculation"""
        cost_model = CostModel()
        
        cost = cost_model.calculate_memory_cost(retrievals=3, hops=2, stores=1)
        
        assert cost > 0
        expected = (3 * 0.0005) + (2 * 0.0001) + (1 * 0.0002)
        assert abs(cost - expected) < 0.0001
    
    def test_tool_cost_lookup(self):
        """Test tool-specific cost lookup"""
        cost_model = CostModel()
        
        # Should return specific cost for known tools
        assert cost_model.get_tool_cost("analyze_costs") == 0.005
        
        # Should return default for unknown tools
        assert cost_model.get_tool_cost("unknown_tool") == 0.001


@pytest.mark.unit
class TestPlanningConfig:
    """Test planning configuration"""
    
    def test_config_defaults(self):
        """Test configuration has reasonable defaults"""
        config = PlanningConfig()
        
        assert config.planning_model == "gpt-4o-mini"
        assert 0 <= config.planning_temperature <= 2.0
        assert config.max_tasks_per_plan >= 1
        assert config.max_budget_usd > 0
        assert config.memory_retrieval_limit >= 1


@pytest.mark.integration
class TestPlannerIntegration:
    """Integration tests for planner system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_planning_flow(self, mock_memory_service):
        """Test complete planning flow"""
        # Setup realistic LLM response
        llm_response = '''
        {
            "tasks": [
                {
                    "tool_name": "analyze_costs",
                    "inputs": {"time_period": "last_30_days"},
                    "estimate_tokens": 250,
                    "name": "Monthly cost analysis",
                    "description": "Analyze AWS costs for the last 30 days",
                    "parallel_group_id": null
                },
                {
                    "tool_name": "track_budget", 
                    "inputs": {"budget_name": "monthly_aws"},
                    "estimate_tokens": 150,
                    "name": "Budget tracking",
                    "description": "Track monthly AWS budget utilization"
                }
            ],
            "dependencies": [],
            "reasoning": "User wants comprehensive FinOps analysis including both cost analysis and budget tracking"
        }
        '''
        
        llm_client = MockLLMClient(response=llm_response)
        planner = AgentPlanner(memory_service=mock_memory_service, llm_client=llm_client)
        
        # Create agent
        agent = FinOpsAgent(memory_service=mock_memory_service, agent_id="finops_agent")
        
        # Create request
        request = ChatRequest(
            message="I need a comprehensive analysis of my AWS costs and budget status",
            user_id="user123",
            budget_usd=5.0,
            conversation_history=[
                {"role": "user", "content": "I'm worried about my cloud spending"},
                {"role": "assistant", "content": "I can help analyze your costs"}
            ]
        )
        
        # Execute planning
        result = await planner.create_plan(request, agent)
        
        # Verify success
        assert result.success == True
        assert result.plan is not None
        
        # Verify plan structure
        plan = result.plan
        assert len(plan.tasks) == 2
        assert plan.tasks[0].tool_name == "analyze_costs"
        assert plan.tasks[1].tool_name == "track_budget"
        
        # Verify cost estimation
        assert plan.cost.usd > 0
        assert plan.cost.tokens > 0
        assert plan.cost.usd < request.budget_usd  # Should be under budget
        
        # Verify metadata
        assert plan.request_id == request.request_id
        assert plan.user_id == request.user_id
        assert plan.original_message == request.message
        assert len(plan.planning_reasoning) > 0
        
        # Verify metrics
        assert result.planning_time_ms > 0
        assert result.memory_queries > 0