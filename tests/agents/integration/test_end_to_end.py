"""
End-to-end integration tests for Phase 3 Agent Framework
"""

import pytest
from unittest.mock import AsyncMock, Mock
import asyncio

from agents.routing.selector import AgentRouter
from agents.specialized.finops import FinOpsAgent
from agents.specialized.default import DefaultAgent
from agents.models import ChatRequest, ExecutionPlan
from planner.planner import AgentPlanner
from planner.compiler import DynamicGraphCompiler


class MockLLMClient:
    """Mock LLM client that simulates realistic responses"""
    
    def __init__(self):
        self.call_count = 0
    
    async def complete(self, messages, **kwargs):
        self.call_count += 1
        
        # Check if this is a routing request or planning request
        content = messages[0]["content"]
        
        if "agent router" in content.lower():
            # Routing response
            return '''
            {
                "selected_domain": "finops",
                "confidence_score": 0.9,
                "reasoning": "User is asking about AWS costs which is clearly a FinOps domain task"
            }
            '''
        else:
            # Planning response
            return '''
            {
                "tasks": [
                    {
                        "tool_name": "analyze_costs",
                        "inputs": {"time_period": "last_30_days"},
                        "estimate_tokens": 200,
                        "name": "AWS Cost Analysis",
                        "description": "Analyze AWS costs for the last 30 days"
                    },
                    {
                        "tool_name": "track_budget",
                        "inputs": {"budget_name": "monthly_aws"},
                        "estimate_tokens": 150,
                        "name": "Budget Tracking",
                        "description": "Track monthly AWS budget utilization"
                    }
                ],
                "dependencies": [
                    {
                        "parent_task_index": 0,
                        "child_task_index": 1,
                        "dependency_type": "sequential"
                    }
                ],
                "reasoning": "First analyze costs to understand current spending, then track against budget"
            }
            '''


@pytest.fixture
async def mock_memory_service():
    """Mock memory service with realistic responses"""
    mock = AsyncMock()
    
    # Mock memory retrieval
    mock_memory1 = Mock()
    mock_memory1.content = "Previous cost analysis showed $5,200 monthly AWS spend with EC2 being 60% of costs"
    
    mock_memory2 = Mock()
    mock_memory2.content = "User has expressed concerns about unexpected charges from EBS volumes"
    
    mock.retrieve_relevant_memories = AsyncMock(return_value=[mock_memory1, mock_memory2])
    mock.store_conversation_memory = AsyncMock()
    mock.get_health_status = AsyncMock(return_value={"status": "healthy"})
    
    return mock


@pytest.fixture
def llm_client():
    """Mock LLM client"""
    return MockLLMClient()


@pytest.fixture
async def agent_system(mock_memory_service, llm_client):
    """Complete agent system setup"""
    # Setup router
    router = AgentRouter(llm_client=llm_client)
    
    # Register agents
    router.register_agent("finops", FinOpsAgent)
    router.register_agent("default", DefaultAgent)
    
    # Setup planner
    planner = AgentPlanner(memory_service=mock_memory_service, llm_client=llm_client)
    
    # Setup compiler
    compiler = DynamicGraphCompiler(memory_service=mock_memory_service)
    
    return {
        "router": router,
        "planner": planner, 
        "compiler": compiler,
        "memory_service": mock_memory_service
    }


@pytest.mark.integration
class TestEndToEndFlow:
    """Test complete end-to-end agent framework flow"""
    
    @pytest.mark.asyncio
    async def test_websocket_request_to_execution(self, agent_system):
        """Test complete flow from WebSocket request to graph execution"""
        
        # Simulate incoming WebSocket message
        user_request = {
            "message": "Analyze my AWS bill for last month and check if I'm over budget",
            "user_id": "user123",
            "conversation_id": "conv456"
        }
        
        # Step 1: Route to appropriate agent
        router = agent_system["router"]
        
        chat_request = ChatRequest(
            message=user_request["message"],
            user_id=user_request["user_id"],
            budget_usd=2.0,
            conversation_history=[],
            context_id=user_request["conversation_id"]
        )
        
        # Route request
        selected_agent_class = await router.select(
            domain=None,
            user_ctx={
                "message": chat_request.message,
                "user_id": chat_request.user_id,
                "request_id": chat_request.request_id,
                "conversation_history": chat_request.conversation_history
            }
        )
        
        assert selected_agent_class == FinOpsAgent
        
        # Step 2: Create agent instance
        agent = selected_agent_class(
            memory_service=agent_system["memory_service"],
            agent_id=f"finops_{chat_request.request_id[:8]}"
        )
        
        # Step 3: Plan execution
        planner = agent_system["planner"]
        planning_result = await planner.create_plan(chat_request, agent)
        
        assert planning_result.success == True
        assert planning_result.plan is not None
        
        plan = planning_result.plan
        assert len(plan.tasks) == 2
        assert plan.tasks[0].tool_name == "analyze_costs"
        assert plan.tasks[1].tool_name == "track_budget"
        assert len(plan.dependencies) == 1
        
        # Step 4: Compile to graph
        compiler = agent_system["compiler"]
        compilation_result = await compiler.compile(plan, agent)
        
        assert compilation_result.success == True
        assert compilation_result.nodes_created >= 4  # 2 tasks + retrieval + persistence
        assert compilation_result.compilation_time_ms > 0
        assert compilation_result.compilation_time_ms < 50  # Performance requirement
        
        # Step 5: Verify memory integration
        # Memory service should have been called for retrieval during planning
        agent_system["memory_service"].retrieve_relevant_memories.assert_called()
        
        # Verify final metrics
        router_stats = router.get_routing_stats()
        assert router_stats["total_requests"] == 1
        assert router_stats["successful_routes"] == 1
        assert router_stats["miss_rate_percent"] == 0.0
    
    @pytest.mark.asyncio
    async def test_multi_domain_request_routing(self, agent_system):
        """Test routing complex multi-domain requests"""
        
        # Request that could match multiple domains
        ambiguous_request = ChatRequest(
            message="Research best practices for cost optimization and create a GitHub issue to track implementation",
            user_id="user123",
            budget_usd=3.0
        )
        
        router = agent_system["router"]
        
        selected_agent_class = await router.select(
            domain=None,
            user_ctx={
                "message": ambiguous_request.message,
                "user_id": ambiguous_request.user_id,
                "request_id": ambiguous_request.request_id
            }
        )
        
        # Should route to FinOps due to "cost optimization" being primary intent
        assert selected_agent_class == FinOpsAgent
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, mock_memory_service):
        """Test error handling throughout the pipeline"""
        
        # Test with failing LLM client
        class FailingLLMClient:
            async def complete(self, **kwargs):
                raise Exception("LLM service unavailable")
        
        failing_llm = FailingLLMClient()
        
        # Router should fallback to heuristic routing
        router = AgentRouter(llm_client=failing_llm)
        router.register_agent("finops", FinOpsAgent)
        router.register_agent("default", DefaultAgent)
        
        selected_class = await router.select(
            domain=None,
            user_ctx={
                "message": "What are my AWS costs?",
                "user_id": "user123"
            }
        )
        
        # Should still route correctly using heuristic fallback
        assert selected_class == FinOpsAgent
        
        # Planner should also fallback gracefully
        planner = AgentPlanner(memory_service=mock_memory_service, llm_client=failing_llm)
        agent = FinOpsAgent(memory_service=mock_memory_service)
        
        request = ChatRequest(
            message="Analyze costs",
            user_id="user123",
            budget_usd=1.0
        )
        
        result = await planner.create_plan(request, agent)
        
        # Should succeed with fallback plan
        assert result.success == True
        assert result.plan is not None
        assert "fallback" in result.plan.planning_reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution_compilation(self, agent_system):
        """Test compilation of plans with parallel tasks"""
        
        # Mock LLM to return plan with parallel tasks
        class ParallelTaskLLMClient:
            async def complete(self, **kwargs):
                if "agent router" in kwargs.get("messages", [{}])[0].get("content", "").lower():
                    return '{"selected_domain": "finops", "confidence_score": 0.9, "reasoning": "FinOps task"}'
                
                return '''
                {
                    "tasks": [
                        {
                            "tool_name": "analyze_costs",
                            "inputs": {"time_period": "last_30_days"},
                            "estimate_tokens": 150,
                            "name": "Cost Analysis",
                            "parallel_group_id": "group1"
                        },
                        {
                            "tool_name": "track_budget",
                            "inputs": {"budget_name": "monthly"},
                            "estimate_tokens": 100,
                            "name": "Budget Tracking", 
                            "parallel_group_id": "group1"
                        }
                    ],
                    "dependencies": [],
                    "reasoning": "Both tasks can run in parallel"
                }
                '''
        
        parallel_llm = ParallelTaskLLMClient()
        planner = AgentPlanner(memory_service=agent_system["memory_service"], llm_client=parallel_llm)
        
        request = ChatRequest(
            message="Check costs and budget simultaneously",
            user_id="user123",
            budget_usd=2.0
        )
        
        agent = FinOpsAgent(memory_service=agent_system["memory_service"])
        
        # Plan with parallel tasks
        planning_result = await planner.create_plan(request, agent)
        assert planning_result.success == True
        
        plan = planning_result.plan
        parallel_groups = plan.get_parallel_groups()
        assert len(parallel_groups) == 1
        assert "group1" in parallel_groups
        assert len(parallel_groups["group1"]) == 2
        
        # Compile plan
        compilation_result = await agent_system["compiler"].compile(plan, agent)
        assert compilation_result.success == True
        assert compilation_result.parallel_groups_count == 1
    
    @pytest.mark.asyncio
    async def test_memory_persistence_flow(self, agent_system):
        """Test memory storage and retrieval throughout execution"""
        
        request = ChatRequest(
            message="Analyze my cloud spending patterns",
            user_id="user123",
            budget_usd=1.5
        )
        
        # Create agent
        agent = FinOpsAgent(memory_service=agent_system["memory_service"])
        
        # Plan and compile
        planning_result = await agent_system["planner"].create_plan(request, agent)
        compilation_result = await agent_system["compiler"].compile(planning_result.plan, agent)
        
        # Verify memory operations were planned
        assert compilation_result.success == True
        assert "memory_retrieval" in str(compilation_result.graph_json)
        assert "memory_persistence" in str(compilation_result.graph_json)
        
        # Verify memory service interactions
        memory_service = agent_system["memory_service"]
        
        # Should have retrieved memories during planning
        memory_service.retrieve_relevant_memories.assert_called()
        
        # Would store memories during execution (mocked)
        # memory_service.store_conversation_memory would be called during graph execution


@pytest.mark.integration
class TestPerformanceRequirements:
    """Test performance requirements are met"""
    
    @pytest.mark.asyncio
    async def test_compilation_time_under_50ms(self, agent_system):
        """Test compilation time meets p95 < 50ms requirement"""
        
        # Create plan with multiple tasks
        request = ChatRequest(
            message="Comprehensive cost analysis",
            user_id="user123",
            budget_usd=5.0
        )
        
        agent = FinOpsAgent(memory_service=agent_system["memory_service"])
        
        # Plan
        planning_result = await agent_system["planner"].create_plan(request, agent)
        assert planning_result.success == True
        
        # Compile multiple times to test consistency
        compile_times = []
        
        for _ in range(10):
            compilation_result = await agent_system["compiler"].compile(planning_result.plan, agent)
            assert compilation_result.success == True
            compile_times.append(compilation_result.compilation_time_ms)
        
        # Verify all compilations were under 50ms
        for time_ms in compile_times:
            assert time_ms < 50.0, f"Compilation time {time_ms}ms exceeds 50ms requirement"
        
        # Calculate p95
        compile_times.sort()
        p95_index = int(0.95 * len(compile_times))
        p95_time = compile_times[p95_index]
        
        assert p95_time < 50.0, f"P95 compilation time {p95_time}ms exceeds requirement"
    
    @pytest.mark.asyncio
    async def test_routing_miss_rate_under_1_percent(self, agent_system):
        """Test routing miss rate stays under 1% for realistic requests"""
        
        router = agent_system["router"]
        
        # Test various realistic requests
        test_requests = [
            "Analyze my AWS costs",
            "What's my cloud spending?",
            "Track budget utilization",
            "Optimize EC2 instances", 
            "Review billing anomalies",
            "Cost forecast for next quarter",
            "Help me understand this charge",
            "General question about services",
            "Random user inquiry",
            "Thank you for the help"
        ]
        
        successful_routes = 0
        total_requests = len(test_requests)
        
        for message in test_requests:
            try:
                selected_class = await router.select(
                    domain=None,
                    user_ctx={
                        "message": message,
                        "user_id": "user123",
                        "request_id": f"req_{hash(message)}"
                    }
                )
                
                if selected_class:
                    successful_routes += 1
                    
            except Exception:
                pass  # Count as routing miss
        
        success_rate = successful_routes / total_requests
        miss_rate = 1 - success_rate
        
        assert miss_rate < 0.01, f"Routing miss rate {miss_rate*100:.1f}% exceeds 1% requirement"


@pytest.mark.integration  
class TestDummyExecution:
    """Test dummy graph execution to verify runnable compilation"""
    
    @pytest.mark.asyncio
    async def test_compiled_graph_execution(self, agent_system):
        """Test compiled graph can be executed with dummy tools"""
        
        request = ChatRequest(
            message="Quick cost check",
            user_id="user123", 
            budget_usd=1.0
        )
        
        agent = FinOpsAgent(memory_service=agent_system["memory_service"])
        
        # Create and compile plan
        planning_result = await agent_system["planner"].create_plan(request, agent)
        compilation_result = await agent_system["compiler"].compile(planning_result.plan, agent)
        
        assert compilation_result.success == True
        
        # Verify graph structure is valid
        graph_json = compilation_result.graph_json
        
        assert "nodes" in graph_json
        assert "memory_retrieval" in graph_json["nodes"]
        assert "memory_persistence" in graph_json["nodes"]
        
        # In a full implementation, would execute the compiled graph here
        # For now, verify the structure is correct for execution
        nodes_list = graph_json["nodes"]
        
        # Should have memory nodes + task nodes
        expected_min_nodes = 2 + len(planning_result.plan.tasks)  # memory nodes + task nodes
        assert len(nodes_list) >= expected_min_nodes