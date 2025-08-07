"""
Unit tests for HardenedAgent base class
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from agents.base.agent import HardenedAgent
from agents.base.registry import tool
from agents.base.exceptions import AgentError, ExecutionError


class TestAgent(HardenedAgent):
    """Test implementation of HardenedAgent"""
    
    def get_capabilities(self) -> List[str]:
        return ["test_capability", "mock_processing"]
    
    def get_domain(self) -> str:
        return "test"
    
    @tool(description="Test tool for unit testing")
    async def test_tool(self, input_text: str) -> str:
        return f"Processed: {input_text}"
    
    async def _process_message(self, message: str, memory_context: List[str], plan: dict) -> str:
        return f"Test response to: {message}"


@pytest.fixture
def mock_memory_service():
    """Mock memory service for testing"""
    mock = AsyncMock()
    mock.store_conversation_memory = AsyncMock()
    mock.retrieve_relevant_memories = AsyncMock(return_value=[])
    return mock


@pytest.fixture
def test_agent(mock_memory_service):
    """Create test agent instance"""
    return TestAgent(memory_service=mock_memory_service, agent_id="test_agent")


@pytest.mark.unit
class TestHardenedAgent:
    """Test HardenedAgent functionality"""
    
    def test_agent_initialization(self, mock_memory_service):
        """Test agent initializes correctly"""
        agent = TestAgent(memory_service=mock_memory_service, agent_id="test_id")
        
        assert agent.agent_id == "test_id"
        assert agent.memory_service == mock_memory_service
        assert agent._initialized == True
        assert len(agent._tools) == 1  # test_tool should be extracted
        assert "test_tool" in agent._tools
    
    def test_agent_auto_id_generation(self, mock_memory_service):
        """Test agent generates ID when not provided"""
        agent = TestAgent(memory_service=mock_memory_service)
        
        assert agent.agent_id is not None
        assert agent.agent_id.startswith("TestAgent_")
    
    def test_get_capabilities(self, test_agent):
        """Test capabilities are returned correctly"""
        capabilities = test_agent.get_capabilities()
        assert capabilities == ["test_capability", "mock_processing"]
    
    def test_get_domain(self, test_agent):
        """Test domain is returned correctly"""
        domain = test_agent.get_domain()
        assert domain == "test"
    
    def test_get_tools(self, test_agent):
        """Test tools are extracted correctly"""
        tools = test_agent.get_tools()
        assert len(tools) == 1
        assert "test_tool" in tools
        assert callable(tools["test_tool"])
    
    def test_get_tool_schema(self, test_agent):
        """Test tool schema generation"""
        schema = test_agent.get_tool_schema()
        
        assert "tools_definitions" in schema
        assert "test_tool" in schema["tools_definitions"]
        assert schema["tools_definitions"]["test_tool"]["name"] == "test_tool"
        assert "description" in schema["tools_definitions"]["test_tool"]
    
    @pytest.mark.asyncio
    async def test_send_basic_flow(self, test_agent):
        """Test basic send() flow"""
        response = await test_agent.send(
            message="test message",
            user_id="user123"
        )
        
        assert response == "Test response to: test message"
        
        # Verify memory service calls
        test_agent.memory_service.store_conversation_memory.assert_called()
        test_agent.memory_service.retrieve_relevant_memories.assert_called()
    
    @pytest.mark.asyncio
    async def test_plan_creation(self, test_agent):
        """Test plan() method"""
        request = {
            "message": "test message",
            "user_id": "user123",
            "request_id": "req123"
        }
        
        plan = await test_agent.plan(request)
        
        assert plan["request_id"] == "req123"
        assert plan["agent_id"] == "test_agent"
        assert plan["user_id"] == "user123"
        assert plan["message"] == "test message"
        assert plan["capabilities_required"] == ["test_capability", "mock_processing"]
        assert "test_tool" in plan["tools_available"]
    
    @pytest.mark.asyncio
    async def test_execute_with_memory_context(self, test_agent):
        """Test execute() method with memory integration"""
        mock_memories = [Mock(content="Previous conversation")]
        test_agent.memory_service.retrieve_relevant_memories.return_value = mock_memories
        
        plan = {
            "id": "plan123",
            "request_id": "req123",
            "message": "test message",
            "user_id": "user123"
        }
        
        result = await test_agent.execute(plan)
        
        assert result["plan_id"] == "plan123"
        assert result["request_id"] == "req123"
        assert result["agent_id"] == "test_agent"
        assert result["response"] == "Test response to: test message"
        assert result["memory_context_used"] == 1
    
    @pytest.mark.asyncio
    async def test_persist_results(self, test_agent):
        """Test persist() method"""
        result = {
            "request_id": "req123",
            "response": "test response",
            "plan_id": "plan123"
        }
        
        await test_agent.persist(result, "user123")
        
        test_agent.memory_service.store_conversation_memory.assert_called()
        call_args = test_agent.memory_service.store_conversation_memory.call_args
        assert call_args[1]["user_id"] == "user123"
        assert call_args[1]["message"] == "test response"
    
    @pytest.mark.asyncio
    async def test_lifecycle_hooks(self, test_agent):
        """Test lifecycle hooks are called"""
        with patch.object(test_agent, 'before_plan', new_callable=AsyncMock) as mock_before:
            with patch.object(test_agent, 'after_execute', new_callable=AsyncMock) as mock_after:
                mock_before.return_value = {"message": "test", "user_id": "user123"}
                mock_after.return_value = {"response": "modified response"}
                
                await test_agent.send("test message", "user123")
                
                mock_before.assert_called_once()
                mock_after.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_agent):
        """Test error handling and on_error hook"""
        # Mock error in planning
        with patch.object(test_agent, '_process_message', side_effect=Exception("Test error")):
            with patch.object(test_agent, 'on_error', new_callable=AsyncMock) as mock_on_error:
                with pytest.raises(ExecutionError):
                    await test_agent.send("test message", "user123")
                
                mock_on_error.assert_called_once()
                assert "Test error" in str(mock_on_error.call_args[0][0])
    
    def test_register_tool_manually(self, test_agent):
        """Test manual tool registration"""
        def custom_tool():
            return "custom result"
        
        initial_count = len(test_agent.get_tools())
        test_agent.register_tool("custom_tool", custom_tool)
        
        tools = test_agent.get_tools()
        assert len(tools) == initial_count + 1
        assert "custom_tool" in tools
        assert tools["custom_tool"] == custom_tool