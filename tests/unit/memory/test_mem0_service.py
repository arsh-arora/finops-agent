"""
Tests for FinOps Memory Service using Mem0.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from datetime import datetime

from src.memory.mem0_service import FinOpsMemoryService
from src.memory.models import (
    ConversationContext, 
    MemorySearchFilters, 
    FinOpsMemoryCategory,
    MemoryPriority
)
from src.memory.exceptions import (
    MemoryServiceError,
    MemoryStorageError, 
    MemoryRetrievalError
)


@pytest.mark.unit
@pytest.mark.memory
class TestFinOpsMemoryService:
    """Test FinOps Memory Service functionality."""
    
    def test_memory_service_initialization(self):
        """Test MemoryService initialization."""
        service = FinOpsMemoryService()
        
        assert service.memory is None
        assert service.is_initialized is False
        assert service.graph_mode is True
        assert service.vector_mode is True
        assert service.config is not None
    
    def test_memory_service_with_custom_config(self):
        """Test MemoryService with custom configuration."""
        custom_config = {
            "vector_store": {"provider": "qdrant"},
            "graph_store": {"provider": "neo4j"}
        }
        
        service = FinOpsMemoryService(config=custom_config)
        assert service.config == custom_config
    
    @patch('src.memory.mem0_service.Memory')
    async def test_initialize_success(self, mock_memory_class):
        """Test successful memory service initialization."""
        # Mock Mem0 Memory class
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = {"id": "test_memory_id"}
        mock_memory_instance.delete_all.return_value = True
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        service = FinOpsMemoryService()
        result = await service.initialize()
        
        assert result is True
        assert service.is_initialized is True
        assert service.memory is not None
        mock_memory_class.from_config.assert_called_once()
    
    @patch('src.memory.mem0_service.Memory')
    async def test_initialize_fallback_mode(self, mock_memory_class):
        """Test initialization fallback mode when graph store fails."""
        # First call fails, second call (fallback) succeeds
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = {"id": "test_memory_id"}
        mock_memory_instance.delete_all.return_value = True
        
        mock_memory_class.from_config.side_effect = [
            Exception("Neo4j connection failed"),  # First call fails
            mock_memory_instance  # Fallback succeeds
        ]
        
        service = FinOpsMemoryService()
        result = await service.initialize()
        
        assert result is True
        assert service.is_initialized is True
        assert service.graph_mode is False  # Fallback mode
        assert mock_memory_class.from_config.call_count == 2
    
    @patch('src.memory.mem0_service.Memory')
    async def test_initialize_complete_failure(self, mock_memory_class):
        """Test initialization complete failure."""
        mock_memory_class.from_config.side_effect = Exception("Complete failure")
        
        service = FinOpsMemoryService()
        
        with pytest.raises(MemoryServiceError, match="Failed to initialize memory service"):
            await service.initialize()
    
    async def test_store_conversation_memory_not_initialized(self):
        """Test storing memory when service not initialized."""
        service = FinOpsMemoryService()
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv"
        )
        
        with pytest.raises(MemoryServiceError, match="Memory service not initialized"):
            await service.store_conversation_memory("test message", context)
    
    @patch('src.memory.mem0_service.Memory')
    async def test_store_conversation_memory_success(self, mock_memory_class):
        """Test successful conversation memory storage."""
        # Setup mocks
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = {"id": "stored_memory_id"}
        mock_memory_instance.delete_all.return_value = True
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        service = FinOpsMemoryService()
        await service.initialize()
        
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            category=FinOpsMemoryCategory.COST_ANALYSIS,
            priority=MemoryPriority.HIGH
        )
        
        messages = [
            {"role": "user", "content": "What are my AWS costs?"},
            {"role": "assistant", "content": "Let me analyze your AWS costs..."}
        ]
        
        result = await service.store_conversation_memory(messages, context)
        
        assert result == "stored_memory_id"
        mock_memory_instance.add.assert_called_once()
        
        # Verify add was called with correct parameters
        call_args = mock_memory_instance.add.call_args
        assert call_args[1]["user_id"] == "test_user"
        assert call_args[1]["agent_id"] is None
        assert "conversation_id" in call_args[1]["metadata"]
        assert call_args[1]["metadata"]["category"] == "cost_analysis"
    
    @patch('src.memory.mem0_service.Memory')
    async def test_store_conversation_memory_failure(self, mock_memory_class):
        """Test conversation memory storage failure."""
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = None  # Simulate failure
        mock_memory_instance.delete_all.return_value = True
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        service = FinOpsMemoryService()
        await service.initialize()
        
        context = ConversationContext(
            user_id="test_user", 
            conversation_id="test_conv"
        )
        
        with pytest.raises(MemoryStorageError, match="Failed to store memory"):
            await service.store_conversation_memory("test message", context)
    
    @patch('src.memory.mem0_service.Memory')
    async def test_retrieve_relevant_memories_success(self, mock_memory_class):
        """Test successful memory retrieval."""
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = {"id": "test_memory_id"}
        mock_memory_instance.delete_all.return_value = True
        mock_memory_instance.search.return_value = [
            {
                "id": "memory_1",
                "memory": "AWS costs are high this month",
                "score": 0.95,
                "metadata": {"category": "cost_analysis"},
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "id": "memory_2", 
                "memory": "Consider rightsizing EC2 instances",
                "score": 0.87,
                "metadata": {"category": "resource_optimization"},
                "created_at": "2024-01-01T01:00:00Z"
            }
        ]
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        service = FinOpsMemoryService()
        await service.initialize()
        
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv"
        )
        
        memories = await service.retrieve_relevant_memories(
            "high AWS costs", 
            context
        )
        
        assert len(memories) == 2
        assert memories[0]["memory_id"] == "memory_1"
        assert memories[0]["content"] == "AWS costs are high this month"
        assert memories[0]["score"] == 0.95
        assert memories[1]["memory_id"] == "memory_2"
        
        mock_memory_instance.search.assert_called_once()
    
    @patch('src.memory.mem0_service.Memory')
    async def test_retrieve_relevant_memories_with_filters(self, mock_memory_class):
        """Test memory retrieval with search filters."""
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = {"id": "test_memory_id"}
        mock_memory_instance.delete_all.return_value = True
        mock_memory_instance.search.return_value = []
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        service = FinOpsMemoryService()
        await service.initialize()
        
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv"
        )
        
        filters = MemorySearchFilters(
            category=FinOpsMemoryCategory.BUDGET_TRACKING,
            limit=5
        )
        
        await service.retrieve_relevant_memories(
            "budget status",
            context, 
            search_filters=filters
        )
        
        # Verify search was called with correct filters
        call_args = mock_memory_instance.search.call_args
        assert call_args[1]["limit"] == 5
        assert call_args[1]["metadata"]["category"] == "budget_tracking"
    
    @patch('src.memory.mem0_service.Memory')
    async def test_get_user_memories(self, mock_memory_class):
        """Test getting all user memories."""
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = {"id": "test_memory_id"}
        mock_memory_instance.delete_all.return_value = True
        mock_memory_instance.get_all.return_value = [
            {"id": "mem1", "memory": "Memory 1"},
            {"id": "mem2", "memory": "Memory 2"}
        ]
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        service = FinOpsMemoryService()
        await service.initialize()
        
        memories = await service.get_user_memories("test_user")
        
        assert len(memories) == 2
        mock_memory_instance.get_all.assert_called_once_with(
            user_id="test_user",
            agent_id=None
        )
    
    @patch('src.memory.mem0_service.Memory')
    async def test_delete_memory(self, mock_memory_class):
        """Test memory deletion."""
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = {"id": "test_memory_id"}
        mock_memory_instance.delete_all.return_value = True
        mock_memory_instance.delete.return_value = True
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        service = FinOpsMemoryService()
        await service.initialize()
        
        result = await service.delete_memory("memory_to_delete")
        
        assert result is True
        mock_memory_instance.delete.assert_called_once_with(memory_id="memory_to_delete")
    
    def test_get_health_status(self):
        """Test health status reporting."""
        service = FinOpsMemoryService()
        status = service.get_health_status()
        
        assert "initialized" in status
        assert "graph_mode" in status
        assert "vector_mode" in status
        assert "memory_client_available" in status
        assert "config_valid" in status
        assert "last_check" in status
        
        assert status["initialized"] is False
        assert status["graph_mode"] is True
        assert status["vector_mode"] is True
        assert status["memory_client_available"] is False
        assert status["config_valid"] is True
    
    @patch('src.memory.mem0_service.Memory')
    async def test_get_memory_stats(self, mock_memory_class):
        """Test memory statistics generation."""
        mock_memory_instance = AsyncMock()
        mock_memory_instance.add.return_value = {"id": "test_memory_id"}
        mock_memory_instance.delete_all.return_value = True
        mock_memory_instance.get_all.return_value = [
            {"metadata": {"category": "cost_analysis", "priority": "high"}},
            {"metadata": {"category": "cost_analysis", "priority": "medium"}},
            {"metadata": {"category": "budget_tracking", "priority": "high"}}
        ]
        mock_memory_class.from_config.return_value = mock_memory_instance
        
        service = FinOpsMemoryService()
        await service.initialize()
        
        stats = await service.get_memory_stats("test_user")
        
        assert stats.total_memories == 3
        assert stats.memories_by_category["cost_analysis"] == 2
        assert stats.memories_by_category["budget_tracking"] == 1
        assert stats.memories_by_priority["high"] == 2
        assert stats.memories_by_priority["medium"] == 1
        assert stats.last_updated is not None


@pytest.mark.unit
@pytest.mark.memory
class TestMemoryModels:
    """Test memory system models."""
    
    def test_conversation_context_creation(self):
        """Test ConversationContext model creation."""
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            category=FinOpsMemoryCategory.COST_ANALYSIS,
            priority=MemoryPriority.HIGH
        )
        
        assert context.user_id == "test_user"
        assert context.conversation_id == "test_conv"
        assert context.category == FinOpsMemoryCategory.COST_ANALYSIS
        assert context.priority == MemoryPriority.HIGH
        assert context.agent_id is None
        assert context.metadata == {}
    
    def test_memory_search_filters_defaults(self):
        """Test MemorySearchFilters default values."""
        filters = MemorySearchFilters()
        
        assert filters.category is None
        assert filters.priority is None
        assert filters.limit == 10
        assert filters.include_metadata is True
        assert filters.date_from is None
        assert filters.date_to is None
    
    def test_memory_search_filters_validation(self):
        """Test MemorySearchFilters validation."""
        # Test limit boundaries
        filters = MemorySearchFilters(limit=5)
        assert filters.limit == 5
        
        # Test with category and priority
        filters = MemorySearchFilters(
            category=FinOpsMemoryCategory.BUDGET_TRACKING,
            priority=MemoryPriority.CRITICAL,
            limit=20
        )
        
        assert filters.category == FinOpsMemoryCategory.BUDGET_TRACKING
        assert filters.priority == MemoryPriority.CRITICAL
        assert filters.limit == 20