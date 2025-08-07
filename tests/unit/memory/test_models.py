"""
Tests for Memory System models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.memory.models import (
    FinOpsMemoryCategory,
    MemoryPriority,
    ConversationContext,
    MemorySearchFilters,
    FinOpsMemoryRecord,
    MemoryStats
)


@pytest.mark.unit
@pytest.mark.memory
class TestFinOpsMemoryCategory:
    """Test FinOpsMemoryCategory enum."""
    
    def test_finops_memory_categories(self):
        """Test all FinOps memory categories."""
        categories = list(FinOpsMemoryCategory)
        
        expected_categories = [
            "cost_analysis",
            "budget_tracking", 
            "resource_optimization",
            "billing_insights",
            "compliance_governance",
            "performance_metrics",
            "vendor_management",
            "forecasting"
        ]
        
        assert len(categories) == len(expected_categories)
        
        for expected in expected_categories:
            assert any(cat.value == expected for cat in categories)
    
    def test_category_values(self):
        """Test specific category values."""
        assert FinOpsMemoryCategory.COST_ANALYSIS.value == "cost_analysis"
        assert FinOpsMemoryCategory.BUDGET_TRACKING.value == "budget_tracking"
        assert FinOpsMemoryCategory.RESOURCE_OPTIMIZATION.value == "resource_optimization"


@pytest.mark.unit
@pytest.mark.memory  
class TestMemoryPriority:
    """Test MemoryPriority enum."""
    
    def test_memory_priorities(self):
        """Test all memory priority levels."""
        priorities = list(MemoryPriority)
        expected_priorities = ["low", "medium", "high", "critical"]
        
        assert len(priorities) == len(expected_priorities)
        
        for expected in expected_priorities:
            assert any(p.value == expected for p in priorities)
    
    def test_priority_values(self):
        """Test specific priority values."""
        assert MemoryPriority.LOW.value == "low"
        assert MemoryPriority.MEDIUM.value == "medium"
        assert MemoryPriority.HIGH.value == "high"
        assert MemoryPriority.CRITICAL.value == "critical"


@pytest.mark.unit
@pytest.mark.memory
class TestConversationContext:
    """Test ConversationContext model."""
    
    def test_conversation_context_minimal(self):
        """Test ConversationContext with minimal required fields."""
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv"
        )
        
        assert context.user_id == "test_user"
        assert context.conversation_id == "test_conv"
        assert context.agent_id is None
        assert context.category is None
        assert context.priority == MemoryPriority.MEDIUM
        assert context.metadata == {}
        assert context.relevant_memories == []
        assert context.session_data == {}
    
    def test_conversation_context_complete(self):
        """Test ConversationContext with all fields."""
        metadata = {"source": "web", "client_version": "1.0"}
        memories = [{"id": "mem1", "content": "Previous memory"}]
        session_data = {"preference": "detailed_reports"}
        
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            agent_id="finops_agent",
            category=FinOpsMemoryCategory.COST_ANALYSIS,
            priority=MemoryPriority.HIGH,
            metadata=metadata,
            relevant_memories=memories,
            session_data=session_data
        )
        
        assert context.user_id == "test_user"
        assert context.conversation_id == "test_conv"
        assert context.agent_id == "finops_agent"
        assert context.category == FinOpsMemoryCategory.COST_ANALYSIS
        assert context.priority == MemoryPriority.HIGH
        assert context.metadata == metadata
        assert context.relevant_memories == memories
        assert context.session_data == session_data
    
    def test_conversation_context_validation(self):
        """Test ConversationContext validation."""
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            ConversationContext()  # Missing user_id and conversation_id
        
        with pytest.raises(ValidationError):
            ConversationContext(user_id="test_user")  # Missing conversation_id


@pytest.mark.unit
@pytest.mark.memory
class TestMemorySearchFilters:
    """Test MemorySearchFilters model."""
    
    def test_memory_search_filters_defaults(self):
        """Test MemorySearchFilters default values."""
        filters = MemorySearchFilters()
        
        assert filters.category is None
        assert filters.priority is None
        assert filters.date_from is None
        assert filters.date_to is None
        assert filters.limit == 10
        assert filters.include_metadata is True
    
    def test_memory_search_filters_complete(self):
        """Test MemorySearchFilters with all fields."""
        date_from = datetime(2024, 1, 1)
        date_to = datetime(2024, 1, 31)
        
        filters = MemorySearchFilters(
            category=FinOpsMemoryCategory.BUDGET_TRACKING,
            priority=MemoryPriority.HIGH,
            date_from=date_from,
            date_to=date_to,
            limit=25,
            include_metadata=False
        )
        
        assert filters.category == FinOpsMemoryCategory.BUDGET_TRACKING
        assert filters.priority == MemoryPriority.HIGH
        assert filters.date_from == date_from
        assert filters.date_to == date_to
        assert filters.limit == 25
        assert filters.include_metadata is False
    
    def test_memory_search_filters_limit_validation(self):
        """Test MemorySearchFilters limit validation."""
        # Valid limits
        filters = MemorySearchFilters(limit=1)
        assert filters.limit == 1
        
        filters = MemorySearchFilters(limit=100)
        assert filters.limit == 100
        
        # Invalid limits should raise ValidationError
        with pytest.raises(ValidationError):
            MemorySearchFilters(limit=0)  # Below minimum
        
        with pytest.raises(ValidationError):
            MemorySearchFilters(limit=101)  # Above maximum


@pytest.mark.unit
@pytest.mark.memory
class TestFinOpsMemoryRecord:
    """Test FinOpsMemoryRecord model."""
    
    def test_finops_memory_record_minimal(self):
        """Test FinOpsMemoryRecord with minimal required fields."""
        record = FinOpsMemoryRecord(
            memory_id="mem_123",
            content="AWS costs increased by 20%",
            user_id="test_user"
        )
        
        assert record.memory_id == "mem_123"
        assert record.content == "AWS costs increased by 20%"
        assert record.user_id == "test_user"
        assert record.agent_id is None
        assert record.category is None
        assert record.priority == MemoryPriority.MEDIUM
        assert record.metadata == {}
        assert record.entities == []
        assert record.relationships == []
        assert record.created_at is None
        assert record.updated_at is None
    
    def test_finops_memory_record_complete(self):
        """Test FinOpsMemoryRecord with all fields."""
        created_at = datetime(2024, 1, 15, 10, 30)
        updated_at = datetime(2024, 1, 15, 11, 0)
        metadata = {"confidence": 0.95, "source": "billing_dashboard"}
        entities = [{"name": "AWS", "type": "vendor"}]
        relationships = [{"source": "AWS", "target": "costs", "type": "AFFECTS"}]
        
        record = FinOpsMemoryRecord(
            memory_id="mem_123",
            content="AWS costs increased by 20%", 
            user_id="test_user",
            agent_id="cost_analyzer",
            category=FinOpsMemoryCategory.COST_ANALYSIS,
            priority=MemoryPriority.CRITICAL,
            metadata=metadata,
            entities=entities,
            relationships=relationships,
            created_at=created_at,
            updated_at=updated_at
        )
        
        assert record.memory_id == "mem_123"
        assert record.content == "AWS costs increased by 20%"
        assert record.user_id == "test_user"
        assert record.agent_id == "cost_analyzer"
        assert record.category == FinOpsMemoryCategory.COST_ANALYSIS
        assert record.priority == MemoryPriority.CRITICAL
        assert record.metadata == metadata
        assert record.entities == entities
        assert record.relationships == relationships
        assert record.created_at == created_at
        assert record.updated_at == updated_at
    
    def test_finops_memory_record_validation(self):
        """Test FinOpsMemoryRecord validation."""
        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            FinOpsMemoryRecord()  # Missing all required fields
        
        with pytest.raises(ValidationError):
            FinOpsMemoryRecord(memory_id="mem_123")  # Missing content and user_id


@pytest.mark.unit
@pytest.mark.memory
class TestMemoryStats:
    """Test MemoryStats model."""
    
    def test_memory_stats_defaults(self):
        """Test MemoryStats default values."""
        stats = MemoryStats()
        
        assert stats.total_memories == 0
        assert stats.memories_by_category == {}
        assert stats.memories_by_priority == {}
        assert stats.graph_entities == 0
        assert stats.graph_relationships == 0
        assert stats.vector_embeddings == 0
        assert stats.last_updated is None
    
    def test_memory_stats_complete(self):
        """Test MemoryStats with all fields."""
        last_updated = datetime(2024, 1, 15, 12, 0)
        memories_by_category = {
            "cost_analysis": 15,
            "budget_tracking": 8,
            "resource_optimization": 12
        }
        memories_by_priority = {
            "low": 5,
            "medium": 20,
            "high": 8,
            "critical": 2
        }
        
        stats = MemoryStats(
            total_memories=35,
            memories_by_category=memories_by_category,
            memories_by_priority=memories_by_priority,
            graph_entities=150,
            graph_relationships=300,
            vector_embeddings=35,
            last_updated=last_updated
        )
        
        assert stats.total_memories == 35
        assert stats.memories_by_category == memories_by_category
        assert stats.memories_by_priority == memories_by_priority
        assert stats.graph_entities == 150
        assert stats.graph_relationships == 300
        assert stats.vector_embeddings == 35
        assert stats.last_updated == last_updated
    
    def test_memory_stats_category_totals(self):
        """Test that category totals make sense."""
        memories_by_category = {
            "cost_analysis": 10,
            "budget_tracking": 5,
            "forecasting": 8
        }
        
        stats = MemoryStats(
            total_memories=23,
            memories_by_category=memories_by_category
        )
        
        category_sum = sum(memories_by_category.values())
        assert stats.total_memories == category_sum


@pytest.mark.unit
@pytest.mark.memory
class TestModelIntegration:
    """Test model integration and relationships."""
    
    def test_conversation_context_with_memory_record(self):
        """Test using ConversationContext to create MemoryRecord."""
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            category=FinOpsMemoryCategory.RESOURCE_OPTIMIZATION,
            priority=MemoryPriority.HIGH
        )
        
        record = FinOpsMemoryRecord(
            memory_id="mem_456",
            content="Consider rightsizing EC2 instances to save 30%",
            user_id=context.user_id,
            category=context.category,
            priority=context.priority,
            metadata={"conversation_id": context.conversation_id}
        )
        
        assert record.user_id == context.user_id
        assert record.category == context.category
        assert record.priority == context.priority
        assert record.metadata["conversation_id"] == context.conversation_id
    
    def test_search_filters_with_context_category(self):
        """Test using search filters that match context category."""
        context = ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            category=FinOpsMemoryCategory.BILLING_INSIGHTS
        )
        
        filters = MemorySearchFilters(
            category=context.category,
            priority=MemoryPriority.HIGH,
            limit=5
        )
        
        assert filters.category == context.category
        assert filters.category == FinOpsMemoryCategory.BILLING_INSIGHTS