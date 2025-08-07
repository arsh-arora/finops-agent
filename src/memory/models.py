"""
Pydantic models for the Memory System.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class FinOpsMemoryCategory(str, Enum):
    """Categories for FinOps-specific memory organization."""
    COST_ANALYSIS = "cost_analysis"
    BUDGET_TRACKING = "budget_tracking"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    BILLING_INSIGHTS = "billing_insights"
    COMPLIANCE_GOVERNANCE = "compliance_governance"
    PERFORMANCE_METRICS = "performance_metrics"
    VENDOR_MANAGEMENT = "vendor_management"
    FORECASTING = "forecasting"


class MemoryPriority(str, Enum):
    """Priority levels for memory storage and retrieval."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConversationContext(BaseModel):
    """Context information for conversation memory management."""
    user_id: str
    conversation_id: str
    agent_id: Optional[str] = None
    category: Optional[FinOpsMemoryCategory] = None
    priority: MemoryPriority = MemoryPriority.MEDIUM
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevant_memories: List[Dict[str, Any]] = Field(default_factory=list)
    session_data: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = {"use_enum_values": True}


class MemorySearchFilters(BaseModel):
    """Filters for memory search operations."""
    category: Optional[FinOpsMemoryCategory] = None
    priority: Optional[MemoryPriority] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=10, ge=1, le=100)
    include_metadata: bool = True
    
    model_config = {"use_enum_values": True}


class FinOpsMemoryRecord(BaseModel):
    """Structured representation of a FinOps memory record."""
    memory_id: str
    content: str
    user_id: str
    agent_id: Optional[str] = None
    category: Optional[FinOpsMemoryCategory] = None
    priority: MemoryPriority = MemoryPriority.MEDIUM
    metadata: Dict[str, Any] = Field(default_factory=dict)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    model_config = {"use_enum_values": True}


class MemoryStats(BaseModel):
    """Statistics about memory usage and performance."""
    total_memories: int = 0
    memories_by_category: Dict[str, int] = Field(default_factory=dict)
    memories_by_priority: Dict[str, int] = Field(default_factory=dict)
    graph_entities: int = 0
    graph_relationships: int = 0
    vector_embeddings: int = 0
    last_updated: Optional[datetime] = None