"""
Data Classes for Agent Framework - Pydantic v2 Implementation
Typed, serializable objects shared by planner, compiler, and agents
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
from uuid import UUID, uuid4


class TaskStatus(str, Enum):
    """Status of task execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DependencyType(str, Enum):
    """Types of task dependencies"""
    SEQUENTIAL = "sequential"     # Task must complete before dependent starts
    DATA = "data"                # Task output required as input
    RESOURCE = "resource"        # Shared resource constraint


class CostEstimate(BaseModel):
    """Cost estimation for plan execution"""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True
    )
    
    tokens: int = Field(..., ge=0, description="Estimated token usage")
    graph_hops: int = Field(default=0, ge=0, description="Memory graph traversal hops")
    usd: float = Field(..., ge=0, description="Estimated cost in USD")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Estimation confidence")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Cost breakdown by component")
    
    @field_validator('usd')
    @classmethod
    def validate_usd_precision(cls, v: float) -> float:
        """Ensure USD amount has reasonable precision"""
        return round(v, 4)  # 4 decimal places for pricing precision


class Task(BaseModel):
    """Individual task within an execution plan"""
    model_config = ConfigDict(
        json_encoders={UUID: str, datetime: lambda v: v.isoformat()},
        validate_assignment=True
    )
    
    id: UUID = Field(default_factory=uuid4, description="Unique task identifier")
    tool_name: str = Field(..., min_length=1, description="Tool to execute this task")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Tool input parameters")
    estimate_tokens: int = Field(..., ge=0, description="Estimated token usage for this task")
    parallel_group_id: Optional[str] = Field(default=None, description="Parallel execution group identifier")
    
    # Metadata
    name: str = Field(default="", description="Human-readable task name")
    description: str = Field(default="", description="Task description")
    priority: Priority = Field(default=Priority.MEDIUM, description="Task priority")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    
    # Execution context
    timeout_seconds: Optional[int] = Field(default=None, ge=1, description="Task timeout in seconds")
    retry_count: int = Field(default=0, ge=0, le=3, description="Number of retries allowed")
    memory_context: Dict[str, Any] = Field(default_factory=dict, description="Memory context for task")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Task creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Task completion timestamp")
    
    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool name is valid"""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Tool name must be alphanumeric with underscores/hyphens")
        return v


class Dependency(BaseModel):
    """Task dependency relationship"""
    model_config = ConfigDict(
        json_encoders={UUID: str},
        validate_assignment=True
    )
    
    parent_id: UUID = Field(..., description="Parent task that must complete first")
    child_id: UUID = Field(..., description="Dependent task that waits for parent")
    dependency_type: DependencyType = Field(default=DependencyType.SEQUENTIAL, description="Type of dependency")
    data_mapping: Dict[str, str] = Field(default_factory=dict, description="Output->Input data mapping")
    
    @field_validator('child_id')
    @classmethod
    def validate_no_self_dependency(cls, v: UUID, info) -> UUID:
        """Ensure task doesn't depend on itself"""
        if 'parent_id' in info.data and v == info.data['parent_id']:
            raise ValueError("Task cannot depend on itself")
        return v


class ChatRequest(BaseModel):
    """Incoming chat request with context and constraints"""
    model_config = ConfigDict(
        json_encoders={UUID: str, datetime: lambda v: v.isoformat()},
        validate_assignment=True
    )
    
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    context_id: Optional[str] = Field(default=None, description="Conversation context identifier")  
    user_id: str = Field(..., min_length=1, description="User identifier")
    budget_usd: float = Field(default=1.0, ge=0, le=100, description="Maximum spend allowed for request")
    
    # Optional context
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Previous messages")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences and settings")
    domain_hint: Optional[str] = Field(default=None, description="Suggested domain for routing")
    
    # Request metadata
    request_id: str = Field(default_factory=lambda: uuid4().hex, description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    source: str = Field(default="websocket", description="Request source (websocket, api, etc)")
    
    @field_validator('budget_usd')
    @classmethod
    def validate_budget_precision(cls, v: float) -> float:
        """Ensure budget has reasonable precision"""
        return round(v, 2)  # 2 decimal places for USD


class ExecutionPlan(BaseModel):
    """Complete execution plan with tasks, dependencies, and cost estimate"""
    model_config = ConfigDict(
        json_encoders={UUID: str, datetime: lambda v: v.isoformat()},
        validate_assignment=True,
        schema_extra={
            "example": {
                "tasks": [
                    {
                        "id": "12345678-1234-1234-1234-123456789abc",
                        "tool_name": "analyze_costs",
                        "inputs": {"time_period": "last_30_days"},
                        "estimate_tokens": 150
                    }
                ],
                "dependencies": [],
                "cost": {
                    "tokens": 150,
                    "graph_hops": 2,
                    "usd": 0.003
                }
            }
        }
    )
    
    # Core plan data
    tasks: List[Task] = Field(..., min_length=1, description="Tasks to execute")
    dependencies: List[Dependency] = Field(default_factory=list, description="Task dependencies")
    cost: CostEstimate = Field(..., description="Execution cost estimate")
    
    # Plan metadata
    id: UUID = Field(default_factory=uuid4, description="Unique plan identifier")
    request_id: str = Field(..., description="Original request identifier")
    user_id: str = Field(..., description="User who initiated the request")
    agent_domain: str = Field(..., description="Agent domain handling this plan")
    agent_id: str = Field(..., description="Specific agent instance identifier")
    
    # Planning context
    original_message: str = Field(..., description="Original user message")
    planning_reasoning: str = Field(default="", description="LLM reasoning for plan creation")
    memory_context_used: List[str] = Field(default_factory=list, description="Memory contexts used in planning")
    
    # Execution tracking
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Overall plan status")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Execution progress (0-1)")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Plan creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Execution start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Execution completion timestamp")
    
    # Schema versioning for forward compatibility
    schema_version: str = Field(default="1.0", description="Schema version for compatibility")
    
    @field_validator('tasks')
    @classmethod
    def validate_tasks_not_empty(cls, v: List[Task]) -> List[Task]:
        """Ensure plan has at least one task"""
        if not v:
            raise ValueError("Plan must contain at least one task")
        return v
    
    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, v: List[Dependency], info) -> List[Dependency]:
        """Validate dependency consistency"""
        if 'tasks' not in info.data:
            return v
            
        task_ids = {task.id for task in info.data['tasks']}
        
        for dep in v:
            if dep.parent_id not in task_ids:
                raise ValueError(f"Dependency parent_id {dep.parent_id} not found in tasks")
            if dep.child_id not in task_ids:
                raise ValueError(f"Dependency child_id {dep.child_id} not found in tasks")
        
        return v
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that have all dependencies satisfied and are ready to execute"""
        if not self.dependencies:
            return [task for task in self.tasks if task.status == TaskStatus.PENDING]
        
        completed_task_ids = {task.id for task in self.tasks if task.status == TaskStatus.COMPLETED}
        blocked_task_ids = set()
        
        for dep in self.dependencies:
            if dep.parent_id not in completed_task_ids:
                blocked_task_ids.add(dep.child_id)
        
        return [
            task for task in self.tasks 
            if task.status == TaskStatus.PENDING and task.id not in blocked_task_ids
        ]
    
    def get_parallel_groups(self) -> Dict[str, List[Task]]:
        """Group tasks by parallel execution group"""
        groups = {}
        for task in self.tasks:
            if task.parallel_group_id:
                if task.parallel_group_id not in groups:
                    groups[task.parallel_group_id] = []
                groups[task.parallel_group_id].append(task)
        return groups
    
    def calculate_progress(self) -> float:
        """Calculate current execution progress"""
        if not self.tasks:
            return 0.0
        
        completed = len([task for task in self.tasks if task.status == TaskStatus.COMPLETED])
        total = len(self.tasks)
        return completed / total
    
    def update_progress(self) -> None:
        """Update the progress field based on task statuses"""
        self.progress = self.calculate_progress()
        
        # Update overall plan status
        if self.progress == 1.0:
            self.status = TaskStatus.COMPLETED
            if not self.completed_at:
                self.completed_at = datetime.utcnow()
        elif self.progress > 0:
            self.status = TaskStatus.IN_PROGRESS
            if not self.started_at:
                self.started_at = datetime.utcnow()
        
        # Check for failed tasks
        failed_tasks = [task for task in self.tasks if task.status == TaskStatus.FAILED]
        if failed_tasks:
            self.status = TaskStatus.FAILED


class PlanningResult(BaseModel):
    """Result of plan creation process"""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True
    )
    
    plan: Optional[ExecutionPlan] = Field(default=None, description="Generated execution plan")
    success: bool = Field(..., description="Whether planning succeeded")
    error_message: str = Field(default="", description="Error message if planning failed")
    warnings: List[str] = Field(default_factory=list, description="Planning warnings")
    
    # Planning metrics
    planning_time_ms: float = Field(..., ge=0, description="Time taken for planning in milliseconds")
    llm_tokens_used: int = Field(default=0, ge=0, description="LLM tokens consumed during planning")
    memory_queries: int = Field(default=0, ge=0, description="Number of memory queries performed")
    
    @field_validator('planning_time_ms')
    @classmethod
    def validate_planning_time(cls, v: float) -> float:
        """Ensure planning time is reasonable"""
        if v > 30000:  # 30 seconds
            raise ValueError("Planning time exceeds maximum threshold")
        return round(v, 2)


class CompilationResult(BaseModel):
    """Result of graph compilation process"""  
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        validate_assignment=True
    )
    
    success: bool = Field(..., description="Whether compilation succeeded")
    graph_json: Optional[Dict[str, Any]] = Field(default=None, description="Serialized StateGraph")
    error_message: str = Field(default="", description="Error message if compilation failed")
    warnings: List[str] = Field(default_factory=list, description="Compilation warnings")
    
    # Compilation metrics
    compilation_time_ms: float = Field(..., ge=0, description="Time taken for compilation in milliseconds") 
    nodes_created: int = Field(default=0, ge=0, description="Number of nodes in compiled graph")
    edges_created: int = Field(default=0, ge=0, description="Number of edges in compiled graph")
    
    # Graph analysis
    has_cycles: bool = Field(default=False, description="Whether graph contains cycles")
    parallel_groups_count: int = Field(default=0, ge=0, description="Number of parallel execution groups")
    
    @field_validator('compilation_time_ms')
    @classmethod  
    def validate_compilation_time(cls, v: float) -> float:
        """Ensure compilation time meets performance requirements"""
        if v > 50:  # 50ms p95 requirement
            raise ValueError("Compilation time exceeds p95 requirement of 50ms")
        return round(v, 2)