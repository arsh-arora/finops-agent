"""
Unit tests for data classes and models
"""

import pytest
from datetime import datetime
from uuid import uuid4

from agents.models import (
    ChatRequest, ExecutionPlan, Task, Dependency, CostEstimate,
    TaskStatus, Priority, DependencyType, PlanningResult, CompilationResult
)
from planner.models import CostModel, PlanningConfig


@pytest.mark.unit
class TestDataClassValidation:
    """Test Pydantic v2 data class validation"""
    
    def test_chat_request_validation(self):
        """Test ChatRequest validation"""
        # Valid request
        request = ChatRequest(
            message="Test message",
            user_id="user123",
            budget_usd=1.50
        )
        
        assert request.message == "Test message"
        assert request.user_id == "user123"
        assert request.budget_usd == 1.50
        assert request.request_id is not None
    
    def test_chat_request_validation_errors(self):
        """Test ChatRequest validation failures"""
        # Empty message
        with pytest.raises(ValueError):
            ChatRequest(message="", user_id="user123")
        
        # Negative budget
        with pytest.raises(ValueError):
            ChatRequest(message="test", user_id="user123", budget_usd=-1.0)
        
        # Budget too high
        with pytest.raises(ValueError):
            ChatRequest(message="test", user_id="user123", budget_usd=200.0)
    
    def test_task_validation(self):
        """Test Task validation"""
        task = Task(
            tool_name="test_tool",
            estimate_tokens=100,
            inputs={"param": "value"}
        )
        
        assert task.tool_name == "test_tool"
        assert task.estimate_tokens == 100
        assert task.status == TaskStatus.PENDING
        assert task.priority == Priority.MEDIUM
        assert task.id is not None
    
    def test_task_validation_errors(self):
        """Test Task validation failures"""
        # Invalid tool name
        with pytest.raises(ValueError):
            Task(tool_name="invalid@tool!", estimate_tokens=100)
        
        # Negative tokens
        with pytest.raises(ValueError):
            Task(tool_name="test_tool", estimate_tokens=-1)
        
        # Too many retries
        with pytest.raises(ValueError):
            Task(tool_name="test_tool", estimate_tokens=100, retry_count=5)
    
    def test_dependency_validation(self):
        """Test Dependency validation"""
        parent_id = uuid4()
        child_id = uuid4()
        
        dep = Dependency(
            parent_id=parent_id,
            child_id=child_id,
            dependency_type=DependencyType.SEQUENTIAL
        )
        
        assert dep.parent_id == parent_id
        assert dep.child_id == child_id
        assert dep.dependency_type == DependencyType.SEQUENTIAL
    
    def test_dependency_self_reference_error(self):
        """Test dependency can't reference itself"""
        task_id = uuid4()
        
        with pytest.raises(ValueError):
            Dependency(parent_id=task_id, child_id=task_id)
    
    def test_cost_estimate_validation(self):
        """Test CostEstimate validation"""
        cost = CostEstimate(
            tokens=1500,
            graph_hops=3,
            usd=0.0025
        )
        
        assert cost.tokens == 1500
        assert cost.graph_hops == 3
        assert cost.usd == 0.0025
        assert 0 <= cost.confidence <= 1.0
    
    def test_execution_plan_validation(self):
        """Test ExecutionPlan validation"""
        task = Task(tool_name="test_tool", estimate_tokens=100)
        cost = CostEstimate(tokens=100, usd=0.001)
        
        plan = ExecutionPlan(
            tasks=[task],
            cost=cost,
            request_id="req123",
            user_id="user123",
            agent_domain="test",
            agent_id="agent123",
            original_message="Test message"
        )
        
        assert len(plan.tasks) == 1
        assert plan.cost == cost
        assert plan.status == TaskStatus.PENDING
        assert plan.progress == 0.0
    
    def test_execution_plan_validation_errors(self):
        """Test ExecutionPlan validation failures"""
        cost = CostEstimate(tokens=100, usd=0.001)
        
        # Empty tasks list
        with pytest.raises(ValueError):
            ExecutionPlan(
                tasks=[],
                cost=cost,
                request_id="req123",
                user_id="user123",
                agent_domain="test",
                agent_id="agent123",
                original_message="Test"
            )
    
    def test_dependency_task_reference_validation(self):
        """Test dependencies reference existing tasks"""
        task1 = Task(tool_name="tool1", estimate_tokens=100)
        task2 = Task(tool_name="tool2", estimate_tokens=100)
        unknown_id = uuid4()
        
        # Valid dependency
        valid_dep = Dependency(parent_id=task1.id, child_id=task2.id)
        
        cost = CostEstimate(tokens=200, usd=0.002)
        
        plan = ExecutionPlan(
            tasks=[task1, task2],
            dependencies=[valid_dep],
            cost=cost,
            request_id="req123",
            user_id="user123",
            agent_domain="test",
            agent_id="agent123",
            original_message="Test"
        )
        
        assert len(plan.dependencies) == 1
        
        # Invalid dependency - references unknown task
        invalid_dep = Dependency(parent_id=unknown_id, child_id=task1.id)
        
        with pytest.raises(ValueError):
            ExecutionPlan(
                tasks=[task1, task2],
                dependencies=[invalid_dep],
                cost=cost,
                request_id="req123",
                user_id="user123",
                agent_domain="test",
                agent_id="agent123",
                original_message="Test"
            )


@pytest.mark.unit
class TestExecutionPlanMethods:
    """Test ExecutionPlan helper methods"""
    
    def test_get_ready_tasks_no_dependencies(self):
        """Test getting ready tasks when no dependencies"""
        task1 = Task(tool_name="tool1", estimate_tokens=100)
        task2 = Task(tool_name="tool2", estimate_tokens=100)
        cost = CostEstimate(tokens=200, usd=0.002)
        
        plan = ExecutionPlan(
            tasks=[task1, task2],
            cost=cost,
            request_id="req123",
            user_id="user123",
            agent_domain="test",
            agent_id="agent123",
            original_message="Test"
        )
        
        ready_tasks = plan.get_ready_tasks()
        assert len(ready_tasks) == 2
        assert task1 in ready_tasks
        assert task2 in ready_tasks
    
    def test_get_ready_tasks_with_dependencies(self):
        """Test getting ready tasks with dependencies"""
        task1 = Task(tool_name="tool1", estimate_tokens=100)
        task2 = Task(tool_name="tool2", estimate_tokens=100)
        task3 = Task(tool_name="tool3", estimate_tokens=100)
        
        # task2 depends on task1, task3 is independent
        dep = Dependency(parent_id=task1.id, child_id=task2.id)
        cost = CostEstimate(tokens=300, usd=0.003)
        
        plan = ExecutionPlan(
            tasks=[task1, task2, task3],
            dependencies=[dep],
            cost=cost,
            request_id="req123",
            user_id="user123",
            agent_domain="test",
            agent_id="agent123",
            original_message="Test"
        )
        
        ready_tasks = plan.get_ready_tasks()
        assert len(ready_tasks) == 2  # task1 and task3
        assert task1 in ready_tasks
        assert task3 in ready_tasks
        assert task2 not in ready_tasks  # Blocked by dependency
    
    def test_get_parallel_groups(self):
        """Test getting parallel execution groups"""
        task1 = Task(tool_name="tool1", estimate_tokens=100, parallel_group_id="group1")
        task2 = Task(tool_name="tool2", estimate_tokens=100, parallel_group_id="group1") 
        task3 = Task(tool_name="tool3", estimate_tokens=100, parallel_group_id="group2")
        task4 = Task(tool_name="tool4", estimate_tokens=100)  # No group
        
        cost = CostEstimate(tokens=400, usd=0.004)
        
        plan = ExecutionPlan(
            tasks=[task1, task2, task3, task4],
            cost=cost,
            request_id="req123",
            user_id="user123",
            agent_domain="test",
            agent_id="agent123",
            original_message="Test"
        )
        
        groups = plan.get_parallel_groups()
        
        assert len(groups) == 2
        assert "group1" in groups
        assert "group2" in groups
        assert len(groups["group1"]) == 2
        assert len(groups["group2"]) == 1
    
    def test_progress_calculation(self):
        """Test progress calculation"""
        task1 = Task(tool_name="tool1", estimate_tokens=100, status=TaskStatus.COMPLETED)
        task2 = Task(tool_name="tool2", estimate_tokens=100, status=TaskStatus.PENDING)
        cost = CostEstimate(tokens=200, usd=0.002)
        
        plan = ExecutionPlan(
            tasks=[task1, task2],
            cost=cost,
            request_id="req123",
            user_id="user123",
            agent_domain="test",
            agent_id="agent123",
            original_message="Test"
        )
        
        progress = plan.calculate_progress()
        assert progress == 0.5  # 1 of 2 tasks completed
        
        # Update progress
        plan.update_progress()
        assert plan.progress == 0.5
        assert plan.status == TaskStatus.IN_PROGRESS


@pytest.mark.unit
class TestJsonSerialization:
    """Test JSON round-trip serialization"""
    
    def test_task_json_roundtrip(self):
        """Test Task JSON serialization"""
        task = Task(
            tool_name="test_tool",
            estimate_tokens=150,
            inputs={"param": "value"},
            name="Test Task",
            priority=Priority.HIGH
        )
        
        # Serialize to JSON
        json_data = task.model_dump()
        
        # Deserialize from JSON
        restored_task = Task.model_validate(json_data)
        
        assert restored_task.tool_name == task.tool_name
        assert restored_task.estimate_tokens == task.estimate_tokens
        assert restored_task.inputs == task.inputs
        assert restored_task.priority == task.priority
        assert restored_task.id == task.id
    
    def test_execution_plan_json_roundtrip(self):
        """Test ExecutionPlan JSON serialization"""
        task = Task(tool_name="test_tool", estimate_tokens=100)
        cost = CostEstimate(tokens=100, usd=0.001)
        
        plan = ExecutionPlan(
            tasks=[task],
            cost=cost,
            request_id="req123",
            user_id="user123",
            agent_domain="test",
            agent_id="agent123",
            original_message="Test message"
        )
        
        # Serialize to JSON
        json_data = plan.model_dump()
        
        # Deserialize from JSON
        restored_plan = ExecutionPlan.model_validate(json_data)
        
        assert len(restored_plan.tasks) == 1
        assert restored_plan.cost.tokens == 100
        assert restored_plan.request_id == "req123"
        assert restored_plan.schema_version == "1.0"


@pytest.mark.unit 
class TestResultClasses:
    """Test result data classes"""
    
    def test_planning_result(self):
        """Test PlanningResult creation"""
        result = PlanningResult(
            plan=None,
            success=False,
            error_message="Test error",
            planning_time_ms=125.5,
            llm_tokens_used=200
        )
        
        assert result.success == False
        assert result.error_message == "Test error"
        assert result.planning_time_ms == 125.5
        assert result.llm_tokens_used == 200
    
    def test_compilation_result(self):
        """Test CompilationResult creation"""
        result = CompilationResult(
            success=True,
            compilation_time_ms=45.2,
            nodes_created=5,
            edges_created=4,
            parallel_groups_count=1
        )
        
        assert result.success == True
        assert result.compilation_time_ms == 45.2
        assert result.nodes_created == 5
        assert result.edges_created == 4
    
    def test_compilation_result_time_validation(self):
        """Test compilation time validation"""
        # Should pass - under 50ms limit
        result = CompilationResult(
            success=True,
            compilation_time_ms=25.0,
            nodes_created=3,
            edges_created=2
        )
        assert result.compilation_time_ms == 25.0
        
        # Should fail - over 50ms limit
        with pytest.raises(ValueError):
            CompilationResult(
                success=True,
                compilation_time_ms=75.0,  # Exceeds 50ms limit
                nodes_created=3,
                edges_created=2
            )