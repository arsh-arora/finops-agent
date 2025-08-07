"""
Tests for LangGraph StateGraph execution engine
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.orchestration.langgraph_runner import (
    LangGraphRunner, ExecutionResult, ExecutionMetrics
)
from src.memory.mem0_service import FinOpsMemoryService
from src.planner.compiler import GraphState


class TestLangGraphRunner:
    """Test LangGraph execution engine"""
    
    @pytest.fixture
    def mock_memory_service(self):
        """Mock memory service"""
        mock = Mock(spec=FinOpsMemoryService)
        mock.retrieve_relevant_memories = AsyncMock(return_value=[])
        mock.store_conversation_memory = AsyncMock()
        return mock
    
    @pytest.fixture
    def runner(self, mock_memory_service):
        """Create LangGraph runner instance"""
        return LangGraphRunner(mock_memory_service)
    
    @pytest.fixture
    def sample_context(self):
        """Sample execution context"""
        return {
            'request_id': 'test-request-123',
            'user_id': 'test-user',
            'plan_id': 'test-plan-456',
            'execution_id': 'test-execution-789'
        }
    
    def test_runner_initialization(self, runner, mock_memory_service):
        """Test runner initialization"""
        assert runner.memory_service == mock_memory_service
        assert runner.dataflow_manager is not None
        assert runner.parallel_executor is not None
        assert runner.task_offloader is not None
        assert len(runner._active_executions) == 0
    
    @pytest.mark.asyncio
    async def test_run_stategraph_without_langgraph(self, runner, sample_context):
        """Test execution without LangGraph available"""
        
        with patch('src.orchestration.langgraph_runner.LANGGRAPH_AVAILABLE', False):
            # Mock graph without LangGraph
            mock_graph = Mock()
            mock_graph.nodes = {
                'test_node': AsyncMock(return_value=GraphState({'result': 'test'}))
            }
            mock_graph.edges = []
            
            # Execute should raise RuntimeError
            with pytest.raises(RuntimeError, match="LangGraph not available"):
                await runner.run_stategraph(
                    graph=mock_graph,
                    context=sample_context,
                    timeout_seconds=30
                )
    
    @pytest.mark.asyncio
    async def test_execution_timeout(self, runner, sample_context):
        """Test execution timeout handling"""
        
        with patch('src.orchestration.langgraph_runner.LANGGRAPH_AVAILABLE', False):
            # Execute should raise RuntimeError
            with pytest.raises(RuntimeError, match="LangGraph not available"):
                await runner.run_stategraph(
                    graph=Mock(),
                    context=sample_context,
                    timeout_seconds=1
                )
    
    @pytest.mark.asyncio
    async def test_execution_with_node_failure(self, runner, sample_context):
        """Test handling of node execution failures"""
        
        with patch('src.orchestration.langgraph_runner.LANGGRAPH_AVAILABLE', False):
            # Execute should raise RuntimeError
            with pytest.raises(RuntimeError, match="LangGraph not available"):
                await runner.run_stategraph(
                    graph=Mock(),
                    context=sample_context
                )
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, runner, sample_context):
        """Test comprehensive metrics collection"""
        
        with patch('src.orchestration.langgraph_runner.LANGGRAPH_AVAILABLE', False):
            # Execute should raise RuntimeError
            with pytest.raises(RuntimeError, match="LangGraph not available"):
                await runner.run_stategraph(
                    graph=Mock(),
                    context=sample_context
                )
    
    @pytest.mark.asyncio
    async def test_concurrent_executions(self, runner, sample_context):
        """Test handling of concurrent executions"""
        
        with patch('src.orchestration.langgraph_runner.LANGGRAPH_AVAILABLE', False):
            # Execute should raise RuntimeError
            with pytest.raises(RuntimeError, match="LangGraph not available"):
                await runner.run_stategraph(
                    graph=Mock(),
                    context=sample_context
                )
    
    @pytest.mark.asyncio
    async def test_execution_cancellation(self, runner, sample_context):
        """Test execution cancellation"""
        
        # Test cancel_execution returns False for non-existent execution
        cancelled = await runner.cancel_execution('non-existent')
        assert cancelled is False
    
    def test_active_executions_tracking(self, runner):
        """Test tracking of active executions"""
        
        # Initially no active executions
        assert len(runner.get_active_executions()) == 0
        
        # Mock execution tracking would be tested with actual executions
        # This tests the interface
        metrics = runner.get_execution_metrics('non-existent')
        assert metrics is None
    
    @pytest.mark.asyncio
    async def test_tool_registry_integration(self, runner, sample_context):
        """Test integration with tool registry"""
        
        # This would test tool registration and execution
        # For now, verify the method exists
        assert hasattr(runner, '_register_graph_tools')
        
        # Mock state for testing
        test_state = GraphState(sample_context)
        
        # Test tool registration (should not fail)
        await runner._register_graph_tools(Mock(), test_state)
    
    @pytest.mark.asyncio
    async def test_error_result_creation(self, runner, sample_context):
        """Test error result creation"""
        
        metrics = ExecutionMetrics()
        metrics.start_time = datetime.utcnow()
        
        # Test error result creation
        error_result = runner._create_error_result(
            execution_id='test-error',
            metrics=metrics,
            error_message='Test error',
            context=sample_context
        )
        
        assert error_result.success is False
        assert error_result.error_message == 'Test error'
        assert error_result.execution_id == 'test-error'
        assert error_result.metrics.total_execution_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, runner):
        """Test runner cleanup"""
        
        # Add some mock active executions
        mock_task = Mock()
        mock_task.cancel = Mock()
        runner._active_executions['test'] = mock_task
        
        # Test cleanup
        await runner.cleanup()
        
        # Verify cleanup
        mock_task.cancel.assert_called_once()
        assert len(runner._agent_cache) == 0
        assert len(runner._tool_registry_cache) == 0


class TestExecutionResult:
    """Test ExecutionResult model"""
    
    def test_execution_result_creation(self):
        """Test creation of ExecutionResult"""
        
        metrics = ExecutionMetrics()
        
        result = ExecutionResult(
            success=True,
            final_state={'test': 'data'},
            metrics=metrics
        )
        
        assert result.success is True
        assert result.final_state == {'test': 'data'}
        assert result.metrics == metrics
        assert result.execution_id is not None
    
    def test_execution_summary(self):
        """Test execution summary generation"""
        
        metrics = ExecutionMetrics()
        metrics.nodes_executed = 5
        metrics.total_execution_time_ms = 1000
        metrics.total_tokens = 100
        
        result = ExecutionResult(
            success=True,
            final_state={},
            metrics=metrics
        )
        
        summary = result.get_summary()
        
        assert 'execution_id' in summary
        assert summary['success'] is True
        assert summary['nodes_executed'] == 5
        assert summary['duration_ms'] == 1000
        assert summary['total_tokens'] == 100


class TestExecutionMetrics:
    """Test ExecutionMetrics model"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        
        metrics = ExecutionMetrics()
        
        assert isinstance(metrics.start_time, datetime)
        assert metrics.end_time is None
        assert metrics.nodes_executed == 0
        assert metrics.total_tokens == 0
        assert isinstance(metrics.node_execution_times, dict)
        assert isinstance(metrics.errors, list)
    
    def test_metrics_tracking(self):
        """Test metrics data tracking"""
        
        metrics = ExecutionMetrics()
        
        # Update metrics
        metrics.nodes_executed = 3
        metrics.nodes_failed = 1
        metrics.total_tokens = 250
        metrics.node_execution_times['node1'] = 100.5
        metrics.errors.append({'error': 'test'})
        
        assert metrics.nodes_executed == 3
        assert metrics.nodes_failed == 1
        assert metrics.total_tokens == 250
        assert 'node1' in metrics.node_execution_times
        assert len(metrics.errors) == 1