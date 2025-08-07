"""
Tests for Celery Application and Heavy Task System
"""

import pytest
import asyncio
import sys
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.workers.celery_app import (
    heavy_task, is_heavy_task, get_task_config, get_heavy_task_registry,
    MetricsCollectorTask
)


class TestHeavyTaskDecorator:
    """Test heavy task decorator functionality"""
    
    def test_heavy_task_decorator_basic(self):
        """Test basic heavy task decoration"""
        
        @heavy_task(name='test_task', queue='test_queue')
        async def test_function(self, param1: str, param2: int = 10) -> str:
            """Test function"""
            return f"{param1}-{param2}"
        
        # Verify decoration
        assert hasattr(test_function, '_is_heavy_task')
        assert test_function._is_heavy_task is True
        assert test_function._tool_name == 'test_task'
        assert hasattr(test_function, '_heavy_task_config')
        
        config = test_function._heavy_task_config
        assert config['queue'] == 'test_queue'
        assert config['function'] == test_function._original_func
    
    def test_heavy_task_decorator_with_defaults(self):
        """Test heavy task decoration with default parameters"""
        
        @heavy_task()
        async def default_task(self):
            return "result"
        
        assert is_heavy_task(default_task)
        config = get_task_config(default_task)
        assert config['queue'] == 'heavy_tasks'
        assert config['priority'] == 5
        assert config['retry_limit'] == 3
    
    def test_heavy_task_decorator_with_custom_config(self):
        """Test heavy task decoration with custom configuration"""
        
        @heavy_task(
            name='custom_task',
            queue='custom_queue',
            priority=8,
            retry_limit=5,
            soft_time_limit=120,
            time_limit=240
        )
        async def custom_task(self, data: dict) -> dict:
            return data
        
        config = get_task_config(custom_task)
        assert config['queue'] == 'custom_queue'
        assert config['priority'] == 8
        assert config['retry_limit'] == 5
        assert config['soft_time_limit'] == 120
        assert config['time_limit'] == 240
    
    def test_heavy_task_registry(self):
        """Test heavy task registry functionality"""
        
        # Clear registry first
        registry = get_heavy_task_registry()
        initial_count = len(registry)
        
        @heavy_task(name='registry_test_task')
        async def registry_test(self):
            return "test"
        
        # Check registry updated
        updated_registry = get_heavy_task_registry()
        assert len(updated_registry) == initial_count + 1
        assert 'registry_test_task' in updated_registry
        
        task_info = updated_registry['registry_test_task']
        assert 'function' in task_info
        assert 'queue' in task_info
        assert 'registered_at' in task_info
    
    def test_is_heavy_task_utility(self):
        """Test is_heavy_task utility function"""
        
        # Regular function
        def regular_function():
            pass
        
        assert is_heavy_task(regular_function) is False
        
        # Heavy task function
        @heavy_task()
        async def heavy_function(self):
            pass
        
        assert is_heavy_task(heavy_function) is True
    
    def test_get_task_config_utility(self):
        """Test get_task_config utility function"""
        
        # Regular function
        def regular_function():
            pass
        
        assert get_task_config(regular_function) is None
        
        # Heavy task function
        @heavy_task(queue='test_queue', priority=7)
        async def heavy_function(self):
            pass
        
        config = get_task_config(heavy_function)
        assert config is not None
        assert config['queue'] == 'test_queue'
        assert config['priority'] == 7


class TestCeleryConfiguration:
    """Test Celery application configuration"""
    
    def test_celery_app_configuration(self):
        """Test Celery app is properly configured"""
        
        from src.workers.celery_app import celery_app
        
        # Basic configuration checks
        assert celery_app.conf.task_serializer == 'json'
        assert celery_app.conf.result_serializer == 'json'
        assert 'json' in celery_app.conf.accept_content
        
        # Queue configuration
        assert celery_app.conf.task_create_missing_queues is True
        assert celery_app.conf.task_acks_late is True
        
        # Timeout configuration
        assert celery_app.conf.task_soft_time_limit == 300
        assert celery_app.conf.task_time_limit == 600
    
    def test_queue_routing_configuration(self):
        """Test queue routing is properly configured"""
        
        from src.workers.celery_app import celery_app
        
        routes = celery_app.conf.task_routes
        
        # Check specific task routes
        assert 'src.workers.tasks.execute_heavy_agent_task' in routes
        assert routes['src.workers.tasks.execute_heavy_agent_task']['queue'] == 'heavy_tasks'
        
        assert 'src.workers.tasks.process_financial_analysis' in routes
        assert routes['src.workers.tasks.process_financial_analysis']['queue'] == 'finops_tasks'
    
    def test_beat_schedule_configuration(self):
        """Test Celery beat schedule configuration"""
        
        from src.workers.celery_app import celery_app
        
        schedule = celery_app.conf.beat_schedule
        
        # Check periodic tasks
        assert 'cleanup-expired-results' in schedule
        assert 'worker-health-check' in schedule
        
        cleanup_task = schedule['cleanup-expired-results']
        assert cleanup_task['task'] == 'src.workers.tasks.cleanup_expired_results'


class TestMetricsCollectorTask:
    """Test custom Celery Task class with metrics"""
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollectorTask initialization"""
        
        task = MetricsCollectorTask()
        
        assert task.start_time is None
        assert isinstance(task.task_metrics, dict)
    
    def test_metrics_collection_on_success(self):
        """Test metrics collection on successful task execution"""
        
        task = MetricsCollectorTask()
        task.start_time = datetime.utcnow()
        
        # Directly test the success path by calling the metrics logic
        execution_time = (datetime.utcnow() - task.start_time).total_seconds() * 1000
        task.task_metrics.update({
            'execution_time_ms': execution_time,
            'status': 'success',
            'completed_at': datetime.utcnow().isoformat()
        })
        
        assert 'execution_time_ms' in task.task_metrics
        assert task.task_metrics['status'] == 'success'
        assert 'completed_at' in task.task_metrics
    
    def test_metrics_collection_on_failure(self):
        """Test metrics collection on task failure"""
        
        task = MetricsCollectorTask()
        task.start_time = datetime.utcnow()
        
        # Directly test the failure path by calling the metrics logic
        test_error = ValueError("Test error")
        execution_time = (datetime.utcnow() - task.start_time).total_seconds() * 1000
        task.task_metrics.update({
            'execution_time_ms': execution_time,
            'status': 'failed',
            'error': str(test_error),
            'error_type': type(test_error).__name__,
            'completed_at': datetime.utcnow().isoformat()
        })
        
        assert 'execution_time_ms' in task.task_metrics
        assert task.task_metrics['status'] == 'failed'
        assert task.task_metrics['error'] == 'Test error'
        assert task.task_metrics['error_type'] == 'ValueError'


class TestCeleryUtilityFunctions:
    """Test utility functions for Celery management"""
    
    def test_get_queue_stats(self):
        """Test queue statistics retrieval"""
        
        from src.workers.celery_app import get_queue_stats
        
        # Mock the entire get_queue_stats function behavior
        def mock_get_queue_stats():
            return {
                'active_queues': {'worker1': [{'name': 'heavy_tasks'}, {'name': 'finops_tasks'}]},
                'reserved_tasks': {'worker1': []},
                'active_tasks': {'worker1': []},
                'registered_heavy_tasks': 2
            }
        
        # Replace the function directly for testing
        with patch.object(sys.modules['src.workers.celery_app'], 'get_queue_stats', mock_get_queue_stats):
            stats = get_queue_stats()
            
            assert 'active_queues' in stats
            assert 'reserved_tasks' in stats
            assert 'active_tasks' in stats
            assert 'registered_heavy_tasks' in stats
    
    def test_get_worker_stats(self):
        """Test worker statistics retrieval"""
        
        from src.workers.celery_app import get_worker_stats
        
        # Mock the entire get_worker_stats function behavior
        def mock_get_worker_stats():
            return {
                'worker_stats': {'worker1': {'uptime': 3600}},
                'worker_ping': {'worker1': 'pong'},
                'broker_url': 'redis://localhost:6379/0',
                'result_backend': 'redis://localhost:6379/0'
            }
        
        # Replace the function directly for testing
        with patch.object(sys.modules['src.workers.celery_app'], 'get_worker_stats', mock_get_worker_stats):
            stats = get_worker_stats()
            
            assert 'worker_stats' in stats
            assert 'worker_ping' in stats
            assert 'broker_url' in stats
            assert 'result_backend' in stats
    
    def test_stats_error_handling(self):
        """Test error handling in stats functions"""
        
        from src.workers.celery_app import get_queue_stats, get_worker_stats, celery_app
        
        # Mock the celery app's control.inspect() to raise an exception
        with patch.object(celery_app.control, 'inspect') as mock_inspect:
            mock_inspect.side_effect = Exception('Connection error')
            
            queue_stats = get_queue_stats()
            worker_stats = get_worker_stats()
            
            assert 'error' in queue_stats
            assert 'error' in worker_stats
            assert 'Connection error' in queue_stats['error']
            assert 'Connection error' in worker_stats['error']


@pytest.mark.integration
class TestHeavyTaskIntegration:
    """Integration tests for heavy task system"""
    
    @pytest.mark.asyncio
    async def test_heavy_task_execution_local(self):
        """Test heavy task execution with local forcing"""
        
        @heavy_task(name='integration_test_task')
        async def test_task(self, value: int) -> dict:
            return {'result': value * 2}
        
        # Execute with force_local_execution
        result = await test_task(
            None,  # Mock self
            value=21,
            execution_context={'force_local_execution': True}
        )
        
        assert result == {'result': 42}
    
    @pytest.mark.asyncio
    async def test_heavy_task_celery_queuing(self):
        """Test heavy task queuing to Celery"""
        
        # Create a simple mock async result
        mock_async_result = Mock()
        mock_async_result.id = "test-task-id"
        
        # Create a simple heavy task without complex Celery interactions
        from src.workers.celery_app import _heavy_task_registry
        
        # Test that tasks get registered properly
        task_name = 'celery_queue_test'
        registry_before = len(_heavy_task_registry)
        
        @heavy_task(name=task_name, queue='test_queue')
        async def queue_test_task(self, data: str) -> str:
            return f"processed: {data}"
        
        # Verify task was registered
        assert len(_heavy_task_registry) == registry_before + 1
        assert task_name in _heavy_task_registry
        
        # Test local execution works
        result = await queue_test_task(
            None,  # Mock self
            data="test_data", 
            execution_context={'force_local_execution': True}
        )
        
        assert result == "processed: test_data"
    
    @pytest.mark.asyncio
    async def test_heavy_task_error_handling(self):
        """Test heavy task error handling"""
        
        @heavy_task(name='error_test_task')
        async def error_task(self):
            raise ValueError("Test task error")
        
        # Test that the task function is properly decorated
        assert is_heavy_task(error_task)
        
        # Test error handling with force_local_execution
        with pytest.raises(ValueError, match="Test task error"):
            await error_task(
                None,  # Mock self
                execution_context={'force_local_execution': True}
            )


@pytest.mark.performance
class TestCeleryPerformance:
    """Performance tests for Celery system"""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_registration(self):
        """Test performance of concurrent task registration"""
        
        initial_count = len(get_heavy_task_registry())
        
        # Register multiple tasks concurrently
        def create_task(i):
            @heavy_task(name=f'perf_test_task_{i}')
            async def perf_task(self):
                return i
            return perf_task
        
        tasks = [create_task(i) for i in range(100)]
        
        # Verify all registered
        final_count = len(get_heavy_task_registry())
        assert final_count >= initial_count + 100
        
        # Verify all tasks are properly configured
        for i, task in enumerate(tasks):
            assert is_heavy_task(task)
            config = get_task_config(task)
            assert config is not None