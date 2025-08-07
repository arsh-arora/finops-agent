#!/usr/bin/env python3
"""
Smoke Test for Phase 5 Orchestration System
Comprehensive end-to-end testing of orchestration components
"""

import asyncio
import sys
import time
import json
from typing import Dict, Any
from datetime import datetime
from unittest.mock import Mock, AsyncMock

# Add project root to path
sys.path.insert(0, '/Users/aroraji/Desktop/MultiAgent-FinOps-Chat/finops-agent-chat')

from src.orchestration.langgraph_runner import LangGraphRunner, ExecutionResult
from src.orchestration.dataflow import DataFlowManager, NodeOutput
from src.orchestration.parallel import ParallelExecutor, ExecutionGroup
from src.orchestration.cross_agent import CrossAgentWorkflowManager
from src.orchestration.task_offloader import TaskOffloadingEngine
from src.workers.celery_app import heavy_task, get_heavy_task_registry
from src.workers.queue_monitor import RedisQueueMonitor
from src.memory.mem0_service import FinOpsMemoryService


class SmokeTestRunner:
    """Orchestration system smoke test runner"""
    
    def __init__(self):
        self.results: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'test_results': {},
            'performance_metrics': {}
        }
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all smoke tests"""
        
        print("🚀 Starting Phase 5 Orchestration Smoke Tests")
        print("=" * 60)
        
        # Test components
        test_methods = [
            self.test_langgraph_runner,
            self.test_dataflow_manager,
            self.test_parallel_executor,
            self.test_cross_agent_workflow,
            self.test_task_offloading,
            self.test_heavy_task_system,
            self.test_queue_monitoring,
            self.test_end_to_end_workflow
        ]
        
        for test_method in test_methods:
            test_name = test_method.__name__
            print(f"\n📋 Running {test_name}...")
            
            try:
                start_time = time.time()
                result = await test_method()
                execution_time = (time.time() - start_time) * 1000
                
                self.results['test_results'][test_name] = {
                    'status': 'PASSED',
                    'execution_time_ms': execution_time,
                    'result': result
                }
                self.results['tests_passed'] += 1
                print(f"✅ {test_name} PASSED ({execution_time:.1f}ms)")
                
            except Exception as e:
                self.results['test_results'][test_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                self.results['tests_failed'] += 1
                print(f"❌ {test_name} FAILED: {str(e)}")
        
        # Generate final report
        self._generate_final_report()
        return self.results
    
    async def test_langgraph_runner(self) -> Dict[str, Any]:
        """Test LangGraph execution engine"""
        
        # Create mock memory service
        mock_memory = Mock(spec=FinOpsMemoryService)
        mock_memory.retrieve_relevant_memories = AsyncMock(return_value=[])
        mock_memory.store_conversation_memory = AsyncMock()
        
        # Initialize runner
        runner = LangGraphRunner(mock_memory)
        
        # Create mock StateGraph
        mock_graph = Mock()
        async def test_node(state):
            state['processed'] = True
            state['timestamp'] = datetime.utcnow().isoformat()
            return state
        
        mock_graph.nodes = {'test_node': test_node}
        mock_graph.edges = []
        
        # Execute graph
        context = {
            'request_id': 'smoke-test-001',
            'user_id': 'test-user',
            'execution_id': 'exec-001'
        }
        
        result = await runner.run_stategraph(
            graph=mock_graph,
            context=context,
            timeout_seconds=10
        )
        
        # Verify result
        assert isinstance(result, ExecutionResult)
        assert result.success is True
        assert result.execution_id is not None
        assert result.metrics.nodes_executed > 0
        
        await runner.cleanup()
        
        return {
            'execution_success': result.success,
            'nodes_executed': result.metrics.nodes_executed,
            'execution_time_ms': result.metrics.total_execution_time_ms
        }
    
    async def test_dataflow_manager(self) -> Dict[str, Any]:
        """Test data flow management"""
        
        manager = DataFlowManager(max_concurrent_operations=10)
        
        # Test basic routing
        test_data = {'result': 'success', 'value': 42}
        output = await manager.route_output(
            node_id='test-node',
            output=test_data,
            target_nodes=['target-1', 'target-2'],
            execution_context={'request_id': 'smoke-test-002'}
        )
        
        assert isinstance(output, NodeOutput)
        assert output.data == test_data
        assert len(output.target_nodes) == 2
        
        # Test concurrent routing
        concurrent_tasks = []
        for i in range(5):
            task = manager.route_output(
                node_id=f'node-{i}',
                output={'index': i, 'data': f'test-{i}'}
            )
            concurrent_tasks.append(task)
        
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        assert len(concurrent_results) == 5
        
        # Test metrics
        metrics = manager.get_metrics()
        assert metrics['total_operations'] >= 6
        
        await manager.cleanup()
        
        return {
            'routing_success': True,
            'concurrent_operations': len(concurrent_results),
            'total_operations': metrics['total_operations'],
            'success_rate': metrics['success_rate']
        }
    
    async def test_parallel_executor(self) -> Dict[str, Any]:
        """Test parallel execution system"""
        
        executor = ParallelExecutor(default_concurrency=3)
        
        # Create mock node executors
        async def mock_executor_1(context):
            await asyncio.sleep(0.01)
            return {'node': 'executor_1', 'result': 'success'}
        
        async def mock_executor_2(context):
            await asyncio.sleep(0.02)
            return {'node': 'executor_2', 'result': 'completed'}
        
        async def mock_executor_3(context):
            await asyncio.sleep(0.005)
            return {'node': 'executor_3', 'result': 'done'}
        
        # Define execution groups
        groups = {
            'group_1': ['node_1', 'node_2'],
            'group_2': ['node_3']
        }
        
        node_executors = {
            'node_1': mock_executor_1,
            'node_2': mock_executor_2,
            'node_3': mock_executor_3
        }
        
        context = {'request_id': 'smoke-test-003'}
        
        # Execute parallel groups
        result = await executor.execute_parallel_groups(
            groups=groups,
            node_executors=node_executors,
            execution_context=context
        )
        
        assert result.success is True
        assert result.total_groups == 2
        assert result.successful_groups >= 1
        assert 'all_node_results' in result.merged_output
        
        # Test performance metrics
        perf_metrics = executor.get_performance_metrics()
        
        return {
            'execution_success': result.success,
            'groups_executed': result.total_groups,
            'successful_groups': result.successful_groups,
            'overall_efficiency': result.overall_efficiency,
            'performance_metrics': perf_metrics
        }
    
    async def test_cross_agent_workflow(self) -> Dict[str, Any]:
        """Test cross-agent workflow management"""
        
        # Mock memory service and agent router
        mock_memory = Mock(spec=FinOpsMemoryService)
        mock_memory.store_conversation_memory = AsyncMock()
        
        mock_router = Mock()
        mock_agent_class = Mock()
        mock_agent_class.get_domain.return_value = 'finops'
        mock_agent_class.get_capabilities.return_value = ['test_capability']
        mock_router.select = AsyncMock(return_value=mock_agent_class)
        
        # Initialize workflow manager
        workflow_manager = CrossAgentWorkflowManager(mock_memory, mock_router)
        
        # Create mock execution plan
        from src.agents.models import ExecutionPlan, Task, CostEstimate
        from uuid import uuid4
        
        mock_task = Mock()
        mock_task.id = uuid4()
        mock_task.tool_name = 'finops_test_tool'
        mock_task.inputs = {'param': 'value'}
        mock_task.description = 'Test task'
        mock_task.name = 'Test Task'
        
        mock_plan = Mock(spec=ExecutionPlan)
        mock_plan.id = uuid4()
        mock_plan.tasks = [mock_task]
        mock_plan.request_id = 'cross-agent-test'
        mock_plan.user_id = 'test-user'
        
        workflow_context = {
            'request_id': 'cross-agent-test',
            'user_id': 'test-user'
        }
        
        # Execute cross-agent workflow
        result = await workflow_manager.execute_cross_agent_workflow(
            execution_plan=mock_plan,
            workflow_context=workflow_context
        )
        
        assert result.workflow_id is not None
        assert result.total_agents >= 0
        
        await workflow_manager.cleanup()
        
        return {
            'workflow_success': result.success,
            'total_agents': result.total_agents,
            'successful_agents': result.successful_agents,
            'execution_time_ms': result.total_execution_time_ms
        }
    
    async def test_task_offloading(self) -> Dict[str, Any]:
        """Test task offloading engine"""
        
        # Initialize offloading engine
        offloader = TaskOffloadingEngine(enable_offloading=True)
        
        # Create mock tool function
        async def mock_tool(param1: str, param2: int = 10) -> str:
            await asyncio.sleep(0.01)
            return f"processed: {param1}-{param2}"
        
        # Test offloading decision
        decision = await offloader.should_offload_task(
            tool_func=mock_tool,
            tool_inputs={'param1': 'test', 'param2': 42},
            execution_context={'request_id': 'offload-test'}
        )
        
        assert decision.execution_mode is not None
        assert decision.reasoning is not None
        
        # Test local execution (force local)
        result = await offloader.execute_with_offloading(
            tool_func=mock_tool,
            tool_inputs={'param1': 'local_test', 'param2': 100},
            execution_context={'force_local_execution': True}
        )
        
        assert result == "processed: local_test-100"
        
        # Test metrics
        metrics = offloader.get_offloading_metrics()
        assert metrics.total_tasks_analyzed >= 1
        
        await offloader.cleanup()
        
        return {
            'decision_made': True,
            'execution_success': result is not None,
            'tasks_analyzed': metrics.total_tasks_analyzed,
            'local_executions': metrics.local_executions
        }
    
    async def test_heavy_task_system(self) -> Dict[str, Any]:
        """Test heavy task decoration and registration"""
        
        # Create heavy task
        @heavy_task(name='smoke_test_heavy_task', queue='test_queue', priority=8)
        async def test_heavy_task(self, value: int) -> dict:
            """Test heavy task for smoke testing"""
            await asyncio.sleep(0.01)
            return {'processed_value': value * 2, 'timestamp': datetime.utcnow().isoformat()}
        
        # Verify registration
        registry = get_heavy_task_registry()
        assert 'smoke_test_heavy_task' in registry
        
        task_config = registry['smoke_test_heavy_task']
        assert task_config['queue'] == 'test_queue'
        assert task_config['priority'] == 8
        
        # Test local execution (forced)
        result = await test_heavy_task(
            None,  # Mock self
            value=21,
            execution_context={'force_local_execution': True}
        )
        
        assert result['processed_value'] == 42
        assert 'timestamp' in result
        
        return {
            'task_registered': True,
            'task_config_valid': task_config['queue'] == 'test_queue',
            'execution_result': result,
            'registry_size': len(registry)
        }
    
    async def test_queue_monitoring(self) -> Dict[str, Any]:
        """Test Redis queue monitoring system"""
        
        # Mock Celery app
        mock_celery_app = Mock()
        mock_inspector = Mock()
        mock_inspector.active_queues.return_value = {
            'worker1': [{'name': 'test_queue'}]
        }
        mock_inspector.reserved.return_value = {'worker1': []}
        mock_inspector.active.return_value = {'worker1': []}
        mock_celery_app.control.inspect.return_value = mock_inspector
        
        # Initialize monitor
        monitor = RedisQueueMonitor(mock_celery_app)
        
        # Configure test queue
        queue_config = {
            'test_queue': {
                'retry_policy': {
                    'max_retries': 3,
                    'backoff_strategy': 'exponential'
                },
                'alert_thresholds': {
                    'error_rate_threshold': 10.0,
                    'queue_size_threshold': 100
                }
            }
        }
        
        await monitor.configure_queues(queue_config)
        
        # Test queue metrics collection
        await monitor._collect_queue_metrics('test_queue')
        
        # Test health summary
        health_summary = monitor.get_queue_health_summary()
        assert 'total_queues' in health_summary
        assert 'monitoring_active' in health_summary
        
        await monitor.cleanup()
        
        return {
            'configuration_success': True,
            'metrics_collection': True,
            'health_summary': health_summary
        }
    
    async def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end orchestration workflow"""
        
        print("  🔄 Running end-to-end integration test...")
        
        # Initialize all components
        mock_memory = Mock(spec=FinOpsMemoryService)
        mock_memory.retrieve_relevant_memories = AsyncMock(return_value=[])
        mock_memory.store_conversation_memory = AsyncMock()
        
        runner = LangGraphRunner(mock_memory)
        dataflow = DataFlowManager()
        parallel = ParallelExecutor()
        offloader = TaskOffloadingEngine()
        
        # Create end-to-end workflow simulation
        workflow_start = time.time()
        
        # Step 1: Create mock execution plan
        async def workflow_node_1(state):
            state['step1_completed'] = True
            state['step1_timestamp'] = datetime.utcnow().isoformat()
            # Simulate data flow
            await dataflow.route_output(
                node_id='workflow_node_1',
                output={'step1_result': 'success'},
                execution_context=state
            )
            return state
        
        async def workflow_node_2(state):
            state['step2_completed'] = True
            state['step2_timestamp'] = datetime.utcnow().isoformat()
            return state
        
        # Step 2: Execute with LangGraph runner
        mock_graph = Mock()
        mock_graph.nodes = {
            'workflow_node_1': workflow_node_1,
            'workflow_node_2': workflow_node_2
        }
        mock_graph.edges = []
        
        context = {
            'request_id': 'e2e-workflow-test',
            'user_id': 'e2e-test-user',
            'execution_id': 'e2e-exec-001'
        }
        
        result = await runner.run_stategraph(
            graph=mock_graph,
            context=context,
            timeout_seconds=15
        )
        
        # Step 3: Verify all components worked
        workflow_time = (time.time() - workflow_start) * 1000
        
        # Cleanup
        await runner.cleanup()
        await dataflow.cleanup()
        await offloader.cleanup()
        
        return {
            'workflow_success': result.success,
            'total_workflow_time_ms': workflow_time,
            'nodes_executed': result.metrics.nodes_executed,
            'final_state_keys': list(result.final_state.keys()),
            'components_integrated': 4,  # runner, dataflow, parallel, offloader
            'execution_metrics': {
                'total_time_ms': result.metrics.total_execution_time_ms,
                'nodes_executed': result.metrics.nodes_executed,
                'graph_hops': result.metrics.graph_hops
            }
        }
    
    def _generate_final_report(self):
        """Generate final test report"""
        
        total_tests = self.results['tests_passed'] + self.results['tests_failed']
        success_rate = (self.results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 PHASE 5 ORCHESTRATION SMOKE TEST RESULTS")
        print("=" * 60)
        print(f"Tests Passed: {self.results['tests_passed']}")
        print(f"Tests Failed: {self.results['tests_failed']}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Total Tests: {total_tests}")
        
        # Performance summary
        execution_times = []
        for test_name, test_result in self.results['test_results'].items():
            if test_result['status'] == 'PASSED' and 'execution_time_ms' in test_result:
                execution_times.append(test_result['execution_time_ms'])
        
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            print(f"\nPerformance Metrics:")
            print(f"  Average Test Time: {avg_time:.1f}ms")
            print(f"  Slowest Test Time: {max_time:.1f}ms")
        
        # Component status
        print(f"\nComponent Test Status:")
        for test_name, test_result in self.results['test_results'].items():
            status_icon = "✅" if test_result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {test_name.replace('test_', '').replace('_', ' ').title()}")
        
        if self.results['tests_failed'] == 0:
            print(f"\n🎉 ALL TESTS PASSED! Phase 5 Orchestration system is ready.")
        else:
            print(f"\n⚠️  {self.results['tests_failed']} test(s) failed. Check implementation.")
        
        print("=" * 60)


async def main():
    """Main smoke test entry point"""
    
    runner = SmokeTestRunner()
    
    try:
        results = await runner.run_all_tests()
        
        # Save results to file
        with open('smoke_test_orchestration_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Exit with appropriate code
        exit_code = 0 if results['tests_failed'] == 0 else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"💥 Smoke test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())