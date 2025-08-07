"""
Tests for Node-to-Node Data Flow Management
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from pydantic import BaseModel

from src.orchestration.dataflow import (
    DataFlowManager, NodeOutput, DataFlowOperation, DataFlowStatus
)


class TestDataModel(BaseModel):
    """Test Pydantic model for validation"""
    name: str
    value: int
    data: dict = {}


class TestNodeOutput:
    """Test NodeOutput class"""
    
    def test_node_output_creation(self):
        """Test NodeOutput creation"""
        
        output = NodeOutput(
            node_id='test-node',
            data={'result': 'test'},
            target_nodes=['node-2', 'node-3']
        )
        
        assert output.node_id == 'test-node'
        assert output.data == {'result': 'test'}
        assert output.target_nodes == ['node-2', 'node-3']
        assert output.execution_id is not None
        assert isinstance(output.timestamp, datetime)
    
    def test_node_output_serialization(self):
        """Test NodeOutput serialization"""
        
        # Test with simple data
        output = NodeOutput(
            node_id='test-node',
            data={'result': 'test', 'count': 42}
        )
        
        serialized = output.to_serializable()
        
        assert isinstance(serialized, dict)
        assert serialized['node_id'] == 'test-node'
        assert serialized['data'] == {'result': 'test', 'count': 42}
        assert 'timestamp' in serialized
        assert 'serialization_time_ms' in serialized
    
    def test_pydantic_model_serialization(self):
        """Test serialization of Pydantic models"""
        
        test_model = TestDataModel(name='test', value=42, data={'key': 'value'})
        
        output = NodeOutput(
            node_id='test-node',
            data=test_model
        )
        
        serialized = output.to_serializable()
        
        assert serialized['output_type'] == 'TestDataModel'
        assert serialized['data']['name'] == 'test'
        assert serialized['data']['value'] == 42
    
    def test_serialization_error_handling(self):
        """Test handling of serialization errors"""
        
        # Mock the datetime.utcnow to raise an exception during serialization
        from unittest.mock import patch
        
        output = NodeOutput(
            node_id='test-node',
            data={'test': 'data'}
        )
        
        # Patch datetime to cause a failure
        with patch('src.orchestration.dataflow.datetime') as mock_dt:
            mock_dt.utcnow.side_effect = Exception('Time failure')
            serialized = output.to_serializable()
        
        assert 'error' in serialized
        assert serialized['output_type'] == 'error'


class TestDataFlowManager:
    """Test DataFlowManager"""
    
    @pytest.fixture
    def manager(self):
        """Create DataFlowManager instance"""
        return DataFlowManager(max_concurrent_operations=5)
    
    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager.max_concurrent_operations == 5
        assert len(manager._active_operations) == 0
        assert len(manager._output_buffer) == 0
        assert manager._total_operations == 0
    
    @pytest.mark.asyncio
    async def test_route_output_basic(self, manager):
        """Test basic output routing"""
        
        test_data = {'result': 'success', 'value': 100}
        target_nodes = ['node-2', 'node-3']
        
        # Route output
        output = await manager.route_output(
            node_id='node-1',
            output=test_data,
            target_nodes=target_nodes,
            execution_context={'request_id': 'test-123'}
        )
        
        # Verify output
        assert isinstance(output, NodeOutput)
        assert output.node_id == 'node-1'
        assert output.data == test_data
        assert output.target_nodes == target_nodes
        assert output.request_id == 'test-123'
        assert output.schema_validated is False  # No schema registered
        
        # Verify metrics updated
        assert manager._total_operations == 1
    
    @pytest.mark.asyncio
    async def test_output_validation(self, manager):
        """Test output validation with registered schema"""
        
        # Register schema for node
        manager.register_node_schema('test-node', TestDataModel)
        
        # Valid data
        valid_data = {'name': 'test', 'value': 42}
        
        output = await manager.route_output(
            node_id='test-node',
            output=valid_data
        )
        
        assert output.schema_validated is True
        assert len(output.validation_errors) == 0
        assert isinstance(output.data, TestDataModel)
    
    @pytest.mark.asyncio
    async def test_output_validation_failure(self, manager):
        """Test output validation with invalid data"""
        
        # Register schema for node
        manager.register_node_schema('test-node', TestDataModel)
        
        # Invalid data (missing required fields)
        invalid_data = {'invalid': 'data'}
        
        output = await manager.route_output(
            node_id='test-node',
            output=invalid_data
        )
        
        assert output.schema_validated is False
        assert len(output.validation_errors) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_routing(self, manager):
        """Test concurrent output routing"""
        
        async def route_output(node_id):
            return await manager.route_output(
                node_id=node_id,
                output={'result': f'data-{node_id}'}
            )
        
        # Route multiple outputs concurrently
        tasks = [route_output(f'node-{i}') for i in range(3)]
        outputs = await asyncio.gather(*tasks)
        
        # Verify all outputs processed
        assert len(outputs) == 3
        for i, output in enumerate(outputs):
            assert output.node_id == f'node-{i}'
            assert output.data['result'] == f'data-node-{i}'
        
        assert manager._total_operations == 3
    
    @pytest.mark.asyncio
    async def test_routing_with_error(self, manager):
        """Test routing behavior with errors"""
        
        # Mock routing that fails
        with patch.object(manager, '_perform_routing', side_effect=Exception('Routing failed')):
            output = await manager.route_output(
                node_id='test-node',
                output={'test': 'data'}
            )
            
            assert len(output.validation_errors) > 0
            assert 'Routing error' in output.validation_errors[0]
    
    def test_routing_rule_registration(self, manager):
        """Test routing rule registration"""
        
        rule = {
            'default_targets': ['node-2', 'node-3'],
            'conditions': {'type': 'success'}
        }
        
        manager.register_routing_rule('node-1', rule)
        
        assert 'node-1' in manager._routing_rules
        assert manager._routing_rules['node-1'] == rule
    
    def test_node_output_retrieval(self, manager):
        """Test node output retrieval from buffer"""
        
        # Add output to buffer
        output = NodeOutput(node_id='test-node', data={'result': 'test'})
        manager._output_buffer['test-node:123'] = output
        
        # Retrieve by node and operation ID
        retrieved = manager.get_node_output('test-node', '123')
        assert retrieved == output
        
        # Retrieve most recent for node
        recent = manager.get_node_output('test-node')
        assert recent == output
        
        # Non-existent node
        none_output = manager.get_node_output('non-existent')
        assert none_output is None
    
    def test_metrics_collection(self, manager):
        """Test metrics collection"""
        
        # Update some metrics
        manager._total_operations = 10
        manager._failed_operations = 2
        manager._total_data_transferred_bytes = 1024
        
        metrics = manager.get_metrics()
        
        assert metrics['total_operations'] == 10
        assert metrics['failed_operations'] == 2
        assert metrics['success_rate'] == 0.8  # 8/10
        assert metrics['total_data_transferred_gb'] > 0
    
    def test_buffer_cleanup(self, manager):
        """Test output buffer cleanup"""
        
        # Add old output to buffer
        old_output = NodeOutput(node_id='old-node', data={'old': 'data'})
        old_output.timestamp = datetime.utcnow() - timedelta(hours=2)
        manager._output_buffer['old:123'] = old_output
        
        # Add recent output
        recent_output = NodeOutput(node_id='recent-node', data={'recent': 'data'})
        manager._output_buffer['recent:456'] = recent_output
        
        # Clean up old entries (older than 1 hour)
        removed = manager.clear_buffer(max_age_minutes=60)
        
        assert removed == 1
        assert 'old:123' not in manager._output_buffer
        assert 'recent:456' in manager._output_buffer
    
    @pytest.mark.asyncio
    async def test_manager_cleanup(self, manager):
        """Test manager cleanup"""
        
        # Add some data
        manager._output_buffer['test:123'] = NodeOutput(node_id='test', data={})
        manager._active_operations['op-1'] = Mock()
        
        await manager.cleanup()
        
        assert len(manager._output_buffer) == 0
        assert len(manager._active_operations) == 0


class TestDataFlowOperation:
    """Test DataFlowOperation class"""
    
    def test_operation_creation(self):
        """Test operation creation"""
        
        operation = DataFlowOperation(
            from_node='node-1',
            to_node='node-2',
            data_size_bytes=1024
        )
        
        assert operation.from_node == 'node-1'
        assert operation.to_node == 'node-2'
        assert operation.data_size_bytes == 1024
        assert operation.status == DataFlowStatus.PENDING
        assert operation.operation_id is not None
        assert isinstance(operation.started_at, datetime)
    
    def test_operation_lifecycle(self):
        """Test operation status lifecycle"""
        
        operation = DataFlowOperation()
        
        # Initial state
        assert operation.status == DataFlowStatus.PENDING
        assert operation.completed_at is None
        assert operation.duration_ms == 0.0
        
        # Update to completed
        operation.status = DataFlowStatus.COMPLETED
        operation.completed_at = datetime.utcnow()
        operation.duration_ms = 150.5
        
        assert operation.status == DataFlowStatus.COMPLETED
        assert operation.completed_at is not None
        assert operation.duration_ms == 150.5


@pytest.mark.asyncio
async def test_end_to_end_dataflow():
    """Test end-to-end data flow scenario"""
    
    manager = DataFlowManager()
    
    # Register schemas and routing rules
    manager.register_node_schema('processor', TestDataModel)
    manager.register_routing_rule('processor', {
        'default_targets': ['output', 'logger']
    })
    
    # Route data through processor
    input_data = {'name': 'test-item', 'value': 100, 'data': {'extra': 'info'}}
    
    output = await manager.route_output(
        node_id='processor',
        output=input_data,
        execution_context={'request_id': 'end-to-end-test'}
    )
    
    # Verify processing
    assert output.schema_validated is True
    assert isinstance(output.data, TestDataModel)
    assert output.data.name == 'test-item'
    assert output.data.value == 100
    
    # Verify routing - should use rules and get 2 target nodes
    assert len(output.target_nodes) == 2  # Should use rules: ['output', 'logger']
    
    # Check metrics
    metrics = manager.get_metrics()
    assert metrics['total_operations'] == 1
    assert metrics['registered_schemas'] == 1
    assert metrics['routing_rules'] == 1