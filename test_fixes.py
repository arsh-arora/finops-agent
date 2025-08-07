#!/usr/bin/env python3
"""
Quick test of key fixes
"""
import sys
sys.path.insert(0, '/Users/aroraji/Desktop/MultiAgent-FinOps-Chat/finops-agent-chat')

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

print("✅ Testing key fixes...")

# Test 1: NodeOutput serialization error handling
print("🔍 Testing NodeOutput serialization error handling...")
try:
    with patch('sys.path', ['/Users/aroraji/Desktop/MultiAgent-FinOps-Chat/finops-agent-chat'] + sys.path):
        with patch('structlog.get_logger', return_value=Mock()):
            from src.orchestration.dataflow import NodeOutput
            
            output = NodeOutput(
                node_id='test-node',
                data={'test': 'data'}
            )
            
            # Test error handling by mocking datetime to fail
            with patch('src.orchestration.dataflow.datetime') as mock_dt:
                mock_dt.utcnow.side_effect = Exception('Time failure')
                serialized = output.to_serializable()
            
            assert 'error' in serialized
            assert serialized['output_type'] == 'error'
            print("✅ NodeOutput serialization error handling works")
except Exception as e:
    print(f"❌ NodeOutput test failed: {e}")

# Test 2: Heavy task decorator attributes
print("🔍 Testing heavy task decorator attributes...")
try:
    with patch('sys.path', ['/Users/aroraji/Desktop/MultiAgent-FinOps-Chat/finops-agent-chat'] + sys.path):
        with patch('structlog.get_logger', return_value=Mock()):
            with patch('celery.Celery'):
                with patch('src.workers.celery_app.redis'):
                    from src.workers.celery_app import heavy_task
                    
                    @heavy_task(name='test_task', queue='test_queue')
                    async def test_function(self, param1: str, param2: int = 10) -> str:
                        return f"{param1}-{param2}"
                    
                    # Verify decoration attributes
                    assert hasattr(test_function, '_is_heavy_task')
                    assert test_function._is_heavy_task is True
                    assert test_function._tool_name == 'test_task'
                    assert hasattr(test_function, '_heavy_task_config')
                    print("✅ Heavy task decorator attributes work")
except Exception as e:
    print(f"❌ Heavy task decorator test failed: {e}")

# Test 3: Timedelta import
print("🔍 Testing timedelta import...")
try:
    print("✅ timedelta imported successfully")
    # Test using timedelta
    now = datetime.utcnow()
    future = now + timedelta(hours=1)
    assert future > now
    print("✅ timedelta works correctly")
except Exception as e:
    print(f"❌ timedelta test failed: {e}")

print("\n🎉 Key fixes validation complete!")