"""
E2E Test Helper Functions

Provides utility functions for common E2E testing patterns including
condition waiting, event sequence validation, performance measurement,
and metric extraction.
"""

import asyncio
import time
import logging
from typing import Any, Callable, List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


async def wait_for_condition(
    condition_func: Callable[[], bool],
    timeout: int = 10,
    check_interval: float = 0.1,
    timeout_message: str = "Condition not met within timeout"
) -> bool:
    """
    Wait for a condition to become true within timeout.
    
    Args:
        condition_func: Function that returns bool when condition is met
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        timeout_message: Message to raise if timeout occurs
        
    Returns:
        True if condition met, raises TimeoutError if not
        
    Raises:
        TimeoutError: If condition not met within timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            if condition_func():
                return True
        except Exception as e:
            logger.debug(f"Condition check failed: {e}")
            
        await asyncio.sleep(check_interval)
    
    raise TimeoutError(timeout_message)


async def wait_for_async_condition(
    condition_func: Callable[[], Any],
    timeout: int = 10,
    check_interval: float = 0.1,
    timeout_message: str = "Async condition not met within timeout"
) -> bool:
    """
    Wait for an async condition to become true within timeout.
    
    Args:
        condition_func: Async function that returns truthy value when condition is met
        timeout: Maximum time to wait in seconds  
        check_interval: Time between checks in seconds
        timeout_message: Message to raise if timeout occurs
        
    Returns:
        True if condition met
        
    Raises:
        TimeoutError: If condition not met within timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            result = await condition_func()
            if result:
                return True
        except Exception as e:
            logger.debug(f"Async condition check failed: {e}")
            
        await asyncio.sleep(check_interval)
    
    raise TimeoutError(timeout_message)


def assert_event_sequence(
    events: List[Dict[str, Any]],
    expected_sequence: List[str],
    allow_extra_events: bool = True,
    strict_order: bool = True
) -> None:
    """
    Assert that events follow expected sequence.
    
    Args:
        events: List of event dictionaries
        expected_sequence: Expected event type sequence
        allow_extra_events: Whether to allow events not in expected sequence
        strict_order: Whether events must be in exact order
        
    Raises:
        AssertionError: If sequence doesn't match expectations
    """
    event_types = [event.get('type', event.get('event', 'unknown')) for event in events]
    
    if not allow_extra_events and len(event_types) != len(expected_sequence):
        raise AssertionError(
            f"Event count mismatch: expected {len(expected_sequence)}, got {len(event_types)}\n"
            f"Expected: {expected_sequence}\n"
            f"Actual: {event_types}"
        )
    
    if strict_order:
        # Check that expected events appear in order
        expected_index = 0
        for event_type in event_types:
            if expected_index < len(expected_sequence) and event_type == expected_sequence[expected_index]:
                expected_index += 1
                
        if expected_index != len(expected_sequence):
            missing_events = expected_sequence[expected_index:]
            raise AssertionError(
                f"Missing events in sequence: {missing_events}\n"
                f"Expected: {expected_sequence}\n"
                f"Actual: {event_types}"
            )
    else:
        # Check that all expected events are present (order doesn't matter)
        missing_events = set(expected_sequence) - set(event_types)
        if missing_events:
            raise AssertionError(
                f"Missing events: {missing_events}\n"
                f"Expected: {expected_sequence}\n"
                f"Actual: {event_types}"
            )


def measure_ttfb(
    start_time: float,
    events: List[Dict[str, Any]],
    first_response_types: Optional[List[str]] = None
) -> Optional[float]:
    """
    Measure Time to First Byte (TTFB) from events.
    
    Args:
        start_time: Request start time
        events: List of received events
        first_response_types: Event types that count as first response
        
    Returns:
        TTFB in milliseconds, None if no valid first response found
    """
    if not first_response_types:
        first_response_types = [
            'partial_result',
            'node_started',
            'agent_thinking',
            'streaming_started'
        ]
    
    for event in events:
        event_type = event.get('type', event.get('event', ''))
        if event_type in first_response_types:
            event_timestamp = event.get('timestamp')
            if event_timestamp:
                try:
                    # Handle ISO timestamp
                    if isinstance(event_timestamp, str):
                        event_time = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00')).timestamp()
                    else:
                        event_time = float(event_timestamp)
                    
                    ttfb_ms = (event_time - start_time) * 1000
                    return ttfb_ms
                except (ValueError, TypeError):
                    continue
    
    return None


def extract_execution_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract execution metrics from event sequence.
    
    Args:
        events: List of execution events
        
    Returns:
        Dictionary with execution metrics
    """
    metrics = {
        'total_events': len(events),
        'event_types': {},
        'node_executions': {},
        'parallel_groups': 0,
        'memory_operations': 0,
        'agent_switches': 0,
        'error_count': 0,
        'execution_timeline': []
    }
    
    # Track node execution times
    node_start_times = {}
    
    for i, event in enumerate(events):
        event_type = event.get('type', event.get('event', 'unknown'))
        event_timestamp = event.get('timestamp', time.time())
        node_id = event.get('node_id')
        
        # Count event types
        metrics['event_types'][event_type] = metrics['event_types'].get(event_type, 0) + 1
        
        # Track timeline
        metrics['execution_timeline'].append({
            'sequence': i,
            'event_type': event_type,
            'timestamp': event_timestamp,
            'node_id': node_id
        })
        
        # Track node execution
        if event_type == 'node_started' and node_id:
            try:
                if isinstance(event_timestamp, str):
                    start_time = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00')).timestamp()
                else:
                    start_time = float(event_timestamp)
                node_start_times[node_id] = start_time
            except (ValueError, TypeError):
                pass
                
        elif event_type == 'node_completed' and node_id and node_id in node_start_times:
            try:
                if isinstance(event_timestamp, str):
                    end_time = datetime.fromisoformat(event_timestamp.replace('Z', '+00:00')).timestamp()
                else:
                    end_time = float(event_timestamp)
                    
                execution_time = (end_time - node_start_times[node_id]) * 1000
                metrics['node_executions'][node_id] = {
                    'execution_time_ms': execution_time,
                    'start_time': node_start_times[node_id],
                    'end_time': end_time
                }
            except (ValueError, TypeError):
                pass
        
        # Count specific event types
        if event_type == 'parallel_group_started':
            metrics['parallel_groups'] += 1
            
        elif 'memory' in event_type.lower():
            metrics['memory_operations'] += 1
            
        elif event_type == 'agent_selected':
            metrics['agent_switches'] += 1
            
        elif event_type in ['error', 'node_failed', 'execution_failed']:
            metrics['error_count'] += 1
    
    # Calculate summary statistics
    if metrics['node_executions']:
        execution_times = [n['execution_time_ms'] for n in metrics['node_executions'].values()]
        metrics['avg_node_execution_ms'] = sum(execution_times) / len(execution_times)
        metrics['max_node_execution_ms'] = max(execution_times)
        metrics['min_node_execution_ms'] = min(execution_times)
    
    return metrics


def assert_performance_requirements(
    metrics: Dict[str, Any],
    max_ttfb_ms: Optional[float] = None,
    max_total_execution_ms: Optional[float] = None,
    max_node_execution_ms: Optional[float] = None,
    min_parallel_efficiency: Optional[float] = None
) -> None:
    """
    Assert that performance requirements are met.
    
    Args:
        metrics: Execution metrics dictionary
        max_ttfb_ms: Maximum allowed TTFB in milliseconds
        max_total_execution_ms: Maximum allowed total execution time
        max_node_execution_ms: Maximum allowed individual node execution time  
        min_parallel_efficiency: Minimum parallel execution efficiency (0-1)
        
    Raises:
        AssertionError: If performance requirements not met
    """
    if max_ttfb_ms is not None and 'ttfb_ms' in metrics:
        ttfb = metrics['ttfb_ms']
        assert ttfb <= max_ttfb_ms, f"TTFB {ttfb:.1f}ms exceeds limit of {max_ttfb_ms}ms"
    
    if max_total_execution_ms is not None and 'total_execution_ms' in metrics:
        total_time = metrics['total_execution_ms']
        assert total_time <= max_total_execution_ms, \
            f"Total execution {total_time:.1f}ms exceeds limit of {max_total_execution_ms}ms"
    
    if max_node_execution_ms is not None and 'max_node_execution_ms' in metrics:
        max_node_time = metrics['max_node_execution_ms']
        assert max_node_time <= max_node_execution_ms, \
            f"Max node execution {max_node_time:.1f}ms exceeds limit of {max_node_execution_ms}ms"
    
    if min_parallel_efficiency is not None:
        # Calculate parallel efficiency if we have the data
        parallel_groups = metrics.get('parallel_groups', 0)
        total_nodes = len(metrics.get('node_executions', {}))
        
        if parallel_groups > 0 and total_nodes > 0:
            efficiency = parallel_groups / total_nodes
            assert efficiency >= min_parallel_efficiency, \
                f"Parallel efficiency {efficiency:.2f} below minimum {min_parallel_efficiency}"


def find_events_by_criteria(
    events: List[Dict[str, Any]],
    event_type: Optional[str] = None,
    node_id: Optional[str] = None,
    has_field: Optional[str] = None,
    field_value: Optional[Tuple[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Find events matching specific criteria.
    
    Args:
        events: List of events to search
        event_type: Filter by event type
        node_id: Filter by node ID
        has_field: Filter events that have specific field
        field_value: Filter by field value (field_name, expected_value)
        
    Returns:
        List of matching events
    """
    matching_events = []
    
    for event in events:
        # Check event type
        if event_type is not None:
            if event.get('type', event.get('event', '')) != event_type:
                continue
        
        # Check node ID
        if node_id is not None:
            if event.get('node_id') != node_id:
                continue
        
        # Check field presence
        if has_field is not None:
            if has_field not in event:
                continue
        
        # Check field value
        if field_value is not None:
            field_name, expected_value = field_value
            if event.get(field_name) != expected_value:
                continue
        
        matching_events.append(event)
    
    return matching_events


def validate_event_data(
    event: Dict[str, Any],
    required_fields: List[str],
    optional_fields: Optional[List[str]] = None,
    field_types: Optional[Dict[str, type]] = None
) -> bool:
    """
    Validate event data structure.
    
    Args:
        event: Event to validate
        required_fields: Fields that must be present
        optional_fields: Fields that may be present
        field_types: Expected types for fields
        
    Returns:
        True if validation passes
        
    Raises:
        AssertionError: If validation fails
    """
    # Check required fields
    for field in required_fields:
        assert field in event, f"Required field '{field}' missing from event"
    
    # Check field types
    if field_types:
        for field, expected_type in field_types.items():
            if field in event:
                actual_type = type(event[field])
                assert actual_type == expected_type or isinstance(event[field], expected_type), \
                    f"Field '{field}' has type {actual_type}, expected {expected_type}"
    
    # Check for unexpected fields (if optional_fields specified)
    if optional_fields is not None:
        allowed_fields = set(required_fields + optional_fields)
        actual_fields = set(event.keys())
        unexpected_fields = actual_fields - allowed_fields
        
        assert not unexpected_fields, f"Unexpected fields in event: {unexpected_fields}"
    
    return True


def create_test_message(
    message_type: str = "user_message",
    text: str = "Test message",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized test message.
    
    Args:
        message_type: Type of message
        text: Message text content
        metadata: Additional metadata
        
    Returns:
        Formatted message dictionary
    """
    import uuid
    from datetime import datetime, timezone
    
    message = {
        "type": message_type,
        "text": text,
        "metadata": {
            "source": "e2e_test",
            "test_generated": True,
            **(metadata or {})
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_id": str(uuid.uuid4())
    }
    
    return message


def format_duration(milliseconds: float) -> str:
    """Format duration in milliseconds to human readable string."""
    if milliseconds < 1000:
        return f"{milliseconds:.1f}ms"
    elif milliseconds < 60000:
        return f"{milliseconds/1000:.1f}s"
    else:
        minutes = int(milliseconds / 60000)
        seconds = (milliseconds % 60000) / 1000
        return f"{minutes}m {seconds:.1f}s"


def summarize_test_results(
    test_name: str,
    metrics: Dict[str, Any],
    success: bool,
    duration_ms: float
) -> str:
    """
    Create summary of test results for reporting.
    
    Args:
        test_name: Name of the test
        metrics: Test execution metrics
        success: Whether test passed
        duration_ms: Total test duration
        
    Returns:
        Formatted summary string
    """
    status = "✅ PASSED" if success else "❌ FAILED"
    duration_str = format_duration(duration_ms)
    
    summary = f"{status} {test_name} ({duration_str})\n"
    
    if 'ttfb_ms' in metrics:
        summary += f"  TTFB: {format_duration(metrics['ttfb_ms'])}\n"
        
    if 'node_executions' in metrics:
        node_count = len(metrics['node_executions'])
        summary += f"  Nodes executed: {node_count}\n"
        
    if 'parallel_groups' in metrics and metrics['parallel_groups'] > 0:
        summary += f"  Parallel groups: {metrics['parallel_groups']}\n"
        
    if 'error_count' in metrics and metrics['error_count'] > 0:
        summary += f"  Errors: {metrics['error_count']}\n"
    
    return summary