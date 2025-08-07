"""
E2E WebSocket Client for Testing

Provides a comprehensive WebSocket client specifically designed for end-to-end
testing of the FinOps Agent Chat system. Supports connection management,
message streaming, event tracking, and performance monitoring.
"""

import asyncio
import json
import time
import websockets
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from urllib.parse import urlencode
import uuid

logger = logging.getLogger(__name__)


@dataclass
class WebSocketEvent:
    """Represents a WebSocket event with metadata."""
    event_type: str
    data: Dict[str, Any]
    timestamp: float
    sequence_number: int
    node_id: Optional[str] = None
    execution_id: Optional[str] = None


@dataclass
class ConnectionStats:
    """Connection statistics and metrics."""
    connected_at: float
    messages_sent: int = 0
    messages_received: int = 0
    events_received: int = 0
    connection_errors: int = 0
    last_activity: float = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()


@dataclass
class ExecutionMetrics:
    """Execution performance metrics."""
    ttfb_ms: Optional[float] = None  # Time to first byte
    total_execution_ms: Optional[float] = None
    node_execution_times: Dict[str, float] = field(default_factory=dict)
    event_sequence: List[str] = field(default_factory=list)
    parallel_node_count: int = 0
    memory_operations: int = 0
    agent_switches: int = 0


class E2EWebSocketClient:
    """
    End-to-end WebSocket client for comprehensive system testing.
    
    Features:
    - JWT authentication via query parameters
    - Event-driven message handling with filtering
    - Performance monitoring and metrics collection  
    - Streaming execution tracking
    - Connection health monitoring
    - Deterministic test support
    """
    
    def __init__(
        self, 
        base_url: str = "ws://localhost:8000/api/v1/ws/chat",
        timeout: int = 30
    ):
        """
        Initialize E2E WebSocket client.
        
        Args:
            base_url: WebSocket endpoint URL
            timeout: Default timeout for operations
        """
        self.base_url = base_url
        self.timeout = timeout
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        
        # Event tracking
        self.events: List[WebSocketEvent] = []
        self.event_filters: Dict[str, Callable[[Dict[str, Any]], bool]] = {}
        self.event_handlers: Dict[str, Callable[[WebSocketEvent], None]] = {}
        self.sequence_counter = 0
        
        # Connection management
        self.stats = ConnectionStats(connected_at=0)
        self.is_connected = False
        self.connection_id: Optional[str] = None
        self.user_id: Optional[str] = None
        
        # Execution tracking
        self.current_execution: Optional[Dict[str, Any]] = None
        self.execution_metrics = ExecutionMetrics()
        
        # Message queue for testing
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.background_task: Optional[asyncio.Task] = None
        
    async def connect(self, jwt_token: str, conversation_id: Optional[str] = None) -> None:
        """
        Connect to WebSocket with JWT authentication.
        
        Args:
            jwt_token: JWT authentication token
            conversation_id: Optional conversation ID for context
            
        Raises:
            ConnectionError: If connection fails
            ValueError: If token is invalid
        """
        try:
            # Build connection URL with query parameters
            params = {"token": jwt_token}
            if conversation_id:
                params["conversation_id"] = conversation_id
                
            url = f"{self.base_url}?{urlencode(params)}"
            
            logger.info(f"Connecting to WebSocket: {url}")
            
            # Establish WebSocket connection
            self.websocket = await websockets.connect(
                url,
                timeout=self.timeout,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_connected = True
            self.stats.connected_at = time.time()
            self.stats.update_activity()
            
            # Start background message handler
            self.background_task = asyncio.create_task(self._message_handler())
            
            # Wait for welcome message
            welcome_event = await self.wait_for_event("connection_established", timeout=5)
            self.connection_id = welcome_event.data.get("connection_id")
            
            logger.info(f"WebSocket connected successfully, connection_id: {self.connection_id}")
            
        except Exception as e:
            self.stats.connection_errors += 1
            logger.error(f"WebSocket connection failed: {e}")
            raise ConnectionError(f"Failed to connect to WebSocket: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket and cleanup resources."""
        if self.background_task and not self.background_task.done():
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
                
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            
        self.is_connected = False
        logger.info(f"WebSocket disconnected, connection_id: {self.connection_id}")
    
    async def send_message(
        self, 
        message: Dict[str, Any], 
        context_id: Optional[str] = None
    ) -> None:
        """
        Send message to WebSocket server.
        
        Args:
            message: Message data to send
            context_id: Optional context identifier for tracking
            
        Raises:
            RuntimeError: If not connected
        """
        if not self.is_connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
            
        # Add metadata if missing
        if "timestamp" not in message:
            message["timestamp"] = datetime.now(timezone.utc).isoformat()
            
        if "message_id" not in message:
            message["message_id"] = str(uuid.uuid4())
            
        if context_id:
            message["context_id"] = context_id
        
        # Send message
        message_json = json.dumps(message)
        await self.websocket.send(message_json)
        
        # Update stats
        self.stats.messages_sent += 1
        self.stats.total_bytes_sent += len(message_json)
        self.stats.update_activity()
        
        logger.debug(f"Sent message: {message.get('type', 'unknown')} - {message.get('text', '')[:50]}")
    
    async def receive_frames(self, timeout: int = 10) -> List[Dict[str, Any]]:
        """
        Receive all available WebSocket frames within timeout.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            List of received message frames
        """
        frames = []
        end_time = time.time() + timeout
        
        while time.time() < end_time:
            try:
                # Check message queue with short timeout
                remaining_time = end_time - time.time()
                if remaining_time <= 0:
                    break
                    
                frame = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=min(remaining_time, 0.1)
                )
                frames.append(frame)
                
            except asyncio.TimeoutError:
                # Continue checking for more messages
                continue
                
        return frames
    
    async def wait_for_event(
        self, 
        event_type: str, 
        timeout: int = 5,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> WebSocketEvent:
        """
        Wait for specific event type with optional filtering.
        
        Args:
            event_type: Event type to wait for
            timeout: Timeout in seconds
            filter_func: Optional filter function for event data
            
        Returns:
            Matching WebSocketEvent
            
        Raises:
            TimeoutError: If event not received within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check existing events first
            for event in self.events:
                if event.event_type == event_type:
                    if not filter_func or filter_func(event.data):
                        return event
            
            # Wait for new messages
            try:
                await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
                
        raise TimeoutError(f"Event '{event_type}' not received within {timeout} seconds")
    
    async def stream_execution(
        self, 
        message: Dict[str, Any], 
        context_id: Optional[str] = None
    ) -> AsyncGenerator[WebSocketEvent, None]:
        """
        Send message and stream execution events as they arrive.
        
        Args:
            message: Message to send
            context_id: Optional context identifier
            
        Yields:
            WebSocketEvent objects as execution progresses
        """
        # Start execution tracking
        execution_id = str(uuid.uuid4())
        self.current_execution = {
            'execution_id': execution_id,
            'start_time': time.time(),
            'context_id': context_id
        }
        self.execution_metrics = ExecutionMetrics()
        
        # Send initial message
        await self.send_message(message, context_id)
        
        # Track TTFB
        ttfb_start = time.time()
        first_response_received = False
        
        # Stream events until completion
        while True:
            try:
                # Wait for next event
                frame = await asyncio.wait_for(self.message_queue.get(), timeout=30)
                
                # Track TTFB on first response
                if not first_response_received and frame.get('type') != 'message_received':
                    self.execution_metrics.ttfb_ms = (time.time() - ttfb_start) * 1000
                    first_response_received = True
                
                # Create event object
                event = self._frame_to_event(frame)
                
                # Update execution metrics
                self._update_execution_metrics(event)
                
                yield event
                
                # Check for completion
                if self._is_execution_complete(event):
                    self.execution_metrics.total_execution_ms = \
                        (time.time() - self.current_execution['start_time']) * 1000
                    break
                    
            except asyncio.TimeoutError:
                logger.warning("Execution streaming timeout - ending stream")
                break
    
    async def execute_workflow(
        self, 
        message: Dict[str, Any],
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Execute complete workflow and return final result.
        
        Args:
            message: Message to execute
            timeout: Execution timeout
            
        Returns:
            Final execution result with metrics
        """
        execution_events = []
        
        async for event in self.stream_execution(message):
            execution_events.append(event)
            
            # Break on timeout
            if event.timestamp - execution_events[0].timestamp > timeout:
                raise TimeoutError(f"Workflow execution exceeded {timeout} seconds")
        
        # Extract final result
        completion_events = [
            e for e in execution_events 
            if e.event_type in ['graph_completed', 'execution_completed']
        ]
        
        final_result = completion_events[-1].data if completion_events else {}
        
        return {
            'result': final_result,
            'execution_events': execution_events,
            'metrics': self.execution_metrics,
            'stats': self.get_connection_stats()
        }
    
    def add_event_filter(self, event_type: str, filter_func: Callable[[Dict[str, Any]], bool]) -> None:
        """Add event filter for specific event type."""
        self.event_filters[event_type] = filter_func
    
    def add_event_handler(self, event_type: str, handler: Callable[[WebSocketEvent], None]) -> None:
        """Add event handler for specific event type.""" 
        self.event_handlers[event_type] = handler
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            'connected_at': self.stats.connected_at,
            'connection_duration': time.time() - self.stats.connected_at if self.stats.connected_at > 0 else 0,
            'messages_sent': self.stats.messages_sent,
            'messages_received': self.stats.messages_received,
            'events_received': self.stats.events_received,
            'connection_errors': self.stats.connection_errors,
            'bytes_sent': self.stats.total_bytes_sent,
            'bytes_received': self.stats.total_bytes_received,
            'connection_id': self.connection_id,
            'is_connected': self.is_connected
        }
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get current execution metrics."""
        return {
            'ttfb_ms': self.execution_metrics.ttfb_ms,
            'total_execution_ms': self.execution_metrics.total_execution_ms,
            'node_execution_times': self.execution_metrics.node_execution_times,
            'event_sequence': self.execution_metrics.event_sequence,
            'parallel_node_count': self.execution_metrics.parallel_node_count,
            'memory_operations': self.execution_metrics.memory_operations,
            'agent_switches': self.execution_metrics.agent_switches
        }
    
    def get_events_by_type(self, event_type: str) -> List[WebSocketEvent]:
        """Get all events of specific type."""
        return [e for e in self.events if e.event_type == event_type]
    
    async def ping(self) -> float:
        """Send ping and measure response time."""
        if not self.is_connected:
            raise RuntimeError("WebSocket not connected")
            
        ping_start = time.time()
        ping_message = {
            "type": "ping",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await self.send_message(ping_message)
        
        # Wait for pong response
        pong_event = await self.wait_for_event("pong", timeout=5)
        ping_time = (time.time() - ping_start) * 1000
        
        return ping_time
    
    async def _message_handler(self) -> None:
        """Background task to handle incoming WebSocket messages."""
        try:
            while self.is_connected and self.websocket:
                try:
                    # Receive message with timeout
                    message_raw = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=1.0
                    )
                    
                    # Parse JSON
                    message = json.loads(message_raw)
                    
                    # Update stats
                    self.stats.messages_received += 1
                    self.stats.total_bytes_received += len(message_raw)
                    self.stats.update_activity()
                    
                    # Create event
                    event = self._frame_to_event(message)
                    
                    # Apply filters
                    event_type = event.event_type
                    if event_type in self.event_filters:
                        if not self.event_filters[event_type](event.data):
                            continue
                    
                    # Store event
                    self.events.append(event)
                    self.stats.events_received += 1
                    
                    # Call handlers
                    if event_type in self.event_handlers:
                        self.event_handlers[event_type](event)
                    
                    # Add to message queue for receive_frames
                    await self.message_queue.put(message)
                    
                except asyncio.TimeoutError:
                    # Normal timeout - continue listening
                    continue
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info("WebSocket connection closed by server")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
                    self.stats.connection_errors += 1
                    
        except Exception as e:
            logger.error(f"Message handler crashed: {e}")
        finally:
            self.is_connected = False
    
    def _frame_to_event(self, frame: Dict[str, Any]) -> WebSocketEvent:
        """Convert WebSocket frame to WebSocketEvent."""
        self.sequence_counter += 1
        
        return WebSocketEvent(
            event_type=frame.get("type", "unknown"),
            data=frame,
            timestamp=time.time(),
            sequence_number=self.sequence_counter,
            node_id=frame.get("node_id"),
            execution_id=frame.get("execution_id")
        )
    
    def _update_execution_metrics(self, event: WebSocketEvent) -> None:
        """Update execution metrics based on event."""
        # Track event sequence
        self.execution_metrics.event_sequence.append(event.event_type)
        
        # Track node execution times
        if event.event_type == "node_started" and event.node_id:
            # Store start time
            setattr(self.execution_metrics, f"_node_start_{event.node_id}", event.timestamp)
            
        elif event.event_type == "node_completed" and event.node_id:
            # Calculate execution time
            start_attr = f"_node_start_{event.node_id}"
            if hasattr(self.execution_metrics, start_attr):
                start_time = getattr(self.execution_metrics, start_attr)
                execution_time = (event.timestamp - start_time) * 1000
                self.execution_metrics.node_execution_times[event.node_id] = execution_time
        
        # Track parallel execution
        if event.event_type == "parallel_group_started":
            self.execution_metrics.parallel_node_count += event.data.get("node_count", 0)
        
        # Track memory operations
        if "memory" in event.event_type.lower():
            self.execution_metrics.memory_operations += 1
            
        # Track agent switches
        if event.event_type == "agent_selected":
            self.execution_metrics.agent_switches += 1
    
    def _is_execution_complete(self, event: WebSocketEvent) -> bool:
        """Check if execution is complete based on event."""
        completion_events = [
            "graph_completed",
            "execution_completed", 
            "workflow_completed",
            "error",
            "execution_failed"
        ]
        return event.event_type in completion_events
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()