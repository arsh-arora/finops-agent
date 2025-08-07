"""
Task Result Streaming to WebSocket
Real-time streaming of Celery task results to WebSocket connections
"""

import asyncio
import json
import structlog
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

import redis.asyncio as redis
from celery import Celery
from celery.events.state import State
from celery.events import EventReceiver

from src.database.redis_client import redis_client
from src.websocket.connection_manager import connection_manager
from .langgraph_runner import ExecutionResult

logger = structlog.get_logger(__name__)


class StreamEventType(str, Enum):
    """Types of streaming events"""
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_RETRYING = "task_retrying"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_PROGRESS = "execution_progress"
    EXECUTION_COMPLETED = "execution_completed"
    NODE_COMPLETED = "node_completed"


@dataclass
class StreamEvent:
    """Event for streaming to WebSocket connections"""
    
    event_type: StreamEventType
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Context information
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    execution_id: Optional[str] = None
    task_id: Optional[str] = None
    node_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Progress information
    progress_percent: Optional[float] = None
    estimated_completion: Optional[datetime] = None
    
    def to_websocket_message(self) -> Dict[str, Any]:
        """Convert to WebSocket message format"""
        
        return {
            'type': 'task_stream_event',
            'event_type': self.event_type.value,
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'request_id': self.request_id,
            'user_id': self.user_id,
            'execution_id': self.execution_id,
            'task_id': self.task_id,
            'node_id': self.node_id,
            'data': self.data,
            'metadata': self.metadata,
            'progress_percent': self.progress_percent,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None
        }


@dataclass
class ExecutionStream:
    """Stream tracking for an execution session"""
    
    execution_id: str
    request_id: str
    user_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    # WebSocket connections
    connection_ids: Set[str] = field(default_factory=set)
    
    # Event tracking
    events_sent: int = 0
    last_event_time: Optional[datetime] = None
    
    # Progress tracking
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    
    def get_progress_percent(self) -> float:
        """Calculate overall progress percentage"""
        if self.total_nodes == 0:
            return 0.0
        return (self.completed_nodes / self.total_nodes) * 100


class ResultStreamingManager:
    """
    Manages real-time streaming of task results to WebSocket connections
    
    Features:
    - Celery result backend event subscription
    - Real-time progress updates
    - Multi-connection broadcasting
    - Event filtering and routing
    - Connection lifecycle management
    """
    
    def __init__(self, celery_app: Celery):
        """
        Initialize result streaming manager
        
        Args:
            celery_app: Celery application instance
        """
        self.celery_app = celery_app
        
        # Event processing
        self._event_receiver: Optional[EventReceiver] = None
        self._event_processing_active = False
        self._event_processing_task: Optional[asyncio.Task] = None
        
        # Stream management
        self._active_streams: Dict[str, ExecutionStream] = {}  # execution_id -> stream
        self._request_streams: Dict[str, str] = {}  # request_id -> execution_id
        
        # Redis pub/sub
        self._pubsub_task: Optional[asyncio.Task] = None
        self._subscription_channels: Set[str] = set()
        
        # Event handlers
        self._event_handlers: Dict[StreamEventType, List[Callable]] = {}
        
        # Performance metrics
        self._streaming_metrics = {
            'events_processed': 0,
            'events_streamed': 0,
            'active_streams': 0,
            'failed_deliveries': 0
        }
        
        logger.info("result_streaming_manager_initialized")

    async def start_streaming(self) -> None:
        """Start result streaming system"""
        
        if self._event_processing_active:
            logger.warning("result_streaming_already_active")
            return
        
        self._event_processing_active = True
        
        # Start Celery event monitoring
        await self._start_celery_event_monitoring()
        
        # Start Redis pub/sub monitoring
        self._pubsub_task = asyncio.create_task(self._redis_pubsub_monitor())
        
        logger.info("result_streaming_started")

    async def stop_streaming(self) -> None:
        """Stop result streaming system"""
        
        self._event_processing_active = False
        
        # Stop Celery event monitoring
        await self._stop_celery_event_monitoring()
        
        # Stop Redis pub/sub
        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass
        
        # Clear active streams
        self._active_streams.clear()
        self._request_streams.clear()
        
        logger.info("result_streaming_stopped")

    async def register_execution_stream(
        self,
        execution_id: str,
        request_id: str,
        user_id: str,
        connection_ids: List[str],
        total_nodes: int = 0
    ) -> None:
        """Register a new execution stream"""
        
        # Create execution stream
        stream = ExecutionStream(
            execution_id=execution_id,
            request_id=request_id,
            user_id=user_id,
            connection_ids=set(connection_ids),
            total_nodes=total_nodes
        )
        
        self._active_streams[execution_id] = stream
        self._request_streams[request_id] = execution_id
        self._streaming_metrics['active_streams'] = len(self._active_streams)
        
        # Subscribe to Redis channels for this execution
        await self._subscribe_to_execution_channels(execution_id, request_id)
        
        # Send initial event
        await self._send_stream_event(StreamEvent(
            event_type=StreamEventType.EXECUTION_STARTED,
            request_id=request_id,
            user_id=user_id,
            execution_id=execution_id,
            data={
                'total_nodes': total_nodes,
                'started_at': datetime.utcnow().isoformat()
            }
        ))
        
        logger.info(
            "execution_stream_registered",
            execution_id=execution_id,
            request_id=request_id,
            user_id=user_id,
            connection_count=len(connection_ids),
            total_nodes=total_nodes
        )

    async def unregister_execution_stream(self, execution_id: str) -> None:
        """Unregister an execution stream"""
        
        if execution_id not in self._active_streams:
            return
        
        stream = self._active_streams[execution_id]
        
        # Unsubscribe from Redis channels
        await self._unsubscribe_from_execution_channels(execution_id, stream.request_id)
        
        # Remove from tracking
        self._active_streams.pop(execution_id, None)
        self._request_streams.pop(stream.request_id, None)
        self._streaming_metrics['active_streams'] = len(self._active_streams)
        
        logger.info(
            "execution_stream_unregistered",
            execution_id=execution_id,
            events_sent=stream.events_sent
        )

    async def stream_execution_result(self, result: ExecutionResult) -> None:
        """Stream execution result to connected clients"""
        
        execution_id = result.execution_id
        
        if execution_id not in self._active_streams:
            logger.debug("no_stream_for_execution", execution_id=execution_id)
            return
        
        stream = self._active_streams[execution_id]
        
        # Create completion event
        event = StreamEvent(
            event_type=StreamEventType.EXECUTION_COMPLETED,
            request_id=stream.request_id,
            user_id=stream.user_id,
            execution_id=execution_id,
            data={
                'success': result.success,
                'final_state': result.final_state,
                'task_results': result.task_results,
                'execution_summary': result.get_summary(),
                'error_message': result.error_message if not result.success else None
            },
            progress_percent=100.0
        )
        
        await self._send_stream_event(event)
        
        # Auto-unregister completed streams
        await self.unregister_execution_stream(execution_id)

    async def stream_node_completion(
        self,
        execution_id: str,
        node_id: str,
        node_result: Any,
        execution_time_ms: float
    ) -> None:
        """Stream node completion event"""
        
        if execution_id not in self._active_streams:
            return
        
        stream = self._active_streams[execution_id]
        stream.completed_nodes += 1
        
        # Create node completion event
        event = StreamEvent(
            event_type=StreamEventType.NODE_COMPLETED,
            request_id=stream.request_id,
            user_id=stream.user_id,
            execution_id=execution_id,
            node_id=node_id,
            data={
                'node_id': node_id,
                'result': self._serialize_node_result(node_result),
                'execution_time_ms': execution_time_ms,
                'completed_at': datetime.utcnow().isoformat()
            },
            progress_percent=stream.get_progress_percent()
        )
        
        await self._send_stream_event(event)

    async def stream_task_progress(
        self,
        task_id: str,
        progress_data: Dict[str, Any]
    ) -> None:
        """Stream task progress update"""
        
        # Find execution stream for this task
        execution_id = self._find_execution_for_task(task_id)
        if not execution_id or execution_id not in self._active_streams:
            return
        
        stream = self._active_streams[execution_id]
        
        # Create progress event
        event = StreamEvent(
            event_type=StreamEventType.TASK_PROGRESS,
            request_id=stream.request_id,
            user_id=stream.user_id,
            execution_id=execution_id,
            task_id=task_id,
            data=progress_data,
            progress_percent=progress_data.get('progress_percent')
        )
        
        await self._send_stream_event(event)

    async def _start_celery_event_monitoring(self) -> None:
        """Start monitoring Celery events"""
        
        try:
            # Create event receiver
            self._event_receiver = EventReceiver(
                self.celery_app.connection(),
                handlers={
                    'task-sent': self._handle_task_sent,
                    'task-started': self._handle_task_started,
                    'task-succeeded': self._handle_task_succeeded,
                    'task-failed': self._handle_task_failed,
                    'task-retry': self._handle_task_retry,
                }
            )
            
            # Start event processing task
            self._event_processing_task = asyncio.create_task(
                self._celery_event_processing_loop()
            )
            
            logger.info("celery_event_monitoring_started")
            
        except Exception as e:
            logger.error("celery_event_monitoring_start_failed", error=str(e))

    async def _stop_celery_event_monitoring(self) -> None:
        """Stop Celery event monitoring"""
        
        if self._event_processing_task:
            self._event_processing_task.cancel()
            try:
                await self._event_processing_task
            except asyncio.CancelledError:
                pass
        
        if self._event_receiver:
            try:
                self._event_receiver.close()
            except:
                pass
        
        logger.info("celery_event_monitoring_stopped")

    async def _celery_event_processing_loop(self) -> None:
        """Process Celery events in async loop"""
        
        while self._event_processing_active:
            try:
                if self._event_receiver:
                    # Process events (this blocks, so run in executor)
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._event_receiver.capture, True, 1.0
                    )
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("celery_event_processing_error", error=str(e))
                await asyncio.sleep(1.0)

    def _handle_task_sent(self, event: Dict[str, Any]) -> None:
        """Handle task sent event"""
        self._streaming_metrics['events_processed'] += 1
        # Task sent events are not typically streamed to clients

    def _handle_task_started(self, event: Dict[str, Any]) -> None:
        """Handle task started event"""
        
        self._streaming_metrics['events_processed'] += 1
        
        task_id = event.get('uuid')
        if not task_id:
            return
        
        # Create and queue stream event
        asyncio.create_task(self._queue_task_event(
            StreamEventType.TASK_STARTED,
            task_id,
            event
        ))

    def _handle_task_succeeded(self, event: Dict[str, Any]) -> None:
        """Handle task succeeded event"""
        
        self._streaming_metrics['events_processed'] += 1
        
        task_id = event.get('uuid')
        if not task_id:
            return
        
        # Create and queue stream event
        asyncio.create_task(self._queue_task_event(
            StreamEventType.TASK_COMPLETED,
            task_id,
            event
        ))

    def _handle_task_failed(self, event: Dict[str, Any]) -> None:
        """Handle task failed event"""
        
        self._streaming_metrics['events_processed'] += 1
        
        task_id = event.get('uuid')
        if not task_id:
            return
        
        # Create and queue stream event
        asyncio.create_task(self._queue_task_event(
            StreamEventType.TASK_FAILED,
            task_id,
            event
        ))

    def _handle_task_retry(self, event: Dict[str, Any]) -> None:
        """Handle task retry event"""
        
        self._streaming_metrics['events_processed'] += 1
        
        task_id = event.get('uuid')
        if not task_id:
            return
        
        # Create and queue stream event
        asyncio.create_task(self._queue_task_event(
            StreamEventType.TASK_RETRYING,
            task_id,
            event
        ))

    async def _queue_task_event(
        self,
        event_type: StreamEventType,
        task_id: str,
        celery_event: Dict[str, Any]
    ) -> None:
        """Queue a task event for streaming"""
        
        # Find execution stream for this task
        execution_id = self._find_execution_for_task(task_id)
        if not execution_id or execution_id not in self._active_streams:
            return
        
        stream = self._active_streams[execution_id]
        
        # Create stream event
        event = StreamEvent(
            event_type=event_type,
            request_id=stream.request_id,
            user_id=stream.user_id,
            execution_id=execution_id,
            task_id=task_id,
            data=self._extract_task_event_data(celery_event),
            metadata={
                'hostname': celery_event.get('hostname'),
                'timestamp': celery_event.get('timestamp')
            }
        )
        
        await self._send_stream_event(event)

    async def _redis_pubsub_monitor(self) -> None:
        """Monitor Redis pub/sub for task results"""
        
        try:
            pubsub = redis_client.redis_client.pubsub()
            
            while self._event_processing_active:
                try:
                    # Subscribe to active channels
                    if self._subscription_channels:
                        for channel in self._subscription_channels:
                            await pubsub.subscribe(channel)
                    
                    # Process messages
                    async for message in pubsub.listen():
                        if message['type'] == 'message':
                            await self._process_redis_message(message)
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error("redis_pubsub_error", error=str(e))
                    await asyncio.sleep(1.0)
        
        except Exception as e:
            logger.error("redis_pubsub_monitor_failed", error=str(e))

    async def _process_redis_message(self, message: Dict[str, Any]) -> None:
        """Process Redis pub/sub message"""
        
        try:
            channel = message['channel'].decode()
            data = json.loads(message['data'])
            
            # Route message based on channel pattern
            if channel.startswith('ws_updates:'):
                request_id = channel.split(':', 1)[1]
                await self._handle_websocket_update(request_id, data)
            
        except Exception as e:
            logger.error("redis_message_processing_failed", error=str(e))

    async def _handle_websocket_update(
        self,
        request_id: str,
        update_data: Dict[str, Any]
    ) -> None:
        """Handle WebSocket update from Redis"""
        
        if request_id not in self._request_streams:
            return
        
        execution_id = self._request_streams[request_id]
        stream = self._active_streams.get(execution_id)
        
        if not stream:
            return
        
        # Forward update to WebSocket connections
        for connection_id in stream.connection_ids:
            await connection_manager.send_personal_message(connection_id, update_data)

    async def _send_stream_event(self, event: StreamEvent) -> None:
        """Send stream event to WebSocket connections"""
        
        if not event.request_id or event.request_id not in self._request_streams:
            return
        
        execution_id = self._request_streams[event.request_id]
        stream = self._active_streams.get(execution_id)
        
        if not stream:
            return
        
        # Convert to WebSocket message
        ws_message = event.to_websocket_message()
        
        # Send to all connected clients
        successful_sends = 0
        
        for connection_id in list(stream.connection_ids):  # Copy to avoid modification during iteration
            try:
                await connection_manager.send_personal_message(connection_id, ws_message)
                successful_sends += 1
            except Exception as e:
                logger.debug("websocket_send_failed", connection_id=connection_id, error=str(e))
                # Remove failed connection
                stream.connection_ids.discard(connection_id)
                self._streaming_metrics['failed_deliveries'] += 1
        
        # Update stream metrics
        stream.events_sent += 1
        stream.last_event_time = datetime.utcnow()
        self._streaming_metrics['events_streamed'] += 1
        
        # Execute event handlers
        await self._execute_event_handlers(event)
        
        logger.debug(
            "stream_event_sent",
            event_type=event.event_type.value,
            event_id=event.event_id,
            execution_id=execution_id,
            successful_sends=successful_sends,
            total_connections=len(stream.connection_ids)
        )

    async def _subscribe_to_execution_channels(
        self,
        execution_id: str,
        request_id: str
    ) -> None:
        """Subscribe to Redis channels for execution"""
        
        channels = [
            f"ws_updates:{request_id}",
            f"task_updates:{execution_id}",
            f"execution_updates:{execution_id}"
        ]
        
        for channel in channels:
            self._subscription_channels.add(channel)
        
        logger.debug("subscribed_to_channels", execution_id=execution_id, channels=channels)

    async def _unsubscribe_from_execution_channels(
        self,
        execution_id: str,
        request_id: str
    ) -> None:
        """Unsubscribe from Redis channels for execution"""
        
        channels = [
            f"ws_updates:{request_id}",
            f"task_updates:{execution_id}",
            f"execution_updates:{execution_id}"
        ]
        
        for channel in channels:
            self._subscription_channels.discard(channel)
        
        logger.debug("unsubscribed_from_channels", execution_id=execution_id, channels=channels)

    def _find_execution_for_task(self, task_id: str) -> Optional[str]:
        """Find execution ID for a task ID"""
        
        # This would typically involve looking up task metadata
        # For now, we'll implement a simple heuristic
        
        # Check if task_id contains execution context
        for execution_id, stream in self._active_streams.items():
            # This is a placeholder - in production, we'd have proper task->execution mapping
            if execution_id in task_id or stream.request_id in task_id:
                return execution_id
        
        return None

    def _serialize_node_result(self, result: Any) -> Any:
        """Serialize node result for streaming"""
        
        try:
            # Handle Pydantic models
            if hasattr(result, 'model_dump'):
                return result.model_dump()
            
            # Handle basic types
            if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                return result
            
            # Fallback to string representation
            return str(result)
            
        except Exception as e:
            logger.debug("node_result_serialization_failed", error=str(e))
            return {"serialization_error": str(e)}

    def _extract_task_event_data(self, celery_event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data from Celery event"""
        
        return {
            'task_name': celery_event.get('name'),
            'args': celery_event.get('args'),
            'kwargs': celery_event.get('kwargs'),
            'result': celery_event.get('result'),
            'exception': celery_event.get('exception'),
            'traceback': celery_event.get('traceback'),
            'runtime': celery_event.get('runtime'),
            'retries': celery_event.get('retries', 0)
        }

    async def _execute_event_handlers(self, event: StreamEvent) -> None:
        """Execute registered event handlers"""
        
        handlers = self._event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error("event_handler_failed", handler=str(handler), error=str(e))

    def register_event_handler(
        self,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None]
    ) -> None:
        """Register event handler"""
        
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(handler)
        
        logger.debug(
            "event_handler_registered",
            event_type=event_type.value,
            handler_count=len(self._event_handlers[event_type])
        )

    def get_streaming_metrics(self) -> Dict[str, Any]:
        """Get streaming performance metrics"""
        
        return {
            **self._streaming_metrics,
            'active_streams_detail': {
                execution_id: {
                    'request_id': stream.request_id,
                    'user_id': stream.user_id,
                    'events_sent': stream.events_sent,
                    'progress_percent': stream.get_progress_percent(),
                    'connection_count': len(stream.connection_ids)
                }
                for execution_id, stream in self._active_streams.items()
            },
            'subscription_channels': len(self._subscription_channels),
            'event_handlers': sum(len(handlers) for handlers in self._event_handlers.values())
        }

    def get_active_streams(self) -> Dict[str, ExecutionStream]:
        """Get active execution streams"""
        return self._active_streams.copy()

    async def cleanup(self) -> None:
        """Cleanup streaming resources"""
        
        await self.stop_streaming()
        
        self._event_handlers.clear()
        self._streaming_metrics.clear()
        
        logger.info("result_streaming_manager_cleanup_completed")