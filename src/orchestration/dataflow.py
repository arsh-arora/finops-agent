"""
Node-to-Node Data Flow Management
Handles routing outputs between nodes with Pydantic validation and serialization
"""

import asyncio
import structlog
from typing import Dict, Any, List, Optional, Union, Type, get_origin, get_args
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from uuid import uuid4, UUID
from enum import Enum

from pydantic import BaseModel, ValidationError, Field
import json

logger = structlog.get_logger(__name__)


class DataFlowStatus(str, Enum):
    """Data flow operation status"""
    PENDING = "pending"
    ROUTING = "routing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class NodeOutput:
    """Structured node output with metadata and routing information"""
    
    # Core data
    node_id: str
    data: Any
    output_type: str = "unknown"
    
    # Routing metadata
    target_nodes: List[str] = field(default_factory=list)
    routing_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Execution context
    execution_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str = ""
    
    # Data validation
    schema_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Performance tracking
    serialization_time_ms: float = 0.0
    routing_time_ms: float = 0.0
    
    def to_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format"""
        try:
            start_time = datetime.utcnow()
            
            # Handle Pydantic models
            if isinstance(self.data, BaseModel):
                serialized_data = self.data.model_dump()
                self.output_type = self.data.__class__.__name__
            # Handle built-in types
            elif isinstance(self.data, (dict, list, str, int, float, bool, type(None))):
                serialized_data = self.data
                self.output_type = type(self.data).__name__
            else:
                # Fallback: attempt JSON serialization
                try:
                    serialized_data = json.loads(json.dumps(self.data, default=str))
                    self.output_type = type(self.data).__name__
                except (TypeError, ValueError):
                    serialized_data = str(self.data)
                    self.output_type = "string_fallback"
            
            # Calculate serialization time
            self.serialization_time_ms = (
                datetime.utcnow() - start_time
            ).total_seconds() * 1000
            
            return {
                "node_id": self.node_id,
                "data": serialized_data,
                "output_type": self.output_type,
                "target_nodes": self.target_nodes,
                "routing_rules": self.routing_rules,
                "execution_id": self.execution_id,
                "timestamp": self.timestamp.isoformat(),
                "request_id": self.request_id,
                "schema_validated": self.schema_validated,
                "validation_errors": self.validation_errors,
                "serialization_time_ms": self.serialization_time_ms
            }
            
        except Exception as e:
            logger.error(
                "node_output_serialization_failed",
                node_id=self.node_id,
                error=str(e)
            )
            return {
                "node_id": self.node_id,
                "data": f"SERIALIZATION_ERROR: {str(e)}",
                "output_type": "error",
                "error": str(e)
            }


@dataclass
class DataFlowOperation:
    """Data flow operation tracking"""
    
    operation_id: str = field(default_factory=lambda: uuid4().hex)
    from_node: str = ""
    to_node: str = ""
    data_size_bytes: int = 0
    status: DataFlowStatus = DataFlowStatus.PENDING
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Error handling
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3


class DataFlowManager:
    """
    Manages data flow between StateGraph nodes
    
    Features:
    - Pydantic model validation and serialization
    - Node output routing with dependency resolution
    - Performance tracking and metrics
    - Error handling and retry logic
    - Request ID propagation for tracing
    """
    
    def __init__(self, max_concurrent_operations: int = 50):
        """
        Initialize data flow manager
        
        Args:
            max_concurrent_operations: Maximum concurrent routing operations
        """
        self.max_concurrent_operations = max_concurrent_operations
        
        # Operation tracking
        self._active_operations: Dict[str, DataFlowOperation] = {}
        self._operation_semaphore = asyncio.Semaphore(max_concurrent_operations)
        
        # Node output buffer
        self._output_buffer: Dict[str, NodeOutput] = {}
        
        # Routing configuration
        self._routing_rules: Dict[str, Dict[str, Any]] = {}
        self._node_schemas: Dict[str, Type[BaseModel]] = {}
        
        # Performance metrics
        self._total_operations = 0
        self._failed_operations = 0
        self._total_data_transferred_bytes = 0
        
        logger.info(
            "dataflow_manager_initialized",
            max_concurrent_operations=max_concurrent_operations
        )

    async def route_output(
        self,
        node_id: str,
        output: Any,
        target_nodes: Optional[List[str]] = None,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> NodeOutput:
        """
        Route node output to dependent nodes with validation and serialization
        
        Args:
            node_id: Source node identifier
            output: Node output data
            target_nodes: Explicit target nodes (overrides graph edges)
            execution_context: Execution context for tracing
            
        Returns:
            NodeOutput with routing metadata and validation results
        """
        operation_id = uuid4().hex
        context = execution_context or {}
        request_id = context.get('request_id', '')
        
        # Create node output
        node_output = NodeOutput(
            node_id=node_id,
            data=output,
            target_nodes=target_nodes or [],
            execution_id=context.get('execution_id', ''),
            request_id=request_id
        )
        
        # Create operation tracking
        operation = DataFlowOperation(
            operation_id=operation_id,
            from_node=node_id,
            to_node=",".join(target_nodes) if target_nodes else "auto",
            status=DataFlowStatus.ROUTING
        )
        
        logger.info(
            "dataflow_routing_started",
            operation_id=operation_id,
            node_id=node_id,
            target_nodes=target_nodes,
            request_id=request_id,
            output_type=type(output).__name__
        )
        
        async with self._operation_semaphore:
            self._active_operations[operation_id] = operation
            
            try:
                # Validate output against schema if available
                await self._validate_node_output(node_output)
                
                # Serialize output data
                serialized = node_output.to_serializable()
                operation.data_size_bytes = len(json.dumps(serialized))
                
                # Route to target nodes
                routing_start = datetime.utcnow()
                await self._perform_routing(node_output, operation)
                node_output.routing_time_ms = (
                    datetime.utcnow() - routing_start
                ).total_seconds() * 1000
                
                # Store in output buffer for downstream consumption
                self._output_buffer[f"{node_id}:{operation_id}"] = node_output
                
                # Mark operation complete
                operation.status = DataFlowStatus.COMPLETED
                operation.completed_at = datetime.utcnow()
                operation.duration_ms = (
                    operation.completed_at - operation.started_at
                ).total_seconds() * 1000
                
                # Update metrics
                self._total_operations += 1
                self._total_data_transferred_bytes += operation.data_size_bytes
                
                logger.info(
                    "dataflow_routing_completed",
                    operation_id=operation_id,
                    node_id=node_id,
                    duration_ms=operation.duration_ms,
                    data_size_bytes=operation.data_size_bytes,
                    target_count=len(node_output.target_nodes),
                    schema_validated=node_output.schema_validated
                )
                
                return node_output
                
            except Exception as e:
                # Handle routing failure
                operation.status = DataFlowStatus.FAILED
                operation.error_message = str(e)
                operation.completed_at = datetime.utcnow()
                operation.duration_ms = (
                    operation.completed_at - operation.started_at
                ).total_seconds() * 1000
                
                self._failed_operations += 1
                
                logger.error(
                    "dataflow_routing_failed",
                    operation_id=operation_id,
                    node_id=node_id,
                    error=str(e),
                    error_type=type(e).__name__
                )
                
                # Add error to node output
                node_output.validation_errors.append(f"Routing error: {str(e)}")
                
                return node_output
                
            finally:
                self._active_operations.pop(operation_id, None)

    async def _validate_node_output(self, node_output: NodeOutput) -> None:
        """Validate node output against registered schema"""
        
        node_id = node_output.node_id
        
        # Check if we have a schema for this node
        if node_id not in self._node_schemas:
            # No schema registered - skip validation
            node_output.schema_validated = False
            return
        
        schema_class = self._node_schemas[node_id]
        
        try:
            # Attempt to validate using Pydantic model
            if isinstance(node_output.data, BaseModel):
                # Data is already a Pydantic model
                validated_data = node_output.data
            else:
                # Try to create model from data
                validated_data = schema_class(**node_output.data) if isinstance(node_output.data, dict) else schema_class(node_output.data)
            
            # Replace data with validated version
            node_output.data = validated_data
            node_output.schema_validated = True
            
            logger.debug(
                "node_output_validation_success",
                node_id=node_id,
                schema_class=schema_class.__name__
            )
            
        except ValidationError as e:
            # Validation failed - log errors but continue
            validation_errors = []
            for error in e.errors():
                validation_errors.append(f"{error['loc']}: {error['msg']}")
            
            node_output.validation_errors = validation_errors
            node_output.schema_validated = False
            
            logger.warning(
                "node_output_validation_failed",
                node_id=node_id,
                schema_class=schema_class.__name__,
                errors=validation_errors
            )
            
        except Exception as e:
            # Unexpected validation error
            node_output.validation_errors = [f"Validation exception: {str(e)}"]
            node_output.schema_validated = False
            
            logger.error(
                "node_output_validation_error",
                node_id=node_id,
                error=str(e)
            )

    async def _perform_routing(
        self, 
        node_output: NodeOutput, 
        operation: DataFlowOperation
    ) -> None:
        """Perform actual routing of data to target nodes"""
        
        node_id = node_output.node_id
        
        # Get target nodes from explicit list or routing rules
        target_nodes = node_output.target_nodes
        if not target_nodes and node_id in self._routing_rules:
            routing_rule = self._routing_rules[node_id]
            target_nodes = routing_rule.get('default_targets', [])
            # Update node output with resolved targets
            node_output.target_nodes = target_nodes
            node_output.routing_rules = routing_rule
        
        if not target_nodes:
            logger.debug(
                "no_routing_targets",
                node_id=node_id,
                operation_id=operation.operation_id
            )
            return
        
        # Route to each target node
        routing_tasks = []
        for target_node in target_nodes:
            task = self._route_to_single_node(node_output, target_node, operation)
            routing_tasks.append(task)
        
        # Wait for all routing operations to complete
        if routing_tasks:
            await asyncio.gather(*routing_tasks, return_exceptions=True)
        
        # Update target nodes in output
        node_output.target_nodes = target_nodes

    async def _route_to_single_node(
        self,
        node_output: NodeOutput,
        target_node: str,
        operation: DataFlowOperation
    ) -> None:
        """Route data to a single target node"""
        
        try:
            logger.debug(
                "routing_to_node",
                from_node=node_output.node_id,
                to_node=target_node,
                operation_id=operation.operation_id,
                request_id=node_output.request_id
            )
            
            # This is where we would actually deliver data to the target node
            # In a real implementation, this might involve:
            # - Queuing data for the target node
            # - Updating graph state
            # - Triggering target node execution
            
            # For now, we'll simulate the routing with logging
            await asyncio.sleep(0.001)  # Simulate routing latency
            
        except Exception as e:
            logger.error(
                "single_node_routing_failed",
                from_node=node_output.node_id,
                to_node=target_node,
                error=str(e)
            )
            raise

    def register_node_schema(self, node_id: str, schema: Type[BaseModel]) -> None:
        """Register Pydantic schema for node output validation"""
        
        self._node_schemas[node_id] = schema
        
        logger.debug(
            "node_schema_registered",
            node_id=node_id,
            schema_class=schema.__name__
        )

    def register_routing_rule(
        self,
        node_id: str,
        rule: Dict[str, Any]
    ) -> None:
        """Register routing rule for a node"""
        
        self._routing_rules[node_id] = rule
        
        logger.debug(
            "routing_rule_registered",
            node_id=node_id,
            rule=rule
        )

    def get_node_output(
        self, 
        node_id: str, 
        operation_id: Optional[str] = None
    ) -> Optional[NodeOutput]:
        """Get cached node output"""
        
        if operation_id:
            key = f"{node_id}:{operation_id}"
            return self._output_buffer.get(key)
        
        # Find most recent output for node
        matching_keys = [k for k in self._output_buffer.keys() if k.startswith(f"{node_id}:")]
        if not matching_keys:
            return None
        
        # Return most recent (assuming keys are chronologically ordered)
        latest_key = max(matching_keys)
        return self._output_buffer[latest_key]

    def get_operation_status(self, operation_id: str) -> Optional[DataFlowOperation]:
        """Get status of a data flow operation"""
        return self._active_operations.get(operation_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Get data flow performance metrics"""
        
        active_operations = len(self._active_operations)
        success_rate = (
            (self._total_operations - self._failed_operations) / self._total_operations
            if self._total_operations > 0 else 0.0
        )
        
        return {
            "total_operations": self._total_operations,
            "failed_operations": self._failed_operations,
            "active_operations": active_operations,
            "success_rate": round(success_rate, 3),
            "total_data_transferred_gb": round(
                self._total_data_transferred_bytes / (1024 ** 3), 6
            ),
            "buffer_size": len(self._output_buffer),
            "registered_schemas": len(self._node_schemas),
            "routing_rules": len(self._routing_rules)
        }

    async def cleanup(self) -> None:
        """Cleanup resources and clear buffers"""
        
        # Cancel active operations
        for operation in self._active_operations.values():
            # Operations would be cancelled here in production
            pass
        
        # Clear buffers
        self._output_buffer.clear()
        self._active_operations.clear()
        
        logger.info("dataflow_manager_cleanup_completed")

    def clear_buffer(self, max_age_minutes: int = 60) -> int:
        """Clear old entries from output buffer"""
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        removed_count = 0
        
        keys_to_remove = []
        for key, output in self._output_buffer.items():
            if output.timestamp < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._output_buffer[key]
            removed_count += 1
        
        if removed_count > 0:
            logger.info(
                "output_buffer_cleaned",
                removed_entries=removed_count,
                remaining_entries=len(self._output_buffer)
            )
        
        return removed_count