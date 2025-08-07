"""
Tool registry and @tool decorator for agent framework
Auto-generates JSON schemas from function signatures for planner integration
"""

import inspect
import logging
from typing import Callable, Any, Dict, Optional, List, get_type_hints, Union
from functools import wraps
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None, 
    schema: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Decorator to mark agent methods as tools available for execution
    
    Args:
        name: Optional tool name (defaults to function name)
        description: Optional tool description (defaults to docstring)
        schema: Optional manual JSON schema (auto-generated if not provided)
    
    Returns:
        Decorated function with tool metadata
        
    Example:
        @tool(description="Calculate cost optimization recommendations")
        async def optimize_costs(self, resource_type: str, budget_limit: float) -> str:
            return "optimization_result"
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Import here to avoid circular imports
                from .exceptions import ToolError
                
                agent_instance = args[0] if args else None
                agent_id = getattr(agent_instance, 'agent_id', 'unknown') if agent_instance else 'unknown'
                
                logger.error(
                    "tool_execution_failed",
                    tool_name=func.__name__,
                    agent_id=agent_id,
                    error=str(e)
                )
                
                raise ToolError(
                    f"Tool execution failed: {str(e)}",
                    tool_name=func.__name__,
                    agent_id=agent_id
                )
        
        # Attach tool metadata
        wrapper._tool_name = name or func.__name__
        wrapper._tool_description = description or func.__doc__ or f"Tool: {func.__name__}"
        wrapper._tool_schema = schema or _generate_schema_from_signature(func)
        wrapper._is_tool = True
        wrapper._original_func = func
        
        logger.debug(
            "tool_decorated",
            tool_name=wrapper._tool_name,
            has_custom_schema=schema is not None
        )
        
        return wrapper
    
    return decorator


class ToolRegistry:
    """Registry for managing agent tools and schema generation"""
    
    @staticmethod
    def extract_tools(agent_instance: Any) -> Dict[str, Callable]:
        """
        Extract all @tool decorated methods from an agent instance
        
        Args:
            agent_instance: Agent instance to extract tools from
            
        Returns:
            Dictionary mapping tool names to callable functions
        """
        tools = {}
        
        # Inspect all methods of the agent instance
        for attr_name in dir(agent_instance):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(agent_instance, attr_name)
            
            # Check if it's a tool-decorated method
            # Must be callable, have _is_tool=True, and _tool_name that's a string
            if (callable(attr) and 
                hasattr(attr, '_is_tool') and 
                attr._is_tool == True and
                hasattr(attr, '_tool_name') and
                isinstance(attr._tool_name, str)):
                
                tool_name = attr._tool_name
                tools[tool_name] = attr
                
                logger.debug(
                    "tool_extracted",
                    agent_type=agent_instance.__class__.__name__,
                    agent_id=getattr(agent_instance, 'agent_id', 'unknown'),
                    tool_name=tool_name
                )
        
        logger.info(
            "tools_extraction_completed",
            agent_type=agent_instance.__class__.__name__,
            agent_id=getattr(agent_instance, 'agent_id', 'unknown'),
            tools_count=len(tools),
            tool_names=list(tools.keys())
        )
        
        return tools

    @staticmethod
    def generate_schema(tools: Dict[str, Callable]) -> Dict[str, Any]:
        """
        Generate JSON schema for all tools for planner integration
        
        Args:
            tools: Dictionary of tool names to callable functions
            
        Returns:
            JSON schema dictionary for all tools
        """
        schema = {
            "type": "object",
            "properties": {
                "tools": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "parameters": {"type": "object"}
                        },
                        "required": ["name", "description", "parameters"]
                    }
                }
            },
            "tools_definitions": {}
        }
        
        tools_list = []
        
        for tool_name, tool_func in tools.items():
            tool_schema = {
                "name": tool_name,
                "description": getattr(tool_func, '_tool_description', f"Tool: {tool_name}"),
                "parameters": getattr(tool_func, '_tool_schema', {})
            }
            
            tools_list.append(tool_schema)
            schema["tools_definitions"][tool_name] = tool_schema
            
        schema["properties"]["tools"]["items"]["enum"] = tools_list
        
        logger.debug(
            "schema_generated",
            tools_count=len(tools),
            schema_size=len(str(schema))
        )
        
        return schema

    @staticmethod
    def validate_tool_call(
        tool_name: str, 
        parameters: Dict[str, Any], 
        tools: Dict[str, Callable]
    ) -> bool:
        """
        Validate a tool call against available tools and their schemas
        
        Args:
            tool_name: Name of tool to call
            parameters: Parameters for the tool call
            tools: Available tools dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if tool_name not in tools:
            logger.warning(
                "tool_validation_failed",
                tool_name=tool_name,
                reason="tool_not_found",
                available_tools=list(tools.keys())
            )
            return False
            
        tool_func = tools[tool_name]
        tool_schema = getattr(tool_func, '_tool_schema', {})
        
        # Basic parameter validation against schema
        if 'properties' in tool_schema:
            required_params = tool_schema.get('required', [])
            
            # Check required parameters are present
            for required_param in required_params:
                if required_param not in parameters:
                    logger.warning(
                        "tool_validation_failed",
                        tool_name=tool_name,
                        reason="missing_required_parameter",
                        missing_param=required_param
                    )
                    return False
        
        logger.debug(
            "tool_validation_passed",
            tool_name=tool_name,
            parameters_count=len(parameters)
        )
        
        return True


def _generate_schema_from_signature(func: Callable) -> Dict[str, Any]:
    """
    Auto-generate JSON schema from function signature and type hints
    
    Args:
        func: Function to generate schema for
        
    Returns:
        JSON schema dictionary
    """
    try:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter
            if param_name == 'self':
                continue
                
            param_schema = {}
            param_type = type_hints.get(param_name, str)
            
            # Map Python types to JSON schema types
            param_schema["type"] = _map_python_type_to_json_type(param_type)
            
            # Add description from parameter annotation if available
            if hasattr(param, 'annotation') and hasattr(param.annotation, '__doc__'):
                param_schema["description"] = param.annotation.__doc__
            
            # Handle default values
            if param.default != inspect.Parameter.empty:
                param_schema["default"] = param.default
            else:
                # Required parameter
                schema["required"].append(param_name)
            
            schema["properties"][param_name] = param_schema
        
        logger.debug(
            "schema_auto_generated",
            function_name=func.__name__,
            parameters_count=len(schema["properties"]),
            required_count=len(schema["required"])
        )
        
        return schema
        
    except Exception as e:
        logger.warning(
            "schema_generation_failed",
            function_name=func.__name__,
            error=str(e)
        )
        # Return basic schema as fallback
        return {
            "type": "object",
            "properties": {},
            "required": []
        }


def _map_python_type_to_json_type(python_type: Any) -> str:
    """
    Map Python types to JSON schema types
    
    Args:
        python_type: Python type to map
        
    Returns:
        JSON schema type string
    """
    # Handle Union types (Optional[T] is Union[T, None])
    if hasattr(python_type, '__origin__'):
        if python_type.__origin__ is Union:
            # Get the non-None type from Optional
            non_none_types = [t for t in python_type.__args__ if t != type(None)]
            if non_none_types:
                return _map_python_type_to_json_type(non_none_types[0])
        elif python_type.__origin__ is list:
            return "array"
        elif python_type.__origin__ is dict:
            return "object"
    
    # Basic type mapping
    type_mapping = {
        str: "string",
        int: "integer", 
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        List: "array",
        Dict: "object"
    }
    
    # Handle enum types
    if isinstance(python_type, type) and issubclass(python_type, Enum):
        return "string"
    
    return type_mapping.get(python_type, "string")