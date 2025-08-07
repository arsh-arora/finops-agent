"""
Unit tests for tool registry and @tool decorator
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Optional

from agents.base.registry import tool, ToolRegistry
from agents.base.exceptions import ToolError


class TestToolClass:
    """Test class with various tool decorations"""
    
    def __init__(self):
        self.agent_id = "test_agent"
    
    @tool()
    async def simple_tool(self, input_text: str) -> str:
        """Simple tool with basic types"""
        return f"Result: {input_text}"
    
    @tool(name="custom_name", description="Custom tool description")
    async def renamed_tool(self, value: int, optional_param: Optional[str] = None) -> str:
        """Tool with custom name and optional parameters"""
        return f"Value: {value}, Optional: {optional_param}"
    
    @tool(schema={
        "type": "object",
        "properties": {
            "manual_param": {"type": "string", "description": "Manually defined parameter"}
        },
        "required": ["manual_param"]
    })
    async def manual_schema_tool(self, manual_param: str) -> str:
        """Tool with manually defined schema"""
        return f"Manual: {manual_param}"
    
    @tool()
    async def error_tool(self) -> str:
        """Tool that raises an exception"""
        raise ValueError("Test error")
    
    def regular_method(self) -> str:
        """Regular method without @tool decorator"""
        return "not a tool"


@pytest.mark.unit
class TestToolDecorator:
    """Test @tool decorator functionality"""
    
    @pytest.mark.asyncio
    async def test_simple_tool_decoration(self):
        """Test basic @tool decoration"""
        instance = TestToolClass()
        
        # Test tool is callable and works
        result = await instance.simple_tool("test input")
        assert result == "Result: test input"
        
        # Test tool metadata is attached
        assert hasattr(instance.simple_tool, '_is_tool')
        assert instance.simple_tool._is_tool == True
        assert instance.simple_tool._tool_name == "simple_tool"
        assert "Simple tool with basic types" in instance.simple_tool._tool_description
        assert isinstance(instance.simple_tool._tool_schema, dict)
    
    @pytest.mark.asyncio  
    async def test_custom_name_and_description(self):
        """Test @tool with custom name and description"""
        instance = TestToolClass()
        
        result = await instance.renamed_tool(42, "optional_value")
        assert result == "Value: 42, Optional: optional_value"
        
        # Test custom metadata
        assert instance.renamed_tool._tool_name == "custom_name"
        assert instance.renamed_tool._tool_description == "Custom tool description"
    
    @pytest.mark.asyncio
    async def test_manual_schema(self):
        """Test @tool with manually provided schema"""
        instance = TestToolClass()
        
        result = await instance.manual_schema_tool("manual_value")
        assert result == "Manual: manual_value"
        
        # Test manual schema is used
        schema = instance.manual_schema_tool._tool_schema
        assert schema["properties"]["manual_param"]["type"] == "string"
        assert "manual_param" in schema["required"]
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test @tool error handling and ToolError raising"""
        instance = TestToolClass()
        
        with pytest.raises(ToolError) as exc_info:
            await instance.error_tool()
        
        assert "Test error" in str(exc_info.value)
        assert exc_info.value.tool_name == "error_tool"
        assert exc_info.value.agent_id == "test_agent"
    
    def test_auto_schema_generation(self):
        """Test automatic schema generation from function signature"""
        instance = TestToolClass()
        
        schema = instance.simple_tool._tool_schema
        
        # Should have input_text parameter
        assert "properties" in schema
        assert "input_text" in schema["properties"]
        assert schema["properties"]["input_text"]["type"] == "string"
        assert "input_text" in schema["required"]
    
    def test_optional_parameter_handling(self):
        """Test optional parameters in auto-generated schema"""
        instance = TestToolClass()
        
        schema = instance.renamed_tool._tool_schema
        
        # Required parameter
        assert "value" in schema["properties"]
        assert "value" in schema["required"]
        
        # Optional parameter
        assert "optional_param" in schema["properties"] 
        assert "optional_param" not in schema["required"]


@pytest.mark.unit 
class TestToolRegistry:
    """Test ToolRegistry functionality"""
    
    def test_extract_tools(self):
        """Test tool extraction from agent instance"""
        instance = TestToolClass()
        
        tools = ToolRegistry.extract_tools(instance)
        
        # Should extract all @tool decorated methods
        assert len(tools) == 4  # simple_tool, custom_name, manual_schema_tool, error_tool
        assert "simple_tool" in tools
        assert "custom_name" in tools  # renamed tool
        assert "manual_schema_tool" in tools
        assert "error_tool" in tools
        
        # Should not include regular methods
        assert "regular_method" not in tools
        
        # Tools should be callable
        assert callable(tools["simple_tool"])
    
    def test_generate_schema(self):
        """Test schema generation for all tools"""
        instance = TestToolClass()
        tools = ToolRegistry.extract_tools(instance)
        
        schema = ToolRegistry.generate_schema(tools)
        
        # Check schema structure
        assert "tools_definitions" in schema
        assert "properties" in schema
        assert "tools" in schema["properties"]
        
        # Check individual tool definitions
        assert "simple_tool" in schema["tools_definitions"]
        assert "custom_name" in schema["tools_definitions"]  # renamed tool
        
        # Check tool definition structure
        simple_def = schema["tools_definitions"]["simple_tool"]
        assert "name" in simple_def
        assert "description" in simple_def
        assert "parameters" in simple_def
    
    def test_validate_tool_call_success(self):
        """Test successful tool call validation"""
        instance = TestToolClass()
        tools = ToolRegistry.extract_tools(instance)
        
        # Valid tool call
        is_valid = ToolRegistry.validate_tool_call(
            tool_name="simple_tool",
            parameters={"input_text": "test"},
            tools=tools
        )
        
        assert is_valid == True
    
    def test_validate_tool_call_missing_tool(self):
        """Test validation failure for non-existent tool"""
        instance = TestToolClass()
        tools = ToolRegistry.extract_tools(instance)
        
        # Non-existent tool
        is_valid = ToolRegistry.validate_tool_call(
            tool_name="non_existent_tool",
            parameters={},
            tools=tools
        )
        
        assert is_valid == False
    
    def test_validate_tool_call_missing_parameter(self):
        """Test validation failure for missing required parameter"""
        instance = TestToolClass()
        tools = ToolRegistry.extract_tools(instance)
        
        # Missing required parameter
        is_valid = ToolRegistry.validate_tool_call(
            tool_name="simple_tool", 
            parameters={},  # missing input_text
            tools=tools
        )
        
        assert is_valid == False
    
    def test_validate_tool_call_optional_parameter(self):
        """Test validation success with optional parameter missing"""
        instance = TestToolClass()
        tools = ToolRegistry.extract_tools(instance)
        
        # Optional parameter missing should still be valid
        is_valid = ToolRegistry.validate_tool_call(
            tool_name="custom_name",  # renamed_tool
            parameters={"value": 42},  # optional_param missing
            tools=tools
        )
        
        assert is_valid == True


@pytest.mark.unit
class TestSchemaGeneration:
    """Test automatic schema generation utilities"""
    
    def test_basic_type_mapping(self):
        """Test basic Python type to JSON schema mapping"""
        from agents.base.registry import _map_python_type_to_json_type
        
        assert _map_python_type_to_json_type(str) == "string"
        assert _map_python_type_to_json_type(int) == "integer"
        assert _map_python_type_to_json_type(float) == "number"
        assert _map_python_type_to_json_type(bool) == "boolean"
        assert _map_python_type_to_json_type(list) == "array"
        assert _map_python_type_to_json_type(dict) == "object"
    
    def test_optional_type_mapping(self):
        """Test Optional[Type] mapping"""
        from agents.base.registry import _map_python_type_to_json_type
        from typing import Optional
        
        # Optional[str] should map to "string"
        assert _map_python_type_to_json_type(Optional[str]) == "string"
        assert _map_python_type_to_json_type(Optional[int]) == "integer"
    
    def test_schema_generation_with_defaults(self):
        """Test schema generation includes default values"""
        @tool()
        async def tool_with_defaults(required_param: str, optional_param: str = "default_value") -> str:
            return "result"
        
        schema = tool_with_defaults._tool_schema
        
        # Required parameter
        assert "required_param" in schema["required"]
        assert "default" not in schema["properties"]["required_param"]
        
        # Optional parameter with default
        assert "optional_param" not in schema["required"]
        assert schema["properties"]["optional_param"]["default"] == "default_value"