"""
Agent framework custom exceptions
"""

from typing import Optional


class AgentError(Exception):
    """Base exception for all agent-related errors"""
    
    def __init__(
        self, 
        message: str, 
        agent_id: Optional[str] = None, 
        request_id: Optional[str] = None
    ):
        super().__init__(message)
        self.agent_id = agent_id
        self.request_id = request_id
        self.message = message

    def __str__(self) -> str:
        context = []
        if self.agent_id:
            context.append(f"agent_id={self.agent_id}")
        if self.request_id:
            context.append(f"request_id={self.request_id}")
        
        context_str = f" ({', '.join(context)})" if context else ""
        return f"{self.message}{context_str}"


class ToolError(AgentError):
    """Exception raised when tool execution fails"""
    
    def __init__(
        self, 
        message: str, 
        tool_name: str, 
        agent_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, agent_id, request_id)
        self.tool_name = tool_name

    def __str__(self) -> str:
        base_str = super().__str__()
        return f"Tool '{self.tool_name}' failed: {base_str}"


class PlanningError(AgentError):
    """Exception raised during plan creation or validation"""
    pass


class CompilationError(AgentError):
    """Exception raised during graph compilation"""
    pass


class ExecutionError(AgentError):
    """Exception raised during plan execution"""
    
    def __init__(
        self, 
        message: str, 
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        super().__init__(message, agent_id, request_id)
        self.task_id = task_id

    def __str__(self) -> str:
        base_str = super().__str__()
        task_info = f" (task_id={self.task_id})" if self.task_id else ""
        return f"Execution failed{task_info}: {base_str}"