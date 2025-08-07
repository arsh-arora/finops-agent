"""
Planning system exceptions
"""

from agents.base.exceptions import AgentError


class PlanningError(AgentError):
    """Base exception for planning failures"""
    pass


class BudgetExceededError(PlanningError):
    """Exception raised when estimated cost exceeds user budget"""
    
    def __init__(self, estimated_cost: float, budget: float, **kwargs):
        self.estimated_cost = estimated_cost
        self.budget = budget
        super().__init__(
            f"Estimated cost ${estimated_cost:.4f} exceeds budget ${budget:.2f}",
            **kwargs
        )


class InvalidRequestError(PlanningError):
    """Exception raised when request fails schema validation"""
    pass


class ToolNotFoundError(PlanningError):
    """Exception raised when plan references unknown tools"""
    
    def __init__(self, tool_name: str, available_tools: list, **kwargs):
        self.tool_name = tool_name
        self.available_tools = available_tools
        super().__init__(
            f"Tool '{tool_name}' not found. Available: {', '.join(available_tools)}",
            **kwargs
        )