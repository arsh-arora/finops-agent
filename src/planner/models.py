"""
Planning system models and configuration
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class CostModel(BaseModel):
    """Cost calculation parameters for different services"""
    
    # Token pricing (per 1K tokens)
    gpt4_mini_input_cost: float = Field(default=0.00015, description="GPT-4 Mini input cost per 1K tokens")
    gpt4_mini_output_cost: float = Field(default=0.0006, description="GPT-4 Mini output cost per 1K tokens")
    
    # Memory graph costs
    graph_hop_base_cost: float = Field(default=0.0001, description="Base cost per memory graph hop")
    memory_retrieval_cost: float = Field(default=0.0005, description="Cost per memory retrieval operation")
    memory_storage_cost: float = Field(default=0.0002, description="Cost per memory storage operation")
    
    # Tool execution estimates
    default_tool_cost: float = Field(default=0.001, description="Default tool execution cost estimate")
    tool_specific_costs: Dict[str, float] = Field(
        default_factory=lambda: {
            "analyze_costs": 0.005,      # Higher cost for complex analysis
            "track_budget": 0.002,       # Medium cost for budget tracking
            "general_help": 0.0005,      # Low cost for simple help
            "answer_question": 0.001     # Standard cost for Q&A
        },
        description="Tool-specific cost overrides"
    )
    
    def calculate_llm_cost(self, input_tokens: int, output_tokens: int, model: str = "gpt-4o-mini") -> float:
        """Calculate LLM usage cost"""
        if model == "gpt-4o-mini":
            input_cost = (input_tokens / 1000) * self.gpt4_mini_input_cost
            output_cost = (output_tokens / 1000) * self.gpt4_mini_output_cost
            return input_cost + output_cost
        
        # Default fallback
        return ((input_tokens + output_tokens) / 1000) * 0.0006
    
    def calculate_memory_cost(self, retrievals: int, hops: int, stores: int = 1) -> float:
        """Calculate memory system costs"""
        retrieval_cost = retrievals * self.memory_retrieval_cost
        hop_cost = hops * self.graph_hop_base_cost
        storage_cost = stores * self.memory_storage_cost
        return retrieval_cost + hop_cost + storage_cost
    
    def get_tool_cost(self, tool_name: str) -> float:
        """Get estimated cost for specific tool"""
        return self.tool_specific_costs.get(tool_name, self.default_tool_cost)


class PlanningConfig(BaseModel):
    """Configuration for planning system behavior"""
    
    # LLM settings
    planning_model: str = Field(default="gpt-4o-mini", description="Model for plan generation")
    planning_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Temperature for planning LLM")
    max_planning_tokens: int = Field(default=1000, ge=100, le=4000, description="Max tokens for planning response")
    
    # Safety limits
    max_tasks_per_plan: int = Field(default=10, ge=1, le=50, description="Maximum tasks allowed in single plan")
    max_budget_usd: float = Field(default=10.0, ge=0.01, le=100.0, description="Maximum budget allowed")
    default_task_timeout: int = Field(default=300, ge=30, le=3600, description="Default task timeout in seconds")
    
    # Memory integration
    memory_retrieval_limit: int = Field(default=5, ge=1, le=20, description="Max memories to retrieve for context")
    memory_graph_hop_limit: int = Field(default=3, ge=1, le=10, description="Max graph hops for memory traversal")
    
    # Cost thresholds
    cost_estimation_confidence: float = Field(default=0.8, ge=0.1, le=1.0, description="Required confidence for cost estimates")
    budget_safety_margin: float = Field(default=0.1, ge=0.0, le=0.5, description="Safety margin for budget calculations")
    
    # Optimization settings
    enable_task_optimization: bool = Field(default=True, description="Enable task reordering optimization")
    enable_parallel_grouping: bool = Field(default=True, description="Enable parallel task grouping")
    cost_optimization_weight: float = Field(default=0.7, ge=0.0, le=1.0, description="Weight for cost vs speed optimization")