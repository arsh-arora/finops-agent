"""
LLM-Aware Agent Planner
Translates natural-language instructions into ExecutionPlan objects
"""

import json
import time
import structlog
from typing import Dict, Any, List, Optional
from uuid import uuid4

from memory import FinOpsMemoryService
from agents.models import ChatRequest, ExecutionPlan, Task, Dependency, CostEstimate, PlanningResult
from agents.base.agent import HardenedAgent
from .models import CostModel, PlanningConfig
from .exceptions import PlanningError, BudgetExceededError, InvalidRequestError, ToolNotFoundError

logger = structlog.get_logger(__name__)


class AgentPlanner:
    """
    LLM-aware planner that creates ExecutionPlan objects from natural language requests
    
    Features:
    - Single LLM call with mem0-aware context
    - JSON schema validation for inputs and outputs
    - Cost estimation with token and graph hop modeling
    - Safety checks for budget and tool availability
    """
    
    def __init__(
        self, 
        memory_service: FinOpsMemoryService,
        llm_client=None,
        config: Optional[PlanningConfig] = None,
        cost_model: Optional[CostModel] = None
    ):
        """
        Initialize planner with memory service and LLM client
        
        Args:
            memory_service: Memory service for context retrieval
            llm_client: LLM client for plan generation
            config: Planning configuration
            cost_model: Cost calculation model
        """
        self.memory_service = memory_service
        self.llm_client = llm_client
        self.config = config or PlanningConfig()
        self.cost_model = cost_model or CostModel()
        
        logger.info(
            "agent_planner_initialized",
            has_llm_client=llm_client is not None,
            max_tasks=self.config.max_tasks_per_plan,
            max_budget=self.config.max_budget_usd
        )

    async def create_plan(
        self, 
        request: ChatRequest, 
        agent: HardenedAgent
    ) -> PlanningResult:
        """
        Create execution plan from chat request
        
        Args:
            request: Validated chat request
            agent: Agent that will execute the plan
            
        Returns:
            PlanningResult with ExecutionPlan or error details
        """
        start_time = time.time()
        request_id = request.request_id
        
        logger.info(
            "plan_creation_started",
            request_id=request_id,
            user_id=request.user_id,
            agent_domain=agent.get_domain(),
            budget_usd=request.budget_usd
        )
        
        try:
            # Validate request schema
            validated_request = self._validate_request(request)
            
            # Retrieve memory context
            memory_context = await self._get_memory_context(validated_request)
            
            # Get agent capabilities and tools
            agent_capabilities = agent.get_capabilities()
            agent_tools = agent.get_tool_schema()
            
            # Generate plan using LLM
            if self.llm_client:
                plan = await self._llm_generate_plan(
                    validated_request, 
                    memory_context,
                    agent_capabilities,
                    agent_tools
                )
            else:
                # Fallback to simple plan generation
                plan = await self._fallback_generate_plan(
                    validated_request,
                    agent_capabilities,
                    list(agent.get_tools().keys())
                )
            
            # Validate tools are available
            self._validate_plan_tools(plan, agent.get_tools())
            
            # Calculate costs
            cost_estimate = await self._calculate_plan_cost(
                plan, 
                len(memory_context),
                validated_request
            )
            plan.cost = cost_estimate
            
            # Safety checks
            self._check_budget_limits(cost_estimate.usd, validated_request.budget_usd)
            self._check_plan_limits(plan)
            
            # Optimize plan if enabled
            if self.config.enable_task_optimization:
                plan = self._optimize_plan(plan)
            
            planning_time = (time.time() - start_time) * 1000
            
            logger.info(
                "plan_creation_completed",
                request_id=request_id,
                tasks_count=len(plan.tasks),
                dependencies_count=len(plan.dependencies),
                estimated_cost=cost_estimate.usd,
                planning_time_ms=planning_time
            )
            
            return PlanningResult(
                plan=plan,
                success=True,
                planning_time_ms=planning_time,
                llm_tokens_used=getattr(plan, '_planning_tokens', 0),
                memory_queries=len(memory_context)
            )
            
        except Exception as e:
            planning_time = (time.time() - start_time) * 1000
            
            logger.error(
                "plan_creation_failed",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
                planning_time_ms=planning_time
            )
            
            return PlanningResult(
                plan=None,
                success=False,
                error_message=str(e),
                planning_time_ms=planning_time
            )

    async def _get_memory_context(self, request: ChatRequest) -> List[str]:
        """Retrieve relevant memory context for planning"""
        try:
            memories = await self.memory_service.retrieve_relevant_memories(
                query=request.message,
                user_id=request.user_id,
                limit=self.config.memory_retrieval_limit
            )
            
            context = []
            if memories:
                for memory in memories:
                    context.append(memory.content)
                    
            logger.debug(
                "memory_context_retrieved",
                request_id=request.request_id,
                memories_count=len(context)
            )
            
            return context
            
        except Exception as e:
            logger.warning(
                "memory_context_retrieval_failed",
                request_id=request.request_id,
                error=str(e)
            )
            return []

    async def _llm_generate_plan(
        self,
        request: ChatRequest,
        memory_context: List[str],
        agent_capabilities: List[str],
        agent_tools: Dict[str, Any]
    ) -> ExecutionPlan:
        """Generate plan using LLM with mem0 context"""
        
        # Build planning prompt
        planning_prompt = self._build_planning_prompt(
            request,
            memory_context,
            agent_capabilities,
            agent_tools
        )
        
        try:
            # Call LLM for plan generation
            response = await self.llm_client.complete(
                messages=[{"role": "user", "content": planning_prompt}],
                model=self.config.planning_model,
                max_tokens=self.config.max_planning_tokens,
                temperature=self.config.planning_temperature
            )
            
            # Parse LLM response into ExecutionPlan
            plan = self._parse_planning_response(response, request)
            
            # Store planning tokens used
            plan._planning_tokens = len(response.split())
            
            return plan
            
        except Exception as e:
            logger.error(
                "llm_plan_generation_failed",
                request_id=request.request_id,
                error=str(e)
            )
            
            # Fallback to simple plan
            return await self._fallback_generate_plan(
                request,
                agent_capabilities,
                list(agent_tools.get('tools_definitions', {}).keys())
            )

    def _build_planning_prompt(
        self,
        request: ChatRequest,
        memory_context: List[str],
        agent_capabilities: List[str],
        agent_tools: Dict[str, Any]
    ) -> str:
        """Build comprehensive planning prompt for LLM"""
        
        # Format memory context
        memory_str = ""
        if memory_context:
            memory_str = f"\n\nRelevant Memory Context:\n" + "\n".join([
                f"- {context[:200]}..." if len(context) > 200 else f"- {context}"
                for context in memory_context[:3]  # Limit context to avoid token bloat
            ])
        
        # Format agent capabilities
        capabilities_str = ", ".join(agent_capabilities)
        
        # Format available tools
        tools_str = ""
        if 'tools_definitions' in agent_tools:
            tool_list = []
            for tool_name, tool_info in agent_tools['tools_definitions'].items():
                description = tool_info.get('description', 'No description')
                tool_list.append(f"- {tool_name}: {description}")
            tools_str = "\n".join(tool_list)
        
        return f"""You are an intelligent agent planner. Create an execution plan for the user's request.

User Request: "{request.message}"
User Budget: ${request.budget_usd:.2f}
Agent Capabilities: {capabilities_str}
{memory_str}

Available Tools:
{tools_str}

Instructions:
1. Analyze the request and break it down into specific, actionable tasks
2. Each task should use one of the available tools
3. Consider dependencies between tasks (sequential or data dependencies)  
4. Estimate token usage for each task realistically
5. Keep total estimated cost under ${request.budget_usd:.2f}
6. Group independent tasks for parallel execution where beneficial

Respond with a JSON object in this exact format:
{{
  "tasks": [
    {{
      "tool_name": "tool_name",
      "inputs": {{"parameter": "value"}},
      "estimate_tokens": 150,
      "name": "Human readable task name",
      "description": "What this task accomplishes",
      "parallel_group_id": "group1" // Optional, for parallel tasks
    }}
  ],
  "dependencies": [
    {{
      "parent_task_index": 0,
      "child_task_index": 1,
      "dependency_type": "sequential"
    }}
  ],
  "reasoning": "Brief explanation of the plan approach"
}}

Requirements:
- Maximum {self.config.max_tasks_per_plan} tasks
- Use only the listed available tools
- Provide realistic token estimates (typical range: 50-500 per task)
- Include reasoning for your planning decisions"""

    def _parse_planning_response(self, llm_response: str, request: ChatRequest) -> ExecutionPlan:
        """Parse LLM response into ExecutionPlan object"""
        try:
            # Extract JSON from response
            response_text = llm_response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(response_text)
            
            # Create tasks from parsed response
            tasks = []
            for i, task_data in enumerate(parsed.get("tasks", [])):
                task = Task(
                    tool_name=task_data["tool_name"],
                    inputs=task_data.get("inputs", {}),
                    estimate_tokens=task_data.get("estimate_tokens", 100),
                    name=task_data.get("name", f"Task {i+1}"),
                    description=task_data.get("description", ""),
                    parallel_group_id=task_data.get("parallel_group_id")
                )
                tasks.append(task)
            
            # Create dependencies from parsed response  
            dependencies = []
            for dep_data in parsed.get("dependencies", []):
                parent_idx = dep_data["parent_task_index"]
                child_idx = dep_data["child_task_index"]
                
                if 0 <= parent_idx < len(tasks) and 0 <= child_idx < len(tasks):
                    dependency = Dependency(
                        parent_id=tasks[parent_idx].id,
                        child_id=tasks[child_idx].id,
                        dependency_type=dep_data.get("dependency_type", "sequential")
                    )
                    dependencies.append(dependency)
            
            # Create ExecutionPlan
            plan = ExecutionPlan(
                tasks=tasks,
                dependencies=dependencies,
                cost=CostEstimate(tokens=0, usd=0.0),  # Will be calculated separately
                request_id=request.request_id,
                user_id=request.user_id,
                agent_domain="unknown",  # Will be set by caller
                agent_id="unknown",      # Will be set by caller
                original_message=request.message,
                planning_reasoning=parsed.get("reasoning", "LLM-generated plan")
            )
            
            return plan
            
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(
                "llm_response_parse_failed",
                request_id=request.request_id,
                response=llm_response[:500],
                error=str(e)
            )
            
            # Fallback to simple single-task plan
            return ExecutionPlan(
                tasks=[Task(
                    tool_name="general_help",
                    inputs={"query": request.message},
                    estimate_tokens=100,
                    name="General assistance",
                    description="Provide general help for the user's request"
                )],
                dependencies=[],
                cost=CostEstimate(tokens=100, usd=0.001),
                request_id=request.request_id,
                user_id=request.user_id,
                agent_domain="default",
                agent_id="fallback",
                original_message=request.message,
                planning_reasoning="Fallback plan due to LLM parsing failure"
            )

    async def _fallback_generate_plan(
        self,
        request: ChatRequest,
        agent_capabilities: List[str],
        available_tools: List[str]
    ) -> ExecutionPlan:
        """Generate simple fallback plan when LLM is unavailable"""
        
        # Select appropriate tool based on capabilities
        selected_tool = "general_help"  # Default fallback
        
        if "cost_analysis" in agent_capabilities and any("cost" in request.message.lower() for _ in [True]):
            selected_tool = "analyze_costs" if "analyze_costs" in available_tools else selected_tool
        elif "budget_tracking" in agent_capabilities and "budget" in request.message.lower():
            selected_tool = "track_budget" if "track_budget" in available_tools else selected_tool
        
        # Create simple single-task plan
        task = Task(
            tool_name=selected_tool,
            inputs={"query": request.message} if selected_tool == "general_help" else {},
            estimate_tokens=150,
            name="Process user request",
            description=f"Handle user request using {selected_tool}"
        )
        
        return ExecutionPlan(
            tasks=[task],
            dependencies=[],
            cost=CostEstimate(tokens=150, usd=self.cost_model.get_tool_cost(selected_tool)),
            request_id=request.request_id,
            user_id=request.user_id,
            agent_domain="fallback",
            agent_id="fallback",
            original_message=request.message,
            planning_reasoning="Fallback plan generation without LLM"
        )

    async def _calculate_plan_cost(
        self,
        plan: ExecutionPlan,
        memory_context_size: int,
        request: ChatRequest
    ) -> CostEstimate:
        """Calculate comprehensive cost estimate for plan execution"""
        
        # Sum up token estimates from tasks
        total_tokens = sum(task.estimate_tokens for task in plan.tasks)
        
        # Add planning overhead tokens
        planning_tokens = getattr(plan, '_planning_tokens', 200)  # Default estimate
        total_tokens += planning_tokens
        
        # Calculate LLM costs (assume input:output ratio of 3:1)
        input_tokens = int(total_tokens * 0.75)
        output_tokens = int(total_tokens * 0.25)
        llm_cost = self.cost_model.calculate_llm_cost(input_tokens, output_tokens)
        
        # Calculate memory system costs
        graph_hops = min(memory_context_size * 2, self.config.memory_graph_hop_limit)  # Estimate hops
        memory_cost = self.cost_model.calculate_memory_cost(
            retrievals=len(plan.tasks) + 1,  # One per task + initial retrieval
            hops=graph_hops,
            stores=2  # Store initial request + final result
        )
        
        # Calculate tool execution costs
        tool_cost = sum(
            self.cost_model.get_tool_cost(task.tool_name) for task in plan.tasks
        )
        
        # Total cost
        total_cost = llm_cost + memory_cost + tool_cost
        
        # Cost breakdown
        breakdown = {
            "llm_cost": llm_cost,
            "memory_cost": memory_cost,  
            "tool_execution": tool_cost,
            "total": total_cost
        }
        
        return CostEstimate(
            tokens=total_tokens,
            graph_hops=graph_hops,
            usd=total_cost,
            confidence=self.config.cost_estimation_confidence,
            breakdown=breakdown
        )

    def _validate_request(self, request: ChatRequest) -> ChatRequest:
        """Validate request conforms to schema"""
        try:
            # Pydantic will validate automatically, but we can add custom checks
            if len(request.message.strip()) == 0:
                raise InvalidRequestError("Message cannot be empty")
            
            if request.budget_usd <= 0:
                raise InvalidRequestError("Budget must be greater than 0")
                
            if request.budget_usd > self.config.max_budget_usd:
                raise InvalidRequestError(f"Budget exceeds maximum allowed: ${self.config.max_budget_usd}")
            
            return request
            
        except Exception as e:
            raise InvalidRequestError(f"Request validation failed: {str(e)}")

    def _validate_plan_tools(self, plan: ExecutionPlan, available_tools: Dict[str, Any]) -> None:
        """Validate all plan tools are available in agent"""
        available_tool_names = set(available_tools.keys())
        
        for task in plan.tasks:
            if task.tool_name not in available_tool_names:
                raise ToolNotFoundError(
                    tool_name=task.tool_name,
                    available_tools=list(available_tool_names)
                )

    def _check_budget_limits(self, estimated_cost: float, budget: float) -> None:
        """Check if estimated cost exceeds budget with safety margin"""
        safety_budget = budget * (1 - self.config.budget_safety_margin)
        
        if estimated_cost > safety_budget:
            raise BudgetExceededError(
                estimated_cost=estimated_cost,
                budget=safety_budget
            )

    def _check_plan_limits(self, plan: ExecutionPlan) -> None:
        """Check plan doesn't exceed safety limits"""
        if len(plan.tasks) > self.config.max_tasks_per_plan:
            raise PlanningError(f"Plan exceeds maximum tasks limit: {self.config.max_tasks_per_plan}")
        
        if len(plan.tasks) == 0:
            raise PlanningError("Plan must contain at least one task")

    def _optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Optimize plan for cost and performance"""
        if not self.config.enable_task_optimization:
            return plan
        
        # Sort tasks by cost (lowest first) to optimize for budget usage
        if self.config.cost_optimization_weight > 0.5:
            plan.tasks.sort(key=lambda task: self.cost_model.get_tool_cost(task.tool_name))
        
        # Group independent tasks for parallel execution
        if self.config.enable_parallel_grouping:
            plan = self._create_parallel_groups(plan)
        
        return plan

    def _create_parallel_groups(self, plan: ExecutionPlan) -> ExecutionPlan:
        """Create parallel execution groups for independent tasks"""
        if not plan.dependencies:
            # All tasks are independent - can be parallelized
            group_id = f"parallel_{uuid4().hex[:8]}"
            for task in plan.tasks:
                task.parallel_group_id = group_id
        
        # For now, keep it simple. Complex dependency analysis would go here.
        return plan