"""
LLM-based Agent Routing & Selection
Routes requests to appropriate agent classes using intelligent analysis
"""

import json
import structlog
from typing import Dict, Type, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime

logger = structlog.get_logger(__name__)


@dataclass
class RoutingDecision:
    """Result of LLM-based routing analysis"""
    selected_domain: str
    confidence_score: float
    reasoning: str
    fallback_used: bool
    analysis_tokens: int


class AgentRouter:
    """
    Intelligent agent selection using LLM analysis instead of rule-based routing
    Routes every request to exactly one concrete agent class or DefaultAgent fallback
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize router with LLM client for intelligent routing
        
        Args:
            llm_client: LLM client for routing decisions (will use OpenRouter/OpenAI)
        """
        self._registry: Dict[str, Type] = {}
        self._agent_capabilities: Dict[str, Dict[str, Any]] = {}
        self._llm_client = llm_client
        self._routing_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "fallback_used": 0,
            "routing_misses": 0
        }
        
        logger.info(
            "agent_router_initialized",
            has_llm_client=llm_client is not None
        )

    def register_agent(
        self, 
        domain: str, 
        agent_class: Type,
        capabilities: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register an agent class with its domain and capabilities
        
        Args:
            domain: Domain identifier (e.g., 'finops', 'github', 'research')
            agent_class: Agent class to handle requests for this domain
            capabilities: Agent capabilities and metadata for LLM routing
        """
        if not hasattr(agent_class, 'get_domain') or not hasattr(agent_class, 'get_capabilities'):
            raise ValueError(f"Agent class {agent_class.__name__} must implement get_domain() and get_capabilities()")
        
        self._registry[domain] = agent_class
        
        # Store enhanced capabilities for LLM routing
        self._agent_capabilities[domain] = {
            "class_name": agent_class.__name__,
            "capabilities": capabilities or self._extract_agent_capabilities(agent_class),
            "description": getattr(agent_class, '__doc__', f"{agent_class.__name__} agent"),
            "tools": self._extract_agent_tools(agent_class) if hasattr(agent_class, '_tools') else [],
            "examples": self._get_domain_examples(domain)
        }
        
        logger.info(
            "agent_registered",
            domain=domain,
            agent_class=agent_class.__name__,
            capabilities_count=len(self._agent_capabilities[domain]["capabilities"])
        )

    async def select(self, domain: Optional[str], user_ctx: Dict[str, Any]) -> Type:
        """
        Select appropriate agent class using LLM-based intelligent routing
        
        Args:
            domain: Optional explicit domain hint
            user_ctx: User context including message, history, preferences
            
        Returns:
            Selected HardenedAgent class
            
        Raises:
            ValueError: If routing fails completely
        """
        request_id = user_ctx.get('request_id', 'unknown')
        message = user_ctx.get('message', '')
        
        self._routing_stats["total_requests"] += 1
        
        logger.info(
            "agent_selection_started",
            request_id=request_id,
            explicit_domain=domain,
            message_length=len(message),
            available_domains=list(self._registry.keys())
        )
        
        try:
            # If explicit domain provided and valid, use it
            if domain and domain in self._registry:
                self._routing_stats["successful_routes"] += 1
                selected_class = self._registry[domain]
                
                logger.info(
                    "agent_selected_explicit",
                    request_id=request_id,
                    domain=domain,
                    agent_class=selected_class.__name__
                )
                
                return selected_class
            
            # Use LLM-based intelligent routing
            routing_decision = await self._llm_route_request(user_ctx)
            
            if routing_decision.selected_domain in self._registry:
                self._routing_stats["successful_routes"] += 1
                selected_class = self._registry[routing_decision.selected_domain]
                
                logger.info(
                    "agent_selected_llm",
                    request_id=request_id,
                    domain=routing_decision.selected_domain,
                    agent_class=selected_class.__name__,
                    confidence=routing_decision.confidence_score,
                    reasoning=routing_decision.reasoning[:100] + "..." if len(routing_decision.reasoning) > 100 else routing_decision.reasoning
                )
                
                return selected_class
            
            # Fallback to default agent
            return self._fallback_to_default(request_id, routing_decision)
            
        except Exception as e:
            logger.error(
                "agent_selection_error",
                request_id=request_id,
                error=str(e)
            )
            
            # Emergency fallback
            return self._emergency_fallback(request_id)

    async def _llm_route_request(self, user_ctx: Dict[str, Any]) -> RoutingDecision:
        """
        Use LLM to intelligently analyze request and select appropriate agent
        
        Args:
            user_ctx: Full user context for analysis
            
        Returns:
            RoutingDecision with selected domain and reasoning
        """
        if not self._llm_client:
            # Fallback to basic heuristic routing if no LLM available
            return self._heuristic_fallback_routing(user_ctx)
        
        # Construct intelligent routing prompt
        routing_prompt = self._build_routing_prompt(user_ctx)
        
        try:
            # Call LLM for routing decision
            response = await self._llm_client.complete(
                messages=[{"role": "user", "content": routing_prompt}],
                model="gpt-4o-mini",  # Fast, cost-effective for routing decisions
                max_tokens=200,
                temperature=0.1  # Low temperature for consistent routing
            )
            
            # Parse LLM response
            return self._parse_routing_response(response, user_ctx)
            
        except Exception as e:
            logger.warning(
                "llm_routing_failed",
                error=str(e),
                fallback_to_heuristic=True
            )
            
            # Fallback to heuristic routing
            return self._heuristic_fallback_routing(user_ctx)

    def _build_routing_prompt(self, user_ctx: Dict[str, Any]) -> str:
        """Build intelligent routing prompt for LLM"""
        message = user_ctx.get('message', '')
        conversation_history = user_ctx.get('conversation_history', [])
        user_preferences = user_ctx.get('preferences', {})
        
        # Build agent capability descriptions
        agent_descriptions = []
        for domain, info in self._agent_capabilities.items():
            if domain == 'default':
                continue
                
            capabilities_str = ", ".join(info["capabilities"])
            examples_str = "; ".join(info["examples"])
            
            agent_descriptions.append(
                f"- **{domain}**: {info['description']}\n"
                f"  Capabilities: {capabilities_str}\n"
                f"  Examples: {examples_str}"
            )
        
        recent_context = ""
        if conversation_history:
            recent_messages = conversation_history[-3:]  # Last 3 messages for context
            recent_context = f"\nRecent conversation context:\n" + "\n".join([
                f"- {msg.get('role', 'user')}: {msg.get('content', '')[:100]}..."
                for msg in recent_messages
            ])
        
        return f"""You are an intelligent agent router. Analyze the user's request and select the most appropriate specialized agent.

Available Agents:
{chr(10).join(agent_descriptions)}

User Request: "{message}"
{recent_context}

Instructions:
1. Analyze the request content, intent, and context
2. Consider any domain-specific keywords, technical terms, or use cases
3. Select the most appropriate agent domain
4. If the request spans multiple domains, choose the PRIMARY domain
5. If unsure or the request is too general, select "default"

Respond in this exact JSON format:
{{
  "selected_domain": "domain_name",
  "confidence_score": 0.85,
  "reasoning": "Brief explanation of why this agent was selected"
}}

Domain must be one of: {', '.join(list(self._agent_capabilities.keys()))}"""

    def _parse_routing_response(self, llm_response: str, user_ctx: Dict[str, Any]) -> RoutingDecision:
        """Parse LLM routing response into RoutingDecision"""
        try:
            # Extract JSON from response
            response_text = llm_response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(response_text)
            
            return RoutingDecision(
                selected_domain=parsed.get("selected_domain", "default"),
                confidence_score=float(parsed.get("confidence_score", 0.5)),
                reasoning=parsed.get("reasoning", "LLM routing decision"),
                fallback_used=False,
                analysis_tokens=len(response_text.split())  # Rough token estimate
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(
                "llm_response_parse_failed",
                response=llm_response[:200],
                error=str(e)
            )
            
            # Fallback to heuristic analysis
            return self._heuristic_fallback_routing(user_ctx)

    def _heuristic_fallback_routing(self, user_ctx: Dict[str, Any]) -> RoutingDecision:
        """
        Fallback heuristic routing when LLM is unavailable
        Uses keyword analysis and context clues
        """
        message = user_ctx.get('message', '').lower()
        
        # Domain-specific keyword mapping
        domain_keywords = {
            'finops': [
                'cost', 'budget', 'billing', 'spend', 'expense', 'financial', 'finops',
                'aws', 'azure', 'gcp', 'cloud', 'ec2', 'lambda', 'optimization',
                'savings', 'waste', 'utilization', 'forecast', 'charge', 'npv', 'irr',
                'investment', 'anomaly', 'anomalies', 'roi', 'profit', 'revenue'
            ],
            'github': [
                'github', 'git', 'repository', 'repo', 'commit', 'pull request', 'pr',
                'issue', 'branch', 'merge', 'clone', 'push', 'code', 'development',
                'security', 'vulnerability', 'vuln', 'cve', 'epss', 'contributor',
                'analysis', 'scan', 'audit'
            ],
            'document': [
                'document', 'pdf', 'word', 'excel', 'powerpoint', 'file', 'parse',
                'extract', 'bounding box', 'bbox', 'content', 'text', 'docling',
                'analyze document', 'process file', 'entity', 'topic', 'sentiment'
            ],
            'research': [
                'research', 'search', 'web', 'internet', 'find', 'investigate', 'study',
                'tavily', 'fact check', 'verify', 'credibility', 'bias', 'consensus',
                'sources', 'information', 'data', 'insights', 'analysis'
            ],
            'deep_research': [
                'comprehensive', 'multi-hop', 'orchestrate', 'coordinate', 'synthesize',
                'cross-domain', 'multi-agent', 'complex analysis', 'thorough investigation',
                'validate across', 'deep dive', 'holistic', 'integrate findings'
            ]
        }
        
        # Score each domain based on keyword matches
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            if domain in self._registry:
                score = sum(1 for keyword in keywords if keyword in message)
                if score > 0:
                    domain_scores[domain] = score / len(keywords)  # Normalize
        
        # Select highest scoring domain
        if domain_scores:
            selected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
            confidence = min(domain_scores[selected_domain] * 2, 1.0)  # Scale confidence
            
            return RoutingDecision(
                selected_domain=selected_domain,
                confidence_score=confidence,
                reasoning=f"Heuristic analysis: matched {selected_domain} keywords",
                fallback_used=True,
                analysis_tokens=0
            )
        
        # No clear match - use default
        return RoutingDecision(
            selected_domain="default",
            confidence_score=0.3,
            reasoning="No clear domain match found, using default agent",
            fallback_used=True,
            analysis_tokens=0
        )

    def _fallback_to_default(self, request_id: str, routing_decision: RoutingDecision) -> Type:
        """Handle fallback to default agent"""
        self._routing_stats["fallback_used"] += 1
        
        if 'default' in self._registry:
            selected_class = self._registry['default']
            
            logger.info(
                "agent_selected_fallback",
                request_id=request_id,
                agent_class=selected_class.__name__,
                original_decision=routing_decision.selected_domain
            )
            
            return selected_class
        
        # No default agent registered
        self._routing_stats["routing_misses"] += 1
        
        logger.error(
            "agent_routing_failed",
            request_id=request_id,
            reason="no_default_agent",
            available_domains=list(self._registry.keys())
        )
        
        raise ValueError(f"No suitable agent found and no default agent registered. Available domains: {list(self._registry.keys())}")

    def _emergency_fallback(self, request_id: str) -> Type:
        """Emergency fallback when everything fails"""
        self._routing_stats["routing_misses"] += 1
        
        # Try to return any available agent as last resort
        if self._registry:
            fallback_class = next(iter(self._registry.values()))
            
            logger.warning(
                "emergency_fallback_used",
                request_id=request_id,
                agent_class=fallback_class.__name__
            )
            
            return fallback_class
        
        raise ValueError("No agents registered in router")

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        total = self._routing_stats["total_requests"]
        if total == 0:
            miss_rate = 0.0
        else:
            miss_rate = self._routing_stats["routing_misses"] / total
        
        return {
            **self._routing_stats,
            "miss_rate_percent": round(miss_rate * 100, 2),
            "success_rate_percent": round((self._routing_stats["successful_routes"] / total * 100) if total > 0 else 0, 2)
        }

    def clear_registry(self) -> None:
        """Clear all registered agents (for testing)"""
        self._registry.clear()
        self._agent_capabilities.clear()
        
        logger.info("agent_registry_cleared")

    def _extract_agent_capabilities(self, agent_class: Type) -> List[str]:
        """Extract capabilities from agent class"""
        if hasattr(agent_class, '_capabilities'):
            return list(agent_class._capabilities)
        elif hasattr(agent_class, 'get_capabilities'):
            # Try to call static method if available
            try:
                return agent_class.get_capabilities() if callable(agent_class.get_capabilities) else []
            except TypeError:
                return []
        return []

    def _extract_agent_tools(self, agent_class: Type) -> List[str]:
        """Extract available tools from agent class"""
        tools = []
        for attr_name in dir(agent_class):
            if not attr_name.startswith('_'):
                attr = getattr(agent_class, attr_name)
                if (hasattr(attr, '_is_tool') and 
                    attr._is_tool and 
                    hasattr(attr, '_tool_name')):
                    tools.append(attr._tool_name)
        return tools

    def _get_domain_examples(self, domain: str) -> List[str]:
        """Get example requests for each domain"""
        examples = {
            'finops': [
                "Calculate NPV for this investment with sensitivity analysis",
                "Detect cost anomalies in our AWS spending using ML",
                "Optimize our budget allocation across multiple projects", 
                "Perform IRR analysis with market-aware discount rates"
            ],
            'github': [
                "Perform comprehensive security analysis of this repository",
                "Analyze contributor behavior patterns for anomalies",
                "Assess vulnerability risk with EPSS scoring",
                "Compare security posture across multiple repositories"
            ],
            'document': [
                "Extract bounding boxes from this PDF with high precision",
                "Analyze document content with ML-driven insights",
                "Compare multiple documents for similarities and differences",
                "Process multi-format documents with entity extraction"
            ],
            'research': [
                "Conduct comprehensive web research with credibility analysis",
                "Perform fact verification across multiple sources",
                "Research with bias detection and consensus analysis",
                "Cross-validate findings using multiple search strategies"
            ],
            'deep_research': [
                "Orchestrate multi-hop research across financial and security domains",
                "Coordinate multiple agents for comprehensive investigation",
                "Synthesize insights from document, web, and financial analysis",
                "Perform adaptive research with cross-domain validation"
            ],
            'default': [
                "Help me with general questions",
                "I need assistance",
                "Can you help me understand this?"
            ]
        }
        return examples.get(domain, [f"Handle {domain} related requests"])