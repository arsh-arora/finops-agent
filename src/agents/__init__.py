"""
Advanced Multi-Agent System - Phase 4
Production-grade agents with intelligent routing and orchestration
"""

import structlog
from typing import Dict, Any, Optional, Type
from .routing.selector import AgentRouter
from .base.agent import HardenedAgent

# Import Phase 3 core components
from .base.registry import tool, ToolRegistry
from .base.exceptions import (
    AgentError,
    ToolError,
    PlanningError,
    CompilationError,
    ExecutionError
)
from .specialized.default import DefaultAgent
from .models import ChatRequest, ExecutionPlan, Task, CostEstimate

# Import all Phase 4 agents
from .finops import AdvancedFinOpsAgent
from .github import AdvancedGitHubAgent
from .document import AdvancedDocumentAgent
from .research import AdvancedResearchAgent
from .deep_research import AdvancedDeepResearchAgent

logger = structlog.get_logger(__name__)


class Phase4AgentRegistry:
    """
    Phase 4 Agent Registry with intelligent routing and orchestration capabilities
    """
    
    def __init__(self, memory_service, llm_client=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Phase 4 agent system
        
        Args:
            memory_service: Memory service instance for agents
            llm_client: LLM client for intelligent routing
            config: Configuration options
        """
        self.memory_service = memory_service
        self.config = config or {}
        
        # Initialize intelligent router
        self.router = AgentRouter(llm_client=llm_client)
        
        # Agent registry for instantiated agents
        self._agent_instances: Dict[str, HardenedAgent] = {}
        
        # Register all Phase 4 agents
        self._register_phase4_agents()
        
        logger.info(
            "phase4_agent_registry_initialized",
            registered_agents=list(self._agent_instances.keys()),
            has_llm_client=llm_client is not None
        )
    
    def _register_phase4_agents(self):
        """Register all Phase 4 specialized agents with enhanced capabilities"""
        
        # FinOps Agent - Advanced financial modeling and analysis
        self.router.register_agent(
            domain="finops",
            agent_class=AdvancedFinOpsAgent,
            capabilities={
                "advanced_financial_modeling": "NPV, IRR, and complex financial calculations",
                "intelligent_anomaly_detection": "ML-driven cost anomaly detection with ensemble methods",
                "multi_objective_optimization": "Budget optimization with mathematical programming",
                "risk_assessment_analysis": "Risk-adjusted financial modeling and analysis", 
                "market_aware_calculations": "Market context integration for discount rates",
                "adaptive_algorithm_selection": "Data-driven algorithm selection and tuning",
                "memory_driven_learning": "Historical pattern-based optimization"
            }
        )
        
        # GitHub Agent - Intelligent security and code analysis  
        self.router.register_agent(
            domain="github",
            agent_class=AdvancedGitHubAgent,
            capabilities={
                "repository_security_analysis": "Multi-tool security vulnerability scanning",
                "vulnerability_assessment": "EPSS-enhanced vulnerability scoring and prioritization",
                "contributor_behavior_analysis": "ML-driven contributor pattern analysis and anomaly detection",
                "dependency_security_scanning": "Comprehensive dependency vulnerability assessment",
                "code_quality_evaluation": "Advanced code metrics and quality assessment",
                "repository_comparison": "Intelligent multi-repository comparative analysis",
                "ml_driven_risk_assessment": "Machine learning enhanced security risk scoring",
                "epss_vulnerability_scoring": "Exploit Prediction Scoring System integration"
            }
        )
        
        # Document Agent - Advanced document processing with Docling
        self.router.register_agent(
            domain="document", 
            agent_class=AdvancedDocumentAgent,
            capabilities={
                "document_processing": "Multi-format document analysis and conversion",
                "bounding_box_extraction": "Docling-powered precise element location and extraction",
                "content_analysis": "ML-driven content understanding and insight generation",
                "entity_extraction": "Named entity recognition and relationship mapping",
                "topic_modeling": "Advanced topic discovery and clustering",
                "sentiment_analysis": "Multi-dimensional sentiment and bias analysis",
                "document_comparison": "Intelligent document similarity and difference analysis",
                "quality_assessment": "Comprehensive document quality and confidence scoring",
                "multi_format_support": "PDF, Word, Excel, PowerPoint, and image processing",
                "ml_driven_insights": "AI-powered content insights and recommendations"
            }
        )
        
        # Research Agent - Intelligent web research with Tavily
        self.router.register_agent(
            domain="research",
            agent_class=AdvancedResearchAgent, 
            capabilities={
                "web_research": "Comprehensive web search and analysis",
                "tavily_integration": "Advanced search with Tavily API for enhanced results",
                "quality_assessment": "Multi-dimensional source quality and credibility scoring",
                "bias_detection": "Intelligent bias pattern detection and analysis",
                "credibility_scoring": "Authority-based source credibility assessment",
                "consensus_analysis": "Cross-source consensus and contradiction detection",
                "source_verification": "Multi-source fact verification and validation",
                "temporal_analysis": "Information recency and temporal pattern analysis",
                "multi_query_research": "Comparative research across multiple queries",
                "comparative_analysis": "Cross-query pattern analysis and synthesis",
                "ml_driven_insights": "AI-enhanced research insights and recommendations",
                "memory_optimization": "Historical pattern-driven query optimization"
            }
        )
        
        # Deep Research Agent - Multi-hop orchestration
        self.router.register_agent(
            domain="deep_research",
            agent_class=AdvancedDeepResearchAgent,
            capabilities={
                "multi_hop_orchestration": "Intelligent multi-step research coordination",
                "cross_domain_synthesis": "Integration of insights across different domains",
                "agent_coordination": "Multi-agent workflow orchestration and management",
                "adaptive_planning": "Dynamic research strategy adjustment based on results",
                "quality_assessment": "Comprehensive research quality and confidence scoring",
                "cross_validation": "Multi-agent finding validation and consistency checking",
                "insight_generation": "Advanced cross-domain insight discovery and analysis",
                "research_optimization": "Memory-driven research strategy optimization",
                "graph_execution": "Graph-based execution tracking and dependency management",
                "memory_integration": "Advanced memory-driven learning and pattern recognition"
            }
        )
        
        logger.info(
            "phase4_agents_registered",
            total_agents=5,
            domains=["finops", "github", "document", "research", "deep_research"]
        )
    
    async def get_agent(self, domain: Optional[str] = None, user_context: Optional[Dict[str, Any]] = None) -> HardenedAgent:
        """
        Get appropriate agent instance using intelligent routing
        
        Args:
            domain: Optional explicit domain specification
            user_context: User context for intelligent routing decisions
            
        Returns:
            Appropriate agent instance for the request
        """
        user_ctx = user_context or {}
        
        # Use router to select appropriate agent class
        agent_class = await self.router.select(domain, user_ctx)
        
        # Get agent domain for caching
        if hasattr(agent_class, '_domain'):
            agent_domain = agent_class._domain
        else:
            agent_domain = domain or "unknown"
        
        # Return cached instance or create new one
        if agent_domain not in self._agent_instances:
            # Create agent instance with memory service
            if agent_class == AdvancedDeepResearchAgent:
                # Deep research agent gets special initialization
                agent_instance = agent_class(self.memory_service)
            else:
                # Standard agent initialization
                agent_instance = agent_class(self.memory_service)
            
            self._agent_instances[agent_domain] = agent_instance
            
            logger.info(
                "agent_instance_created",
                domain=agent_domain,
                agent_class=agent_class.__name__
            )
        
        return self._agent_instances[agent_domain]
    
    def get_available_domains(self) -> list:
        """Get list of available agent domains"""
        return list(self.router._registry.keys())
    
    def get_agent_capabilities(self, domain: str) -> Dict[str, Any]:
        """Get capabilities for specific agent domain"""
        return self.router._agent_capabilities.get(domain, {})
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        return self.router.get_routing_stats()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all registered agents"""
        health_status = {
            "registry_status": "healthy",
            "total_agents": len(self._agent_instances),
            "agent_health": {},
            "routing_stats": self.get_routing_stats(),
            "memory_service_status": "connected" if self.memory_service else "unavailable"
        }
        
        # Check each instantiated agent
        for domain, agent in self._agent_instances.items():
            try:
                # Basic health check - verify agent has required methods
                agent_health = {
                    "status": "healthy",
                    "capabilities_count": len(agent.get_capabilities()) if hasattr(agent, 'get_capabilities') else 0,
                    "domain": agent.get_domain() if hasattr(agent, 'get_domain') else domain,
                    "memory_connected": hasattr(agent, 'memory_service') and agent.memory_service is not None
                }
                
                health_status["agent_health"][domain] = agent_health
                
            except Exception as e:
                health_status["agent_health"][domain] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["registry_status"] = "degraded"
        
        return health_status


# Global registry instance (will be initialized by main application)
_registry_instance: Optional[Phase4AgentRegistry] = None


def initialize_agent_registry(memory_service, llm_client=None, config: Optional[Dict[str, Any]] = None) -> Phase4AgentRegistry:
    """
    Initialize the global Phase 4 agent registry
    
    Args:
        memory_service: Memory service instance
        llm_client: LLM client for intelligent routing
        config: Configuration options
        
    Returns:
        Initialized agent registry
    """
    global _registry_instance
    
    _registry_instance = Phase4AgentRegistry(
        memory_service=memory_service,
        llm_client=llm_client, 
        config=config
    )
    
    logger.info(
        "global_agent_registry_initialized",
        available_domains=_registry_instance.get_available_domains()
    )
    
    return _registry_instance


def get_agent_registry() -> Phase4AgentRegistry:
    """
    Get the global agent registry instance
    
    Returns:
        Global agent registry
        
    Raises:
        RuntimeError: If registry not initialized
    """
    if _registry_instance is None:
        raise RuntimeError("Agent registry not initialized. Call initialize_agent_registry() first.")
    
    return _registry_instance


# Export all components for backwards compatibility and new Phase 4 features
__all__ = [
    # Phase 3 core components (backwards compatibility)
    "HardenedAgent",
    "tool",
    "ToolRegistry", 
    "AgentRouter",
    "DefaultAgent",
    "ChatRequest",
    "ExecutionPlan", 
    "Task",
    "CostEstimate",
    "AgentError",
    "ToolError",
    "PlanningError", 
    "CompilationError",
    "ExecutionError",
    
    # Phase 4 new components
    "Phase4AgentRegistry",
    "initialize_agent_registry", 
    "get_agent_registry",
    "AdvancedFinOpsAgent",
    "AdvancedGitHubAgent",
    "AdvancedDocumentAgent", 
    "AdvancedResearchAgent",
    "AdvancedDeepResearchAgent"
]