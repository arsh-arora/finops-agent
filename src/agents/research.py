"""
Advanced Research Agent - Production Grade
Intelligent web research with Tavily integration and ML-driven insights
"""

import asyncio
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4
from pydantic import BaseModel, Field, validator
from enum import Enum

from .base.agent import HardenedAgent
from .base.registry import tool
from .base.exceptions import ToolError
from src.adapters.research.intelligent_searcher import (
    IntelligentResearchEngine,
    ResearchProfile,
    SearchScope,
    SearchComplexity,
    SearchResult,
    SearchInsight,
    SourceType,
    CredibilityLevel
)

import numpy as np

logger = structlog.get_logger(__name__)


# Pydantic Models for Input/Output Validation
class ResearchQuery(BaseModel):
    """Research query with configuration"""
    query: str = Field(..., min_length=3, max_length=500, description="Research query")
    search_scope: str = Field("general", description="Search scope")
    max_results: int = Field(20, ge=5, le=100, description="Maximum number of results")
    quality_threshold: float = Field(0.6, ge=0.1, le=1.0, description="Quality threshold for filtering")
    include_recent_only: bool = Field(False, description="Include only recent sources")
    
    @validator('search_scope')
    def validate_search_scope(cls, v):
        allowed = ['general', 'academic', 'news', 'technical', 'financial', 'legal', 'medical']
        if v not in allowed:
            raise ValueError(f'search_scope must be one of {allowed}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "impact of artificial intelligence on job market",
                "search_scope": "academic",
                "max_results": 20,
                "quality_threshold": 0.7,
                "include_recent_only": False
            }
        }


class ResearchOptions(BaseModel):
    """Advanced research configuration options"""
    search_depth: str = Field("advanced", description="Search depth level")
    fact_checking: bool = Field(True, description="Enable fact checking analysis")
    bias_detection: bool = Field(True, description="Enable bias detection")
    sentiment_analysis: bool = Field(True, description="Enable sentiment analysis")
    entity_extraction: bool = Field(True, description="Enable entity extraction")
    consensus_analysis: bool = Field(True, description="Enable consensus analysis")
    source_verification: bool = Field(True, description="Enable source credibility verification")
    
    @validator('search_depth')
    def validate_search_depth(cls, v):
        allowed = ['basic', 'standard', 'advanced', 'comprehensive']
        if v not in allowed:
            raise ValueError(f'search_depth must be one of {allowed}')
        return v


class SearchResultSummary(BaseModel):
    """Summary of individual search result"""
    result_id: str
    title: str
    url: str
    snippet: str
    source_type: str
    credibility_score: float = Field(..., ge=0, le=1)
    relevance_score: float = Field(..., ge=0, le=1)
    recency_score: float = Field(..., ge=0, le=1)
    bias_score: float = Field(..., ge=0, le=1)
    key_entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    sentiment: Dict[str, float] = Field(default_factory=dict)


class ResearchInsightSummary(BaseModel):
    """Summary of research insight"""
    insight_type: str
    confidence: float = Field(..., ge=0, le=1)
    description: str
    supporting_sources: List[str] = Field(default_factory=list)
    key_evidence: List[str] = Field(default_factory=list)
    consensus_level: float = Field(..., ge=0, le=1)


class ResearchAnalysisResult(BaseModel):
    """Comprehensive research analysis results"""
    search_id: str
    query: str
    search_scope: str
    complexity_level: str
    analysis_timestamp: datetime
    
    total_results_found: int = Field(..., ge=0)
    high_quality_results: int = Field(..., ge=0)
    
    search_results: List[SearchResultSummary]
    research_insights: List[ResearchInsightSummary]
    
    consensus_analysis: Dict[str, Any] = Field(default_factory=dict)
    credibility_assessment: Dict[str, Any] = Field(default_factory=dict)
    bias_analysis: Dict[str, Any] = Field(default_factory=dict)
    temporal_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    source_diversity: Dict[str, Any] = Field(default_factory=dict)
    research_gaps: List[str] = Field(default_factory=list)
    confidence_metrics: Dict[str, float] = Field(default_factory=dict)
    
    executive_summary: str
    recommendations: List[str] = Field(default_factory=list)
    schema_version: str = Field(default="1.0")


class MultiQueryResearchRequest(BaseModel):
    """Request for multi-query research analysis"""
    queries: List[ResearchQuery] = Field(..., min_items=2, max_items=5)
    comparative_analysis: bool = Field(True, description="Perform comparative analysis")
    synthesis_required: bool = Field(True, description="Synthesize findings across queries")
    research_options: Optional[ResearchOptions] = None


class ComparativeResearchResult(BaseModel):
    """Results from comparative research analysis"""
    research_timestamp: datetime
    queries_analyzed: List[str]
    
    individual_results: List[ResearchAnalysisResult]
    comparative_insights: List[ResearchInsightSummary]
    cross_query_patterns: Dict[str, Any] = Field(default_factory=dict)
    synthesis_summary: str
    
    overall_confidence: float = Field(..., ge=0, le=1)
    research_completeness: float = Field(..., ge=0, le=1)
    schema_version: str = Field(default="1.0")


class AdvancedResearchAgent(HardenedAgent):
    """
    Advanced Research Agent with intelligent web search capabilities
    
    Features:
    - Tavily integration for comprehensive web search
    - ML-driven result analysis and insights
    - Multi-dimensional quality assessment
    - Bias detection and credibility scoring
    - Consensus analysis across sources
    - Temporal pattern recognition
    - Memory-driven query optimization
    - Comparative multi-query research
    """
    
    _domain = "research"
    _capabilities = [
        "web_research",
        "tavily_integration",
        "quality_assessment",
        "bias_detection",
        "credibility_scoring",
        "consensus_analysis",
        "source_verification",
        "temporal_analysis",
        "multi_query_research",
        "comparative_analysis",
        "ml_driven_insights",
        "memory_optimization"
    ]
    
    def __init__(self, memory_service, agent_id: Optional[str] = None, tavily_api_key: Optional[str] = None):
        super().__init__(memory_service, agent_id)
        self.research_engine = IntelligentResearchEngine(memory_service, tavily_api_key)
        self._research_cache = {}
        
        logger.info(
            "advanced_research_agent_initialized",
            agent_id=self.agent_id,
            capabilities=self._capabilities,
            tavily_available=bool(tavily_api_key)
        )
    
    def get_capabilities(self) -> List[str]:
        return self._capabilities.copy()
    
    def get_domain(self) -> str:
        return self._domain
    
    async def _process_message(
        self, 
        message: str, 
        memory_context: List[str], 
        plan: Dict[str, Any]
    ) -> str:
        """Intelligent message processing with research context awareness"""
        
        request_id = plan.get('request_id', uuid4().hex)
        
        logger.info(
            "research_message_processing",
            agent_id=self.agent_id,
            request_id=request_id,
            message_type=await self._classify_research_intent(message),
            memory_context_size=len(memory_context)
        )
        
        intent = await self._classify_research_intent(message)
        
        if intent == "web_research":
            return f"I can conduct comprehensive web research with Tavily integration, ML-driven analysis, and quality assessment. Memory context: {len(memory_context)} research patterns available."
        elif intent == "fact_checking":
            return f"I'll perform fact verification using multi-source analysis, credibility scoring, and bias detection. {len(memory_context)} fact-checking patterns in memory."
        elif intent == "comparative_research":
            return f"I can compare multiple research queries with cross-query synthesis and pattern analysis. {len(memory_context)} comparative patterns available."
        elif intent == "source_analysis":
            return f"I'll analyze source credibility, bias patterns, and consensus levels across research results. {len(memory_context)} source analysis patterns in memory."
        else:
            return f"I can assist with comprehensive web research including quality assessment, bias detection, consensus analysis, and multi-query comparative research. {len(memory_context)} relevant memories found."
    
    @tool(description="Comprehensive web research with ML-driven analysis and quality assessment")
    async def conduct_research(
        self,
        research_query: ResearchQuery,
        research_options: Optional[ResearchOptions] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ResearchAnalysisResult:
        """
        Conduct comprehensive web research with:
        - Tavily-powered intelligent search
        - ML-driven result analysis and insights
        - Multi-dimensional quality assessment
        - Bias detection and credibility scoring
        - Consensus analysis across sources
        - Temporal pattern recognition
        
        Complexity: O(n * log(n)) for n search results
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "comprehensive_research_started",
            agent_id=self.agent_id,
            request_id=request_id,
            query=research_query.query,
            search_scope=research_query.search_scope,
            max_results=research_query.max_results,
            estimated_cost=0.20
        )
        
        try:
            # Convert search scope to enum
            scope_mapping = {
                'general': SearchScope.GENERAL,
                'academic': SearchScope.ACADEMIC,
                'news': SearchScope.NEWS,
                'technical': SearchScope.TECHNICAL,
                'financial': SearchScope.FINANCIAL,
                'legal': SearchScope.LEGAL,
                'medical': SearchScope.MEDICAL
            }
            search_scope = scope_mapping.get(research_query.search_scope, SearchScope.GENERAL)
            
            # Configure research options
            research_opts = self._convert_research_options(research_options or ResearchOptions())
            
            # Execute comprehensive research
            research_profile = await self.research_engine.conduct_research_comprehensive(
                query=research_query.query,
                search_scope=search_scope,
                max_results=research_query.max_results,
                quality_threshold=research_query.quality_threshold,
                research_options=research_opts,
                context=context or {}
            )
            
            # Convert to API result format
            result = await self._convert_research_profile_to_result(
                research_profile, research_query
            )
            
            # Store research pattern for optimization
            await self._store_research_pattern("web_research", result, context)
            
            logger.info(
                "comprehensive_research_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                search_id=result.search_id,
                high_quality_results=result.high_quality_results,
                insights_generated=len(result.research_insights),
                overall_confidence=result.confidence_metrics.get("overall_confidence", 0)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "comprehensive_research_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                query=research_query.query,
                error=str(e),
                error_type=type(e).__name__
            )
            raise ToolError(
                f"Research failed: {str(e)}",
                tool_name="conduct_research",
                agent_id=self.agent_id
            )
    
    @tool(description="Multi-query comparative research with cross-query synthesis")
    async def conduct_comparative_research(
        self,
        research_request: MultiQueryResearchRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> ComparativeResearchResult:
        """
        Conduct comparative research across multiple queries:
        - Parallel execution of multiple research queries
        - Cross-query pattern analysis
        - Comparative insights generation
        - Synthesis of findings across queries
        - Overall confidence and completeness assessment
        
        Complexity: O(q * n * log(n)) for q queries with n results each
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "comparative_research_started",
            agent_id=self.agent_id,
            request_id=request_id,
            query_count=len(research_request.queries),
            comparative_analysis=research_request.comparative_analysis,
            estimated_cost=len(research_request.queries) * 0.20
        )
        
        try:
            # Execute research for each query in parallel
            research_tasks = []
            for query in research_request.queries:
                task = self.conduct_research(
                    research_query=query,
                    research_options=research_request.research_options,
                    context=context
                )
                research_tasks.append(task)
            
            individual_results = await asyncio.gather(*research_tasks)
            
            # Perform comparative analysis
            comparative_insights = []
            cross_query_patterns = {}
            
            if research_request.comparative_analysis:
                comparative_insights = await self._generate_comparative_insights(individual_results)
                cross_query_patterns = await self._analyze_cross_query_patterns(individual_results)
            
            # Generate synthesis
            synthesis_summary = ""
            if research_request.synthesis_required:
                synthesis_summary = await self._synthesize_research_findings(individual_results)
            
            # Compute overall metrics
            overall_confidence = await self._compute_overall_confidence(individual_results)
            research_completeness = await self._compute_research_completeness(individual_results)
            
            result = ComparativeResearchResult(
                research_timestamp=datetime.now(),
                queries_analyzed=[query.query for query in research_request.queries],
                individual_results=individual_results,
                comparative_insights=comparative_insights,
                cross_query_patterns=cross_query_patterns,
                synthesis_summary=synthesis_summary,
                overall_confidence=overall_confidence,
                research_completeness=research_completeness
            )
            
            # Store comparative research pattern
            await self._store_research_pattern("comparative_research", result.dict(), context)
            
            logger.info(
                "comparative_research_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                queries_analyzed=len(research_request.queries),
                comparative_insights=len(comparative_insights),
                overall_confidence=overall_confidence
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "comparative_research_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise ToolError(
                f"Comparative research failed: {str(e)}",
                tool_name="conduct_comparative_research",
                agent_id=self.agent_id
            )
    
    @tool(description="Advanced fact verification and source credibility analysis")
    async def verify_information(
        self,
        claim: str,
        verification_depth: str = "standard",
        source_requirements: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advanced fact verification with:
        - Multi-source verification
        - Credibility-weighted consensus
        - Bias detection and analysis
        - Source authority assessment
        - Confidence scoring
        
        Complexity: O(n * m) for n sources and m verification checks
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "fact_verification_started",
            agent_id=self.agent_id,
            request_id=request_id,
            claim_length=len(claim),
            verification_depth=verification_depth,
            estimated_cost=0.25
        )
        
        try:
            # Create research query for fact verification
            verification_query = ResearchQuery(
                query=f"fact check verify: {claim}",
                search_scope="academic",  # Prioritize academic sources
                max_results=15,
                quality_threshold=0.7  # Higher threshold for fact checking
            )
            
            # Configure options for fact verification
            verification_options = ResearchOptions(
                search_depth=verification_depth,
                fact_checking=True,
                bias_detection=True,
                source_verification=True,
                consensus_analysis=True
            )
            
            # Conduct research for verification
            research_result = await self.conduct_research(
                research_query=verification_query,
                research_options=verification_options,
                context=context
            )
            
            # Perform specialized fact verification analysis
            verification_result = await self._perform_fact_verification_analysis(
                research_result, claim, source_requirements
            )
            
            logger.info(
                "fact_verification_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                verification_confidence=verification_result.get("verification_confidence", 0),
                supporting_sources=len(verification_result.get("supporting_sources", []))
            )
            
            return verification_result
            
        except Exception as e:
            logger.error(
                "fact_verification_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise ToolError(
                f"Fact verification failed: {str(e)}",
                tool_name="verify_information",
                agent_id=self.agent_id
            )
    
    @tool(description="Source credibility assessment and bias analysis")
    async def analyze_sources(
        self,
        source_urls: List[str],
        analysis_depth: str = "comprehensive",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive source analysis:
        - Credibility scoring using multiple factors
        - Bias detection and classification
        - Authority assessment
        - Temporal relevance analysis
        - Cross-source validation
        
        Complexity: O(n * k) for n sources and k analysis factors
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "source_analysis_started",
            agent_id=self.agent_id,
            request_id=request_id,
            source_count=len(source_urls),
            analysis_depth=analysis_depth,
            estimated_cost=len(source_urls) * 0.08
        )
        
        try:
            # Analyze each source
            source_analyses = []
            
            for url in source_urls:
                # Create focused research query for source analysis
                source_query = ResearchQuery(
                    query=f"site:{url}",
                    search_scope="general",
                    max_results=5,
                    quality_threshold=0.5
                )
                
                # Conduct targeted research
                source_research = await self.conduct_research(
                    research_query=source_query,
                    context=context
                )
                
                # Extract source-specific analysis
                source_analysis = await self._analyze_individual_source(
                    url, source_research, analysis_depth
                )
                source_analyses.append(source_analysis)
            
            # Generate comparative source analysis
            comparative_analysis = await self._generate_source_comparison(source_analyses)
            
            result = {
                "analysis_timestamp": datetime.now(),
                "sources_analyzed": len(source_urls),
                "individual_source_analyses": source_analyses,
                "comparative_analysis": comparative_analysis,
                "overall_credibility_distribution": await self._compute_credibility_distribution(source_analyses),
                "bias_patterns": await self._identify_bias_patterns(source_analyses),
                "recommendations": await self._generate_source_recommendations(source_analyses)
            }
            
            # Store source analysis pattern
            await self._store_research_pattern("source_analysis", result, context)
            
            logger.info(
                "source_analysis_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                sources_analyzed=len(source_analyses)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "source_analysis_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise ToolError(
                f"Source analysis failed: {str(e)}",
                tool_name="analyze_sources",
                agent_id=self.agent_id
            )
    
    # Helper methods
    
    async def _classify_research_intent(self, message: str) -> str:
        """Classify user message intent for research operations"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["fact check", "verify", "true", "false", "accurate"]):
            return "fact_checking"
        elif any(word in message_lower for word in ["compare", "versus", "vs", "difference", "similar"]):
            return "comparative_research"
        elif any(word in message_lower for word in ["source", "credible", "reliable", "bias", "trust"]):
            return "source_analysis"
        elif any(word in message_lower for word in ["research", "search", "find", "information", "data"]):
            return "web_research"
        else:
            return "general_research"
    
    def _convert_research_options(self, options: ResearchOptions) -> Dict[str, Any]:
        """Convert Pydantic research options to engine format"""
        
        return {
            "search_depth": options.search_depth,
            "fact_checking": options.fact_checking,
            "bias_detection": options.bias_detection,
            "sentiment_analysis": options.sentiment_analysis,
            "entity_extraction": options.entity_extraction,
            "consensus_analysis": options.consensus_analysis,
            "source_verification": options.source_verification
        }
    
    async def _convert_research_profile_to_result(
        self,
        profile: ResearchProfile,
        query: ResearchQuery
    ) -> ResearchAnalysisResult:
        """Convert internal research profile to API result"""
        
        # Convert search results
        search_results = []
        for result in profile.filtered_results:
            result_summary = SearchResultSummary(
                result_id=result.result_id,
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                source_type=result.source_type.value,
                credibility_score=result.credibility_score,
                relevance_score=result.relevance_score,
                recency_score=result.recency_score,
                bias_score=result.bias_score,
                key_entities=result.key_entities,
                topics=result.topics,
                sentiment=result.sentiment
            )
            search_results.append(result_summary)
        
        # Convert insights
        insights = []
        for insight in profile.research_insights:
            insight_summary = ResearchInsightSummary(
                insight_type=insight.insight_type,
                confidence=insight.confidence,
                description=insight.description,
                supporting_sources=insight.supporting_sources,
                key_evidence=insight.key_evidence,
                consensus_level=insight.consensus_level
            )
            insights.append(insight_summary)
        
        # Generate executive summary
        executive_summary = await self._generate_executive_summary(profile, query)
        
        # Generate recommendations
        recommendations = await self._generate_research_recommendations(profile)
        
        return ResearchAnalysisResult(
            search_id=profile.search_id,
            query=profile.query,
            search_scope=profile.search_scope.value,
            complexity_level=profile.complexity.value,
            analysis_timestamp=profile.search_timestamp,
            total_results_found=profile.total_results,
            high_quality_results=len(profile.filtered_results),
            search_results=search_results,
            research_insights=insights,
            consensus_analysis=profile.consensus_analysis,
            credibility_assessment=profile.credibility_assessment,
            bias_analysis=profile.bias_analysis,
            temporal_analysis=profile.temporal_analysis,
            source_diversity=profile.source_diversity,
            research_gaps=profile.research_gaps,
            confidence_metrics=profile.confidence_metrics,
            executive_summary=executive_summary,
            recommendations=recommendations
        )
    
    async def _generate_executive_summary(
        self,
        profile: ResearchProfile,
        query: ResearchQuery
    ) -> str:
        """Generate intelligent executive summary"""
        
        summary_parts = []
        
        # Results overview
        quality_ratio = len(profile.filtered_results) / max(1, profile.total_results)
        summary_parts.append(
            f"Research on '{profile.query}' yielded {profile.total_results} results, "
            f"with {len(profile.filtered_results)} meeting quality standards ({quality_ratio:.1%})."
        )
        
        # Confidence and credibility
        overall_confidence = profile.confidence_metrics.get("overall_confidence", 0)
        credibility_score = profile.credibility_assessment.get("overall_credibility_score", 0)
        
        summary_parts.append(
            f"Analysis confidence: {overall_confidence:.1%}, "
            f"average source credibility: {credibility_score:.1%}."
        )
        
        # Key insights
        if profile.research_insights:
            high_conf_insights = [i for i in profile.research_insights if i.confidence > 0.7]
            if high_conf_insights:
                summary_parts.append(
                    f"Generated {len(high_conf_insights)} high-confidence insights "
                    f"including {', '.join(set(i.insight_type for i in high_conf_insights[:3]))}."
                )
        
        # Consensus and bias
        consensus_level = profile.consensus_analysis.get("overall_consensus_level", 0)
        bias_level = profile.bias_analysis.get("overall_bias_level", 0)
        
        if consensus_level > 0.6:
            summary_parts.append(f"Strong consensus ({consensus_level:.1%}) observed among sources.")
        elif bias_level > 0.4:
            summary_parts.append(f"Moderate bias detected ({bias_level:.1%}) - source diversity recommended.")
        
        return " ".join(summary_parts)
    
    async def _generate_research_recommendations(self, profile: ResearchProfile) -> List[str]:
        """Generate intelligent research recommendations"""
        
        recommendations = []
        
        # Quality-based recommendations
        quality_ratio = len(profile.filtered_results) / max(1, profile.total_results)
        if quality_ratio < 0.5:
            recommendations.append("Consider refining search terms or expanding search scope for higher quality results")
        
        # Credibility recommendations
        credibility_score = profile.credibility_assessment.get("overall_credibility_score", 0)
        if credibility_score < 0.6:
            recommendations.append("Seek additional high-credibility sources to strengthen research foundation")
        
        # Consensus recommendations
        consensus_level = profile.consensus_analysis.get("overall_consensus_level", 0)
        if consensus_level < 0.4:
            recommendations.append("Conflicting viewpoints detected - consider additional sources for clarification")
        
        # Temporal recommendations
        freshness = profile.temporal_analysis.get("information_freshness", 0)
        if freshness < 0.5:
            recommendations.append("Consider seeking more recent sources for up-to-date information")
        
        # Gap-based recommendations
        if profile.research_gaps:
            recommendations.extend([f"Address research gap: {gap}" for gap in profile.research_gaps[:2]])
        
        # Bias recommendations
        bias_level = profile.bias_analysis.get("overall_bias_level", 0)
        if bias_level > 0.5:
            recommendations.append("High bias detected - cross-reference with neutral sources")
        
        return recommendations
    
    async def _generate_comparative_insights(
        self,
        results: List[ResearchAnalysisResult]
    ) -> List[ResearchInsightSummary]:
        """Generate comparative insights across multiple research results"""
        
        insights = []
        
        if len(results) < 2:
            return insights
        
        # Confidence comparison
        confidences = [r.confidence_metrics.get("overall_confidence", 0) for r in results]
        if max(confidences) - min(confidences) > 0.3:
            best_query = results[np.argmax(confidences)].query
            insights.append(ResearchInsightSummary(
                insight_type="confidence_comparison",
                confidence=0.8,
                description=f"Research quality varies significantly; '{best_query}' yielded highest confidence results",
                supporting_sources=[results[np.argmax(confidences)].search_id],
                key_evidence=[f"Confidence range: {min(confidences):.2f} - {max(confidences):.2f}"],
                consensus_level=0.7
            ))
        
        # Source diversity comparison
        diversity_scores = [r.source_diversity.get("diversity_score", 0) for r in results]
        if any(score > 0.7 for score in diversity_scores):
            insights.append(ResearchInsightSummary(
                insight_type="source_diversity_comparison",
                confidence=0.7,
                description="Good source diversity achieved across research queries",
                supporting_sources=[r.search_id for r, score in zip(results, diversity_scores) if score > 0.7],
                key_evidence=[f"Diversity scores: {', '.join(f'{s:.2f}' for s in diversity_scores)}"],
                consensus_level=0.8
            ))
        
        return insights
    
    async def _analyze_cross_query_patterns(
        self,
        results: List[ResearchAnalysisResult]
    ) -> Dict[str, Any]:
        """Analyze patterns across multiple research queries"""
        
        patterns = {
            "common_topics": [],
            "shared_sources": [],
            "consensus_alignment": 0.0,
            "credibility_consistency": 0.0
        }
        
        if len(results) < 2:
            return patterns
        
        # Find common topics
        all_topics = []
        for result in results:
            for search_result in result.search_results:
                all_topics.extend(search_result.topics)
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Topics that appear in multiple queries
        threshold = len(results) * 0.5
        patterns["common_topics"] = [topic for topic, count in topic_counts.items() if count >= threshold]
        
        # Find shared sources (URLs that appear across queries)
        all_urls = []
        for result in results:
            result_urls = [sr.url for sr in result.search_results]
            all_urls.append(set(result_urls))
        
        if len(all_urls) > 1:
            shared_urls = all_urls[0]
            for url_set in all_urls[1:]:
                shared_urls = shared_urls.intersection(url_set)
            patterns["shared_sources"] = list(shared_urls)
        
        # Consensus alignment
        consensus_levels = [r.consensus_analysis.get("overall_consensus_level", 0) for r in results]
        patterns["consensus_alignment"] = float(np.mean(consensus_levels))
        
        # Credibility consistency
        credibility_scores = [r.credibility_assessment.get("overall_credibility_score", 0) for r in results]
        patterns["credibility_consistency"] = float(1.0 - np.std(credibility_scores))
        
        return patterns
    
    async def _synthesize_research_findings(
        self,
        results: List[ResearchAnalysisResult]
    ) -> str:
        """Synthesize findings across multiple research results"""
        
        if not results:
            return "No research results to synthesize."
        
        synthesis_parts = []
        
        # Overall assessment
        total_results = sum(r.total_results_found for r in results)
        high_quality_total = sum(r.high_quality_results for r in results)
        
        synthesis_parts.append(
            f"Comprehensive analysis across {len(results)} research queries yielded {total_results} total results, "
            f"with {high_quality_total} meeting quality standards."
        )
        
        # Confidence synthesis
        confidences = [r.confidence_metrics.get("overall_confidence", 0) for r in results]
        avg_confidence = np.mean(confidences)
        synthesis_parts.append(f"Average research confidence: {avg_confidence:.1%}")
        
        # Key patterns
        if len(set(r.search_scope for r in results)) == 1:
            synthesis_parts.append(f"Consistent {results[0].search_scope} focus maintained across queries.")
        
        # Insights synthesis
        total_insights = sum(len(r.research_insights) for r in results)
        if total_insights > 0:
            synthesis_parts.append(f"Generated {total_insights} insights across all research queries.")
        
        return " ".join(synthesis_parts)
    
    async def _compute_overall_confidence(self, results: List[ResearchAnalysisResult]) -> float:
        """Compute overall confidence across multiple research results"""
        
        if not results:
            return 0.0
        
        confidences = [r.confidence_metrics.get("overall_confidence", 0) for r in results]
        return float(np.mean(confidences))
    
    async def _compute_research_completeness(self, results: List[ResearchAnalysisResult]) -> float:
        """Compute research completeness score"""
        
        if not results:
            return 0.0
        
        # Factors contributing to completeness
        factors = []
        
        # Quality ratio
        for result in results:
            if result.total_results_found > 0:
                quality_ratio = result.high_quality_results / result.total_results_found
                factors.append(quality_ratio)
        
        # Source diversity
        diversity_scores = [r.source_diversity.get("diversity_score", 0) for r in results]
        factors.extend(diversity_scores)
        
        # Research gaps (inverse factor)
        gap_penalties = [1.0 - (len(r.research_gaps) * 0.1) for r in results]
        factors.extend([max(0, penalty) for penalty in gap_penalties])
        
        return float(np.mean(factors)) if factors else 0.0
    
    async def _perform_fact_verification_analysis(
        self,
        research_result: ResearchAnalysisResult,
        claim: str,
        source_requirements: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Perform specialized fact verification analysis"""
        
        verification = {
            "claim": claim,
            "verification_timestamp": datetime.now(),
            "verification_confidence": 0.0,
            "supporting_sources": [],
            "contradicting_sources": [],
            "neutral_sources": [],
            "credibility_weighted_consensus": 0.0,
            "fact_check_summary": "",
            "verification_limitations": []
        }
        
        # Analyze sources for claim support
        supporting = []
        contradicting = []
        neutral = []
        
        for result in research_result.search_results:
            # Simple sentiment-based classification (could be enhanced with NLP)
            sentiment_score = result.sentiment.get("polarity", 0)
            credibility = result.credibility_score
            
            if sentiment_score > 0.1 and credibility > 0.5:
                supporting.append({
                    "url": result.url,
                    "title": result.title,
                    "credibility": credibility,
                    "relevance": result.relevance_score
                })
            elif sentiment_score < -0.1 and credibility > 0.5:
                contradicting.append({
                    "url": result.url,
                    "title": result.title,
                    "credibility": credibility,
                    "relevance": result.relevance_score
                })
            else:
                neutral.append({
                    "url": result.url,
                    "title": result.title,
                    "credibility": credibility,
                    "relevance": result.relevance_score
                })
        
        verification["supporting_sources"] = supporting
        verification["contradicting_sources"] = contradicting
        verification["neutral_sources"] = neutral
        
        # Credibility-weighted consensus
        if supporting or contradicting:
            support_weight = sum(s["credibility"] * s["relevance"] for s in supporting)
            contradict_weight = sum(c["credibility"] * c["relevance"] for c in contradicting)
            
            total_weight = support_weight + contradict_weight
            if total_weight > 0:
                verification["credibility_weighted_consensus"] = support_weight / total_weight
            
            # Overall verification confidence
            high_cred_sources = len([s for s in supporting + contradicting if s["credibility"] > 0.7])
            total_relevant = len(supporting) + len(contradicting)
            
            if total_relevant > 0:
                confidence_base = high_cred_sources / total_relevant
                consensus_factor = abs(verification["credibility_weighted_consensus"] - 0.5) * 2
                verification["verification_confidence"] = (confidence_base * 0.6) + (consensus_factor * 0.4)
        
        # Generate fact check summary
        if verification["credibility_weighted_consensus"] > 0.7:
            verification["fact_check_summary"] = f"Claim appears to be supported by available evidence (confidence: {verification['verification_confidence']:.1%})"
        elif verification["credibility_weighted_consensus"] < 0.3:
            verification["fact_check_summary"] = f"Claim appears to be contradicted by available evidence (confidence: {verification['verification_confidence']:.1%})"
        else:
            verification["fact_check_summary"] = f"Evidence is mixed or inconclusive (confidence: {verification['verification_confidence']:.1%})"
        
        # Identify limitations
        if len(supporting) + len(contradicting) < 3:
            verification["verification_limitations"].append("Limited number of relevant sources found")
        
        if research_result.credibility_assessment.get("overall_credibility_score", 0) < 0.6:
            verification["verification_limitations"].append("Overall source credibility below optimal threshold")
        
        return verification
    
    async def _analyze_individual_source(
        self,
        url: str,
        research_result: ResearchAnalysisResult,
        analysis_depth: str
    ) -> Dict[str, Any]:
        """Analyze individual source comprehensively"""
        
        analysis = {
            "url": url,
            "analysis_timestamp": datetime.now(),
            "analysis_depth": analysis_depth,
            "credibility_score": 0.0,
            "bias_score": 0.0,
            "authority_score": 0.0,
            "recency_score": 0.0,
            "content_quality": 0.0,
            "source_classification": "unknown",
            "key_topics": [],
            "potential_issues": []
        }
        
        # Find matching results for this URL
        matching_results = [r for r in research_result.search_results if r.url == url]
        
        if matching_results:
            result = matching_results[0]  # Use first match
            
            analysis.update({
                "credibility_score": result.credibility_score,
                "bias_score": result.bias_score,
                "source_classification": result.source_type,
                "key_topics": result.topics,
                "recency_score": result.recency_score
            })
            
            # Content quality assessment
            content_indicators = [
                result.credibility_score > 0.6,
                result.bias_score < 0.4,
                len(result.key_entities) > 2,
                len(result.topics) > 1
            ]
            analysis["content_quality"] = sum(content_indicators) / len(content_indicators)
            
            # Identify potential issues
            if result.bias_score > 0.5:
                analysis["potential_issues"].append("High bias detected")
            if result.credibility_score < 0.5:
                analysis["potential_issues"].append("Low credibility score")
            if result.recency_score < 0.3:
                analysis["potential_issues"].append("Information may be outdated")
        
        return analysis
    
    async def _generate_source_comparison(self, source_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparative analysis across sources"""
        
        comparison = {
            "credibility_distribution": {},
            "bias_patterns": {},
            "quality_rankings": [],
            "source_type_diversity": {}
        }
        
        if not source_analyses:
            return comparison
        
        # Credibility distribution
        credibility_scores = [s.get("credibility_score", 0) for s in source_analyses]
        comparison["credibility_distribution"] = {
            "mean": float(np.mean(credibility_scores)),
            "std": float(np.std(credibility_scores)),
            "min": float(np.min(credibility_scores)),
            "max": float(np.max(credibility_scores))
        }
        
        # Quality rankings
        ranked_sources = sorted(
            source_analyses,
            key=lambda s: (s.get("credibility_score", 0) + s.get("content_quality", 0)) / 2,
            reverse=True
        )
        comparison["quality_rankings"] = [s["url"] for s in ranked_sources]
        
        # Source type diversity
        source_types = [s.get("source_classification", "unknown") for s in source_analyses]
        type_counts = {}
        for source_type in source_types:
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
        
        comparison["source_type_diversity"] = type_counts
        
        return comparison
    
    async def _compute_credibility_distribution(self, source_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute credibility distribution across sources"""
        
        if not source_analyses:
            return {}
        
        scores = [s.get("credibility_score", 0) for s in source_analyses]
        
        return {
            "very_high": len([s for s in scores if s > 0.9]),
            "high": len([s for s in scores if 0.7 < s <= 0.9]),
            "moderate": len([s for s in scores if 0.5 < s <= 0.7]),
            "low": len([s for s in scores if 0.3 < s <= 0.5]),
            "very_low": len([s for s in scores if s <= 0.3]),
            "average": float(np.mean(scores)),
            "median": float(np.median(scores))
        }
    
    async def _identify_bias_patterns(self, source_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify bias patterns across sources"""
        
        if not source_analyses:
            return {}
        
        bias_scores = [s.get("bias_score", 0) for s in source_analyses]
        high_bias_sources = [s["url"] for s in source_analyses if s.get("bias_score", 0) > 0.6]
        
        return {
            "average_bias": float(np.mean(bias_scores)),
            "high_bias_sources": high_bias_sources,
            "bias_distribution": {
                "low": len([s for s in bias_scores if s <= 0.3]),
                "moderate": len([s for s in bias_scores if 0.3 < s <= 0.6]),
                "high": len([s for s in bias_scores if s > 0.6])
            }
        }
    
    async def _generate_source_recommendations(self, source_analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on source analysis"""
        
        recommendations = []
        
        if not source_analyses:
            return recommendations
        
        # Quality recommendations
        high_quality_count = len([s for s in source_analyses if s.get("content_quality", 0) > 0.7])
        if high_quality_count < len(source_analyses) * 0.5:
            recommendations.append("Consider seeking higher quality sources")
        
        # Bias recommendations
        high_bias_count = len([s for s in source_analyses if s.get("bias_score", 0) > 0.6])
        if high_bias_count > 0:
            recommendations.append(f"Review {high_bias_count} sources with potential bias")
        
        # Diversity recommendations
        source_types = set(s.get("source_classification", "unknown") for s in source_analyses)
        if len(source_types) < 3:
            recommendations.append("Increase source type diversity for comprehensive coverage")
        
        return recommendations
    
    async def _store_research_pattern(
        self,
        research_type: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ):
        """Store research patterns for future optimization"""
        if not self.memory_service or not context:
            return
        
        try:
            memory_content = {
                "research_type": f"web_{research_type}",
                "result_summary": {
                    "success": True,
                    "execution_time": datetime.now().isoformat(),
                    "key_metrics": result
                },
                "context": context,
                "agent_id": self.agent_id
            }
            
            await self.memory_service.store_conversation_memory(
                user_id=context.get("user_id", "system"),
                message=f"Web research: {research_type}",
                context={
                    "category": "web_research",
                    "research_type": research_type,
                    "agent_id": self.agent_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store research pattern: {e}")