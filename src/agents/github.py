"""
Advanced GitHub Agent - Production Grade
Intelligent repository security analysis with ML-driven vulnerability assessment
"""

import asyncio
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from pydantic import BaseModel, Field, validator, HttpUrl
from enum import Enum

from .base.agent import HardenedAgent
from .base.registry import tool
from .base.exceptions import ToolError
from src.adapters.github.intelligent_analyzer import (
    IntelligentGitHubAnalyzer,
    RepositorySecurityProfile,
    SecurityFinding,
    ContributorBehaviorProfile,
    SecurityRiskLevel,
    VulnerabilityType
)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


# Pydantic Models for Input/Output Validation
class RepositoryIdentifier(BaseModel):
    """Repository identification with validation"""
    url: HttpUrl = Field(..., description="GitHub repository URL")
    branch: Optional[str] = Field("main", description="Branch to analyze")
    clone_depth: int = Field(100, ge=1, le=1000, description="Clone depth for analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "url": "https://github.com/owner/repository",
                "branch": "main",
                "clone_depth": 100
            }
        }


class SecurityAnalysisRequest(BaseModel):
    """Security analysis configuration"""
    repository: RepositoryIdentifier
    analysis_scope: List[str] = Field(
        default=["vulnerabilities", "dependencies", "contributors", "code_quality"],
        description="Analysis components to include"
    )
    risk_tolerance: str = Field("medium", description="Risk tolerance level")
    include_ml_analysis: bool = Field(True, description="Include ML-based analysis")
    
    @validator('risk_tolerance')
    def validate_risk_tolerance(cls, v):
        allowed = ['low', 'medium', 'high']
        if v not in allowed:
            raise ValueError(f'risk_tolerance must be one of {allowed}')
        return v


class VulnerabilityFinding(BaseModel):
    """Individual vulnerability finding"""
    finding_id: str
    vulnerability_type: str
    severity: str  
    title: str
    description: str
    file_path: str
    line_number: Optional[int]
    confidence_score: float = Field(..., ge=0, le=1)
    epss_score: Optional[float] = Field(None, ge=0, le=1)
    cve_references: List[str] = Field(default_factory=list)
    mitigation_steps: List[str] = Field(default_factory=list)
    tool_source: str


class ContributorRiskProfile(BaseModel):
    """Contributor risk assessment"""
    contributor_id: str
    commit_frequency: float = Field(..., ge=0)
    security_awareness_score: float = Field(..., ge=0, le=1)
    anomaly_score: float = Field(..., ge=-1, le=1)
    risk_indicators: List[str] = Field(default_factory=list)
    contribution_quality_score: float = Field(..., ge=0, le=1)


class SecurityAnalysisResult(BaseModel):
    """Comprehensive security analysis results"""
    repository_url: str
    analysis_timestamp: datetime
    overall_risk_score: float = Field(..., ge=0, le=1)
    risk_level: str
    
    vulnerability_findings: List[VulnerabilityFinding]
    contributor_profiles: List[ContributorRiskProfile]
    dependency_issues: List[Dict[str, Any]] = Field(default_factory=list)
    
    code_quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendation_matrix: Dict[str, List[str]] = Field(default_factory=dict)
    confidence_assessment: Dict[str, float] = Field(default_factory=dict)
    
    executive_summary: str
    schema_version: str = Field(default="1.0")


class RepositoryComparisonRequest(BaseModel):
    """Request for repository comparison analysis"""
    repositories: List[RepositoryIdentifier] = Field(..., min_items=2, max_items=5)
    comparison_metrics: List[str] = Field(
        default=["security", "quality", "contributors"],
        description="Metrics to compare"
    )


class ComparisonResult(BaseModel):
    """Repository comparison results"""
    repositories: List[str]
    comparison_timestamp: datetime
    metric_comparisons: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    rankings: Dict[str, List[str]] = Field(default_factory=dict)
    recommendations: Dict[str, List[str]] = Field(default_factory=dict)
    schema_version: str = Field(default="1.0")


class AdvancedGitHubAgent(HardenedAgent):
    """
    Advanced GitHub Agent with intelligent repository security analysis
    
    Features:
    - Multi-tool security vulnerability scanning
    - ML-driven contributor behavior analysis  
    - EPSS-enhanced vulnerability scoring
    - Intelligent repository comparison
    - Memory-driven optimization
    - Advanced dependency analysis
    """
    
    _domain = "github"
    _capabilities = [
        "repository_security_analysis",
        "vulnerability_assessment", 
        "contributor_behavior_analysis",
        "dependency_security_scanning",
        "code_quality_evaluation",
        "repository_comparison",
        "ml_driven_risk_assessment",
        "epss_vulnerability_scoring"
    ]
    
    def __init__(self, memory_service, agent_id: Optional[str] = None):
        super().__init__(memory_service, agent_id)
        self.github_analyzer = IntelligentGitHubAnalyzer(memory_service)
        self._analysis_cache = {}
        
        logger.info(
            "advanced_github_agent_initialized",
            agent_id=self.agent_id,
            capabilities=self._capabilities
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
        """Intelligent message processing with GitHub context awareness"""
        
        request_id = plan.get('request_id', uuid4().hex)
        
        logger.info(
            "github_message_processing",
            agent_id=self.agent_id,
            request_id=request_id,
            message_type=await self._classify_github_intent(message),
            memory_context_size=len(memory_context)
        )
        
        intent = await self._classify_github_intent(message)
        
        if intent == "security_analysis":
            return f"I can perform comprehensive repository security analysis using multi-tool scanning, ML-driven vulnerability assessment, and EPSS scoring. Memory context: {len(memory_context)} relevant patterns found."
        elif intent == "vulnerability_assessment":
            return f"I'll analyze vulnerabilities using Bandit, Semgrep, and custom ML models with EPSS-enhanced scoring. {len(memory_context)} previous assessments available."
        elif intent == "contributor_analysis":
            return f"I can analyze contributor behavior patterns using ML anomaly detection and risk profiling. {len(memory_context)} contributor patterns in memory."
        elif intent == "repository_comparison":
            return f"I'll compare repositories across security, quality, and contributor metrics with intelligent ranking. {len(memory_context)} comparison patterns available."
        else:
            return f"I can assist with GitHub repository analysis including security scanning, contributor behavior analysis, dependency checking, and intelligent repository comparison. {len(memory_context)} relevant memories found."
    
    @tool(description="Comprehensive repository security analysis with ML-driven insights")
    async def analyze_repository_security(
        self,
        analysis_request: SecurityAnalysisRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> SecurityAnalysisResult:
        """
        Perform comprehensive repository security analysis including:
        - Multi-tool vulnerability scanning (Bandit, Semgrep, ML)
        - ML-driven contributor behavior analysis
        - EPSS-enhanced vulnerability scoring
        - Dependency security assessment
        - Code quality evaluation
        
        Complexity: O(n log n) for n files in repository
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "repository_security_analysis_started",
            agent_id=self.agent_id,
            request_id=request_id,
            repository_url=str(analysis_request.repository.url),
            analysis_scope=analysis_request.analysis_scope,
            estimated_cost=0.25
        )
        
        try:
            # Execute comprehensive analysis using intelligent analyzer
            security_profile = await self.github_analyzer.analyze_repository_comprehensive(
                repo_url=str(analysis_request.repository.url),
                clone_depth=analysis_request.repository.clone_depth,
                context=context or {}
            )
            
            # Convert internal format to API format
            result = await self._convert_security_profile_to_result(
                security_profile, analysis_request
            )
            
            # Store analysis pattern for future optimization
            await self._store_analysis_pattern("security_analysis", result, context)
            
            logger.info(
                "repository_security_analysis_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                repository_url=str(analysis_request.repository.url),
                overall_risk_score=result.overall_risk_score,
                findings_count=len(result.vulnerability_findings)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "repository_security_analysis_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                repository_url=str(analysis_request.repository.url),
                error=str(e),
                error_type=type(e).__name__
            )
            raise ToolError(
                f"Repository security analysis failed: {str(e)}",
                tool_name="analyze_repository_security",
                agent_id=self.agent_id
            )
    
    @tool(description="Advanced vulnerability assessment with EPSS scoring and ML ranking")
    async def assess_vulnerabilities(
        self,
        repository_url: HttpUrl,
        vulnerability_types: Optional[List[str]] = None,
        include_epss: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Focused vulnerability assessment with advanced scoring:
        - Multi-tool scanning integration
        - EPSS (Exploit Prediction Scoring System) enhancement
        - ML-based risk ranking
        - Intelligent deduplication
        
        Complexity: O(n * m) for n files and m vulnerability patterns
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "vulnerability_assessment_started",
            agent_id=self.agent_id,
            request_id=request_id,
            repository_url=str(repository_url),
            include_epss=include_epss,
            estimated_cost=0.15
        )
        
        try:
            # Create minimal analysis request focused on vulnerabilities
            analysis_request = SecurityAnalysisRequest(
                repository=RepositoryIdentifier(url=repository_url),
                analysis_scope=["vulnerabilities"],
                include_ml_analysis=True
            )
            
            # Execute focused vulnerability analysis
            security_profile = await self.github_analyzer.analyze_repository_comprehensive(
                repo_url=str(repository_url),
                clone_depth=100,  # Default depth
                context=context or {}
            )
            
            # Filter and enhance vulnerability data
            vulnerability_data = await self._process_vulnerability_findings(
                security_profile.security_findings,
                vulnerability_types,
                include_epss
            )
            
            result = {
                "repository_url": str(repository_url),
                "assessment_timestamp": datetime.now(),
                "total_vulnerabilities": len(vulnerability_data["findings"]),
                "critical_vulnerabilities": vulnerability_data["severity_breakdown"]["critical"],
                "high_vulnerabilities": vulnerability_data["severity_breakdown"]["high"],
                "vulnerability_findings": vulnerability_data["findings"],
                "risk_score": vulnerability_data["overall_risk_score"],
                "epss_statistics": vulnerability_data.get("epss_stats", {}),
                "tool_coverage": vulnerability_data["tool_coverage"],
                "recommendations": vulnerability_data["recommendations"]
            }
            
            # Store vulnerability assessment pattern
            await self._store_analysis_pattern("vulnerability_assessment", result, context)
            
            logger.info(
                "vulnerability_assessment_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                repository_url=str(repository_url),
                vulnerabilities_found=len(vulnerability_data["findings"])
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "vulnerability_assessment_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                repository_url=str(repository_url),
                error=str(e)
            )
            raise ToolError(
                f"Vulnerability assessment failed: {str(e)}",
                tool_name="assess_vulnerabilities", 
                agent_id=self.agent_id
            )
    
    @tool(description="ML-driven contributor behavior analysis with anomaly detection")
    async def analyze_contributor_behavior(
        self,
        repository_url: HttpUrl,
        analysis_depth: int = 100,
        include_anomaly_detection: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advanced contributor behavior analysis:
        - Commit pattern analysis
        - Code complexity trend evaluation
        - Security awareness scoring
        - ML-based anomaly detection
        - Risk profiling and scoring
        
        Complexity: O(c * log(c)) for c commits analyzed
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "contributor_behavior_analysis_started",
            agent_id=self.agent_id,
            request_id=request_id,
            repository_url=str(repository_url),
            analysis_depth=analysis_depth,
            estimated_cost=0.12
        )
        
        try:
            # Execute contributor-focused analysis
            security_profile = await self.github_analyzer.analyze_repository_comprehensive(
                repo_url=str(repository_url),
                clone_depth=analysis_depth,
                context=context or {}
            )
            
            # Process contributor profiles
            contributor_analysis = await self._process_contributor_profiles(
                security_profile.contributor_profiles,
                include_anomaly_detection
            )
            
            result = {
                "repository_url": str(repository_url),
                "analysis_timestamp": datetime.now(),
                "total_contributors": len(contributor_analysis["profiles"]),
                "contributor_profiles": contributor_analysis["profiles"],
                "anomalous_contributors": contributor_analysis["anomalous_count"],
                "risk_distribution": contributor_analysis["risk_distribution"],
                "collaboration_insights": contributor_analysis["collaboration_patterns"],
                "security_awareness_stats": contributor_analysis["security_awareness"],
                "recommendations": contributor_analysis["recommendations"]
            }
            
            # Store contributor analysis pattern
            await self._store_analysis_pattern("contributor_analysis", result, context)
            
            logger.info(
                "contributor_behavior_analysis_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                repository_url=str(repository_url),
                contributors_analyzed=len(contributor_analysis["profiles"]),
                anomalous_contributors=contributor_analysis["anomalous_count"]
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "contributor_behavior_analysis_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                repository_url=str(repository_url),
                error=str(e)
            )
            raise ToolError(
                f"Contributor behavior analysis failed: {str(e)}",
                tool_name="analyze_contributor_behavior",
                agent_id=self.agent_id
            )
    
    @tool(description="Intelligent repository comparison with multi-dimensional analysis")
    async def compare_repositories(
        self,
        comparison_request: RepositoryComparisonRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> ComparisonResult:
        """
        Advanced repository comparison across multiple dimensions:
        - Security posture comparison
        - Code quality metrics
        - Contributor patterns
        - Dependency health
        - Intelligent ranking and recommendations
        
        Complexity: O(r * n * log(n)) for r repositories with n files each
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "repository_comparison_started",
            agent_id=self.agent_id,
            request_id=request_id,
            repository_count=len(comparison_request.repositories),
            comparison_metrics=comparison_request.comparison_metrics,
            estimated_cost=len(comparison_request.repositories) * 0.20
        )
        
        try:
            # Analyze all repositories in parallel
            analysis_tasks = []
            for repo in comparison_request.repositories:
                task = self.github_analyzer.analyze_repository_comprehensive(
                    repo_url=str(repo.url),
                    clone_depth=repo.clone_depth,
                    context=context or {}
                )
                analysis_tasks.append(task)
            
            repository_profiles = await asyncio.gather(*analysis_tasks)
            
            # Perform comparative analysis
            comparison_data = await self._perform_repository_comparison(
                repository_profiles,
                comparison_request.comparison_metrics
            )
            
            result = ComparisonResult(
                repositories=[str(repo.url) for repo in comparison_request.repositories],
                comparison_timestamp=datetime.now(),
                metric_comparisons=comparison_data["metric_comparisons"],
                rankings=comparison_data["rankings"],
                recommendations=comparison_data["recommendations"]
            )
            
            # Store comparison pattern
            await self._store_analysis_pattern("repository_comparison", result.dict(), context)
            
            logger.info(
                "repository_comparison_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                repositories_compared=len(comparison_request.repositories),
                metrics_analyzed=len(comparison_request.comparison_metrics)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "repository_comparison_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise ToolError(
                f"Repository comparison failed: {str(e)}",
                tool_name="compare_repositories",
                agent_id=self.agent_id
            )
    
    # Helper methods
    
    async def _classify_github_intent(self, message: str) -> str:
        """Classify user message intent for appropriate GitHub response"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["security", "vulnerability", "vuln", "cve"]):
            return "security_analysis" if "analysis" in message_lower else "vulnerability_assessment"
        elif any(word in message_lower for word in ["contributor", "author", "committer", "behavior"]):
            return "contributor_analysis"
        elif any(word in message_lower for word in ["compare", "comparison", "vs", "versus"]):
            return "repository_comparison"
        elif any(word in message_lower for word in ["repository", "repo", "github", "analyze"]):
            return "repository_analysis"
        else:
            return "general_github"
    
    async def _convert_security_profile_to_result(
        self,
        profile: RepositorySecurityProfile,
        request: SecurityAnalysisRequest
    ) -> SecurityAnalysisResult:
        """Convert internal security profile to API result format"""
        
        # Convert vulnerability findings
        vulnerability_findings = []
        for finding in profile.security_findings:
            vuln_finding = VulnerabilityFinding(
                finding_id=finding.finding_id,
                vulnerability_type=finding.vulnerability_type.value,
                severity=finding.risk_level.value,
                title=finding.title,
                description=finding.description,
                file_path=finding.file_path,
                line_number=finding.line_number,
                confidence_score=finding.confidence_score,
                epss_score=finding.epss_score,
                cve_references=finding.cve_references,
                mitigation_steps=finding.mitigation_suggestions,
                tool_source=finding.tool_source
            )
            vulnerability_findings.append(vuln_finding)
        
        # Convert contributor profiles
        contributor_profiles = []
        for contrib in profile.contributor_profiles:
            risk_profile = ContributorRiskProfile(
                contributor_id=contrib.contributor_id,
                commit_frequency=contrib.commit_frequency,
                security_awareness_score=contrib.security_awareness_score,
                anomaly_score=contrib.anomaly_score,
                risk_indicators=contrib.risk_indicators,
                contribution_quality_score=contrib.contribution_quality_score
            )
            contributor_profiles.append(risk_profile)
        
        # Determine risk level from score
        risk_level = "low"
        if profile.overall_risk_score >= 0.8:
            risk_level = "critical"
        elif profile.overall_risk_score >= 0.6:
            risk_level = "high"
        elif profile.overall_risk_score >= 0.4:
            risk_level = "medium"
        
        # Generate executive summary
        executive_summary = await self._generate_executive_summary(
            profile, len(vulnerability_findings), len(contributor_profiles)
        )
        
        return SecurityAnalysisResult(
            repository_url=profile.repository_url,
            analysis_timestamp=profile.analysis_timestamp,
            overall_risk_score=profile.overall_risk_score,
            risk_level=risk_level,
            vulnerability_findings=vulnerability_findings,
            contributor_profiles=contributor_profiles,
            dependency_issues=profile.dependency_vulnerabilities,
            code_quality_metrics=profile.code_quality_metrics,
            recommendation_matrix=profile.recommendation_priority_matrix,
            confidence_assessment=profile.confidence_metrics,
            executive_summary=executive_summary
        )
    
    async def _process_vulnerability_findings(
        self,
        findings: List[SecurityFinding],
        vulnerability_types: Optional[List[str]],
        include_epss: bool
    ) -> Dict[str, Any]:
        """Process and filter vulnerability findings"""
        
        # Filter by type if specified
        if vulnerability_types:
            type_filter = set(vulnerability_types)
            findings = [f for f in findings if f.vulnerability_type.value in type_filter]
        
        # Count by severity
        severity_counts = {
            "critical": len([f for f in findings if f.risk_level == SecurityRiskLevel.CRITICAL]),
            "high": len([f for f in findings if f.risk_level == SecurityRiskLevel.HIGH]),
            "medium": len([f for f in findings if f.risk_level == SecurityRiskLevel.MEDIUM]),
            "low": len([f for f in findings if f.risk_level == SecurityRiskLevel.LOW])
        }
        
        # Calculate risk score
        risk_weights = {"critical": 10, "high": 7, "medium": 4, "low": 2}
        total_risk = sum(severity_counts[sev] * weight for sev, weight in risk_weights.items())
        max_possible_risk = len(findings) * risk_weights["critical"] if findings else 1
        overall_risk_score = min(1.0, total_risk / max_possible_risk)
        
        # EPSS statistics
        epss_stats = {}
        if include_epss:
            epss_scores = [f.epss_score for f in findings if f.epss_score is not None]
            if epss_scores:
                epss_stats = {
                    "mean_epss": float(np.mean(epss_scores)),
                    "max_epss": float(np.max(epss_scores)),
                    "high_epss_count": len([s for s in epss_scores if s > 0.7])
                }
        
        # Tool coverage
        tools_used = set(f.tool_source for f in findings)
        tool_coverage = {
            "tools_used": list(tools_used),
            "coverage_score": min(1.0, len(tools_used) / 3)  # Expect 3 main tools
        }
        
        # Recommendations
        recommendations = []
        if severity_counts["critical"] > 0:
            recommendations.append("Immediately address critical vulnerabilities")
        if severity_counts["high"] > 3:
            recommendations.append("Prioritize high-severity vulnerability remediation")
        if overall_risk_score > 0.7:
            recommendations.append("Consider comprehensive security review")
        
        return {
            "findings": [
                {
                    "id": f.finding_id,
                    "type": f.vulnerability_type.value,
                    "severity": f.risk_level.value,
                    "title": f.title,
                    "file": f.file_path,
                    "line": f.line_number,
                    "confidence": f.confidence_score,
                    "epss_score": f.epss_score,
                    "tool": f.tool_source
                } for f in findings
            ],
            "severity_breakdown": severity_counts,
            "overall_risk_score": overall_risk_score,
            "epss_stats": epss_stats,
            "tool_coverage": tool_coverage,
            "recommendations": recommendations
        }
    
    async def _process_contributor_profiles(
        self,
        profiles: List[ContributorBehaviorProfile],
        include_anomaly_detection: bool
    ) -> Dict[str, Any]:
        """Process contributor profiles for analysis results"""
        
        # Count anomalous contributors
        anomalous_count = len([p for p in profiles if 'behavioral_anomaly_detected' in p.risk_indicators])
        
        # Risk distribution
        risk_levels = []
        for profile in profiles:
            if len(profile.risk_indicators) >= 3:
                risk_levels.append("high")
            elif len(profile.risk_indicators) >= 1:
                risk_levels.append("medium")
            else:
                risk_levels.append("low")
        
        risk_distribution = {
            "high": risk_levels.count("high"),
            "medium": risk_levels.count("medium"),
            "low": risk_levels.count("low")
        }
        
        # Collaboration patterns
        if profiles:
            collaboration_patterns = {
                "avg_quality_score": float(np.mean([p.contribution_quality_score for p in profiles])),
                "avg_security_awareness": float(np.mean([p.security_awareness_score for p in profiles])),
                "high_frequency_contributors": len([p for p in profiles if p.commit_frequency > 5])
            }
        else:
            collaboration_patterns = {"avg_quality_score": 0, "avg_security_awareness": 0, "high_frequency_contributors": 0}
        
        # Security awareness stats
        security_awareness = {
            "mean_score": collaboration_patterns["avg_security_awareness"],
            "low_awareness_count": len([p for p in profiles if p.security_awareness_score < 0.3])
        }
        
        # Recommendations
        recommendations = []
        if anomalous_count > 0:
            recommendations.append(f"Investigate {anomalous_count} contributors with anomalous behavior")
        if security_awareness["low_awareness_count"] > len(profiles) // 2:
            recommendations.append("Consider security training for development team")
        if risk_distribution["high"] > 0:
            recommendations.append("Review high-risk contributor access and permissions")
        
        return {
            "profiles": [
                {
                    "contributor": p.contributor_id,
                    "commit_frequency": p.commit_frequency,
                    "security_awareness": p.security_awareness_score,
                    "quality_score": p.contribution_quality_score,
                    "risk_indicators": p.risk_indicators,
                    "anomaly_score": p.anomaly_score
                } for p in profiles
            ],
            "anomalous_count": anomalous_count,
            "risk_distribution": risk_distribution,
            "collaboration_patterns": collaboration_patterns,
            "security_awareness": security_awareness,
            "recommendations": recommendations
        }
    
    async def _perform_repository_comparison(
        self,
        profiles: List[RepositorySecurityProfile],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform intelligent repository comparison"""
        
        comparison_data = {
            "metric_comparisons": {},
            "rankings": {},
            "recommendations": {}
        }
        
        repo_urls = [p.repository_url for p in profiles]
        
        # Security comparison
        if "security" in metrics:
            security_scores = [p.overall_risk_score for p in profiles]
            comparison_data["metric_comparisons"]["security_risk"] = dict(zip(repo_urls, security_scores))
            
            # Ranking (lower risk = better)
            security_ranking = sorted(zip(repo_urls, security_scores), key=lambda x: x[1])
            comparison_data["rankings"]["security"] = [repo for repo, _ in security_ranking]
        
        # Quality comparison  
        if "quality" in metrics:
            quality_scores = []
            for profile in profiles:
                # Simple quality score based on test coverage and complexity
                test_coverage = profile.code_quality_metrics.get('test_coverage_estimate', 0)
                complexity = profile.code_quality_metrics.get('complexity_score', 10)
                quality_score = test_coverage * 0.6 + max(0, (10 - complexity) / 10) * 0.4
                quality_scores.append(quality_score)
            
            comparison_data["metric_comparisons"]["quality_score"] = dict(zip(repo_urls, quality_scores))
            
            # Ranking (higher quality = better)
            quality_ranking = sorted(zip(repo_urls, quality_scores), key=lambda x: x[1], reverse=True)
            comparison_data["rankings"]["quality"] = [repo for repo, _ in quality_ranking]
        
        # Contributor comparison
        if "contributors" in metrics:
            contributor_scores = []
            for profile in profiles:
                if profile.contributor_profiles:
                    avg_quality = np.mean([c.contribution_quality_score for c in profile.contributor_profiles])
                    anomaly_penalty = len([c for c in profile.contributor_profiles if 'behavioral_anomaly_detected' in c.risk_indicators]) / len(profile.contributor_profiles)
                    contributor_score = avg_quality * (1 - anomaly_penalty)
                else:
                    contributor_score = 0.5  # Neutral score for repos with no contributor data
                contributor_scores.append(contributor_score)
            
            comparison_data["metric_comparisons"]["contributor_quality"] = dict(zip(repo_urls, contributor_scores))
            
            # Ranking (higher contributor quality = better)
            contributor_ranking = sorted(zip(repo_urls, contributor_scores), key=lambda x: x[1], reverse=True)
            comparison_data["rankings"]["contributors"] = [repo for repo, _ in contributor_ranking]
        
        # Generate recommendations for each repository
        for i, profile in enumerate(profiles):
            repo_url = profile.repository_url
            recommendations = []
            
            # Security recommendations
            if profile.overall_risk_score > 0.6:
                recommendations.append("High security risk - prioritize vulnerability remediation")
            
            # Quality recommendations
            test_coverage = profile.code_quality_metrics.get('test_coverage_estimate', 0)
            if test_coverage < 0.3:
                recommendations.append("Low test coverage - improve testing practices")
            
            # Contributor recommendations
            anomalous_contributors = len([c for c in profile.contributor_profiles if 'behavioral_anomaly_detected' in c.risk_indicators])
            if anomalous_contributors > 0:
                recommendations.append(f"Review {anomalous_contributors} contributors with unusual patterns")
            
            comparison_data["recommendations"][repo_url] = recommendations
        
        return comparison_data
    
    async def _generate_executive_summary(
        self,
        profile: RepositorySecurityProfile,
        findings_count: int,
        contributors_count: int
    ) -> str:
        """Generate intelligent executive summary"""
        
        risk_adjective = "low"
        if profile.overall_risk_score >= 0.8:
            risk_adjective = "critical"
        elif profile.overall_risk_score >= 0.6:
            risk_adjective = "high"  
        elif profile.overall_risk_score >= 0.4:
            risk_adjective = "moderate"
        
        critical_findings = len([f for f in profile.security_findings if f.risk_level == SecurityRiskLevel.CRITICAL])
        high_findings = len([f for f in profile.security_findings if f.risk_level == SecurityRiskLevel.HIGH])
        
        summary_parts = [
            f"Repository security analysis reveals {risk_adjective} overall risk (score: {profile.overall_risk_score:.2f}).",
            f"Found {findings_count} security findings across {len(profile.code_quality_metrics.get('language_distribution', {}))} programming languages."
        ]
        
        if critical_findings > 0 or high_findings > 0:
            summary_parts.append(f"Priority issues: {critical_findings} critical and {high_findings} high-severity vulnerabilities require immediate attention.")
        
        if contributors_count > 0:
            anomalous_contributors = len([c for c in profile.contributor_profiles if 'behavioral_anomaly_detected' in c.risk_indicators])
            if anomalous_contributors > 0:
                summary_parts.append(f"Contributor analysis identified {anomalous_contributors} individuals with unusual behavior patterns.")
        
        summary_parts.append(f"Code quality metrics indicate {profile.code_quality_metrics.get('total_files', 0)} files analyzed with {profile.code_quality_metrics.get('test_coverage_estimate', 0):.1%} estimated test coverage.")
        
        return " ".join(summary_parts)
    
    async def _store_analysis_pattern(
        self,
        analysis_type: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ):
        """Store GitHub analysis patterns for future optimization"""
        if not self.memory_service or not context:
            return
        
        try:
            memory_content = {
                "analysis_type": f"github_{analysis_type}",
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
                message=f"GitHub analysis: {analysis_type}",
                context={
                    "category": "github_analysis",
                    "analysis_type": analysis_type,
                    "agent_id": self.agent_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store analysis pattern: {e}")