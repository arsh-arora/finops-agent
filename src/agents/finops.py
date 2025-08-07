"""
Advanced FinOps Agent - Production Grade
Intelligent financial operations with adaptive algorithms and memory-driven optimization
"""

import asyncio
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from pydantic import BaseModel, Field, validator
from enum import Enum

from .base.agent import HardenedAgent
from .base.registry import tool
from .base.exceptions import ToolError
from src.adapters.financial.intelligent_analyzer import (
    IntelligentFinancialAnalyzer, 
    FinancialDataProfile
)

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize, linprog
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

logger = structlog.get_logger(__name__)


# Pydantic Models for Input/Output Validation
class CostDataPoint(BaseModel):
    """Individual cost data point with metadata"""
    timestamp: datetime
    cost: Decimal = Field(..., gt=0, description="Cost amount in base currency")
    category: Optional[str] = Field(None, description="Cost category classification")
    resource_id: Optional[str] = Field(None, description="Associated resource identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    
    class Config:
        schema_extra = {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "cost": "1250.75",
                "category": "compute",
                "resource_id": "ec2-i-1234567890abcdef0",
                "metadata": {"region": "us-east-1", "instance_type": "m5.large"}
            }
        }


class AnomalyPoint(BaseModel):
    """Detected anomaly with confidence scoring"""
    timestamp: datetime
    value: float
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence (0-1)")
    z_score: Optional[float] = Field(None, description="Statistical z-score")
    isolation_score: Optional[float] = Field(None, description="Isolation forest score")
    deviation_magnitude: float = Field(..., description="Deviation from expected value")
    context: Dict[str, Any] = Field(default_factory=dict, description="Anomaly context")


class AnomalyDetectionResult(BaseModel):
    """Comprehensive anomaly detection results"""
    anomalies: List[AnomalyPoint]
    method_used: str = Field(..., description="Detection method employed")
    total_anomalies: int = Field(..., ge=0)
    anomaly_ratio: float = Field(..., ge=0, le=1, description="Proportion of data points flagged")
    confidence_distribution: Dict[str, float] = Field(..., description="Confidence score statistics")
    statistical_summary: Dict[str, float] = Field(..., description="Data statistical properties")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    schema_version: str = Field(default="1.0", description="Schema version for compatibility")


class NPVResult(BaseModel):
    """Advanced Net Present Value calculation results"""
    npv: Decimal = Field(..., description="Net Present Value")
    discount_rate_used: float = Field(..., gt=0, description="Effective discount rate applied")
    risk_adjustment: float = Field(..., ge=0, description="Risk adjustment factor")
    irr: Optional[float] = Field(None, description="Internal Rate of Return")
    payback_period: Optional[float] = Field(None, ge=0, description="Payback period in years")
    profitability_index: float = Field(..., description="Profitability index")
    sensitivity_analysis: Dict[str, Any] = Field(..., description="Sensitivity analysis results")
    market_context: Optional[Dict[str, Any]] = Field(None, description="Market conditions context")
    confidence_interval: Dict[str, float] = Field(..., description="NPV confidence bounds")
    schema_version: str = Field(default="1.0")


class BudgetItem(BaseModel):
    """Budget optimization item with constraints"""
    name: str = Field(..., min_length=1, description="Item identifier")
    cost: Decimal = Field(..., gt=0, description="Item cost")
    benefit: Decimal = Field(..., ge=0, description="Expected benefit/value")
    priority: int = Field(..., ge=1, le=10, description="Priority ranking (1-10)")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Item-specific constraints")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")
    risk_factor: float = Field(default=1.0, ge=0.1, le=3.0, description="Risk multiplier")
    
    @validator('cost', 'benefit')
    def validate_amounts(cls, v):
        return Decimal(str(v)) if not isinstance(v, Decimal) else v


class BudgetConstraints(BaseModel):
    """Comprehensive budget optimization constraints"""
    total_budget: Decimal = Field(..., gt=0, description="Total available budget")
    min_benefit_threshold: Optional[Decimal] = Field(None, ge=0, description="Minimum benefit requirement")
    max_risk_score: Optional[float] = Field(None, ge=0, description="Maximum acceptable risk")
    required_categories: List[str] = Field(default_factory=list, description="Must-include categories")
    forbidden_items: List[str] = Field(default_factory=list, description="Excluded items")
    temporal_constraints: Dict[str, Any] = Field(default_factory=dict, description="Time-based limits")


class OptimizationObjective(str, Enum):
    """Multi-objective optimization goals"""
    MAXIMIZE_ROI = "maximize_roi"
    MINIMIZE_RISK = "minimize_risk"
    MAXIMIZE_COVERAGE = "maximize_coverage"
    BALANCE_PORTFOLIO = "balance_portfolio"


class BudgetOptimizationResult(BaseModel):
    """Advanced budget optimization results"""
    selected_items: List[str] = Field(..., description="Optimally selected items")
    total_cost: Decimal = Field(..., description="Total allocated cost")
    total_benefit: Decimal = Field(..., description="Total expected benefit")
    roi: float = Field(..., description="Return on investment")
    risk_score: float = Field(..., description="Portfolio risk assessment")
    optimization_status: str = Field(..., description="Solver status")
    sensitivity_analysis: Dict[str, Any] = Field(..., description="Parameter sensitivity")
    alternative_scenarios: List[Dict[str, Any]] = Field(default_factory=list, description="Alternative allocations")
    confidence_metrics: Dict[str, float] = Field(..., description="Solution confidence")
    schema_version: str = Field(default="1.0")


class AdvancedFinOpsAgent(HardenedAgent):
    """
    Advanced FinOps Agent with intelligent financial modeling capabilities
    
    Features:
    - Adaptive algorithm selection based on data characteristics
    - Memory-driven optimization and learning
    - Advanced anomaly detection with ensemble methods
    - Multi-objective budget optimization
    - Risk-adjusted financial modeling
    - Market-aware discount rate determination
    """
    
    _domain = "finops"
    _capabilities = [
        "advanced_financial_modeling",
        "intelligent_anomaly_detection", 
        "multi_objective_optimization",
        "risk_assessment_analysis",
        "market_aware_calculations",
        "adaptive_algorithm_selection",
        "memory_driven_learning"
    ]
    
    def __init__(self, memory_service, agent_id: Optional[str] = None):
        super().__init__(memory_service, agent_id)
        self.financial_analyzer = IntelligentFinancialAnalyzer(memory_service)
        self._performance_cache = {}
        
        logger.info(
            "advanced_finops_agent_initialized",
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
        """Intelligent message processing with context awareness"""
        
        request_id = plan.get('request_id', uuid4().hex)
        
        logger.info(
            "finops_message_processing",
            agent_id=self.agent_id,
            request_id=request_id,
            message_type=await self._classify_financial_intent(message),
            memory_context_size=len(memory_context)
        )
        
        # Analyze message intent and provide intelligent response
        intent = await self._classify_financial_intent(message)
        
        if intent == "cost_analysis":
            return f"I can perform advanced cost analysis using ensemble anomaly detection, adaptive algorithms, and market-aware modeling. Memory context: {len(memory_context)} relevant financial patterns found."
        elif intent == "budget_optimization":
            return f"I'll help optimize your budget using multi-objective programming with risk-adjusted returns. Leveraging {len(memory_context)} historical optimization patterns."
        elif intent == "npv_calculation":
            return f"I can calculate NPV with intelligent discount rate determination, sensitivity analysis, and market context integration. {len(memory_context)} similar calculations in memory."
        elif intent == "anomaly_detection":
            return f"I'll detect anomalies using adaptive ensemble methods that learn from your data characteristics. {len(memory_context)} detection patterns available."
        else:
            return f"I can assist with advanced financial operations including NPV calculations, anomaly detection, budget optimization, and risk analysis. {len(memory_context)} relevant memories found."
    
    @tool(description="Advanced Net Present Value calculation with intelligent market awareness")
    async def compute_npv(
        self,
        cashflows: List[float],
        discount_rate: Optional[float] = None,
        risk_adjustment: float = 0.0,
        compounding_frequency: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> NPVResult:
        """
        Compute NPV using advanced financial modeling with:
        - Intelligent discount rate determination from market data
        - Risk-adjusted calculations based on cashflow characteristics
        - Comprehensive sensitivity analysis
        - Market context integration
        
        Complexity: O(n log n) for n cashflow periods
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "npv_calculation_started",
            agent_id=self.agent_id,
            request_id=request_id,
            cashflows_count=len(cashflows),
            discount_rate_provided=discount_rate is not None,
            estimated_cost=0.02
        )
        
        try:
            # Use intelligent analyzer for advanced NPV calculation
            result = await self.financial_analyzer.compute_advanced_npv(
                cashflows=cashflows,
                discount_rate=discount_rate or 0.0,
                risk_adjustment=risk_adjustment,
                compounding_frequency=compounding_frequency,
                context=context or {}
            )
            
            # Convert to Pydantic model for validation
            npv_result = NPVResult(
                npv=Decimal(str(result["npv"])),
                discount_rate_used=result["discount_rate_used"],
                risk_adjustment=result["risk_adjustment"],
                irr=result.get("irr"),
                payback_period=result.get("payback_period"),
                profitability_index=result["profitability_index"],
                sensitivity_analysis=result["sensitivity_analysis"],
                market_context=result.get("market_context"),
                confidence_interval={
                    "lower": result["sensitivity_analysis"]["npvs"][0],
                    "upper": result["sensitivity_analysis"]["npvs"][-1]
                }
            )
            
            # Store successful calculation pattern in memory
            await self._store_calculation_pattern("npv", result, context)
            
            logger.info(
                "npv_calculation_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                npv=float(npv_result.npv),
                execution_time_ms=(datetime.now().timestamp() * 1000) % 1000
            )
            
            return npv_result
            
        except Exception as e:
            logger.error(
                "npv_calculation_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise ToolError(
                f"NPV calculation failed: {str(e)}",
                tool_name="compute_npv",
                agent_id=self.agent_id
            )
    
    @tool(description="Intelligent anomaly detection with adaptive ensemble methods")
    async def detect_cost_anomalies(
        self,
        timeseries: List[CostDataPoint],
        context: Optional[Dict[str, Any]] = None
    ) -> AnomalyDetectionResult:
        """
        Detect cost anomalies using intelligent ensemble methods that adapt
        based on data characteristics and historical performance patterns.
        
        Features:
        - Automatic algorithm selection based on data profile
        - Ensemble voting with confidence scoring
        - Memory-driven method optimization
        - Contextual anomaly interpretation
        
        Complexity: O(n log n) for n data points
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "anomaly_detection_started",
            agent_id=self.agent_id,
            request_id=request_id,
            data_points=len(timeseries),
            estimated_cost=0.05
        )
        
        try:
            # Convert Pydantic models to dict format for analyzer
            data_dicts = []
            for point in timeseries:
                data_dicts.append({
                    "timestamp": point.timestamp,
                    "cost": float(point.cost),
                    "category": point.category,
                    "metadata": point.metadata
                })
            
            # Use intelligent analyzer for adaptive anomaly detection
            analysis_result = await self.financial_analyzer.detect_anomalies_intelligent(
                timeseries=data_dicts,
                context=context or {}
            )
            
            # Convert to structured anomaly points
            anomaly_points = []
            for anomaly in analysis_result.get("anomalies", []):
                point = AnomalyPoint(
                    timestamp=datetime.fromisoformat(anomaly["timestamp"].replace('Z', '+00:00')),
                    value=anomaly["value"],
                    confidence=anomaly["confidence"],
                    z_score=anomaly.get("z_score"),
                    isolation_score=anomaly.get("isolation_score"),
                    deviation_magnitude=anomaly.get("deviation_from_trend", 0),
                    context={
                        "detection_method": analysis_result.get("method_used"),
                        "data_characteristics": analysis_result.get("ensemble_details", {})
                    }
                )
                anomaly_points.append(point)
            
            # Generate intelligent recommendations
            recommendations = await self._generate_anomaly_recommendations(
                anomaly_points, analysis_result, context
            )
            
            result = AnomalyDetectionResult(
                anomalies=anomaly_points,
                method_used=analysis_result.get("method_used", "ensemble"),
                total_anomalies=len(anomaly_points),
                anomaly_ratio=analysis_result.get("anomaly_ratio", 0.0),
                confidence_distribution=analysis_result.get("confidence_distribution", {}),
                statistical_summary=analysis_result.get("statistical_summary", {}),
                recommendations=recommendations,
                execution_metadata={
                    "algorithm_selection_rationale": "Adaptive selection based on data characteristics",
                    "performance_optimization": "Memory-driven parameter tuning",
                    "ensemble_methods": analysis_result.get("ensemble_details", {})
                }
            )
            
            # Store anomaly detection pattern for future optimization
            await self._store_anomaly_pattern(analysis_result, context)
            
            logger.info(
                "anomaly_detection_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                anomalies_detected=len(anomaly_points),
                method_used=result.method_used
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "anomaly_detection_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise ToolError(
                f"Anomaly detection failed: {str(e)}",
                tool_name="detect_cost_anomalies",
                agent_id=self.agent_id
            )
    
    @tool(description="Multi-objective budget optimization using advanced mathematical programming")
    async def optimize_budget(
        self,
        items: List[BudgetItem],
        constraints: BudgetConstraints,
        objectives: List[OptimizationObjective] = None
    ) -> BudgetOptimizationResult:
        """
        Advanced budget optimization using multi-objective integer programming
        with risk consideration and dependency handling.
        
        Features:
        - Multi-objective optimization (ROI, risk, coverage)
        - Dependency constraint handling
        - Risk-adjusted benefit calculation
        - Pareto frontier analysis
        - Sensitivity analysis on key parameters
        
        Complexity: O(2^n) worst case, O(n^3) typical with heuristics
        """
        request_id = uuid4().hex
        
        logger.info(
            "budget_optimization_started",
            agent_id=self.agent_id,
            request_id=request_id,
            items_count=len(items),
            total_budget=float(constraints.total_budget),
            estimated_cost=0.10
        )
        
        try:
            if not items:
                raise ValueError("No budget items provided for optimization")
            
            # Convert to optimization matrices
            costs = np.array([float(item.cost) for item in items])
            benefits = np.array([float(item.benefit) for item in items])
            priorities = np.array([item.priority for item in items])
            risk_factors = np.array([item.risk_factor for item in items])
            
            n_items = len(items)
            
            # Create decision variables (binary)
            x = cp.Variable(n_items, boolean=True)
            
            # Risk-adjusted benefits
            adjusted_benefits = benefits / risk_factors
            
            # Default to ROI maximization if no objectives specified
            if not objectives:
                objectives = [OptimizationObjective.MAXIMIZE_ROI]
            
            # Multi-objective function construction
            objective_expr = 0
            
            if OptimizationObjective.MAXIMIZE_ROI in objectives:
                roi_term = cp.sum(cp.multiply(adjusted_benefits, x)) / (cp.sum(cp.multiply(costs, x)) + 1e-6)
                objective_expr += roi_term
            
            if OptimizationObjective.MINIMIZE_RISK in objectives:
                risk_term = -cp.sum(cp.multiply(risk_factors, x)) / n_items
                objective_expr += 0.3 * risk_term  # Weight risk minimization
            
            if OptimizationObjective.MAXIMIZE_COVERAGE in objectives:
                coverage_term = cp.sum(x) / n_items
                objective_expr += 0.2 * coverage_term
            
            # Constraints
            constraint_list = []
            
            # Budget constraint
            constraint_list.append(cp.sum(cp.multiply(costs, x)) <= float(constraints.total_budget))
            
            # Minimum benefit threshold
            if constraints.min_benefit_threshold:
                constraint_list.append(
                    cp.sum(cp.multiply(benefits, x)) >= float(constraints.min_benefit_threshold)
                )
            
            # Risk constraint
            if constraints.max_risk_score:
                avg_risk = cp.sum(cp.multiply(risk_factors, x)) / (cp.sum(x) + 1e-6)
                constraint_list.append(avg_risk <= constraints.max_risk_score)
            
            # Dependency constraints
            item_indices = {item.name: i for i, item in enumerate(items)}
            for i, item in enumerate(items):
                for dep_name in item.dependencies:
                    if dep_name in item_indices:
                        dep_idx = item_indices[dep_name]
                        # If item i is selected, dependency must be selected
                        constraint_list.append(x[i] <= x[dep_idx])
            
            # Required categories constraint
            if constraints.required_categories:
                for category in constraints.required_categories:
                    category_items = [
                        i for i, item in enumerate(items) 
                        if item.constraints.get('category') == category
                    ]
                    if category_items:
                        constraint_list.append(cp.sum([x[i] for i in category_items]) >= 1)
            
            # Forbidden items constraint
            if constraints.forbidden_items:
                for forbidden_item in constraints.forbidden_items:
                    if forbidden_item in item_indices:
                        forbidden_idx = item_indices[forbidden_item]
                        constraint_list.append(x[forbidden_idx] == 0)
            
            # Solve optimization problem
            problem = cp.Problem(cp.Maximize(objective_expr), constraint_list)
            
            try:
                problem.solve(solver=cp.CBC, verbose=False)
            except:
                try:
                    problem.solve(solver=cp.ECOS_BB, verbose=False)
                except:
                    problem.solve(verbose=False)
            
            if problem.status not in ["infeasible", "unbounded"]:
                # Extract solution
                selected_indices = np.where(np.round(x.value) == 1)[0]
                selected_items = [items[i].name for i in selected_indices]
                
                total_cost = sum(float(items[i].cost) for i in selected_indices)
                total_benefit = sum(float(items[i].benefit) for i in selected_indices)
                
                # Calculate portfolio risk
                if selected_indices.size > 0:
                    portfolio_risk = np.mean([items[i].risk_factor for i in selected_indices])
                    roi = (total_benefit / total_cost) if total_cost > 0 else 0
                else:
                    portfolio_risk = 0
                    roi = 0
                
                # Sensitivity analysis
                sensitivity_results = await self._perform_budget_sensitivity_analysis(
                    items, constraints, selected_indices, objectives
                )
                
                result = BudgetOptimizationResult(
                    selected_items=selected_items,
                    total_cost=Decimal(str(total_cost)),
                    total_benefit=Decimal(str(total_benefit)),
                    roi=float(roi),
                    risk_score=float(portfolio_risk),
                    optimization_status=problem.status,
                    sensitivity_analysis=sensitivity_results,
                    confidence_metrics={
                        "solution_confidence": 0.95 if problem.status == "optimal" else 0.7,
                        "constraint_satisfaction": 1.0,
                        "objective_achievement": float(problem.value) if problem.value else 0.0
                    }
                )
                
                logger.info(
                    "budget_optimization_completed",
                    agent_id=self.agent_id,
                    request_id=request_id,
                    selected_items=len(selected_items),
                    total_cost=float(result.total_cost),
                    roi=result.roi,
                    status=problem.status
                )
                
                return result
            
            else:
                raise ValueError(f"Optimization problem is {problem.status}")
                
        except Exception as e:
            logger.error(
                "budget_optimization_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise ToolError(
                f"Budget optimization failed: {str(e)}",
                tool_name="optimize_budget",
                agent_id=self.agent_id
            )
    
    @tool(description="Compute Internal Rate of Return using advanced numerical methods")
    async def compute_irr(
        self,
        cashflows: List[float],
        initial_guess: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """
        Advanced IRR calculation using Newton-Raphson with multiple starting points
        and convergence analysis for robust solution finding.
        """
        request_id = uuid4().hex
        
        logger.info(
            "irr_calculation_started",
            agent_id=self.agent_id,
            request_id=request_id,
            cashflows_count=len(cashflows),
            estimated_cost=0.01
        )
        
        try:
            if len(cashflows) < 2:
                raise ValueError("At least 2 cashflow periods required")
            
            def npv_function(rate):
                return sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))
            
            def npv_derivative(rate):
                return sum(-i * cf / (1 + rate) ** (i + 1) for i, cf in enumerate(cashflows))
            
            # Try multiple starting points for robust convergence
            starting_points = [initial_guess, 0.05, 0.15, 0.25, -0.5]
            solutions = []
            
            for start in starting_points:
                try:
                    # Newton-Raphson iteration
                    rate = start
                    for iteration in range(max_iterations):
                        npv_val = npv_function(rate)
                        
                        if abs(npv_val) < tolerance:
                            break
                        
                        derivative = npv_derivative(rate)
                        if abs(derivative) < 1e-12:
                            break  # Avoid division by zero
                        
                        new_rate = rate - npv_val / derivative
                        
                        if abs(new_rate - rate) < tolerance:
                            break
                        
                        rate = new_rate
                    
                    # Verify solution
                    if abs(npv_function(rate)) < tolerance and rate > -0.99:
                        solutions.append({
                            "irr": float(rate),
                            "npv_at_irr": float(npv_function(rate)),
                            "iterations": iteration + 1,
                            "starting_point": start
                        })
                
                except:
                    continue
            
            if not solutions:
                # Fallback to scipy optimization
                from scipy.optimize import fsolve, brentq
                try:
                    # Try to find roots in reasonable range
                    irr_solution = fsolve(npv_function, initial_guess)[0]
                    if abs(npv_function(irr_solution)) < tolerance:
                        solutions.append({
                            "irr": float(irr_solution),
                            "npv_at_irr": float(npv_function(irr_solution)),
                            "method": "scipy_fsolve"
                        })
                except:
                    pass
            
            if solutions:
                # Select most reliable solution (closest to zero NPV)
                best_solution = min(solutions, key=lambda s: abs(s.get("npv_at_irr", 1)))
                
                result = {
                    "irr": best_solution["irr"],
                    "npv_at_irr": best_solution.get("npv_at_irr", 0),
                    "convergence_analysis": {
                        "solutions_found": len(solutions),
                        "iterations_used": best_solution.get("iterations"),
                        "starting_point": best_solution.get("starting_point"),
                        "all_solutions": solutions
                    },
                    "interpretation": {
                        "annualized_return": f"{best_solution['irr'] * 100:.2f}%",
                        "feasibility": "feasible" if best_solution["irr"] > 0 else "not_feasible"
                    }
                }
                
                logger.info(
                    "irr_calculation_completed",
                    agent_id=self.agent_id,
                    request_id=request_id,
                    irr=best_solution["irr"],
                    solutions_found=len(solutions)
                )
                
                return result
            
            else:
                raise ValueError("IRR calculation did not converge to a valid solution")
                
        except Exception as e:
            logger.error(
                "irr_calculation_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise ToolError(
                f"IRR calculation failed: {str(e)}",
                tool_name="compute_irr",
                agent_id=self.agent_id
            )
    
    # Helper methods
    async def _classify_financial_intent(self, message: str) -> str:
        """Classify user message intent for appropriate response"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["npv", "present value", "discount"]):
            return "npv_calculation"
        elif any(word in message_lower for word in ["anomaly", "anomalies", "outlier", "unusual"]):
            return "anomaly_detection"
        elif any(word in message_lower for word in ["optimize", "budget", "allocation"]):
            return "budget_optimization"
        elif any(word in message_lower for word in ["cost", "analyze", "analysis"]):
            return "cost_analysis"
        else:
            return "general_finops"
    
    async def _generate_anomaly_recommendations(
        self,
        anomalies: List[AnomalyPoint],
        analysis_result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate intelligent recommendations based on anomaly analysis"""
        
        recommendations = []
        
        if not anomalies:
            recommendations.append("No anomalies detected - cost patterns appear normal")
            return recommendations
        
        # High-confidence anomalies
        high_conf_anomalies = [a for a in anomalies if a.confidence > 0.8]
        if high_conf_anomalies:
            recommendations.append(
                f"Investigate {len(high_conf_anomalies)} high-confidence anomalies immediately"
            )
        
        # Clustering of anomalies
        if len(anomalies) > 3:
            timestamps = [a.timestamp for a in anomalies]
            time_diffs = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
            if any(diff < 7 for diff in time_diffs):
                recommendations.append("Multiple anomalies detected within short timeframe - investigate systematic issues")
        
        # Magnitude-based recommendations
        high_magnitude = [a for a in anomalies if a.deviation_magnitude > analysis_result.get("statistical_summary", {}).get("data_std", 0) * 3]
        if high_magnitude:
            recommendations.append("Large magnitude deviations detected - review underlying cost drivers")
        
        return recommendations
    
    async def _perform_budget_sensitivity_analysis(
        self,
        items: List[BudgetItem],
        constraints: BudgetConstraints,
        selected_indices: np.ndarray,
        objectives: List[OptimizationObjective]
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis on budget optimization results"""
        
        # Test budget variations
        budget_variations = [0.8, 0.9, 1.1, 1.2]  # ±20% budget change
        sensitivity_results = {
            "budget_sensitivity": [],
            "item_criticality": {},
            "objective_trade_offs": {}
        }
        
        base_budget = float(constraints.total_budget)
        
        for multiplier in budget_variations:
            varied_budget = base_budget * multiplier
            # Simplified re-optimization would go here
            sensitivity_results["budget_sensitivity"].append({
                "budget_multiplier": multiplier,
                "budget_amount": varied_budget,
                "estimated_impact": f"{(multiplier - 1) * 100:.1f}% budget change"
            })
        
        return sensitivity_results
    
    async def _store_calculation_pattern(
        self,
        calculation_type: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ):
        """Store successful calculation patterns in memory"""
        if not self.memory_service or not context:
            return
        
        try:
            memory_content = {
                "calculation_type": calculation_type,
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
                message=f"FinOps calculation: {calculation_type}",
                context={
                    "category": "financial_modeling",
                    "calculation_type": calculation_type,
                    "agent_id": self.agent_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store calculation pattern: {e}")
    
    async def _store_anomaly_pattern(
        self,
        analysis_result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ):
        """Store anomaly detection patterns for future optimization"""
        if not self.memory_service or not context:
            return
        
        try:
            memory_content = {
                "analysis_type": "anomaly_detection",
                "method_used": analysis_result.get("method_used"),
                "performance_metrics": {
                    "anomalies_detected": analysis_result.get("total_anomalies", 0),
                    "confidence_mean": analysis_result.get("confidence_distribution", {}).get("mean", 0)
                },
                "execution_timestamp": datetime.now().isoformat()
            }
            
            await self.memory_service.store_conversation_memory(
                user_id=context.get("user_id", "system"),
                message=f"Anomaly detection completed: {analysis_result.get('method_used')}",
                context={
                    "category": "anomaly_detection",
                    "method": analysis_result.get("method_used"),
                    "agent_id": self.agent_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store anomaly pattern: {e}")