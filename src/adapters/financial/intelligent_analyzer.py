"""
Intelligent Financial Analysis Engine
Self-adapting financial modeling with memory-driven optimization
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from dataclasses import dataclass
from enum import Enum
import json

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats, optimize
from scipy.stats import jarque_bera, shapiro, anderson
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
import cvxpy as cp
import yfinance as yf
from fredapi import Fred
import structlog

logger = structlog.get_logger(__name__)


class DataCharacteristics(Enum):
    """Data profile characteristics for adaptive algorithm selection"""
    TRENDING = "trending"
    SEASONAL = "seasonal" 
    VOLATILE = "volatile"
    STABLE = "stable"
    SPARSE = "sparse"
    DENSE = "dense"
    ANOMALOUS = "anomalous"
    NORMAL = "normal"


@dataclass
class FinancialDataProfile:
    """Comprehensive analysis of financial data characteristics"""
    size: int
    volatility: float
    trend_strength: float
    seasonality_strength: float
    stationarity_pvalue: float
    normality_pvalue: float
    outlier_ratio: float
    missing_ratio: float
    characteristics: List[DataCharacteristics]
    recommended_models: List[str]
    confidence_interval: Tuple[float, float]


@dataclass
class ModelPerformanceHistory:
    """Historical performance tracking for model selection"""
    model_name: str
    accuracy_scores: List[float]
    execution_times: List[float]
    user_satisfaction_scores: List[float]
    data_profile_matches: List[FinancialDataProfile]
    last_updated: datetime


class IntelligentFinancialAnalyzer:
    """
    Adaptive financial analysis engine that learns from historical patterns
    and user feedback to optimize model selection and parameters
    """
    
    def __init__(self, memory_service=None):
        self.memory_service = memory_service
        self.model_performance_cache: Dict[str, ModelPerformanceHistory] = {}
        self.risk_free_rate_cache: Optional[float] = None
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        
        # Initialize economic data API (FRED)
        try:
            self.fred = Fred(api_key=None)  # Will use environment variable
        except:
            self.fred = None
            logger.warning("FRED API not available, using fallback rates")
    
    async def analyze_data_characteristics(
        self, 
        data: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]] = None
    ) -> FinancialDataProfile:
        """
        Intelligent analysis of financial data characteristics to inform
        optimal algorithm selection and parameter tuning
        """
        try:
            # Convert to pandas for analysis
            df = pd.DataFrame(data)
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').set_index('timestamp')
            
            # Extract numeric columns for analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for analysis")
            
            primary_col = numeric_cols[0]  # Use first numeric column as primary
            values = df[primary_col].dropna()
            
            if len(values) < 10:
                raise ValueError("Insufficient data points for meaningful analysis")
            
            # Comprehensive statistical analysis
            profile = await self._compute_data_profile(values, df, context)
            
            # Store analysis results in memory for future optimization
            if self.memory_service and context:
                await self._store_data_profile(profile, context)
            
            return profile
            
        except Exception as e:
            logger.error(
                "data_characteristics_analysis_failed",
                error=str(e),
                data_size=len(data)
            )
            raise
    
    async def _compute_data_profile(
        self, 
        values: pd.Series, 
        full_df: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> FinancialDataProfile:
        """Compute comprehensive data profile for intelligent model selection"""
        
        # Basic statistics
        size = len(values)
        volatility = float(values.std() / values.mean()) if values.mean() != 0 else 0.0
        missing_ratio = float(values.isna().sum() / len(values))
        
        # Trend analysis using linear regression
        x = np.arange(len(values))
        trend_slope, _, r_value, p_value, _ = stats.linregress(x, values)
        trend_strength = float(abs(r_value))
        
        # Seasonality detection (if time series)
        seasonality_strength = 0.0
        if len(values) >= 24:  # Minimum for seasonal analysis
            try:
                decomposition = seasonal_decompose(values, model='additive', period=12)
                seasonal_var = decomposition.seasonal.var()
                total_var = values.var()
                seasonality_strength = float(seasonal_var / total_var) if total_var > 0 else 0.0
            except:
                pass
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(values)
            stationarity_pvalue = float(adf_result[1])
        except:
            stationarity_pvalue = 1.0
        
        # Normality tests
        if len(values) >= 8:
            _, normality_pvalue = jarque_bera(values)
            normality_pvalue = float(normality_pvalue)
        else:
            normality_pvalue = 1.0
        
        # Outlier detection using IQR method
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = len(values[(values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)])
        outlier_ratio = float(outlier_count / len(values))
        
        # Characterize data
        characteristics = []
        
        if trend_strength > 0.3:
            characteristics.append(DataCharacteristics.TRENDING)
        if seasonality_strength > 0.1:
            characteristics.append(DataCharacteristics.SEASONAL)
        if volatility > 0.2:
            characteristics.append(DataCharacteristics.VOLATILE)
        else:
            characteristics.append(DataCharacteristics.STABLE)
        if size < 50:
            characteristics.append(DataCharacteristics.SPARSE)
        else:
            characteristics.append(DataCharacteristics.DENSE)
        if outlier_ratio > 0.05:
            characteristics.append(DataCharacteristics.ANOMALOUS)
        else:
            characteristics.append(DataCharacteristics.NORMAL)
        
        # Model recommendations based on characteristics
        recommended_models = await self._recommend_models(characteristics, size)
        
        # Confidence interval estimation
        confidence_interval = (
            float(values.mean() - 1.96 * values.std() / np.sqrt(len(values))),
            float(values.mean() + 1.96 * values.std() / np.sqrt(len(values)))
        )
        
        return FinancialDataProfile(
            size=size,
            volatility=volatility,
            trend_strength=trend_strength,
            seasonality_strength=seasonality_strength,
            stationarity_pvalue=stationarity_pvalue,
            normality_pvalue=normality_pvalue,
            outlier_ratio=outlier_ratio,
            missing_ratio=missing_ratio,
            characteristics=characteristics,
            recommended_models=recommended_models,
            confidence_interval=confidence_interval
        )
    
    async def _recommend_models(
        self, 
        characteristics: List[DataCharacteristics], 
        data_size: int
    ) -> List[str]:
        """Intelligent model recommendation based on data characteristics"""
        
        recommendations = []
        
        # Query memory for similar data profiles and their successful models
        if self.memory_service:
            try:
                similar_profiles = await self.memory_service.retrieve_relevant_memories(
                    query=f"financial modeling {' '.join([c.value for c in characteristics])}",
                    limit=5
                )
                
                # Extract successful models from memory
                for memory in similar_profiles:
                    if hasattr(memory, 'content'):
                        try:
                            memory_data = json.loads(memory.content)
                            if 'successful_models' in memory_data:
                                recommendations.extend(memory_data['successful_models'])
                        except:
                            pass
            except:
                pass
        
        # Fallback rule-based recommendations if no memory available
        if not recommendations:
            if DataCharacteristics.SEASONAL in characteristics:
                recommendations.extend(['holt_winters', 'sarima', 'seasonal_decompose'])
            
            if DataCharacteristics.TRENDING in characteristics:
                recommendations.extend(['linear_regression', 'polynomial_regression', 'arima'])
            
            if DataCharacteristics.VOLATILE in characteristics:
                recommendations.extend(['garch', 'ewma', 'robust_regression'])
            
            if DataCharacteristics.ANOMALOUS in characteristics:
                recommendations.extend(['isolation_forest', 'local_outlier_factor', 'robust_scaler'])
            
            if data_size < 30:
                recommendations.extend(['bootstrap', 'bayesian_methods'])
            elif data_size > 1000:
                recommendations.extend(['neural_networks', 'random_forest', 'gradient_boosting'])
        
        return list(set(recommendations))  # Remove duplicates
    
    async def detect_anomalies_intelligent(
        self,
        timeseries: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Intelligent anomaly detection with adaptive algorithm selection
        based on data characteristics and historical performance
        """
        try:
            # Analyze data characteristics
            data_profile = await self.analyze_data_characteristics(timeseries, context)
            
            # Convert to working format
            df = pd.DataFrame(timeseries)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')
            
            # Select optimal anomaly detection method based on profile
            method = await self._select_anomaly_method(data_profile, context)
            
            logger.info(
                "anomaly_detection_started",
                method=method,
                data_size=data_profile.size,
                characteristics=[c.value for c in data_profile.characteristics]
            )
            
            # Execute selected method
            if method == "ensemble":
                result = await self._ensemble_anomaly_detection(df, data_profile)
            elif method == "isolation_forest":
                result = await self._isolation_forest_detection(df, data_profile)
            elif method == "statistical":
                result = await self._statistical_anomaly_detection(df, data_profile)
            elif method == "time_series":
                result = await self._time_series_anomaly_detection(df, data_profile)
            else:
                # Adaptive ensemble as fallback
                result = await self._ensemble_anomaly_detection(df, data_profile)
            
            # Store performance metrics for future optimization
            if self.memory_service and context:
                await self._store_anomaly_performance(method, result, context, data_profile)
            
            return result
            
        except Exception as e:
            logger.error(
                "intelligent_anomaly_detection_failed",
                error=str(e),
                data_size=len(timeseries)
            )
            raise
    
    async def _select_anomaly_method(
        self, 
        profile: FinancialDataProfile, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Intelligently select anomaly detection method based on data and history"""
        
        # Query memory for performance of different methods on similar data
        best_method = "ensemble"  # Default
        
        if self.memory_service and context:
            try:
                method_memories = await self.memory_service.retrieve_relevant_memories(
                    query=f"anomaly detection performance {' '.join([c.value for c in profile.characteristics])}",
                    user_id=context.get("user_id"),
                    limit=10
                )
                
                method_scores = {}
                for memory in method_memories:
                    try:
                        memory_data = json.loads(memory.content)
                        if 'method' in memory_data and 'accuracy_score' in memory_data:
                            method = memory_data['method']
                            score = memory_data['accuracy_score']
                            if method not in method_scores:
                                method_scores[method] = []
                            method_scores[method].append(score)
                    except:
                        pass
                
                # Select method with best average performance
                if method_scores:
                    method_averages = {
                        method: np.mean(scores) 
                        for method, scores in method_scores.items()
                    }
                    best_method = max(method_averages, key=method_averages.get)
            except:
                pass
        
        # Fallback selection based on data characteristics
        if best_method == "ensemble":
            if DataCharacteristics.SEASONAL in profile.characteristics:
                best_method = "time_series"
            elif DataCharacteristics.VOLATILE in profile.characteristics:
                best_method = "isolation_forest"
            elif profile.size < 50:
                best_method = "statistical"
        
        return best_method
    
    async def _ensemble_anomaly_detection(
        self, 
        df: pd.DataFrame, 
        profile: FinancialDataProfile
    ) -> Dict[str, Any]:
        """Advanced ensemble anomaly detection combining multiple methods"""
        
        values = df.iloc[:, 0].values  # First numeric column
        
        # Method 1: Isolation Forest
        iso_forest = IsolationForest(
            contamination='auto',
            random_state=42,
            n_estimators=100
        )
        iso_scores = iso_forest.decision_function(values.reshape(-1, 1))
        iso_predictions = iso_forest.predict(values.reshape(-1, 1))
        
        # Method 2: Statistical (Z-score with robust estimates)
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        modified_z_scores = 0.6745 * (values - median) / mad
        stat_predictions = np.where(np.abs(modified_z_scores) > 3.5, -1, 1)
        
        # Method 3: Time series decomposition (if sufficient data)
        ts_predictions = np.ones_like(values)
        if len(values) >= 24:
            try:
                decomposition = seasonal_decompose(
                    pd.Series(values, index=df.index), 
                    model='additive', 
                    period=min(12, len(values) // 2)
                )
                residuals = decomposition.resid.dropna()
                residual_threshold = 2 * residuals.std()
                ts_anomalies = np.abs(residuals) > residual_threshold
                ts_predictions[ts_anomalies.index] = -1
            except:
                pass
        
        # Ensemble voting
        ensemble_votes = iso_predictions + stat_predictions + ts_predictions
        final_predictions = np.where(ensemble_votes <= 0, -1, 1)
        
        # Confidence scoring
        confidence_scores = []
        for i in range(len(values)):
            vote_agreement = abs(ensemble_votes[i]) / 3.0  # Normalized agreement
            confidence_scores.append(vote_agreement)
        
        # Extract anomaly points
        anomaly_indices = np.where(final_predictions == -1)[0]
        anomalies = []
        
        for idx in anomaly_indices:
            anomalies.append({
                "timestamp": df.index[idx].isoformat(),
                "value": float(values[idx]),
                "confidence": float(confidence_scores[idx]),
                "z_score": float(modified_z_scores[idx]),
                "isolation_score": float(iso_scores[idx]),
                "deviation_from_trend": float(abs(values[idx] - np.mean(values)))
            })
        
        return {
            "anomalies": anomalies,
            "method_used": "ensemble_voting",
            "total_anomalies": len(anomalies),
            "anomaly_ratio": len(anomalies) / len(values),
            "confidence_distribution": {
                "mean": float(np.mean(confidence_scores)),
                "std": float(np.std(confidence_scores)),
                "min": float(np.min(confidence_scores)),
                "max": float(np.max(confidence_scores))
            },
            "statistical_summary": {
                "data_mean": float(np.mean(values)),
                "data_std": float(np.std(values)),
                "data_median": float(np.median(values)),
                "data_mad": float(mad)
            },
            "ensemble_details": {
                "isolation_forest_anomalies": int(np.sum(iso_predictions == -1)),
                "statistical_anomalies": int(np.sum(stat_predictions == -1)),
                "time_series_anomalies": int(np.sum(ts_predictions == -1))
            }
        }
    
    async def _isolation_forest_detection(
        self, 
        df: pd.DataFrame, 
        profile: FinancialDataProfile
    ) -> Dict[str, Any]:
        """Advanced isolation forest with parameter optimization"""
        
        values = df.iloc[:, 0].values.reshape(-1, 1)
        
        # Optimize contamination parameter based on data characteristics
        contamination = min(0.1, profile.outlier_ratio + 0.02)  # Adaptive contamination
        
        # Create and fit model
        model = IsolationForest(
            contamination=contamination,
            n_estimators=200,  # More trees for better accuracy
            max_samples='auto',
            random_state=42,
            n_jobs=-1
        )
        
        predictions = model.fit_predict(values)
        scores = model.decision_function(values)
        
        # Extract anomalies
        anomaly_indices = np.where(predictions == -1)[0]
        anomalies = []
        
        for idx in anomaly_indices:
            anomalies.append({
                "timestamp": df.index[idx].isoformat(),
                "value": float(values[idx, 0]),
                "isolation_score": float(scores[idx]),
                "confidence": float(1 / (1 + np.exp(scores[idx])))  # Sigmoid transformation
            })
        
        return {
            "anomalies": anomalies,
            "method_used": "isolation_forest_optimized",
            "total_anomalies": len(anomalies),
            "parameters_used": {
                "contamination": contamination,
                "n_estimators": 200
            }
        }
    
    async def compute_advanced_npv(
        self,
        cashflows: List[float],
        discount_rate: float,
        risk_adjustment: float = 0.0,
        compounding_frequency: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advanced NPV calculation with risk adjustment, sensitivity analysis,
        and intelligent parameter optimization based on market conditions
        """
        try:
            if len(cashflows) < 2:
                raise ValueError("At least 2 cashflow periods required")
            
            # Get current risk-free rate if not provided
            if discount_rate <= 0:
                discount_rate = await self._get_intelligent_discount_rate(context)
            
            # Apply risk adjustment intelligently based on cashflow volatility
            cf_volatility = np.std(cashflows) / np.mean([abs(cf) for cf in cashflows if cf != 0])
            intelligent_risk_adjustment = risk_adjustment + (cf_volatility * 0.01)  # Dynamic risk adjustment
            
            adjusted_rate = discount_rate + intelligent_risk_adjustment
            
            # Compute NPV with different compounding frequencies
            npv_results = {}
            for freq in [1, 2, 4, 12]:  # Annual, semi-annual, quarterly, monthly
                periods = np.arange(len(cashflows))
                discount_factors = (1 + adjusted_rate / freq) ** (-freq * periods)
                npv = np.sum(np.array(cashflows) * discount_factors)
                npv_results[f"frequency_{freq}"] = float(npv)
            
            # Use requested frequency or optimal one
            if compounding_frequency in npv_results:
                final_npv = npv_results[f"frequency_{compounding_frequency}"]
            else:
                final_npv = npv_results["frequency_1"]  # Default to annual
            
            # Sensitivity analysis
            sensitivity_rates = np.linspace(
                max(0.001, discount_rate - 0.05), 
                discount_rate + 0.05, 
                11
            )
            
            sensitivity_npvs = []
            for rate in sensitivity_rates:
                periods = np.arange(len(cashflows))
                discount_factors = (1 + rate + intelligent_risk_adjustment) ** (-periods)
                sens_npv = np.sum(np.array(cashflows) * discount_factors)
                sensitivity_npvs.append(float(sens_npv))
            
            # IRR calculation for comparison
            try:
                irr = float(np.irr(cashflows)) if hasattr(np, 'irr') else None
                if irr is None:
                    # Custom IRR calculation using optimization
                    def npv_func(rate):
                        periods = np.arange(len(cashflows))
                        return np.sum(np.array(cashflows) / (1 + rate) ** periods)
                    
                    irr_result = optimize.fsolve(npv_func, 0.1)[0]
                    irr = float(irr_result) if abs(npv_func(irr_result)) < 0.01 else None
            except:
                irr = None
            
            # Payback period calculation
            cumulative_cf = np.cumsum(cashflows)
            payback_period = None
            for i, cum_cf in enumerate(cumulative_cf):
                if cum_cf > 0:
                    payback_period = float(i)
                    break
            
            # Profitability index
            initial_investment = abs(cashflows[0]) if cashflows[0] < 0 else 1
            pv_future_flows = final_npv + initial_investment
            profitability_index = pv_future_flows / initial_investment if initial_investment > 0 else 0
            
            result = {
                "npv": final_npv,
                "discount_rate_used": discount_rate,
                "risk_adjustment": intelligent_risk_adjustment,
                "compounding_frequency": compounding_frequency,
                "irr": irr,
                "payback_period": payback_period,
                "profitability_index": float(profitability_index),
                "sensitivity_analysis": {
                    "rates": [float(r) for r in sensitivity_rates],
                    "npvs": sensitivity_npvs,
                    "elasticity": float(np.std(sensitivity_npvs) / abs(final_npv)) if final_npv != 0 else 0
                },
                "compounding_comparison": npv_results,
                "cashflow_analysis": {
                    "total_inflows": float(sum(cf for cf in cashflows if cf > 0)),
                    "total_outflows": float(abs(sum(cf for cf in cashflows if cf < 0))),
                    "net_cashflow": float(sum(cashflows)),
                    "volatility": float(cf_volatility)
                },
                "market_context": await self._get_market_context(context) if context else None
            }
            
            # Store successful calculation in memory for future optimization
            if self.memory_service and context:
                await self._store_npv_calculation(result, context)
            
            return result
            
        except Exception as e:
            logger.error(
                "advanced_npv_calculation_failed",
                error=str(e),
                cashflows_count=len(cashflows),
                discount_rate=discount_rate
            )
            raise
    
    async def _get_intelligent_discount_rate(
        self, 
        context: Optional[Dict[str, Any]]
    ) -> float:
        """Intelligently determine discount rate based on market conditions"""
        
        # Try to get current risk-free rate
        if self.risk_free_rate_cache is None:
            try:
                # Get 10-year Treasury rate as risk-free proxy
                if self.fred:
                    rate_data = self.fred.get_series('GS10', limit=1)
                    self.risk_free_rate_cache = float(rate_data.iloc[-1] / 100)  # Convert to decimal
                else:
                    # Fallback to reasonable default
                    self.risk_free_rate_cache = 0.04  # 4% default
            except:
                self.risk_free_rate_cache = 0.04
        
        base_rate = self.risk_free_rate_cache
        
        # Add market risk premium based on context
        if context:
            industry = context.get('industry', 'generic')
            company_size = context.get('company_size', 'medium')
            
            # Industry risk adjustments (simplified model)
            industry_premiums = {
                'technology': 0.02,
                'healthcare': 0.015,
                'finance': 0.01,
                'energy': 0.025,
                'manufacturing': 0.015,
                'retail': 0.02
            }
            
            size_premiums = {
                'small': 0.03,
                'medium': 0.02,
                'large': 0.01
            }
            
            industry_premium = industry_premiums.get(industry, 0.015)
            size_premium = size_premiums.get(company_size, 0.02)
            
            return base_rate + industry_premium + size_premium
        
        return base_rate + 0.06  # Default market risk premium
    
    async def _get_market_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant market context for financial calculations"""
        
        market_data = {
            "risk_free_rate": self.risk_free_rate_cache or 0.04,
            "collection_timestamp": datetime.now().isoformat()
        }
        
        # Add economic indicators if available
        try:
            if self.fred:
                # Get key economic indicators
                indicators = {
                    'unemployment_rate': 'UNRATE',
                    'inflation_rate': 'CPIAUCNS',
                    'gdp_growth': 'GDP'
                }
                
                for name, series_id in indicators.items():
                    try:
                        data = self.fred.get_series(series_id, limit=1)
                        market_data[name] = float(data.iloc[-1])
                    except:
                        pass
        except:
            pass
        
        return market_data
    
    async def _store_data_profile(
        self, 
        profile: FinancialDataProfile, 
        context: Dict[str, Any]
    ):
        """Store data profile analysis for future optimization"""
        try:
            memory_content = {
                "analysis_type": "financial_data_profile",
                "profile": {
                    "size": profile.size,
                    "volatility": profile.volatility,
                    "characteristics": [c.value for c in profile.characteristics],
                    "recommended_models": profile.recommended_models
                },
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
            
            await self.memory_service.store_conversation_memory(
                user_id=context.get("user_id", "system"),
                message=json.dumps(memory_content),
                context={
                    "category": "financial_modeling",
                    "analysis_type": "data_profile",
                    "data_size": profile.size
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store data profile: {e}")
    
    async def _store_anomaly_performance(
        self,
        method: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
        profile: FinancialDataProfile
    ):
        """Store anomaly detection performance for future optimization"""
        try:
            performance_data = {
                "analysis_type": "anomaly_detection_performance",
                "method": method,
                "accuracy_metrics": {
                    "anomaly_count": result.get("total_anomalies", 0),
                    "anomaly_ratio": result.get("anomaly_ratio", 0),
                    "confidence_mean": result.get("confidence_distribution", {}).get("mean", 0)
                },
                "data_characteristics": [c.value for c in profile.characteristics],
                "execution_timestamp": datetime.now().isoformat()
            }
            
            await self.memory_service.store_conversation_memory(
                user_id=context.get("user_id", "system"),
                message=json.dumps(performance_data),
                context={
                    "category": "anomaly_detection",
                    "method_used": method,
                    "performance_tracking": True
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store anomaly performance: {e}")
    
    async def _store_npv_calculation(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Store NPV calculation results for future reference"""
        try:
            calculation_data = {
                "analysis_type": "npv_calculation",
                "npv": result["npv"],
                "discount_rate": result["discount_rate_used"],
                "market_context": result.get("market_context"),
                "calculation_timestamp": datetime.now().isoformat()
            }
            
            await self.memory_service.store_conversation_memory(
                user_id=context.get("user_id", "system"),
                message=json.dumps(calculation_data),
                context={
                    "category": "financial_modeling",
                    "calculation_type": "npv",
                    "industry": context.get("industry")
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store NPV calculation: {e}")
    
    # Additional methods for statistical and time series anomaly detection
    async def _statistical_anomaly_detection(
        self, 
        df: pd.DataFrame, 
        profile: FinancialDataProfile
    ) -> Dict[str, Any]:
        """Statistical anomaly detection using robust methods"""
        
        values = df.iloc[:, 0].values
        
        # Use robust statistics (median, MAD) instead of mean/std
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        # Modified Z-score using MAD
        modified_z_scores = 0.6745 * (values - median) / mad
        threshold = 3.5  # More conservative threshold
        
        anomaly_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
        anomalies = []
        
        for idx in anomaly_indices:
            anomalies.append({
                "timestamp": df.index[idx].isoformat(),
                "value": float(values[idx]),
                "z_score": float(modified_z_scores[idx]),
                "deviation": float(abs(values[idx] - median)),
                "confidence": float(min(1.0, abs(modified_z_scores[idx]) / 5.0))
            })
        
        return {
            "anomalies": anomalies,
            "method_used": "robust_statistical",
            "total_anomalies": len(anomalies),
            "parameters_used": {
                "threshold": threshold,
                "median": float(median),
                "mad": float(mad)
            }
        }
    
    async def _time_series_anomaly_detection(
        self, 
        df: pd.DataFrame, 
        profile: FinancialDataProfile
    ) -> Dict[str, Any]:
        """Time series specific anomaly detection"""
        
        values = df.iloc[:, 0]
        
        if len(values) < 24:
            # Fall back to statistical method for short series
            return await self._statistical_anomaly_detection(df, profile)
        
        # Seasonal decomposition
        period = min(12, len(values) // 3)  # Adaptive period
        decomposition = seasonal_decompose(values, model='additive', period=period)
        
        # Detect anomalies in residuals
        residuals = decomposition.resid.dropna()
        residual_threshold = 2.5 * residuals.std()
        
        anomaly_mask = np.abs(residuals) > residual_threshold
        anomaly_indices = residuals[anomaly_mask].index
        
        anomalies = []
        for idx in anomaly_indices:
            if idx in values.index:
                anomalies.append({
                    "timestamp": idx.isoformat(),
                    "value": float(values[idx]),
                    "residual": float(residuals[idx]),
                    "trend_component": float(decomposition.trend[idx]) if idx in decomposition.trend.dropna().index else None,
                    "seasonal_component": float(decomposition.seasonal[idx]) if idx in decomposition.seasonal.dropna().index else None,
                    "confidence": float(min(1.0, abs(residuals[idx]) / (3 * residuals.std())))
                })
        
        return {
            "anomalies": anomalies,
            "method_used": "time_series_decomposition",
            "total_anomalies": len(anomalies),
            "decomposition_period": period,
            "residual_statistics": {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "threshold_used": float(residual_threshold)
            }
        }