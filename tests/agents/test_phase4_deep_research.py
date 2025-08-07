"""
Test suite for Phase 4 AdvancedDeepResearchAgent
Comprehensive testing of multi-hop orchestration and cross-domain synthesis capabilities
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.deep_research import AdvancedDeepResearchAgent
from src.agents.models import ChatRequest


@pytest.fixture
def mock_memory_service():
    """Mock memory service for testing"""
    mock = Mock()
    mock.search_memories = AsyncMock(return_value=[])
    mock.add_memories = AsyncMock(return_value=None)
    mock.delete_memories = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def mock_other_agents():
    """Mock other agents for orchestration testing"""
    mock_finops = Mock()
    mock_finops.compute_npv = AsyncMock(return_value={
        'base_npv': 15000,
        'confidence_level': 0.89
    })
    
    mock_github = Mock()
    mock_github.analyze_repository_security = AsyncMock(return_value={
        'overall_security_score': 0.75,
        'vulnerability_summary': {'total': 3}
    })
    
    mock_document = Mock()
    mock_document.analyze_content = AsyncMock(return_value={
        'content_insights': {'main_topics': ['security', 'costs']},
        'confidence_score': 0.92
    })
    
    mock_research = Mock()
    mock_research.conduct_research = AsyncMock(return_value={
        'search_results': [{'title': 'Cloud Security Best Practices'}],
        'confidence_score': 0.88
    })
    
    return {
        'finops': mock_finops,
        'github': mock_github,
        'document': mock_document,
        'research': mock_research
    }


@pytest.fixture
def deep_research_agent(mock_memory_service):
    """Create AdvancedDeepResearchAgent instance for testing"""
    return AdvancedDeepResearchAgent(memory_service=mock_memory_service)


@pytest.fixture
def sample_orchestration_data():
    """Sample orchestration data for testing"""
    return {
        'research_objective': 'Analyze the financial and security implications of migrating to cloud infrastructure',
        'domains_involved': ['finops', 'github', 'research', 'document'],
        'execution_plan': [
            {
                'step': 1,
                'agent': 'research',
                'task': 'Research cloud migration cost factors',
                'dependencies': []
            },
            {
                'step': 2,
                'agent': 'finops',
                'task': 'Calculate migration costs and ROI',
                'dependencies': ['step_1']
            },
            {
                'step': 3,
                'agent': 'github',
                'task': 'Assess current codebase security',
                'dependencies': []
            },
            {
                'step': 4,
                'agent': 'document',
                'task': 'Analyze migration documentation',
                'dependencies': ['step_1']
            }
        ],
        'synthesis_requirements': [
            'Combine cost analysis with security assessment',
            'Identify trade-offs between cost and security',
            'Generate comprehensive migration recommendation'
        ]
    }


class TestAdvancedDeepResearchAgent:
    """Test cases for AdvancedDeepResearchAgent"""
    
    def test_agent_initialization(self, deep_research_agent):
        """Test agent initialization and basic properties"""
        assert deep_research_agent.get_domain() == "deep_research"
        assert "multi_hop_orchestration" in deep_research_agent.get_capabilities()
        assert "cross_domain_synthesis" in deep_research_agent.get_capabilities()
        assert "agent_coordination" in deep_research_agent.get_capabilities()
        assert hasattr(deep_research_agent, 'orchestration_engine')
    
    @pytest.mark.asyncio
    async def test_orchestrate_research_tool(self, deep_research_agent, sample_orchestration_data, mock_other_agents):
        """Test multi-hop research orchestration"""
        with patch.object(deep_research_agent.orchestration_engine, 'orchestrate_multi_hop_research') as mock_orchestrate:
            mock_orchestrate.return_value = {
                'execution_results': {
                    'step_1': {
                        'agent': 'research',
                        'status': 'completed',
                        'result': {'confidence_score': 0.88},
                        'execution_time': 3.2
                    },
                    'step_2': {
                        'agent': 'finops',
                        'status': 'completed',
                        'result': {'base_npv': 15000},
                        'execution_time': 2.8
                    },
                    'step_3': {
                        'agent': 'github',
                        'status': 'completed',
                        'result': {'overall_security_score': 0.75},
                        'execution_time': 4.1
                    },
                    'step_4': {
                        'agent': 'document',
                        'status': 'completed',
                        'result': {'confidence_score': 0.92},
                        'execution_time': 2.5
                    }
                },
                'orchestration_metadata': {
                    'total_steps': 4,
                    'successful_steps': 4,
                    'failed_steps': 0,
                    'total_execution_time': 12.6,
                    'dependency_resolution_time': 0.3,
                    'coordination_efficiency': 0.94
                },
                'synthesis_readiness': {
                    'data_completeness': 0.96,
                    'cross_domain_coverage': 0.89,
                    'synthesis_confidence': 0.91
                }
            }
            
            result = await deep_research_agent.orchestrate_research(
                research_objective=sample_orchestration_data['research_objective'],
                domains_required=sample_orchestration_data['domains_involved']
            )
            
            assert 'execution_results' in result
            assert result['orchestration_metadata']['successful_steps'] == 4
            assert result['orchestration_metadata']['failed_steps'] == 0
            assert result['synthesis_readiness']['data_completeness'] > 0.9
            mock_orchestrate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_synthesize_findings_tool(self, deep_research_agent, sample_orchestration_data):
        """Test cross-domain findings synthesis"""
        findings_data = {
            'finops_findings': {
                'migration_cost': 500000,
                'annual_savings': 200000,
                'roi_timeline': '2.5 years',
                'confidence': 0.87
            },
            'security_findings': {
                'current_vulnerabilities': 12,
                'cloud_security_improvement': 0.35,
                'compliance_requirements': ['SOC2', 'ISO27001'],
                'confidence': 0.82
            },
            'research_findings': {
                'industry_trends': ['hybrid_cloud', 'zero_trust'],
                'best_practices': ['gradual_migration', 'security_first'],
                'confidence': 0.91
            },
            'document_findings': {
                'migration_readiness': 0.73,
                'documentation_gaps': 3,
                'confidence': 0.89
            }
        }
        
        with patch.object(deep_research_agent.orchestration_engine, 'synthesize_cross_domain_findings') as mock_synthesize:
            mock_synthesize.return_value = {
                'synthesis_summary': {
                    'primary_recommendation': 'Proceed with cloud migration using phased approach',
                    'confidence_level': 0.88,
                    'decision_factors': [
                        'Strong financial case with 2.5 year ROI',
                        'Security improvements outweigh initial risks',
                        'Industry alignment with hybrid cloud trends'
                    ],
                    'risk_factors': [
                        'Current vulnerability count requires immediate attention',
                        'Documentation gaps may slow migration',
                        'Compliance requirements add complexity'
                    ]
                },
                'cross_domain_insights': [
                    {
                        'insight': 'Security investment should precede cost optimization',
                        'supporting_domains': ['finops', 'github'],
                        'confidence': 0.91
                    },
                    {
                        'insight': 'Gradual migration reduces both cost and security risks',
                        'supporting_domains': ['research', 'document'],
                        'confidence': 0.85
                    }
                ],
                'actionable_recommendations': [
                    {
                        'action': 'Address current vulnerabilities before migration',
                        'priority': 'high',
                        'estimated_effort': '4-6 weeks',
                        'cost_impact': 75000
                    },
                    {
                        'action': 'Complete documentation gaps analysis',
                        'priority': 'medium',
                        'estimated_effort': '2-3 weeks',
                        'cost_impact': 15000
                    }
                ],
                'quality_metrics': {
                    'synthesis_completeness': 0.93,
                    'cross_validation_score': 0.87,
                    'recommendation_confidence': 0.88
                }
            }
            
            result = await deep_research_agent.synthesize_findings(
                findings_data=findings_data
            )
            
            assert 'synthesis_summary' in result
            assert result['synthesis_summary']['confidence_level'] > 0.8
            assert len(result['cross_domain_insights']) > 0
            assert len(result['actionable_recommendations']) > 0
            assert result['quality_metrics']['synthesis_completeness'] > 0.9
            mock_synthesize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adaptive_planning_tool(self, deep_research_agent):
        """Test adaptive research planning"""
        initial_plan = {
            'research_objective': 'Cloud migration analysis',
            'planned_steps': 4,
            'estimated_duration': '2 hours'
        }
        
        execution_feedback = {
            'completed_steps': 2,
            'step_results': [
                {'quality_score': 0.95, 'execution_time': 1.2},
                {'quality_score': 0.78, 'execution_time': 2.8}
            ],
            'emerging_requirements': [
                'Need additional security analysis',
                'Compliance requirements more complex than expected'
            ]
        }
        
        with patch.object(deep_research_agent.orchestration_engine, 'adapt_research_plan') as mock_adapt:
            mock_adapt.return_value = {
                'updated_plan': {
                    'additional_steps': [
                        {
                            'step': 5,
                            'agent': 'github',
                            'task': 'Deep compliance vulnerability scan',
                            'estimated_time': 1.5
                        },
                        {
                            'step': 6,
                            'agent': 'research',
                            'task': 'Research compliance requirements',
                            'estimated_time': 1.0
                        }
                    ],
                    'revised_duration': '3.5 hours',
                    'confidence_adjustment': -0.05
                },
                'adaptation_rationale': [
                    'Low quality score in step 2 indicates need for additional analysis',
                    'Compliance complexity requires specialized research'
                ],
                'optimization_applied': {
                    'parallel_execution_opportunities': 2,
                    'resource_reallocation': True,
                    'efficiency_gain': 0.15
                },
                'adaptation_confidence': 0.89
            }
            
            result = await deep_research_agent.adaptive_planning(
                initial_plan=initial_plan,
                execution_feedback=execution_feedback
            )
            
            assert 'updated_plan' in result
            assert len(result['updated_plan']['additional_steps']) > 0
            assert 'adaptation_rationale' in result
            assert result['adaptation_confidence'] > 0.8
            mock_adapt.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_request_processing(self, deep_research_agent):
        """Test processing of complex chat requests"""
        request = ChatRequest(
            message="Conduct comprehensive analysis of cloud migration including costs, security, and compliance",
            user_id="test_user",
            session_id="test_session"
        )
        
        with patch.object(deep_research_agent, 'orchestrate_research') as mock_orchestrate:
            mock_orchestrate.return_value = {
                'execution_results': {'step_1': {'status': 'completed'}},
                'synthesis_readiness': {'data_completeness': 0.95}
            }
            
            response = await deep_research_agent.process_request(request)
            
            assert response is not None
            assert isinstance(response, dict)
    
    def test_tool_registration(self, deep_research_agent):
        """Test that all tools are properly registered"""
        tools = deep_research_agent.get_available_tools()
        
        expected_tools = [
            'orchestrate_research',
            'synthesize_findings',
            'adaptive_planning'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, deep_research_agent, mock_memory_service):
        """Test memory service integration for orchestration patterns"""
        # Test memory search for previous orchestration patterns
        mock_memory_service.search_memories.return_value = [
            {
                'content': 'Previous orchestration pattern for cloud migration',
                'metadata': {
                    'orchestration_type': 'multi_hop_research',
                    'domains': ['finops', 'security'],
                    'success_rate': 0.92
                }
            }
        ]
        
        memories = await deep_research_agent.memory_service.search_memories(
            query="orchestration patterns",
            limit=5
        )
        
        assert len(memories) == 1
        mock_memory_service.search_memories.assert_called_once()
    
    def test_error_handling(self, deep_research_agent):
        """Test error handling for orchestration failures"""
        # Test with invalid research objective
        with pytest.raises(ValueError):
            deep_research_agent.orchestrate_research.run_tool(
                research_objective="",  # Should not be empty
                domains_required=[]
            )
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self, deep_research_agent, sample_orchestration_data):
        """Test dependency resolution in orchestration"""
        with patch.object(deep_research_agent.orchestration_engine, 'orchestrate_multi_hop_research') as mock_orchestrate:
            mock_orchestrate.return_value = {
                'dependency_resolution': {
                    'dependency_graph': {
                        'nodes': 4,
                        'edges': 2,
                        'cycles_detected': 0
                    },
                    'execution_order': ['step_1', 'step_3', 'step_2', 'step_4'],
                    'parallel_opportunities': [
                        {'steps': ['step_1', 'step_3'], 'savings': 2.1}
                    ],
                    'critical_path': ['step_1', 'step_2'],
                    'optimization_score': 0.87
                },
                'execution_results': {'successful_steps': 4}
            }
            
            result = await deep_research_agent.orchestrate_research(
                research_objective=sample_orchestration_data['research_objective'],
                domains_required=sample_orchestration_data['domains_involved']
            )
            
            assert 'dependency_resolution' in result
            assert result['dependency_resolution']['cycles_detected'] == 0
            assert len(result['dependency_resolution']['execution_order']) == 4
    
    @pytest.mark.asyncio
    async def test_large_scale_orchestration(self, deep_research_agent):
        """Test performance with large-scale orchestration"""
        with patch.object(deep_research_agent.orchestration_engine, 'orchestrate_multi_hop_research') as mock_large:
            # Mock result for large orchestration
            mock_large.return_value = {
                'execution_results': {f'step_{i}': {'status': 'completed'} for i in range(1, 21)},
                'orchestration_metadata': {
                    'total_steps': 20,
                    'successful_steps': 18,
                    'failed_steps': 2,
                    'total_execution_time': 45.6,
                    'coordination_efficiency': 0.78
                },
                'performance_metrics': {
                    'parallel_execution_ratio': 0.65,
                    'resource_utilization': 0.82,
                    'memory_usage_mb': 256
                }
            }
            
            start_time = datetime.now()
            result = await deep_research_agent.orchestrate_research(
                research_objective='Comprehensive multi-domain analysis',
                domains_required=['finops', 'github', 'document', 'research']
            )
            end_time = datetime.now()
            
            # Verify performance
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time < 120.0  # Should complete within reasonable time
            assert result['orchestration_metadata']['total_steps'] >= 10
            assert result['orchestration_metadata']['successful_steps'] > 10


class TestDeepResearchAgentIntegration:
    """Integration tests for Deep Research agent orchestration"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_orchestration_workflow(self, deep_research_agent, sample_orchestration_data):
        """Test complete orchestration workflow"""
        objective = sample_orchestration_data['research_objective']
        domains = sample_orchestration_data['domains_involved']
        
        # Step 1: Orchestrate Research
        with patch.object(deep_research_agent.orchestration_engine, 'orchestrate_multi_hop_research') as mock_orchestrate:
            mock_orchestrate.return_value = {
                'execution_results': {
                    'step_1': {'result': {'confidence_score': 0.89}},
                    'step_2': {'result': {'base_npv': 15000}}
                },
                'synthesis_readiness': {'data_completeness': 0.94}
            }
            
            orchestration_result = await deep_research_agent.orchestrate_research(
                research_objective=objective,
                domains_required=domains
            )
        
        # Step 2: Synthesize Findings
        with patch.object(deep_research_agent.orchestration_engine, 'synthesize_cross_domain_findings') as mock_synthesize:
            mock_synthesize.return_value = {
                'synthesis_summary': {'confidence_level': 0.88},
                'cross_domain_insights': [{'confidence': 0.91}],
                'quality_metrics': {'synthesis_completeness': 0.93}
            }
            
            synthesis_result = await deep_research_agent.synthesize_findings(
                findings_data={'domain1': 'data1', 'domain2': 'data2'}
            )
        
        # Step 3: Adaptive Planning (if needed)
        with patch.object(deep_research_agent.orchestration_engine, 'adapt_research_plan') as mock_adapt:
            mock_adapt.return_value = {
                'updated_plan': {'additional_steps': []},
                'adaptation_confidence': 0.85
            }
            
            adaptation_result = await deep_research_agent.adaptive_planning(
                initial_plan={'research_objective': objective},
                execution_feedback={'quality_feedback': 'good'}
            )
        
        # Verify workflow completion
        assert orchestration_result['synthesis_readiness']['data_completeness'] > 0.9
        assert synthesis_result['synthesis_summary']['confidence_level'] > 0.8
        assert adaptation_result['adaptation_confidence'] > 0.8
    
    @pytest.mark.asyncio
    async def test_cross_domain_validation(self, deep_research_agent):
        """Test validation across multiple domains"""
        with patch.object(deep_research_agent.orchestration_engine, 'synthesize_cross_domain_findings') as mock_validate:
            mock_validate.return_value = {
                'cross_validation_results': {
                    'finops_github_consistency': 0.87,
                    'research_document_alignment': 0.92,
                    'overall_consistency_score': 0.89
                },
                'validation_insights': [
                    'Financial projections align with security cost estimates',
                    'Research findings support document recommendations'
                ],
                'confidence_boosters': [
                    'Multiple independent sources confirm key findings',
                    'Cross-domain validation increases overall confidence'
                ]
            }
            
            result = await deep_research_agent.synthesize_findings(
                findings_data={
                    'finops': {'cost_analysis': 'data'},
                    'github': {'security_analysis': 'data'},
                    'research': {'market_analysis': 'data'},
                    'document': {'policy_analysis': 'data'}
                }
            )
            
            assert 'cross_validation_results' in result
            assert result['cross_validation_results']['overall_consistency_score'] > 0.8
    
    def test_concurrent_orchestration(self, deep_research_agent):
        """Test handling of concurrent orchestration requests"""
        # This would test thread safety and concurrent orchestration
        # Implementation would depend on actual concurrency requirements
        pass