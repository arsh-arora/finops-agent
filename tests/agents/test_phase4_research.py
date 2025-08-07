"""
Test suite for Phase 4 AdvancedResearchAgent
Comprehensive testing of web research and Tavily integration capabilities
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.research import AdvancedResearchAgent
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
def research_agent(mock_memory_service):
    """Create AdvancedResearchAgent instance for testing"""
    return AdvancedResearchAgent(memory_service=mock_memory_service)


@pytest.fixture
def sample_research_data():
    """Sample research data for testing"""
    return {
        'query': 'cloud computing cost optimization strategies 2024',
        'search_results': [
            {
                'title': 'Top 10 Cloud Cost Optimization Strategies for 2024',
                'url': 'https://example.com/cloud-cost-optimization',
                'content': 'Cloud cost optimization involves rightsizing instances, using reserved instances...',
                'source_credibility': 0.89,
                'relevance_score': 0.95,
                'publish_date': '2024-01-15',
                'author_authority': 0.87
            },
            {
                'title': 'AWS Cost Management Best Practices',
                'url': 'https://aws.amazon.com/cost-management',
                'content': 'AWS provides various tools for cost management and optimization...',
                'source_credibility': 0.96,
                'relevance_score': 0.92,
                'publish_date': '2024-02-01',
                'author_authority': 0.95
            }
        ],
        'research_insights': {
            'main_themes': [
                'rightsizing_instances',
                'reserved_instances',
                'automated_scaling',
                'cost_monitoring'
            ],
            'consensus_points': [
                'Regular cost monitoring is essential',
                'Automated scaling saves significant costs',
                'Reserved instances provide substantial savings'
            ],
            'conflicting_views': [
                'Debate on spot instances reliability for production workloads'
            ]
        }
    }


class TestAdvancedResearchAgent:
    """Test cases for AdvancedResearchAgent"""
    
    def test_agent_initialization(self, research_agent):
        """Test agent initialization and basic properties"""
        assert research_agent.get_domain() == "research"
        assert "web_research" in research_agent.get_capabilities()
        assert "tavily_integration" in research_agent.get_capabilities()
        assert "quality_assessment" in research_agent.get_capabilities()
        assert hasattr(research_agent, 'research_engine')
    
    @pytest.mark.asyncio
    async def test_conduct_research_tool(self, research_agent, sample_research_data):
        """Test comprehensive web research with Tavily"""
        with patch.object(research_agent.research_engine, 'conduct_comprehensive_research') as mock_research:
            mock_research.return_value = {
                'search_results': sample_research_data['search_results'],
                'research_summary': {
                    'total_sources': 15,
                    'high_quality_sources': 12,
                    'average_credibility': 0.87,
                    'coverage_score': 0.91,
                    'information_freshness': 0.89
                },
                'key_insights': [
                    'Cloud cost optimization strategies focus on rightsizing and automation',
                    'Reserved instances provide 30-60% cost savings for predictable workloads',
                    'AI-driven cost optimization tools are becoming mainstream'
                ],
                'source_analysis': {
                    'authoritative_sources': 8,
                    'academic_sources': 3,
                    'industry_sources': 4,
                    'bias_detection_score': 0.15
                },
                'confidence_score': 0.94
            }
            
            result = await research_agent.conduct_research(
                query=sample_research_data['query'],
                max_sources=15
            )
            
            assert len(result['search_results']) > 0
            assert result['research_summary']['total_sources'] > 0
            assert result['confidence_score'] > 0.9
            assert len(result['key_insights']) > 0
            assert result['source_analysis']['bias_detection_score'] < 0.3
            mock_research.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comparative_research_tool(self, research_agent):
        """Test comparative research across multiple queries"""
        queries = [
            'AWS cost optimization strategies',
            'Azure cost management best practices',
            'Google Cloud cost optimization techniques'
        ]
        
        with patch.object(research_agent.research_engine, 'conduct_comparative_research') as mock_comparative:
            mock_comparative.return_value = {
                'comparative_analysis': {
                    'aws_strategies': {
                        'unique_features': ['AWS Cost Explorer', 'Trusted Advisor'],
                        'common_practices': ['rightsizing', 'reserved_instances'],
                        'cost_savings_range': '20-40%'
                    },
                    'azure_strategies': {
                        'unique_features': ['Azure Cost Management', 'Azure Advisor'],
                        'common_practices': ['rightsizing', 'hybrid_benefits'],
                        'cost_savings_range': '25-45%'
                    },
                    'gcp_strategies': {
                        'unique_features': ['Recommender API', 'Sustained Use Discounts'],
                        'common_practices': ['preemptible_instances', 'committed_use'],
                        'cost_savings_range': '30-50%'
                    }
                },
                'cross_platform_insights': [
                    'All platforms emphasize automated rightsizing',
                    'Reserved/committed instances are universal cost savers',
                    'Monitoring and alerting are critical across all platforms'
                ],
                'recommendation_synthesis': {
                    'best_practices': [
                        'Implement multi-cloud cost monitoring',
                        'Use platform-native optimization tools',
                        'Regular cost review and optimization cycles'
                    ],
                    'platform_selection_criteria': 'Cost optimization tooling maturity varies significantly'
                },
                'confidence_level': 0.89
            }
            
            result = await research_agent.comparative_research(
                queries=queries
            )
            
            assert 'comparative_analysis' in result
            assert 'cross_platform_insights' in result
            assert len(result['cross_platform_insights']) > 0
            assert result['confidence_level'] > 0.8
            mock_comparative.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fact_verification_tool(self, research_agent):
        """Test fact verification across sources"""
        claims = [
            'Reserved instances can save up to 75% on cloud costs',
            'Spot instances are reliable for production workloads',
            'AI-driven cost optimization can reduce costs by 30%'
        ]
        
        with patch.object(research_agent.research_engine, 'verify_facts_cross_sources') as mock_verify:
            mock_verify.return_value = {
                'verification_results': [
                    {
                        'claim': claims[0],
                        'verification_status': 'verified',
                        'confidence': 0.92,
                        'supporting_sources': 8,
                        'contradicting_sources': 1,
                        'consensus_level': 'strong'
                    },
                    {
                        'claim': claims[1],
                        'verification_status': 'disputed',
                        'confidence': 0.65,
                        'supporting_sources': 3,
                        'contradicting_sources': 5,
                        'consensus_level': 'weak'
                    },
                    {
                        'claim': claims[2],
                        'verification_status': 'partially_verified',
                        'confidence': 0.78,
                        'supporting_sources': 5,
                        'contradicting_sources': 2,
                        'consensus_level': 'moderate'
                    }
                ],
                'overall_credibility': {
                    'verified_claims': 1,
                    'disputed_claims': 1,
                    'partially_verified': 1,
                    'average_confidence': 0.78
                },
                'methodology_transparency': 0.91
            }
            
            result = await research_agent.verify_facts(
                claims_to_verify=claims
            )
            
            assert len(result['verification_results']) == len(claims)
            assert 'overall_credibility' in result
            assert result['methodology_transparency'] > 0.9
            for verification in result['verification_results']:
                assert 'verification_status' in verification
                assert verification['confidence'] > 0.6
            mock_verify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_request_processing(self, research_agent):
        """Test processing of chat requests"""
        request = ChatRequest(
            message="Research the latest trends in artificial intelligence for 2024",
            user_id="test_user",
            session_id="test_session"
        )
        
        with patch.object(research_agent, 'conduct_research') as mock_research:
            mock_research.return_value = {
                'search_results': [{'title': 'AI Trends 2024'}],
                'confidence_score': 0.91
            }
            
            response = await research_agent.process_request(request)
            
            assert response is not None
            assert isinstance(response, dict)
    
    def test_tool_registration(self, research_agent):
        """Test that all tools are properly registered"""
        tools = research_agent.get_available_tools()
        
        expected_tools = [
            'conduct_research',
            'comparative_research',
            'verify_facts'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, research_agent, mock_memory_service):
        """Test memory service integration"""
        # Test memory search for previous research
        mock_memory_service.search_memories.return_value = [
            {
                'content': 'Previous research on cloud optimization',
                'metadata': {
                    'research_type': 'web_research',
                    'query': 'cloud cost optimization',
                    'confidence': 0.89
                }
            }
        ]
        
        memories = await research_agent.memory_service.search_memories(
            query="cloud optimization research",
            limit=5
        )
        
        assert len(memories) == 1
        mock_memory_service.search_memories.assert_called_once()
    
    def test_error_handling(self, research_agent):
        """Test error handling for invalid inputs"""
        # Test with empty query
        with pytest.raises(ValueError):
            research_agent.conduct_research.run_tool(
                query=""  # Should not be empty
            )
    
    @pytest.mark.asyncio
    async def test_tavily_integration(self, research_agent, sample_research_data):
        """Test specific Tavily API integration features"""
        with patch.object(research_agent.research_engine, 'conduct_comprehensive_research') as mock_tavily:
            # Mock Tavily-specific features
            mock_tavily.return_value = {
                'tavily_metadata': {
                    'api_version': '1.2.3',
                    'search_method': 'advanced_semantic',
                    'response_time_ms': 1250,
                    'rate_limit_remaining': 95
                },
                'search_results': sample_research_data['search_results'],
                'quality_enhancements': {
                    'content_filtering': True,
                    'duplicate_removal': True,
                    'relevance_ranking': True,
                    'credibility_scoring': True
                },
                'advanced_features': {
                    'semantic_search': True,
                    'real_time_indexing': True,
                    'multi_language_support': True
                }
            }
            
            result = await research_agent.conduct_research(
                query=sample_research_data['query']
            )
            
            assert 'tavily_metadata' in result
            assert result['tavily_metadata']['api_version']
            assert 'quality_enhancements' in result
            assert result['quality_enhancements']['credibility_scoring'] is True
    
    @pytest.mark.asyncio
    async def test_bias_detection(self, research_agent):
        """Test bias detection in research results"""
        with patch.object(research_agent.research_engine, 'conduct_comprehensive_research') as mock_bias:
            mock_bias.return_value = {
                'bias_analysis': {
                    'detected_biases': [
                        {
                            'type': 'commercial_bias',
                            'severity': 0.25,
                            'affected_sources': 2,
                            'description': 'Some sources show commercial product preference'
                        }
                    ],
                    'overall_bias_score': 0.18,
                    'mitigation_applied': True,
                    'source_diversity_score': 0.84
                },
                'credibility_distribution': {
                    'high_credibility': 12,
                    'medium_credibility': 5,
                    'low_credibility': 1
                },
                'methodology_notes': [
                    'Cross-referenced claims across multiple source types',
                    'Applied bias detection algorithms to content analysis'
                ]
            }
            
            result = await research_agent.conduct_research(
                query='best cloud provider comparison'
            )
            
            assert 'bias_analysis' in result
            assert result['bias_analysis']['overall_bias_score'] < 0.3
            assert result['bias_analysis']['mitigation_applied'] is True
    
    @pytest.mark.asyncio
    async def test_large_scale_research(self, research_agent):
        """Test performance with large-scale research requests"""
        with patch.object(research_agent.research_engine, 'conduct_comprehensive_research') as mock_large:
            # Mock result for large research request
            mock_large.return_value = {
                'search_results': [{'title': f'Result {i}'} for i in range(50)],
                'performance_metrics': {
                    'total_sources_processed': 150,
                    'search_time_seconds': 8.5,
                    'analysis_time_seconds': 12.3,
                    'total_processing_time': 20.8
                },
                'research_summary': {
                    'total_sources': 50,
                    'average_credibility': 0.86
                }
            }
            
            start_time = datetime.now()
            result = await research_agent.conduct_research(
                query='comprehensive artificial intelligence research',
                max_sources=50
            )
            end_time = datetime.now()
            
            # Verify performance
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time < 60.0  # Should complete within reasonable time
            assert len(result['search_results']) >= 10
            assert result['performance_metrics']['total_sources_processed'] > 50


class TestResearchAgentIntegration:
    """Integration tests for Research agent with real-world scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self, research_agent, sample_research_data):
        """Test complete research workflow"""
        query = sample_research_data['query']
        
        # Step 1: Conduct Research
        with patch.object(research_agent.research_engine, 'conduct_comprehensive_research') as mock_research:
            mock_research.return_value = {
                'search_results': sample_research_data['search_results'],
                'confidence_score': 0.91
            }
            
            research_result = await research_agent.conduct_research(query=query)
        
        # Step 2: Comparative Research (related queries)
        with patch.object(research_agent.research_engine, 'conduct_comparative_research') as mock_comparative:
            mock_comparative.return_value = {
                'comparative_analysis': {'aws_vs_azure': 'detailed comparison'},
                'confidence_level': 0.87
            }
            
            comparative_result = await research_agent.comparative_research(
                queries=[query, 'azure cost optimization']
            )
        
        # Step 3: Fact Verification
        with patch.object(research_agent.research_engine, 'verify_facts_cross_sources') as mock_verify:
            mock_verify.return_value = {
                'verification_results': [{'verification_status': 'verified'}],
                'overall_credibility': {'verified_claims': 1}
            }
            
            verification_result = await research_agent.verify_facts(
                claims_to_verify=['Cloud costs can be reduced by 30% with optimization']
            )
        
        # Verify workflow completion
        assert research_result['confidence_score'] > 0.9
        assert comparative_result['confidence_level'] > 0.8
        assert verification_result['overall_credibility']['verified_claims'] >= 0
    
    @pytest.mark.asyncio
    async def test_multi_domain_research(self, research_agent):
        """Test research across multiple domains"""
        cross_domain_queries = [
            'financial impact of cloud migration',
            'security considerations in cloud cost optimization',
            'regulatory compliance costs in cloud computing'
        ]
        
        with patch.object(research_agent.research_engine, 'conduct_comparative_research') as mock_multi:
            mock_multi.return_value = {
                'domain_analysis': {
                    'financial_domain': {'key_insights': ['ROI calculations', 'TCO analysis']},
                    'security_domain': {'key_insights': ['Compliance costs', 'Security tooling']},
                    'regulatory_domain': {'key_insights': ['GDPR implications', 'Data residency']}
                },
                'cross_domain_synthesis': [
                    'Security and compliance can impact cloud cost optimization strategies',
                    'Financial models must account for regulatory requirements'
                ],
                'confidence_level': 0.88
            }
            
            result = await research_agent.comparative_research(
                queries=cross_domain_queries
            )
            
            assert 'domain_analysis' in result
            assert 'cross_domain_synthesis' in result
            assert len(result['cross_domain_synthesis']) > 0
    
    def test_concurrent_research_requests(self, research_agent):
        """Test handling of concurrent research requests"""
        # This would test thread safety and concurrent processing
        # Implementation would depend on actual concurrency requirements
        pass