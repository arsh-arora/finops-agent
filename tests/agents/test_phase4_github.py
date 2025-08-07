"""
Test suite for Phase 4 AdvancedGitHubAgent
Comprehensive testing of security analysis and repository intelligence capabilities
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.github import AdvancedGitHubAgent
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
def github_agent(mock_memory_service):
    """Create AdvancedGitHubAgent instance for testing"""
    return AdvancedGitHubAgent(memory_service=mock_memory_service)


@pytest.fixture
def sample_repository_data():
    """Sample repository data for testing"""
    return {
        'repo_url': 'https://github.com/example/test-repo',
        'repo_path': '/tmp/test-repo',
        'vulnerabilities': [
            {
                'id': 'CVE-2023-1234',
                'severity': 'high',
                'description': 'SQL injection vulnerability',
                'file': 'src/database.py',
                'line': 42,
                'epss_score': 0.85
            }
        ],
        'contributors': [
            {
                'username': 'alice',
                'commits': 150,
                'lines_added': 5000,
                'lines_deleted': 1200,
                'first_commit': '2023-01-15',
                'last_commit': '2024-01-10'
            },
            {
                'username': 'bob',
                'commits': 80,
                'lines_added': 2500,
                'lines_deleted': 800,
                'first_commit': '2023-06-01',
                'last_commit': '2024-01-05'
            }
        ],
        'security_metrics': {
            'total_vulnerabilities': 3,
            'critical_vulnerabilities': 1,
            'high_vulnerabilities': 2,
            'security_score': 0.72
        }
    }


class TestAdvancedGitHubAgent:
    """Test cases for AdvancedGitHubAgent"""
    
    def test_agent_initialization(self, github_agent):
        """Test agent initialization and basic properties"""
        assert github_agent.get_domain() == "github"
        assert "repository_security_analysis" in github_agent.get_capabilities()
        assert "vulnerability_assessment" in github_agent.get_capabilities()
        assert "contributor_behavior_analysis" in github_agent.get_capabilities()
        assert hasattr(github_agent, 'security_analyzer')
    
    @pytest.mark.asyncio
    async def test_analyze_repository_security_tool(self, github_agent, sample_repository_data):
        """Test comprehensive repository security analysis"""
        with patch.object(github_agent.security_analyzer, 'analyze_repository_security') as mock_security:
            mock_security.return_value = {
                'overall_security_score': 0.72,
                'vulnerability_summary': {
                    'total': 3,
                    'critical': 1,
                    'high': 2,
                    'medium': 0,
                    'low': 0
                },
                'security_tools_used': ['bandit', 'semgrep', 'ml_analysis'],
                'recommendations': [
                    'Fix critical SQL injection vulnerability in src/database.py',
                    'Implement input validation in user authentication module'
                ],
                'confidence_score': 0.91,
                'analysis_completeness': 0.95
            }
            
            result = await github_agent.analyze_repository_security(
                repo_url=sample_repository_data['repo_url']
            )
            
            assert result['overall_security_score'] > 0
            assert result['vulnerability_summary']['total'] >= 0
            assert 'security_tools_used' in result
            assert len(result['recommendations']) > 0
            assert result['confidence_score'] > 0.9
            mock_security.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_assess_vulnerabilities_tool(self, github_agent, sample_repository_data):
        """Test vulnerability assessment with EPSS scoring"""
        with patch.object(github_agent.security_analyzer, 'assess_vulnerabilities_with_epss') as mock_assess:
            mock_assess.return_value = {
                'vulnerabilities': [
                    {
                        'id': 'CVE-2023-1234',
                        'severity': 'high',
                        'epss_score': 0.85,
                        'exploit_probability': 'very_high',
                        'priority_score': 9.2,
                        'remediation_effort': 'medium',
                        'business_impact': 'high'
                    }
                ],
                'risk_summary': {
                    'total_risk_score': 8.7,
                    'exploitation_likelihood': 0.85,
                    'business_impact_rating': 'high'
                },
                'prioritized_fixes': [
                    {
                        'vulnerability_id': 'CVE-2023-1234',
                        'priority': 1,
                        'estimated_fix_time': '4 hours'
                    }
                ],
                'confidence_level': 0.92
            }
            
            result = await github_agent.assess_vulnerabilities(
                repo_path=sample_repository_data['repo_path']
            )
            
            assert 'vulnerabilities' in result
            assert 'risk_summary' in result
            assert result['risk_summary']['total_risk_score'] > 0
            assert len(result['prioritized_fixes']) > 0
            assert result['confidence_level'] > 0.9
            mock_assess.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_contributors_tool(self, github_agent, sample_repository_data):
        """Test contributor behavior analysis"""
        with patch.object(github_agent.security_analyzer, 'analyze_contributor_patterns') as mock_contributors:
            mock_contributors.return_value = {
                'contributor_profiles': [
                    {
                        'username': 'alice',
                        'risk_score': 0.15,
                        'activity_pattern': 'normal',
                        'anomalies_detected': [],
                        'trust_level': 'high',
                        'contribution_quality': 0.88
                    },
                    {
                        'username': 'bob',
                        'risk_score': 0.32,
                        'activity_pattern': 'irregular',
                        'anomalies_detected': ['unusual_commit_timing'],
                        'trust_level': 'medium',
                        'contribution_quality': 0.75
                    }
                ],
                'team_security_metrics': {
                    'average_trust_level': 0.82,
                    'high_risk_contributors': 0,
                    'anomaly_detection_confidence': 0.89
                },
                'recommendations': [
                    'Monitor bob for unusual commit patterns',
                    'Consider code review requirements for medium-trust contributors'
                ]
            }
            
            result = await github_agent.analyze_contributors(
                repo_path=sample_repository_data['repo_path']
            )
            
            assert 'contributor_profiles' in result
            assert 'team_security_metrics' in result
            assert len(result['contributor_profiles']) > 0
            assert result['team_security_metrics']['average_trust_level'] > 0
            mock_contributors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compare_repositories_tool(self, github_agent):
        """Test cross-repository security comparison"""
        repos = [
            'https://github.com/example/repo1',
            'https://github.com/example/repo2'
        ]
        
        with patch.object(github_agent.security_analyzer, 'compare_repository_security') as mock_compare:
            mock_compare.return_value = {
                'comparison_summary': {
                    'repo1': {
                        'security_score': 0.85,
                        'vulnerability_count': 2,
                        'risk_level': 'low'
                    },
                    'repo2': {
                        'security_score': 0.72,
                        'vulnerability_count': 5,
                        'risk_level': 'medium'
                    }
                },
                'relative_analysis': {
                    'most_secure': 'repo1',
                    'needs_attention': 'repo2',
                    'security_gap': 0.13
                },
                'common_patterns': [
                    'Both repositories use similar authentication methods',
                    'Dependency management practices vary significantly'
                ],
                'recommendations': [
                    'Apply repo1 security practices to repo2',
                    'Standardize dependency scanning across repositories'
                ]
            }
            
            result = await github_agent.compare_repositories(
                repository_urls=repos
            )
            
            assert 'comparison_summary' in result
            assert 'relative_analysis' in result
            assert 'most_secure' in result['relative_analysis']
            assert len(result['recommendations']) > 0
            mock_compare.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_request_processing(self, github_agent):
        """Test processing of chat requests"""
        request = ChatRequest(
            message="Analyze the security of https://github.com/example/test-repo",
            user_id="test_user",
            session_id="test_session"
        )
        
        with patch.object(github_agent, 'analyze_repository_security') as mock_analyze:
            mock_analyze.return_value = {
                'overall_security_score': 0.78,
                'vulnerability_summary': {'total': 2},
                'confidence_score': 0.89
            }
            
            response = await github_agent.process_request(request)
            
            assert response is not None
            assert isinstance(response, dict)
    
    def test_tool_registration(self, github_agent):
        """Test that all tools are properly registered"""
        tools = github_agent.get_available_tools()
        
        expected_tools = [
            'analyze_repository_security',
            'assess_vulnerabilities',
            'analyze_contributors',
            'compare_repositories'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, github_agent, mock_memory_service):
        """Test memory service integration"""
        # Test memory search for previous analyses
        mock_memory_service.search_memories.return_value = [
            {
                'content': 'Previous security analysis result',
                'metadata': {
                    'analysis_type': 'security',
                    'repo_url': 'https://github.com/example/test-repo',
                    'security_score': 0.75
                }
            }
        ]
        
        memories = await github_agent.memory_service.search_memories(
            query="security analysis",
            limit=5
        )
        
        assert len(memories) == 1
        mock_memory_service.search_memories.assert_called_once()
    
    def test_error_handling(self, github_agent):
        """Test error handling for invalid inputs"""
        # Test with invalid repository URL
        with pytest.raises((ValueError, TypeError)):
            github_agent.analyze_repository_security.run_tool(
                repo_url="invalid-url"  # Should be valid GitHub URL
            )
    
    @pytest.mark.asyncio
    async def test_security_tool_integration(self, github_agent, sample_repository_data):
        """Test integration with security tools (Bandit, Semgrep)"""
        with patch.object(github_agent.security_analyzer, 'analyze_repository_security') as mock_security:
            # Mock result with multiple security tools
            mock_security.return_value = {
                'tool_results': {
                    'bandit': {
                        'issues_found': 3,
                        'severity_breakdown': {'high': 1, 'medium': 2}
                    },
                    'semgrep': {
                        'rules_matched': 5,
                        'vulnerabilities': 2
                    },
                    'ml_analysis': {
                        'anomaly_score': 0.15,
                        'patterns_detected': ['safe_coding_practices']
                    }
                },
                'overall_security_score': 0.82,
                'confidence_score': 0.94
            }
            
            result = await github_agent.analyze_repository_security(
                repo_url=sample_repository_data['repo_url']
            )
            
            assert 'tool_results' in result
            assert 'bandit' in result['tool_results']
            assert 'semgrep' in result['tool_results']
            assert 'ml_analysis' in result['tool_results']
            assert result['confidence_score'] > 0.9


class TestGitHubAgentIntegration:
    """Integration tests for GitHub agent with real-world scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_workflow(self, github_agent, sample_repository_data):
        """Test complete security analysis workflow"""
        repo_url = sample_repository_data['repo_url']
        
        # Step 1: Repository Security Analysis
        with patch.object(github_agent.security_analyzer, 'analyze_repository_security') as mock_security:
            mock_security.return_value = {
                'overall_security_score': 0.75,
                'vulnerability_summary': {'total': 3}
            }
            
            security_result = await github_agent.analyze_repository_security(repo_url=repo_url)
        
        # Step 2: Vulnerability Assessment
        with patch.object(github_agent.security_analyzer, 'assess_vulnerabilities_with_epss') as mock_vuln:
            mock_vuln.return_value = {
                'risk_summary': {'total_risk_score': 7.5},
                'prioritized_fixes': [{'priority': 1}]
            }
            
            vuln_result = await github_agent.assess_vulnerabilities(repo_path='/tmp/repo')
        
        # Step 3: Contributor Analysis
        with patch.object(github_agent.security_analyzer, 'analyze_contributor_patterns') as mock_contrib:
            mock_contrib.return_value = {
                'team_security_metrics': {'average_trust_level': 0.85},
                'contributor_profiles': [{'risk_score': 0.2}]
            }
            
            contrib_result = await github_agent.analyze_contributors(repo_path='/tmp/repo')
        
        # Verify workflow completion
        assert security_result['overall_security_score'] > 0
        assert vuln_result['risk_summary']['total_risk_score'] > 0
        assert contrib_result['team_security_metrics']['average_trust_level'] > 0.8
    
    @pytest.mark.asyncio
    async def test_large_repository_analysis(self, github_agent):
        """Test performance with large repositories"""
        with patch.object(github_agent.security_analyzer, 'analyze_repository_security') as mock_security:
            # Mock result for large repository
            mock_security.return_value = {
                'overall_security_score': 0.68,
                'files_analyzed': 15000,
                'analysis_time_seconds': 45.2,
                'memory_usage_mb': 512,
                'vulnerability_summary': {'total': 25}
            }
            
            start_time = datetime.now()
            result = await github_agent.analyze_repository_security(
                repo_url='https://github.com/large/repository'
            )
            end_time = datetime.now()
            
            # Verify performance metrics
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time < 60.0  # Should complete within reasonable time
            assert result['files_analyzed'] > 1000
    
    def test_concurrent_analyses(self, github_agent):
        """Test handling of concurrent repository analyses"""
        # This would test thread safety and concurrent processing
        # Implementation would depend on actual concurrency requirements
        pass