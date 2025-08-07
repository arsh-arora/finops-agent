"""
Intelligent GitHub Repository Security Analyzer
Advanced security analysis with ML-driven vulnerability assessment and contributor analysis
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import json
import re
import subprocess
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib

import git
from git import Repo
from pydriller import Repository
import bandit
from bandit.core import config as bandit_config
from bandit.core import manager
import semgrep
import requests
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import networkx as nx
import structlog

logger = structlog.get_logger(__name__)


class SecurityRiskLevel(Enum):
    """Security risk assessment levels"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class VulnerabilityType(Enum):
    """Types of vulnerabilities detected"""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entities"
    BROKEN_ACCESS = "broken_access_control"
    SECURITY_MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    KNOWN_VULN = "known_vulnerabilities"
    INSUFFICIENT_LOGGING = "insufficient_logging"


@dataclass
class SecurityFinding:
    """Individual security finding with ML-enhanced scoring"""
    finding_id: str
    vulnerability_type: VulnerabilityType
    risk_level: SecurityRiskLevel
    title: str
    description: str
    file_path: str
    line_number: Optional[int]
    confidence_score: float
    epss_score: Optional[float]  # EPSS (Exploit Prediction Scoring System)
    cve_references: List[str]
    mitigation_suggestions: List[str]
    tool_source: str
    context: Dict[str, Any]


@dataclass
class ContributorBehaviorProfile:
    """ML-based contributor behavior analysis"""
    contributor_id: str
    commit_frequency: float
    code_complexity_trend: float
    security_awareness_score: float
    collaboration_patterns: Dict[str, Any]
    risk_indicators: List[str]
    contribution_quality_score: float
    anomaly_score: float


@dataclass
class RepositorySecurityProfile:
    """Comprehensive repository security assessment"""
    repository_url: str
    analysis_timestamp: datetime
    overall_risk_score: float
    security_findings: List[SecurityFinding]
    contributor_profiles: List[ContributorBehaviorProfile]
    dependency_vulnerabilities: List[Dict[str, Any]]
    code_quality_metrics: Dict[str, Any]
    recommendation_priority_matrix: Dict[str, List[str]]
    confidence_metrics: Dict[str, float]


class IntelligentGitHubAnalyzer:
    """
    Advanced GitHub repository security analyzer with ML-driven insights
    """
    
    def __init__(self, memory_service=None):
        self.memory_service = memory_service
        self.vulnerability_cache: Dict[str, Dict[str, Any]] = {}
        self.contributor_patterns: Dict[str, Any] = {}
        
        # ML models for intelligent analysis
        self.risk_assessment_model = None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Security tools configuration
        self.bandit_config = bandit_config.BanditConfig()
    
    async def analyze_repository_comprehensive(
        self,
        repo_url: str,
        clone_depth: int = 100,
        context: Optional[Dict[str, Any]] = None
    ) -> RepositorySecurityProfile:
        """
        Comprehensive repository security analysis with ML-driven insights
        """
        logger.info(
            "repository_security_analysis_started",
            repo_url=repo_url,
            clone_depth=clone_depth
        )
        
        try:
            # Clone repository to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_path = Path(temp_dir) / "repo"
                
                # Intelligent cloning with optimization
                repo = await self._clone_repository_intelligent(repo_url, repo_path, clone_depth)
                
                # Parallel analysis execution
                analysis_tasks = [
                    self._analyze_security_vulnerabilities(repo_path),
                    self._analyze_contributor_behavior(repo_url, repo_path),
                    self._analyze_dependency_vulnerabilities(repo_path),
                    self._compute_code_quality_metrics(repo_path),
                ]
                
                (security_findings, contributor_profiles, 
                 dependency_vulns, code_metrics) = await asyncio.gather(*analysis_tasks)
                
                # ML-driven overall risk assessment
                overall_risk = await self._compute_intelligent_risk_score(
                    security_findings, contributor_profiles, dependency_vulns, code_metrics
                )
                
                # Generate intelligent recommendations
                recommendations = await self._generate_intelligent_recommendations(
                    security_findings, contributor_profiles, context
                )
                
                profile = RepositorySecurityProfile(
                    repository_url=repo_url,
                    analysis_timestamp=datetime.now(),
                    overall_risk_score=overall_risk,
                    security_findings=security_findings,
                    contributor_profiles=contributor_profiles,
                    dependency_vulnerabilities=dependency_vulns,
                    code_quality_metrics=code_metrics,
                    recommendation_priority_matrix=recommendations,
                    confidence_metrics=await self._compute_confidence_metrics(
                        security_findings, contributor_profiles
                    )
                )
                
                # Store analysis patterns for learning
                if self.memory_service and context:
                    await self._store_analysis_patterns(profile, context)
                
                logger.info(
                    "repository_security_analysis_completed",
                    repo_url=repo_url,
                    overall_risk=overall_risk,
                    findings_count=len(security_findings),
                    contributors_analyzed=len(contributor_profiles)
                )
                
                return profile
                
        except Exception as e:
            logger.error(
                "repository_security_analysis_failed",
                repo_url=repo_url,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _clone_repository_intelligent(
        self, 
        repo_url: str, 
        repo_path: Path, 
        depth: int
    ) -> Repo:
        """Intelligent repository cloning with optimization"""
        
        try:
            # Determine optimal clone strategy based on repository characteristics
            clone_options = {
                'depth': depth,
                'single_branch': True,  # Start with main branch only
                'recurse_submodules': False  # Skip submodules initially
            }
            
            # Check if shallow clone is beneficial based on memory patterns
            if self.memory_service:
                similar_repos = await self._get_similar_repository_patterns(repo_url)
                if similar_repos and any(r.get('large_repository', False) for r in similar_repos):
                    clone_options['depth'] = min(50, depth)  # More conservative for large repos
            
            logger.info(
                "cloning_repository",
                repo_url=repo_url,
                clone_options=clone_options
            )
            
            repo = Repo.clone_from(repo_url, repo_path, **clone_options)
            
            # Collect repository metadata
            repo_info = {
                'total_commits': len(list(repo.iter_commits())),
                'branch_count': len(list(repo.branches)),
                'file_count': len(list(Path(repo_path).rglob('*'))),
                'repository_size': sum(f.stat().st_size for f in Path(repo_path).rglob('*') if f.is_file())
            }
            
            logger.info("repository_cloned", repo_info=repo_info)
            return repo
            
        except Exception as e:
            logger.error("repository_clone_failed", repo_url=repo_url, error=str(e))
            raise
    
    async def _analyze_security_vulnerabilities(self, repo_path: Path) -> List[SecurityFinding]:
        """Multi-tool security vulnerability analysis with ML enhancement"""
        
        findings = []
        
        try:
            # Bandit analysis for Python security issues
            bandit_findings = await self._run_bandit_analysis(repo_path)
            findings.extend(bandit_findings)
            
            # Semgrep analysis for multi-language security patterns
            semgrep_findings = await self._run_semgrep_analysis(repo_path)
            findings.extend(semgrep_findings)
            
            # Custom ML-based pattern detection
            ml_findings = await self._run_ml_security_analysis(repo_path)
            findings.extend(ml_findings)
            
            # EPSS scoring for known vulnerabilities
            findings = await self._enrich_with_epss_scores(findings)
            
            # Intelligent deduplication and ranking
            findings = await self._deduplicate_and_rank_findings(findings)
            
            return findings
            
        except Exception as e:
            logger.error("security_vulnerability_analysis_failed", error=str(e))
            return []
    
    async def _run_bandit_analysis(self, repo_path: Path) -> List[SecurityFinding]:
        """Run Bandit security analysis on Python files"""
        
        findings = []
        
        try:
            # Find Python files
            python_files = list(repo_path.rglob("*.py"))
            if not python_files:
                return findings
            
            # Configure Bandit manager
            b_mgr = manager.BanditManager(self.bandit_config, 'file')
            
            for py_file in python_files:
                try:
                    # Run Bandit on individual file
                    b_mgr.discover_files([str(py_file)])
                    b_mgr.run_tests()
                    
                    # Extract findings
                    for issue in b_mgr.get_issue_list():
                        finding = SecurityFinding(
                            finding_id=hashlib.md5(f"{py_file}:{issue.lineno}:{issue.test_id}".encode()).hexdigest(),
                            vulnerability_type=self._map_bandit_to_vuln_type(issue.test_id),
                            risk_level=self._map_bandit_severity(issue.severity),
                            title=f"Bandit: {issue.test_id}",
                            description=issue.text,
                            file_path=str(py_file.relative_to(repo_path)),
                            line_number=issue.lineno,
                            confidence_score=self._map_bandit_confidence(issue.confidence),
                            epss_score=None,  # Will be enriched later
                            cve_references=[],
                            mitigation_suggestions=self._generate_bandit_mitigation(issue),
                            tool_source="bandit",
                            context={
                                "test_id": issue.test_id,
                                "severity": issue.severity.name,
                                "confidence": issue.confidence.name
                            }
                        )
                        findings.append(finding)
                    
                    # Reset manager for next file
                    b_mgr = manager.BanditManager(self.bandit_config, 'file')
                    
                except Exception as e:
                    logger.warning(f"Bandit analysis failed for {py_file}: {e}")
                    continue
            
            return findings
            
        except Exception as e:
            logger.error("bandit_analysis_failed", error=str(e))
            return []
    
    async def _run_semgrep_analysis(self, repo_path: Path) -> List[SecurityFinding]:
        """Run Semgrep analysis for multi-language security patterns"""
        
        findings = []
        
        try:
            # Run semgrep via subprocess (as semgrep Python API may not be available)
            cmd = [
                'semgrep',
                '--config=auto',  # Use Semgrep's curated rulesets
                '--json',
                '--quiet',
                str(repo_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                semgrep_results = json.loads(result.stdout)
                
                for result_item in semgrep_results.get('results', []):
                    finding = SecurityFinding(
                        finding_id=hashlib.md5(f"semgrep:{result_item.get('check_id')}:{result_item.get('path')}:{result_item.get('start', {}).get('line', 0)}".encode()).hexdigest(),
                        vulnerability_type=self._map_semgrep_to_vuln_type(result_item.get('check_id', '')),
                        risk_level=self._map_semgrep_severity(result_item.get('extra', {}).get('severity', 'INFO')),
                        title=f"Semgrep: {result_item.get('check_id', 'Unknown')}",
                        description=result_item.get('extra', {}).get('message', 'Security finding detected'),
                        file_path=result_item.get('path', ''),
                        line_number=result_item.get('start', {}).get('line'),
                        confidence_score=0.8,  # Semgrep generally has high confidence
                        epss_score=None,
                        cve_references=result_item.get('extra', {}).get('references', []),
                        mitigation_suggestions=self._generate_semgrep_mitigation(result_item),
                        tool_source="semgrep",
                        context=result_item.get('extra', {})
                    )
                    findings.append(finding)
            
            return findings
            
        except subprocess.TimeoutExpired:
            logger.warning("Semgrep analysis timed out")
            return []
        except FileNotFoundError:
            logger.warning("Semgrep not found, skipping analysis")
            return []
        except Exception as e:
            logger.error("semgrep_analysis_failed", error=str(e))
            return []
    
    async def _run_ml_security_analysis(self, repo_path: Path) -> List[SecurityFinding]:
        """Custom ML-based security pattern detection"""
        
        findings = []
        
        try:
            # Collect code features for ML analysis
            code_features = await self._extract_code_features(repo_path)
            
            if not code_features:
                return findings
            
            # Pattern-based detection using ML
            suspicious_patterns = await self._detect_suspicious_patterns(code_features)
            
            for pattern in suspicious_patterns:
                finding = SecurityFinding(
                    finding_id=hashlib.md5(f"ml:{pattern['file']}:{pattern['pattern_type']}:{pattern.get('line', 0)}".encode()).hexdigest(),
                    vulnerability_type=VulnerabilityType.SECURITY_MISCONFIG,  # Default type
                    risk_level=SecurityRiskLevel.MEDIUM,
                    title=f"ML Detection: {pattern['pattern_type']}",
                    description=pattern['description'],
                    file_path=pattern['file'],
                    line_number=pattern.get('line'),
                    confidence_score=pattern['confidence'],
                    epss_score=None,
                    cve_references=[],
                    mitigation_suggestions=pattern.get('suggestions', []),
                    tool_source="ml_analysis",
                    context=pattern.get('context', {})
                )
                findings.append(finding)
            
            return findings
            
        except Exception as e:
            logger.error("ml_security_analysis_failed", error=str(e))
            return []
    
    async def _extract_code_features(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Extract features from code for ML analysis"""
        
        features = []
        
        # Patterns to look for
        security_patterns = [
            (r'eval\s*\(', 'dangerous_eval', 'Use of eval() function'),
            (r'exec\s*\(', 'dangerous_exec', 'Use of exec() function'),
            (r'input\s*\(', 'user_input', 'Direct user input usage'),
            (r'os\.system\s*\(', 'os_command', 'OS command execution'),
            (r'subprocess\.(call|run|Popen)', 'subprocess_usage', 'Subprocess execution'),
            (r'pickle\.loads?\s*\(', 'pickle_usage', 'Pickle deserialization'),
            (r'yaml\.load\s*\(', 'yaml_load', 'Unsafe YAML loading'),
            (r'shell=True', 'shell_injection', 'Shell injection risk'),
            (r'password\s*=\s*["\'][^"\']{1,}["\']', 'hardcoded_password', 'Hardcoded password'),
            (r'api[_-]?key\s*=\s*["\'][^"\']{10,}["\']', 'hardcoded_api_key', 'Hardcoded API key'),
            (r'secret\s*=\s*["\'][^"\']{1,}["\']', 'hardcoded_secret', 'Hardcoded secret'),
        ]
        
        # Analyze files
        code_files = []
        for ext in ['*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.c', '*.go', '*.rs', '*.php']:
            code_files.extend(repo_path.rglob(ext))
        
        for file_path in code_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern, pattern_type, description in security_patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            features.append({
                                'file': str(file_path.relative_to(repo_path)),
                                'line': line_num,
                                'pattern_type': pattern_type,
                                'description': description,
                                'matched_text': match.group(),
                                'confidence': 0.7,  # Base confidence
                                'context': {
                                    'line_content': line.strip(),
                                    'file_extension': file_path.suffix
                                }
                            })
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
                continue
        
        return features
    
    async def _detect_suspicious_patterns(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect suspicious patterns using ML clustering and anomaly detection"""
        
        if not features:
            return []
        
        suspicious = []
        
        # Group features by type for analysis
        pattern_groups = {}
        for feature in features:
            pattern_type = feature['pattern_type']
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(feature)
        
        # Apply ML analysis to each pattern group
        for pattern_type, group_features in pattern_groups.items():
            if len(group_features) < 2:
                # If only one instance, mark as suspicious if it's a high-risk pattern
                high_risk_patterns = ['dangerous_eval', 'dangerous_exec', 'hardcoded_password', 'hardcoded_api_key']
                if pattern_type in high_risk_patterns:
                    suspicious.extend(group_features)
                continue
            
            # Use file frequency and context to determine suspiciousness
            file_counts = {}
            for feature in group_features:
                file_path = feature['file']
                file_counts[file_path] = file_counts.get(file_path, 0) + 1
            
            # Files with many instances of risky patterns are more suspicious
            for feature in group_features:
                file_count = file_counts[feature['file']]
                if file_count > 3:  # Threshold for suspiciousness
                    feature['confidence'] = min(0.95, feature['confidence'] + 0.2)
                    feature['suggestions'] = [
                        f"Multiple {pattern_type} instances found in this file",
                        "Review and validate all instances for security implications",
                        "Consider refactoring to reduce security risks"
                    ]
                    suspicious.append(feature)
        
        return suspicious
    
    async def _analyze_contributor_behavior(
        self, 
        repo_url: str, 
        repo_path: Path
    ) -> List[ContributorBehaviorProfile]:
        """Analyze contributor behavior patterns with ML insights"""
        
        profiles = []
        
        try:
            # Use PyDriller for detailed commit analysis
            repository = Repository(str(repo_path))
            
            contributor_data = {}
            
            # Collect contributor statistics
            for commit in repository.traverse_commits():
                author = commit.author.email
                
                if author not in contributor_data:
                    contributor_data[author] = {
                        'commits': [],
                        'files_modified': set(),
                        'lines_added': 0,
                        'lines_deleted': 0,
                        'commit_messages': [],
                        'commit_times': []
                    }
                
                data = contributor_data[author]
                data['commits'].append(commit)
                data['lines_added'] += commit.insertions
                data['lines_deleted'] += commit.deletions
                data['commit_messages'].append(commit.msg)
                data['commit_times'].append(commit.committer_date)
                
                for modified_file in commit.modified_files:
                    data['files_modified'].add(modified_file.filename)
            
            # Analyze each contributor
            for author, data in contributor_data.items():
                if len(data['commits']) < 2:  # Skip single-commit contributors
                    continue
                
                profile = await self._create_contributor_profile(author, data)
                profiles.append(profile)
            
            # ML-based anomaly detection on contributor patterns
            profiles = await self._detect_contributor_anomalies(profiles)
            
            return profiles
            
        except Exception as e:
            logger.error("contributor_behavior_analysis_failed", error=str(e))
            return []
    
    async def _create_contributor_profile(
        self, 
        author: str, 
        data: Dict[str, Any]
    ) -> ContributorBehaviorProfile:
        """Create detailed contributor behavior profile"""
        
        commits = data['commits']
        commit_times = data['commit_times']
        
        # Calculate metrics
        commit_frequency = len(commits) / max(1, (max(commit_times) - min(commit_times)).days or 1)
        
        # Code complexity trend (simplified metric based on lines changed)
        recent_commits = sorted(commits, key=lambda c: c.committer_date)[-10:]  # Last 10 commits
        complexity_scores = []
        for commit in recent_commits:
            # Complexity based on lines changed and files modified
            complexity = (commit.insertions + commit.deletions) * len(commit.modified_files)
            complexity_scores.append(complexity)
        
        complexity_trend = np.polyfit(range(len(complexity_scores)), complexity_scores, 1)[0] if len(complexity_scores) > 1 else 0
        
        # Security awareness score based on commit messages
        security_keywords = [
            'security', 'vulnerability', 'fix', 'patch', 'cve', 'auth', 'authorization',
            'sanitize', 'validate', 'escape', 'injection', 'xss', 'csrf'
        ]
        
        security_mentions = 0
        for msg in data['commit_messages']:
            msg_lower = msg.lower()
            security_mentions += sum(1 for keyword in security_keywords if keyword in msg_lower)
        
        security_awareness_score = min(1.0, security_mentions / max(1, len(commits)) * 10)
        
        # Collaboration patterns (simplified)
        collaboration_patterns = {
            'avg_files_per_commit': len(data['files_modified']) / len(commits),
            'commit_message_length_avg': np.mean([len(msg) for msg in data['commit_messages']]),
            'lines_per_commit_avg': (data['lines_added'] + data['lines_deleted']) / len(commits)
        }
        
        # Risk indicators
        risk_indicators = []
        if commit_frequency > 10:  # Very high frequency might indicate automated commits
            risk_indicators.append("unusually_high_commit_frequency")
        if security_awareness_score < 0.1:
            risk_indicators.append("low_security_awareness")
        if collaboration_patterns['lines_per_commit_avg'] > 1000:
            risk_indicators.append("large_commits_pattern")
        
        # Contribution quality score (simplified heuristic)
        quality_factors = [
            min(1.0, security_awareness_score * 2),  # Security awareness
            min(1.0, collaboration_patterns['commit_message_length_avg'] / 50),  # Message quality
            max(0.0, 1.0 - len(risk_indicators) * 0.3)  # Risk penalty
        ]
        contribution_quality_score = np.mean(quality_factors)
        
        return ContributorBehaviorProfile(
            contributor_id=author,
            commit_frequency=float(commit_frequency),
            code_complexity_trend=float(complexity_trend),
            security_awareness_score=float(security_awareness_score),
            collaboration_patterns=collaboration_patterns,
            risk_indicators=risk_indicators,
            contribution_quality_score=float(contribution_quality_score),
            anomaly_score=0.0  # Will be set by anomaly detection
        )
    
    async def _detect_contributor_anomalies(
        self, 
        profiles: List[ContributorBehaviorProfile]
    ) -> List[ContributorBehaviorProfile]:
        """Use ML to detect anomalous contributor behavior"""
        
        if len(profiles) < 3:  # Need minimum data for anomaly detection
            return profiles
        
        try:
            # Extract features for anomaly detection
            features = []
            for profile in profiles:
                feature_vector = [
                    profile.commit_frequency,
                    profile.code_complexity_trend,
                    profile.security_awareness_score,
                    profile.collaboration_patterns['avg_files_per_commit'],
                    profile.collaboration_patterns['lines_per_commit_avg'],
                    len(profile.risk_indicators),
                    profile.contribution_quality_score
                ]
                features.append(feature_vector)
            
            # Normalize features
            scaler = StandardScaler()
            features_normalized = scaler.fit_transform(features)
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.fit(features_normalized).decision_function(features_normalized)
            anomaly_predictions = self.anomaly_detector.predict(features_normalized)
            
            # Update profiles with anomaly scores
            for i, profile in enumerate(profiles):
                profile.anomaly_score = float(anomaly_scores[i])
                if anomaly_predictions[i] == -1:  # Anomaly detected
                    profile.risk_indicators.append("behavioral_anomaly_detected")
            
            return profiles
            
        except Exception as e:
            logger.error("contributor_anomaly_detection_failed", error=str(e))
            return profiles
    
    async def _analyze_dependency_vulnerabilities(self, repo_path: Path) -> List[Dict[str, Any]]:
        """Analyze dependencies for known vulnerabilities"""
        
        vulnerabilities = []
        
        try:
            # Check for package files
            package_files = {
                'requirements.txt': 'python',
                'package.json': 'node',
                'pom.xml': 'java',
                'Cargo.toml': 'rust',
                'go.mod': 'go'
            }
            
            for file_name, ecosystem in package_files.items():
                package_file = repo_path / file_name
                if package_file.exists():
                    deps = await self._parse_dependencies(package_file, ecosystem)
                    vulns = await self._check_dependencies_for_vulnerabilities(deps, ecosystem)
                    vulnerabilities.extend(vulns)
            
            return vulnerabilities
            
        except Exception as e:
            logger.error("dependency_vulnerability_analysis_failed", error=str(e))
            return []
    
    async def _parse_dependencies(self, package_file: Path, ecosystem: str) -> List[Dict[str, Any]]:
        """Parse dependencies from package files"""
        
        dependencies = []
        
        try:
            content = package_file.read_text(encoding='utf-8')
            
            if ecosystem == 'python' and package_file.name == 'requirements.txt':
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Simple parsing - could be enhanced
                        parts = re.split(r'[>=<~!]', line)
                        if parts:
                            dep_name = parts[0].strip()
                            version = parts[1].strip() if len(parts) > 1 else None
                            dependencies.append({
                                'name': dep_name,
                                'version': version,
                                'ecosystem': ecosystem
                            })
            
            elif ecosystem == 'node' and package_file.name == 'package.json':
                try:
                    package_data = json.loads(content)
                    for dep_type in ['dependencies', 'devDependencies']:
                        deps = package_data.get(dep_type, {})
                        for name, version in deps.items():
                            dependencies.append({
                                'name': name,
                                'version': version.lstrip('^~'),
                                'ecosystem': ecosystem,
                                'type': dep_type
                            })
                except json.JSONDecodeError:
                    pass
            
            return dependencies
            
        except Exception as e:
            logger.error(f"dependency_parsing_failed for {package_file}: {e}")
            return []
    
    async def _check_dependencies_for_vulnerabilities(
        self, 
        dependencies: List[Dict[str, Any]], 
        ecosystem: str
    ) -> List[Dict[str, Any]]:
        """Check dependencies against vulnerability databases"""
        
        vulnerabilities = []
        
        # This would typically use services like:
        # - GitHub Advisory Database
        # - OSV (Open Source Vulnerabilities)
        # - Snyk API
        # - NIST NVD
        
        # For demonstration, we'll implement a basic pattern-based check
        # In production, this would use actual vulnerability APIs
        
        high_risk_packages = {
            'python': [
                'pickle', 'yaml', 'requests', 'urllib3', 'cryptography',
                'pycrypto', 'paramiko', 'sqlalchemy', 'django', 'flask'
            ],
            'node': [
                'lodash', 'moment', 'axios', 'express', 'socket.io',
                'jsonwebtoken', 'bcrypt', 'crypto-js', 'node-sass'
            ]
        }
        
        risky_packages = high_risk_packages.get(ecosystem, [])
        
        for dep in dependencies:
            if dep['name'].lower() in risky_packages:
                vulnerabilities.append({
                    'dependency_name': dep['name'],
                    'version': dep.get('version', 'unknown'),
                    'ecosystem': ecosystem,
                    'severity': 'medium',  # Would be determined by actual vulnerability data
                    'description': f"Potentially vulnerable package: {dep['name']}",
                    'recommendations': [
                        'Update to latest version',
                        'Review security advisories',
                        'Consider alternative packages'
                    ]
                })
        
        return vulnerabilities
    
    # Helper methods for mapping security tool outputs
    
    def _map_bandit_to_vuln_type(self, test_id: str) -> VulnerabilityType:
        """Map Bandit test ID to vulnerability type"""
        mapping = {
            'B101': VulnerabilityType.BROKEN_AUTH,  # assert_used
            'B102': VulnerabilityType.INJECTION,    # exec_used
            'B103': VulnerabilityType.SECURITY_MISCONFIG,  # set_bad_file_permissions
            'B104': VulnerabilityType.BROKEN_ACCESS,  # hardcoded_bind_all_interfaces
            'B105': VulnerabilityType.SENSITIVE_DATA,  # hardcoded_password_string
            'B106': VulnerabilityType.SENSITIVE_DATA,  # hardcoded_password_funcarg
            'B107': VulnerabilityType.SENSITIVE_DATA,  # hardcoded_password_default
            'B108': VulnerabilityType.INSECURE_DESERIALIZATION,  # hardcoded_tmp_directory
            # Add more mappings as needed
        }
        return mapping.get(test_id, VulnerabilityType.SECURITY_MISCONFIG)
    
    def _map_bandit_severity(self, severity) -> SecurityRiskLevel:
        """Map Bandit severity to risk level"""
        if hasattr(severity, 'name'):
            severity = severity.name
        
        mapping = {
            'HIGH': SecurityRiskLevel.HIGH,
            'MEDIUM': SecurityRiskLevel.MEDIUM,
            'LOW': SecurityRiskLevel.LOW
        }
        return mapping.get(str(severity).upper(), SecurityRiskLevel.MEDIUM)
    
    def _map_bandit_confidence(self, confidence) -> float:
        """Map Bandit confidence to float score"""
        if hasattr(confidence, 'name'):
            confidence = confidence.name
        
        mapping = {
            'HIGH': 0.9,
            'MEDIUM': 0.7,
            'LOW': 0.5
        }
        return mapping.get(str(confidence).upper(), 0.7)
    
    def _generate_bandit_mitigation(self, issue) -> List[str]:
        """Generate mitigation suggestions for Bandit issues"""
        suggestions = [
            f"Review {issue.test_id} security issue",
            "Follow security best practices for this code pattern",
            "Consider using safer alternatives"
        ]
        
        # Add specific suggestions based on test ID
        specific_suggestions = {
            'B102': ["Avoid using exec()", "Use safer alternatives like ast.literal_eval()"],
            'B105': ["Remove hardcoded passwords", "Use environment variables or secure configuration"],
            'B108': ["Avoid hardcoded temporary directory paths", "Use tempfile module"]
        }
        
        if issue.test_id in specific_suggestions:
            suggestions.extend(specific_suggestions[issue.test_id])
        
        return suggestions
    
    def _map_semgrep_to_vuln_type(self, check_id: str) -> VulnerabilityType:
        """Map Semgrep check ID to vulnerability type"""
        if 'injection' in check_id.lower() or 'sql' in check_id.lower():
            return VulnerabilityType.INJECTION
        elif 'xss' in check_id.lower():
            return VulnerabilityType.XSS
        elif 'auth' in check_id.lower():
            return VulnerabilityType.BROKEN_AUTH
        elif 'secret' in check_id.lower() or 'password' in check_id.lower():
            return VulnerabilityType.SENSITIVE_DATA
        else:
            return VulnerabilityType.SECURITY_MISCONFIG
    
    def _map_semgrep_severity(self, severity: str) -> SecurityRiskLevel:
        """Map Semgrep severity to risk level"""
        mapping = {
            'ERROR': SecurityRiskLevel.HIGH,
            'WARNING': SecurityRiskLevel.MEDIUM,
            'INFO': SecurityRiskLevel.LOW
        }
        return mapping.get(severity.upper(), SecurityRiskLevel.MEDIUM)
    
    def _generate_semgrep_mitigation(self, result: Dict[str, Any]) -> List[str]:
        """Generate mitigation suggestions for Semgrep findings"""
        suggestions = [
            "Review identified security pattern",
            "Follow secure coding practices"
        ]
        
        # Extract suggestions from Semgrep metadata if available
        extra = result.get('extra', {})
        if 'fix' in extra:
            suggestions.append(f"Suggested fix: {extra['fix']}")
        
        if 'references' in extra:
            suggestions.extend([f"Reference: {ref}" for ref in extra['references'][:3]])
        
        return suggestions
    
    async def _enrich_with_epss_scores(self, findings: List[SecurityFinding]) -> List[SecurityFinding]:
        """Enrich findings with EPSS (Exploit Prediction Scoring System) scores"""
        
        # In production, this would query the EPSS API
        # For demonstration, we'll assign mock EPSS scores
        
        for finding in findings:
            if finding.cve_references:
                # Mock EPSS score based on risk level
                if finding.risk_level == SecurityRiskLevel.CRITICAL:
                    finding.epss_score = 0.8 + (0.2 * np.random.random())
                elif finding.risk_level == SecurityRiskLevel.HIGH:
                    finding.epss_score = 0.5 + (0.3 * np.random.random())
                else:
                    finding.epss_score = 0.1 + (0.4 * np.random.random())
        
        return findings
    
    async def _deduplicate_and_rank_findings(self, findings: List[SecurityFinding]) -> List[SecurityFinding]:
        """Deduplicate similar findings and rank by risk"""
        
        if not findings:
            return findings
        
        # Simple deduplication based on file and vulnerability type
        seen_combinations = set()
        deduplicated = []
        
        for finding in findings:
            key = (finding.file_path, finding.vulnerability_type, finding.title)
            if key not in seen_combinations:
                seen_combinations.add(key)
                deduplicated.append(finding)
        
        # Rank by risk level and confidence
        risk_order = {
            SecurityRiskLevel.CRITICAL: 4,
            SecurityRiskLevel.HIGH: 3,
            SecurityRiskLevel.MEDIUM: 2,
            SecurityRiskLevel.LOW: 1,
            SecurityRiskLevel.INFO: 0
        }
        
        deduplicated.sort(
            key=lambda f: (risk_order.get(f.risk_level, 0), f.confidence_score),
            reverse=True
        )
        
        return deduplicated
    
    async def _compute_code_quality_metrics(self, repo_path: Path) -> Dict[str, Any]:
        """Compute comprehensive code quality metrics"""
        
        metrics = {
            'total_files': 0,
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'complexity_score': 0.0,
            'duplication_ratio': 0.0,
            'test_coverage_estimate': 0.0,
            'language_distribution': {},
            'file_size_distribution': {}
        }
        
        try:
            code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb'}
            
            total_complexity = 0
            file_count = 0
            language_counts = {}
            
            for file_path in repo_path.rglob('*'):
                if file_path.is_file() and file_path.suffix in code_extensions:
                    metrics['total_files'] += 1
                    
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        lines = content.split('\n')
                        
                        metrics['total_lines'] += len(lines)
                        
                        # Simple line classification
                        code_lines = 0
                        comment_lines = 0
                        blank_lines = 0
                        
                        for line in lines:
                            stripped = line.strip()
                            if not stripped:
                                blank_lines += 1
                            elif stripped.startswith(('#', '//', '/*', '*')):
                                comment_lines += 1
                            else:
                                code_lines += 1
                        
                        metrics['code_lines'] += code_lines
                        metrics['comment_lines'] += comment_lines
                        metrics['blank_lines'] += blank_lines
                        
                        # Language distribution
                        lang = file_path.suffix
                        language_counts[lang] = language_counts.get(lang, 0) + 1
                        
                        # Simple complexity estimation (count of control structures)
                        complexity_keywords = ['if', 'for', 'while', 'switch', 'case', 'catch', 'except']
                        complexity = sum(content.lower().count(keyword) for keyword in complexity_keywords)
                        total_complexity += complexity
                        file_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze {file_path}: {e}")
                        continue
            
            # Calculate derived metrics
            if file_count > 0:
                metrics['complexity_score'] = total_complexity / file_count
            
            metrics['language_distribution'] = language_counts
            
            # Simple test coverage estimate based on test file presence
            test_files = list(repo_path.rglob('*test*')) + list(repo_path.rglob('*spec*'))
            total_files = metrics['total_files'] or 1
            metrics['test_coverage_estimate'] = len(test_files) / total_files
            
            return metrics
            
        except Exception as e:
            logger.error("code_quality_metrics_failed", error=str(e))
            return metrics
    
    async def _compute_intelligent_risk_score(
        self,
        security_findings: List[SecurityFinding],
        contributor_profiles: List[ContributorBehaviorProfile],
        dependency_vulns: List[Dict[str, Any]],
        code_metrics: Dict[str, Any]
    ) -> float:
        """Compute overall repository risk score using ML"""
        
        try:
            # Security findings score (0-40 points)
            security_score = 0.0
            risk_weights = {
                SecurityRiskLevel.CRITICAL: 10,
                SecurityRiskLevel.HIGH: 7,
                SecurityRiskLevel.MEDIUM: 4,
                SecurityRiskLevel.LOW: 2,
                SecurityRiskLevel.INFO: 1
            }
            
            for finding in security_findings:
                weight = risk_weights.get(finding.risk_level, 1)
                security_score += weight * finding.confidence_score
            
            security_score = min(40.0, security_score)  # Cap at 40
            
            # Contributor risk score (0-25 points)
            contributor_score = 0.0
            if contributor_profiles:
                anomaly_contributors = [p for p in contributor_profiles if 'behavioral_anomaly_detected' in p.risk_indicators]
                low_quality_contributors = [p for p in contributor_profiles if p.contribution_quality_score < 0.5]
                
                contributor_score = min(25.0, len(anomaly_contributors) * 5 + len(low_quality_contributors) * 3)
            
            # Dependency risk score (0-20 points)
            dependency_score = min(20.0, len(dependency_vulns) * 2)
            
            # Code quality score (0-15 points)
            quality_score = 0.0
            if code_metrics.get('complexity_score', 0) > 10:
                quality_score += 5
            if code_metrics.get('test_coverage_estimate', 1) < 0.3:
                quality_score += 5
            if code_metrics.get('total_files', 0) > 1000:  # Large codebase
                quality_score += 5
            
            # Overall risk score (0-100)
            overall_risk = security_score + contributor_score + dependency_score + quality_score
            
            # Normalize to 0-1 scale
            normalized_risk = min(1.0, overall_risk / 100.0)
            
            return float(normalized_risk)
            
        except Exception as e:
            logger.error("risk_score_computation_failed", error=str(e))
            return 0.5  # Default medium risk
    
    async def _generate_intelligent_recommendations(
        self,
        security_findings: List[SecurityFinding],
        contributor_profiles: List[ContributorBehaviorProfile],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """Generate intelligent, prioritized recommendations"""
        
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': [],
            'monitoring': []
        }
        
        # Immediate actions (critical/high severity)
        critical_findings = [f for f in security_findings if f.risk_level in [SecurityRiskLevel.CRITICAL, SecurityRiskLevel.HIGH]]
        if critical_findings:
            recommendations['immediate'].append(f"Address {len(critical_findings)} critical/high severity security issues")
            
            # Group by vulnerability type for better recommendations
            vuln_groups = {}
            for finding in critical_findings:
                vuln_type = finding.vulnerability_type
                if vuln_type not in vuln_groups:
                    vuln_groups[vuln_type] = []
                vuln_groups[vuln_type].append(finding)
            
            for vuln_type, findings in vuln_groups.items():
                recommendations['immediate'].append(f"Fix {len(findings)} {vuln_type.value} vulnerabilities")
        
        # Short-term actions (medium severity + code quality)
        medium_findings = [f for f in security_findings if f.risk_level == SecurityRiskLevel.MEDIUM]
        if medium_findings:
            recommendations['short_term'].append(f"Review and address {len(medium_findings)} medium severity issues")
        
        if any('behavioral_anomaly_detected' in p.risk_indicators for p in contributor_profiles):
            recommendations['short_term'].append("Investigate unusual contributor behavior patterns")
        
        # Long-term improvements
        recommendations['long_term'].extend([
            "Implement automated security scanning in CI/CD pipeline",
            "Establish security code review process",
            "Conduct security training for development team"
        ])
        
        # Monitoring recommendations
        recommendations['monitoring'].extend([
            "Set up dependency vulnerability monitoring",
            "Monitor for new security advisories affecting your dependencies",
            "Implement logging and monitoring for security events"
        ])
        
        return recommendations
    
    async def _compute_confidence_metrics(
        self,
        security_findings: List[SecurityFinding],
        contributor_profiles: List[ContributorBehaviorProfile]
    ) -> Dict[str, float]:
        """Compute confidence metrics for the analysis"""
        
        if not security_findings and not contributor_profiles:
            return {'overall_confidence': 0.0}
        
        # Security analysis confidence
        security_confidence = 0.0
        if security_findings:
            confidence_scores = [f.confidence_score for f in security_findings]
            security_confidence = np.mean(confidence_scores)
        
        # Contributor analysis confidence
        contributor_confidence = 0.8  # High confidence in behavioral analysis
        if len(contributor_profiles) < 3:
            contributor_confidence = 0.6  # Lower confidence with few contributors
        
        # Overall confidence (weighted average)
        weights = [0.6, 0.4]  # Security findings weighted higher
        confidences = [security_confidence, contributor_confidence]
        overall_confidence = np.average(confidences, weights=weights)
        
        return {
            'overall_confidence': float(overall_confidence),
            'security_analysis_confidence': float(security_confidence),
            'contributor_analysis_confidence': float(contributor_confidence),
            'sample_size_adequacy': len(security_findings) >= 5  # Adequate sample for analysis
        }
    
    async def _get_similar_repository_patterns(self, repo_url: str) -> List[Dict[str, Any]]:
        """Retrieve similar repository analysis patterns from memory"""
        
        if not self.memory_service:
            return []
        
        try:
            similar_patterns = await self.memory_service.retrieve_relevant_memories(
                query=f"github repository analysis {repo_url}",
                limit=5
            )
            
            patterns = []
            for memory in similar_patterns:
                try:
                    if hasattr(memory, 'content'):
                        pattern_data = json.loads(memory.content)
                        patterns.append(pattern_data)
                except:
                    continue
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Failed to retrieve similar repository patterns: {e}")
            return []
    
    async def _store_analysis_patterns(
        self,
        profile: RepositorySecurityProfile,
        context: Dict[str, Any]
    ):
        """Store repository analysis patterns for future learning"""
        
        if not self.memory_service:
            return
        
        try:
            analysis_data = {
                'analysis_type': 'github_repository_security',
                'repository_url': profile.repository_url,
                'overall_risk_score': profile.overall_risk_score,
                'findings_summary': {
                    'total_findings': len(profile.security_findings),
                    'critical_findings': len([f for f in profile.security_findings if f.risk_level == SecurityRiskLevel.CRITICAL]),
                    'high_findings': len([f for f in profile.security_findings if f.risk_level == SecurityRiskLevel.HIGH])
                },
                'contributor_insights': {
                    'total_contributors': len(profile.contributor_profiles),
                    'anomalous_contributors': len([p for p in profile.contributor_profiles if 'behavioral_anomaly_detected' in p.risk_indicators])
                },
                'code_metrics_summary': profile.code_quality_metrics,
                'analysis_timestamp': profile.analysis_timestamp.isoformat(),
                'successful_analysis': True
            }
            
            await self.memory_service.store_conversation_memory(
                user_id=context.get('user_id', 'system'),
                message=json.dumps(analysis_data),
                context={
                    'category': 'github_security_analysis',
                    'repository_url': profile.repository_url,
                    'overall_risk': profile.overall_risk_score
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to store analysis patterns: {e}")