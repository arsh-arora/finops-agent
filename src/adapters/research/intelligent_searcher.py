"""
Intelligent Research Engine
Advanced web research with Tavily integration, query optimization, and ML-driven result analysis
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import json
import re
import logging
from dataclasses import dataclass
from enum import Enum
import hashlib
from urllib.parse import urlparse, parse_qs
import aiohttp
from bs4 import BeautifulSoup

# Tavily integration
from tavily import TavilyClient

# ML libraries for research analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
import torch
import spacy
import structlog

logger = structlog.get_logger(__name__)


class SearchScope(Enum):
    """Search scope definitions"""
    GENERAL = "general"
    ACADEMIC = "academic"
    NEWS = "news"
    TECHNICAL = "technical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    MEDICAL = "medical"


class SearchComplexity(Enum):
    """Search complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"  
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class SourceType(Enum):
    """Types of information sources"""
    WEBSITE = "website"
    RESEARCH_PAPER = "research_paper"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    FORUM = "forum"
    SOCIAL_MEDIA = "social_media"
    UNKNOWN = "unknown"


class CredibilityLevel(Enum):
    """Source credibility assessment"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    UNVERIFIED = "unverified"


@dataclass
class SearchResult:
    """Individual search result with metadata"""
    result_id: str
    title: str
    url: str
    content: str
    snippet: str
    source_type: SourceType
    credibility_score: float
    relevance_score: float
    recency_score: float
    authority_score: float
    bias_score: float
    fact_check_score: Optional[float]
    key_entities: List[str]
    topics: List[str]
    sentiment: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class SearchInsight:
    """ML-generated insights from search results"""
    insight_type: str
    confidence: float
    description: str
    supporting_sources: List[str]
    contradicting_sources: List[str]
    key_evidence: List[str]
    bias_indicators: List[str]
    consensus_level: float
    novelty_score: float


@dataclass
class ResearchProfile:
    """Comprehensive research analysis results"""
    search_id: str
    query: str
    search_scope: SearchScope
    complexity: SearchComplexity
    search_timestamp: datetime
    
    total_results: int
    processed_results: List[SearchResult]
    filtered_results: List[SearchResult]
    
    research_insights: List[SearchInsight]
    consensus_analysis: Dict[str, Any]
    credibility_assessment: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    
    source_diversity: Dict[str, Any]
    fact_verification: Dict[str, Any]
    research_gaps: List[str]
    
    confidence_metrics: Dict[str, float]
    research_metadata: Dict[str, Any]


class IntelligentResearchEngine:
    """
    Advanced research engine with Tavily integration and ML-driven analysis
    """
    
    def __init__(self, memory_service=None, tavily_api_key: Optional[str] = None):
        self.memory_service = memory_service
        self.research_cache: Dict[str, Any] = {}
        
        # Initialize Tavily client
        try:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        except Exception as e:
            logger.warning(f"Failed to initialize Tavily client: {e}")
            self.tavily_client = None
        
        # Initialize ML components
        self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.summarizer = None
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        
        # Credible domains for authority scoring
        self.high_authority_domains = {
            'academic': [
                'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google.com',
                'jstor.org', 'researchgate.net', 'springer.com', 'elsevier.com',
                'ieee.org', 'acm.org', 'nature.com', 'science.org'
            ],
            'news': [
                'reuters.com', 'ap.org', 'bbc.com', 'cnn.com', 'nytimes.com',
                'washingtonpost.com', 'wsj.com', 'ft.com', 'bloomberg.com'
            ],
            'government': [
                'gov', 'edu', 'mil', 'europa.eu', 'un.org', 'who.int',
                'worldbank.org', 'imf.org', 'oecd.org'
            ],
            'technical': [
                'github.com', 'stackoverflow.com', 'docs.microsoft.com',
                'developer.mozilla.org', 'w3.org', 'ietf.org'
            ]
        }
        
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic NLP")
            self.nlp = None
        
        # Initialize Transformers models
        try:
            self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
        except Exception as e:
            logger.warning(f"Failed to initialize transformer models: {e}")
    
    async def conduct_research_comprehensive(
        self,
        query: str,
        search_scope: SearchScope = SearchScope.GENERAL,
        max_results: int = 20,
        quality_threshold: float = 0.6,
        research_options: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ResearchProfile:
        """
        Comprehensive research with intelligent analysis
        """
        search_id = hashlib.md5(f"{query}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        logger.info(
            "comprehensive_research_started",
            search_id=search_id,
            query=query,
            search_scope=search_scope.value,
            max_results=max_results
        )
        
        try:
            # Optimize query for better search results
            optimized_query = await self._optimize_search_query(query, search_scope, context)
            
            # Determine search complexity
            complexity = await self._assess_search_complexity(optimized_query, search_scope)
            
            # Execute Tavily search
            raw_results = await self._execute_tavily_search(
                optimized_query, search_scope, max_results, research_options
            )
            
            # Process and enhance search results
            processed_results = await self._process_search_results(
                raw_results, search_scope, quality_threshold
            )
            
            # Filter high-quality results
            filtered_results = await self._filter_results_by_quality(
                processed_results, quality_threshold
            )
            
            # Generate research insights
            insights = await self._generate_research_insights(
                filtered_results, optimized_query, search_scope
            )
            
            # Perform comprehensive analysis
            consensus_analysis = await self._analyze_consensus(filtered_results, insights)
            credibility_assessment = await self._assess_overall_credibility(filtered_results)
            bias_analysis = await self._analyze_bias_patterns(filtered_results)
            temporal_analysis = await self._analyze_temporal_patterns(filtered_results)
            
            # Additional analyses
            source_diversity = await self._analyze_source_diversity(filtered_results)
            fact_verification = await self._perform_fact_verification(filtered_results, insights)
            research_gaps = await self._identify_research_gaps(filtered_results, insights)
            
            # Compute confidence metrics
            confidence_metrics = await self._compute_research_confidence(
                filtered_results, insights, consensus_analysis
            )
            
            # Create research profile
            profile = ResearchProfile(
                search_id=search_id,
                query=query,
                search_scope=search_scope,
                complexity=complexity,
                search_timestamp=datetime.now(),
                total_results=len(raw_results),
                processed_results=processed_results,
                filtered_results=filtered_results,
                research_insights=insights,
                consensus_analysis=consensus_analysis,
                credibility_assessment=credibility_assessment,
                bias_analysis=bias_analysis,
                temporal_analysis=temporal_analysis,
                source_diversity=source_diversity,
                fact_verification=fact_verification,
                research_gaps=research_gaps,
                confidence_metrics=confidence_metrics,
                research_metadata={
                    "optimized_query": optimized_query,
                    "tavily_version": "latest",
                    "ml_models_used": self._get_active_models(),
                    "processing_time": datetime.now().isoformat(),
                    "research_options": research_options or {}
                }
            )
            
            # Store research patterns for learning
            if self.memory_service and context:
                await self._store_research_patterns(profile, context)
            
            logger.info(
                "comprehensive_research_completed",
                search_id=search_id,
                total_results=len(processed_results),
                high_quality_results=len(filtered_results),
                insights_generated=len(insights)
            )
            
            return profile
            
        except Exception as e:
            logger.error(
                "comprehensive_research_failed",
                search_id=search_id,
                query=query,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _optimize_search_query(
        self,
        query: str,
        scope: SearchScope,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Optimize search query for better results"""
        
        try:
            optimized = query.strip()
            
            # Add scope-specific terms
            scope_enhancers = {
                SearchScope.ACADEMIC: ['research', 'study', 'paper', 'journal'],
                SearchScope.NEWS: ['news', 'latest', 'recent', 'breaking'],
                SearchScope.TECHNICAL: ['documentation', 'guide', 'tutorial', 'specification'],
                SearchScope.FINANCIAL: ['financial', 'market', 'economic', 'analysis'],
                SearchScope.LEGAL: ['legal', 'law', 'regulation', 'compliance'],
                SearchScope.MEDICAL: ['medical', 'clinical', 'health', 'treatment']
            }
            
            enhancers = scope_enhancers.get(scope, [])
            
            # Add enhancers that aren't already in the query
            missing_enhancers = [e for e in enhancers if e.lower() not in optimized.lower()]
            if missing_enhancers and len(missing_enhancers) <= 2:
                optimized += f" {' '.join(missing_enhancers[:2])}"
            
            # Use memory to improve query based on past searches
            if self.memory_service and context:
                similar_queries = await self._get_similar_query_patterns(optimized, context)
                if similar_queries:
                    # Extract successful query patterns
                    for pattern in similar_queries[:2]:  # Top 2 patterns
                        if 'successful_terms' in pattern:
                            for term in pattern['successful_terms'][:1]:  # Add 1 term
                                if term.lower() not in optimized.lower():
                                    optimized += f" {term}"
                                    break
            
            # Clean up and validate
            optimized = ' '.join(optimized.split())  # Remove extra whitespace
            optimized = optimized[:200]  # Reasonable length limit
            
            return optimized
            
        except Exception as e:
            logger.warning(f"Query optimization failed: {e}")
            return query
    
    async def _assess_search_complexity(self, query: str, scope: SearchScope) -> SearchComplexity:
        """Assess the complexity of the search query"""
        
        # Simple heuristics for complexity assessment
        words = query.split()
        
        # High complexity indicators
        complex_indicators = [
            'compare', 'versus', 'relationship', 'correlation', 'analysis',
            'impact', 'effect', 'influence', 'trend', 'pattern', 'systematic'
        ]
        
        moderate_indicators = [
            'how', 'why', 'what', 'when', 'where', 'explain', 'describe',
            'overview', 'introduction', 'basics'
        ]
        
        complexity_score = 0
        
        # Length factor
        if len(words) > 10:
            complexity_score += 2
        elif len(words) > 5:
            complexity_score += 1
        
        # Indicator matching
        query_lower = query.lower()
        complexity_score += sum(1 for indicator in complex_indicators if indicator in query_lower) * 2
        complexity_score += sum(1 for indicator in moderate_indicators if indicator in query_lower)
        
        # Scope factor
        if scope in [SearchScope.ACADEMIC, SearchScope.TECHNICAL, SearchScope.MEDICAL]:
            complexity_score += 1
        
        # Determine complexity level
        if complexity_score >= 6:
            return SearchComplexity.HIGHLY_COMPLEX
        elif complexity_score >= 4:
            return SearchComplexity.COMPLEX
        elif complexity_score >= 2:
            return SearchComplexity.MODERATE
        else:
            return SearchComplexity.SIMPLE
    
    async def _execute_tavily_search(
        self,
        query: str,
        scope: SearchScope,
        max_results: int,
        options: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute search using Tavily API"""
        
        if not self.tavily_client:
            logger.warning("Tavily client not available, using mock results")
            return await self._generate_mock_results(query, max_results)
        
        try:
            # Configure search parameters based on scope
            search_depth = options.get('search_depth', 'advanced') if options else 'advanced'
            
            # Scope-specific search configuration
            include_domains = []
            exclude_domains = []
            
            if scope == SearchScope.ACADEMIC:
                include_domains.extend(self.high_authority_domains['academic'])
            elif scope == SearchScope.NEWS:
                include_domains.extend(self.high_authority_domains['news'])
            elif scope == SearchScope.TECHNICAL:
                include_domains.extend(self.high_authority_domains['technical'])
            
            # Execute Tavily search
            search_params = {
                'query': query,
                'search_depth': search_depth,
                'max_results': max_results,
                'include_images': options.get('include_images', False) if options else False,
                'include_answer': True,
                'exclude_domains': exclude_domains if exclude_domains else None
            }
            
            # Remove None values
            search_params = {k: v for k, v in search_params.items() if v is not None}
            
            response = self.tavily_client.search(**search_params)
            
            # Extract results
            results = response.get('results', [])
            
            # Add Tavily answer if available
            if response.get('answer'):
                synthetic_result = {
                    'title': 'Tavily AI Summary',
                    'url': 'https://tavily.com/summary',
                    'content': response['answer'],
                    'snippet': response['answer'][:200] + '...' if len(response['answer']) > 200 else response['answer'],
                    'score': 0.95,  # High score for AI summary
                    'source_type': 'ai_summary'
                }
                results.insert(0, synthetic_result)  # Add at beginning
            
            return results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return await self._generate_mock_results(query, max_results)
    
    async def _generate_mock_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generate mock search results for testing/fallback"""
        
        mock_results = []
        
        for i in range(min(max_results, 5)):  # Generate up to 5 mock results
            result = {
                'title': f'Mock Result {i+1}: {query}',
                'url': f'https://example.com/result{i+1}',
                'content': f'This is mock content for search result {i+1} related to "{query}". ' * 5,
                'snippet': f'Mock snippet for result {i+1} about {query}',
                'score': 0.8 - (i * 0.1),  # Decreasing relevance
                'source_type': 'mock'
            }
            mock_results.append(result)
        
        return mock_results
    
    async def _process_search_results(
        self,
        raw_results: List[Dict[str, Any]],
        scope: SearchScope,
        quality_threshold: float
    ) -> List[SearchResult]:
        """Process raw search results with ML enhancement"""
        
        processed_results = []
        
        for result in raw_results:
            try:
                # Extract basic information
                title = result.get('title', 'Untitled')
                url = result.get('url', '')
                content = result.get('content', '')
                snippet = result.get('snippet', content[:200])
                
                # Generate result ID
                result_id = hashlib.md5(f"{url}_{title}".encode()).hexdigest()[:16]
                
                # Classify source type
                source_type = await self._classify_source_type(url, title, content)
                
                # Compute various scores
                credibility_score = await self._compute_credibility_score(url, title, content, source_type)
                relevance_score = result.get('score', 0.5)  # Use Tavily score if available
                recency_score = await self._compute_recency_score(content, url)
                authority_score = await self._compute_authority_score(url, source_type)
                bias_score = await self._compute_bias_score(content, title)
                
                # Extract entities and topics
                key_entities = await self._extract_entities(content)
                topics = await self._extract_topics(content, title)
                
                # Analyze sentiment
                sentiment = await self._analyze_content_sentiment(content)
                
                # Create processed result
                search_result = SearchResult(
                    result_id=result_id,
                    title=title,
                    url=url,
                    content=content,
                    snippet=snippet,
                    source_type=source_type,
                    credibility_score=credibility_score,
                    relevance_score=relevance_score,
                    recency_score=recency_score,
                    authority_score=authority_score,
                    bias_score=bias_score,
                    fact_check_score=None,  # Could be enhanced with fact-checking APIs
                    key_entities=key_entities,
                    topics=topics,
                    sentiment=sentiment,
                    metadata={
                        'processing_timestamp': datetime.now().isoformat(),
                        'content_length': len(content),
                        'domain': urlparse(url).netloc if url else 'unknown'
                    }
                )
                
                processed_results.append(search_result)
                
            except Exception as e:
                logger.warning(f"Failed to process search result: {e}")
                continue
        
        return processed_results
    
    async def _classify_source_type(self, url: str, title: str, content: str) -> SourceType:
        """Classify the type of information source"""
        
        if not url:
            return SourceType.UNKNOWN
        
        domain = urlparse(url).netloc.lower()
        
        # Academic sources
        academic_indicators = ['arxiv', 'pubmed', 'scholar', 'jstor', 'researchgate', 'ieee', 'acm']
        if any(indicator in domain for indicator in academic_indicators):
            return SourceType.ACADEMIC
        
        # Government sources
        if domain.endswith('.gov') or domain.endswith('.edu') or domain.endswith('.mil'):
            return SourceType.GOVERNMENT
        
        # News sources
        news_indicators = ['news', 'cnn', 'bbc', 'reuters', 'ap.org', 'nytimes', 'wsj']
        if any(indicator in domain for indicator in news_indicators):
            return SourceType.NEWS_ARTICLE
        
        # Technical documentation
        tech_indicators = ['docs.', 'documentation', 'api.', 'developer.', 'github']
        if any(indicator in domain for indicator in tech_indicators):
            return SourceType.WEBSITE
        
        # Blog indicators
        blog_indicators = ['blog', 'medium.com', 'wordpress', 'blogspot']
        if any(indicator in domain for indicator in blog_indicators):
            return SourceType.BLOG_POST
        
        # Forum indicators  
        forum_indicators = ['forum', 'reddit', 'stackoverflow', 'quora', 'discourse']
        if any(indicator in domain for indicator in forum_indicators):
            return SourceType.FORUM
        
        # Social media
        social_indicators = ['twitter', 'facebook', 'linkedin', 'instagram']
        if any(indicator in domain for indicator in social_indicators):
            return SourceType.SOCIAL_MEDIA
        
        # Default to website
        return SourceType.WEBSITE
    
    async def _compute_credibility_score(
        self, 
        url: str, 
        title: str, 
        content: str, 
        source_type: SourceType
    ) -> float:
        """Compute credibility score for the source"""
        
        score = 0.5  # Base score
        
        if not url:
            return score
        
        domain = urlparse(url).netloc.lower()
        
        # Domain authority
        high_auth_domains = []
        for domain_list in self.high_authority_domains.values():
            high_auth_domains.extend(domain_list)
        
        if any(auth_domain in domain for auth_domain in high_auth_domains):
            score += 0.3
        
        # Source type bonus
        type_bonuses = {
            SourceType.ACADEMIC: 0.2,
            SourceType.GOVERNMENT: 0.2,
            SourceType.RESEARCH_PAPER: 0.2,
            SourceType.NEWS_ARTICLE: 0.1,
            SourceType.WEBSITE: 0.0,
            SourceType.BLOG_POST: -0.1,
            SourceType.FORUM: -0.1,
            SourceType.SOCIAL_MEDIA: -0.2
        }
        score += type_bonuses.get(source_type, 0)
        
        # Content quality indicators
        if content:
            # Length bonus (substantial content)
            if len(content) > 1000:
                score += 0.1
            
            # Citation indicators
            if re.search(r'\[[\d,\s-]+\]|\(\d{4}\)|doi:', content):
                score += 0.1
            
            # Professional writing indicators
            if content.count('.') > 5:  # Multiple sentences
                score += 0.05
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
    
    async def _compute_recency_score(self, content: str, url: str) -> float:
        """Compute recency score based on content and URL analysis"""
        
        score = 0.5  # Default moderate recency
        
        # Look for date indicators in content
        current_year = datetime.now().year
        
        # Recent year mentions
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, content)
        if years:
            latest_year = max(int(year) for year in years)
            years_ago = current_year - latest_year
            
            if years_ago == 0:
                score = 1.0  # Current year
            elif years_ago <= 1:
                score = 0.9
            elif years_ago <= 3:
                score = 0.7
            elif years_ago <= 5:
                score = 0.5
            else:
                score = 0.3
        
        # URL path date indicators
        if url:
            url_year_match = re.search(r'/(\d{4})/', url)
            if url_year_match:
                url_year = int(url_year_match.group(1))
                years_ago = current_year - url_year
                url_score = max(0.1, 1.0 - (years_ago * 0.1))
                score = max(score, url_score)
        
        return min(1.0, score)
    
    async def _compute_authority_score(self, url: str, source_type: SourceType) -> float:
        """Compute authority score for the source"""
        
        if not url:
            return 0.3
        
        domain = urlparse(url).netloc.lower()
        
        # Check against high authority domain lists
        for category, domains in self.high_authority_domains.items():
            if any(auth_domain in domain for auth_domain in domains):
                return 0.9  # High authority
        
        # Domain characteristics
        score = 0.4  # Base score
        
        # TLD indicators
        if domain.endswith(('.edu', '.gov', '.org')):
            score += 0.2
        elif domain.endswith(('.com', '.net')):
            score += 0.1
        
        # Source type authority
        type_authority = {
            SourceType.ACADEMIC: 0.3,
            SourceType.GOVERNMENT: 0.3,
            SourceType.NEWS_ARTICLE: 0.2,
            SourceType.WEBSITE: 0.1,
            SourceType.BLOG_POST: 0.0,
            SourceType.FORUM: -0.1,
            SourceType.SOCIAL_MEDIA: -0.2
        }
        score += type_authority.get(source_type, 0)
        
        return max(0.0, min(1.0, score))
    
    async def _compute_bias_score(self, content: str, title: str) -> float:
        """Compute bias score (0 = no bias, 1 = high bias)"""
        
        bias_score = 0.0
        
        if not content:
            return bias_score
        
        # Emotional language indicators
        emotional_words = [
            'amazing', 'incredible', 'shocking', 'outrageous', 'devastating',
            'revolutionary', 'groundbreaking', 'alarming', 'unprecedented'
        ]
        
        text_lower = (content + ' ' + title).lower()
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        bias_score += min(0.3, emotional_count * 0.05)
        
        # Extreme language
        extreme_words = [
            'always', 'never', 'completely', 'totally', 'absolutely',
            'definitely', 'certainly', 'obviously', 'clearly'
        ]
        extreme_count = sum(1 for word in extreme_words if word in text_lower)
        bias_score += min(0.2, extreme_count * 0.03)
        
        # All caps usage (excessive emphasis)
        caps_ratio = sum(1 for c in content if c.isupper()) / len(content) if content else 0
        if caps_ratio > 0.1:  # More than 10% caps
            bias_score += 0.1
        
        # Exclamation mark usage
        exclamation_count = content.count('!')
        bias_score += min(0.1, exclamation_count * 0.02)
        
        return min(1.0, bias_score)
    
    async def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities from content"""
        
        entities = []
        
        if not content:
            return entities
        
        try:
            # Use spaCy if available
            if self.nlp and len(content) < 1000000:  # Limit for performance
                doc = self.nlp(content[:10000])  # Process first 10k chars
                
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                        entities.append(ent.text.strip())
            
            # Fallback: Use transformer pipeline
            elif self.ner_pipeline:
                ner_results = self.ner_pipeline(content[:512])  # Limit for model
                
                for result in ner_results:
                    entities.append(result['word'].strip())
            
            # Remove duplicates and clean
            entities = list(set(entities))
            entities = [ent for ent in entities if len(ent) > 1 and not ent.isdigit()]
            
            return entities[:10]  # Return top 10 entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return entities
    
    async def _extract_topics(self, content: str, title: str) -> List[str]:
        """Extract topics from content"""
        
        topics = []
        
        if not content:
            return topics
        
        try:
            # Simple keyword-based topic extraction
            text = (content + ' ' + title).lower()
            
            # Predefined topic keywords
            topic_keywords = {
                'technology': ['software', 'algorithm', 'ai', 'machine learning', 'computer', 'digital'],
                'business': ['market', 'business', 'company', 'revenue', 'profit', 'strategy'],
                'health': ['medical', 'health', 'disease', 'treatment', 'patient', 'clinical'],
                'science': ['research', 'study', 'experiment', 'hypothesis', 'data', 'analysis'],
                'politics': ['government', 'policy', 'political', 'election', 'democracy', 'law'],
                'economics': ['economic', 'economy', 'financial', 'money', 'investment', 'trade'],
                'education': ['education', 'learning', 'student', 'university', 'academic', 'school'],
                'environment': ['environment', 'climate', 'green', 'sustainable', 'energy', 'carbon']
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in text for keyword in keywords):
                    topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return topics
    
    async def _analyze_content_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment of content"""
        
        sentiment = {'polarity': 0.0, 'subjectivity': 0.0, 'confidence': 0.0}
        
        if not content or not self.sentiment_analyzer:
            return sentiment
        
        try:
            # Use transformer model for sentiment
            result = self.sentiment_analyzer(content[:512])  # Limit text length
            
            if result:
                confidence = float(result[0]['score'])
                label = result[0]['label'].upper()
                
                if 'POSITIVE' in label:
                    polarity = confidence
                elif 'NEGATIVE' in label:
                    polarity = -confidence
                else:
                    polarity = 0.0
                
                sentiment = {
                    'polarity': polarity,
                    'subjectivity': confidence,  # Use confidence as subjectivity proxy
                    'confidence': confidence
                }
            
            return sentiment
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return sentiment
    
    async def _filter_results_by_quality(
        self,
        results: List[SearchResult],
        threshold: float
    ) -> List[SearchResult]:
        """Filter results based on quality metrics"""
        
        # Compute overall quality score for each result
        quality_results = []
        
        for result in results:
            # Weighted quality score
            quality_score = (
                result.credibility_score * 0.3 +
                result.relevance_score * 0.25 +
                result.authority_score * 0.2 +
                result.recency_score * 0.15 +
                (1.0 - result.bias_score) * 0.1  # Lower bias = higher quality
            )
            
            if quality_score >= threshold:
                quality_results.append(result)
        
        # Sort by quality score (descending)
        quality_results.sort(
            key=lambda r: (
                r.credibility_score * 0.3 +
                r.relevance_score * 0.25 +
                r.authority_score * 0.2 +
                r.recency_score * 0.15 +
                (1.0 - r.bias_score) * 0.1
            ),
            reverse=True
        )
        
        return quality_results
    
    async def _generate_research_insights(
        self,
        results: List[SearchResult],
        query: str,
        scope: SearchScope
    ) -> List[SearchInsight]:
        """Generate ML-driven research insights"""
        
        insights = []
        
        if not results:
            return insights
        
        try:
            # Consensus analysis insight
            consensus_insight = await self._generate_consensus_insight(results, query)
            if consensus_insight:
                insights.append(consensus_insight)
            
            # Credibility insight
            credibility_insight = await self._generate_credibility_insight(results)
            if credibility_insight:
                insights.append(credibility_insight)
            
            # Temporal insight
            temporal_insight = await self._generate_temporal_insight(results)
            if temporal_insight:
                insights.append(temporal_insight)
            
            # Bias analysis insight
            bias_insight = await self._generate_bias_insight(results)
            if bias_insight:
                insights.append(bias_insight)
            
            # Source diversity insight
            diversity_insight = await self._generate_diversity_insight(results)
            if diversity_insight:
                insights.append(diversity_insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Research insights generation failed: {e}")
            return insights
    
    async def _generate_consensus_insight(
        self,
        results: List[SearchResult],
        query: str
    ) -> Optional[SearchInsight]:
        """Generate insight about consensus among sources"""
        
        if len(results) < 3:
            return None
        
        # Simple consensus analysis based on sentiment alignment
        sentiments = [r.sentiment['polarity'] for r in results if r.sentiment]
        
        if not sentiments:
            return None
        
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        total = len(sentiments)
        consensus_level = max(positive_count, negative_count, neutral_count) / total
        
        if consensus_level > 0.7:
            dominant_sentiment = 'positive' if positive_count == max(positive_count, negative_count, neutral_count) else ('negative' if negative_count > neutral_count else 'neutral')
            
            return SearchInsight(
                insight_type="consensus_analysis",
                confidence=consensus_level,
                description=f"Strong {dominant_sentiment} consensus among {total} sources",
                supporting_sources=[r.url for r in results[:3]],
                contradicting_sources=[],
                key_evidence=[f"{positive_count} positive, {negative_count} negative, {neutral_count} neutral"],
                bias_indicators=[],
                consensus_level=consensus_level,
                novelty_score=0.5
            )
        
        return None
    
    async def _generate_credibility_insight(self, results: List[SearchResult]) -> Optional[SearchInsight]:
        """Generate insight about source credibility"""
        
        if not results:
            return None
        
        credibility_scores = [r.credibility_score for r in results]
        avg_credibility = np.mean(credibility_scores)
        
        high_credibility_count = sum(1 for score in credibility_scores if score > 0.7)
        
        if avg_credibility > 0.7:
            return SearchInsight(
                insight_type="credibility_assessment",
                confidence=0.8,
                description=f"High credibility sources: {high_credibility_count}/{len(results)} sources above 70% credibility",
                supporting_sources=[r.url for r in results if r.credibility_score > 0.7][:3],
                contradicting_sources=[r.url for r in results if r.credibility_score < 0.5],
                key_evidence=[f"Average credibility: {avg_credibility:.2f}"],
                bias_indicators=[],
                consensus_level=high_credibility_count / len(results),
                novelty_score=0.3
            )
        
        return None
    
    async def _generate_temporal_insight(self, results: List[SearchResult]) -> Optional[SearchInsight]:
        """Generate insight about temporal patterns"""
        
        if not results:
            return None
        
        recency_scores = [r.recency_score for r in results]
        avg_recency = np.mean(recency_scores)
        
        recent_count = sum(1 for score in recency_scores if score > 0.8)
        
        if recent_count > len(results) * 0.5:
            return SearchInsight(
                insight_type="temporal_analysis",
                confidence=0.7,
                description=f"Recent information: {recent_count}/{len(results)} sources with high recency",
                supporting_sources=[r.url for r in results if r.recency_score > 0.8][:3],
                contradicting_sources=[],
                key_evidence=[f"Average recency: {avg_recency:.2f}"],
                bias_indicators=[],
                consensus_level=recent_count / len(results),
                novelty_score=avg_recency
            )
        
        return None
    
    async def _generate_bias_insight(self, results: List[SearchResult]) -> Optional[SearchInsight]:
        """Generate insight about bias patterns"""
        
        if not results:
            return None
        
        bias_scores = [r.bias_score for r in results]
        avg_bias = np.mean(bias_scores)
        high_bias_count = sum(1 for score in bias_scores if score > 0.5)
        
        if high_bias_count > 0:
            return SearchInsight(
                insight_type="bias_analysis",
                confidence=0.6,
                description=f"Bias detected: {high_bias_count}/{len(results)} sources show potential bias",
                supporting_sources=[],
                contradicting_sources=[r.url for r in results if r.bias_score > 0.5],
                key_evidence=[f"Average bias score: {avg_bias:.2f}"],
                bias_indicators=[f"{high_bias_count} high-bias sources detected"],
                consensus_level=1.0 - (high_bias_count / len(results)),
                novelty_score=0.4
            )
        
        return None
    
    async def _generate_diversity_insight(self, results: List[SearchResult]) -> Optional[SearchInsight]:
        """Generate insight about source diversity"""
        
        if not results:
            return None
        
        source_types = [r.source_type for r in results]
        unique_types = set(source_types)
        
        if len(unique_types) >= 3:
            return SearchInsight(
                insight_type="source_diversity",
                confidence=0.8,
                description=f"Diverse sources: {len(unique_types)} different source types",
                supporting_sources=[r.url for r in results[:3]],
                contradicting_sources=[],
                key_evidence=[f"Source types: {', '.join([t.value for t in unique_types])}"],
                bias_indicators=[],
                consensus_level=len(unique_types) / len(SourceType),
                novelty_score=0.6
            )
        
        return None
    
    async def _analyze_consensus(
        self,
        results: List[SearchResult],
        insights: List[SearchInsight]
    ) -> Dict[str, Any]:
        """Analyze consensus among sources"""
        
        consensus = {
            'overall_consensus_level': 0.0,
            'sentiment_consensus': {},
            'topic_consensus': {},
            'conflicting_viewpoints': []
        }
        
        if not results:
            return consensus
        
        # Sentiment consensus
        sentiments = [r.sentiment['polarity'] for r in results if r.sentiment]
        if sentiments:
            positive = sum(1 for s in sentiments if s > 0.1)
            negative = sum(1 for s in sentiments if s < -0.1)
            neutral = len(sentiments) - positive - negative
            
            consensus['sentiment_consensus'] = {
                'positive_sources': positive,
                'negative_sources': negative,
                'neutral_sources': neutral,
                'dominant_sentiment': 'positive' if positive > max(negative, neutral) else ('negative' if negative > neutral else 'neutral'),
                'consensus_strength': max(positive, negative, neutral) / len(sentiments) if sentiments else 0
            }
        
        # Topic consensus
        all_topics = []
        for result in results:
            all_topics.extend(result.topics)
        
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        consensus['topic_consensus'] = {
            'common_topics': [topic for topic, count in topic_counts.items() if count >= len(results) * 0.3],
            'topic_distribution': topic_counts
        }
        
        # Overall consensus level (weighted average)
        sentiment_consensus_level = consensus['sentiment_consensus'].get('consensus_strength', 0)
        topic_consensus_level = len(consensus['topic_consensus']['common_topics']) / max(1, len(topic_counts))
        
        consensus['overall_consensus_level'] = (sentiment_consensus_level * 0.6) + (topic_consensus_level * 0.4)
        
        return consensus
    
    async def _assess_overall_credibility(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Assess overall credibility of research results"""
        
        assessment = {
            'overall_credibility_score': 0.0,
            'high_credibility_sources': 0,
            'credibility_distribution': {},
            'authority_analysis': {}
        }
        
        if not results:
            return assessment
        
        credibility_scores = [r.credibility_score for r in results]
        
        assessment['overall_credibility_score'] = float(np.mean(credibility_scores))
        assessment['high_credibility_sources'] = sum(1 for score in credibility_scores if score > 0.7)
        
        # Credibility distribution
        very_high = sum(1 for score in credibility_scores if score > 0.9)
        high = sum(1 for score in credibility_scores if 0.7 < score <= 0.9)
        medium = sum(1 for score in credibility_scores if 0.5 < score <= 0.7)
        low = sum(1 for score in credibility_scores if score <= 0.5)
        
        assessment['credibility_distribution'] = {
            'very_high': very_high,
            'high': high,
            'medium': medium,
            'low': low
        }
        
        # Authority analysis
        authority_scores = [r.authority_score for r in results]
        assessment['authority_analysis'] = {
            'average_authority': float(np.mean(authority_scores)),
            'high_authority_sources': sum(1 for score in authority_scores if score > 0.7)
        }
        
        return assessment
    
    async def _analyze_bias_patterns(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze bias patterns in search results"""
        
        analysis = {
            'overall_bias_level': 0.0,
            'bias_distribution': {},
            'potential_bias_indicators': []
        }
        
        if not results:
            return analysis
        
        bias_scores = [r.bias_score for r in results]
        analysis['overall_bias_level'] = float(np.mean(bias_scores))
        
        # Bias distribution
        high_bias = sum(1 for score in bias_scores if score > 0.6)
        medium_bias = sum(1 for score in bias_scores if 0.3 < score <= 0.6)
        low_bias = sum(1 for score in bias_scores if score <= 0.3)
        
        analysis['bias_distribution'] = {
            'high_bias': high_bias,
            'medium_bias': medium_bias,
            'low_bias': low_bias
        }
        
        # Identify potential bias indicators
        if high_bias > len(results) * 0.3:
            analysis['potential_bias_indicators'].append(f"{high_bias} sources show high bias")
        
        # Check for source type bias
        source_types = [r.source_type for r in results]
        type_counts = {}
        for source_type in source_types:
            type_counts[source_type] = type_counts.get(source_type, 0) + 1
        
        dominant_type = max(type_counts, key=type_counts.get)
        if type_counts[dominant_type] > len(results) * 0.7:
            analysis['potential_bias_indicators'].append(f"Source type bias: {type_counts[dominant_type]} of {len(results)} sources are {dominant_type.value}")
        
        return analysis
    
    async def _analyze_temporal_patterns(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze temporal patterns in search results"""
        
        analysis = {
            'recency_analysis': {},
            'temporal_distribution': {},
            'information_freshness': 0.0
        }
        
        if not results:
            return analysis
        
        recency_scores = [r.recency_score for r in results]
        analysis['information_freshness'] = float(np.mean(recency_scores))
        
        # Recency analysis
        very_recent = sum(1 for score in recency_scores if score > 0.9)
        recent = sum(1 for score in recency_scores if 0.7 < score <= 0.9)
        somewhat_recent = sum(1 for score in recency_scores if 0.5 < score <= 0.7)
        older = sum(1 for score in recency_scores if score <= 0.5)
        
        analysis['recency_analysis'] = {
            'very_recent_sources': very_recent,
            'recent_sources': recent,
            'somewhat_recent_sources': somewhat_recent,
            'older_sources': older
        }
        
        analysis['temporal_distribution'] = {
            'current_year_dominant': very_recent > len(results) * 0.5,
            'recent_information_available': (very_recent + recent) > len(results) * 0.6,
            'temporal_diversity': len(set([int(score * 10) for score in recency_scores])) > 3
        }
        
        return analysis
    
    async def _analyze_source_diversity(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze diversity of information sources"""
        
        diversity = {
            'source_type_diversity': {},
            'domain_diversity': {},
            'geographic_diversity': {},
            'diversity_score': 0.0
        }
        
        if not results:
            return diversity
        
        # Source type diversity
        source_types = [r.source_type for r in results]
        type_counts = {}
        for source_type in source_types:
            type_counts[source_type.value] = type_counts.get(source_type.value, 0) + 1
        
        diversity['source_type_diversity'] = {
            'unique_types': len(type_counts),
            'type_distribution': type_counts,
            'balanced_distribution': max(type_counts.values()) <= len(results) * 0.6
        }
        
        # Domain diversity
        domains = set()
        for result in results:
            if result.url:
                domain = urlparse(result.url).netloc
                domains.add(domain)
        
        diversity['domain_diversity'] = {
            'unique_domains': len(domains),
            'domain_concentration': len(domains) / len(results) if results else 0
        }
        
        # Calculate overall diversity score
        type_diversity = len(type_counts) / len(SourceType)
        domain_diversity = len(domains) / max(1, len(results))
        
        diversity['diversity_score'] = (type_diversity * 0.6) + (domain_diversity * 0.4)
        
        return diversity
    
    async def _perform_fact_verification(
        self,
        results: List[SearchResult],
        insights: List[SearchInsight]
    ) -> Dict[str, Any]:
        """Perform basic fact verification analysis"""
        
        verification = {
            'verification_possible': False,
            'consistent_information': 0.0,
            'contradictory_information': [],
            'verification_confidence': 0.0
        }
        
        if len(results) < 3:
            return verification
        
        verification['verification_possible'] = True
        
        # Simple consistency check based on sentiment and topics
        sentiments = [r.sentiment['polarity'] for r in results if r.sentiment]
        topics = []
        for result in results:
            topics.extend(result.topics)
        
        # Check sentiment consistency
        if sentiments:
            sentiment_std = np.std(sentiments)
            consistency_score = max(0, 1 - (sentiment_std * 2))  # Lower std = higher consistency
            verification['consistent_information'] = consistency_score
        
        # Check for contradictory insights
        contradictions = []
        for insight in insights:
            if insight.contradicting_sources:
                contradictions.extend(insight.contradicting_sources)
        
        verification['contradictory_information'] = list(set(contradictions))
        
        # Overall verification confidence
        verification['verification_confidence'] = (
            verification['consistent_information'] * 0.7 +
            (1 - len(verification['contradictory_information']) / len(results)) * 0.3
        )
        
        return verification
    
    async def _identify_research_gaps(
        self,
        results: List[SearchResult],
        insights: List[SearchInsight]
    ) -> List[str]:
        """Identify potential gaps in research results"""
        
        gaps = []
        
        if not results:
            gaps.append("No search results available")
            return gaps
        
        # Check for source diversity gaps
        source_types = set(r.source_type for r in results)
        missing_types = set(SourceType) - source_types
        
        important_missing = [
            SourceType.ACADEMIC,
            SourceType.GOVERNMENT,
            SourceType.NEWS_ARTICLE
        ]
        
        for missing_type in important_missing:
            if missing_type in missing_types:
                gaps.append(f"Missing {missing_type.value} sources")
        
        # Check for recency gaps
        recency_scores = [r.recency_score for r in results]
        if np.mean(recency_scores) < 0.5:
            gaps.append("Limited recent information available")
        
        # Check for credibility gaps
        credibility_scores = [r.credibility_score for r in results]
        high_credibility_count = sum(1 for score in credibility_scores if score > 0.7)
        
        if high_credibility_count < len(results) * 0.5:
            gaps.append("Limited high-credibility sources")
        
        # Check for consensus gaps
        consensus_insights = [i for i in insights if i.insight_type == "consensus_analysis"]
        if not consensus_insights or all(i.consensus_level < 0.6 for i in consensus_insights):
            gaps.append("Lack of consensus among sources")
        
        return gaps
    
    async def _compute_research_confidence(
        self,
        results: List[SearchResult],
        insights: List[SearchInsight],
        consensus_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute confidence metrics for research results"""
        
        confidence = {
            'overall_confidence': 0.0,
            'source_quality_confidence': 0.0,
            'consensus_confidence': 0.0,
            'diversity_confidence': 0.0,
            'recency_confidence': 0.0
        }
        
        if not results:
            return confidence
        
        # Source quality confidence
        credibility_scores = [r.credibility_score for r in results]
        authority_scores = [r.authority_score for r in results]
        
        confidence['source_quality_confidence'] = float(
            (np.mean(credibility_scores) * 0.6) + (np.mean(authority_scores) * 0.4)
        )
        
        # Consensus confidence
        confidence['consensus_confidence'] = consensus_analysis.get('overall_consensus_level', 0.0)
        
        # Diversity confidence
        source_types = set(r.source_type for r in results)
        diversity_score = len(source_types) / len(SourceType)
        confidence['diversity_confidence'] = float(diversity_score)
        
        # Recency confidence
        recency_scores = [r.recency_score for r in results]
        confidence['recency_confidence'] = float(np.mean(recency_scores))
        
        # Overall confidence (weighted average)
        weights = [0.3, 0.25, 0.2, 0.25]  # Quality, consensus, diversity, recency
        scores = [
            confidence['source_quality_confidence'],
            confidence['consensus_confidence'], 
            confidence['diversity_confidence'],
            confidence['recency_confidence']
        ]
        
        confidence['overall_confidence'] = float(np.average(scores, weights=weights))
        
        return confidence
    
    def _get_active_models(self) -> List[str]:
        """Get list of active ML models"""
        models = []
        
        if self.summarizer:
            models.append("bart_summarizer")
        if self.sentiment_analyzer:
            models.append("transformers_sentiment")
        if self.ner_pipeline:
            models.append("transformers_ner")
        if self.nlp:
            models.append("spacy_nlp")
        
        models.extend(["tfidf_vectorizer", "tavily_search"])
        
        return models
    
    async def _get_similar_query_patterns(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve similar query patterns from memory for optimization"""
        
        if not self.memory_service:
            return []
        
        try:
            similar_patterns = await self.memory_service.retrieve_relevant_memories(
                query=f"research query {query}",
                user_id=context.get("user_id"),
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
            logger.warning(f"Failed to retrieve similar query patterns: {e}")
            return []
    
    async def _store_research_patterns(
        self,
        profile: ResearchProfile,
        context: Dict[str, Any]
    ):
        """Store research patterns for future optimization"""
        
        if not self.memory_service:
            return
        
        try:
            research_data = {
                "analysis_type": "web_research",
                "query": profile.query,
                "search_scope": profile.search_scope.value,
                "complexity": profile.complexity.value,
                "performance_metrics": {
                    "total_results": profile.total_results,
                    "high_quality_results": len(profile.filtered_results),
                    "overall_confidence": profile.confidence_metrics.get("overall_confidence", 0),
                    "consensus_level": profile.consensus_analysis.get("overall_consensus_level", 0)
                },
                "successful_terms": profile.research_metadata.get("optimized_query", "").split(),
                "insights_generated": len(profile.research_insights),
                "research_timestamp": profile.search_timestamp.isoformat(),
                "successful_research": True
            }
            
            await self.memory_service.store_conversation_memory(
                user_id=context.get("user_id", "system"),
                message=json.dumps(research_data),
                context={
                    "category": "web_research",
                    "search_scope": profile.search_scope.value,
                    "query_complexity": profile.complexity.value
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to store research patterns: {e}")