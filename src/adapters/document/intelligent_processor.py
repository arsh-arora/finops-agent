"""
Intelligent Document Processing Engine
Advanced document analysis with Docling bounding box extraction and ML-driven content understanding
"""

import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
import json
import re
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import mimetypes
from io import BytesIO
import base64

# Document processing libraries
from docling.document_converter import DocumentConverter
from docling.datamodel.document import ConversionResult
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import cv2

# ML libraries for content analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import spacy
import structlog

logger = structlog.get_logger(__name__)


class DocumentType(Enum):
    """Document types with specialized processing"""
    PDF = "pdf"
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Content classification types"""
    FINANCIAL_REPORT = "financial_report"
    LEGAL_DOCUMENT = "legal_document"
    TECHNICAL_SPEC = "technical_specification"
    RESEARCH_PAPER = "research_paper"
    BUSINESS_PLAN = "business_plan"
    CONTRACT = "contract"
    INVOICE = "invoice"
    RESUME = "resume"
    PRESENTATION = "presentation"
    MANUAL = "manual"
    GENERIC = "generic"


class ProcessingComplexity(Enum):
    """Document processing complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class BoundingBox:
    """Bounding box with coordinate and metadata"""
    x: float
    y: float
    width: float
    height: float
    page: int
    confidence: float
    element_type: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class DocumentElement:
    """Document element with semantic understanding"""
    element_id: str
    element_type: str
    content: str
    bounding_box: BoundingBox
    confidence: float
    semantic_label: str
    relationships: List[str]
    properties: Dict[str, Any]


@dataclass
class ContentInsight:
    """ML-driven content insights"""
    insight_type: str
    confidence: float
    description: str
    key_entities: List[str]
    sentiment_score: Optional[float]
    topic_labels: List[str]
    importance_score: float
    supporting_evidence: List[str]


@dataclass
class DocumentAnalysisProfile:
    """Comprehensive document analysis results"""
    document_id: str
    document_type: DocumentType
    content_type: ContentType
    processing_complexity: ProcessingComplexity
    total_pages: int
    total_elements: int
    
    text_content: str
    structured_elements: List[DocumentElement]
    content_insights: List[ContentInsight]
    
    key_entities: Dict[str, List[str]]
    topic_modeling: Dict[str, Any]
    sentiment_analysis: Dict[str, float]
    readability_metrics: Dict[str, float]
    
    bounding_box_summary: Dict[str, Any]
    quality_assessment: Dict[str, float]
    processing_metadata: Dict[str, Any]
    confidence_metrics: Dict[str, float]


class IntelligentDocumentProcessor:
    """
    Advanced document processing engine with Docling integration and ML-driven analysis
    """
    
    def __init__(self, memory_service=None):
        self.memory_service = memory_service
        self.processing_cache: Dict[str, Any] = {}
        
        # Initialize Docling converter
        self.docling_converter = DocumentConverter()
        
        # Initialize ML models
        self.text_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.topic_model = LatentDirichletAllocation(n_components=10, random_state=42)
        
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic NLP")
            self.nlp = None
        
        # Initialize Transformers models
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
            self.ner_pipeline = pipeline("ner", aggregation_strategy="simple")
        except Exception as e:
            logger.warning(f"Failed to initialize transformer models: {e}")
    
    async def process_document_comprehensive(
        self,
        document_path: Union[str, Path, BytesIO],
        document_id: Optional[str] = None,
        processing_options: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DocumentAnalysisProfile:
        """
        Comprehensive document processing with intelligent analysis
        """
        if not document_id:
            if isinstance(document_path, (str, Path)):
                document_id = hashlib.md5(str(document_path).encode()).hexdigest()
            else:
                document_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()
        
        logger.info(
            "document_processing_started",
            document_id=document_id,
            processing_options=processing_options or {}
        )
        
        try:
            # Determine document type and complexity
            doc_type, complexity = await self._analyze_document_characteristics(
                document_path, processing_options
            )
            
            # Execute Docling conversion with bounding box extraction
            conversion_result = await self._execute_docling_conversion(
                document_path, doc_type, processing_options
            )
            
            # Extract structured content and bounding boxes
            structured_elements = await self._extract_structured_elements(
                conversion_result, doc_type
            )
            
            # Perform ML-driven content analysis
            content_analysis = await self._perform_content_analysis(
                conversion_result, structured_elements
            )
            
            # Generate comprehensive insights
            insights = await self._generate_content_insights(
                content_analysis, structured_elements, context
            )
            
            # Assess document quality and confidence
            quality_metrics = await self._assess_document_quality(
                conversion_result, structured_elements
            )
            
            # Create comprehensive analysis profile
            profile = DocumentAnalysisProfile(
                document_id=document_id,
                document_type=doc_type,
                content_type=content_analysis["content_type"],
                processing_complexity=complexity,
                total_pages=conversion_result.pages if hasattr(conversion_result, 'pages') else 1,
                total_elements=len(structured_elements),
                text_content=content_analysis["full_text"],
                structured_elements=structured_elements,
                content_insights=insights,
                key_entities=content_analysis["entities"],
                topic_modeling=content_analysis["topics"],
                sentiment_analysis=content_analysis["sentiment"],
                readability_metrics=content_analysis["readability"],
                bounding_box_summary=await self._summarize_bounding_boxes(structured_elements),
                quality_assessment=quality_metrics,
                processing_metadata={
                    "processing_time": datetime.now().isoformat(),
                    "docling_version": "latest",
                    "ml_models_used": self._get_active_models(),
                    "processing_options": processing_options or {}
                },
                confidence_metrics=await self._compute_confidence_metrics(
                    quality_metrics, len(structured_elements)
                )
            )
            
            # Store processing patterns for optimization
            if self.memory_service and context:
                await self._store_processing_patterns(profile, context)
            
            logger.info(
                "document_processing_completed",
                document_id=document_id,
                document_type=doc_type.value,
                total_elements=len(structured_elements),
                content_type=content_analysis["content_type"].value
            )
            
            return profile
            
        except Exception as e:
            logger.error(
                "document_processing_failed",
                document_id=document_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    async def _analyze_document_characteristics(
        self,
        document_path: Union[str, Path, BytesIO],
        options: Optional[Dict[str, Any]]
    ) -> Tuple[DocumentType, ProcessingComplexity]:
        """Analyze document type and processing complexity"""
        
        try:
            if isinstance(document_path, BytesIO):
                # For byte streams, try to detect from magic bytes
                document_path.seek(0)
                magic_bytes = document_path.read(16)
                document_path.seek(0)
                
                if magic_bytes.startswith(b'%PDF'):
                    doc_type = DocumentType.PDF
                elif magic_bytes.startswith(b'PK'):  # ZIP-based formats
                    doc_type = DocumentType.WORD  # Could be DOCX, XLSX, PPTX
                else:
                    doc_type = DocumentType.UNKNOWN
            else:
                # Determine type from file extension
                path = Path(document_path)
                mime_type, _ = mimetypes.guess_type(str(path))
                
                if mime_type:
                    if 'pdf' in mime_type:
                        doc_type = DocumentType.PDF
                    elif 'word' in mime_type or 'officedocument.wordprocessingml' in mime_type:
                        doc_type = DocumentType.WORD
                    elif 'excel' in mime_type or 'officedocument.spreadsheetml' in mime_type:
                        doc_type = DocumentType.EXCEL
                    elif 'powerpoint' in mime_type or 'officedocument.presentationml' in mime_type:
                        doc_type = DocumentType.POWERPOINT
                    elif 'text' in mime_type:
                        doc_type = DocumentType.TEXT
                    elif 'html' in mime_type:
                        doc_type = DocumentType.HTML
                    elif 'image' in mime_type:
                        doc_type = DocumentType.IMAGE
                    else:
                        doc_type = DocumentType.UNKNOWN
                else:
                    # Fallback to extension
                    ext = path.suffix.lower()
                    ext_mapping = {
                        '.pdf': DocumentType.PDF,
                        '.docx': DocumentType.WORD,
                        '.doc': DocumentType.WORD,
                        '.xlsx': DocumentType.EXCEL,
                        '.xls': DocumentType.EXCEL,
                        '.pptx': DocumentType.POWERPOINT,
                        '.ppt': DocumentType.POWERPOINT,
                        '.txt': DocumentType.TEXT,
                        '.html': DocumentType.HTML,
                        '.md': DocumentType.MARKDOWN,
                        '.jpg': DocumentType.IMAGE,
                        '.jpeg': DocumentType.IMAGE,
                        '.png': DocumentType.IMAGE
                    }
                    doc_type = ext_mapping.get(ext, DocumentType.UNKNOWN)
            
            # Determine processing complexity
            complexity = ProcessingComplexity.MODERATE  # Default
            
            if doc_type == DocumentType.PDF:
                # For PDFs, check if processing options suggest complexity
                if options and options.get('extract_tables', False):
                    complexity = ProcessingComplexity.COMPLEX
                elif options and options.get('ocr_required', False):
                    complexity = ProcessingComplexity.HIGHLY_COMPLEX
            elif doc_type == DocumentType.IMAGE:
                complexity = ProcessingComplexity.COMPLEX  # OCR required
            elif doc_type == DocumentType.TEXT:
                complexity = ProcessingComplexity.SIMPLE
            
            return doc_type, complexity
            
        except Exception as e:
            logger.warning(f"Failed to analyze document characteristics: {e}")
            return DocumentType.UNKNOWN, ProcessingComplexity.MODERATE
    
    async def _execute_docling_conversion(
        self,
        document_path: Union[str, Path, BytesIO],
        doc_type: DocumentType,
        options: Optional[Dict[str, Any]]
    ) -> ConversionResult:
        """Execute Docling conversion with optimized settings"""
        
        try:
            # Use simplified Docling API - converter auto-configures based on document type
            converter = DocumentConverter()
            
            # Perform conversion
            if isinstance(document_path, BytesIO):
                # Handle byte stream
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(document_path.read())
                    tmp_path = tmp_file.name
                
                try:
                    conversion_result = converter.convert(tmp_path)
                finally:
                    Path(tmp_path).unlink()  # Clean up temp file
            else:
                conversion_result = converter.convert(str(document_path))
            
            return conversion_result
            
        except Exception as e:
            logger.error(f"Docling conversion failed: {e}")
            # Fallback conversion attempt
            try:
                return converter.convert(str(document_path))
            except Exception as fallback_error:
                logger.error(f"Fallback conversion also failed: {fallback_error}")
                raise
    
    async def _extract_structured_elements(
        self,
        conversion_result: ConversionResult,
        doc_type: DocumentType
    ) -> List[DocumentElement]:
        """Extract structured elements with bounding boxes from Docling result"""
        
        elements = []
        
        try:
            # Extract elements from Docling document structure
            doc = conversion_result.document
            
            if hasattr(doc, 'body') and doc.body:
                for page_num, page in enumerate(doc.body):
                    for element in page:
                        # Create bounding box from Docling element
                        bbox = None
                        if hasattr(element, 'prov') and element.prov:
                            for prov in element.prov:
                                if hasattr(prov, 'bbox') and prov.bbox:
                                    bbox = BoundingBox(
                                        x=prov.bbox.l,
                                        y=prov.bbox.t,
                                        width=prov.bbox.r - prov.bbox.l,
                                        height=prov.bbox.b - prov.bbox.t,
                                        page=page_num,
                                        confidence=getattr(prov, 'confidence', 0.9),
                                        element_type=element.name if hasattr(element, 'name') else 'unknown',
                                        content=element.text if hasattr(element, 'text') else '',
                                        metadata={}
                                    )
                                    break
                        
                        if bbox is None:
                            # Create default bounding box if not available
                            bbox = BoundingBox(
                                x=0, y=0, width=0, height=0,
                                page=page_num,
                                confidence=0.5,
                                element_type='unknown',
                                content=element.text if hasattr(element, 'text') else '',
                                metadata={}
                            )
                        
                        # Create document element
                        doc_element = DocumentElement(
                            element_id=hashlib.md5(f"{page_num}_{element.text if hasattr(element, 'text') else 'empty'}".encode()).hexdigest()[:16],
                            element_type=element.name if hasattr(element, 'name') else 'text',
                            content=element.text if hasattr(element, 'text') else '',
                            bounding_box=bbox,
                            confidence=bbox.confidence,
                            semantic_label=await self._classify_element_semantics(element),
                            relationships=await self._identify_element_relationships(element, elements),
                            properties=await self._extract_element_properties(element)
                        )
                        
                        elements.append(doc_element)
            
            return elements
            
        except Exception as e:
            logger.error(f"Structured element extraction failed: {e}")
            return []
    
    async def _classify_element_semantics(self, element) -> str:
        """Classify element semantic meaning using ML"""
        
        if not hasattr(element, 'text') or not element.text:
            return 'empty'
        
        text = element.text.lower().strip()
        
        # Simple rule-based classification (could be enhanced with ML)
        if re.match(r'^#+ ', text) or len(text) < 100 and text.isupper():
            return 'heading'
        elif re.match(r'^\d+\.\s', text) or re.match(r'^[•\-\*]\s', text):
            return 'list_item'
        elif '$' in text or '€' in text or '£' in text or re.search(r'\d+,\d+', text):
            return 'financial_data'
        elif '@' in text and '.' in text:
            return 'contact_info'
        elif re.search(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text):
            return 'date_info'
        elif len(text) > 200:
            return 'paragraph'
        else:
            return 'text'
    
    async def _identify_element_relationships(self, element, existing_elements: List[DocumentElement]) -> List[str]:
        """Identify relationships between elements"""
        
        relationships = []
        
        # For now, return simple positional relationships
        # Could be enhanced with more sophisticated relationship detection
        if existing_elements:
            relationships.append(f"follows_{existing_elements[-1].element_id}")
        
        return relationships
    
    async def _extract_element_properties(self, element) -> Dict[str, Any]:
        """Extract additional properties from element"""
        
        properties = {}
        
        if hasattr(element, 'text') and element.text:
            properties['text_length'] = len(element.text)
            properties['word_count'] = len(element.text.split())
            properties['has_numbers'] = bool(re.search(r'\d', element.text))
            properties['has_uppercase'] = any(c.isupper() for c in element.text)
        
        if hasattr(element, 'name'):
            properties['docling_type'] = element.name
        
        return properties
    
    async def _perform_content_analysis(
        self,
        conversion_result: ConversionResult,
        elements: List[DocumentElement]
    ) -> Dict[str, Any]:
        """Perform comprehensive ML-driven content analysis"""
        
        analysis = {
            "full_text": "",
            "content_type": ContentType.GENERIC,
            "entities": {},
            "topics": {},
            "sentiment": {},
            "readability": {}
        }
        
        try:
            # Extract full text
            if hasattr(conversion_result, 'document') and conversion_result.document:
                analysis["full_text"] = conversion_result.document.to_text()
            else:
                analysis["full_text"] = " ".join([elem.content for elem in elements if elem.content])
            
            text = analysis["full_text"]
            
            if not text.strip():
                return analysis
            
            # Content type classification
            analysis["content_type"] = await self._classify_content_type(text, elements)
            
            # Named Entity Recognition
            analysis["entities"] = await self._extract_named_entities(text)
            
            # Topic modeling
            analysis["topics"] = await self._perform_topic_modeling(text)
            
            # Sentiment analysis
            analysis["sentiment"] = await self._analyze_sentiment(text)
            
            # Readability metrics
            analysis["readability"] = await self._compute_readability_metrics(text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return analysis
    
    async def _classify_content_type(self, text: str, elements: List[DocumentElement]) -> ContentType:
        """Classify document content type using ML"""
        
        text_lower = text.lower()
        
        # Financial indicators
        financial_keywords = [
            'revenue', 'profit', 'loss', 'balance sheet', 'income statement',
            'cash flow', 'assets', 'liabilities', 'equity', 'roi', 'investment'
        ]
        financial_score = sum(1 for keyword in financial_keywords if keyword in text_lower)
        
        # Legal indicators
        legal_keywords = [
            'contract', 'agreement', 'terms', 'conditions', 'whereas',
            'hereby', 'shall', 'party', 'parties', 'jurisdiction'
        ]
        legal_score = sum(1 for keyword in legal_keywords if keyword in text_lower)
        
        # Technical indicators
        technical_keywords = [
            'specification', 'requirements', 'architecture', 'design',
            'implementation', 'algorithm', 'system', 'method', 'process'
        ]
        technical_score = sum(1 for keyword in technical_keywords if keyword in text_lower)
        
        # Research indicators
        research_keywords = [
            'abstract', 'introduction', 'methodology', 'results',
            'conclusion', 'references', 'hypothesis', 'experiment'
        ]
        research_score = sum(1 for keyword in research_keywords if keyword in text_lower)
        
        # Determine highest scoring category
        scores = {
            ContentType.FINANCIAL_REPORT: financial_score,
            ContentType.LEGAL_DOCUMENT: legal_score,
            ContentType.TECHNICAL_SPEC: technical_score,
            ContentType.RESEARCH_PAPER: research_score
        }
        
        if max(scores.values()) > 2:  # Threshold for classification
            return max(scores, key=scores.get)
        else:
            return ContentType.GENERIC
    
    async def _extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using NLP models"""
        
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'money': [],
            'misc': []
        }
        
        try:
            # Use spaCy if available
            if self.nlp:
                doc = self.nlp(text[:1000000])  # Limit text length for performance
                
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'PER']:
                        entities['persons'].append(ent.text)
                    elif ent.label_ in ['ORG', 'ORGANIZATION']:
                        entities['organizations'].append(ent.text)
                    elif ent.label_ in ['GPE', 'LOC', 'LOCATION']:
                        entities['locations'].append(ent.text)
                    elif ent.label_ in ['DATE', 'TIME']:
                        entities['dates'].append(ent.text)
                    elif ent.label_ in ['MONEY', 'PERCENT']:
                        entities['money'].append(ent.text)
                    else:
                        entities['misc'].append(ent.text)
            
            # Use Transformers NER pipeline if available
            elif self.ner_pipeline:
                ner_results = self.ner_pipeline(text[:512])  # Truncate for model limits
                
                for result in ner_results:
                    entity_type = result['entity_group'].lower()
                    if 'per' in entity_type:
                        entities['persons'].append(result['word'])
                    elif 'org' in entity_type:
                        entities['organizations'].append(result['word'])
                    elif 'loc' in entity_type:
                        entities['locations'].append(result['word'])
                    else:
                        entities['misc'].append(result['word'])
            
            # Remove duplicates and clean up
            for key in entities:
                entities[key] = list(set(entities[key]))
                entities[key] = [ent.strip() for ent in entities[key] if len(ent.strip()) > 1]
            
            return entities
            
        except Exception as e:
            logger.error(f"Named entity extraction failed: {e}")
            return entities
    
    async def _perform_topic_modeling(self, text: str) -> Dict[str, Any]:
        """Perform topic modeling on document text"""
        
        topics = {
            'num_topics': 0,
            'main_topics': [],
            'topic_distribution': {},
            'keywords_per_topic': {}
        }
        
        try:
            if len(text.strip()) < 100:  # Too short for meaningful topic modeling
                return topics
            
            # Vectorize text
            text_vector = self.text_vectorizer.fit_transform([text])
            
            # Perform LDA topic modeling
            if text_vector.shape[1] > 10:  # Ensure sufficient vocabulary
                n_topics = min(5, max(2, text_vector.shape[1] // 100))  # Adaptive number of topics
                self.topic_model.n_components = n_topics
                
                topic_dist = self.topic_model.fit_transform(text_vector)
                
                topics['num_topics'] = n_topics
                topics['topic_distribution'] = {
                    f'topic_{i}': float(prob) for i, prob in enumerate(topic_dist[0])
                }
                
                # Get top topics
                top_topic_indices = np.argsort(topic_dist[0])[-3:][::-1]  # Top 3 topics
                topics['main_topics'] = [f'topic_{i}' for i in top_topic_indices]
                
                # Get keywords per topic
                feature_names = self.text_vectorizer.get_feature_names_out()
                for topic_idx in range(n_topics):
                    top_words_idx = np.argsort(self.topic_model.components_[topic_idx])[-10:][::-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    topics['keywords_per_topic'][f'topic_{topic_idx}'] = top_words
            
            return topics
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}")
            return topics
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze document sentiment"""
        
        sentiment = {
            'overall_polarity': 0.0,
            'overall_subjectivity': 0.0,
            'confidence': 0.0
        }
        
        try:
            if self.sentiment_analyzer and text.strip():
                # Use transformer-based sentiment analysis
                result = self.sentiment_analyzer(text[:512])  # Truncate for model limits
                
                if result:
                    sentiment['confidence'] = float(result[0]['score'])
                    
                    # Convert label to polarity score
                    label = result[0]['label'].upper()
                    if 'POSITIVE' in label:
                        sentiment['overall_polarity'] = sentiment['confidence']
                    elif 'NEGATIVE' in label:
                        sentiment['overall_polarity'] = -sentiment['confidence']
                    else:
                        sentiment['overall_polarity'] = 0.0
                    
                    # Simple subjectivity estimation
                    sentiment['overall_subjectivity'] = sentiment['confidence']
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return sentiment
    
    async def _compute_readability_metrics(self, text: str) -> Dict[str, float]:
        """Compute readability metrics"""
        
        metrics = {
            'flesch_reading_ease': 0.0,
            'flesch_kincaid_grade': 0.0,
            'avg_sentence_length': 0.0,
            'avg_word_length': 0.0,
            'complexity_score': 0.0
        }
        
        try:
            if not text.strip():
                return metrics
            
            # Basic text statistics
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = text.split()
            
            if not sentences or not words:
                return metrics
            
            total_sentences = len(sentences)
            total_words = len(words)
            total_syllables = sum(self._count_syllables(word) for word in words)
            
            # Average sentence length
            metrics['avg_sentence_length'] = total_words / total_sentences
            
            # Average word length
            metrics['avg_word_length'] = sum(len(word) for word in words) / total_words
            
            # Flesch Reading Ease
            if total_sentences > 0 and total_words > 0:
                metrics['flesch_reading_ease'] = (
                    206.835 - (1.015 * (total_words / total_sentences)) -
                    (84.6 * (total_syllables / total_words))
                )
            
            # Flesch-Kincaid Grade Level
            if total_sentences > 0 and total_words > 0:
                metrics['flesch_kincaid_grade'] = (
                    (0.39 * (total_words / total_sentences)) +
                    (11.8 * (total_syllables / total_words)) - 15.59
                )
            
            # Complexity score (0-1, higher = more complex)
            long_words = len([w for w in words if len(w) > 6])
            complex_sentences = len([s for s in sentences if len(s.split()) > 20])
            
            metrics['complexity_score'] = min(1.0, (
                (long_words / total_words) * 0.5 +
                (complex_sentences / total_sentences) * 0.5
            ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Readability metrics computation failed: {e}")
            return metrics
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple heuristic)"""
        word = word.lower()
        vowels = "aeiouy"
        syllables = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllables > 1:
            syllables -= 1
        
        return max(1, syllables)
    
    async def _generate_content_insights(
        self,
        content_analysis: Dict[str, Any],
        elements: List[DocumentElement],
        context: Optional[Dict[str, Any]]
    ) -> List[ContentInsight]:
        """Generate ML-driven content insights"""
        
        insights = []
        
        try:
            text = content_analysis["full_text"]
            entities = content_analysis["entities"]
            topics = content_analysis["topics"]
            sentiment = content_analysis["sentiment"]
            
            # Entity-based insights
            if entities['organizations']:
                insight = ContentInsight(
                    insight_type="entity_analysis",
                    confidence=0.9,
                    description=f"Document mentions {len(entities['organizations'])} organizations",
                    key_entities=entities['organizations'][:5],
                    sentiment_score=None,
                    topic_labels=['organizations'],
                    importance_score=0.8,
                    supporting_evidence=entities['organizations'][:3]
                )
                insights.append(insight)
            
            # Financial insights
            if content_analysis["content_type"] == ContentType.FINANCIAL_REPORT:
                financial_terms = [term for term in text.lower().split() 
                                 if term in ['revenue', 'profit', 'loss', 'earnings', 'roi']]
                if financial_terms:
                    insight = ContentInsight(
                        insight_type="financial_analysis",
                        confidence=0.85,
                        description=f"Financial document with {len(financial_terms)} financial terms",
                        key_entities=financial_terms[:5],
                        sentiment_score=sentiment.get('overall_polarity', 0),
                        topic_labels=['finance', 'business'],
                        importance_score=0.9,
                        supporting_evidence=financial_terms[:3]
                    )
                    insights.append(insight)
            
            # Topic-based insights
            if topics.get('main_topics'):
                for topic in topics['main_topics'][:2]:  # Top 2 topics
                    keywords = topics.get('keywords_per_topic', {}).get(topic, [])
                    if keywords:
                        insight = ContentInsight(
                            insight_type="topic_analysis",
                            confidence=topics['topic_distribution'].get(topic, 0.5),
                            description=f"Key topic identified: {', '.join(keywords[:3])}",
                            key_entities=keywords[:5],
                            sentiment_score=None,
                            topic_labels=[topic],
                            importance_score=topics['topic_distribution'].get(topic, 0.5),
                            supporting_evidence=keywords[:3]
                        )
                        insights.append(insight)
            
            # Sentiment insights
            if abs(sentiment.get('overall_polarity', 0)) > 0.3:
                polarity_label = "positive" if sentiment['overall_polarity'] > 0 else "negative"
                insight = ContentInsight(
                    insight_type="sentiment_analysis",
                    confidence=sentiment.get('confidence', 0.7),
                    description=f"Document has {polarity_label} sentiment (score: {sentiment['overall_polarity']:.2f})",
                    key_entities=[],
                    sentiment_score=sentiment['overall_polarity'],
                    topic_labels=['sentiment'],
                    importance_score=abs(sentiment['overall_polarity']),
                    supporting_evidence=[f"Sentiment polarity: {sentiment['overall_polarity']:.2f}"]
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Content insights generation failed: {e}")
            return insights
    
    async def _summarize_bounding_boxes(self, elements: List[DocumentElement]) -> Dict[str, Any]:
        """Summarize bounding box information"""
        
        summary = {
            'total_bounding_boxes': len(elements),
            'elements_by_type': {},
            'elements_by_page': {},
            'confidence_distribution': {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            },
            'spatial_distribution': {}
        }
        
        if not elements:
            return summary
        
        # Count by element type
        for element in elements:
            elem_type = element.element_type
            summary['elements_by_type'][elem_type] = summary['elements_by_type'].get(elem_type, 0) + 1
        
        # Count by page
        for element in elements:
            page = element.bounding_box.page
            summary['elements_by_page'][f'page_{page}'] = summary['elements_by_page'].get(f'page_{page}', 0) + 1
        
        # Confidence statistics
        confidences = [element.confidence for element in elements]
        summary['confidence_distribution'] = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
        
        return summary
    
    async def _assess_document_quality(
        self,
        conversion_result: ConversionResult,
        elements: List[DocumentElement]
    ) -> Dict[str, float]:
        """Assess document processing quality"""
        
        quality = {
            'extraction_completeness': 0.0,
            'text_quality_score': 0.0,
            'structure_quality_score': 0.0,
            'overall_quality_score': 0.0
        }
        
        try:
            # Text quality assessment
            total_text = " ".join([elem.content for elem in elements if elem.content])
            
            if total_text.strip():
                # Check for garbled text, OCR errors
                alpha_ratio = sum(1 for c in total_text if c.isalpha()) / len(total_text)
                quality['text_quality_score'] = min(1.0, alpha_ratio * 1.2)  # Boost good ratios
                
                # Structure quality based on element diversity
                unique_types = len(set(elem.element_type for elem in elements))
                quality['structure_quality_score'] = min(1.0, unique_types / 5.0)  # Expect up to 5 types
                
                # Extraction completeness based on confidence
                if elements:
                    avg_confidence = np.mean([elem.confidence for elem in elements])
                    quality['extraction_completeness'] = avg_confidence
                
                # Overall quality (weighted average)
                weights = [0.4, 0.3, 0.3]
                scores = [
                    quality['text_quality_score'],
                    quality['structure_quality_score'],
                    quality['extraction_completeness']
                ]
                quality['overall_quality_score'] = np.average(scores, weights=weights)
            
            return quality
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return quality
    
    async def _compute_confidence_metrics(
        self,
        quality_metrics: Dict[str, float],
        element_count: int
    ) -> Dict[str, float]:
        """Compute confidence metrics for the analysis"""
        
        confidence = {
            'processing_confidence': 0.0,
            'content_analysis_confidence': 0.0,
            'overall_confidence': 0.0
        }
        
        try:
            # Processing confidence based on quality and element count
            quality_score = quality_metrics.get('overall_quality_score', 0.5)
            element_score = min(1.0, element_count / 20.0)  # Expect ~20 elements for good confidence
            
            confidence['processing_confidence'] = (quality_score * 0.7) + (element_score * 0.3)
            
            # Content analysis confidence based on text availability
            text_available_score = 1.0 if element_count > 0 else 0.0
            confidence['content_analysis_confidence'] = text_available_score * quality_score
            
            # Overall confidence
            confidence['overall_confidence'] = (
                confidence['processing_confidence'] * 0.6 +
                confidence['content_analysis_confidence'] * 0.4
            )
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence metrics computation failed: {e}")
            return confidence
    
    def _get_active_models(self) -> List[str]:
        """Get list of active ML models"""
        models = []
        
        if self.sentiment_analyzer:
            models.append("transformers_sentiment")
        if self.ner_pipeline:
            models.append("transformers_ner")
        if self.nlp:
            models.append("spacy_nlp")
        
        models.extend(["tfidf_vectorizer", "lda_topic_model", "docling_converter"])
        
        return models
    
    async def _store_processing_patterns(
        self,
        profile: DocumentAnalysisProfile,
        context: Dict[str, Any]
    ):
        """Store document processing patterns for future optimization"""
        
        if not self.memory_service:
            return
        
        try:
            processing_data = {
                "analysis_type": "document_processing",
                "document_type": profile.document_type.value,
                "content_type": profile.content_type.value,
                "processing_complexity": profile.processing_complexity.value,
                "performance_metrics": {
                    "total_elements": profile.total_elements,
                    "total_pages": profile.total_pages,
                    "quality_score": profile.quality_assessment.get("overall_quality_score", 0),
                    "confidence": profile.confidence_metrics.get("overall_confidence", 0)
                },
                "ml_insights": {
                    "content_insights_count": len(profile.content_insights),
                    "entity_types_found": len(profile.key_entities),
                    "topics_identified": profile.topic_modeling.get("num_topics", 0)
                },
                "processing_timestamp": datetime.now().isoformat(),
                "successful_processing": True
            }
            
            await self.memory_service.store_conversation_memory(
                user_id=context.get("user_id", "system"),
                message=json.dumps(processing_data),
                context={
                    "category": "document_processing",
                    "document_type": profile.document_type.value,
                    "content_type": profile.content_type.value
                }
            )
            
        except Exception as e:
            logger.warning(f"Failed to store processing patterns: {e}")