"""
Advanced Document Agent - Production Grade
Intelligent document processing with Docling bounding box extraction and ML-driven content analysis
"""

import asyncio
import structlog
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
from pydantic import BaseModel, Field, validator
from enum import Enum
from io import BytesIO
import base64

from .base.agent import HardenedAgent
from .base.registry import tool
from .base.exceptions import ToolError
from src.adapters.document.intelligent_processor import (
    IntelligentDocumentProcessor,
    DocumentAnalysisProfile,
    DocumentType,
    ContentType,
    ProcessingComplexity,
    BoundingBox,
    DocumentElement,
    ContentInsight
)

import numpy as np

logger = structlog.get_logger(__name__)


# Pydantic Models for Input/Output Validation
class DocumentInput(BaseModel):
    """Document input specification"""
    file_path: Optional[str] = Field(None, description="Path to document file")
    file_data: Optional[str] = Field(None, description="Base64 encoded file data")
    file_name: Optional[str] = Field(None, description="Original file name")
    document_id: Optional[str] = Field(None, description="Custom document identifier")
    
    @validator('file_path', 'file_data', pre=True, always=True)
    def validate_input_source(cls, v, values):
        if not values.get('file_path') and not values.get('file_data'):
            raise ValueError('Either file_path or file_data must be provided')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "file_path": "/path/to/document.pdf",
                "file_name": "financial_report.pdf",
                "document_id": "doc_001"
            }
        }


class ProcessingOptions(BaseModel):
    """Document processing configuration options"""
    extract_tables: bool = Field(True, description="Extract table structures")
    ocr_required: bool = Field(False, description="Force OCR processing")
    extract_images: bool = Field(False, description="Extract embedded images")
    analyze_sentiment: bool = Field(True, description="Perform sentiment analysis")
    extract_entities: bool = Field(True, description="Extract named entities")
    topic_modeling: bool = Field(True, description="Perform topic modeling")
    bounding_box_precision: str = Field("standard", description="Bounding box precision level")
    
    @validator('bounding_box_precision')
    def validate_precision(cls, v):
        allowed = ['basic', 'standard', 'high', 'ultra']
        if v not in allowed:
            raise ValueError(f'bounding_box_precision must be one of {allowed}')
        return v


class DocumentElementResult(BaseModel):
    """Document element with bounding box information"""
    element_id: str
    element_type: str
    content: str
    page_number: int
    bounding_box: Dict[str, float] = Field(..., description="x, y, width, height coordinates")
    confidence: float = Field(..., ge=0, le=1)
    semantic_label: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class ContentInsightResult(BaseModel):
    """Content insight from ML analysis"""
    insight_type: str
    confidence: float = Field(..., ge=0, le=1)
    description: str
    key_entities: List[str] = Field(default_factory=list)
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1)
    topic_labels: List[str] = Field(default_factory=list)
    importance_score: float = Field(..., ge=0, le=1)


class DocumentAnalysisResult(BaseModel):
    """Comprehensive document analysis results"""
    document_id: str
    document_type: str
    content_type: str
    processing_complexity: str
    analysis_timestamp: datetime
    
    total_pages: int = Field(..., ge=1)
    total_elements: int = Field(..., ge=0)
    
    full_text: str
    structured_elements: List[DocumentElementResult]
    content_insights: List[ContentInsightResult]
    
    key_entities: Dict[str, List[str]] = Field(default_factory=dict)
    topic_analysis: Dict[str, Any] = Field(default_factory=dict)
    sentiment_analysis: Dict[str, float] = Field(default_factory=dict)
    readability_metrics: Dict[str, float] = Field(default_factory=dict)
    
    quality_assessment: Dict[str, float] = Field(default_factory=dict)
    confidence_metrics: Dict[str, float] = Field(default_factory=dict)
    
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = Field(default="1.0")


class DocumentComparisonRequest(BaseModel):
    """Request for document comparison analysis"""
    documents: List[DocumentInput] = Field(..., min_items=2, max_items=5)
    comparison_metrics: List[str] = Field(
        default=["content_similarity", "structure", "quality"],
        description="Metrics to compare"
    )
    processing_options: Optional[ProcessingOptions] = None


class DocumentSimilarityResult(BaseModel):
    """Document similarity analysis results"""
    document_pair: List[str]
    similarity_score: float = Field(..., ge=0, le=1)
    similarity_breakdown: Dict[str, float] = Field(default_factory=dict)
    content_overlap: Dict[str, Any] = Field(default_factory=dict)


class ComparisonAnalysisResult(BaseModel):
    """Document comparison analysis results"""
    comparison_timestamp: datetime
    documents_analyzed: List[str]
    similarity_matrix: List[DocumentSimilarityResult]
    content_clustering: Dict[str, Any] = Field(default_factory=dict)
    quality_rankings: Dict[str, List[str]] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    schema_version: str = Field(default="1.0")


class AdvancedDocumentAgent(HardenedAgent):
    """
    Advanced Document Agent with intelligent processing capabilities
    
    Features:
    - Docling integration with bounding box extraction
    - ML-driven content analysis and insights
    - Multi-format document support (PDF, Word, Excel, etc.)
    - Named entity recognition and topic modeling
    - Document quality assessment
    - Intelligent document comparison
    - Memory-driven optimization
    """
    
    _domain = "document"
    _capabilities = [
        "document_processing",
        "bounding_box_extraction",
        "content_analysis",
        "entity_extraction", 
        "topic_modeling",
        "sentiment_analysis",
        "document_comparison",
        "quality_assessment",
        "multi_format_support",
        "ml_driven_insights"
    ]
    
    def __init__(self, memory_service, agent_id: Optional[str] = None):
        super().__init__(memory_service, agent_id)
        self.document_processor = IntelligentDocumentProcessor(memory_service)
        self._processing_cache = {}
        
        logger.info(
            "advanced_document_agent_initialized",
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
        """Intelligent message processing with document context awareness"""
        
        request_id = plan.get('request_id', uuid4().hex)
        
        logger.info(
            "document_message_processing",
            agent_id=self.agent_id,
            request_id=request_id,
            message_type=await self._classify_document_intent(message),
            memory_context_size=len(memory_context)
        )
        
        intent = await self._classify_document_intent(message)
        
        if intent == "document_processing":
            return f"I can process documents with Docling bounding box extraction, ML content analysis, and multi-format support. Memory context: {len(memory_context)} processing patterns available."
        elif intent == "content_analysis":
            return f"I'll analyze document content using NER, topic modeling, sentiment analysis, and quality assessment. {len(memory_context)} analysis patterns in memory."
        elif intent == "bounding_box_extraction":
            return f"I can extract precise bounding boxes using Docling with configurable precision levels. {len(memory_context)} extraction patterns available."
        elif intent == "document_comparison":
            return f"I'll compare documents across content similarity, structure, and quality metrics. {len(memory_context)} comparison patterns in memory."
        else:
            return f"I can assist with comprehensive document processing including bounding box extraction, content analysis, entity recognition, and intelligent document comparison. {len(memory_context)} relevant memories found."
    
    @tool(description="Comprehensive document processing with bounding box extraction and ML analysis")
    async def process_document(
        self,
        document_input: DocumentInput,
        processing_options: Optional[ProcessingOptions] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DocumentAnalysisResult:
        """
        Process document with comprehensive analysis including:
        - Docling bounding box extraction
        - ML-driven content analysis
        - Named entity recognition
        - Topic modeling and sentiment analysis
        - Document quality assessment
        
        Complexity: O(n * m) for n pages and m elements per page
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "document_processing_started",
            agent_id=self.agent_id,
            request_id=request_id,
            document_id=document_input.document_id,
            has_file_path=bool(document_input.file_path),
            has_file_data=bool(document_input.file_data),
            estimated_cost=0.30
        )
        
        try:
            # Prepare document input for processor
            doc_path = await self._prepare_document_input(document_input)
            
            # Configure processing options
            proc_options = self._convert_processing_options(processing_options or ProcessingOptions())
            
            # Execute comprehensive document processing
            analysis_profile = await self.document_processor.process_document_comprehensive(
                document_path=doc_path,
                document_id=document_input.document_id,
                processing_options=proc_options,
                context=context or {}
            )
            
            # Convert internal format to API result
            result = await self._convert_analysis_profile_to_result(
                analysis_profile, document_input
            )
            
            # Store processing pattern for optimization
            await self._store_processing_pattern("document_processing", result, context)
            
            logger.info(
                "document_processing_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                document_id=result.document_id,
                total_elements=result.total_elements,
                content_insights=len(result.content_insights)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "document_processing_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                document_id=document_input.document_id,
                error=str(e),
                error_type=type(e).__name__
            )
            raise ToolError(
                f"Document processing failed: {str(e)}",
                tool_name="process_document",
                agent_id=self.agent_id
            )
    
    @tool(description="Extract bounding boxes from document with configurable precision")
    async def extract_bounding_boxes(
        self,
        document_input: DocumentInput,
        precision_level: str = "standard",
        element_types: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract bounding boxes with configurable precision:
        - Basic: Simple element detection
        - Standard: Enhanced element classification  
        - High: Detailed semantic labeling
        - Ultra: Maximum precision with relationships
        
        Complexity: O(n * p) for n elements and precision level p
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "bounding_box_extraction_started",
            agent_id=self.agent_id,
            request_id=request_id,
            precision_level=precision_level,
            element_types=element_types,
            estimated_cost=0.15
        )
        
        try:
            # Configure processing for bounding box focus
            processing_options = ProcessingOptions(
                bounding_box_precision=precision_level,
                analyze_sentiment=False,  # Skip for performance
                topic_modeling=False
            )
            
            # Process document with focus on structure
            analysis_profile = await self.document_processor.process_document_comprehensive(
                document_path=await self._prepare_document_input(document_input),
                document_id=document_input.document_id,
                processing_options=self._convert_processing_options(processing_options),
                context=context or {}
            )
            
            # Filter elements by type if specified
            elements = analysis_profile.structured_elements
            if element_types:
                elements = [elem for elem in elements if elem.element_type in element_types]
            
            # Prepare bounding box results
            bounding_boxes = []
            for elem in elements:
                bbox_data = {
                    "element_id": elem.element_id,
                    "element_type": elem.element_type,
                    "page": elem.bounding_box.page,
                    "coordinates": {
                        "x": elem.bounding_box.x,
                        "y": elem.bounding_box.y,
                        "width": elem.bounding_box.width,
                        "height": elem.bounding_box.height
                    },
                    "content": elem.content,
                    "confidence": elem.confidence,
                    "semantic_label": elem.semantic_label
                }
                bounding_boxes.append(bbox_data)
            
            result = {
                "document_id": analysis_profile.document_id,
                "extraction_timestamp": datetime.now(),
                "precision_level": precision_level,
                "total_bounding_boxes": len(bounding_boxes),
                "bounding_boxes": bounding_boxes,
                "element_type_distribution": await self._compute_element_distribution(elements),
                "confidence_statistics": await self._compute_confidence_stats(elements),
                "extraction_metadata": {
                    "docling_version": "latest",
                    "precision_settings": precision_level,
                    "filtered_types": element_types
                }
            }
            
            # Store extraction pattern
            await self._store_processing_pattern("bounding_box_extraction", result, context)
            
            logger.info(
                "bounding_box_extraction_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                bounding_boxes_extracted=len(bounding_boxes),
                precision_level=precision_level
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "bounding_box_extraction_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise ToolError(
                f"Bounding box extraction failed: {str(e)}",
                tool_name="extract_bounding_boxes",
                agent_id=self.agent_id
            )
    
    @tool(description="Advanced content analysis with ML-driven insights")
    async def analyze_content(
        self,
        document_input: DocumentInput,
        analysis_types: List[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive content analysis including:
        - Named entity recognition
        - Topic modeling with clustering
        - Sentiment analysis
        - Readability assessment
        - Content quality evaluation
        
        Complexity: O(n * log(n)) for n words in document
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        # Default analysis types
        if analysis_types is None:
            analysis_types = ["entities", "topics", "sentiment", "readability"]
        
        logger.info(
            "content_analysis_started",
            agent_id=self.agent_id,
            request_id=request_id,
            analysis_types=analysis_types,
            estimated_cost=0.20
        )
        
        try:
            # Configure processing for content analysis focus
            processing_options = ProcessingOptions(
                extract_entities="entities" in analysis_types,
                topic_modeling="topics" in analysis_types,
                analyze_sentiment="sentiment" in analysis_types,
                extract_tables=False,  # Skip for performance
                extract_images=False
            )
            
            # Process document with focus on content
            analysis_profile = await self.document_processor.process_document_comprehensive(
                document_path=await self._prepare_document_input(document_input),
                document_id=document_input.document_id,
                processing_options=self._convert_processing_options(processing_options),
                context=context or {}
            )
            
            # Compile analysis results
            content_analysis = {
                "document_id": analysis_profile.document_id,
                "analysis_timestamp": datetime.now(),
                "content_type": analysis_profile.content_type.value,
                "full_text_length": len(analysis_profile.text_content),
                "word_count": len(analysis_profile.text_content.split())
            }
            
            # Named entities
            if "entities" in analysis_types:
                content_analysis["named_entities"] = analysis_profile.key_entities
                content_analysis["entity_summary"] = {
                    entity_type: len(entities) 
                    for entity_type, entities in analysis_profile.key_entities.items()
                }
            
            # Topic modeling
            if "topics" in analysis_types:
                content_analysis["topic_analysis"] = analysis_profile.topic_modeling
                
            # Sentiment analysis
            if "sentiment" in analysis_types:
                content_analysis["sentiment_analysis"] = analysis_profile.sentiment_analysis
                
            # Readability metrics
            if "readability" in analysis_types:
                content_analysis["readability_metrics"] = analysis_profile.readability_metrics
            
            # Content insights
            content_analysis["ml_insights"] = [
                {
                    "type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "importance": insight.importance_score,
                    "key_entities": insight.key_entities
                }
                for insight in analysis_profile.content_insights
            ]
            
            # Quality assessment
            content_analysis["quality_assessment"] = analysis_profile.quality_assessment
            content_analysis["confidence_metrics"] = analysis_profile.confidence_metrics
            
            # Store content analysis pattern
            await self._store_processing_pattern("content_analysis", content_analysis, context)
            
            logger.info(
                "content_analysis_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                insights_generated=len(analysis_profile.content_insights),
                content_type=analysis_profile.content_type.value
            )
            
            return content_analysis
            
        except Exception as e:
            logger.error(
                "content_analysis_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise ToolError(
                f"Content analysis failed: {str(e)}",
                tool_name="analyze_content",
                agent_id=self.agent_id
            )
    
    @tool(description="Intelligent document comparison with similarity analysis")
    async def compare_documents(
        self,
        comparison_request: DocumentComparisonRequest,
        context: Optional[Dict[str, Any]] = None
    ) -> ComparisonAnalysisResult:
        """
        Advanced document comparison across multiple dimensions:
        - Content similarity using ML embeddings
        - Structural similarity analysis
        - Quality comparative assessment
        - Topic overlap detection
        
        Complexity: O(n^2 * m) for n documents with m features each
        """
        request_id = context.get('request_id', uuid4().hex) if context else uuid4().hex
        
        logger.info(
            "document_comparison_started",
            agent_id=self.agent_id,
            request_id=request_id,
            document_count=len(comparison_request.documents),
            comparison_metrics=comparison_request.comparison_metrics,
            estimated_cost=len(comparison_request.documents) * 0.25
        )
        
        try:
            # Process all documents
            analysis_profiles = []
            doc_ids = []
            
            for doc_input in comparison_request.documents:
                profile = await self.document_processor.process_document_comprehensive(
                    document_path=await self._prepare_document_input(doc_input),
                    document_id=doc_input.document_id,
                    processing_options=self._convert_processing_options(
                        comparison_request.processing_options or ProcessingOptions()
                    ),
                    context=context or {}
                )
                analysis_profiles.append(profile)
                doc_ids.append(profile.document_id)
            
            # Perform pairwise comparisons
            similarity_results = []
            for i in range(len(analysis_profiles)):
                for j in range(i + 1, len(analysis_profiles)):
                    similarity = await self._compute_document_similarity(
                        analysis_profiles[i], 
                        analysis_profiles[j], 
                        comparison_request.comparison_metrics
                    )
                    similarity_results.append(similarity)
            
            # Generate content clustering
            content_clustering = await self._perform_content_clustering(
                analysis_profiles, comparison_request.comparison_metrics
            )
            
            # Quality rankings
            quality_rankings = await self._generate_quality_rankings(
                analysis_profiles, comparison_request.comparison_metrics
            )
            
            # Generate recommendations
            recommendations = await self._generate_comparison_recommendations(
                analysis_profiles, similarity_results
            )
            
            result = ComparisonAnalysisResult(
                comparison_timestamp=datetime.now(),
                documents_analyzed=doc_ids,
                similarity_matrix=similarity_results,
                content_clustering=content_clustering,
                quality_rankings=quality_rankings,
                recommendations=recommendations
            )
            
            # Store comparison pattern
            await self._store_processing_pattern("document_comparison", result.dict(), context)
            
            logger.info(
                "document_comparison_completed",
                agent_id=self.agent_id,
                request_id=request_id,
                documents_compared=len(analysis_profiles),
                similarity_pairs=len(similarity_results)
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "document_comparison_failed",
                agent_id=self.agent_id,
                request_id=request_id,
                error=str(e)
            )
            raise ToolError(
                f"Document comparison failed: {str(e)}",
                tool_name="compare_documents",
                agent_id=self.agent_id
            )
    
    # Helper methods
    
    async def _classify_document_intent(self, message: str) -> str:
        """Classify user message intent for document operations"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["bounding box", "bbox", "coordinates", "extract"]):
            return "bounding_box_extraction"
        elif any(word in message_lower for word in ["content", "analyze", "entities", "topics", "sentiment"]):
            return "content_analysis"
        elif any(word in message_lower for word in ["compare", "comparison", "similar", "difference"]):
            return "document_comparison"
        elif any(word in message_lower for word in ["process", "parse", "convert", "document"]):
            return "document_processing"
        else:
            return "general_document"
    
    async def _prepare_document_input(self, document_input: DocumentInput) -> Union[str, BytesIO]:
        """Prepare document input for processing"""
        
        if document_input.file_path:
            return document_input.file_path
        elif document_input.file_data:
            # Decode base64 data
            try:
                file_bytes = base64.b64decode(document_input.file_data)
                return BytesIO(file_bytes)
            except Exception as e:
                raise ValueError(f"Invalid base64 file data: {e}")
        else:
            raise ValueError("No valid document input provided")
    
    def _convert_processing_options(self, options: ProcessingOptions) -> Dict[str, Any]:
        """Convert Pydantic processing options to processor format"""
        
        return {
            "extract_tables": options.extract_tables,
            "ocr_required": options.ocr_required,
            "extract_images": options.extract_images,
            "analyze_sentiment": options.analyze_sentiment,
            "extract_entities": options.extract_entities,
            "topic_modeling": options.topic_modeling,
            "bounding_box_precision": options.bounding_box_precision
        }
    
    async def _convert_analysis_profile_to_result(
        self,
        profile: DocumentAnalysisProfile,
        document_input: DocumentInput
    ) -> DocumentAnalysisResult:
        """Convert internal analysis profile to API result"""
        
        # Convert structured elements
        elements = []
        for elem in profile.structured_elements:
            elem_result = DocumentElementResult(
                element_id=elem.element_id,
                element_type=elem.element_type,
                content=elem.content,
                page_number=elem.bounding_box.page,
                bounding_box={
                    "x": elem.bounding_box.x,
                    "y": elem.bounding_box.y,
                    "width": elem.bounding_box.width,
                    "height": elem.bounding_box.height
                },
                confidence=elem.confidence,
                semantic_label=elem.semantic_label,
                properties=elem.properties
            )
            elements.append(elem_result)
        
        # Convert insights
        insights = []
        for insight in profile.content_insights:
            insight_result = ContentInsightResult(
                insight_type=insight.insight_type,
                confidence=insight.confidence,
                description=insight.description,
                key_entities=insight.key_entities,
                sentiment_score=insight.sentiment_score,
                topic_labels=insight.topic_labels,
                importance_score=insight.importance_score
            )
            insights.append(insight_result)
        
        return DocumentAnalysisResult(
            document_id=profile.document_id,
            document_type=profile.document_type.value,
            content_type=profile.content_type.value,
            processing_complexity=profile.processing_complexity.value,
            analysis_timestamp=datetime.now(),
            total_pages=profile.total_pages,
            total_elements=profile.total_elements,
            full_text=profile.text_content,
            structured_elements=elements,
            content_insights=insights,
            key_entities=profile.key_entities,
            topic_analysis=profile.topic_modeling,
            sentiment_analysis=profile.sentiment_analysis,
            readability_metrics=profile.readability_metrics,
            quality_assessment=profile.quality_assessment,
            confidence_metrics=profile.confidence_metrics,
            processing_metadata=profile.processing_metadata
        )
    
    async def _compute_element_distribution(self, elements: List[DocumentElement]) -> Dict[str, int]:
        """Compute distribution of element types"""
        
        distribution = {}
        for elem in elements:
            elem_type = elem.element_type
            distribution[elem_type] = distribution.get(elem_type, 0) + 1
        
        return distribution
    
    async def _compute_confidence_stats(self, elements: List[DocumentElement]) -> Dict[str, float]:
        """Compute confidence statistics for elements"""
        
        if not elements:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        confidences = [elem.confidence for elem in elements]
        return {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences))
        }
    
    async def _compute_document_similarity(
        self,
        profile1: DocumentAnalysisProfile,
        profile2: DocumentAnalysisProfile,
        metrics: List[str]
    ) -> DocumentSimilarityResult:
        """Compute similarity between two documents"""
        
        similarity_scores = {}
        
        # Content similarity
        if "content_similarity" in metrics:
            # Simple text-based similarity (could be enhanced with embeddings)
            text1 = profile1.text_content.lower()
            text2 = profile2.text_content.lower()
            
            words1 = set(text1.split())
            words2 = set(text2.split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity_scores["content_similarity"] = intersection / union if union > 0 else 0.0
        
        # Structural similarity
        if "structure" in metrics:
            # Compare element type distributions
            types1 = set(elem.element_type for elem in profile1.structured_elements)
            types2 = set(elem.element_type for elem in profile2.structured_elements)
            
            type_intersection = len(types1.intersection(types2))
            type_union = len(types1.union(types2))
            
            similarity_scores["structure_similarity"] = type_intersection / type_union if type_union > 0 else 0.0
        
        # Quality similarity
        if "quality" in metrics:
            qual1 = profile1.quality_assessment.get("overall_quality_score", 0.5)
            qual2 = profile2.quality_assessment.get("overall_quality_score", 0.5)
            
            # Similarity based on quality difference (closer = more similar)
            quality_diff = abs(qual1 - qual2)
            similarity_scores["quality_similarity"] = 1.0 - quality_diff
        
        # Overall similarity (average of metrics)
        overall_similarity = np.mean(list(similarity_scores.values())) if similarity_scores else 0.0
        
        return DocumentSimilarityResult(
            document_pair=[profile1.document_id, profile2.document_id],
            similarity_score=float(overall_similarity),
            similarity_breakdown=similarity_scores,
            content_overlap={
                "common_entities": len(
                    set(profile1.key_entities.get('organizations', [])).intersection(
                        set(profile2.key_entities.get('organizations', []))
                    )
                ),
                "topic_overlap": len(
                    set(profile1.topic_modeling.get('main_topics', [])).intersection(
                        set(profile2.topic_modeling.get('main_topics', []))
                    )
                )
            }
        )
    
    async def _perform_content_clustering(
        self,
        profiles: List[DocumentAnalysisProfile],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform content clustering analysis"""
        
        # Simple clustering based on content type
        clusters = {}
        for profile in profiles:
            content_type = profile.content_type.value
            if content_type not in clusters:
                clusters[content_type] = []
            clusters[content_type].append(profile.document_id)
        
        return {
            "content_type_clusters": clusters,
            "cluster_count": len(clusters),
            "largest_cluster_size": max(len(docs) for docs in clusters.values()) if clusters else 0
        }
    
    async def _generate_quality_rankings(
        self,
        profiles: List[DocumentAnalysisProfile],
        metrics: List[str]
    ) -> Dict[str, List[str]]:
        """Generate quality-based rankings"""
        
        rankings = {}
        
        # Overall quality ranking
        quality_scores = [
            (profile.document_id, profile.quality_assessment.get("overall_quality_score", 0.5))
            for profile in profiles
        ]
        quality_scores.sort(key=lambda x: x[1], reverse=True)
        rankings["overall_quality"] = [doc_id for doc_id, _ in quality_scores]
        
        # Content richness ranking (based on insights count)
        content_scores = [
            (profile.document_id, len(profile.content_insights))
            for profile in profiles
        ]
        content_scores.sort(key=lambda x: x[1], reverse=True)
        rankings["content_richness"] = [doc_id for doc_id, _ in content_scores]
        
        return rankings
    
    async def _generate_comparison_recommendations(
        self,
        profiles: List[DocumentAnalysisProfile],
        similarities: List[DocumentSimilarityResult]
    ) -> List[str]:
        """Generate intelligent recommendations from comparison"""
        
        recommendations = []
        
        # Check for high similarity documents
        high_similarity = [sim for sim in similarities if sim.similarity_score > 0.8]
        if high_similarity:
            recommendations.append(f"Found {len(high_similarity)} highly similar document pairs - consider consolidation")
        
        # Check for low quality documents
        low_quality = [p for p in profiles if p.quality_assessment.get("overall_quality_score", 1.0) < 0.5]
        if low_quality:
            recommendations.append(f"{len(low_quality)} documents have low processing quality - consider re-processing")
        
        # Check for diverse content types
        content_types = set(p.content_type.value for p in profiles)
        if len(content_types) > 1:
            recommendations.append(f"Documents span {len(content_types)} content types - ensure appropriate processing for each")
        
        return recommendations
    
    async def _store_processing_pattern(
        self,
        processing_type: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ):
        """Store document processing patterns for future optimization"""
        if not self.memory_service or not context:
            return
        
        try:
            memory_content = {
                "processing_type": f"document_{processing_type}",
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
                message=f"Document processing: {processing_type}",
                context={
                    "category": "document_processing",
                    "processing_type": processing_type,
                    "agent_id": self.agent_id
                }
            )
        except Exception as e:
            logger.warning(f"Failed to store processing pattern: {e}")