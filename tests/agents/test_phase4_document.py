"""
Test suite for Phase 4 AdvancedDocumentAgent
Comprehensive testing of document processing and Docling integration capabilities
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.document import AdvancedDocumentAgent
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
def document_agent(mock_memory_service):
    """Create AdvancedDocumentAgent instance for testing"""
    return AdvancedDocumentAgent(memory_service=mock_memory_service)


@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        'document_path': '/tmp/test_document.pdf',
        'document_content': 'This is a test document with financial information and key insights.',
        'bounding_boxes': [
            {
                'element_type': 'heading',
                'text': 'Financial Report 2024',
                'bbox': [100, 200, 400, 250],
                'confidence': 0.95,
                'page': 1
            },
            {
                'element_type': 'paragraph',
                'text': 'Revenue increased by 15% compared to previous quarter.',
                'bbox': [100, 300, 500, 350],
                'confidence': 0.88,
                'page': 1
            }
        ],
        'entities': [
            {
                'text': '2024',
                'label': 'DATE',
                'confidence': 0.95
            },
            {
                'text': '15%',
                'label': 'PERCENT',
                'confidence': 0.92
            },
            {
                'text': 'Revenue',
                'label': 'FINANCIAL_METRIC',
                'confidence': 0.89
            }
        ],
        'topics': [
            {
                'topic': 'financial_performance',
                'weight': 0.85,
                'keywords': ['revenue', 'growth', 'quarter']
            },
            {
                'topic': 'business_metrics',
                'weight': 0.72,
                'keywords': ['performance', 'analysis', 'trends']
            }
        ]
    }


class TestAdvancedDocumentAgent:
    """Test cases for AdvancedDocumentAgent"""
    
    def test_agent_initialization(self, document_agent):
        """Test agent initialization and basic properties"""
        assert document_agent.get_domain() == "document"
        assert "document_processing" in document_agent.get_capabilities()
        assert "bounding_box_extraction" in document_agent.get_capabilities()
        assert "content_analysis" in document_agent.get_capabilities()
        assert hasattr(document_agent, 'document_processor')
    
    @pytest.mark.asyncio
    async def test_process_document_tool(self, document_agent, sample_document_data):
        """Test comprehensive document processing"""
        with patch.object(document_agent.document_processor, 'process_document') as mock_process:
            mock_process.return_value = {
                'document_metadata': {
                    'file_type': 'pdf',
                    'pages': 5,
                    'file_size_mb': 2.3,
                    'processing_time_seconds': 4.2
                },
                'content_summary': {
                    'total_text_length': 15000,
                    'paragraph_count': 45,
                    'heading_count': 12,
                    'table_count': 3,
                    'image_count': 2
                },
                'quality_metrics': {
                    'text_extraction_confidence': 0.94,
                    'structure_detection_confidence': 0.89,
                    'overall_quality_score': 0.91
                },
                'processing_status': 'completed',
                'extracted_content': sample_document_data['document_content']
            }
            
            result = await document_agent.process_document(
                document_path=sample_document_data['document_path']
            )
            
            assert result['processing_status'] == 'completed'
            assert result['quality_metrics']['overall_quality_score'] > 0.9
            assert result['content_summary']['total_text_length'] > 0
            assert 'extracted_content' in result
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_bounding_boxes_tool(self, document_agent, sample_document_data):
        """Test bounding box extraction with Docling"""
        with patch.object(document_agent.document_processor, 'extract_bounding_boxes') as mock_extract:
            mock_extract.return_value = {
                'bounding_boxes': sample_document_data['bounding_boxes'],
                'extraction_metadata': {
                    'total_elements': 25,
                    'high_confidence_elements': 22,
                    'extraction_confidence': 0.92,
                    'processing_method': 'docling_advanced'
                },
                'element_statistics': {
                    'headings': 5,
                    'paragraphs': 15,
                    'tables': 3,
                    'images': 2
                },
                'quality_assessment': {
                    'bbox_accuracy_score': 0.94,
                    'text_completeness': 0.96
                }
            }
            
            result = await document_agent.extract_bounding_boxes(
                document_path=sample_document_data['document_path']
            )
            
            assert len(result['bounding_boxes']) > 0
            assert result['extraction_metadata']['extraction_confidence'] > 0.9
            assert 'element_statistics' in result
            assert result['quality_assessment']['bbox_accuracy_score'] > 0.9
            mock_extract.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_content_tool(self, document_agent, sample_document_data):
        """Test ML-driven content analysis"""
        with patch.object(document_agent.document_processor, 'analyze_content_intelligence') as mock_analyze:
            mock_analyze.return_value = {
                'content_insights': {
                    'main_topics': sample_document_data['topics'],
                    'sentiment_analysis': {
                        'overall_sentiment': 'positive',
                        'confidence': 0.87,
                        'sentiment_distribution': {
                            'positive': 0.65,
                            'neutral': 0.30,
                            'negative': 0.05
                        }
                    },
                    'complexity_metrics': {
                        'readability_score': 0.78,
                        'technical_complexity': 0.65,
                        'information_density': 0.82
                    }
                },
                'entity_extraction': {
                    'entities': sample_document_data['entities'],
                    'entity_confidence': 0.91,
                    'relationship_graph': {
                        'nodes': 15,
                        'edges': 23
                    }
                },
                'recommendations': [
                    'Document contains high-value financial insights',
                    'Consider extracting key metrics for further analysis',
                    'Strong information density suggests comprehensive content'
                ],
                'confidence_score': 0.93
            }
            
            result = await document_agent.analyze_content(
                document_path=sample_document_data['document_path']
            )
            
            assert 'content_insights' in result
            assert 'entity_extraction' in result
            assert len(result['content_insights']['main_topics']) > 0
            assert result['confidence_score'] > 0.9
            assert len(result['recommendations']) > 0
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compare_documents_tool(self, document_agent):
        """Test intelligent document comparison"""
        document_paths = [
            '/tmp/document1.pdf',
            '/tmp/document2.pdf'
        ]
        
        with patch.object(document_agent.document_processor, 'compare_documents_intelligence') as mock_compare:
            mock_compare.return_value = {
                'similarity_analysis': {
                    'overall_similarity': 0.73,
                    'content_similarity': 0.78,
                    'structure_similarity': 0.65,
                    'topic_overlap': 0.82
                },
                'key_differences': [
                    {
                        'type': 'content',
                        'description': 'Document 1 focuses more on financial metrics',
                        'significance': 0.85
                    },
                    {
                        'type': 'structure',
                        'description': 'Document 2 has additional appendix section',
                        'significance': 0.65
                    }
                ],
                'common_elements': [
                    'Both documents discuss quarterly performance',
                    'Similar executive summary structure',
                    'Consistent financial terminology usage'
                ],
                'recommendation': {
                    'merge_potential': 0.72,
                    'complementary_content': True,
                    'suggested_action': 'Consider creating unified analysis document'
                },
                'confidence_level': 0.88
            }
            
            result = await document_agent.compare_documents(
                document_paths=document_paths
            )
            
            assert 'similarity_analysis' in result
            assert result['similarity_analysis']['overall_similarity'] > 0
            assert len(result['key_differences']) > 0
            assert len(result['common_elements']) > 0
            assert result['confidence_level'] > 0.8
            mock_compare.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_chat_request_processing(self, document_agent):
        """Test processing of chat requests"""
        request = ChatRequest(
            message="Process the PDF document at /tmp/financial_report.pdf and extract key insights",
            user_id="test_user",
            session_id="test_session"
        )
        
        with patch.object(document_agent, 'process_document') as mock_process:
            mock_process.return_value = {
                'processing_status': 'completed',
                'quality_metrics': {'overall_quality_score': 0.92},
                'extracted_content': 'Financial report content...'
            }
            
            response = await document_agent.process_request(request)
            
            assert response is not None
            assert isinstance(response, dict)
    
    def test_tool_registration(self, document_agent):
        """Test that all tools are properly registered"""
        tools = document_agent.get_available_tools()
        
        expected_tools = [
            'process_document',
            'extract_bounding_boxes',
            'analyze_content',
            'compare_documents'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, document_agent, mock_memory_service):
        """Test memory service integration"""
        # Test memory search for previous document analyses
        mock_memory_service.search_memories.return_value = [
            {
                'content': 'Previous document analysis result',
                'metadata': {
                    'analysis_type': 'document_processing',
                    'document_path': '/tmp/test.pdf',
                    'quality_score': 0.88
                }
            }
        ]
        
        memories = await document_agent.memory_service.search_memories(
            query="document processing",
            limit=5
        )
        
        assert len(memories) == 1
        mock_memory_service.search_memories.assert_called_once()
    
    def test_error_handling(self, document_agent):
        """Test error handling for invalid inputs"""
        # Test with non-existent document path
        with pytest.raises((ValueError, FileNotFoundError)):
            document_agent.process_document.run_tool(
                document_path="/non/existent/file.pdf"
            )
    
    @pytest.mark.asyncio
    async def test_multi_format_support(self, document_agent):
        """Test support for multiple document formats"""
        test_formats = [
            '/tmp/document.pdf',
            '/tmp/document.docx',
            '/tmp/document.xlsx',
            '/tmp/document.pptx'
        ]
        
        for doc_path in test_formats:
            with patch.object(document_agent.document_processor, 'process_document') as mock_process:
                mock_process.return_value = {
                    'document_metadata': {
                        'file_type': doc_path.split('.')[-1],
                        'processing_status': 'completed'
                    },
                    'quality_metrics': {'overall_quality_score': 0.89}
                }
                
                result = await document_agent.process_document(document_path=doc_path)
                
                assert result['document_metadata']['processing_status'] == 'completed'
                assert result['quality_metrics']['overall_quality_score'] > 0.8
    
    @pytest.mark.asyncio
    async def test_large_document_processing(self, document_agent):
        """Test performance with large documents"""
        with patch.object(document_agent.document_processor, 'process_document') as mock_process:
            # Mock result for large document
            mock_process.return_value = {
                'document_metadata': {
                    'file_size_mb': 25.6,
                    'pages': 150,
                    'processing_time_seconds': 35.2
                },
                'content_summary': {
                    'total_text_length': 500000,
                    'paragraph_count': 2500
                },
                'quality_metrics': {
                    'overall_quality_score': 0.86
                }
            }
            
            start_time = datetime.now()
            result = await document_agent.process_document(
                document_path='/tmp/large_document.pdf'
            )
            end_time = datetime.now()
            
            # Verify performance
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time < 60.0  # Should complete within reasonable time
            assert result['document_metadata']['pages'] > 100
            assert result['content_summary']['total_text_length'] > 100000


class TestDocumentAgentIntegration:
    """Integration tests for Document agent with real-world scenarios"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_document_workflow(self, document_agent, sample_document_data):
        """Test complete document processing workflow"""
        doc_path = sample_document_data['document_path']
        
        # Step 1: Process Document
        with patch.object(document_agent.document_processor, 'process_document') as mock_process:
            mock_process.return_value = {
                'processing_status': 'completed',
                'quality_metrics': {'overall_quality_score': 0.91}
            }
            
            process_result = await document_agent.process_document(document_path=doc_path)
        
        # Step 2: Extract Bounding Boxes
        with patch.object(document_agent.document_processor, 'extract_bounding_boxes') as mock_bbox:
            mock_bbox.return_value = {
                'bounding_boxes': sample_document_data['bounding_boxes'],
                'extraction_metadata': {'extraction_confidence': 0.94}
            }
            
            bbox_result = await document_agent.extract_bounding_boxes(document_path=doc_path)
        
        # Step 3: Content Analysis
        with patch.object(document_agent.document_processor, 'analyze_content_intelligence') as mock_analyze:
            mock_analyze.return_value = {
                'content_insights': {'main_topics': sample_document_data['topics']},
                'confidence_score': 0.92
            }
            
            content_result = await document_agent.analyze_content(document_path=doc_path)
        
        # Verify workflow completion
        assert process_result['processing_status'] == 'completed'
        assert bbox_result['extraction_metadata']['extraction_confidence'] > 0.9
        assert content_result['confidence_score'] > 0.9
    
    @pytest.mark.asyncio
    async def test_docling_integration(self, document_agent):
        """Test specific Docling integration features"""
        with patch.object(document_agent.document_processor, 'extract_bounding_boxes') as mock_docling:
            # Mock Docling-specific features
            mock_docling.return_value = {
                'docling_version': '2.0.1',
                'processing_method': 'docling_advanced',
                'bounding_boxes': [
                    {
                        'element_type': 'table',
                        'bbox': [100, 200, 500, 400],
                        'docling_confidence': 0.96,
                        'table_structure': {
                            'rows': 5,
                            'columns': 4,
                            'header_detected': True
                        }
                    }
                ],
                'advanced_features': {
                    'table_structure_detection': True,
                    'formula_recognition': True,
                    'chart_analysis': True
                }
            }
            
            result = await document_agent.extract_bounding_boxes(
                document_path='/tmp/test.pdf'
            )
            
            assert 'docling_version' in result
            assert result['processing_method'] == 'docling_advanced'
            assert 'advanced_features' in result
    
    def test_concurrent_document_processing(self, document_agent):
        """Test handling of concurrent document processing"""
        # This would test thread safety and concurrent processing
        # Implementation would depend on actual concurrency requirements
        pass