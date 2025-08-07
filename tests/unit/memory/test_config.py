"""
Tests for Memory System configuration.
"""

import pytest
from unittest.mock import patch, Mock
import os

from src.memory.config import (
    get_mem0_config,
    get_finops_graph_config,
    get_finops_entity_extraction_prompt,
    get_finops_fact_extraction_prompt,
    get_finops_memory_update_prompt,
    _validate_config
)
from src.memory.exceptions import MemoryConfigurationError


@pytest.mark.unit
@pytest.mark.memory
class TestMemoryConfiguration:
    """Test memory configuration functionality."""
    
    @patch('src.memory.config.settings')
    def test_get_mem0_config_success(self, mock_settings):
        """Test successful Mem0 configuration generation."""
        # Mock settings
        mock_settings.QDRANT_HOST = "localhost"
        mock_settings.QDRANT_PORT = 6333
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USERNAME = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"
        
        config = get_mem0_config()
        
        # Verify configuration structure
        assert "vector_store" in config
        assert "graph_store" in config
        assert "llm" in config
        assert "embedder" in config
        assert "history_db_path" in config
        assert "version" in config
        
        # Verify vector store configuration
        vector_config = config["vector_store"]
        assert vector_config["provider"] == "qdrant"
        assert vector_config["config"]["host"] == "localhost"
        assert vector_config["config"]["port"] == 6333
        assert vector_config["config"]["collection_name"] == "finops_memories"
        
        # Verify graph store configuration
        graph_config = config["graph_store"]
        assert graph_config["provider"] == "neo4j"
        assert graph_config["config"]["url"] == "bolt://localhost:7687"
        assert graph_config["config"]["username"] == "neo4j"
        assert graph_config["config"]["password"] == "password"
        
        # Verify LLM configuration
        llm_config = config["llm"]
        assert llm_config["provider"] == "groq"
        assert llm_config["config"]["model"] == "llama-3.1-8b-instant"
        assert llm_config["config"]["temperature"] == 0.2
        
        # Verify embedder configuration
        embedder_config = config["embedder"]
        assert embedder_config["provider"] == "huggingface"
        assert embedder_config["config"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
        
        # Verify version for graph memory
        assert config["version"] == "v1.1"
        
        # Verify custom prompts
        assert "custom_fact_extraction_prompt" in config
        assert "custom_update_memory_prompt" in config
    
    @patch('src.memory.config.settings')
    def test_get_mem0_config_with_qdrant_api_key(self, mock_settings):
        """Test Mem0 configuration with Qdrant API key for cloud deployment."""
        # Mock settings with Qdrant API key
        mock_settings.QDRANT_HOST = "cloud.qdrant.io"
        mock_settings.QDRANT_PORT = 6333
        mock_settings.QDRANT_API_KEY = "test-qdrant-key"
        mock_settings.NEO4J_URI = "neo4j+s://example.databases.neo4j.io"
        mock_settings.NEO4J_USERNAME = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"
        
        config = get_mem0_config()
        
        # Verify Qdrant cloud configuration
        vector_config = config["vector_store"]["config"]
        assert vector_config["api_key"] == "test-qdrant-key"
        assert vector_config["https"] is True
    
    @patch('src.memory.config.settings')
    def test_get_finops_graph_config(self, mock_settings):
        """Test FinOps graph configuration generation."""
        mock_settings.NEO4J_URI = "bolt://localhost:7687"
        mock_settings.NEO4J_USERNAME = "neo4j"
        mock_settings.NEO4J_PASSWORD = "password"
        
        config = get_finops_graph_config()
        
        assert config["provider"] == "neo4j"
        assert "config" in config
        assert "custom_prompt" in config
        
        # Verify Neo4j configuration
        neo4j_config = config["config"]
        assert neo4j_config["url"] == "bolt://localhost:7687"
        assert neo4j_config["username"] == "neo4j"
        assert neo4j_config["password"] == "password"
        
        # Verify custom prompt exists
        assert len(config["custom_prompt"]) > 0
        assert "FinOps" in config["custom_prompt"]
    
    def test_get_finops_entity_extraction_prompt(self):
        """Test FinOps entity extraction prompt generation."""
        prompt = get_finops_entity_extraction_prompt()
        
        assert len(prompt) > 0
        assert "FinOps" in prompt
        assert "ENTITIES TO EXTRACT" in prompt
        assert "RELATIONSHIPS TO CAPTURE" in prompt
        
        # Check for specific FinOps entities
        assert "Financial Metrics" in prompt
        assert "Cloud Resources" in prompt
        assert "Organizations" in prompt
        assert "Vendors" in prompt
        
        # Check for specific relationships
        assert "BELONGS_TO" in prompt
        assert "COSTS" in prompt
        assert "OPTIMIZES" in prompt
    
    def test_get_finops_fact_extraction_prompt(self):
        """Test FinOps fact extraction prompt generation."""
        prompt = get_finops_fact_extraction_prompt()
        
        assert len(prompt) > 0
        assert "FinOps" in prompt
        assert "PRIORITIZE" in prompt
        assert "CAPTURE" in prompt
        
        # Check for specific FinOps priorities
        assert "Cost optimization" in prompt
        assert "Budget targets" in prompt
        assert "Resource utilization" in prompt
        assert "User preferences" in prompt
    
    def test_get_finops_memory_update_prompt(self):
        """Test FinOps memory update prompt generation."""
        prompt = get_finops_memory_update_prompt()
        
        assert len(prompt) > 0
        assert "FinOps" in prompt
        assert "MERGE STRATEGY" in prompt
        assert "PRESERVE" in prompt
        
        # Check for specific FinOps merge strategies
        assert "cost insights" in prompt
        assert "budget tracking" in prompt
        assert "optimization recommendations" in prompt
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        valid_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333
                }
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                }
            },
            "llm": {
                "provider": "gemini",
                "config": {"model": "gemini-2.5-pro"}
            },
            "embedder": {
                "provider": "gemini", 
                "config": {"model": "models/text-embedding-004"}
            }
        }
        
        # Should not raise any exception
        _validate_config(valid_config)
    
    def test_validate_config_missing_required_keys(self):
        """Test configuration validation with missing required keys."""
        invalid_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {"host": "localhost", "port": 6333}
            },
            # Missing graph_store, llm, embedder
        }
        
        with pytest.raises(MemoryConfigurationError, match="Missing required configuration key"):
            _validate_config(invalid_config)
    
    @patch.dict(os.environ, {"GROQ_API_KEY": ""})
    def test_validate_config_missing_groq_key(self):
        """Test configuration validation with missing Groq API key."""
        invalid_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {"host": "localhost", "port": 6333}
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                }
            },
            "llm": {
                "provider": "groq",
                "config": {"model": "llama-3.1-8b-instant"}
            },
            "embedder": {
                "provider": "huggingface", 
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
            }
        }
        
        with pytest.raises(MemoryConfigurationError, match="Groq API key is required"):
            _validate_config(invalid_config)
    
    def test_validate_config_missing_neo4j_credentials(self):
        """Test configuration validation with missing Neo4j credentials."""
        invalid_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {"host": "localhost", "port": 6333}
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    # Missing username and password
                }
            },
            "llm": {
                "provider": "groq",
                "config": {"model": "llama-3.1-8b-instant"}
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
            }
        }
        
        with pytest.raises(MemoryConfigurationError, match="Neo4j .* is required"):
            _validate_config(invalid_config)
    
    def test_validate_config_missing_qdrant_host(self):
        """Test configuration validation with missing Qdrant host."""
        invalid_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    # Missing host and port
                }
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                }
            },
            "llm": {
                "provider": "groq",
                "config": {"model": "llama-3.1-8b-instant"}
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}
            }
        }
        
        with pytest.raises(MemoryConfigurationError, match="Qdrant host and port are required"):
            _validate_config(invalid_config)


@pytest.mark.unit  
@pytest.mark.memory
class TestFinOpsPrompts:
    """Test FinOps-specific prompt generation."""
    
    def test_prompts_contain_finops_keywords(self):
        """Test that all prompts contain relevant FinOps keywords."""
        entity_prompt = get_finops_entity_extraction_prompt()
        fact_prompt = get_finops_fact_extraction_prompt()
        update_prompt = get_finops_memory_update_prompt()
        
        finops_keywords = [
            "cost", "budget", "optimization", "cloud", "financial",
            "resource", "spending", "savings", "efficiency"
        ]
        
        for keyword in finops_keywords:
            # At least one prompt should contain each keyword
            assert (
                keyword.lower() in entity_prompt.lower() or
                keyword.lower() in fact_prompt.lower() or 
                keyword.lower() in update_prompt.lower()
            ), f"FinOps keyword '{keyword}' not found in any prompt"
    
    def test_prompts_are_sufficiently_detailed(self):
        """Test that prompts are detailed enough to be useful."""
        entity_prompt = get_finops_entity_extraction_prompt()
        fact_prompt = get_finops_fact_extraction_prompt()
        update_prompt = get_finops_memory_update_prompt()
        
        # Each prompt should be reasonably detailed
        assert len(entity_prompt) > 200
        assert len(fact_prompt) > 200  
        assert len(update_prompt) > 200
        
        # Should contain structured sections
        assert "ENTITIES TO EXTRACT" in entity_prompt
        assert "PRIORITIZE" in fact_prompt
        assert "MERGE STRATEGY" in update_prompt