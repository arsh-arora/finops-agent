"""
Configuration for Mem0 with advanced graph memory support.
"""

import os
from typing import Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import settings
from .exceptions import MemoryConfigurationError
from .models import FinOpsMemoryCategory


def get_mem0_config() -> Dict[str, Any]:
    """
    Generate Mem0 configuration with Neo4j graph store and Qdrant vector store.
    
    Returns:
        Dict: Complete Mem0 configuration
        
    Raises:
        MemoryConfigurationError: If required configuration is missing
    """
    try:
        # Set OpenRouter API key for LLM
        os.environ["OPENROUTER_API_KEY"] = "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"
        
        config = {
            # Vector store configuration (Qdrant)
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": settings.QDRANT_HOST,
                    "port": settings.QDRANT_PORT,
                    "collection_name": "finops_memories",
                    "embedding_model_dims": 384  # Match HuggingFace model dimensions
                }
            },
            
            # Graph store configuration (Neo4j) 
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": settings.NEO4J_URI,
                    "username": settings.NEO4J_USERNAME, 
                    "password": settings.NEO4J_PASSWORD
                }
            },
            
            # LLM configuration for entity extraction (using OpenRouter)
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "openai/gpt-oss-20b:free",
                    "api_key": os.environ["OPENROUTER_API_KEY"],
                    "openai_base_url": "https://openrouter.ai/api/v1",
                    "temperature": 0.2,
                    "max_tokens": 2000,
                    "top_p": 1.0
                }
            },
            
            # Embedder configuration (using Hugging Face)
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            },
            
            # Memory storage location
            "history_db_path": "/tmp/finops_memory_history.db",
            
            # Enable graph memory features
            "version": "v1.1",
            
            # Custom prompts for FinOps-specific entity extraction
            "custom_fact_extraction_prompt": get_finops_fact_extraction_prompt(),
            "custom_update_memory_prompt": get_finops_memory_update_prompt()
        }
        
        # Add Qdrant API key if available (for cloud deployment)
        if hasattr(settings, 'QDRANT_API_KEY') and settings.QDRANT_API_KEY:
            config["vector_store"]["config"]["api_key"] = settings.QDRANT_API_KEY
            config["vector_store"]["config"]["https"] = True
        
        # Validate required configuration
        _validate_config(config)
        
        return config
        
    except Exception as e:
        raise MemoryConfigurationError(f"Failed to generate Mem0 configuration: {e}")


def get_finops_graph_config() -> Dict[str, Any]:
    """
    Get Neo4j-specific configuration for FinOps graph memory with custom entity extraction.
    
    Returns:
        Dict: Neo4j graph store configuration with custom prompts
    """
    return {
        "provider": "neo4j",
        "config": {
            "url": settings.NEO4J_URI,
            "username": settings.NEO4J_USERNAME,
            "password": settings.NEO4J_PASSWORD
        },
        "custom_prompt": get_finops_entity_extraction_prompt()
    }


def get_finops_entity_extraction_prompt() -> str:
    """
    Custom prompt for extracting FinOps-specific entities and relationships.
    
    Returns:
        str: Custom extraction prompt for graph memory
    """
    return """
    Extract and identify FinOps-specific entities and their relationships from the conversation:

    ENTITIES TO EXTRACT:
    - Financial Metrics: costs, budgets, spending, savings, ROI
    - Cloud Resources: instances, services, regions, availability zones
    - Organizations: departments, teams, cost centers, business units
    - Time Periods: months, quarters, fiscal years, billing cycles
    - Vendors: AWS, Azure, GCP, cloud providers, SaaS tools
    - Optimization Actions: rightsizing, scheduling, reserved instances
    - Governance Policies: tagging, approval workflows, budgets limits

    RELATIONSHIPS TO CAPTURE:
    - BELONGS_TO: resources belong to departments/cost centers
    - COSTS: services cost money in specific time periods
    - OPTIMIZES: actions optimize specific resources or costs
    - MANAGES: people/teams manage budgets or resources
    - EXCEEDS/UNDER: actual spending vs budget relationships
    - RECOMMENDS: optimization recommendations for resources

    Focus on financial and operational context that will help future FinOps conversations.
    Only extract entities that are relevant to cloud financial operations.
    """


def get_finops_fact_extraction_prompt() -> str:
    """
    Custom prompt for extracting FinOps facts from conversations.
    
    Returns:
        str: Custom fact extraction prompt
    """
    return """
    Extract key FinOps facts and insights from this conversation:
    
    PRIORITIZE:
    - Cost optimization insights and specific recommendations
    - Budget targets, thresholds, and spending patterns
    - Resource utilization metrics and efficiency data
    - User preferences for reporting, alerts, and dashboards
    - Action items with owners and deadlines
    - Policy decisions and governance rules
    - Vendor negotiations and contract details
    
    CAPTURE:
    - Numerical data: costs, percentages, quantities, dates
    - Decisions made and their rationale
    - Problems identified and solutions proposed
    - User preferences and configuration choices
    
    Format facts clearly for future retrieval and context.
    """


def get_finops_memory_update_prompt() -> str:
    """
    Custom prompt for updating existing FinOps memories.
    
    Returns:
        str: Custom memory update prompt
    """
    return """
    Update existing FinOps memory with new information while preserving context:
    
    MERGE STRATEGY:
    - Consolidate new cost insights with existing financial data
    - Update budget tracking preferences and thresholds
    - Combine optimization recommendations without duplication
    - Maintain historical context while adding new developments
    - Preserve user preferences and configuration settings
    
    PRESERVE:
    - Historical cost trends and patterns
    - Previous optimization outcomes and lessons learned
    - Established governance policies and exceptions
    - User workflow preferences and custom configurations
    
    Ensure updated memory maintains financial accuracy and operational context.
    """


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate Mem0 configuration completeness.
    
    Args:
        config: Mem0 configuration dictionary
        
    Raises:
        MemoryConfigurationError: If configuration is invalid
    """
    required_keys = ["vector_store", "graph_store", "llm", "embedder"]
    
    for key in required_keys:
        if key not in config:
            raise MemoryConfigurationError(f"Missing required configuration key: {key}")
    
    # Validate OpenRouter API key
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key or not openrouter_api_key.startswith("sk-or-"):
        raise MemoryConfigurationError("OpenRouter API key is required for LLM operation")
    
    # Validate Neo4j connection details
    neo4j_config = config["graph_store"]["config"]
    required_neo4j = ["url", "username", "password"]
    for key in required_neo4j:
        if not neo4j_config.get(key):
            raise MemoryConfigurationError(f"Neo4j {key} is required for graph memory")
    
    # Validate Qdrant connection details  
    qdrant_config = config["vector_store"]["config"]
    if not qdrant_config.get("host") or not qdrant_config.get("port"):
        raise MemoryConfigurationError("Qdrant host and port are required for vector storage")