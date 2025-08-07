from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from typing import List, Dict, Any, Optional
import logging
import uuid
from config.settings import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant vector database client for embeddings storage and retrieval.
    """
    
    def __init__(self):
        self.client = None
        self.collection_name = "conversation_embeddings"
    
    async def connect(self):
        """Initialize Qdrant client connection."""
        try:
            # Initialize client based on configuration
            if settings.QDRANT_API_KEY:
                self.client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    api_key=settings.QDRANT_API_KEY,
                    https=True
                )
            else:
                self.client = QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT
                )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Qdrant connection established. Collections: {len(collections.collections)}")
            
            # Create collection if it doesn't exist
            await self.create_collection()
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    async def create_collection(self):
        """Create the conversation embeddings collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name not in existing_collections:
                # Create collection with 1536 dimensions (OpenAI ada-002 embedding size)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # Standard OpenAI embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Qdrant collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to create Qdrant collection: {e}")
            raise
    
    async def store_embedding(
        self, 
        vector: List[float], 
        payload: Dict[str, Any],
        point_id: Optional[str] = None
    ) -> str:
        """
        Store an embedding vector with associated metadata.
        
        Args:
            vector: The embedding vector
            payload: Metadata associated with the vector
            point_id: Optional custom point ID
            
        Returns:
            str: The point ID of the stored vector
        """
        try:
            if point_id is None:
                point_id = str(uuid.uuid4())
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            
            logger.debug(f"Stored embedding with ID: {point_id}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            raise
    
    async def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filter_conditions: Optional filter conditions
            
        Returns:
            List[Dict]: List of similar vectors with metadata and scores
        """
        try:
            # Build query filter if provided
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            # Perform similarity search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for point in search_result:
                results.append({
                    "id": point.id,
                    "score": point.score,
                    "payload": point.payload
                })
            
            logger.debug(f"Found {len(results)} similar vectors")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar vectors: {e}")
            return []
    
    async def delete_vectors(self, point_ids: List[str]) -> bool:
        """
        Delete vectors by their point IDs.
        
        Args:
            point_ids: List of point IDs to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=point_ids
                )
            )
            
            logger.info(f"Deleted {len(point_ids)} vectors from Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dict: Collection information
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


# Global Qdrant client instance
qdrant_client = QdrantVectorStore()