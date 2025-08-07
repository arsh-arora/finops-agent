from neo4j import AsyncGraphDatabase, AsyncSession
from typing import Dict, List, Any, Optional
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """
    Neo4j graph database client for knowledge graph relationships.
    """
    
    def __init__(self):
        self.driver = None
    
    async def connect(self):
        """Initialize Neo4j driver connection."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                max_connection_lifetime=3600,  # 1 hour
                max_connection_pool_size=10,
                connection_acquisition_timeout=60
            )
            
            # Test connection
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as test")
                record = await result.single()
                if record and record["test"] == 1:
                    logger.info("Neo4j connection established successfully")
                
            # Initialize schema
            await self.create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def disconnect(self):
        """Close Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")
    
    async def create_indexes(self):
        """Create necessary indexes for performance."""
        indexes = [
            "CREATE INDEX user_id_index IF NOT EXISTS FOR (u:User) ON (u.user_id)",
            "CREATE INDEX conversation_id_index IF NOT EXISTS FOR (c:Conversation) ON (c.conversation_id)",
            "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)"
        ]
        
        try:
            async with self.driver.session() as session:
                for index_query in indexes:
                    await session.run(index_query)
            logger.info("Neo4j indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create Neo4j indexes: {e}")
    
    async def create_user_node(
        self, 
        user_id: str, 
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        Create or update a user node.
        
        Args:
            user_id: Unique user identifier
            properties: Additional user properties
            
        Returns:
            bool: True if successful
        """
        try:
            props = properties or {}
            props["user_id"] = user_id
            
            query = """
            MERGE (u:User {user_id: $user_id})
            SET u += $properties
            RETURN u
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, user_id=user_id, properties=props)
                record = await result.single()
                return record is not None
                
        except Exception as e:
            logger.error(f"Failed to create user node: {e}")
            return False
    
    async def create_memory_node(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Create a memory node and link it to a user.
        
        Args:
            memory_id: Unique memory identifier
            user_id: User who owns this memory
            content: Memory content
            metadata: Additional memory metadata
            
        Returns:
            bool: True if successful
        """
        try:
            props = metadata or {}
            props.update({
                "memory_id": memory_id,
                "content": content,
                "timestamp": "datetime()"  # Neo4j function
            })
            
            query = """
            MATCH (u:User {user_id: $user_id})
            CREATE (m:Memory)
            SET m += $properties
            CREATE (u)-[:HAS_MEMORY]->(m)
            RETURN m
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query, 
                    user_id=user_id, 
                    properties=props
                )
                record = await result.single()
                return record is not None
                
        except Exception as e:
            logger.error(f"Failed to create memory node: {e}")
            return False
    
    async def create_entity_relationship(
        self,
        entity_name: str,
        entity_type: str,
        memory_id: str,
        relationship_type: str = "MENTIONED_IN"
    ) -> bool:
        """
        Create an entity node and link it to a memory.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of entity (person, organization, concept, etc.)
            memory_id: Memory that mentions this entity
            relationship_type: Type of relationship
            
        Returns:
            bool: True if successful
        """
        try:
            query = """
            MATCH (m:Memory {memory_id: $memory_id})
            MERGE (e:Entity {name: $entity_name})
            SET e.type = $entity_type
            MERGE (e)-[:$relationship_type]->(m)
            RETURN e, m
            """.replace("$relationship_type", relationship_type)
            
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    memory_id=memory_id
                )
                record = await result.single()
                return record is not None
                
        except Exception as e:
            logger.error(f"Failed to create entity relationship: {e}")
            return False
    
    async def find_related_memories(
        self,
        user_id: str,
        entity_names: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find memories related to a user and optionally specific entities.
        
        Args:
            user_id: User to search memories for
            entity_names: Optional list of entity names to filter by
            limit: Maximum number of memories to return
            
        Returns:
            List[Dict]: List of related memories with metadata
        """
        try:
            if entity_names:
                query = """
                MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
                MATCH (e:Entity)-[:MENTIONED_IN]->(m)
                WHERE e.name IN $entity_names
                RETURN DISTINCT m, collect(e.name) as entities
                ORDER BY m.timestamp DESC
                LIMIT $limit
                """
                params = {
                    "user_id": user_id,
                    "entity_names": entity_names,
                    "limit": limit
                }
            else:
                query = """
                MATCH (u:User {user_id: $user_id})-[:HAS_MEMORY]->(m:Memory)
                OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(m)
                RETURN m, collect(e.name) as entities
                ORDER BY m.timestamp DESC
                LIMIT $limit
                """
                params = {
                    "user_id": user_id,
                    "limit": limit
                }
            
            async with self.driver.session() as session:
                result = await session.run(query, **params)
                records = await result.data()
                
                memories = []
                for record in records:
                    memory_node = record["m"]
                    entities = [e for e in record["entities"] if e is not None]
                    
                    memories.append({
                        "memory_id": memory_node["memory_id"],
                        "content": memory_node["content"],
                        "timestamp": memory_node.get("timestamp"),
                        "entities": entities,
                        "metadata": {k: v for k, v in memory_node.items() 
                                   if k not in ["memory_id", "content", "timestamp"]}
                    })
                
                logger.debug(f"Found {len(memories)} related memories for user {user_id}")
                return memories
                
        except Exception as e:
            logger.error(f"Failed to find related memories: {e}")
            return []
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive context for a user including memories and entities.
        
        Args:
            user_id: User to get context for
            
        Returns:
            Dict: User context information
        """
        try:
            query = """
            MATCH (u:User {user_id: $user_id})
            OPTIONAL MATCH (u)-[:HAS_MEMORY]->(m:Memory)
            OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(m)
            RETURN u,
                   count(DISTINCT m) as memory_count,
                   collect(DISTINCT e.name) as entities,
                   collect(DISTINCT e.type) as entity_types
            """
            
            async with self.driver.session() as session:
                result = await session.run(query, user_id=user_id)
                record = await result.single()
                
                if not record:
                    return {"user_id": user_id, "exists": False}
                
                user_node = record["u"]
                context = {
                    "user_id": user_id,
                    "exists": True,
                    "memory_count": record["memory_count"],
                    "entities": [e for e in record["entities"] if e is not None],
                    "entity_types": [t for t in record["entity_types"] if t is not None],
                    "user_properties": dict(user_node)
                }
                
                return context
                
        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return {"user_id": user_id, "exists": False, "error": str(e)}


# Global Neo4j client instance
neo4j_client = Neo4jGraphStore()