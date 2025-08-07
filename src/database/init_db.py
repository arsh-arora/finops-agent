"""
Database initialization script for the FinOps Agent Chat system.

This script handles:
1. Database connection testing
2. Table creation
3. Index creation
4. Initial data seeding (if needed)
"""

import asyncio
import logging
from .connection import create_tables, check_database_connection
from .redis_client import redis_client
from .qdrant_client import qdrant_client
from .neo4j_client import neo4j_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_postgresql():
    """Initialize PostgreSQL database."""
    logger.info("Initializing PostgreSQL...")
    
    # Check connection
    if not await check_database_connection():
        raise ConnectionError("Failed to connect to PostgreSQL")
    
    # Create tables
    await create_tables()
    logger.info("PostgreSQL initialization completed")


async def initialize_redis():
    """Initialize Redis connection."""
    logger.info("Initializing Redis...")
    await redis_client.connect()
    logger.info("Redis initialization completed")


async def initialize_qdrant():
    """Initialize Qdrant vector database."""
    logger.info("Initializing Qdrant...")
    await qdrant_client.connect()
    logger.info("Qdrant initialization completed")


async def initialize_neo4j():
    """Initialize Neo4j graph database."""
    logger.info("Initializing Neo4j...")
    await neo4j_client.connect()
    logger.info("Neo4j initialization completed")


async def initialize_all_databases():
    """Initialize all database systems."""
    logger.info("Starting database initialization...")
    
    try:
        # Initialize databases in sequence
        await initialize_postgresql()
        await initialize_redis()
        await initialize_qdrant()
        await initialize_neo4j()
        
        logger.info("All databases initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


async def cleanup_connections():
    """Clean up all database connections."""
    logger.info("Cleaning up database connections...")
    
    try:
        await redis_client.disconnect()
        await neo4j_client.disconnect()
        logger.info("Database cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


async def health_check():
    """Perform health check on all database systems."""
    logger.info("Performing database health check...")
    
    health_status = {
        "postgresql": False,
        "redis": False,
        "qdrant": False,
        "neo4j": False
    }
    
    # PostgreSQL health check
    try:
        health_status["postgresql"] = await check_database_connection()
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
    
    # Redis health check
    try:
        if redis_client.redis_client:
            await redis_client.redis_client.ping()
            health_status["redis"] = True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
    
    # Qdrant health check
    try:
        if qdrant_client.client:
            collections = qdrant_client.client.get_collections()
            health_status["qdrant"] = True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
    
    # Neo4j health check
    try:
        if neo4j_client.driver:
            async with neo4j_client.driver.session() as session:
                result = await session.run("RETURN 1")
                await result.single()
                health_status["neo4j"] = True
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
    
    # Log results
    all_healthy = all(health_status.values())
    if all_healthy:
        logger.info("All databases are healthy!")
    else:
        unhealthy = [db for db, status in health_status.items() if not status]
        logger.warning(f"Unhealthy databases: {unhealthy}")
    
    return health_status


if __name__ == "__main__":
    # Run initialization
    asyncio.run(initialize_all_databases())