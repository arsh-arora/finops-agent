from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import logging
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create async engine with connection pooling
engine = create_async_engine(
    settings.POSTGRES_URL,
    poolclass=NullPool,  # For async, we use NullPool to avoid connection issues
    pool_pre_ping=True,
    echo=settings.DEBUG,  # Log SQL queries in debug mode
    connect_args={
        "server_settings": {
            "application_name": "finops_agent_chat",
        }
    }
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True,
    autocommit=False
)


async def get_db_session() -> AsyncSession:
    """
    Dependency to get database session.
    
    Returns:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def create_tables():
    """Create all database tables."""
    from .models import Base
    
    async with engine.begin() as conn:
        # Create tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")


async def check_database_connection():
    """Check if database connection is working."""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            row = result.fetchone()
            if row and row[0] == 1:
                logger.info("Database connection successful")
                return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
    return False