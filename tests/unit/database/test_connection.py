"""
Tests for database connection management.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.connection import get_db_session, check_database_connection, create_tables


@pytest.mark.unit
@pytest.mark.database
class TestDatabaseConnection:
    """Test database connection functionality."""
    
    async def test_get_db_session_success(self, db_session):
        """Test successful database session creation."""
        async for session in get_db_session():
            assert isinstance(session, AsyncSession)
            assert session is not None
            break  # Only test first iteration
    
    async def test_get_db_session_rollback_on_error(self):
        """Test session rollback on error."""
        with patch('src.database.connection.AsyncSessionLocal') as mock_session_factory:
            mock_session = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session.close = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            mock_session_factory.return_value.__aexit__.side_effect = Exception("Test error")
            
            with pytest.raises(Exception):
                async for session in get_db_session():
                    raise Exception("Test error")
            
            mock_session.rollback.assert_called_once()
    
    async def test_check_database_connection_success(self, db_engine):
        """Test successful database connection check."""
        with patch('src.database.connection.engine', db_engine):
            result = await check_database_connection()
            assert result is True
    
    async def test_check_database_connection_failure(self):
        """Test database connection failure."""
        mock_engine = AsyncMock()
        mock_engine.begin.side_effect = Exception("Connection failed")
        
        with patch('src.database.connection.engine', mock_engine):
            result = await check_database_connection()
            assert result is False
    
    async def test_create_tables_success(self, db_engine):
        """Test successful table creation."""
        with patch('src.database.connection.engine', db_engine):
            await create_tables()
            # If no exception is raised, the test passes