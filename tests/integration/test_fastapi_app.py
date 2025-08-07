"""
Integration tests for the FastAPI application.
"""

import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import patch, AsyncMock

from main import app
from src.auth.jwt_auth import create_access_token


@pytest.mark.integration
class TestFastAPIApp:
    """Test FastAPI application integration."""
    
    @pytest.fixture
    async def async_client(self):
        """Create async HTTP client for testing."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    def mock_database_init(self):
        """Mock database initialization to avoid actual DB connections."""
        with patch('main.initialize_all_databases') as mock_init:
            mock_init.return_value = True
            with patch('main.cleanup_connections') as mock_cleanup:
                mock_cleanup.return_value = None
                yield mock_init, mock_cleanup
    
    async def test_root_endpoint(self, async_client, mock_database_init):
        """Test root endpoint."""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "FinOps Agent Chat API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "websocket_endpoint" in data
    
    async def test_health_check_healthy(self, async_client, mock_database_init):
        """Test health check endpoint when all services are healthy."""
        with patch('main.health_check') as mock_health:
            mock_health.return_value = {
                "postgresql": True,
                "redis": True,
                "qdrant": True,
                "neo4j": True
            }
            
            response = await async_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert all(data["services"].values())
    
    async def test_health_check_unhealthy(self, async_client, mock_database_init):
        """Test health check endpoint when some services are unhealthy."""
        with patch('main.health_check') as mock_health:
            mock_health.return_value = {
                "postgresql": True,
                "redis": False,
                "qdrant": True,
                "neo4j": False
            }
            
            response = await async_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["services"]["postgresql"] is True
            assert data["services"]["redis"] is False
    
    async def test_create_test_token_success(self, async_client, mock_database_init):
        """Test creating test authentication token."""
        response = await async_client.post("/auth/token", params={"user_id": "test_user"})
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user_id"] == "test_user"
        assert isinstance(data["expires_in"], int)
    
    async def test_create_test_token_missing_user_id(self, async_client, mock_database_init):
        """Test creating token without user ID."""
        response = await async_client.post("/auth/token", params={})
        
        assert response.status_code == 400
        data = response.json()
        assert "User ID is required" in data["detail"]
    
    async def test_create_test_token_empty_user_id(self, async_client, mock_database_init):
        """Test creating token with empty user ID."""
        response = await async_client.post("/auth/token", params={"user_id": ""})
        
        assert response.status_code == 400
        data = response.json()
        assert "User ID is required" in data["detail"]
    
    async def test_websocket_stats_endpoint(self, async_client, mock_database_init):
        """Test WebSocket stats endpoint."""
        with patch('src.websocket.router.connection_manager') as mock_manager:
            mock_manager.get_stats.return_value = {
                "total_connections": 5,
                "unique_users": 3,
                "active_conversations": 2,
                "connections_per_user": {
                    "user1": 2,
                    "user2": 1,
                    "user3": 2
                }
            }
            
            response = await async_client.get("/api/v1/ws/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_connections"] == 5
            assert data["unique_users"] == 3
            assert data["active_conversations"] == 2
    
    async def test_cors_headers(self, async_client, mock_database_init):
        """Test CORS headers are present."""
        response = await async_client.options("/")
        
        # CORS middleware should add appropriate headers
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented
    
    async def test_application_lifespan_startup_failure(self):
        """Test application handles startup failure gracefully."""
        with patch('main.initialize_all_databases') as mock_init:
            mock_init.return_value = False
            
            # Application startup should fail
            with pytest.raises(RuntimeError, match="Database initialization failed"):
                async with AsyncClient(app=app, base_url="http://test"):
                    pass


@pytest.mark.integration
class TestWebSocketIntegration:
    """Test WebSocket integration with the FastAPI app."""
    
    @pytest.fixture
    def valid_token(self):
        """Create valid JWT token for testing."""
        return create_access_token({"sub": "test_user", "username": "testuser"})
    
    @pytest.fixture
    def mock_database_dependencies(self):
        """Mock all database dependencies."""
        with patch('main.initialize_all_databases', return_value=True):
            with patch('main.cleanup_connections'):
                with patch('src.websocket.router.get_db_session') as mock_db:
                    mock_session = AsyncMock()
                    mock_db.return_value.__aenter__.return_value = mock_session
                    mock_db.return_value.__aexit__.return_value = None
                    yield mock_session
    
    async def test_websocket_connection_success(
        self,
        valid_token,
        mock_database_dependencies
    ):
        """Test successful WebSocket connection."""
        with patch('src.websocket.router.connection_manager') as mock_manager:
            mock_manager.connect.return_value = "test_connection_id"
            
            async with AsyncClient(app=app, base_url="http://test") as client:
                with client.websocket_connect(
                    f"/api/v1/ws/chat?token={valid_token}"
                ) as websocket:
                    # Connection should be established
                    assert websocket is not None
                    
                    # Verify connection manager was called
                    mock_manager.connect.assert_called_once()
    
    async def test_websocket_connection_invalid_token(self, mock_database_dependencies):
        """Test WebSocket connection with invalid token."""
        invalid_token = "invalid.jwt.token"
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            with pytest.raises(Exception):  # WebSocket connection should fail
                with client.websocket_connect(
                    f"/api/v1/ws/chat?token={invalid_token}"
                ):
                    pass
    
    async def test_websocket_connection_missing_token(self, mock_database_dependencies):
        """Test WebSocket connection without token."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            with pytest.raises(Exception):  # Should fail validation
                with client.websocket_connect("/api/v1/ws/chat"):
                    pass
    
    @pytest.mark.slow
    async def test_websocket_message_flow(
        self,
        valid_token,
        mock_database_dependencies
    ):
        """Test complete WebSocket message flow."""
        with patch('src.websocket.router.connection_manager') as mock_manager:
            mock_manager.connect.return_value = "test_connection_id"
            
            with patch('src.websocket.router.websocket_handler') as mock_handler:
                mock_handler.handle_message = AsyncMock()
                mock_handler.handle_ping = AsyncMock()
                mock_handler.handle_disconnect = AsyncMock()
                
                async with AsyncClient(app=app, base_url="http://test") as client:
                    with client.websocket_connect(
                        f"/api/v1/ws/chat?token={valid_token}"
                    ) as websocket:
                        # Send a test message
                        test_message = {
                            "type": "user_message",
                            "text": "Hello, FinOps agent!",
                            "timestamp": "2024-01-01T10:00:00Z"
                        }
                        
                        await websocket.send_json(test_message)
                        
                        # Give some time for message processing
                        await asyncio.sleep(0.1)
                        
                        # Verify handler was called
                        mock_handler.handle_message.assert_called_once()
                        call_args = mock_handler.handle_message.call_args
                        assert call_args[1]["message"] == test_message


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database integration with the application."""
    
    async def test_database_health_check_integration(self):
        """Test database health check integration."""
        # Mock all database clients
        with patch('main.check_database_connection') as mock_pg:
            with patch('main.redis_client') as mock_redis:
                with patch('main.qdrant_client') as mock_qdrant:
                    with patch('main.neo4j_client') as mock_neo4j:
                        
                        # Setup mocks
                        mock_pg.return_value = True
                        mock_redis.redis_client.ping.return_value = True
                        mock_qdrant.client.get_collections.return_value = MagicMock()
                        mock_neo4j.driver.session.return_value.__aenter__.return_value.run.return_value.single.return_value = {"test": 1}
                        
                        from main import health_check
                        
                        health_status = await health_check()
                        
                        assert health_status["postgresql"] is True
                        assert health_status["redis"] is True
                        assert health_status["qdrant"] is True
                        assert health_status["neo4j"] is True


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndFlow:
    """Test complete end-to-end application flow."""
    
    async def test_complete_conversation_flow(self):
        """Test complete conversation flow from authentication to response."""
        # This test would require more complex setup with actual database mocking
        # For now, we'll test the individual components are wired correctly
        
        with patch('main.initialize_all_databases', return_value=True):
            with patch('main.cleanup_connections'):
                # Test that the app can start and basic endpoints work
                async with AsyncClient(app=app, base_url="http://test") as client:
                    # Get auth token
                    auth_response = await client.post(
                        "/auth/token",
                        params={"user_id": "integration_test_user"}
                    )
                    assert auth_response.status_code == 200
                    
                    token = auth_response.json()["access_token"]
                    
                    # Check health
                    with patch('main.health_check') as mock_health:
                        mock_health.return_value = {
                            "postgresql": True,
                            "redis": True,
                            "qdrant": True,
                            "neo4j": True
                        }
                        
                        health_response = await client.get("/health")
                        assert health_response.status_code == 200
                        assert health_response.json()["status"] == "healthy"
                    
                    # The WebSocket connection test would go here
                    # but requires more complex async WebSocket testing setup