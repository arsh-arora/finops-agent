"""
FinOps Agent Chat - Main FastAPI Application

This is the main entry point for the FinOps Agent Chat system.
It sets up the FastAPI application, initializes database connections,
and configures WebSocket endpoints for real-time agent communication.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import uvicorn

# Import configuration and database components
from config.settings import settings
from src.database.init_db import initialize_all_databases, cleanup_connections, health_check

# Import WebSocket router
from src.websocket import websocket_router

# Import auth components
from src.auth import create_access_token, User
from datetime import timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for database connections
    and other resources.
    """
    # Startup
    logger.info("Starting FinOps Agent Chat application...")
    
    # Initialize all database connections
    success = await initialize_all_databases()
    if not success:
        logger.error("Failed to initialize databases - exiting")
        raise RuntimeError("Database initialization failed")
    
    logger.info("Application startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FinOps Agent Chat application...")
    await cleanup_connections()
    logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="FinOps Agent Chat API",
    description="Multi-Agent FinOps Chat System with WebSocket support",
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include WebSocket router
app.include_router(websocket_router)


# Health check endpoints
@app.get("/health")
async def health_check_endpoint():
    """
    Health check endpoint for load balancers and monitoring.
    
    Returns application health status and database connectivity.
    """
    health_status = await health_check()
    
    overall_healthy = all(health_status.values())
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "services": health_status,
        "version": "1.0.0",
        "debug_mode": settings.DEBUG
    }


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "FinOps Agent Chat API",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket_endpoint": "/api/v1/ws/chat",
        "health_check": "/health"
    }


# Authentication endpoint for testing
@app.post("/auth/token")
async def create_test_token(user_id: str):
    """
    Create a JWT token for testing purposes.
    
    In production, this would be replaced with proper authentication
    that validates user credentials.
    
    Args:
        user_id: User identifier
        
    Returns:
        JWT token for WebSocket authentication
    """
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    # Create token with user information
    token_data = {
        "sub": user_id,
        "username": f"user_{user_id}"
    }
    
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user_id": user_id
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )