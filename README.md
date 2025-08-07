# FinOps Agent Chat - Phase 1: Core Infrastructure

This is the core infrastructure implementation for the Multi-Agent FinOps Chat system. Phase 1 provides the foundational database layers and WebSocket communication framework.

## Architecture Overview

```
WebSocket Frame → FastAPI → JWT Auth → ConversationEvent (PostgreSQL) → Agent Router (Phase 2)
                                                  ↓
Neo4j ← Memory Persistence ← Agent Tools ← Memory Retrieval ← Redis/Qdrant
```

## Core Components Implemented

### Database Layer
- **PostgreSQL**: Conversation events and structured data storage
- **Redis**: Celery task queue and caching layer  
- **Qdrant**: Vector embeddings storage for semantic search
- **Neo4j**: Knowledge graph for relationship-based memory retrieval

### WebSocket Foundation
- **FastAPI WebSocket endpoint**: `/api/v1/ws/chat` with JWT authentication
- **Connection Management**: Multi-user, multi-conversation session handling
- **Message Processing**: Real-time message routing and event persistence
- **ConversationEvent Model**: Complete SQLAlchemy model for conversation tracking

### Authentication System
- **JWT Authentication**: Secure token-based WebSocket authentication
- **User Management**: Basic user context and session management
- **Token Validation**: Middleware for WebSocket connection security

## Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 2. Start with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f app
```

### 3. Manual Setup (Development)

```bash
# Install dependencies
pip install -r requirements/base.txt

# Start databases
docker-compose up -d postgres redis qdrant neo4j

# Initialize databases
python -m src.database.init_db

# Run application
python main.py
```

## API Endpoints

### Health Check
- `GET /health` - Service health status
- `GET /` - API information

### Authentication
- `POST /auth/token` - Create JWT token (testing only)

### WebSocket
- `WS /api/v1/ws/chat?token={jwt_token}&conversation_id={optional}` - Real-time chat

## WebSocket Message Format

### Client → Server
```json
{
  "type": "user_message",
  "text": "Analyze my AWS costs for last month",
  "metadata": {"source": "web"},
  "timestamp": "2024-01-01T10:00:00Z",
  "message_id": "uuid-here"
}
```

### Server → Client
```json
{
  "type": "agent_response",
  "event_id": "event-uuid",
  "conversation_id": "conversation-uuid",
  "agent_type": "finops",
  "message": "I'll analyze your AWS costs...",
  "timestamp": "2024-01-01T10:00:01Z"
}
```

## Database Schema

### ConversationEvent Table
- `id`: Primary key
- `event_id`: Unique event identifier
- `conversation_id`: Groups related messages
- `user_id`: User identifier
- `agent_type`: finops, github, document, research
- `event_type`: user_message, agent_response, system_event
- `message_text`: Message content
- `metadata`: JSON metadata
- `created_at`, `updated_at`: Timestamps
- `processing_time_ms`: Performance tracking

## Configuration

All configuration is handled through environment variables:

```bash
# Application
DEBUG=false
JWT_SECRET_KEY=your-secret-key

# Databases
POSTGRES_URL=postgresql+asyncpg://...
REDIS_URL=redis://localhost:6379/0
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

## Testing WebSocket Connection

```javascript
// Connect to WebSocket
const token = "your-jwt-token";
const ws = new WebSocket(`ws://localhost:8000/api/v1/ws/chat?token=${token}`);

// Send message
ws.send(JSON.stringify({
  type: "user_message",
  text: "Help me optimize my cloud costs",
  timestamp: new Date().toISOString()
}));

// Receive messages
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log("Received:", message);
};
```

## Development

### Database Migrations
```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations  
alembic upgrade head
```

### Health Checks
```bash
# Check all services
curl http://localhost:8000/health

# WebSocket stats
curl http://localhost:8000/api/v1/ws/stats
```

## Next Phase

Phase 2 will implement:
- **Memory System**: Mem0 integration with Neo4j/Qdrant backends
- **Agent Framework**: HardenedAgent base class and specialized agents
- **LangGraph Integration**: Dynamic graph compilation and execution
- **Cross-agent Orchestration**: Multi-agent workflow management

The current Phase 1 infrastructure provides the solid foundation needed for these advanced features.

## Troubleshooting

### Database Connection Issues
```bash
# Test PostgreSQL
docker exec -it finops_postgres psql -U finops -d finops_chat -c "SELECT 1;"

# Test Redis
docker exec -it finops_redis redis-cli ping

# Test Qdrant
curl http://localhost:6333/health

# Test Neo4j
curl http://localhost:7474/
```

### WebSocket Issues
- Ensure JWT token is valid and not expired
- Check CORS configuration for web clients
- Verify all database services are healthy
- Check application logs: `docker-compose logs -f app`# finops-agent
