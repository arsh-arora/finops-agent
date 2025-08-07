"""
Integration tests for database components working together.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import uuid

from src.database.init_db import (
    initialize_postgresql,
    initialize_redis,
    initialize_qdrant,
    initialize_neo4j,
    initialize_all_databases,
    health_check
)


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseInitialization:
    """Test database initialization integration."""
    
    async def test_initialize_postgresql_success(self, db_engine):
        """Test successful PostgreSQL initialization."""
        with patch('src.database.init_db.check_database_connection', return_value=True):
            with patch('src.database.init_db.create_tables') as mock_create:
                mock_create.return_value = None
                
                await initialize_postgresql()
                
                mock_create.assert_called_once()
    
    async def test_initialize_postgresql_failure(self):
        """Test PostgreSQL initialization failure."""
        with patch('src.database.init_db.check_database_connection', return_value=False):
            
            with pytest.raises(ConnectionError, match="Failed to connect to PostgreSQL"):
                await initialize_postgresql()
    
    async def test_initialize_redis_success(self, mock_redis):
        """Test successful Redis initialization."""
        with patch('src.database.init_db.redis_client', mock_redis):
            mock_redis.connect = AsyncMock()
            
            await initialize_redis()
            
            mock_redis.connect.assert_called_once()
    
    async def test_initialize_qdrant_success(self, mock_qdrant):
        """Test successful Qdrant initialization."""
        with patch('src.database.init_db.qdrant_client', mock_qdrant):
            mock_qdrant.connect = AsyncMock()
            
            await initialize_qdrant()
            
            mock_qdrant.connect.assert_called_once()
    
    async def test_initialize_neo4j_success(self, mock_neo4j):
        """Test successful Neo4j initialization."""
        with patch('src.database.init_db.neo4j_client', mock_neo4j):
            mock_neo4j.connect = AsyncMock()
            
            await initialize_neo4j()
            
            mock_neo4j.connect.assert_called_once()
    
    async def test_initialize_all_databases_success(
        self,
        db_engine,
        mock_redis,
        mock_qdrant,
        mock_neo4j
    ):
        """Test successful initialization of all databases."""
        with patch('src.database.init_db.check_database_connection', return_value=True):
            with patch('src.database.init_db.create_tables'):
                with patch('src.database.init_db.redis_client', mock_redis):
                    with patch('src.database.init_db.qdrant_client', mock_qdrant):
                        with patch('src.database.init_db.neo4j_client', mock_neo4j):
                            # Setup mocks
                            mock_redis.connect = AsyncMock()
                            mock_qdrant.connect = AsyncMock()
                            mock_neo4j.connect = AsyncMock()
                            
                            result = await initialize_all_databases()
                            
                            assert result is True
                            mock_redis.connect.assert_called_once()
                            mock_qdrant.connect.assert_called_once()
                            mock_neo4j.connect.assert_called_once()
    
    async def test_initialize_all_databases_failure(self):
        """Test initialization failure handling."""
        with patch('src.database.init_db.check_database_connection', side_effect=Exception("DB Error")):
            
            result = await initialize_all_databases()
            
            assert result is False
    
    async def test_health_check_all_healthy(
        self,
        db_engine,
        mock_redis,
        mock_qdrant,
        mock_neo4j
    ):
        """Test health check when all systems are healthy."""
        with patch('src.database.init_db.check_database_connection', return_value=True):
            with patch('src.database.init_db.redis_client', mock_redis):
                with patch('src.database.init_db.qdrant_client', mock_qdrant):
                    with patch('src.database.init_db.neo4j_client', mock_neo4j):
                        # Setup healthy mocks
                        mock_redis.redis_client = AsyncMock()
                        mock_redis.redis_client.ping.return_value = True
                        
                        mock_qdrant.client = MagicMock()
                        mock_qdrant.client.get_collections.return_value = MagicMock()
                        
                        mock_neo4j.driver = AsyncMock()
                        mock_session = AsyncMock()
                        mock_result = AsyncMock()
                        mock_result.single.return_value = {"test": 1}
                        mock_session.run.return_value = mock_result
                        mock_neo4j.driver.session.return_value.__aenter__.return_value = mock_session
                        
                        health_status = await health_check()
                        
                        assert health_status["postgresql"] is True
                        assert health_status["redis"] is True
                        assert health_status["qdrant"] is True
                        assert health_status["neo4j"] is True
    
    async def test_health_check_mixed_health(self, mock_redis, mock_qdrant, mock_neo4j):
        """Test health check with mixed system health."""
        with patch('src.database.init_db.check_database_connection', return_value=True):
            with patch('src.database.init_db.redis_client', mock_redis):
                with patch('src.database.init_db.qdrant_client', mock_qdrant):
                    with patch('src.database.init_db.neo4j_client', mock_neo4j):
                        # PostgreSQL healthy, Redis unhealthy, Qdrant healthy, Neo4j unhealthy
                        mock_redis.redis_client = None  # Not connected
                        
                        mock_qdrant.client = MagicMock()
                        mock_qdrant.client.get_collections.return_value = MagicMock()
                        
                        mock_neo4j.driver = None  # Not connected
                        
                        health_status = await health_check()
                        
                        assert health_status["postgresql"] is True
                        assert health_status["redis"] is False
                        assert health_status["qdrant"] is True
                        assert health_status["neo4j"] is False


@pytest.mark.integration
@pytest.mark.database
class TestConversationEventIntegration:
    """Test ConversationEvent model integration with database operations."""
    
    async def test_conversation_event_lifecycle(self, db_session):
        """Test complete ConversationEvent lifecycle."""
        from src.database.models import ConversationEvent, ConversationEventType, AgentType
        from sqlalchemy import select, update, delete
        
        # Create event
        conversation_id = str(uuid.uuid4())
        event = ConversationEvent(
            conversation_id=conversation_id,
            user_id="lifecycle_test_user",
            event_type=ConversationEventType.USER_MESSAGE,
            message_text="Test lifecycle message",
            event_metadata={"test": "data"}
        )
        
        db_session.add(event)
        await db_session.commit()
        await db_session.refresh(event)
        
        # Verify creation
        assert event.id is not None
        assert event.event_id is not None
        assert event.created_at is not None
        
        # Read event
        result = await db_session.execute(
            select(ConversationEvent).where(ConversationEvent.id == event.id)
        )
        retrieved_event = result.scalar_one()
        
        assert retrieved_event.conversation_id == conversation_id
        assert retrieved_event.message_text == "Test lifecycle message"
        assert retrieved_event.event_metadata["test"] == "data"
        
        # Update event
        await db_session.execute(
            update(ConversationEvent)
            .where(ConversationEvent.id == event.id)
            .values(message_text="Updated message")
        )
        await db_session.commit()
        
        # Verify update
        result = await db_session.execute(
            select(ConversationEvent).where(ConversationEvent.id == event.id)
        )
        updated_event = result.scalar_one()
        assert updated_event.message_text == "Updated message"
        
        # Delete event
        await db_session.execute(
            delete(ConversationEvent).where(ConversationEvent.id == event.id)
        )
        await db_session.commit()
        
        # Verify deletion
        result = await db_session.execute(
            select(ConversationEvent).where(ConversationEvent.id == event.id)
        )
        deleted_event = result.scalar_one_or_none()
        assert deleted_event is None
    
    async def test_conversation_event_queries(self, db_session, create_conversation_event):
        """Test complex ConversationEvent queries."""
        from src.database.models import ConversationEvent, ConversationEventType, AgentType
        from sqlalchemy import select, and_, or_, func
        
        conversation_id = str(uuid.uuid4())
        user_id = "query_test_user"
        
        # Create test events
        events_data = [
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "event_type": ConversationEventType.USER_MESSAGE,
                "message_text": "First user message",
                "agent_type": None
            },
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "event_type": ConversationEventType.AGENT_RESPONSE,
                "message_text": "Agent response",
                "agent_type": AgentType.FINOPS
            },
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "event_type": ConversationEventType.USER_MESSAGE,
                "message_text": "Second user message",
                "agent_type": None
            },
            {
                "conversation_id": str(uuid.uuid4()),  # Different conversation
                "user_id": user_id,
                "event_type": ConversationEventType.USER_MESSAGE,
                "message_text": "Different conversation message",
                "agent_type": None
            }
        ]
        
        created_events = []
        for event_data in events_data:
            event = await create_conversation_event(db_session, **event_data)
            created_events.append(event)
        
        # Test conversation-specific query
        result = await db_session.execute(
            select(ConversationEvent)
            .where(ConversationEvent.conversation_id == conversation_id)
            .order_by(ConversationEvent.created_at)
        )
        conversation_events = result.scalars().all()
        
        assert len(conversation_events) == 3
        assert all(event.conversation_id == conversation_id for event in conversation_events)
        
        # Test user + event type query
        result = await db_session.execute(
            select(ConversationEvent)
            .where(and_(
                ConversationEvent.user_id == user_id,
                ConversationEvent.event_type == ConversationEventType.USER_MESSAGE
            ))
        )
        user_messages = result.scalars().all()
        
        assert len(user_messages) == 3  # 2 from main conversation + 1 from other
        assert all(event.event_type == ConversationEventType.USER_MESSAGE for event in user_messages)
        
        # Test agent response query
        result = await db_session.execute(
            select(ConversationEvent)
            .where(and_(
                ConversationEvent.user_id == user_id,
                ConversationEvent.agent_type == AgentType.FINOPS
            ))
        )
        finops_responses = result.scalars().all()
        
        assert len(finops_responses) == 1
        assert finops_responses[0].agent_type == AgentType.FINOPS
        
        # Test count query
        result = await db_session.execute(
            select(func.count(ConversationEvent.id))
            .where(ConversationEvent.user_id == user_id)
        )
        total_events = result.scalar()
        
        assert total_events == 4
    
    async def test_conversation_event_metadata_queries(self, db_session, create_conversation_event):
        """Test queries involving JSON metadata."""
        from src.database.models import ConversationEvent
        from sqlalchemy import select, text
        
        # Create events with different metadata
        event1 = await create_conversation_event(
            db_session,
            user_id="metadata_user",
            message_text="Mobile message",
            event_metadata={"source": "mobile", "version": "1.2.3"}
        )
        
        event2 = await create_conversation_event(
            db_session,
            user_id="metadata_user",
            message_text="Web message",
            event_metadata={"source": "web", "browser": "chrome"}
        )
        
        event3 = await create_conversation_event(
            db_session,
            user_id="metadata_user",
            message_text="API message",
            event_metadata={"source": "api", "client_id": "test_client"}
        )
        
        # Query by JSON field (SQLite-compatible approach)
        # Note: This is a simplified approach for testing
        # In production PostgreSQL, you'd use proper JSON operators
        result = await db_session.execute(
            select(ConversationEvent)
            .where(ConversationEvent.user_id == "metadata_user")
        )
        events = result.scalars().all()
        
        # Filter in Python for SQLite compatibility
        mobile_events = [e for e in events if e.event_metadata.get("source") == "mobile"]
        web_events = [e for e in events if e.event_metadata.get("source") == "web"]
        api_events = [e for e in events if e.event_metadata.get("source") == "api"]
        
        assert len(mobile_events) == 1
        assert len(web_events) == 1
        assert len(api_events) == 1
        
        assert mobile_events[0].event_metadata["version"] == "1.2.3"
        assert web_events[0].event_metadata["browser"] == "chrome"
        assert api_events[0].event_metadata["client_id"] == "test_client"


@pytest.mark.integration
@pytest.mark.database
@pytest.mark.slow
class TestDatabasePerformance:
    """Test database performance and concurrent operations."""
    
    async def test_concurrent_event_creation(self, db_session, create_conversation_event):
        """Test concurrent ConversationEvent creation."""
        conversation_id = str(uuid.uuid4())
        
        # Create multiple events concurrently
        tasks = []
        for i in range(10):
            task = create_conversation_event(
                db_session,
                conversation_id=conversation_id,
                user_id=f"concurrent_user_{i}",
                message_text=f"Concurrent message {i}"
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        events = await asyncio.gather(*tasks)
        
        # Verify all events were created
        assert len(events) == 10
        assert all(event.id is not None for event in events)
        assert all(event.conversation_id == conversation_id for event in events)
        
        # Verify unique event IDs
        event_ids = [event.event_id for event in events]
        assert len(set(event_ids)) == 10  # All unique
    
    async def test_large_metadata_handling(self, db_session, create_conversation_event):
        """Test handling of large metadata objects."""
        # Create event with large metadata
        large_metadata = {
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "request_headers": {f"header_{i}": f"value_{i}" for i in range(100)},
            "custom_data": {f"field_{i}": f"data_{i}" * 10 for i in range(50)},
            "nested_object": {
                "level1": {
                    "level2": {
                        "level3": {
                            "data": ["item"] * 100
                        }
                    }
                }
            }
        }
        
        event = await create_conversation_event(
            db_session,
            user_id="large_metadata_user",
            message_text="Event with large metadata",
            event_metadata=large_metadata
        )
        
        # Verify event was created successfully
        assert event.id is not None
        
        # Verify metadata integrity
        assert len(event.event_metadata["request_headers"]) == 100
        assert len(event.event_metadata["custom_data"]) == 50
        assert len(event.event_metadata["nested_object"]["level1"]["level2"]["level3"]["data"]) == 100