"""
Tests for database models.
"""

import pytest
import uuid
from datetime import datetime, timezone

from src.database.models import ConversationEvent, ConversationEventType, AgentType


@pytest.mark.unit
@pytest.mark.database
class TestConversationEventModel:
    """Test ConversationEvent model functionality."""
    
    def test_conversation_event_creation(self, sample_conversation_event_data):
        """Test creating a ConversationEvent instance."""
        event = ConversationEvent(**sample_conversation_event_data)
        
        assert event.conversation_id == sample_conversation_event_data["conversation_id"]
        assert event.user_id == sample_conversation_event_data["user_id"]
        assert event.agent_type == sample_conversation_event_data["agent_type"]
        assert event.event_type == sample_conversation_event_data["event_type"]
        assert event.message_text == sample_conversation_event_data["message_text"]
        assert event.event_metadata == sample_conversation_event_data["event_metadata"]
    
    def test_conversation_event_with_defaults(self):
        """Test ConversationEvent with minimal data."""
        event = ConversationEvent(
            user_id="test_user",
            event_type=ConversationEventType.USER_MESSAGE,
            message_text="Test message"
        )
        
        assert event.user_id == "test_user"
        assert event.event_type == ConversationEventType.USER_MESSAGE
        assert event.message_text == "Test message"
        assert event.event_metadata == {}
        assert event.agent_type is None
    
    def test_conversation_event_to_dict(self, sample_conversation_event_data):
        """Test ConversationEvent to_dict method."""
        event = ConversationEvent(**sample_conversation_event_data)
        event.id = 1
        event.event_id = str(uuid.uuid4())
        event.created_at = datetime.now(timezone.utc)
        event.updated_at = datetime.now(timezone.utc)
        
        result = event.to_dict()
        
        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["user_id"] == sample_conversation_event_data["user_id"]
        assert result["agent_type"] == sample_conversation_event_data["agent_type"].value
        assert result["event_type"] == sample_conversation_event_data["event_type"].value
        assert result["message_text"] == sample_conversation_event_data["message_text"]
        assert result["metadata"] == sample_conversation_event_data["event_metadata"]
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_conversation_event_repr(self, sample_conversation_event_data):
        """Test ConversationEvent __repr__ method."""
        event = ConversationEvent(**sample_conversation_event_data)
        event.id = 1
        
        repr_str = repr(event)
        
        assert "ConversationEvent" in repr_str
        assert "id=1" in repr_str
        assert f"type={sample_conversation_event_data['event_type']}" in repr_str
        assert f"user={sample_conversation_event_data['user_id']}" in repr_str
    
    async def test_conversation_event_database_persistence(self, db_session, sample_conversation_event_data):
        """Test saving and retrieving ConversationEvent from database."""
        # Create and save event
        event = ConversationEvent(**sample_conversation_event_data)
        db_session.add(event)
        await db_session.commit()
        await db_session.refresh(event)
        
        # Verify event was saved
        assert event.id is not None
        assert event.event_id is not None
        assert event.created_at is not None
        
        # Retrieve event
        from sqlalchemy import select
        result = await db_session.execute(
            select(ConversationEvent).where(ConversationEvent.id == event.id)
        )
        retrieved_event = result.scalar_one()
        
        assert retrieved_event.conversation_id == sample_conversation_event_data["conversation_id"]
        assert retrieved_event.user_id == sample_conversation_event_data["user_id"]
        assert retrieved_event.message_text == sample_conversation_event_data["message_text"]
    
    async def test_conversation_event_query_by_user(self, db_session, create_conversation_event):
        """Test querying ConversationEvents by user ID."""
        user_id = "test_user_query"
        
        # Create multiple events for the user
        event1 = await create_conversation_event(
            db_session,
            user_id=user_id,
            message_text="First message",
            event_type=ConversationEventType.USER_MESSAGE
        )
        event2 = await create_conversation_event(
            db_session,
            user_id=user_id,
            message_text="Second message",
            event_type=ConversationEventType.AGENT_RESPONSE
        )
        
        # Create event for different user
        await create_conversation_event(
            db_session,
            user_id="other_user",
            message_text="Other message"
        )
        
        # Query events for specific user
        from sqlalchemy import select
        result = await db_session.execute(
            select(ConversationEvent).where(ConversationEvent.user_id == user_id)
        )
        user_events = result.scalars().all()
        
        assert len(user_events) == 2
        assert all(event.user_id == user_id for event in user_events)
    
    async def test_conversation_event_query_by_conversation(self, db_session, create_conversation_event):
        """Test querying ConversationEvents by conversation ID."""
        conversation_id = str(uuid.uuid4())
        
        # Create multiple events in the conversation
        event1 = await create_conversation_event(
            db_session,
            conversation_id=conversation_id,
            message_text="First message"
        )
        event2 = await create_conversation_event(
            db_session,
            conversation_id=conversation_id,
            message_text="Second message"
        )
        
        # Create event in different conversation
        await create_conversation_event(
            db_session,
            conversation_id=str(uuid.uuid4()),
            message_text="Other conversation"
        )
        
        # Query events for specific conversation
        from sqlalchemy import select
        result = await db_session.execute(
            select(ConversationEvent).where(
                ConversationEvent.conversation_id == conversation_id
            ).order_by(ConversationEvent.created_at)
        )
        conversation_events = result.scalars().all()
        
        assert len(conversation_events) == 2
        assert all(event.conversation_id == conversation_id for event in conversation_events)


@pytest.mark.unit
class TestEnums:
    """Test enum definitions."""
    
    def test_conversation_event_type_enum(self):
        """Test ConversationEventType enum values."""
        assert ConversationEventType.USER_MESSAGE == "user_message"
        assert ConversationEventType.AGENT_RESPONSE == "agent_response"
        assert ConversationEventType.SYSTEM_EVENT == "system_event"
        assert ConversationEventType.ERROR == "error"
    
    def test_agent_type_enum(self):
        """Test AgentType enum values."""
        assert AgentType.FINOPS == "finops"
        assert AgentType.GITHUB == "github"
        assert AgentType.DOCUMENT == "document"
        assert AgentType.RESEARCH == "research"