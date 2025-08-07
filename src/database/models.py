from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum
import uuid

Base = declarative_base()


class ConversationEventType(str, Enum):
    """Types of conversation events."""
    USER_MESSAGE = "user_message"
    AGENT_RESPONSE = "agent_response"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"


class AgentType(str, Enum):
    """Types of agents available in the system."""
    FINOPS = "finops"
    GITHUB = "github"
    DOCUMENT = "document"  
    RESEARCH = "research"


class ConversationEvent(Base):
    """
    Model for storing conversation events and messages.
    
    This table stores all interactions between users and agents,
    including metadata for tracking conversation flow and context.
    """
    __tablename__ = "conversation_events"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Event identification
    event_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), index=True)  # Groups related messages
    
    # User and agent information
    user_id = Column(String(255), index=True, nullable=False)
    agent_type = Column(SQLEnum(AgentType), nullable=True)  # NULL for user messages
    
    # Event details
    event_type = Column(SQLEnum(ConversationEventType), nullable=False)
    message_text = Column(Text)
    
    # Metadata  
    event_metadata = Column(JSON, nullable=False, default=lambda: {})  # Store additional context, tool calls, etc.
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Performance tracking
    processing_time_ms = Column(Integer)  # Time taken to process the message
    
    def __init__(self, **kwargs):
        """Initialize ConversationEvent with proper defaults."""
        if 'event_metadata' not in kwargs or kwargs['event_metadata'] is None:
            kwargs['event_metadata'] = {}
        super().__init__(**kwargs)
    
    def __repr__(self):
        return f"<ConversationEvent(id={self.id}, type={self.event_type}, user={self.user_id})>"
    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "event_id": self.event_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "agent_type": self.agent_type.value if self.agent_type else None,
            "event_type": self.event_type.value,
            "message_text": self.message_text,
            "metadata": self.event_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "processing_time_ms": self.processing_time_ms
        }