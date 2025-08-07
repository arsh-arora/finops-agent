from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class User(BaseModel):
    """User model for authentication and authorization."""
    user_id: str = Field(..., description="Unique user identifier")
    username: Optional[str] = Field(None, description="Username")
    email: Optional[str] = Field(None, description="User email address")
    full_name: Optional[str] = Field(None, description="Full name of the user")
    is_active: bool = Field(True, description="Whether the user is active")
    created_at: Optional[datetime] = Field(None, description="User creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class Token(BaseModel):
    """JWT token model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class TokenData(BaseModel):
    """Token payload data."""
    user_id: Optional[str] = None
    username: Optional[str] = None
    exp: Optional[int] = None