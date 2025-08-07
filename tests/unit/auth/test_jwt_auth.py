"""
Tests for JWT authentication functionality.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from src.auth.jwt_auth import JWTAuth, create_access_token, verify_token, get_current_user
from src.auth.models import User, TokenData


@pytest.mark.unit
@pytest.mark.auth
class TestJWTAuth:
    """Test JWT authentication functionality."""
    
    def test_jwt_auth_initialization(self, jwt_auth, test_settings):
        """Test JWTAuth initialization."""
        with patch('src.auth.jwt_auth.settings', test_settings):
            auth = JWTAuth()
            
            assert auth.secret_key == test_settings.JWT_SECRET_KEY
            assert auth.algorithm == test_settings.JWT_ALGORITHM
            assert auth.access_token_expire_minutes == test_settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    
    def test_verify_password_success(self, jwt_auth):
        """Test successful password verification."""
        plain_password = "test_password"
        hashed_password = jwt_auth.get_password_hash(plain_password)
        
        result = jwt_auth.verify_password(plain_password, hashed_password)
        
        assert result is True
    
    def test_verify_password_failure(self, jwt_auth):
        """Test failed password verification."""
        plain_password = "test_password"
        wrong_password = "wrong_password"
        hashed_password = jwt_auth.get_password_hash(plain_password)
        
        result = jwt_auth.verify_password(wrong_password, hashed_password)
        
        assert result is False
    
    def test_get_password_hash(self, jwt_auth):
        """Test password hashing."""
        password = "test_password"
        
        hashed = jwt_auth.get_password_hash(password)
        
        assert hashed != password
        assert isinstance(hashed, str)
        assert len(hashed) > 0
    
    def test_create_access_token_default_expiry(self, jwt_auth):
        """Test creating access token with default expiry."""
        data = {"sub": "test_user", "username": "testuser"}
        
        token = jwt_auth.create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token can be decoded
        token_data = jwt_auth.verify_token(token)
        assert token_data.user_id == "test_user"
        assert token_data.username == "testuser"
    
    def test_create_access_token_custom_expiry(self, jwt_auth):
        """Test creating access token with custom expiry."""
        data = {"sub": "test_user", "username": "testuser"}
        expires_delta = timedelta(minutes=60)
        
        token = jwt_auth.create_access_token(data, expires_delta)
        
        assert isinstance(token, str)
        
        # Verify token can be decoded
        token_data = jwt_auth.verify_token(token)
        assert token_data.user_id == "test_user"
    
    def test_verify_token_valid(self, jwt_auth):
        """Test verifying valid token."""
        data = {"sub": "test_user", "username": "testuser"}
        token = jwt_auth.create_access_token(data)
        
        token_data = jwt_auth.verify_token(token)
        
        assert token_data is not None
        assert token_data.user_id == "test_user"
        assert token_data.username == "testuser"
        assert isinstance(token_data.exp, int)
    
    def test_verify_token_invalid(self, jwt_auth):
        """Test verifying invalid token."""
        invalid_token = "invalid.jwt.token"
        
        token_data = jwt_auth.verify_token(invalid_token)
        
        assert token_data is None
    
    def test_verify_token_expired(self, jwt_auth):
        """Test verifying expired token."""
        data = {"sub": "test_user", "username": "testuser"}
        # Create token with negative expiry (already expired)
        expires_delta = timedelta(seconds=-1)
        token = jwt_auth.create_access_token(data, expires_delta)
        
        token_data = jwt_auth.verify_token(token)
        
        assert token_data is None
    
    def test_verify_token_no_subject(self, jwt_auth):
        """Test verifying token without subject."""
        # Create token without 'sub' field
        data = {"username": "testuser"}
        token = jwt_auth.create_access_token(data)
        
        token_data = jwt_auth.verify_token(token)
        
        assert token_data is None
    
    async def test_get_current_user_valid_token(self, jwt_auth):
        """Test getting current user with valid token."""
        data = {"sub": "test_user", "username": "testuser"}
        token = jwt_auth.create_access_token(data)
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=token
        )
        
        user = await jwt_auth.get_current_user(credentials)
        
        assert isinstance(user, User)
        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.is_active is True
    
    async def test_get_current_user_invalid_token(self, jwt_auth):
        """Test getting current user with invalid token."""
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid.token"
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await jwt_auth.get_current_user(credentials)
        
        assert exc_info.value.status_code == 401
        assert "Could not validate credentials" in str(exc_info.value.detail)


@pytest.mark.unit
@pytest.mark.auth
class TestTokenData:
    """Test TokenData model."""
    
    def test_token_data_creation(self):
        """Test TokenData creation."""
        token_data = TokenData(
            user_id="test_user",
            username="testuser",
            exp=1234567890
        )
        
        assert token_data.user_id == "test_user"
        assert token_data.username == "testuser"
        assert token_data.exp == 1234567890
    
    def test_token_data_optional_fields(self):
        """Test TokenData with optional fields."""
        token_data = TokenData()
        
        assert token_data.user_id is None
        assert token_data.username is None
        assert token_data.exp is None


@pytest.mark.unit
@pytest.mark.auth
class TestUser:
    """Test User model."""
    
    def test_user_creation(self, sample_user_data):
        """Test User creation."""
        user = User(**sample_user_data)
        
        assert user.user_id == sample_user_data["user_id"]
        assert user.username == sample_user_data["username"]
        assert user.email == sample_user_data["email"]
        assert user.is_active is True
    
    def test_user_minimal_data(self):
        """Test User with minimal required data."""
        user = User(user_id="test_user")
        
        assert user.user_id == "test_user"
        assert user.username is None
        assert user.email is None
        assert user.full_name is None
        assert user.is_active is True
        assert user.created_at is None
    
    def test_user_json_serialization(self, sample_user_data):
        """Test User JSON serialization."""
        user_data = sample_user_data.copy()
        user_data["created_at"] = datetime.now(timezone.utc)
        
        user = User(**user_data)
        json_data = user.model_dump()
        
        assert json_data["user_id"] == user_data["user_id"]
        assert json_data["username"] == user_data["username"]
        assert json_data["email"] == user_data["email"]
        assert "created_at" in json_data


@pytest.mark.unit
@pytest.mark.auth
class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_access_token_function(self, test_settings):
        """Test create_access_token convenience function."""
        with patch('src.auth.jwt_auth.settings', test_settings):
            data = {"sub": "test_user", "username": "testuser"}
            
            token = create_access_token(data)
            
            assert isinstance(token, str)
            assert len(token) > 0
    
    def test_verify_token_function(self, test_settings):
        """Test verify_token convenience function."""
        with patch('src.auth.jwt_auth.settings', test_settings):
            data = {"sub": "test_user", "username": "testuser"}
            token = create_access_token(data)
            
            token_data = verify_token(token)
            
            assert token_data is not None
            assert token_data.user_id == "test_user"
    
    async def test_get_current_user_function(self, test_settings):
        """Test get_current_user convenience function."""
        with patch('src.auth.jwt_auth.settings', test_settings):
            data = {"sub": "test_user", "username": "testuser"}
            token = create_access_token(data)
            
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=token
            )
            
            user = await get_current_user(credentials)
            
            assert isinstance(user, User)
            assert user.user_id == "test_user"


@pytest.mark.unit
@pytest.mark.auth
class TestSecurityIntegration:
    """Test security integration scenarios."""
    
    def test_token_tampering_detection(self, jwt_auth):
        """Test that tampered tokens are rejected."""
        data = {"sub": "test_user", "username": "testuser"}
        token = jwt_auth.create_access_token(data)
        
        # Tamper with the token by changing multiple characters in the signature
        # JWT tokens have 3 parts separated by dots: header.payload.signature
        parts = token.split('.')
        if len(parts) == 3:
            # Tamper with the signature part
            signature = parts[2]
            tampered_signature = 'X' * len(signature)  # Replace entire signature
            tampered_token = f"{parts[0]}.{parts[1]}.{tampered_signature}"
        else:
            # Fallback: change middle of token
            middle = len(token) // 2
            tampered_token = token[:middle] + "TAMPERED" + token[middle+8:]
        
        token_data = jwt_auth.verify_token(tampered_token)
        
        assert token_data is None
    
    def test_different_secret_key_rejection(self):
        """Test that tokens created with different secret keys are rejected."""
        # Create token with one secret
        auth1 = JWTAuth()
        auth1.secret_key = "secret1"
        
        data = {"sub": "test_user", "username": "testuser"}
        token = auth1.create_access_token(data)
        
        # Try to verify with different secret
        auth2 = JWTAuth()
        auth2.secret_key = "secret2"
        
        token_data = auth2.verify_token(token)
        
        assert token_data is None
    
    async def test_expired_token_rejection(self, jwt_auth):
        """Test that expired tokens are properly rejected."""
        data = {"sub": "test_user", "username": "testuser"}
        
        # Create token that expires immediately
        with patch('src.auth.jwt_auth.datetime') as mock_datetime:
            # Mock current time
            mock_now = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now
            
            expires_delta = timedelta(seconds=1)
            token = jwt_auth.create_access_token(data, expires_delta)
            
            # Move time forward past expiration
            mock_datetime.utcnow.return_value = mock_now + timedelta(seconds=2)
            
            credentials = HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials=token
            )
            
            with pytest.raises(HTTPException) as exc_info:
                await jwt_auth.get_current_user(credentials)
            
            assert exc_info.value.status_code == 401