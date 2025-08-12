"""
JWT (JSON Web Token) utilities for authentication and authorization.

This module provides JWT token creation, validation, and decoding functionality
for the trading system's security infrastructure.
"""

import jwt
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Union
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class JWTConfig(BaseModel):
    """JWT configuration settings."""

    secret_key: str = Field(..., description="JWT signing secret")
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration in minutes")
    refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration in days")
    issuer: str = Field(default="trading-system", description="JWT issuer")
    audience: str = Field(default="trading-api", description="JWT audience")


class JWTPayload(BaseModel):
    """Standard JWT payload structure."""

    user_id: str = Field(..., description="User identifier")
    username: Optional[str] = Field(None, description="Username")
    service: Optional[str] = Field(None, description="Service name for service-to-service auth")
    roles: list[str] = Field(default_factory=list, description="User roles")
    permissions: list[str] = Field(default_factory=list, description="User permissions")
    session_id: Optional[str] = Field(None, description="Session identifier")
    api_key_id: Optional[str] = Field(None, description="API key identifier if applicable")

    # Standard JWT claims
    iss: Optional[str] = Field(None, description="Issuer")
    aud: Optional[str] = Field(None, description="Audience")
    exp: Optional[datetime] = Field(None, description="Expiration time")
    iat: Optional[datetime] = Field(None, description="Issued at time")
    nbf: Optional[datetime] = Field(None, description="Not before time")
    jti: Optional[str] = Field(None, description="JWT ID")


class JWTManager:
    """JWT token manager for encoding, decoding, and validation."""

    def __init__(self, jwt_config: Optional[JWTConfig] = None):
        """Initialize JWT manager with configuration."""
        self.config = jwt_config or self._load_default_config()

    def _load_default_config(self) -> JWTConfig:
        """Load JWT configuration from environment variables."""
        import os

        # Avoid circular import by importing config locally
        try:
            from shared.config import config
            shared_jwt_config = config.jwt
            return JWTConfig(
                secret_key=shared_jwt_config.secret_key or "default-jwt-secret-change-in-production",
                algorithm=shared_jwt_config.algorithm,
                access_token_expire_minutes=shared_jwt_config.access_token_expire_minutes,
                refresh_token_expire_days=shared_jwt_config.refresh_token_expire_days,
                issuer=shared_jwt_config.issuer,
                audience=shared_jwt_config.audience
            )
        except ImportError:
            # Fallback to environment variables if config not available
            jwt_secret = os.getenv("JWT_SECRET") or os.getenv("JWT_SECRET_KEY")
            if not jwt_secret:
                logger.warning("JWT_SECRET not found in environment, using default (insecure for production)")
                jwt_secret = "default-jwt-secret-change-in-production"

            return JWTConfig(
                secret_key=jwt_secret,
                algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
                access_token_expire_minutes=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")),
                refresh_token_expire_days=int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7")),
                issuer=os.getenv("JWT_ISSUER", "trading-system"),
                audience=os.getenv("JWT_AUDIENCE", "trading-api")
            )

    def create_access_token(
        self,
        user_id: str,
        username: Optional[str] = None,
        service: Optional[str] = None,
        roles: Optional[list[str]] = None,
        permissions: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new JWT access token."""
        now = datetime.now(timezone.utc)
        expire = now + (expires_delta or timedelta(minutes=self.config.access_token_expire_minutes))

        payload = JWTPayload(
            user_id=user_id,
            username=username,
            service=service,
            roles=roles or [],
            permissions=permissions or [],
            session_id=session_id,
            api_key_id=api_key_id,
            iss=self.config.issuer,
            aud=self.config.audience,
            exp=expire,
            iat=now,
            nbf=now
        )

        # Convert to dict for JWT encoding
        payload_dict = payload.model_dump(exclude_none=True)

        # Convert datetime objects to timestamps
        for key, value in payload_dict.items():
            if isinstance(value, datetime):
                payload_dict[key] = int(value.timestamp())

        try:
            token = jwt.encode(
                payload_dict,
                self.config.secret_key,
                algorithm=self.config.algorithm
            )
            logger.debug(f"Created JWT token for user_id: {user_id}")
            return token
        except Exception as e:
            logger.error(f"Failed to create JWT token: {e}")
            raise

    def create_refresh_token(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT refresh token."""
        now = datetime.now(timezone.utc)
        expire = now + (expires_delta or timedelta(days=self.config.refresh_token_expire_days))

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "token_type": "refresh",
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "exp": int(expire.timestamp()),
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp())
        }

        try:
            token = jwt.encode(
                payload,
                self.config.secret_key,
                algorithm=self.config.algorithm
            )
            logger.debug(f"Created refresh token for user_id: {user_id}")
            return token
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}")
            raise

    def decode_token(self, token: str, verify_exp: bool = True) -> Optional[JWTPayload]:
        """Decode and validate JWT token."""
        try:
            # Decode the token
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"verify_exp": verify_exp}
            )

            # Convert timestamp fields back to datetime
            for field in ["exp", "iat", "nbf"]:
                if field in payload and payload[field] is not None:
                    payload[field] = datetime.fromtimestamp(payload[field], tz=timezone.utc)

            # Create JWTPayload object
            jwt_payload = JWTPayload(**payload)
            logger.debug(f"Successfully decoded JWT token for user_id: {jwt_payload.user_id}")
            return jwt_payload

        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error decoding JWT token: {e}")
            return None

    def validate_token(self, token: str) -> bool:
        """Validate JWT token without decoding payload."""
        payload = self.decode_token(token)
        return payload is not None

    def extract_user_id(self, token: str) -> Optional[str]:
        """Extract user ID from JWT token."""
        payload = self.decode_token(token)
        return payload.user_id if payload else None

    def extract_service_name(self, token: str) -> Optional[str]:
        """Extract service name from JWT token."""
        payload = self.decode_token(token)
        return payload.service if payload else None

    def extract_roles(self, token: str) -> list[str]:
        """Extract user roles from JWT token."""
        payload = self.decode_token(token)
        return payload.roles if payload else []

    def extract_permissions(self, token: str) -> list[str]:
        """Extract user permissions from JWT token."""
        payload = self.decode_token(token)
        return payload.permissions if payload else []

    def is_token_expired(self, token: str) -> bool:
        """Check if token is expired without raising exception."""
        try:
            # Try to decode with expiration verification
            jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"verify_exp": True}
            )
            # If decode succeeds, token is not expired
            return False
        except jwt.ExpiredSignatureError:
            # Token is expired
            return True
        except jwt.InvalidTokenError:
            # Token is invalid (treat as expired for safety)
            return True

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from valid refresh token."""
        try:
            payload = jwt.decode(
                refresh_token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer
            )

            # Verify it's a refresh token
            if payload.get("token_type") != "refresh":
                logger.warning("Token is not a refresh token")
                return None

            # Create new access token
            user_id = payload.get("user_id")
            session_id = payload.get("session_id")

            if not user_id:
                logger.warning("No user_id found in refresh token")
                return None

            return self.create_access_token(
                user_id=user_id,
                session_id=session_id
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error refreshing token: {e}")
            return None


def extract_token_from_header(auth_header: str) -> Optional[str]:
    """Extract JWT token from Authorization header."""
    if not auth_header:
        return None

    if auth_header.startswith("Bearer "):
        return auth_header[7:]  # Remove "Bearer " prefix
    elif auth_header.startswith("JWT "):
        return auth_header[4:]  # Remove "JWT " prefix

    return None


def extract_user_id_from_request_header(auth_header: str, jwt_manager: Optional[JWTManager] = None) -> Optional[str]:
    """Extract user ID from request Authorization header."""
    if not auth_header:
        return None

    token = extract_token_from_header(auth_header)
    if not token:
        return None

    manager = jwt_manager or get_default_jwt_manager()
    return manager.extract_user_id(token)


def validate_token_from_header(auth_header: str, jwt_manager: Optional[JWTManager] = None) -> bool:
    """Validate JWT token from Authorization header."""
    if not auth_header:
        return False

    token = extract_token_from_header(auth_header)
    if not token:
        return False

    manager = jwt_manager or get_default_jwt_manager()
    return manager.validate_token(token)


# Global JWT manager instance
_jwt_manager: Optional[JWTManager] = None


def get_default_jwt_manager() -> JWTManager:
    """Get the default JWT manager instance."""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


def set_jwt_manager(jwt_manager: JWTManager) -> None:
    """Set a custom JWT manager instance."""
    global _jwt_manager
    _jwt_manager = jwt_manager


# Convenience functions using default manager
def create_access_token(**kwargs) -> str:
    """Create access token using default JWT manager."""
    return get_default_jwt_manager().create_access_token(**kwargs)


def create_refresh_token(**kwargs) -> str:
    """Create refresh token using default JWT manager."""
    return get_default_jwt_manager().create_refresh_token(**kwargs)


def decode_token(token: str, **kwargs) -> Optional[JWTPayload]:
    """Decode token using default JWT manager."""
    return get_default_jwt_manager().decode_token(token, **kwargs)


def validate_token(token: str) -> bool:
    """Validate token using default JWT manager."""
    return get_default_jwt_manager().validate_token(token)


def extract_user_id(token: str) -> Optional[str]:
    """Extract user ID using default JWT manager."""
    return get_default_jwt_manager().extract_user_id(token)
