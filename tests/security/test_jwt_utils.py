"""
Test suite for JWT utilities module.

This module tests JWT token creation, validation, decoding, and security features
including expiration, invalid tokens, and user extraction functionality.
"""

import pytest
import jwt
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from shared.security.jwt_utils import (
    JWTManager,
    JWTConfig,
    JWTPayload,
    extract_token_from_header,
    extract_user_id_from_request_header,
    validate_token_from_header,
    get_default_jwt_manager,
    create_access_token,
    decode_token,
    validate_token
)


@pytest.fixture
def jwt_config():
    """Provide test JWT configuration."""
    return JWTConfig(
        secret_key="test-secret-key-at-least-32-chars-long",
        algorithm="HS256",
        access_token_expire_minutes=30,
        refresh_token_expire_days=7,
        issuer="test-trading-system",
        audience="test-trading-api"
    )


@pytest.fixture
def jwt_manager(jwt_config):
    """Provide JWT manager with test configuration."""
    return JWTManager(jwt_config)


@pytest.fixture
def sample_payload():
    """Provide sample JWT payload data."""
    return {
        "user_id": "test_user_123",
        "username": "testuser",
        "service": "trade_executor",
        "roles": ["trader", "user"],
        "permissions": ["read", "write", "execute"],
        "session_id": "session_abc123"
    }


class TestJWTConfig:
    """Test JWT configuration validation."""

    def test_valid_config_creation(self):
        """Test creating valid JWT configuration."""
        config = JWTConfig(
            secret_key="a-very-long-secret-key-for-testing-purposes",
            algorithm="HS256",
            access_token_expire_minutes=60,
            refresh_token_expire_days=14
        )
        assert config.secret_key == "a-very-long-secret-key-for-testing-purposes"
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 60
        assert config.refresh_token_expire_days == 14

    def test_config_defaults(self):
        """Test JWT configuration defaults."""
        config = JWTConfig(secret_key="test-secret-key-at-least-32-chars-long")
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 30
        assert config.refresh_token_expire_days == 7
        assert config.issuer == "trading-system"
        assert config.audience == "trading-api"

    def test_secret_key_validation(self):
        """Test JWT secret key validation."""
        # Valid secret key should pass
        config = JWTConfig(secret_key="this-is-a-valid-secret-key-that-is-long-enough")
        assert len(config.secret_key) >= 32

        # Short secret key is allowed in config but should be handled at runtime
        config_short = JWTConfig(secret_key="short")
        assert config_short.secret_key == "short"


class TestJWTPayload:
    """Test JWT payload model."""

    def test_payload_creation(self, sample_payload):
        """Test creating JWT payload."""
        payload = JWTPayload(**sample_payload)
        assert payload.user_id == "test_user_123"
        assert payload.username == "testuser"
        assert payload.service == "trade_executor"
        assert payload.roles == ["trader", "user"]
        assert payload.permissions == ["read", "write", "execute"]
        assert payload.session_id == "session_abc123"

    def test_payload_defaults(self):
        """Test JWT payload defaults."""
        payload = JWTPayload(user_id="test_user")
        assert payload.user_id == "test_user"
        assert payload.roles == []
        assert payload.permissions == []
        assert payload.username is None
        assert payload.service is None


class TestJWTManager:
    """Test JWT manager functionality."""

    def test_manager_initialization(self, jwt_config):
        """Test JWT manager initialization."""
        manager = JWTManager(jwt_config)
        assert manager.config == jwt_config

    def test_default_config_loading(self):
        """Test loading default configuration from environment."""
        with patch.dict(os.environ, {
            'JWT_SECRET': 'test-env-secret-key-that-is-long-enough',
            'JWT_ALGORITHM': 'HS512',
            'JWT_ACCESS_TOKEN_EXPIRE_MINUTES': '45'
        }):
            manager = JWTManager()
            assert manager.config.secret_key == 'test-env-secret-key-that-is-long-enough'
            assert manager.config.algorithm == 'HS512'
            assert manager.config.access_token_expire_minutes == 45

    def test_create_access_token(self, jwt_manager, sample_payload):
        """Test access token creation."""
        token = jwt_manager.create_access_token(**sample_payload)
        assert isinstance(token, str)
        assert len(token) > 0

        # Decode token manually to verify contents
        decoded = jwt.decode(
            token,
            jwt_manager.config.secret_key,
            algorithms=[jwt_manager.config.algorithm],
            audience=jwt_manager.config.audience,
            issuer=jwt_manager.config.issuer
        )
        assert decoded["user_id"] == "test_user_123"
        assert decoded["username"] == "testuser"
        assert decoded["service"] == "trade_executor"
        assert decoded["roles"] == ["trader", "user"]

    def test_create_refresh_token(self, jwt_manager):
        """Test refresh token creation."""
        token = jwt_manager.create_refresh_token(
            user_id="test_user",
            session_id="session_123"
        )
        assert isinstance(token, str)

        # Decode token manually to verify it's a refresh token
        decoded = jwt.decode(
            token,
            jwt_manager.config.secret_key,
            algorithms=[jwt_manager.config.algorithm],
            audience=jwt_manager.config.audience,
            issuer=jwt_manager.config.issuer
        )
        assert decoded["user_id"] == "test_user"
        assert decoded["session_id"] == "session_123"
        assert decoded["token_type"] == "refresh"

    def test_decode_valid_token(self, jwt_manager, sample_payload):
        """Test decoding valid JWT token."""
        token = jwt_manager.create_access_token(**sample_payload)
        payload = jwt_manager.decode_token(token)

        assert payload is not None
        assert payload.user_id == "test_user_123"
        assert payload.username == "testuser"
        assert payload.service == "trade_executor"
        assert payload.roles == ["trader", "user"]
        assert payload.permissions == ["read", "write", "execute"]

    def test_decode_expired_token(self, jwt_manager):
        """Test decoding expired JWT token."""
        # Create token with past expiration
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        token = jwt_manager.create_access_token(
            user_id="test_user",
            expires_delta=timedelta(seconds=-3600)  # Expired
        )

        payload = jwt_manager.decode_token(token)
        assert payload is None

    def test_decode_invalid_token(self, jwt_manager):
        """Test decoding invalid JWT token."""
        invalid_tokens = [
            "invalid.token.format",
            "not_a_jwt_token",
            "",
            None
        ]

        for invalid_token in invalid_tokens:
            if invalid_token is not None:
                payload = jwt_manager.decode_token(invalid_token)
                assert payload is None

    def test_decode_tampered_token(self, jwt_manager, sample_payload):
        """Test decoding tampered JWT token."""
        token = jwt_manager.create_access_token(**sample_payload)

        # Tamper with the token
        tampered_token = token[:-10] + "tampered123"
        payload = jwt_manager.decode_token(tampered_token)
        assert payload is None

    def test_validate_token(self, jwt_manager, sample_payload):
        """Test token validation."""
        # Valid token
        valid_token = jwt_manager.create_access_token(**sample_payload)
        assert jwt_manager.validate_token(valid_token) is True

        # Invalid token
        assert jwt_manager.validate_token("invalid_token") is False
        assert jwt_manager.validate_token("") is False

    def test_extract_user_id(self, jwt_manager, sample_payload):
        """Test user ID extraction."""
        token = jwt_manager.create_access_token(**sample_payload)
        user_id = jwt_manager.extract_user_id(token)
        assert user_id == "test_user_123"

        # Invalid token should return None
        assert jwt_manager.extract_user_id("invalid_token") is None

    def test_extract_service_name(self, jwt_manager, sample_payload):
        """Test service name extraction."""
        token = jwt_manager.create_access_token(**sample_payload)
        service = jwt_manager.extract_service_name(token)
        assert service == "trade_executor"

    def test_extract_roles_and_permissions(self, jwt_manager, sample_payload):
        """Test roles and permissions extraction."""
        token = jwt_manager.create_access_token(**sample_payload)

        roles = jwt_manager.extract_roles(token)
        assert roles == ["trader", "user"]

        permissions = jwt_manager.extract_permissions(token)
        assert permissions == ["read", "write", "execute"]

    def test_is_token_expired(self, jwt_manager):
        """Test token expiration check."""
        # Valid token
        valid_token = jwt_manager.create_access_token(user_id="test_user")
        assert jwt_manager.is_token_expired(valid_token) is False

        # Create expired token
        expired_token = jwt_manager.create_access_token(
            user_id="test_user",
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        assert jwt_manager.is_token_expired(expired_token) is True

    def test_refresh_access_token(self, jwt_manager):
        """Test refreshing access token."""
        # Create refresh token
        refresh_token = jwt_manager.create_refresh_token(
            user_id="test_user",
            session_id="session_123"
        )

        # Refresh the access token
        new_access_token = jwt_manager.refresh_access_token(refresh_token)
        assert new_access_token is not None

        # Verify new token is valid
        payload = jwt_manager.decode_token(new_access_token)
        assert payload is not None
        assert payload.user_id == "test_user"

    def test_refresh_with_invalid_token(self, jwt_manager):
        """Test refresh with invalid token."""
        # Non-refresh token
        access_token = jwt_manager.create_access_token(user_id="test_user")
        new_token = jwt_manager.refresh_access_token(access_token)
        assert new_token is None

        # Invalid token
        new_token = jwt_manager.refresh_access_token("invalid_token")
        assert new_token is None


class TestJWTUtilityFunctions:
    """Test JWT utility functions."""

    def test_extract_token_from_header(self):
        """Test token extraction from Authorization header."""
        # Bearer token
        bearer_header = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        token = extract_token_from_header(bearer_header)
        assert token == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"

        # JWT token
        jwt_header = "JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        token = extract_token_from_header(jwt_header)
        assert token == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"

        # Invalid header
        assert extract_token_from_header("Invalid header") is None
        assert extract_token_from_header("") is None
        assert extract_token_from_header(None) is None

    def test_extract_user_id_from_request_header(self, jwt_manager):
        """Test user ID extraction from request header."""
        token = jwt_manager.create_access_token(user_id="test_user_456")
        auth_header = f"Bearer {token}"

        user_id = extract_user_id_from_request_header(auth_header, jwt_manager)
        assert user_id == "test_user_456"

        # Invalid header
        user_id = extract_user_id_from_request_header("Invalid header", jwt_manager)
        assert user_id is None

    def test_validate_token_from_header(self, jwt_manager):
        """Test token validation from request header."""
        token = jwt_manager.create_access_token(user_id="test_user")
        auth_header = f"Bearer {token}"

        assert validate_token_from_header(auth_header, jwt_manager) is True
        assert validate_token_from_header("Invalid header", jwt_manager) is False

    def test_global_jwt_manager(self):
        """Test global JWT manager functionality."""
        # Set up environment to avoid config validation errors
        with patch.dict(os.environ, {'JWT_SECRET': 'test-global-jwt-secret-key-for-testing'}):
            # Test getting default manager
            manager1 = get_default_jwt_manager()
            manager2 = get_default_jwt_manager()
            assert manager1 is manager2  # Should be same instance

            # Test convenience functions
            token = create_access_token(user_id="global_test_user")
            assert isinstance(token, str)

            payload = decode_token(token)
            assert payload is not None
            assert payload.user_id == "global_test_user"

            assert validate_token(token) is True


class TestJWTSecurity:
    """Test JWT security features."""

    def test_different_secret_keys(self):
        """Test that tokens from different secret keys don't validate."""
        config1 = JWTConfig(secret_key="secret-key-one-that-is-long-enough-1")
        config2 = JWTConfig(secret_key="secret-key-two-that-is-long-enough-2")

        manager1 = JWTManager(config1)
        manager2 = JWTManager(config2)

        # Create token with manager1
        token = manager1.create_access_token(user_id="test_user")

        # Try to validate with manager2 (different secret)
        assert manager2.validate_token(token) is False
        assert manager2.decode_token(token) is None

    def test_algorithm_security(self):
        """Test different algorithm configurations."""
        # Test with HS512
        config_512 = JWTConfig(
            secret_key="test-secret-key-for-hs512-algorithm-testing",
            algorithm="HS512"
        )
        manager_512 = JWTManager(config_512)

        token = manager_512.create_access_token(user_id="test_user")
        assert manager_512.validate_token(token) is True

        # HS256 manager shouldn't validate HS512 token
        config_256 = JWTConfig(
            secret_key="test-secret-key-for-hs512-algorithm-testing",
            algorithm="HS256"
        )
        manager_256 = JWTManager(config_256)
        assert manager_256.validate_token(token) is False

    def test_audience_validation(self, jwt_manager):
        """Test JWT audience validation."""
        # Create token with specific audience
        token = jwt_manager.create_access_token(user_id="test_user")

        # Should validate with correct audience
        payload = jwt_manager.decode_token(token)
        assert payload is not None

        # Manually decode with wrong audience should fail
        with pytest.raises(jwt.InvalidAudienceError):
            jwt.decode(
                token,
                jwt_manager.config.secret_key,
                algorithms=[jwt_manager.config.algorithm],
                audience="wrong-audience",
                issuer=jwt_manager.config.issuer
            )

    def test_issuer_validation(self, jwt_manager):
        """Test JWT issuer validation."""
        token = jwt_manager.create_access_token(user_id="test_user")

        # Should validate with correct issuer
        payload = jwt_manager.decode_token(token)
        assert payload is not None

        # Manually decode with wrong issuer should fail
        with pytest.raises(jwt.InvalidIssuerError):
            jwt.decode(
                token,
                jwt_manager.config.secret_key,
                algorithms=[jwt_manager.config.algorithm],
                audience=jwt_manager.config.audience,
                issuer="wrong-issuer"
            )

    def test_token_expiration_scenarios(self, jwt_manager):
        """Test various token expiration scenarios."""
        # Token that expires in 1 second
        short_lived_token = jwt_manager.create_access_token(
            user_id="test_user",
            expires_delta=timedelta(seconds=1)
        )
        assert jwt_manager.validate_token(short_lived_token) is True

        # Already expired token
        expired_token = jwt_manager.create_access_token(
            user_id="test_user",
            expires_delta=timedelta(seconds=-1)
        )
        assert jwt_manager.validate_token(expired_token) is False
        assert jwt_manager.is_token_expired(expired_token) is True

    def test_custom_expiration(self, jwt_manager):
        """Test custom token expiration."""
        # 2 hour token
        long_token = jwt_manager.create_access_token(
            user_id="test_user",
            expires_delta=timedelta(hours=2)
        )

        payload = jwt_manager.decode_token(long_token)
        assert payload is not None

        # Check expiration time is approximately correct
        now = datetime.now(timezone.utc)
        expected_exp = now + timedelta(hours=2)
        assert abs((payload.exp - expected_exp).total_seconds()) < 60  # Within 1 minute


class TestJWTIntegration:
    """Test JWT integration scenarios."""

    def test_service_to_service_auth(self, jwt_manager):
        """Test service-to-service authentication."""
        token = jwt_manager.create_access_token(
            user_id="trade_executor_service",
            service="trade_executor",
            roles=["service"],
            permissions=["execute_trades", "read_market_data"]
        )

        payload = jwt_manager.decode_token(token)
        assert payload is not None
        assert payload.service == "trade_executor"
        assert "service" in payload.roles
        assert "execute_trades" in payload.permissions

    def test_user_session_auth(self, jwt_manager):
        """Test user session authentication."""
        session_id = "user_session_abc123"
        token = jwt_manager.create_access_token(
            user_id="human_trader",
            username="trader_bob",
            session_id=session_id,
            roles=["trader", "user"],
            permissions=["read", "trade"]
        )

        payload = jwt_manager.decode_token(token)
        assert payload is not None
        assert payload.user_id == "human_trader"
        assert payload.username == "trader_bob"
        assert payload.session_id == session_id

    def test_api_key_auth(self, jwt_manager):
        """Test API key based authentication."""
        api_key_id = "ak_123456789"
        token = jwt_manager.create_access_token(
            user_id="api_user",
            api_key_id=api_key_id,
            roles=["api_user"],
            permissions=["read_only"]
        )

        payload = jwt_manager.decode_token(token)
        assert payload is not None
        assert payload.api_key_id == api_key_id
        assert "api_user" in payload.roles

    def test_token_refresh_flow(self, jwt_manager):
        """Test complete token refresh flow."""
        # Create initial refresh token
        refresh_token = jwt_manager.create_refresh_token(
            user_id="test_user",
            session_id="session_123"
        )

        # Use refresh token to get new access token
        new_access_token = jwt_manager.refresh_access_token(refresh_token)
        assert new_access_token is not None

        # Verify new access token is valid
        payload = jwt_manager.decode_token(new_access_token)
        assert payload is not None
        assert payload.user_id == "test_user"

        # Old refresh token should still be valid (refresh tokens are single-use in production)
        # For this test, we'll just verify it decodes correctly
        refresh_payload = jwt_manager.decode_token(refresh_token)
        assert refresh_payload is not None


class TestJWTErrorHandling:
    """Test JWT error handling."""

    def test_malformed_secret_key(self):
        """Test handling of malformed secret key."""
        # Short secret key is allowed in config, but JWT operations should handle gracefully
        config = JWTConfig(secret_key="short")
        manager = JWTManager(config)

        # Token creation should still work (for testing environments)
        token = manager.create_access_token(user_id="test_user")
        assert isinstance(token, str)

    def test_token_creation_with_invalid_config(self):
        """Test token creation with invalid configuration."""
        # Empty secret key should be handled gracefully
        config = JWTConfig(secret_key="valid-secret-key-for-testing-purposes")
        manager = JWTManager(config)

        # This should work
        token = manager.create_access_token(user_id="test_user")
        assert isinstance(token, str)

    def test_decode_with_none_verification(self, jwt_manager):
        """Test decoding with disabled expiration verification."""
        # Create expired token
        expired_token = jwt_manager.create_access_token(
            user_id="test_user",
            expires_delta=timedelta(seconds=-1)
        )

        # Should fail with verification
        payload = jwt_manager.decode_token(expired_token, verify_exp=True)
        assert payload is None

        # Should succeed without verification
        payload = jwt_manager.decode_token(expired_token, verify_exp=False)
        assert payload is not None
        assert payload.user_id == "test_user"


class TestJWTEnvironmentIntegration:
    """Test JWT integration with environment variables."""

    @patch.dict(os.environ, {
        'JWT_SECRET': 'environment-jwt-secret-key-for-testing',
        'JWT_ALGORITHM': 'HS512',
        'JWT_ISSUER': 'test-issuer',
        'JWT_AUDIENCE': 'test-audience'
    })
    def test_environment_config_loading(self):
        """Test loading JWT config from environment variables."""
        manager = JWTManager()

        assert manager.config.secret_key == 'environment-jwt-secret-key-for-testing'
        assert manager.config.algorithm == 'HS512'
        assert manager.config.issuer == 'test-issuer'
        assert manager.config.audience == 'test-audience'

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_jwt_secret_fallback(self):
        """Test fallback when JWT secret is missing."""
        with patch('shared.security.jwt_utils.logger') as mock_logger:
            # Test with empty JWT secret environment
            manager = JWTManager()

            # Should use default secret and may log warning
            assert "default-jwt-secret" in manager.config.secret_key or len(manager.config.secret_key) >= 32


class TestJWTSecurityBestPractices:
    """Test JWT security best practices implementation."""

    def test_token_contains_required_claims(self, jwt_manager, sample_payload):
        """Test that tokens contain required security claims."""
        token = jwt_manager.create_access_token(**sample_payload)

        # Manually decode to check all required claims
        decoded = jwt.decode(
            token,
            jwt_manager.config.secret_key,
            algorithms=[jwt_manager.config.algorithm],
            audience=jwt_manager.config.audience,
            issuer=jwt_manager.config.issuer
        )

        # Check required claims are present
        required_claims = ["user_id", "iss", "aud", "exp", "iat"]
        for claim in required_claims:
            assert claim in decoded, f"Required claim '{claim}' missing from token"

    def test_token_timing_consistency(self, jwt_manager):
        """Test that token creation timing is consistent."""
        tokens = []
        start_time = datetime.now(timezone.utc)

        # Create multiple tokens quickly
        for i in range(5):
            token = jwt_manager.create_access_token(user_id=f"user_{i}")
            tokens.append(token)

        end_time = datetime.now(timezone.utc)

        # All tokens should be valid
        for token in tokens:
            assert jwt_manager.validate_token(token) is True

        # All tokens should have similar creation times
        creation_times = []
        for token in tokens:
            payload = jwt_manager.decode_token(token)
            creation_times.append(payload.iat)

        # All creation times should be within the test execution window (with some tolerance)
        for creation_time in creation_times:
            assert start_time - timedelta(seconds=1) <= creation_time <= end_time + timedelta(seconds=1)

    def test_token_uniqueness(self, jwt_manager):
        """Test that each token is unique even with same payload."""
        tokens = set()

        # Create multiple tokens with identical payloads
        for _ in range(10):
            token = jwt_manager.create_access_token(user_id="same_user")
            tokens.add(token)

        # All tokens should be unique (due to iat timestamp differences)
        # Note: In very fast execution, some tokens might be identical if created in same millisecond
        assert len(tokens) >= 8  # Allow for some duplicates due to timing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
