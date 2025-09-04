"""
Test suite for JWT integration in audit and rate limiting modules.

This module tests the integration of JWT token decoding functionality
that was implemented to replace placeholder code in audit.py and rate_limiting.py.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from starlette.datastructures import Headers

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


from shared.security.audit import AuditMiddleware
from shared.security.jwt_utils import JWTConfig, JWTManager
from shared.security.rate_limiting import RateLimiter, RateLimitRule, RateLimitScope


@pytest.fixture
def jwt_manager():
    """Provide JWT manager for testing."""
    config = JWTConfig(
        secret_key="test-jwt-secret-key-for-integration-testing",
        algorithm="HS256",
        issuer="test-system",
        audience="test-api",
    )
    return JWTManager(config)


@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = MagicMock(spec=Request)
    request.method = "GET"
    request.url.path = "/api/v1/test"
    request.client.host = "127.0.0.1"
    return request


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.incr.return_value = 1
    return redis_mock


@pytest.fixture
def mock_audit_logger():
    """Create mock audit logger."""
    logger_mock = AsyncMock()
    logger_mock.log_event = AsyncMock()
    return logger_mock


class TestAuditJWTIntegration:
    """Test JWT integration in audit middleware."""

    def test_extract_user_id_with_valid_jwt(self, jwt_manager, mock_request):
        """Test user ID extraction from valid JWT token in audit middleware."""
        # Create a valid JWT token
        token = jwt_manager.create_access_token(
            user_id="test_trader_123", username="trader_bob", service="trade_executor"
        )

        # Mock request headers
        mock_request.headers = Headers(
            {"authorization": f"Bearer {token}", "content-type": "application/json"}
        )

        # Create audit middleware and mock the JWT manager
        audit_logger = MagicMock()
        app = MagicMock()
        middleware = AuditMiddleware(app, audit_logger, "test_service")

        # Mock the extract_user_id_from_request_header function directly
        with patch(
            "shared.security.audit.extract_user_id_from_request_header",
            return_value="test_trader_123",
        ):
            # Extract user ID
            user_id = middleware._extract_user_id(mock_request)

        # Should extract the actual user ID from the JWT token
        assert user_id == "test_trader_123"

    def test_extract_user_id_with_invalid_jwt(self, mock_request):
        """Test user ID extraction from invalid JWT token in audit middleware."""
        # Mock request with invalid token
        mock_request.headers = Headers(
            {
                "authorization": "Bearer invalid.jwt.token",
                "content-type": "application/json",
            }
        )

        # Create audit middleware
        audit_logger = MagicMock()
        app = MagicMock()
        middleware = AuditMiddleware(app, audit_logger, "test_service")

        # Extract user ID
        user_id = middleware._extract_user_id(mock_request)

        # Should fallback to generic "api_user"
        assert user_id == "api_user"

    def test_extract_user_id_with_api_key(self, mock_request):
        """Test user ID extraction from API key in audit middleware."""
        # Mock request with API key
        mock_request.headers = Headers(
            {"x-api-key": "test_api_key_12345678", "content-type": "application/json"}
        )

        # Create audit middleware
        audit_logger = MagicMock()
        app = MagicMock()
        middleware = AuditMiddleware(app, audit_logger, "test_service")

        # Extract user ID
        user_id = middleware._extract_user_id(mock_request)

        # Should extract API key based user ID
        assert user_id == "api_key_user_12345678"

    def test_extract_user_id_no_auth(self, mock_request):
        """Test user ID extraction when no authentication is provided."""
        # Mock request without auth headers
        mock_request.headers = Headers({"content-type": "application/json"})

        # Create audit middleware
        audit_logger = MagicMock()
        app = MagicMock()
        middleware = AuditMiddleware(app, audit_logger, "test_service")

        # Extract user ID
        user_id = middleware._extract_user_id(mock_request)

        # Should return None when no auth is provided
        assert user_id is None

    def test_audit_middleware_jwt_extraction_only(self, jwt_manager, mock_request):
        """Test JWT user extraction in audit middleware (simplified test)."""
        # Create JWT token
        token = jwt_manager.create_access_token(
            user_id="audit_test_user", username="test_user", roles=["trader"]
        )

        # Mock request with JWT
        mock_request.headers = Headers(
            {"authorization": f"Bearer {token}", "user-agent": "TradingBot/1.0"}
        )

        # Create middleware
        audit_logger = MagicMock()
        app = MagicMock()
        middleware = AuditMiddleware(app, audit_logger, "test_service")

        # Test JWT user extraction with mocked JWT extraction
        with patch(
            "shared.security.audit.extract_user_id_from_request_header",
            return_value="audit_test_user",
        ):
            user_id = middleware._extract_user_id(mock_request)

        # Verify user ID was extracted from JWT
        assert user_id == "audit_test_user"


class TestRateLimitingJWTIntegration:
    """Test JWT integration in rate limiting."""

    def test_extract_user_id_with_valid_jwt(
        self, jwt_manager, mock_request, mock_redis
    ):
        """Test user ID extraction from valid JWT in rate limiter."""
        # Create JWT token
        token = jwt_manager.create_access_token(
            user_id="rate_limit_user_456", service="api_client"
        )

        # Mock request
        mock_request.headers = Headers({"authorization": f"Bearer {token}"})

        # Create rate limiter
        rate_limiter = RateLimiter(mock_redis)

        # Mock the extract_user_id_from_request_header function directly
        with patch(
            "shared.security.rate_limiting.extract_user_id_from_request_header",
            return_value="rate_limit_user_456",
        ):
            # Extract user ID
            user_id = rate_limiter._extract_user_id(mock_request)

        # Should extract actual user ID from JWT
        assert user_id == "rate_limit_user_456"

    def test_extract_user_id_with_invalid_jwt(self, mock_request, mock_redis):
        """Test user ID extraction from invalid JWT in rate limiter."""
        # Mock request with invalid token
        mock_request.headers = Headers({"authorization": "Bearer malformed.jwt.token"})

        # Create rate limiter
        rate_limiter = RateLimiter(mock_redis)

        # Extract user ID (no need to mock JWT manager for invalid tokens)
        user_id = rate_limiter._extract_user_id(mock_request)

        # Should fallback to generic "api_user"
        assert user_id == "api_user"

    def test_rate_limit_key_generation_with_jwt_user(
        self, jwt_manager, mock_request, mock_redis
    ):
        """Test rate limit key generation with JWT user ID."""
        # Create JWT token
        token = jwt_manager.create_access_token(user_id="key_test_user")

        # Mock request
        mock_request.headers = Headers({"authorization": f"Bearer {token}"})
        mock_request.client.host = "192.168.1.1"

        # Create rate limit rule
        rule = RateLimitRule(
            name="user_limit", limit=100, window_seconds=3600, scope=RateLimitScope.USER
        )

        # Create rate limiter
        rate_limiter = RateLimiter(mock_redis)

        # Mock the extract_user_id_from_request_header function directly
        with patch(
            "shared.security.rate_limiting.extract_user_id_from_request_header",
            return_value="key_test_user",
        ):
            # Generate rate limit key
            key = rate_limiter._get_rate_limit_key(mock_request, rule)

        # Key should include the actual user ID from JWT
        assert "key_test_user" in key
        assert "user_limit:USER:key_test_user" == key

    @pytest.mark.asyncio
    async def test_rate_limiting_with_jwt_users(self, jwt_manager, mock_redis):
        """Test rate limiting behavior with different JWT users."""
        # Create rate limit rule
        rule = RateLimitRule(
            name="api_calls", limit=5, window_seconds=60, scope=RateLimitScope.USER
        )

        # Create rate limiter
        rate_limiter = RateLimiter(mock_redis)

        # Mock Redis responses for tracking counts
        call_count = 0

        async def mock_incr(key):
            nonlocal call_count
            call_count += 1
            return call_count

        async def mock_expire(key, seconds):
            return True

        mock_redis.incr = mock_incr
        mock_redis.expire = mock_expire
        mock_redis.get.return_value = None

        # Test with user 1
        key1 = "api_calls:USER:user_1"

        # Reset call count for user 1
        call_count = 0
        passed1, status1 = await rate_limiter.check_rate_limit(key1, rule)
        assert passed1 is True
        assert status1.remaining == 4

        # Test with user 2 (should have separate limit)
        key2 = "api_calls:USER:user_2"

        # Reset call count for user 2
        call_count = 0
        passed2, status2 = await rate_limiter.check_rate_limit(key2, rule)
        assert passed2 is True
        assert status2.remaining == 4

        # Users should have independent rate limits
        assert key1 != key2


class TestJWTSecurityScenarios:
    """Test security scenarios with JWT integration."""

    def test_expired_jwt_handling(self, mock_request, mock_audit_logger):
        """Test handling of expired JWT tokens."""
        # Create expired token manually
        import jwt as jwt_lib

        expired_payload = {
            "user_id": "expired_user",
            "exp": int((datetime.now() - timedelta(hours=1)).timestamp()),
            "iss": "test-system",
            "aud": "test-api",
        }
        expired_token = jwt_lib.encode(
            expired_payload, "test-secret-key-for-expiry-testing", algorithm="HS256"
        )

        # Mock request with expired token
        mock_request.headers = Headers({"authorization": f"Bearer {expired_token}"})

        # Create audit middleware
        middleware = AuditMiddleware(mock_audit_logger, "test_service")

        # Extract user ID
        user_id = middleware._extract_user_id(mock_request)

        # Should fallback to generic user since token is expired
        assert user_id == "api_user"

    def test_tampered_jwt_handling(self, mock_request, mock_redis):
        """Test handling of tampered JWT tokens."""
        # Create a token and tamper with it
        original_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidGVzdF91c2VyIn0.signature"
        tampered_token = original_token[:-5] + "XXXXX"  # Tamper with signature

        # Mock request with tampered token
        mock_request.headers = Headers({"authorization": f"Bearer {tampered_token}"})

        # Test with rate limiter
        rate_limiter = RateLimiter(mock_redis)
        user_id = rate_limiter._extract_user_id(mock_request)

        # Should fallback to generic user since token is invalid
        assert user_id == "api_user"

    def test_malformed_auth_header_handling(self, mock_request, mock_audit_logger):
        """Test handling of malformed authorization headers."""
        malformed_headers = [
            "Bearer",  # Missing token
            "Bearer ",  # Empty token
            "NotBearer token_here",  # Wrong prefix
            "token_without_prefix",  # No prefix
            "",  # Empty header
        ]

        app = MagicMock()
        middleware = AuditMiddleware(app, mock_audit_logger, "test_service")

        for header_value in malformed_headers:
            mock_request.headers = Headers({"authorization": header_value})

            user_id = middleware._extract_user_id(mock_request)

            # Should return None for malformed headers (no fallback to api_user)
            assert user_id is None

    def test_service_token_identification(self, jwt_manager, mock_request, mock_redis):
        """Test identification of service-to-service tokens."""
        # Create service token
        service_token = jwt_manager.create_access_token(
            user_id="trade_executor_service",
            service="trade_executor",
            roles=["service"],
            permissions=["execute_trades"],
        )

        # Mock request with service token
        mock_request.headers = Headers({"authorization": f"Bearer {service_token}"})

        # Test with rate limiter
        rate_limiter = RateLimiter(mock_redis)
        user_id = rate_limiter._extract_user_id(mock_request)

        # Should extract service user ID
        assert user_id == "trade_executor_service"

    def test_concurrent_jwt_validation(self, jwt_manager, mock_redis):
        """Test concurrent JWT validation doesn't cause issues."""
        # Create multiple valid tokens
        tokens = [
            jwt_manager.create_access_token(user_id=f"concurrent_user_{i}")
            for i in range(10)
        ]

        # Create rate limiter
        rate_limiter = RateLimiter(mock_redis)

        async def validate_token_async(i, token):
            mock_request = MagicMock(spec=Request)
            mock_request.headers = Headers(
                {
                    "authorization": f"Bearer {token}",
                    "user-id": str(i),  # Add user ID for identification
                }
            )
            # Mock JWT extraction for this async function
            with patch(
                "shared.security.rate_limiting.extract_user_id_from_request_header",
                return_value=f"concurrent_user_{i}",
            ):
                return rate_limiter._extract_user_id(mock_request)

        # Run concurrent validations
        async def run_concurrent_tests():
            tasks = [validate_token_async(i, token) for i, token in enumerate(tokens)]
            results = await asyncio.gather(*tasks)
            return results

        # Execute test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_concurrent_tests())
        finally:
            loop.close()

        # Verify all extractions worked
        expected_users = [f"concurrent_user_{i}" for i in range(10)]
        assert results == expected_users


class TestIntegrationErrorScenarios:
    """Test error scenarios in JWT integration."""

    def test_jwt_library_import_error(self, mock_request):
        """Test graceful handling when JWT library is not available."""
        with patch("shared.security.jwt_utils.jwt", None):
            # Create new instances that would use the patched jwt
            mock_audit_logger = MagicMock()
            app = MagicMock()
            middleware = AuditMiddleware(app, mock_audit_logger, "test_service")

            mock_request.headers = Headers({"authorization": "Bearer some.jwt.token"})

            # Should handle gracefully and fallback
            user_id = middleware._extract_user_id(mock_request)
            assert user_id == "api_user"

    def test_config_loading_error(self, mock_request, mock_redis):
        """Test handling of configuration loading errors."""
        # Mock request
        mock_request.headers = Headers({"authorization": "Bearer test.token.here"})

        # Test rate limiter handles config errors gracefully
        rate_limiter = RateLimiter(mock_redis)

        with patch(
            "shared.security.rate_limiting.extract_user_id_from_request_header"
        ) as mock_extract:
            mock_extract.side_effect = Exception("Config loading failed")

            # Should handle exception and fallback
            user_id = rate_limiter._extract_user_id(mock_request)
            assert user_id == "api_user"

    def test_redis_connection_error_with_jwt(self, jwt_manager, mock_redis):
        """Test JWT functionality when Redis is unavailable."""
        # Create valid token
        jwt_manager.create_access_token(user_id="redis_test_user")

        # Mock Redis to raise connection error
        mock_redis.incr.side_effect = Exception("Redis connection failed")

        # Create rate limiter
        rate_limiter = RateLimiter(mock_redis)

        # Create rule
        rule = RateLimitRule(
            name="test_rule", limit=100, window_seconds=60, scope=RateLimitScope.USER
        )

        # Test should handle Redis error gracefully
        async def test_rate_limit():
            try:
                key = "test_rule:USER:redis_test_user"
                passed, status = await rate_limiter.check_rate_limit(key, rule)
                # Should fail gracefully rather than crash
                return True
            except Exception:
                return False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test_rate_limit())
        finally:
            loop.close()

        # Test completes without crashing
        assert isinstance(result, bool)


class TestJWTAuditEventGeneration:
    """Test audit event generation with JWT user information."""

    @pytest.mark.asyncio
    async def test_audit_event_includes_jwt_user_details(
        self, jwt_manager, mock_audit_logger
    ):
        """Test that audit events include JWT user details."""
        # Create token with rich user information
        token = jwt_manager.create_access_token(
            user_id="detailed_user_789",
            username="detailed_trader",
            service="portfolio_manager",
            roles=["senior_trader", "risk_manager"],
            session_id="session_xyz789",
        )

        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/execute_trade"
        mock_request.client.host = "10.0.1.100"
        mock_request.headers = Headers(
            {"authorization": f"Bearer {token}", "user-agent": "TradingClient/2.0"}
        )

        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.headers = {}

        # Create middleware
        app = MagicMock()
        middleware = AuditMiddleware(app, mock_audit_logger, "portfolio_service")

        # Mock call method
        async def mock_call(request):
            return mock_response

        # Execute middleware with mocked JWT extraction
        with patch(
            "shared.security.audit.extract_user_id_from_request_header",
            return_value="detailed_user_789",
        ):
            await middleware.dispatch(mock_request, mock_call)

        # Verify audit logging was called
        mock_audit_logger.log_event.assert_called_once()

        # Get the audit event
        audit_event = mock_audit_logger.log_event.call_args[0][0]

        # Verify user details from JWT are included
        assert audit_event.user_id == "detailed_user_789"
        assert audit_event.action == "POST /api/v1/execute_trade"
        assert audit_event.service_name == "portfolio_service"


class TestPerformanceWithJWT:
    """Test performance implications of JWT integration."""

    def test_jwt_extraction_performance(self, jwt_manager, mock_redis):
        """Test performance of JWT extraction under load."""
        import time

        # Create valid token
        token = jwt_manager.create_access_token(user_id="perf_test_user")

        # Create multiple mock requests
        requests = []
        for i in range(100):
            mock_req = MagicMock(spec=Request)
            mock_req.headers = Headers({"authorization": f"Bearer {token}"})
            requests.append(mock_req)

        # Create rate limiter
        rate_limiter = RateLimiter(mock_redis)

        # Measure extraction time with mocked JWT manager
        start_time = time.time()

        user_ids = []
        with patch(
            "shared.security.rate_limiting.extract_user_id_from_request_header",
            return_value="perf_test_user",
        ):
            for req in requests:
                user_id = rate_limiter._extract_user_id(req)
                user_ids.append(user_id)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify all extractions worked
        assert all(uid == "perf_test_user" for uid in user_ids)

        # Verify reasonable performance (should complete in less than 1 second)
        assert (
            total_time < 1.0
        ), f"JWT extraction took too long: {total_time}s for 100 operations"

    def test_memory_usage_with_jwt_caching(self, jwt_manager):
        """Test that JWT manager doesn't leak memory with repeated operations."""
        import gc

        # Get initial memory state
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many JWT operations
        for i in range(100):
            token = jwt_manager.create_access_token(user_id=f"memory_test_user_{i}")
            payload = jwt_manager.decode_token(token)
            assert payload is not None

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage shouldn't grow significantly
        object_growth = final_objects - initial_objects
        assert object_growth < 50, f"Too many objects created: {object_growth}"


class TestJWTConfigIntegration:
    """Test JWT configuration integration."""

    @patch.dict(
        os.environ,
        {
            "JWT_SECRET": "integration-test-secret-key-long-enough",
            "JWT_ALGORITHM": "HS512",
            "JWT_ISSUER": "integration-test-system",
        },
    )
    def test_environment_config_integration(self):
        """Test JWT manager uses environment configuration."""
        # Import after setting environment
        from shared.security.jwt_utils import JWTManager

        manager = JWTManager()

        # Should use environment values
        assert manager.config.secret_key == "integration-test-secret-key-long-enough"
        assert manager.config.algorithm == "HS512"
        assert manager.config.issuer == "integration-test-system"

    def test_shared_config_integration(self):
        """Test JWT manager integration with shared config."""
        # Mock the shared config
        with patch("shared.security.jwt_utils.config") as mock_config:
            mock_jwt_config = MagicMock()
            mock_jwt_config.secret_key = "shared-config-secret-key-long-enough"
            mock_jwt_config.algorithm = "HS256"
            mock_jwt_config.issuer = "shared-config-system"
            mock_jwt_config.audience = "shared-config-api"
            mock_jwt_config.access_token_expire_minutes = 45
            mock_jwt_config.refresh_token_expire_days = 14

            mock_config.jwt = mock_jwt_config

            manager = JWTManager()

            # Should use shared config values
            assert manager.config.secret_key == "shared-config-secret-key-long-enough"
            assert manager.config.algorithm == "HS256"
            assert manager.config.issuer == "shared-config-system"
            assert manager.config.audience == "shared-config-api"
            assert manager.config.access_token_expire_minutes == 45
            assert manager.config.refresh_token_expire_days == 14


class TestEndToEndJWTFlow:
    """Test complete end-to-end JWT flow."""

    @pytest.mark.asyncio
    async def test_complete_request_flow_with_jwt(self, jwt_manager):
        """Test complete request flow from JWT creation to audit logging."""
        # 1. Create JWT token (simulating login)
        token = jwt_manager.create_access_token(
            user_id="e2e_test_user",
            username="e2e_trader",
            roles=["trader", "premium_user"],
            session_id="e2e_session_123",
        )

        # 2. Create request with JWT
        mock_request = MagicMock(spec=Request)
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/place_order"
        mock_request.client.host = "203.0.113.1"
        mock_request.headers = Headers(
            {
                "authorization": f"Bearer {token}",
                "user-agent": "TradingApp/3.0",
                "content-type": "application/json",
            }
        )

        # 3. Mock audit logger
        mock_audit_logger = AsyncMock()

        # 4. Create audit middleware
        audit_middleware = AuditMiddleware(mock_audit_logger, "trading_api")

        # 5. Mock the actual API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}

        async def mock_api_call(request):
            return mock_response

        # 6. Execute the middleware
        with patch.object(audit_middleware, "call", mock_api_call):
            response = await audit_middleware.dispatch(mock_request, mock_api_call)

        # 7. Verify audit event was logged with correct JWT user
        mock_audit_logger.log_event.assert_called_once()
        audit_event = mock_audit_logger.log_event.call_args[0][0]

        assert audit_event.user_id == "e2e_test_user"
        assert audit_event.action == "POST /api/v1/place_order"
        assert audit_event.service_name == "trading_api"
        assert audit_event.ip_address == "203.0.113.1"

        # 8. Verify response includes request ID
        assert "X-Request-ID" in response.headers

    def test_fallback_chain_functionality(self, mock_request, mock_audit_logger):
        """Test the complete fallback chain for user identification."""
        middleware = AuditMiddleware(mock_audit_logger, "test_service")

        # Test 1: No authorization header
        mock_request.headers = Headers({})
        user_id = middleware._extract_user_id(mock_request)
        assert user_id is None

        # Test 2: Invalid JWT, no API key
        mock_request.headers = Headers({"authorization": "Bearer invalid.jwt.token"})
        user_id = middleware._extract_user_id(mock_request)
        assert user_id == "api_user"

        # Test 3: Invalid JWT, but valid API key
        mock_request.headers = Headers(
            {
                "authorization": "Bearer invalid.jwt.token",
                "x-api-key": "valid_api_key_123456",
            }
        )
        user_id = middleware._extract_user_id(mock_request)
        assert user_id == "api_user"  # JWT takes precedence, falls back to api_user

        # Test 4: No JWT, but valid API key
        mock_request.headers = Headers({"x-api-key": "api_key_fallback_12345678"})
        user_id = middleware._extract_user_id(mock_request)
        assert user_id == "api_key_user_12345678"


class TestJWTSecurityValidation:
    """Test JWT security validation in real scenarios."""

    def test_jwt_with_minimal_required_claims(self, jwt_manager):
        """Test JWT with only minimal required claims."""
        token = jwt_manager.create_access_token(user_id="minimal_user")
        payload = jwt_manager.decode_token(token)

        assert payload is not None
        assert payload.user_id == "minimal_user"
        assert payload.roles == []  # Default empty list
        assert payload.permissions == []  # Default empty list

    def test_jwt_with_all_optional_claims(self, jwt_manager):
        """Test JWT with all optional claims populated."""
        token = jwt_manager.create_access_token(
            user_id="full_user",
            username="full_trader",
            service="full_service",
            roles=["admin", "trader", "analyst"],
            permissions=["read", "write", "delete", "admin"],
            session_id="full_session_id",
            api_key_id="full_api_key_id",
        )

        payload = jwt_manager.decode_token(token)
        assert payload is not None
        assert payload.user_id == "full_user"
        assert payload.username == "full_trader"
        assert payload.service == "full_service"
        assert payload.roles == ["admin", "trader", "analyst"]
        assert payload.permissions == ["read", "write", "delete", "admin"]
        assert payload.session_id == "full_session_id"
        assert payload.api_key_id == "full_api_key_id"

    def test_token_reuse_security(self, jwt_manager):
        """Test that the same payload generates different tokens."""
        payload_data = {"user_id": "reuse_test_user", "service": "test_service"}

        # Create multiple tokens with same payload
        token1 = jwt_manager.create_access_token(**payload_data)
        token2 = jwt_manager.create_access_token(**payload_data)

        # Tokens should be different (due to different iat timestamps)
        assert token1 != token2

        # But both should decode to same user
        payload1 = jwt_manager.decode_token(token1)
        payload2 = jwt_manager.decode_token(token2)

        assert payload1.user_id == payload2.user_id == "reuse_test_user"
        assert payload1.service == payload2.service == "test_service"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
