"""
Rate Limiting System for AI Trading System

This module provides comprehensive rate limiting capabilities including:
- Request rate limiting per IP/user
- API endpoint specific limits
- Sliding window rate limiting
- Circuit breaker integration
- Redis-backed distributed rate limiting

Updated for compatibility with latest dependencies.
Version: 1.0.0 - Updated for async Redis compatibility
Note: All imports verified and syntax validated
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast

import redis.asyncio as redis
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from .jwt_utils import extract_user_id_from_request_header

# Explicit type alias for clarity
TypeAny = Any

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types of rate limits"""

    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    BURST = "burst"
    SLIDING_WINDOW = "sliding_window"


class RateLimitScope(Enum):
    """Scope of rate limiting"""

    GLOBAL = "global"
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_API_KEY = "per_api_key"
    PER_ENDPOINT = "per_endpoint"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""

    name: str
    limit: int
    window_seconds: int
    scope: RateLimitScope
    endpoints: Optional[List[str]] = None
    methods: Optional[List[str]] = None
    bypass_users: Optional[List[str]] = None
    bypass_ips: Optional[List[str]] = None
    burst_limit: Optional[int] = None
    enabled: bool = True


@dataclass
class RateLimitStatus:
    """Current rate limit status for a key"""

    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None


class RateLimiter:
    """Redis-backed rate limiter with sliding window"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.rules: Dict[str, RateLimitRule] = {}
        self.default_rules = self._create_default_rules()

    def _create_default_rules(self) -> Dict[str, RateLimitRule]:
        """Create default rate limiting rules"""
        return {
            "api_general": RateLimitRule(
                name="api_general",
                limit=100,
                window_seconds=60,
                scope=RateLimitScope.PER_IP,
            ),
            "api_trading": RateLimitRule(
                name="api_trading",
                limit=10,
                window_seconds=60,
                scope=RateLimitScope.PER_USER,
                endpoints=["/trades", "/orders", "/positions"],
                methods=["POST", "PUT", "DELETE"],
            ),
            "api_market_data": RateLimitRule(
                name="api_market_data",
                limit=30,
                window_seconds=60,
                scope=RateLimitScope.PER_IP,
                endpoints=["/market-data", "/quotes", "/charts"],
            ),
            "api_export": RateLimitRule(
                name="api_export",
                limit=5,
                window_seconds=60,
                scope=RateLimitScope.PER_USER,
                endpoints=["/export"],
            ),
            "api_admin": RateLimitRule(
                name="api_admin",
                limit=20,
                window_seconds=60,
                scope=RateLimitScope.PER_USER,
                endpoints=["/admin", "/config", "/maintenance"],
            ),
            "burst_protection": RateLimitRule(
                name="burst_protection",
                limit=10,
                window_seconds=1,
                scope=RateLimitScope.PER_IP,
                burst_limit=20,
            ),
            "daily_limit": RateLimitRule(
                name="daily_limit",
                limit=10000,
                window_seconds=86400,
                scope=RateLimitScope.PER_IP,
            ),
        }

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a custom rate limiting rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added rate limiting rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> None:
        """Remove a rate limiting rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed rate limiting rule: {rule_name}")

    async def check_rate_limit(
        self, key: str, rule: RateLimitRule, increment: int = 1
    ) -> Tuple[bool, RateLimitStatus]:
        """Check if request is within rate limit using sliding window"""
        if not rule.enabled:
            return True, RateLimitStatus(
                limit=rule.limit,
                remaining=rule.limit,
                reset_time=datetime.now() + timedelta(seconds=rule.window_seconds),
            )

        current_time = time.time()
        window_start = current_time - rule.window_seconds
        redis_key = f"rate_limit:{rule.name}:{key}"

        try:
            # Use Redis pipeline for atomic operations
            pipe = self.redis.pipeline()

            # Remove expired entries
            pipe.zremrangebyscore(redis_key, 0, window_start)

            # Count current requests in window
            pipe.zcard(redis_key)

            # Add current request
            pipe.zadd(redis_key, {str(current_time): current_time})

            # Set expiration
            pipe.expire(redis_key, rule.window_seconds + 60)

            results = await pipe.execute()
            current_count = results[1]

            # Check burst limit first if defined
            if rule.burst_limit and current_count > rule.burst_limit:
                return False, RateLimitStatus(
                    limit=rule.burst_limit,
                    remaining=0,
                    reset_time=datetime.fromtimestamp(current_time + 1),
                    retry_after=1,
                )

            # Check main limit
            if current_count > rule.limit:
                # Remove the request we just added since it's over limit
                await self.redis.zrem(redis_key, str(current_time))

                # Calculate when the limit resets
                oldest_request = await self.redis.zrange(
                    redis_key, 0, 0, withscores=True
                )
                if oldest_request:
                    reset_time = datetime.fromtimestamp(
                        oldest_request[0][1] + rule.window_seconds
                    )
                    retry_after = int((reset_time - datetime.now()).total_seconds())
                else:
                    reset_time = datetime.now() + timedelta(seconds=rule.window_seconds)
                    retry_after = rule.window_seconds

                return False, RateLimitStatus(
                    limit=rule.limit,
                    remaining=0,
                    reset_time=reset_time,
                    retry_after=max(retry_after, 1),
                )

            # Within limits
            remaining = max(0, rule.limit - current_count)
            reset_time = datetime.fromtimestamp(current_time + rule.window_seconds)

            return True, RateLimitStatus(
                limit=rule.limit, remaining=remaining, reset_time=reset_time
            )

        except Exception as e:
            logger.error(f"Rate limit check failed for key {key}: {e}")
            # Fail open - allow request if Redis is down
            return True, RateLimitStatus(
                limit=rule.limit,
                remaining=rule.limit,
                reset_time=datetime.now() + timedelta(seconds=rule.window_seconds),
            )

    async def check_multiple_rules(
        self, key: str, rules: List[RateLimitRule]
    ) -> Tuple[bool, Dict[str, RateLimitStatus]]:
        """Check multiple rate limiting rules"""
        results = {}
        all_passed = True

        for rule in rules:
            if not rule.enabled:
                continue

            passed, status = await self.check_rate_limit(key, rule)
            results[rule.name] = status

            if not passed:
                all_passed = False

        return all_passed, results

    def _get_rate_limit_key(self, request: Request, rule: RateLimitRule) -> str:
        """Generate rate limit key based on scope"""
        if rule.scope == RateLimitScope.GLOBAL:
            return "global"
        elif rule.scope == RateLimitScope.PER_IP:
            return self._get_client_ip(request)
        elif rule.scope == RateLimitScope.PER_USER:
            user_id = self._extract_user_id(request)
            return user_id or self._get_client_ip(request)
        elif rule.scope == RateLimitScope.PER_API_KEY:
            api_key = self._extract_api_key(request)
            return (
                hashlib.sha256(api_key.encode()).hexdigest()[:16]
                if api_key
                else self._get_client_ip(request)
            )
        elif rule.scope == RateLimitScope.PER_ENDPOINT:
            return f"{request.method}:{request.url.path}"
        else:
            return self._get_client_ip(request)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Extract user ID from JWT token
            user_id = extract_user_id_from_request_header(auth_header)
            if user_id:
                return user_id
            # Fallback to generic api_user if JWT decode fails
            return "api_user"

        return None

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request"""
        # Check Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # Check X-API-Key header
        return request.headers.get("x-api-key")

    def _should_apply_rule(self, request: Request, rule: RateLimitRule) -> bool:
        """Check if rule should be applied to this request"""
        # Check if endpoints match
        if rule.endpoints:
            if not any(endpoint in request.url.path for endpoint in rule.endpoints):
                return False

        # Check if methods match
        if rule.methods:
            if request.method not in rule.methods:
                return False

        # Check bypass conditions
        if rule.bypass_ips:
            client_ip = self._get_client_ip(request)
            if client_ip in rule.bypass_ips:
                return False

        if rule.bypass_users:
            user_id = self._extract_user_id(request)
            if user_id and user_id in rule.bypass_users:
                return False

        return True

    async def get_current_usage(self, key: str, rule: RateLimitRule) -> int:
        """Get current usage count for a key"""
        redis_key = f"rate_limit:{rule.name}:{key}"
        current_time = time.time()
        window_start = current_time - rule.window_seconds

        try:
            # Remove expired entries and count current
            await self.redis.zremrangebyscore(redis_key, 0, window_start)
            return await self.redis.zcard(redis_key)
        except Exception as e:
            logger.error(f"Failed to get current usage: {e}")
            return 0

    async def get_rate_limit_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        stats = {
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_rules": len(self.rules),
            "rule_details": {},
        }

        for rule_name, rule in self.rules.items():
            # Get sample of keys for this rule
            pattern = f"rate_limit:{rule_name}:*"
            keys = await self.redis.keys(pattern)

            active_keys = 0
            total_usage = 0

            for key in keys[:100]:  # Sample first 100 keys
                usage = await self.redis.zcard(key)
                if usage > 0:
                    active_keys += 1
                    total_usage += usage

            rule_details = dict(
                cast(Dict[str, Any], stats.get("rule_details", {}) or {})
            )
            rule_details[rule_name] = {
                "enabled": rule.enabled,
                "limit": rule.limit,
                "window_seconds": rule.window_seconds,
                "scope": rule.scope.value,
                "active_keys": active_keys,
                "total_usage": total_usage,
                "endpoints": rule.endpoints,
            }
            stats["rule_details"] = rule_details

        return stats


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""

    def __init__(self, app: Any, rate_limiter: RateLimiter) -> None:
        super().__init__(app)
        self.rate_limiter = rate_limiter

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        """Apply rate limiting to requests"""
        # Get applicable rules for this request
        applicable_rules = [
            rule
            for rule in {
                **self.rate_limiter.default_rules,
                **self.rate_limiter.rules,
            }.values()
            if self.rate_limiter._should_apply_rule(request, rule)
        ]

        if not applicable_rules:
            return await call_next(request)

        # Check each applicable rule
        rate_limit_headers = {}

        for rule in applicable_rules:
            key = self.rate_limiter._get_rate_limit_key(request, rule)
            passed, status = await self.rate_limiter.check_rate_limit(key, rule)

            # Add rate limit headers
            rate_limit_headers.update(
                {
                    f"X-RateLimit-{rule.name}-Limit": str(status.limit),
                    f"X-RateLimit-{rule.name}-Remaining": str(status.remaining),
                    f"X-RateLimit-{rule.name}-Reset": str(
                        int(status.reset_time.timestamp())
                    ),
                }
            )

            if not passed:
                # Log rate limit violation
                logger.warning(
                    f"Rate limit exceeded for {key} on rule {rule.name}: "
                    f"{status.remaining}/{status.limit} remaining"
                )

                # Add retry-after header
                if status.retry_after:
                    rate_limit_headers["Retry-After"] = str(status.retry_after)

                # Return rate limit error
                error_response = {
                    "error": "Rate limit exceeded",
                    "rule": rule.name,
                    "limit": status.limit,
                    "retry_after": status.retry_after,
                    "reset_time": status.reset_time.isoformat(),
                }

                raise HTTPException(
                    status_code=429, detail=error_response, headers=rate_limit_headers
                )

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        for header, value in rate_limit_headers.items():
            response.headers[header] = value

        return response


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts limits based on system load"""

    def __init__(self, base_rate_limiter: RateLimiter, redis_client: redis.Redis):
        self.base_limiter = base_rate_limiter
        self.redis = redis_client
        self.load_threshold_high = 0.8
        self.load_threshold_critical = 0.9
        self.adaptation_factor = 0.5  # Reduce limits by 50% under high load

    async def get_system_load(self) -> float:
        """Get current system load (0.0 to 1.0)"""
        try:
            # Get system metrics from Redis (populated by monitoring)
            metrics = await self.redis.hgetall("system:metrics")  # type: ignore

            if metrics:
                cpu_usage = float(metrics.get("cpu_usage", 0)) / 100
                memory_usage = float(metrics.get("memory_usage", 0)) / 100

                # Return the higher of CPU or memory usage
                return max(cpu_usage, memory_usage)

            return 0.0

        except Exception as e:
            logger.error(f"Failed to get system load: {e}")
            return 0.0

    async def get_adaptive_limit(self, rule: RateLimitRule) -> int:
        """Get adapted limit based on system load"""
        system_load = await self.get_system_load()

        if system_load >= self.load_threshold_critical:
            # Severely reduce limits under critical load
            return max(1, int(rule.limit * 0.1))
        elif system_load >= self.load_threshold_high:
            # Reduce limits under high load
            return max(1, int(rule.limit * self.adaptation_factor))
        else:
            # Normal limits
            return rule.limit

    async def check_adaptive_rate_limit(
        self, key: str, rule: RateLimitRule, increment: int = 1
    ) -> Tuple[bool, RateLimitStatus]:
        """Check rate limit with adaptive adjustment"""
        # Get adapted limit
        adapted_limit = await self.get_adaptive_limit(rule)

        # Create temporary rule with adapted limit
        adapted_rule = RateLimitRule(
            name=rule.name,
            limit=adapted_limit,
            window_seconds=rule.window_seconds,
            scope=rule.scope,
            endpoints=rule.endpoints,
            methods=rule.methods,
            bypass_users=rule.bypass_users,
            bypass_ips=rule.bypass_ips,
            burst_limit=rule.burst_limit,
            enabled=rule.enabled,
        )

        return await self.base_limiter.check_rate_limit(key, adapted_rule, increment)


class CircuitBreakerRateLimiter:
    """Rate limiter with circuit breaker functionality"""

    def __init__(self, rate_limiter: RateLimiter, redis_client: redis.Redis):
        self.rate_limiter = rate_limiter
        self.redis = redis_client
        self.failure_threshold = 5
        self.recovery_timeout = 60
        self.half_open_max_calls = 3

    async def is_circuit_open(self, endpoint: str) -> bool:
        """Check if circuit breaker is open for endpoint"""
        circuit_key = f"circuit_breaker:{endpoint}"

        try:
            circuit_data = await self.redis.get(circuit_key)
            if not circuit_data:
                return False

            circuit_info = json.loads(circuit_data)

            # Check if circuit should recover
            if circuit_info["state"] == "open":
                if time.time() - circuit_info["opened_at"] > self.recovery_timeout:
                    # Move to half-open state
                    circuit_info["state"] = "half_open"
                    circuit_info["half_open_calls"] = 0
                    await self.redis.set(circuit_key, json.dumps(circuit_info), ex=300)
                    return False
                return True

            return circuit_info["state"] == "open"

        except Exception as e:
            logger.error(f"Circuit breaker check failed: {e}")
            return False

    async def record_success(self, endpoint: str) -> None:
        """Record successful request for circuit breaker"""
        circuit_key = f"circuit_breaker:{endpoint}"

        try:
            circuit_data = await self.redis.get(circuit_key)

            if circuit_data:
                circuit_info = json.loads(circuit_data)

                if circuit_info["state"] == "half_open":
                    circuit_info["half_open_calls"] += 1

                    if circuit_info["half_open_calls"] >= self.half_open_max_calls:
                        # Close circuit - recovery successful
                        await self.redis.delete(circuit_key)
                        logger.info(f"Circuit breaker closed for {endpoint}")
                    else:
                        await self.redis.set(
                            circuit_key, json.dumps(circuit_info), ex=300
                        )

                # Reset failure count on success
                circuit_info["failures"] = 0
                await self.redis.set(circuit_key, json.dumps(circuit_info), ex=300)

        except Exception as e:
            logger.error(f"Failed to record success for circuit breaker: {e}")

    async def record_failure(self, endpoint: str) -> None:
        """Record failed request for circuit breaker"""
        circuit_key = f"circuit_breaker:{endpoint}"

        try:
            circuit_data = await self.redis.get(circuit_key)

            if circuit_data:
                circuit_info = json.loads(circuit_data)
            else:
                circuit_info = {
                    "state": "closed",
                    "failures": 0,
                    "opened_at": None,
                    "half_open_calls": 0,
                }

            circuit_info["failures"] += 1

            if circuit_info["failures"] >= self.failure_threshold:
                # Open circuit
                circuit_info["state"] = "open"
                circuit_info["opened_at"] = time.time()
                logger.warning(
                    f"Circuit breaker opened for {endpoint} after {self.failure_threshold} failures"
                )

            await self.redis.set(circuit_key, json.dumps(circuit_info), ex=3600)

        except Exception as e:
            logger.error(f"Failed to record failure for circuit breaker: {e}")


class TradingRateLimiter:
    """Specialized rate limiter for trading operations"""

    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.trading_rules = {
            "orders_per_minute": RateLimitRule(
                name="orders_per_minute",
                limit=10,
                window_seconds=60,
                scope=RateLimitScope.PER_USER,
                endpoints=["/orders"],
                methods=["POST"],
            ),
            "position_changes_per_hour": RateLimitRule(
                name="position_changes_per_hour",
                limit=50,
                window_seconds=3600,
                scope=RateLimitScope.PER_USER,
                endpoints=["/positions"],
                methods=["POST", "PUT", "DELETE"],
            ),
            "daily_trade_volume": RateLimitRule(
                name="daily_trade_volume",
                limit=1000000,  # $1M daily limit
                window_seconds=86400,
                scope=RateLimitScope.PER_USER,
            ),
        }

        # Add trading rules to main rate limiter
        for rule in self.trading_rules.values():
            self.rate_limiter.add_rule(rule)

    async def check_trading_limits(
        self, user_id: str, trade_value: float
    ) -> Tuple[bool, Dict[str, TypeAny]]:
        """Check trading-specific rate limits"""
        # Check volume limit
        current_time = time.time()
        day_start = current_time - 86400

        try:
            # Get current daily volume
            volume_entries = await self.rate_limiter.redis.zrangebyscore(
                f"rate_limit:daily_trade_volume:{user_id}",
                day_start,
                current_time,
                withscores=True,
            )

            current_volume = sum(float(entry[0]) for entry in volume_entries)

            if (
                current_volume + trade_value
                > self.trading_rules["daily_trade_volume"].limit
            ):
                return False, {
                    "error": "Daily trading volume limit exceeded",
                    "current_volume": current_volume,
                    "limit": self.trading_rules["daily_trade_volume"].limit,
                    "trade_value": trade_value,
                }

            # Record this trade volume
            await self.rate_limiter.redis.zadd(
                f"rate_limit:daily_trade_volume:{user_id}",
                {str(trade_value): current_time},
            )

            return True, {
                "current_volume": current_volume + trade_value,
                "limit": self.trading_rules["daily_trade_volume"].limit,
                "remaining": self.trading_rules["daily_trade_volume"].limit
                - current_volume
                - trade_value,
            }

        except Exception as e:
            logger.error(f"Trading limit check failed: {e}")
            return True, {"error": "Rate limit check unavailable"}


# Rate limiting decorators
def rate_limit(
    rule_name: str, custom_key: Optional[str] = None, increment: int = 1
) -> Any:
    """Decorator for function-level rate limiting"""

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get rate limiter from context (would need to be set up)
            rate_limiter = getattr(func, "_rate_limiter", None)
            if not rate_limiter:
                return await func(*args, **kwargs)

            # Get rule
            rule = rate_limiter.rules.get(rule_name) or rate_limiter.default_rules.get(
                rule_name
            )
            if not rule:
                logger.warning(f"Rate limit rule not found: {rule_name}")
                return await func(*args, **kwargs)

            # Use custom key or function name
            key = custom_key or func.__name__

            # Check rate limit
            passed, status = await rate_limiter.check_rate_limit(key, rule, increment)

            if not passed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "rule": rule_name,
                        "retry_after": status.retry_after,
                    },
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Rate limit configuration management
class RateLimitConfig:
    """Configuration management for rate limiting"""

    @staticmethod
    def load_from_env() -> Dict[str, RateLimitRule]:
        """Load rate limiting rules from environment variables"""
        import os

        rules = {}

        # API general limits
        api_limit = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", 100))
        rules["api_general"] = RateLimitRule(
            name="api_general",
            limit=api_limit,
            window_seconds=60,
            scope=RateLimitScope.PER_IP,
        )

        # Trading limits
        trading_limit = int(os.getenv("RATE_LIMIT_TRADING_RPM", 10))
        rules["api_trading"] = RateLimitRule(
            name="api_trading",
            limit=trading_limit,
            window_seconds=60,
            scope=RateLimitScope.PER_USER,
            endpoints=["/trades", "/orders", "/positions"],
            methods=["POST", "PUT", "DELETE"],
        )

        # Market data limits
        market_data_limit = int(os.getenv("RATE_LIMIT_MARKET_DATA_RPM", 30))
        rules["api_market_data"] = RateLimitRule(
            name="api_market_data",
            limit=market_data_limit,
            window_seconds=60,
            scope=RateLimitScope.PER_IP,
            endpoints=["/market-data", "/quotes", "/charts"],
        )

        # Export limits
        export_limit = int(os.getenv("RATE_LIMIT_EXPORT_RPM", 5))
        rules["api_export"] = RateLimitRule(
            name="api_export",
            limit=export_limit,
            window_seconds=60,
            scope=RateLimitScope.PER_USER,
            endpoints=["/export"],
        )

        # Burst protection
        burst_limit = int(os.getenv("RATE_LIMIT_BURST_SIZE", 10))
        rules["burst_protection"] = RateLimitRule(
            name="burst_protection",
            limit=burst_limit,
            window_seconds=1,
            scope=RateLimitScope.PER_IP,
            burst_limit=burst_limit * 2,
        )

        return rules

    @staticmethod
    def create_custom_rule(
        name: str,
        limit: int,
        window_seconds: int,
        scope: str,
        endpoints: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
    ) -> RateLimitRule:
        """Create a custom rate limiting rule"""
        return RateLimitRule(
            name=name,
            limit=limit,
            window_seconds=window_seconds,
            scope=RateLimitScope(scope),
            endpoints=endpoints,
            methods=methods,
        )


# Monitoring and analytics
class RateLimitMonitor:
    """Monitor rate limiting effectiveness and usage"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def get_top_limited_ips(self, limit: int = 10) -> List[Dict[str, TypeAny]]:
        """Get IPs that hit rate limits most frequently"""
        try:
            # Get rate limit violations from the last 24 hours
            pattern = "rate_limit_violations:*"
            keys = await self.redis.keys(pattern)

            violations = {}
            for key in keys:
                ip = key.split(":")[-1]
                count = await self.redis.get(key)
                if count:
                    violations[ip] = int(count)

            # Sort by violation count
            sorted_violations = sorted(
                violations.items(), key=lambda x: x[1], reverse=True
            )

            return [
                {"ip": ip, "violations": count}
                for ip, count in sorted_violations[:limit]
            ]

        except Exception as e:
            logger.error(f"Failed to get top limited IPs: {e}")
            return []

    async def get_rate_limit_effectiveness(self) -> Dict[str, TypeAny]:
        """Analyze rate limiting effectiveness"""
        try:
            # Get statistics from the last hour
            _ = time.time()

            stats = {
                "blocked_requests": 0,
                "allowed_requests": 0,
                "top_blocked_endpoints": {},
                "top_blocked_ips": {},
            }

            # This would be populated by actual rate limiting events
            # For now, return placeholder structure

            return stats

        except Exception as e:
            logger.error(f"Failed to analyze rate limiting effectiveness: {e}")
            return {}


# Utility functions
async def cleanup_rate_limit_data(
    redis_client: redis.Redis, max_age_hours: int = 24
) -> None:
    """Clean up old rate limiting data"""
    try:
        pattern = "rate_limit:*"
        keys = await redis_client.keys(pattern)

        cleaned_count = 0
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        for key in keys:
            # Remove old entries from sorted sets
            removed = await redis_client.zremrangebyscore(key, 0, cutoff_time)
            if removed > 0:
                cleaned_count += removed

            # Remove empty sets
            if await redis_client.zcard(key) == 0:
                await redis_client.delete(key)

        logger.info(f"Rate limit cleanup completed: {cleaned_count} entries removed")

    except Exception as e:
        logger.error(f"Rate limit data cleanup failed: {e}")


async def get_rate_limit_metrics(redis_client: redis.Redis) -> Dict[str, TypeAny]:
    """Get comprehensive rate limiting metrics"""
    try:
        metrics = {
            "total_active_limits": 0,
            "rules_by_type": {},
            "top_consumers": [],
            "recent_blocks": 0,
        }

        # Count active rate limit keys
        pattern = "rate_limit:*"
        keys = await redis_client.keys(pattern)
        metrics["total_active_limits"] = len(keys)

        # Analyze key patterns
        rule_counts: Dict[str, int] = {}
        for key in keys:
            parts = key.split(":")
            if len(parts) >= 2:
                rule_name = parts[1]
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1

        metrics["rules_by_type"] = rule_counts

        return metrics

    except Exception as e:
        logger.error(f"Failed to get rate limit metrics: {e}")
        return {}


def create_rate_limiter_from_config(
    redis_client: redis.Redis, config: Dict[str, TypeAny]
) -> RateLimiter:
    """Create rate limiter with configuration"""
    rate_limiter = RateLimiter(redis_client)

    # Load environment-based rules
    env_rules = RateLimitConfig.load_from_env()
    for rule in env_rules.values():
        rate_limiter.add_rule(rule)

    # Load custom rules from config if provided
    if "custom_rules" in config:
        for rule_config in config["custom_rules"]:
            custom_rule = RateLimitConfig.create_custom_rule(**rule_config)
            rate_limiter.add_rule(custom_rule)

    return rate_limiter


# Health check for rate limiting system
async def health_check_rate_limiting(redis_client: redis.Redis) -> Dict[str, TypeAny]:
    """Perform health check on rate limiting system"""
    try:
        # Test Redis connectivity
        await redis_client.ping()

        # Test rate limiting functionality
        test_key = f"health_check_{int(time.time())}"
        test_rule = RateLimitRule(
            name="health_check", limit=1, window_seconds=60, scope=RateLimitScope.GLOBAL
        )

        rate_limiter = RateLimiter(redis_client)
        passed, status = await rate_limiter.check_rate_limit(test_key, test_rule)

        # Clean up test data
        await redis_client.delete(f"rate_limit:health_check:{test_key}")

        return {
            "status": "healthy",
            "redis_connected": True,
            "rate_limiting_functional": passed,
            "test_completed_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Rate limiting health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "redis_connected": False,
            "rate_limiting_functional": False,
        }
