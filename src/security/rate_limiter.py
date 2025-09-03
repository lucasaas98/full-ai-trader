#!/usr/bin/env python3
"""
AI Trading System Rate Limiter

This module provides comprehensive rate limiting functionality for the AI Trading System
to protect against abuse, ensure fair usage, and maintain system stability. It implements
multiple rate limiting strategies including token bucket, sliding window, and adaptive
rate limiting based on system load.

Features:
- Multiple rate limiting algorithms
- Per-user and per-service limits
- Adaptive rate limiting based on system load
- Redis-backed distributed rate limiting
- Rate limit metrics and monitoring
- Whitelist/blacklist support
- Custom rate limit rules
- Circuit breaker integration
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram
from redis import asyncio as aioredis

logger = structlog.get_logger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"


class RateLimitScope(Enum):
    """Rate limit scope"""

    GLOBAL = "global"
    PER_USER = "per_user"
    PER_IP = "per_ip"
    PER_SERVICE = "per_service"
    PER_ENDPOINT = "per_endpoint"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""

    name: str
    algorithm: RateLimitAlgorithm
    scope: RateLimitScope
    requests: int
    window_seconds: int
    burst_size: Optional[int] = None
    enabled: bool = True
    priority: int = 0

    # Adaptive rate limiting parameters
    min_requests: Optional[int] = None
    max_requests: Optional[int] = None
    load_threshold: float = 0.8

    # Custom conditions
    conditions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = self.requests


@dataclass
class RateLimitResult:
    """Result of rate limit check"""

    allowed: bool
    remaining: int
    reset_time: float
    retry_after: Optional[int] = None
    rule_name: Optional[str] = None
    current_usage: int = 0


class TokenBucket:
    """Token bucket rate limiter implementation"""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        redis_client: aioredis.Redis,
        key_prefix: str,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.redis = redis_client
        self.key_prefix = key_prefix

    async def consume(self, identifier: str, tokens: int = 1) -> RateLimitResult:
        """Consume tokens from the bucket"""
        key = f"{self.key_prefix}:bucket:{identifier}"
        now = time.time()

        # Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or now

        -- Calculate tokens to add based on time elapsed
        local time_elapsed = now - last_refill
        local tokens_to_add = time_elapsed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)

        -- Check if we have enough tokens
        if tokens >= tokens_requested then
            tokens = tokens - tokens_requested
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
            return {1, tokens, now + ((capacity - tokens) / refill_rate)}
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
            redis.call('EXPIRE', key, 3600)
            return {0, tokens, now + ((tokens_requested - tokens) / refill_rate)}
        end
        """

        result = await self.redis.eval(
            lua_script, 1, key, self.capacity, self.refill_rate, tokens, now
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        reset_time = float(result[2])

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=int(reset_time - now) if not allowed else None,
            current_usage=self.capacity - remaining,
        )


class SlidingWindowLimiter:
    """Sliding window rate limiter implementation"""

    def __init__(
        self,
        requests: int,
        window_seconds: int,
        redis_client: aioredis.Redis,
        key_prefix: str,
    ):
        self.requests = requests
        self.window_seconds = window_seconds
        self.redis = redis_client
        self.key_prefix = key_prefix

    async def check_limit(self, identifier: str) -> RateLimitResult:
        """Check rate limit using sliding window"""
        key = f"{self.key_prefix}:sliding:{identifier}"
        now = time.time()
        window_start = now - self.window_seconds

        # Lua script for atomic sliding window operation
        lua_script = """
        local key = KEYS[1]
        local window_start = tonumber(ARGV[1])
        local now = tonumber(ARGV[2])
        local max_requests = tonumber(ARGV[3])

        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

        -- Count current requests in window
        local current_count = redis.call('ZCARD', key)

        if current_count < max_requests then
            -- Add new request
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, 3600)
            return {1, max_requests - current_count - 1, window_start + 3600}
        else
            -- Find oldest entry to determine reset time
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local reset_time = window_start + 3600
            if #oldest > 0 then
                reset_time = tonumber(oldest[2]) + 3600
            end
            return {0, 0, reset_time}
        end
        """

        result = await self.redis.eval(
            lua_script, 1, key, window_start, now, self.requests
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        reset_time = float(result[2])

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=int(reset_time - now) if not allowed else None,
            current_usage=self.requests - remaining,
        )


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on system load"""

    def __init__(
        self, base_rule: RateLimitRule, redis_client: aioredis.Redis, key_prefix: str
    ):
        self.base_rule = base_rule
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.load_key = f"{key_prefix}:system_load"

        # Initialize with base limits
        self.current_limit = base_rule.requests
        self.min_limit = base_rule.min_requests or max(1, base_rule.requests // 4)
        self.max_limit = base_rule.max_requests or base_rule.requests * 2

    async def get_current_system_load(self) -> float:
        """Get current system load metrics"""
        try:
            # Get system metrics from Redis
            load_data = await self.redis.hgetall(self.load_key)

            if not load_data:
                return 0.5  # Default moderate load

            cpu_usage = float(load_data.get("cpu_usage", 0)) / 100
            memory_usage = float(load_data.get("memory_usage", 0)) / 100
            active_connections = int(load_data.get("active_connections", 0))
            error_rate = float(load_data.get("error_rate", 0))

            # Calculate composite load score
            load_score = (
                cpu_usage * 0.3
                + memory_usage * 0.3
                + min(active_connections / 1000, 1.0) * 0.2
                + error_rate * 0.2
            )

            return min(1.0, load_score)

        except Exception as e:
            logger.warning("Failed to get system load", error=str(e))
            return 0.5

    async def adapt_rate_limit(self) -> int:
        """Adapt rate limit based on current system load"""
        system_load = await self.get_current_system_load()

        if system_load > self.base_rule.load_threshold:
            # High load - reduce limits
            reduction_factor = min(
                0.5, (system_load - self.base_rule.load_threshold) * 2
            )
            adapted_limit = int(self.base_rule.requests * (1 - reduction_factor))
            self.current_limit = max(self.min_limit, adapted_limit)
        else:
            # Normal/low load - allow higher limits
            increase_factor = (
                self.base_rule.load_threshold - system_load
            ) / self.base_rule.load_threshold
            adapted_limit = int(self.base_rule.requests * (1 + increase_factor))
            self.current_limit = min(self.max_limit, adapted_limit)

        return self.current_limit

    async def check_limit(self, identifier: str) -> RateLimitResult:
        """Check adaptive rate limit"""
        current_limit = await self.adapt_rate_limit()

        # Use sliding window with adapted limit
        limiter = SlidingWindowLimiter(
            current_limit, self.base_rule.window_seconds, self.redis, self.key_prefix
        )

        result = await limiter.check_limit(identifier)
        result.rule_name = f"{self.base_rule.name}_adaptive"

        return result


class RateLimiter:
    """Main rate limiter with multiple strategies and rules"""

    def __init__(self, redis_client: aioredis.Redis, config: Dict[str, Any]):
        self.redis = redis_client
        self.config = config
        self.key_prefix = config.get("RATE_LIMIT_PREFIX", "rate_limit")

        # Rate limiting rules
        self.rules: Dict[str, RateLimitRule] = {}
        self.limiters: Dict[str, Any] = {}

        # Whitelist/blacklist
        self.whitelisted_ips: set = set(config.get("RATE_LIMIT_WHITELIST_IPS", []))
        self.blacklisted_ips: set = set(config.get("RATE_LIMIT_BLACKLIST_IPS", []))
        self.whitelisted_users: set = set(config.get("RATE_LIMIT_WHITELIST_USERS", []))

        # Metrics
        self.rate_limit_checks_total = Counter(
            "rate_limit_checks_total",
            "Total rate limit checks",
            ["rule_name", "scope", "result"],
        )

        self.rate_limit_duration = Histogram(
            "rate_limit_check_duration_seconds",
            "Time spent on rate limit checks",
            ["algorithm"],
        )

        self.current_rate_limits = Gauge(
            "current_rate_limits", "Current rate limit values", ["rule_name", "scope"]
        )

        # Initialize default rules
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default rate limiting rules"""
        # Global API rate limit
        self.add_rule(
            RateLimitRule(
                name="global_api",
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                scope=RateLimitScope.GLOBAL,
                requests=int(self.config.get("RATE_LIMIT_GLOBAL_REQUESTS", 1000)),
                window_seconds=int(self.config.get("RATE_LIMIT_GLOBAL_WINDOW", 60)),
                priority=100,
            )
        )

        # Per-user rate limit
        self.add_rule(
            RateLimitRule(
                name="per_user",
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                scope=RateLimitScope.PER_USER,
                requests=int(self.config.get("RATE_LIMIT_USER_REQUESTS", 100)),
                window_seconds=int(self.config.get("RATE_LIMIT_USER_WINDOW", 60)),
                burst_size=int(self.config.get("RATE_LIMIT_USER_BURST", 50)),
                priority=80,
            )
        )

        # Per-IP rate limit
        self.add_rule(
            RateLimitRule(
                name="per_ip",
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                scope=RateLimitScope.PER_IP,
                requests=int(self.config.get("RATE_LIMIT_IP_REQUESTS", 200)),
                window_seconds=int(self.config.get("RATE_LIMIT_IP_WINDOW", 60)),
                priority=90,
            )
        )

        # Trading-specific rate limits
        self.add_rule(
            RateLimitRule(
                name="trading_orders",
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                scope=RateLimitScope.PER_USER,
                requests=int(self.config.get("RATE_LIMIT_TRADING_REQUESTS", 20)),
                window_seconds=int(self.config.get("RATE_LIMIT_TRADING_WINDOW", 60)),
                burst_size=5,
                priority=70,
                conditions={"endpoint_pattern": "/api/v1/orders/*"},
            )
        )

        # Market data rate limits
        self.add_rule(
            RateLimitRule(
                name="market_data",
                algorithm=RateLimitAlgorithm.ADAPTIVE,
                scope=RateLimitScope.PER_SERVICE,
                requests=int(self.config.get("RATE_LIMIT_MARKET_DATA_REQUESTS", 500)),
                window_seconds=int(
                    self.config.get("RATE_LIMIT_MARKET_DATA_WINDOW", 60)
                ),
                min_requests=100,
                max_requests=1000,
                load_threshold=0.7,
                priority=60,
                conditions={"service_name": "data_collector"},
            )
        )

        # Admin API rate limits (higher limits)
        self.add_rule(
            RateLimitRule(
                name="admin_api",
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                scope=RateLimitScope.PER_USER,
                requests=int(self.config.get("RATE_LIMIT_ADMIN_REQUESTS", 500)),
                window_seconds=int(self.config.get("RATE_LIMIT_ADMIN_WINDOW", 60)),
                priority=50,
                conditions={"user_role": "admin"},
            )
        )

    def add_rule(self, rule: RateLimitRule):
        """Add a rate limiting rule"""
        self.rules[rule.name] = rule

        # Initialize limiter based on algorithm
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            self.limiters[rule.name] = TokenBucket(
                capacity=rule.burst_size or rule.requests,
                refill_rate=rule.requests / rule.window_seconds,
                redis_client=self.redis,
                key_prefix=f"{self.key_prefix}:{rule.name}",
            )
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            self.limiters[rule.name] = SlidingWindowLimiter(
                requests=rule.requests,
                window_seconds=rule.window_seconds,
                redis_client=self.redis,
                key_prefix=f"{self.key_prefix}:{rule.name}",
            )
        elif rule.algorithm == RateLimitAlgorithm.ADAPTIVE:
            self.limiters[rule.name] = AdaptiveRateLimiter(
                base_rule=rule,
                redis_client=self.redis,
                key_prefix=f"{self.key_prefix}:{rule.name}",
            )

        # Update metrics
        self.current_rate_limits.labels(
            rule_name=rule.name, scope=rule.scope.value
        ).set(rule.requests)

        logger.info(
            "Rate limit rule added",
            rule_name=rule.name,
            algorithm=rule.algorithm.value,
            requests=rule.requests,
        )

    def remove_rule(self, rule_name: str):
        """Remove a rate limiting rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            if rule_name in self.limiters:
                del self.limiters[rule_name]
            logger.info("Rate limit rule removed", rule_name=rule_name)

    async def check_rate_limit(
        self,
        identifier: str,
        scope: RateLimitScope = RateLimitScope.PER_USER,
        endpoint: str = "",
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        service_name: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> RateLimitResult:
        """Check rate limits against applicable rules"""

        # Check whitelist first
        if self._is_whitelisted(user_id, source_ip):
            return RateLimitResult(
                allowed=True,
                remaining=999999,
                reset_time=time.time() + 3600,
                rule_name="whitelist",
            )

        # Check blacklist
        if self._is_blacklisted(source_ip):
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=time.time() + 3600,
                retry_after=3600,
                rule_name="blacklist",
            )

        # Get applicable rules sorted by priority
        applicable_rules = self._get_applicable_rules(
            scope=scope,
            endpoint=endpoint,
            service_name=service_name,
            user_role=user_role,
        )

        # Check each applicable rule
        for rule in applicable_rules:
            _ = time.time()

            try:
                with self.rate_limit_duration.labels(
                    algorithm=rule.algorithm.value
                ).time():
                    # Generate identifier based on scope
                    rate_limit_identifier = self._generate_identifier(
                        rule.scope, identifier, user_id, source_ip, service_name
                    )

                    # Check the specific limiter
                    limiter = self.limiters.get(rule.name)
                    if not limiter:
                        continue

                    if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                        result = await limiter.consume(rate_limit_identifier)
                    else:
                        result = await limiter.check_limit(rate_limit_identifier)

                    result.rule_name = rule.name

                    # Record metrics
                    self.rate_limit_checks_total.labels(
                        rule_name=rule.name,
                        scope=rule.scope.value,
                        result="allowed" if result.allowed else "denied",
                    ).inc()

                    # If any rule denies, return denial
                    if not result.allowed:
                        logger.warning(
                            "Rate limit exceeded",
                            rule_name=rule.name,
                            identifier=rate_limit_identifier,
                            remaining=result.remaining,
                        )

                        # Log rate limit violation for audit
                        await self._log_rate_limit_violation(rule, identifier, result)

                        return result

            except Exception as e:
                logger.error(
                    "Rate limit check failed", rule_name=rule.name, error=str(e)
                )
                # On error, allow request but log the issue
                continue

        # All rules passed
        return RateLimitResult(
            allowed=True,
            remaining=min(
                [
                    r.remaining
                    for r in [
                        await limiter.check_limit(
                            self._generate_identifier(
                                rule.scope, identifier, user_id, source_ip, service_name
                            )
                        )
                        for rule, limiter in [
                            (r, self.limiters.get(r.name)) for r in applicable_rules
                        ]
                        if limiter
                    ]
                    if hasattr(r, "remaining")
                ]
            ),
            reset_time=time.time() + 60,
            rule_name="composite",
        )

    def _get_applicable_rules(
        self,
        scope: RateLimitScope,
        endpoint: str = "",
        service_name: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> List[RateLimitRule]:
        """Get applicable rules for the request"""
        applicable = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check scope compatibility
            if rule.scope != scope and rule.scope != RateLimitScope.GLOBAL:
                continue

            # Check custom conditions
            if rule.conditions:
                if "endpoint_pattern" in rule.conditions:
                    pattern = rule.conditions["endpoint_pattern"]
                    if not self._matches_pattern(endpoint, pattern):
                        continue

                if "service_name" in rule.conditions:
                    if rule.conditions["service_name"] != service_name:
                        continue

                if "user_role" in rule.conditions:
                    if rule.conditions["user_role"] != user_role:
                        continue

            applicable.append(rule)

        # Sort by priority (higher priority first)
        return sorted(applicable, key=lambda r: r.priority, reverse=True)

    def _generate_identifier(
        self,
        scope: RateLimitScope,
        base_identifier: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        service_name: Optional[str] = None,
    ) -> str:
        """Generate rate limit identifier based on scope"""
        if scope == RateLimitScope.GLOBAL:
            return "global"
        elif scope == RateLimitScope.PER_USER:
            return f"user:{user_id or base_identifier}"
        elif scope == RateLimitScope.PER_IP:
            return f"ip:{source_ip or base_identifier}"
        elif scope == RateLimitScope.PER_SERVICE:
            return f"service:{service_name or base_identifier}"
        elif scope == RateLimitScope.PER_ENDPOINT:
            return f"endpoint:{base_identifier}"
        else:
            return base_identifier

    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches pattern (simple wildcard support)"""
        import fnmatch

        return fnmatch.fnmatch(text, pattern)

    def _is_whitelisted(
        self, user_id: Optional[str] = None, source_ip: Optional[str] = None
    ) -> bool:
        """Check if request is whitelisted"""
        if user_id and user_id in self.whitelisted_users:
            return True
        if source_ip and source_ip in self.whitelisted_ips:
            return True
        return False

    def _is_blacklisted(self, source_ip: Optional[str] = None) -> bool:
        """Check if request is blacklisted"""
        if source_ip and source_ip in self.blacklisted_ips:
            return True
        return False

    async def _log_rate_limit_violation(
        self, rule: RateLimitRule, identifier: str, result: RateLimitResult
    ):
        """Log rate limit violation for audit purposes"""
        violation_data = {
            "rule_name": rule.name,
            "identifier": identifier,
            "scope": rule.scope.value,
            "algorithm": rule.algorithm.value,
            "limit": rule.requests,
            "window_seconds": rule.window_seconds,
            "current_usage": result.current_usage,
            "retry_after": result.retry_after,
        }

        # Store violation in Redis for analysis
        violation_key = f"{self.key_prefix}:violations:{rule.name}:{identifier}"
        await self.redis.lpush(violation_key, json.dumps(violation_data))
        await self.redis.expire(violation_key, 86400)  # Keep for 24 hours

    async def get_rate_limit_status(
        self, identifier: str, scope: RateLimitScope
    ) -> Dict[str, Any]:
        """Get current rate limit status for identifier"""
        status = {}

        for rule_name, rule in self.rules.items():
            if rule.scope != scope and rule.scope != RateLimitScope.GLOBAL:
                continue

            limiter = self.limiters.get(rule_name)
            if not limiter:
                continue

            try:
                rate_limit_identifier = self._generate_identifier(scope, identifier)

                if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                    # For token bucket, we need to check without consuming
                    result = await limiter.consume(rate_limit_identifier, 0)
                else:
                    # For other algorithms, get current state
                    key = f"{limiter.key_prefix}:sliding:{rate_limit_identifier}"
                    current_count = await self.redis.zcard(key)
                    result = RateLimitResult(
                        allowed=current_count < rule.requests,
                        remaining=max(0, rule.requests - current_count),
                        reset_time=time.time() + rule.window_seconds,
                        current_usage=current_count,
                    )

                status[rule_name] = {
                    "limit": rule.requests,
                    "remaining": result.remaining,
                    "reset_time": result.reset_time,
                    "window_seconds": rule.window_seconds,
                    "algorithm": rule.algorithm.value,
                }

            except Exception as e:
                logger.error(
                    "Failed to get rate limit status", rule_name=rule_name, error=str(e)
                )
                status[rule_name] = {"error": str(e)}

        return status

    async def update_system_load_metrics(
        self,
        cpu_usage: float,
        memory_usage: float,
        active_connections: int,
        error_rate: float,
    ):
        """Update system load metrics for adaptive rate limiting"""
        load_data = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "active_connections": active_connections,
            "error_rate": error_rate,
            "timestamp": time.time(),
        }

        await self.redis.hset(f"{self.key_prefix}:system_load", mapping=load_data)
        await self.redis.expire(f"{self.key_prefix}:system_load", 300)  # 5 minutes TTL

    async def add_to_whitelist(
        self, user_id: Optional[str] = None, ip_address: Optional[str] = None
    ):
        """Add user or IP to whitelist"""
        if user_id:
            await self.redis.sadd(f"{self.key_prefix}:whitelist:users", user_id)
            logger.info("User added to whitelist", user_id=user_id)

        if ip_address:
            await self.redis.sadd(f"{self.key_prefix}:whitelist:ips", ip_address)
            logger.info("IP added to whitelist", ip_address=ip_address)

    async def add_to_blacklist(self, ip_address: str, duration_seconds: int = 3600):
        """Add IP to blacklist for specified duration"""
        self.blacklisted_ips.add(ip_address)
        await self.redis.sadd(f"{self.key_prefix}:blacklist:ips", ip_address)
        await self.redis.expire(f"{self.key_prefix}:blacklist:ips", duration_seconds)

        logger.warning(
            "IP added to blacklist", ip_address=ip_address, duration=duration_seconds
        )

    async def remove_from_whitelist(
        self, user_id: Optional[str] = None, ip_address: Optional[str] = None
    ):
        """Remove user or IP from whitelist"""
        if user_id and user_id in self.whitelisted_users:
            self.whitelisted_users.remove(user_id)
            await self.redis.srem(f"{self.key_prefix}:whitelist:users", user_id)
            logger.info("User removed from whitelist", user_id=user_id)

        if ip_address and ip_address in self.whitelisted_ips:
            self.whitelisted_ips.remove(ip_address)
            await self.redis.srem(f"{self.key_prefix}:whitelist:ips", ip_address)
            logger.info("IP removed from whitelist", ip_address=ip_address)

    async def get_violation_stats(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get rate limit violation statistics"""
        end_time = time.time()
        start_time = end_time - (time_range_hours * 3600)

        stats = {
            "time_range_hours": time_range_hours,
            "total_violations": 0,
            "violations_by_rule": {},
            "violations_by_hour": {},
            "top_violators": {},
            "violation_trends": [],
        }

        # Search for violations in Redis
        violation_keys = await self.redis.keys(f"{self.key_prefix}:violations:*")

        for key in violation_keys:
            violations = await self.redis.lrange(key, 0, -1)

            for violation_json in violations:
                try:
                    violation = json.loads(violation_json)
                    violation_time = violation.get("timestamp", end_time)

                    if violation_time >= start_time:
                        stats["total_violations"] += 1

                        # Count by rule
                        rule_name = violation.get("rule_name", "unknown")
                        stats["violations_by_rule"][rule_name] = (
                            stats["violations_by_rule"].get(rule_name, 0) + 1
                        )

                        # Count by hour
                        hour = datetime.fromtimestamp(violation_time).strftime("%H")
                        stats["violations_by_hour"][hour] = (
                            stats["violations_by_hour"].get(hour, 0) + 1
                        )

                except Exception as e:
                    logger.warning(f"Failed to parse violation data: {e}")
                    continue

        return stats
