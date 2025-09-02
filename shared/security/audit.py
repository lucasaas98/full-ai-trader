"""
Audit Logging System for AI Trading System

This module provides comprehensive audit logging capabilities for:
- Trade executions and modifications
- Risk management decisions
- Configuration changes
- User actions and API calls
- System events and errors

Version: 1.0.0 - Updated for Pydantic v2 compatibility
Note: All imports verified and syntax validated
"""

import asyncio
import datetime as dt
import json
import logging
from datetime import timezone
from typing import Any, Dict, List, Optional, cast

# Explicit type aliases for clarity
TypeAny = Any
TimeDelta = dt.timedelta
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from enum import Enum

import asyncpg
import redis.asyncio as redis
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .jwt_utils import extract_user_id_from_request_header, get_default_jwt_manager

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of events that can be audited"""

    TRADE_EXECUTION = "trade_execution"
    TRADE_MODIFICATION = "trade_modification"
    TRADE_CANCELLATION = "trade_cancellation"
    POSITION_OPEN = "position_open"
    POSITION_CLOSE = "position_close"
    RISK_DECISION = "risk_decision"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    STRATEGY_EXECUTION = "strategy_execution"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_ACCESS = "api_access"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    MAINTENANCE_START = "maintenance_start"
    MAINTENANCE_END = "maintenance_end"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    SECURITY_EVENT = "security_event"
    ERROR_EVENT = "error_event"


class AuditSeverity(Enum):
    """Severity levels for audit events"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure"""

    timestamp: dt.datetime
    event_type: AuditEventType
    severity: AuditSeverity
    service_name: str
    action: str
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    changes: Optional[Dict[str, Any]] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["event_type"] = self.event_type.value
        result["severity"] = self.severity.value
        return result


class AuditLogger:
    """Core audit logging functionality"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis: redis.Redis = redis_client
        self.batch_size = 100
        self.batch_timeout = 5.0
        self.pending_events: List[AuditEvent] = []
        self.batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize audit logging system"""
        await self._ensure_audit_table()
        self._batch_task = asyncio.create_task(self._batch_processor())
        logger.info("Audit logging system initialized")

    async def shutdown(self):
        """Shutdown audit logging system"""
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Flush any remaining events
        if self.pending_events:
            await self._flush_batch()

        logger.info("Audit logging system shutdown")

    async def _ensure_audit_table(self):
        """Ensure audit logs table exists"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id BIGSERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL,
            event_type VARCHAR(50) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            service_name VARCHAR(50) NOT NULL,
            action VARCHAR(100) NOT NULL,
            entity_type VARCHAR(50),
            entity_id VARCHAR(100),
            user_id VARCHAR(100),
            session_id VARCHAR(100),
            ip_address INET,
            user_agent TEXT,
            request_id VARCHAR(100),
            changes JSONB,
            old_values JSONB,
            new_values JSONB,
            success BOOLEAN NOT NULL DEFAULT true,
            error_message TEXT,
            execution_time_ms FLOAT,
            metadata JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_service ON audit_logs(service_name);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_user ON audit_logs(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_entity ON audit_logs(entity_type, entity_id);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_success ON audit_logs(success);
        CREATE INDEX IF NOT EXISTS idx_audit_logs_severity ON audit_logs(severity);

        -- Create partial index for errors
        CREATE INDEX IF NOT EXISTS idx_audit_logs_errors
        ON audit_logs(timestamp, service_name)
        WHERE success = false;
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(create_table_sql)

    async def log_event(self, event: AuditEvent):
        """Log an audit event (async, batched)"""
        async with self.batch_lock:
            self.pending_events.append(event)

        # Also store in Redis for real-time access
        try:
            await cast(Any, self.redis.lpush)(
                "audit:recent", json.dumps(event.to_dict())
            )
            # Keep only last 1000 events in Redis
            await cast(Any, self.redis.ltrim)("audit:recent", 0, 999)
        except Exception as e:
            logger.error(f"Failed to store audit event in Redis: {e}")

    async def _batch_processor(self):
        """Background task to process batched audit events"""
        while True:
            try:
                await asyncio.sleep(self.batch_timeout)

                async with self.batch_lock:
                    if len(self.pending_events) >= self.batch_size:
                        await self._flush_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in audit batch processor: {e}")

    async def _flush_batch(self):
        """Flush pending events to database"""
        if not self.pending_events:
            return

        events_to_flush = self.pending_events.copy()
        self.pending_events.clear()

        try:
            async with self.db_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO audit_logs (
                        timestamp, event_type, severity, service_name, action,
                        entity_type, entity_id, user_id, session_id, ip_address,
                        user_agent, request_id, changes, old_values, new_values,
                        success, error_message, execution_time_ms, metadata
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                        $11, $12, $13, $14, $15, $16, $17, $18, $19
                    )
                """,
                    [
                        (
                            event.timestamp,
                            event.event_type.value,
                            event.severity.value,
                            event.service_name,
                            event.action,
                            event.entity_type,
                            event.entity_id,
                            event.user_id,
                            event.session_id,
                            event.ip_address,
                            event.user_agent,
                            event.request_id,
                            json.dumps(event.changes) if event.changes else None,
                            json.dumps(event.old_values) if event.old_values else None,
                            json.dumps(event.new_values) if event.new_values else None,
                            event.success,
                            event.error_message,
                            event.execution_time_ms,
                            json.dumps(event.metadata) if event.metadata else None,
                        )
                        for event in events_to_flush
                    ],
                )

            logger.debug(f"Flushed {len(events_to_flush)} audit events to database")

        except Exception as e:
            logger.error(f"Failed to flush audit events: {e}")
            # Put events back for retry
            async with self.batch_lock:
                self.pending_events = events_to_flush + self.pending_events

    async def query_events(
        self,
        start_date: Optional[dt.datetime] = None,
        end_date: Optional[dt.datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        service_names: Optional[List[str]] = None,
        user_ids: Optional[List[str]] = None,
        success_only: Optional[bool] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters"""
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []

        if start_date:
            params.append(start_date)
            query += f" AND timestamp >= ${len(params)}"

        if end_date:
            params.append(end_date)
            query += f" AND timestamp <= ${len(params)}"

        if event_types:
            params.append([et.value for et in event_types])
            query += f" AND event_type = ANY(${len(params)})"

        if service_names:
            params.append(service_names)
            query += f" AND service_name = ANY(${len(params)})"

        if user_ids:
            params.append(user_ids)
            query += f" AND user_id = ANY(${len(params)})"

        if success_only is not None:
            params.append(success_only)
            query += f" AND success = ${len(params)}"

        query += f" ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}"

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [dict(row) for row in rows]

    async def get_recent_events(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit events from Redis cache"""
        try:
            events_data = await cast(Any, self.redis.lrange)(
                "audit:recent", 0, count - 1
            )
            result = []
            for event in events_data:
                event_str = (
                    event.decode("utf-8") if isinstance(event, bytes) else str(event)
                )
                result.append(json.loads(event_str))
            return result
        except Exception as e:
            logger.error(f"Failed to get recent events from Redis: {e}")
            return []


class AuditMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic audit logging"""

    def __init__(self, app, audit_logger: AuditLogger, service_name: str):
        super().__init__(app)
        self.audit_logger = audit_logger
        self.service_name = service_name

    async def dispatch(self, request: Request, call_next):
        """Process request and log audit event"""
        start_time = dt.datetime.now(timezone.utc)
        request_id = f"{self.service_name}_{start_time.strftime('%Y%m%d_%H%M%S_%f')}"

        # Extract request information
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")
        user_id = self._extract_user_id(request)
        session_id = self._extract_session_id(request)

        response = None
        error_message = None
        success = True

        try:
            # Process request
            response = await call_next(request)

            # Check if response indicates an error
            if response.status_code >= 400:
                success = False
                error_message = f"HTTP {response.status_code}"

        except Exception as e:
            success = False
            error_message = str(e)
            # Create error response
            response = Response(
                content=json.dumps({"error": "Internal server error"}),
                status_code=500,
                media_type="application/json",
            )

        # Calculate execution time
        end_time = dt.datetime.now(timezone.utc)
        execution_time_ms = (end_time - start_time).total_seconds() * 1000

        # Determine if this request should be audited
        should_audit = self._should_audit_request(request, response)

        if should_audit:
            # Create audit event
            event = AuditEvent(
                timestamp=start_time,
                event_type=AuditEventType.API_ACCESS,
                severity=AuditSeverity.HIGH if not success else AuditSeverity.LOW,
                service_name=self.service_name,
                action=f"{request.method} {request.url.path}",
                user_id=user_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                success=success,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                metadata={
                    "method": request.method,
                    "path": str(request.url.path),
                    "query_params": dict(request.query_params),
                    "status_code": response.status_code if response else None,
                    "content_length": (
                        response.headers.get("content-length") if response else None
                    ),
                },
            )

            # Log the event
            await self.audit_logger.log_event(event)

        # Add request ID to response headers
        if response:
            response.headers["X-Request-ID"] = request_id

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request (JWT token, API key, etc.)"""
        # Try to extract from Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Extract user ID from JWT token
            user_id = extract_user_id_from_request_header(auth_header)
            if user_id:
                return user_id
            # Fallback to generic api_user if JWT decode fails
            return "api_user"

        # Check for API key in headers
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"api_key_user_{api_key[-8:]}"  # Last 8 chars for identification

        return None

    def _extract_session_id(self, request: Request) -> Optional[str]:
        """Extract session ID from request"""
        # Look for session cookie or header
        session_cookie = request.cookies.get("session_id")
        if session_cookie:
            return session_cookie

        session_header = request.headers.get("x-session-id")
        if session_header:
            return session_header

        return None

    def _should_audit_request(self, request: Request, response: Response) -> bool:
        """Determine if request should be audited"""
        # Always audit non-GET requests
        if request.method != "GET":
            return True

        # Always audit errors
        if response and response.status_code >= 400:
            return True

        # Audit sensitive endpoints
        sensitive_paths = [
            "/trades",
            "/positions",
            "/orders",
            "/risk",
            "/config",
            "/admin",
            "/export",
            "/maintenance",
        ]

        path = request.url.path
        if any(sensitive in path for sensitive in sensitive_paths):
            return True

        # Skip health checks and metrics (too noisy)
        skip_paths = ["/health", "/metrics", "/ping", "/status"]
        if any(skip in path for skip in skip_paths):
            return False

        return False


class TradeAuditor:
    """Specialized auditor for trading operations"""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def log_trade_execution(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy_id: str,
        user_id: str = "system",
        success: bool = True,
        error_message: Optional[str] = None,
        execution_details: Optional[Dict[str, Any]] = None,
    ):
        """Log trade execution event"""
        event = AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=AuditEventType.TRADE_EXECUTION,
            severity=AuditSeverity.HIGH,
            service_name="trade_executor",
            action="execute_trade",
            entity_type="trade",
            entity_id=trade_id,
            user_id=user_id,
            success=success,
            error_message=error_message,
            metadata={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "strategy_id": strategy_id,
                "execution_details": execution_details,
            },
        )

        await self.audit_logger.log_event(event)

    async def log_position_change(
        self,
        symbol: str,
        action: str,  # "open", "close", "modify"
        old_quantity: Optional[float],
        new_quantity: float,
        old_price: Optional[float],
        new_price: float,
        user_id: str = "system",
    ):
        """Log position changes"""
        event = AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=(
                AuditEventType.POSITION_OPEN
                if action == "open"
                else AuditEventType.POSITION_CLOSE
            ),
            severity=AuditSeverity.MEDIUM,
            service_name="trade_executor",
            action=f"position_{action}",
            entity_type="position",
            entity_id=symbol,
            user_id=user_id,
            old_values=(
                {"quantity": old_quantity, "price": old_price}
                if old_quantity is not None
                else None
            ),
            new_values={"quantity": new_quantity, "price": new_price},
            changes={
                "action": action,
                "quantity_change": new_quantity - (old_quantity or 0),
                "price_change": new_price - (old_price or 0) if old_price else 0,
            },
        )

        await self.audit_logger.log_event(event)

    async def log_risk_decision(
        self,
        decision: str,  # "approve", "reject", "modify"
        symbol: str,
        proposed_quantity: float,
        approved_quantity: float,
        risk_score: float,
        risk_factors: Dict[str, Any],
        user_id: str = "system",
    ):
        """Log risk management decisions"""
        event = AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=AuditEventType.RISK_DECISION,
            severity=(
                AuditSeverity.HIGH if decision == "reject" else AuditSeverity.MEDIUM
            ),
            service_name="risk_manager",
            action=f"risk_{decision}",
            entity_type="trade_proposal",
            entity_id=f"{symbol}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id=user_id,
            success=True,
            old_values={"proposed_quantity": proposed_quantity},
            new_values={"approved_quantity": approved_quantity},
            metadata={
                "symbol": symbol,
                "decision": decision,
                "risk_score": risk_score,
                "risk_factors": risk_factors,
                "quantity_reduction": proposed_quantity - approved_quantity,
            },
        )

        await self.audit_logger.log_event(event)


class ConfigAuditor:
    """Specialized auditor for configuration changes"""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def log_config_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        change_reason: str,
        user_id: str,
        service_name: str = "config_manager",
    ):
        """Log configuration changes"""
        event = AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            severity=AuditSeverity.HIGH,
            service_name=service_name,
            action="update_configuration",
            entity_type="configuration",
            entity_id=config_key,
            user_id=user_id,
            old_values={"value": old_value},
            new_values={"value": new_value},
            changes={
                "key": config_key,
                "reason": change_reason,
                "value_changed": old_value != new_value,
            },
        )

        await self.audit_logger.log_event(event)


class SecurityAuditor:
    """Specialized auditor for security events"""

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    async def log_authentication_attempt(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        success: bool,
        method: str = "api_key",
        error_message: Optional[str] = None,
    ):
        """Log authentication attempts"""
        event = AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=AuditEventType.USER_LOGIN,
            severity=AuditSeverity.CRITICAL if not success else AuditSeverity.MEDIUM,
            service_name="authentication",
            action="authenticate",
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            error_message=error_message,
            metadata={"authentication_method": method, "failed_attempt": not success},
        )

        await self.audit_logger.log_event(event)

    async def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: AuditSeverity,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Log general security events"""
        event = AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=AuditEventType.SECURITY_EVENT,
            severity=severity,
            service_name="security",
            action=event_type,
            user_id=user_id,
            ip_address=ip_address,
            metadata={
                "description": description,
                "additional_data": additional_data or {},
            },
        )

        await self.audit_logger.log_event(event)


@asynccontextmanager
async def create_audit_context(db_pool: asyncpg.Pool, redis_client: redis.Redis):
    """Create audit logging context manager"""
    audit_logger = AuditLogger(db_pool, redis_client)
    await audit_logger.initialize()

    try:
        yield audit_logger
    finally:
        await audit_logger.shutdown()


def create_trade_auditor(audit_logger: AuditLogger) -> TradeAuditor:
    """Factory function for trade auditor"""
    return TradeAuditor(audit_logger)


def create_config_auditor(audit_logger: AuditLogger) -> ConfigAuditor:
    """Factory function for config auditor"""
    return ConfigAuditor(audit_logger)


def create_security_auditor(audit_logger: AuditLogger) -> SecurityAuditor:
    """Factory function for security auditor"""
    return SecurityAuditor(audit_logger)


# Decorator for auditing function calls
def audit_operation(
    event_type: AuditEventType,
    severity: AuditSeverity = AuditSeverity.MEDIUM,
    entity_type: Optional[str] = None,
):
    """Decorator to automatically audit function calls"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = dt.datetime.now(timezone.utc)
            success = True
            error_message = None
            result = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_message = str(e)
                raise
            finally:
                # Try to get audit logger from function context
                audit_logger = getattr(func, "_audit_logger", None)
                if audit_logger:
                    execution_time = (
                        dt.datetime.now(timezone.utc) - start_time
                    ).total_seconds() * 1000

                    event = AuditEvent(
                        timestamp=start_time,
                        event_type=event_type,
                        severity=severity,
                        service_name=getattr(func, "_service_name", "unknown"),
                        action=func.__name__,
                        entity_type=entity_type,
                        success=success,
                        error_message=error_message,
                        execution_time_ms=execution_time,
                        metadata={
                            "function": func.__name__,
                            "args_count": len(args),
                            "kwargs_keys": list(kwargs.keys()) if kwargs else [],
                            "result_type": type(result).__name__ if result else None,
                        },
                    )

                    await audit_logger.log_event(event)

        return wrapper

    return decorator


# Utility functions for audit queries
async def get_audit_summary(audit_logger: AuditLogger, days: int = 7) -> Dict[str, Any]:
    """Get audit summary for specified period"""
    end_date = dt.datetime.now(timezone.utc)
    start_date = end_date - TimeDelta(days=days)

    events = await audit_logger.query_events(
        start_date=start_date, end_date=end_date, limit=10000
    )

    # Calculate summary statistics
    total_events = len(events)
    failed_events = len([e for e in events if not e["success"]])
    event_type_counts = {}
    service_counts = {}
    hourly_distribution = {}

    for event in events:
        # Count by event type
        event_type = event["event_type"]
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

        # Count by service
        service = event["service_name"]
        service_counts[service] = service_counts.get(service, 0) + 1

        # Count by hour
        hour = dt.datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).hour
        hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1

    return {
        "period_days": days,
        "total_events": total_events,
        "failed_events": failed_events,
        "success_rate": (
            ((total_events - failed_events) / total_events * 100)
            if total_events > 0
            else 0
        ),
        "event_type_distribution": event_type_counts,
        "service_distribution": service_counts,
        "hourly_distribution": hourly_distribution,
        "most_active_hour": (
            max(hourly_distribution.items(), key=lambda x: x[1])[0]
            if hourly_distribution
            else None
        ),
        "error_rate": (failed_events / total_events * 100) if total_events > 0 else 0,
    }


async def cleanup_old_audit_logs(
    db_pool: asyncpg.Pool, retention_days: int = 2555  # 7 years default
):
    """Clean up old audit logs beyond retention period"""
    cutoff_date = dt.datetime.now(timezone.utc) - TimeDelta(days=retention_days)

    async with db_pool.acquire() as conn:
        # Count records to be deleted
        count_result = await conn.fetchrow(
            "SELECT COUNT(*) FROM audit_logs WHERE timestamp < $1", cutoff_date
        )
        records_to_delete = count_result["count"]

        if records_to_delete > 0:
            logger.info(
                f"Cleaning up {records_to_delete} audit log records older than {retention_days} days"
            )

            # Delete in batches to avoid long locks
            batch_size = 10000
            total_deleted = 0

            while True:
                deleted_count = await conn.fetchval(
                    """
                    WITH batch AS (
                        SELECT id FROM audit_logs
                        WHERE timestamp < $1
                        LIMIT $2
                    )
                    DELETE FROM audit_logs
                    WHERE id IN (SELECT id FROM batch)
                    RETURNING 1
                """,
                    cutoff_date,
                    batch_size,
                )

                if not deleted_count:
                    break

                total_deleted += (
                    len(deleted_count) if isinstance(deleted_count, list) else 1
                )
                logger.debug(
                    f"Deleted batch of audit logs, total so far: {total_deleted}"
                )

            logger.info(f"Audit log cleanup completed: {total_deleted} records deleted")
        else:
            logger.debug("No old audit logs to clean up")


# Export audit data for compliance
async def export_audit_for_compliance(
    audit_logger: AuditLogger,
    start_date: dt.datetime,
    end_date: dt.datetime,
    output_path: str,
) -> str:
    """Export audit data for compliance purposes"""
    events = await audit_logger.query_events(
        start_date=start_date,
        end_date=end_date,
        limit=1000000,  # Large limit for compliance exports
    )

    # Create compliance-friendly format
    compliance_data = {
        "export_metadata": {
            "generated_at": dt.datetime.now(timezone.utc).isoformat(),
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_events": len(events),
            "export_purpose": "regulatory_compliance",
        },
        "audit_events": events,
    }

    # Write to file
    output_file = f"{output_path}/compliance_audit_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"

    with open(output_file, "w") as f:
        json.dump(compliance_data, f, indent=2, default=str)

    logger.info(f"Compliance audit export completed: {output_file}")
    return output_file


# Audit event builders for common operations
class AuditEventBuilder:
    """Builder class for creating consistent audit events"""

    @staticmethod
    def trade_execution(
        trade_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        strategy_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: str = "system",
    ) -> AuditEvent:
        """Build trade execution audit event"""
        return AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=AuditEventType.TRADE_EXECUTION,
            severity=AuditSeverity.CRITICAL if not success else AuditSeverity.HIGH,
            service_name="trade_executor",
            action="execute_trade",
            entity_type="trade",
            entity_id=trade_id,
            user_id=user_id,
            success=success,
            error_message=error_message,
            metadata={
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "strategy_id": strategy_id,
                "dollar_amount": quantity * price,
            },
        )

    @staticmethod
    def system_event(
        action: str,
        service_name: str,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Build system event audit entry"""
        return AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=(
                AuditEventType.SYSTEM_START
                if "start" in action
                else AuditEventType.SYSTEM_STOP
            ),
            severity=AuditSeverity.MEDIUM,
            service_name=service_name,
            action=action,
            entity_type="system",
            user_id="system",
            success=success,
            error_message=error_message,
            metadata=metadata or {},
        )


# Global audit logger instance (to be initialized by each service)
_global_audit_logger: Optional[AuditLogger] = None


def set_global_audit_logger(audit_logger: AuditLogger):
    """Set global audit logger instance"""
    global _global_audit_logger
    _global_audit_logger = audit_logger


def get_global_audit_logger() -> Optional[AuditLogger]:
    """Get global audit logger instance"""
    return _global_audit_logger


async def audit_async(
    event_type: AuditEventType,
    action: str,
    service_name: str,
    severity: AuditSeverity = AuditSeverity.MEDIUM,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    user_id: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convenience function for async audit logging"""
    audit_logger = get_global_audit_logger()
    if audit_logger:
        event = AuditEvent(
            timestamp=dt.datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            service_name=service_name,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            success=success,
            error_message=error_message,
            metadata=metadata,
        )
        await audit_logger.log_event(event)


def audit_sync(
    event_type: AuditEventType,
    action: str,
    service_name: str,
    severity: AuditSeverity = AuditSeverity.MEDIUM,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    user_id: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Convenience function for sync audit logging (creates async task)"""
    asyncio.create_task(
        audit_async(
            event_type,
            action,
            service_name,
            severity,
            entity_type,
            entity_id,
            user_id,
            success,
            error_message,
            metadata,
        )
    )
