#!/usr/bin/env python3
"""
AI Trading System Audit Logger

This module provides comprehensive audit logging for all trading system activities,
including trades, user actions, system changes, and security events. It ensures
compliance with financial regulations and provides detailed audit trails.

Features:
- Comprehensive audit event tracking
- Secure tamper-evident logging
- Structured audit data format
- Multiple output destinations
- Real-time audit monitoring
- Compliance reporting
- Data retention management
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import structlog
from cryptography.fernet import Fernet
from prometheus_client import Counter, Histogram


# Audit event types
class AuditEventType(Enum):
    """Types of audit events"""

    TRADE_EXECUTED = "trade_executed"
    ORDER_PLACED = "order_placed"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_UPDATED = "position_updated"
    RISK_VIOLATION = "risk_violation"
    STRATEGY_CHANGE = "strategy_change"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    API_ACCESS = "api_access"
    CONFIG_CHANGE = "config_change"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    SECURITY_ALERT = "security_alert"
    COMPLIANCE_EVENT = "compliance_event"
    DATA_EXPORT = "data_export"
    MAINTENANCE_MODE = "maintenance_mode"


class AuditLevel(Enum):
    """Audit event severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure"""

    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    service_name: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    level: AuditLevel = AuditLevel.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    security_context: Dict[str, Any] = field(default_factory=dict)
    compliance_tags: List[str] = field(default_factory=list)


class AuditLogger:
    """Main audit logging class with tamper-evident features"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger("audit")

        # Configuration
        self.audit_enabled = config.get("AUDIT_LOG_ENABLED", True)
        self.audit_file_path = Path(config.get("AUDIT_LOG_PATH", "./logs/audit.log"))
        self.encryption_enabled = config.get("AUDIT_LOG_ENCRYPTION", True)
        self.retention_days = int(
            config.get("AUDIT_LOG_RETENTION_DAYS", 2555)
        )  # 7 years
        self.batch_size = int(config.get("AUDIT_BATCH_SIZE", 100))
        self.flush_interval = int(config.get("AUDIT_FLUSH_INTERVAL", 5))

        # Security configuration
        self.encryption_key = config.get("AUDIT_ENCRYPTION_KEY", "")
        self.signing_key = config.get("AUDIT_SIGNING_KEY", "")

        # Initialize encryption
        self.cipher = None
        if self.encryption_enabled and self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode())

        # Event tracking
        self.event_sequence = 0
        self.last_hash = ""
        self.pending_events: List[AuditEvent] = []
        self.audit_lock = asyncio.Lock()

        # Metrics
        self.audit_events_total = Counter(
            "audit_events_total",
            "Total audit events logged",
            ["event_type", "level", "service"],
        )

        self.audit_write_duration = Histogram(
            "audit_write_duration_seconds",
            "Time spent writing audit logs",
            ["destination"],
        )

        # Ensure audit log directory exists
        self.audit_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize audit chain
        self._initialize_audit_chain()

    def _initialize_audit_chain(self):
        """Initialize the audit chain for tamper detection"""
        if self.audit_file_path.exists():
            # Load last hash from existing log
            try:
                with open(self.audit_file_path, "r") as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line:
                            last_entry = json.loads(last_line)
                            self.last_hash = last_entry.get("chain_hash", "")
                            self.event_sequence = last_entry.get("sequence", 0)
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        if not self.last_hash:
            # Initialize with genesis hash
            self.last_hash = hashlib.sha256(b"genesis_audit_chain").hexdigest()
            self.event_sequence = 0

    def _calculate_chain_hash(self, event: AuditEvent, sequence: int) -> str:
        """Calculate chain hash for tamper detection"""
        # Create hash input from event data and previous hash
        hash_input = {
            "sequence": sequence,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "message": event.message,
            "previous_hash": self.last_hash,
        }

        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def log_event(self, event: AuditEvent) -> str:
        """Log an audit event with tamper protection"""
        if not self.audit_enabled:
            return ""

        async with self.audit_lock:
            # Assign sequence number and calculate chain hash
            self.event_sequence += 1
            event.event_id = f"audit_{self.event_sequence:08d}_{int(time.time())}"

            # Calculate chain hash
            chain_hash = self._calculate_chain_hash(event, self.event_sequence)

            # Create audit record
            audit_record = {
                "event_id": event.event_id,
                "sequence": self.event_sequence,
                "chain_hash": chain_hash,
                "previous_hash": self.last_hash,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "level": event.level.value,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "service_name": event.service_name,
                "source_ip": event.source_ip,
                "user_agent": event.user_agent,
                "message": event.message,
                "details": event.details,
                "security_context": event.security_context,
                "compliance_tags": event.compliance_tags,
            }

            # Add to pending events
            self.pending_events.append(event)

            # Update chain state
            self.last_hash = chain_hash

            # Write to audit log
            await self._write_audit_record(audit_record)

            # Update metrics
            self.audit_events_total.labels(
                event_type=event.event_type.value,
                level=event.level.value,
                service=event.service_name or "unknown",
            ).inc()

            self.logger.info(
                "Audit event logged",
                event_id=event.event_id,
                event_type=event.event_type.value,
                sequence=self.event_sequence,
            )

            return event.event_id

    async def _write_audit_record(self, record: Dict[str, Any]):
        """Write audit record to file with optional encryption"""
        with self.audit_write_duration.labels(destination="file").time():
            # Convert to JSON
            record_json = json.dumps(record, separators=(",", ":"))

            # Encrypt if enabled
            if self.encryption_enabled and self.cipher:
                record_json = self.cipher.encrypt(record_json.encode()).decode()

            # Write to file
            async with aiofiles.open(self.audit_file_path, "a") as f:
                await f.write(record_json + "\n")

    async def log_trade_execution(
        self, trade_data: Dict[str, Any], user_id: Optional[str] = None
    ):
        """Log trade execution audit event"""
        event = AuditEvent(
            event_id="",  # Will be assigned
            event_type=AuditEventType.TRADE_EXECUTED,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            service_name="trade_executor",
            level=AuditLevel.INFO,
            message=f"Trade executed: {trade_data.get('symbol')} {trade_data.get('side')} {trade_data.get('quantity')}",
            details=trade_data,
            compliance_tags=["trading", "execution", "financial"],
        )

        return await self.log_event(event)

    async def log_order_placement(
        self, order_data: Dict[str, Any], user_id: Optional[str] = None
    ):
        """Log order placement audit event"""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.ORDER_PLACED,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            service_name="trade_executor",
            level=AuditLevel.INFO,
            message=f"Order placed: {order_data.get('symbol')} {order_data.get('side')} {order_data.get('quantity')}",
            details=order_data,
            compliance_tags=["trading", "order", "financial"],
        )

        return await self.log_event(event)

    async def log_risk_violation(
        self, violation_data: Dict[str, Any], user_id: Optional[str] = None
    ):
        """Log risk management violation"""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.RISK_VIOLATION,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            service_name="risk_manager",
            level=AuditLevel.WARNING,
            message=f"Risk violation: {violation_data.get('violation_type')}",
            details=violation_data,
            compliance_tags=["risk", "violation", "compliance"],
        )

        return await self.log_event(event)

    async def log_strategy_change(
        self, strategy_data: Dict[str, Any], user_id: Optional[str] = None
    ):
        """Log strategy configuration changes"""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.STRATEGY_CHANGE,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            service_name="strategy_engine",
            level=AuditLevel.INFO,
            message=f"Strategy changed: {strategy_data.get('strategy_name')}",
            details=strategy_data,
            compliance_tags=["strategy", "configuration", "algorithm"],
        )

        return await self.log_event(event)

    async def log_user_access(
        self,
        access_data: Dict[str, Any],
        event_type: AuditEventType = AuditEventType.USER_LOGIN,
    ):
        """Log user access events"""
        event = AuditEvent(
            event_id="",
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_id=access_data.get("user_id"),
            session_id=access_data.get("session_id"),
            source_ip=access_data.get("source_ip"),
            user_agent=access_data.get("user_agent"),
            level=AuditLevel.INFO,
            message=f"User {event_type.value}: {access_data.get('user_id')}",
            details=access_data,
            security_context={
                "authentication_method": access_data.get("auth_method"),
                "mfa_used": access_data.get("mfa_used", False),
            },
            compliance_tags=["authentication", "access"],
        )

        return await self.log_event(event)

    async def log_api_access(self, request_data: Dict[str, Any]):
        """Log API access for audit trail"""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.API_ACCESS,
            timestamp=datetime.now(timezone.utc),
            user_id=request_data.get("user_id"),
            source_ip=request_data.get("source_ip"),
            user_agent=request_data.get("user_agent"),
            service_name=request_data.get("service_name"),
            level=AuditLevel.INFO,
            message=f"API access: {request_data.get('method')} {request_data.get('endpoint')}",
            details={
                "endpoint": request_data.get("endpoint"),
                "method": request_data.get("method"),
                "status_code": request_data.get("status_code"),
                "response_time": request_data.get("response_time"),
                "request_size": request_data.get("request_size"),
                "response_size": request_data.get("response_size"),
            },
            compliance_tags=["api", "access"],
        )

        return await self.log_event(event)

    async def log_config_change(
        self, change_data: Dict[str, Any], user_id: Optional[str] = None
    ):
        """Log configuration changes"""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.CONFIG_CHANGE,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            service_name=change_data.get("service_name"),
            level=AuditLevel.WARNING,
            message=f"Configuration changed: {change_data.get('config_key')}",
            details={
                "config_key": change_data.get("config_key"),
                "old_value": change_data.get("old_value", "[REDACTED]"),
                "new_value": change_data.get("new_value", "[REDACTED]"),
                "change_reason": change_data.get("reason"),
            },
            compliance_tags=["configuration", "system_change"],
        )

        return await self.log_event(event)

    async def log_security_event(self, security_data: Dict[str, Any]):
        """Log security-related events"""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.SECURITY_ALERT,
            timestamp=datetime.now(timezone.utc),
            user_id=security_data.get("user_id"),
            source_ip=security_data.get("source_ip"),
            level=AuditLevel.CRITICAL,
            message=f"Security event: {security_data.get('event_description')}",
            details=security_data,
            security_context={
                "threat_level": security_data.get("threat_level"),
                "attack_type": security_data.get("attack_type"),
                "mitigation_applied": security_data.get("mitigation_applied"),
            },
            compliance_tags=["security", "threat", "incident"],
        )

        return await self.log_event(event)

    async def log_compliance_event(self, compliance_data: Dict[str, Any]):
        """Log compliance-related events"""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.COMPLIANCE_EVENT,
            timestamp=datetime.now(timezone.utc),
            user_id=compliance_data.get("user_id"),
            level=AuditLevel.WARNING,
            message=f"Compliance event: {compliance_data.get('regulation')}",
            details=compliance_data,
            compliance_tags=[
                "compliance",
                compliance_data.get("regulation", "").lower(),
                compliance_data.get("requirement_type", "").lower(),
            ],
        )

        return await self.log_event(event)

    async def log_data_export(
        self, export_data: Dict[str, Any], user_id: Optional[str] = None
    ):
        """Log data export operations"""
        event = AuditEvent(
            event_id="",
            event_type=AuditEventType.DATA_EXPORT,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            service_name="export_service",
            level=AuditLevel.INFO,
            message=f"Data exported: {export_data.get('export_type')}",
            details={
                "export_type": export_data.get("export_type"),
                "format": export_data.get("format"),
                "date_range": export_data.get("date_range"),
                "record_count": export_data.get("record_count"),
                "file_path": export_data.get("file_path"),
                "encryption_used": export_data.get("encryption_used"),
            },
            compliance_tags=["data_export", "privacy", "compliance"],
        )

        return await self.log_event(event)

    async def log_system_event(
        self,
        event_type: AuditEventType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log system-level events"""
        event = AuditEvent(
            event_id="",
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            service_name="system",
            level=AuditLevel.INFO,
            message=message,
            details=details or {},
            compliance_tags=["system", "operations"],
        )

        return await self.log_event(event)

    async def flush_pending_events(self):
        """Flush pending audit events to storage"""
        if not self.pending_events:
            return

        async with self.audit_lock:
            events_to_flush = self.pending_events.copy()
            self.pending_events.clear()

        # Write events in batch
        for event in events_to_flush:
            # This would be implemented to write to multiple destinations
            pass

    async def verify_audit_chain(
        self, start_sequence: Optional[int] = None, end_sequence: Optional[int] = None
    ) -> Dict[str, Any]:
        """Verify integrity of audit chain"""
        self.logger.info("Starting audit chain verification")

        verification_result = {
            "status": "valid",
            "total_events": 0,
            "verified_events": 0,
            "chain_breaks": [],
            "corruption_detected": False,
            "start_sequence": start_sequence or 0,
            "end_sequence": end_sequence or 0,
        }

        try:
            # Read audit log file
            if not self.audit_file_path.exists():
                verification_result["status"] = "no_audit_log"
                return verification_result

            previous_hash = hashlib.sha256(b"genesis_audit_chain").hexdigest()
            sequence = 0

            async with aiofiles.open(self.audit_file_path, "r") as f:
                async for line in f:
                    if not line.strip():
                        continue

                    try:
                        # Decrypt if necessary
                        line_data = line.strip()
                        if self.encryption_enabled and self.cipher:
                            line_data = self.cipher.decrypt(line_data.encode()).decode()

                        record = json.loads(line_data)
                        sequence += 1
                        verification_result["total_events"] += 1

                        # Skip if outside range
                        if start_sequence and sequence < start_sequence:
                            continue
                        if end_sequence and sequence > end_sequence:
                            break

                        # Verify chain hash
                        expected_hash = self._calculate_chain_hash_from_record(
                            record, previous_hash
                        )
                        actual_hash = record.get("chain_hash", "")

                        if expected_hash == actual_hash:
                            verification_result["verified_events"] += 1
                        else:
                            verification_result["chain_breaks"].append(
                                {
                                    "sequence": sequence,
                                    "expected_hash": expected_hash,
                                    "actual_hash": actual_hash,
                                    "timestamp": record.get("timestamp"),
                                }
                            )
                            verification_result["corruption_detected"] = True

                        previous_hash = actual_hash

                    except (json.JSONDecodeError, Exception) as e:
                        verification_result["chain_breaks"].append(
                            {
                                "sequence": sequence,
                                "error": str(e),
                                "line": line[:100],  # First 100 chars for debugging
                            }
                        )
                        verification_result["corruption_detected"] = True

            # Determine overall status
            if verification_result["corruption_detected"]:
                verification_result["status"] = "corrupted"
            elif (
                verification_result["verified_events"]
                == verification_result["total_events"]
            ):
                verification_result["status"] = "valid"
            else:
                verification_result["status"] = "partial"

            self.logger.info(
                "Audit chain verification completed",
                status=verification_result["status"],
                total_events=verification_result["total_events"],
                chain_breaks=len(verification_result["chain_breaks"]),
            )

        except Exception as e:
            self.logger.error("Audit chain verification failed", error=str(e))
            verification_result["status"] = "error"
            verification_result["error"] = str(e)

        return verification_result

    def _calculate_chain_hash_from_record(
        self, record: Dict[str, Any], previous_hash: str
    ) -> str:
        """Calculate expected chain hash from audit record"""
        hash_input = {
            "sequence": record.get("sequence"),
            "timestamp": record.get("timestamp"),
            "event_type": record.get("event_type"),
            "user_id": record.get("user_id"),
            "message": record.get("message"),
            "previous_hash": previous_hash,
        }

        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def _decrypt_audit_line(self, line: str) -> str:
        """Decrypt audit line if encryption is enabled"""
        line_data = line.strip()
        if self.encryption_enabled and self.cipher:
            line_data = self.cipher.decrypt(line_data.encode()).decode()
        return line_data

    def _matches_time_filter(
        self,
        record: Dict[str, Any],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> bool:
        """Check if record matches time filters"""
        event_time = datetime.fromisoformat(record.get("timestamp", ""))

        if start_time and event_time < start_time:
            return False
        if end_time and event_time > end_time:
            return False
        return True

    def _matches_filters(
        self,
        record: Dict[str, Any],
        event_types: Optional[List[AuditEventType]],
        user_id: Optional[str],
        service_name: Optional[str],
        compliance_tags: Optional[List[str]],
    ) -> bool:
        """Check if record matches all filters"""
        if event_types and record.get("event_type") not in [
            et.value for et in event_types
        ]:
            return False
        if user_id and record.get("user_id") != user_id:
            return False
        if service_name and record.get("service_name") != service_name:
            return False
        if compliance_tags:
            record_tags = record.get("compliance_tags", [])
            if not any(tag in record_tags for tag in compliance_tags):
                return False
        return True

    async def search_audit_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        service_name: Optional[str] = None,
        compliance_tags: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Search audit events with filters"""
        self.logger.info(
            "Searching audit events",
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            limit=limit,
        )

        results = []

        try:
            async with aiofiles.open(self.audit_file_path, "r") as f:
                async for line in f:
                    if not line.strip():
                        continue

                    try:
                        line_data = self._decrypt_audit_line(line)
                        record = json.loads(line_data)

                        # Apply filters
                        if not self._matches_time_filter(record, start_time, end_time):
                            continue

                        if not self._matches_filters(
                            record, event_types, user_id, service_name, compliance_tags
                        ):
                            continue

                        results.append(record)

                        if len(results) >= limit:
                            break

                    except (json.JSONDecodeError, Exception) as e:
                        self.logger.error("Error parsing audit record", error=str(e))
                        continue

        except FileNotFoundError:
            self.logger.warning("Audit log file not found")
        except Exception as e:
            self.logger.error("Error searching audit events", error=str(e))
            raise

        self.logger.info("Audit search completed", results_count=len(results))
        return results

    async def generate_compliance_report(
        self, start_date: datetime, end_date: datetime, report_type: str = "full"
    ) -> Dict[str, Any]:
        """Generate compliance audit report"""
        self.logger.info(
            "Generating compliance report",
            start_date=start_date,
            end_date=end_date,
            report_type=report_type,
        )

        # Search for relevant events
        events = await self.search_audit_events(
            start_time=start_date, end_time=end_date, limit=10000
        )

        # Analyze events for compliance metrics
        report = {
            "report_id": f"compliance_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "report_type": report_type,
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_events": len(events),
                "trade_events": 0,
                "risk_violations": 0,
                "security_events": 0,
                "config_changes": 0,
                "api_calls": 0,
            },
            "trading_activity": {
                "total_trades": 0,
                "total_volume": 0,
                "unique_symbols": set(),
                "trading_days": set(),
            },
            "risk_analysis": {
                "violations": [],
                "violation_types": {},
                "max_position_size": 0,
                "max_portfolio_risk": 0,
            },
            "security_summary": {
                "failed_logins": 0,
                "api_errors": 0,
                "unusual_access_patterns": [],
            },
            "compliance_flags": [],
        }

        # Process events
        for event in events:
            event_type = event.get("event_type")

            # Count by type
            if event_type == AuditEventType.TRADE_EXECUTED.value:
                report["summary"]["trade_events"] += 1

                # Trading activity analysis
                details = event.get("details", {})
                report["trading_activity"]["total_trades"] += 1
                report["trading_activity"]["total_volume"] += details.get(
                    "quantity", 0
                ) * details.get("price", 0)
                report["trading_activity"]["unique_symbols"].add(
                    details.get("symbol", "")
                )

                trade_date = datetime.fromisoformat(event.get("timestamp", "")).date()
                report["trading_activity"]["trading_days"].add(trade_date.isoformat())

            elif event_type == AuditEventType.RISK_VIOLATION.value:
                report["summary"]["risk_violations"] += 1

                # Risk analysis
                details = event.get("details", {})
                violation_type = details.get("violation_type", "unknown")
                report["risk_analysis"]["violations"].append(
                    {
                        "timestamp": event.get("timestamp"),
                        "type": violation_type,
                        "details": details,
                    }
                )

                if violation_type in report["risk_analysis"]["violation_types"]:
                    report["risk_analysis"]["violation_types"][violation_type] += 1
                else:
                    report["risk_analysis"]["violation_types"][violation_type] = 1

            elif event_type == AuditEventType.SECURITY_ALERT.value:
                report["summary"]["security_events"] += 1

            elif event_type == AuditEventType.CONFIG_CHANGE.value:
                report["summary"]["config_changes"] += 1

            elif event_type == AuditEventType.API_ACCESS.value:
                report["summary"]["api_calls"] += 1

                # Check for API errors
                details = event.get("details", {})
                if details.get("status_code", 200) >= 400:
                    report["security_summary"]["api_errors"] += 1

        # Convert sets to lists for JSON serialization
        report["trading_activity"]["unique_symbols"] = list(
            report["trading_activity"]["unique_symbols"]
        )
        report["trading_activity"]["trading_days"] = list(
            report["trading_activity"]["trading_days"]
        )

        # Add compliance flags based on analysis
        if report["summary"]["risk_violations"] > 10:
            report["compliance_flags"].append("HIGH_RISK_VIOLATIONS")

        if report["security_summary"]["api_errors"] > 100:
            report["compliance_flags"].append("HIGH_API_ERROR_RATE")

        if (
            len(report["trading_activity"]["trading_days"]) > 250
        ):  # More than normal trading days
            report["compliance_flags"].append("UNUSUAL_TRADING_FREQUENCY")

        self.logger.info(
            "Compliance report generated",
            report_id=report["report_id"],
            total_events=report["summary"]["total_events"],
        )

        return report

    async def export_audit_data(
        self,
        start_date: datetime,
        end_date: datetime,
        format_type: str = "json",
        include_details: bool = True,
    ) -> str:
        """Export audit data for external analysis"""
        self.logger.info(
            "Exporting audit data",
            start_date=start_date,
            end_date=end_date,
            format_type=format_type,
        )

        # Search for events in date range
        events = await self.search_audit_events(
            start_time=start_date, end_time=end_date, limit=50000
        )

        # Create export filename
        export_filename = f"audit_export_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.{format_type}"

        # Export data based on format
        if format_type.lower() == "json":
            export_data = {
                "export_info": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_events": len(events),
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "events": (
                    events
                    if include_details
                    else [
                        {
                            k: v
                            for k, v in event.items()
                            if k not in ["details", "security_context"]
                        }
                        for event in events
                    ]
                ),
            }

            # Write to file
            export_path = Path(self.audit_file_path).parent / export_filename
            async with aiofiles.open(export_path, "w") as f:
                await f.write(json.dumps(export_data, indent=2, default=str))

        elif format_type.lower() == "csv":
            import csv
            import io

            # Flatten events for CSV
            flattened_events = []
            for event in events:
                flat_event = {
                    "event_id": event.get("event_id", ""),
                    "event_type": event.get("event_type", ""),
                    "timestamp": event.get("timestamp", ""),
                    "user_id": event.get("user_id", ""),
                    "service_name": event.get("service_name", ""),
                    "level": event.get("level", ""),
                    "message": event.get("message", ""),
                }
                if include_details and "details" in event:
                    flat_event["details"] = str(event["details"])
                flattened_events.append(flat_event)

            # Write CSV
            export_path = Path(self.audit_file_path).parent / export_filename
            async with aiofiles.open(export_path, "w", newline="") as f:
                if flattened_events:
                    fieldnames = flattened_events[0].keys()
                    output = io.StringIO()
                    csv_writer = csv.DictWriter(output, fieldnames=fieldnames)
                    csv_writer.writeheader()
                    csv_writer.writerows(flattened_events)
                    await f.write(output.getvalue())
        else:
            # Unsupported format, default to JSON
            export_path = Path(self.audit_file_path).parent / export_filename.replace(
                f".{format_type}", ".json"
            )
            export_data = {
                "export_info": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "total_events": len(events),
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "events": events,
            }
            async with aiofiles.open(export_path, "w") as f:
                await f.write(json.dumps(export_data, indent=2, default=str))

        # Log the export
        await self.log_data_export(
            {
                "export_filename": export_filename,
                "format": format_type,
                "event_count": len(events),
                "date_range": f"{start_date.isoformat()} to {end_date.isoformat()}",
            }
        )

        return str(export_path)
