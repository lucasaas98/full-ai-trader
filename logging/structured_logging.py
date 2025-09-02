"""
Comprehensive structured logging system for the AI Trading Platform.

This module provides structured logging capabilities with JSON formatting,
log aggregation, search functionality, and audit trail management.
"""

import asyncio
import gzip
import json
import logging
import logging.handlers
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional

import aiofiles
import numpy as np


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"


class LogCategory(str, Enum):
    """Log category enumeration."""

    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    DATA = "data"
    API = "api"
    USER = "user"


@dataclass
class LogContext:
    """Structured log context information."""

    service_name: str
    instance_id: str
    environment: str
    version: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class TradeLogEntry:
    """Structured log entry for trading operations."""

    timestamp: datetime
    level: LogLevel
    category: LogCategory
    service: str
    message: str
    context: LogContext
    data: Optional[Dict[str, Any]] = None
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    duration_ms: Optional[float] = None
    tags: Optional[List[str]] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        log_dict = {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "service": self.service,
            "message": self.message,
            "context": self.context.to_dict(),
        }

        if self.data:
            log_dict["data"] = self.data
        if self.exception:
            log_dict["exception"] = self.exception
        if self.stack_trace:
            log_dict["stack_trace"] = self.stack_trace
        if self.duration_ms is not None:
            log_dict["duration_ms"] = self.duration_ms
        if self.tags:
            log_dict["tags"] = self.tags

        return json.dumps(log_dict, default=str, ensure_ascii=False)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def __init__(self, context: LogContext):
        super().__init__()
        self.context = context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Extract additional data from record
        data = getattr(record, "data", None)
        tags = getattr(record, "tags", None)
        duration_ms = getattr(record, "duration_ms", None)
        category = getattr(record, "category", LogCategory.SYSTEM)

        # Handle exceptions
        exception_info = None
        stack_trace = None
        if record.exc_info:
            exception_info = str(record.exc_info[1])
            stack_trace = traceback.format_exception(*record.exc_info)
            stack_trace = "".join(stack_trace)

        # Create structured log entry
        log_entry = TradeLogEntry(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc),
            level=LogLevel(record.levelname),
            category=category,
            service=self.context.service_name,
            message=record.getMessage(),
            context=self.context,
            data=data,
            exception=exception_info,
            stack_trace=stack_trace,
            duration_ms=duration_ms,
            tags=tags,
        )

        return log_entry.to_json()


class AsyncFileHandler(logging.Handler):
    """Async file handler for high-performance logging."""

    def __init__(
        self, filename: str, max_bytes: int = 100 * 1024 * 1024, backup_count: int = 10
    ):
        super().__init__()
        self.filename = filename
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.current_size = 0
        self.log_queue = Queue()
        self.stop_event = threading.Event()
        self.writer_thread = None

        # Ensure log directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Start writer thread
        self._start_writer_thread()

    def _start_writer_thread(self):
        """Start the async writer thread."""
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _writer_loop(self):
        """Main writer loop running in separate thread."""
        while not self.stop_event.is_set():
            try:
                # Get log entry with timeout
                try:
                    log_entry = self.log_queue.get(timeout=1.0)
                except:
                    continue

                # Write to file
                self._write_to_file(log_entry)
                self.log_queue.task_done()

            except Exception as e:
                print(f"Error in log writer thread: {e}", file=sys.stderr)

    def _write_to_file(self, log_entry: str):
        """Write log entry to file with rotation."""
        # Check if rotation is needed
        if self.current_size > self.max_bytes:
            self._rotate_files()

        # Write log entry
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
            self.current_size += len(log_entry.encode("utf-8")) + 1

    def _rotate_files(self):
        """Rotate log files."""
        if not os.path.exists(self.filename):
            return

        # Compress and move old files
        for i in range(self.backup_count - 1, 0, -1):
            old_file = f"{self.filename}.{i}.gz"
            new_file = f"{self.filename}.{i + 1}.gz"

            if os.path.exists(old_file):
                if i == self.backup_count - 1:
                    os.remove(old_file)
                else:
                    shutil.move(old_file, new_file)

        # Compress current file
        compressed_file = f"{self.filename}.1.gz"
        with open(self.filename, "rb") as f_in:
            with gzip.open(compressed_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Start new file
        os.remove(self.filename)
        self.current_size = 0

    def emit(self, record: logging.LogRecord):
        """Emit log record to queue."""
        try:
            formatted_record = self.format(record)
            self.log_queue.put(formatted_record)
        except Exception:
            self.handleError(record)

    def close(self):
        """Close handler and stop writer thread."""
        self.stop_event.set()
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5.0)
        super().close()


class ElasticsearchHandler(logging.Handler):
    """Handler for sending logs to Elasticsearch."""

    def __init__(self, es_host: str, es_port: int, index_prefix: str = "trading-logs"):
        super().__init__()
        self.es_host = es_host
        self.es_port = es_port
        self.index_prefix = index_prefix
        self.es_client = None
        self.buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        self.last_flush = time.time()

    async def _get_es_client(self):
        """Get Elasticsearch client."""
        if not self.es_client:
            try:
                try:
                    from elasticsearch import AsyncElasticsearch
                except ImportError:
                    raise ImportError(
                        "elasticsearch package required for ElasticsearchHandler"
                    )
                self.es_client = AsyncElasticsearch(
                    [{"host": self.es_host, "port": self.es_port}]
                )
            except ImportError:
                raise ImportError(
                    "elasticsearch package required for ElasticsearchHandler"
                )

        return self.es_client

    def emit(self, record: logging.LogRecord):
        """Emit log record to Elasticsearch buffer."""
        try:
            formatted_record = json.loads(self.format(record))

            # Add to buffer
            self.buffer.append(
                {"_index": self._get_index_name(), "_source": formatted_record}
            )

            # Flush if buffer is full or time interval reached
            if (
                len(self.buffer) >= self.buffer_size
                or time.time() - self.last_flush > self.flush_interval
            ):
                asyncio.create_task(self._flush_buffer())

        except Exception:
            self.handleError(record)

    def _get_index_name(self) -> str:
        """Get index name with date suffix."""
        date_suffix = datetime.now().strftime("%Y-%m-%d")
        return f"{self.index_prefix}-{date_suffix}"

    async def _flush_buffer(self):
        """Flush buffer to Elasticsearch."""
        if not self.buffer:
            return

        try:
            es_client = await self._get_es_client()

            # Bulk index operation
            await es_client.bulk(operations=self.buffer, refresh=False)

            self.buffer.clear()
            self.last_flush = time.time()

        except Exception as e:
            print(f"Failed to flush logs to Elasticsearch: {e}", file=sys.stderr)


class StructuredLogger:
    """Main structured logger class."""

    def __init__(
        self,
        service_name: str,
        log_level: LogLevel = LogLevel.INFO,
        log_file: Optional[str] = None,
        elasticsearch_config: Optional[Dict[str, Any]] = None,
        console_output: bool = True,
    ):

        self.service_name = service_name
        self.instance_id = str(uuid.uuid4())[:8]
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.version = os.getenv("APP_VERSION", "1.0.0")

        # Create context
        self.context = LogContext(
            service_name=service_name,
            instance_id=self.instance_id,
            environment=self.environment,
            version=self.version,
        )

        # Setup logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, log_level.value))
        self.logger.handlers.clear()  # Clear existing handlers

        # Create formatter
        self.formatter = JSONFormatter(self.context)

        # Setup handlers
        self._setup_handlers(log_file, elasticsearch_config, console_output)

        # Performance tracking
        self.performance_stats = {
            "logs_written": 0,
            "errors_logged": 0,
            "warnings_logged": 0,
            "start_time": datetime.now(timezone.utc),
        }

    def _setup_handlers(
        self,
        log_file: Optional[str],
        elasticsearch_config: Optional[Dict[str, Any]],
        console_output: bool,
    ):
        """Setup logging handlers."""

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = AsyncFileHandler(log_file)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        # Elasticsearch handler
        if elasticsearch_config:
            es_handler = ElasticsearchHandler(
                es_host=elasticsearch_config["host"],
                es_port=elasticsearch_config["port"],
                index_prefix=elasticsearch_config.get("index_prefix", "trading-logs"),
            )
            es_handler.setFormatter(self.formatter)
            self.logger.addHandler(es_handler)

    def set_context(self, **kwargs):
        """Update logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

    @contextmanager
    def operation_context(self, operation_name: str, **context_data):
        """Context manager for operation-specific logging."""
        operation_id = str(uuid.uuid4())[:8]
        original_context = dict(self.context.__dict__)
        start_time = time.time()

        try:
            # Set operation context
            self.context.correlation_id = operation_id
            for key, value in context_data.items():
                if hasattr(self.context, key):
                    setattr(self.context, key, value)

            self.info(
                f"Starting operation: {operation_name}",
                data={"operation_id": operation_id, "operation": operation_name},
            )

            yield operation_id

            duration_ms = (time.time() - start_time) * 1000
            self.info(
                f"Completed operation: {operation_name}",
                data={"operation_id": operation_id, "duration_ms": duration_ms},
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.error(
                f"Failed operation: {operation_name}",
                data={"operation_id": operation_id, "duration_ms": duration_ms},
                exception=e,
            )
            raise

        finally:
            # Restore original context
            for key, value in original_context.items():
                setattr(self.context, key, value)

    def debug(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        category: LogCategory = LogCategory.SYSTEM,
        tags: Optional[List[str]] = None,
    ):
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, data, category, tags)

    def info(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        category: LogCategory = LogCategory.SYSTEM,
        tags: Optional[List[str]] = None,
    ):
        """Log info message."""
        self._log(LogLevel.INFO, message, data, category, tags)

    def warning(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        category: LogCategory = LogCategory.SYSTEM,
        tags: Optional[List[str]] = None,
    ):
        """Log warning message."""
        self._log(LogLevel.WARNING, message, data, category, tags)
        self.performance_stats["warnings_logged"] += 1

    def error(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        category: LogCategory = LogCategory.SYSTEM,
        tags: Optional[List[str]] = None,
        exception: Optional[Exception] = None,
    ):
        """Log error message."""
        extra_data = {"data": data, "category": category, "tags": tags}

        if exception:
            extra_data["exc_info"] = (
                type(exception),
                exception,
                exception.__traceback__,
            )

        self.logger.error(message, extra=extra_data)
        self.performance_stats["errors_logged"] += 1

    def critical(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        category: LogCategory = LogCategory.SYSTEM,
        tags: Optional[List[str]] = None,
        exception: Optional[Exception] = None,
    ):
        """Log critical message."""
        extra_data = {"data": data, "category": category, "tags": tags}

        if exception:
            extra_data["exc_info"] = (
                type(exception),
                exception,
                exception.__traceback__,
            )

        self.logger.critical(message, extra=extra_data)

    def audit(
        self,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """Log audit message."""
        # Temporarily update context for audit
        original_user = self.context.user_id
        original_session = self.context.session_id

        if user_id:
            self.context.user_id = user_id
        if session_id:
            self.context.session_id = session_id

        try:
            self._log(LogLevel.AUDIT, message, data, LogCategory.AUDIT, tags)
        finally:
            # Restore context
            self.context.user_id = original_user
            self.context.session_id = original_session

    def _log(
        self,
        level: LogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        category: LogCategory = LogCategory.SYSTEM,
        tags: Optional[List[str]] = None,
    ):
        """Internal logging method."""
        extra_data = {"data": data, "category": category, "tags": tags}

        getattr(self.logger, level.value.lower())(message, extra=extra_data)
        self.performance_stats["logs_written"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get logging performance statistics."""
        uptime = datetime.now(timezone.utc) - self.performance_stats["start_time"]

        return {
            "service_name": self.service_name,
            "instance_id": self.instance_id,
            "uptime_seconds": uptime.total_seconds(),
            "logs_written": self.performance_stats["logs_written"],
            "errors_logged": self.performance_stats["errors_logged"],
            "warnings_logged": self.performance_stats["warnings_logged"],
            "logs_per_minute": self.performance_stats["logs_written"]
            / (uptime.total_seconds() / 60),
            "handler_count": len(self.logger.handlers),
        }


class TradingLogger(StructuredLogger):
    """Specialized logger for trading operations."""

    def trade_executed(self, trade_data: Dict[str, Any]):
        """Log trade execution."""
        self.audit("Trade executed", data=trade_data, tags=["trade", "execution"])

    def signal_generated(self, signal_data: Dict[str, Any]):
        """Log signal generation."""
        self.info(
            "Trading signal generated",
            data=signal_data,
            category=LogCategory.TRADING,
            tags=["signal", "strategy"],
        )

    def risk_check_failed(self, risk_data: Dict[str, Any]):
        """Log risk check failure."""
        self.warning(
            "Risk check failed",
            data=risk_data,
            category=LogCategory.RISK,
            tags=["risk", "rejection"],
        )

    def portfolio_updated(self, portfolio_data: Dict[str, Any]):
        """Log portfolio update."""
        self.info(
            "Portfolio updated",
            data=portfolio_data,
            category=LogCategory.TRADING,
            tags=["portfolio", "update"],
        )

    def market_data_received(self, market_data: Dict[str, Any]):
        """Log market data reception."""
        self.info(
            "Market data received", data=market_data, tags=["market-data", "reception"]
        )

    def api_request(self, api_data: Dict[str, Any], duration_ms: float):
        """Log API request."""
        self.info(
            "API request completed",
            data=api_data,
            category=LogCategory.API,
            tags=["api", "external"],
        )

        # Add duration to record
        if self.logger.handlers:
            record = self.logger.makeRecord(
                self.logger.name,
                logging.INFO,
                __file__,
                0,
                "API request completed",
                (),
                None,
            )
            record.duration_ms = duration_ms

    def security_event(self, security_data: Dict[str, Any]):
        """Log security event."""
        self.warning(
            "Security event detected",
            data=security_data,
            category=LogCategory.SECURITY,
            tags=["security", "alert"],
        )

    def performance_metric(self, metric_data: Dict[str, Any]):
        """Log performance metric."""
        self.info(
            "Performance metric", data=metric_data, tags=["performance", "metric"]
        )


class LogAggregator:
    """Log aggregation and search functionality."""

    def __init__(self, log_directory: str):
        self.log_directory = Path(log_directory)
        self.log_files = []
        self._scan_log_files()

    def _scan_log_files(self):
        """Scan for available log files."""
        self.log_files = []

        if not self.log_directory.exists():
            return

        # Find all log files
        for file_path in self.log_directory.glob("**/*.log"):
            self.log_files.append(file_path)

        # Find compressed log files
        for file_path in self.log_directory.glob("**/*.log.*.gz"):
            self.log_files.append(file_path)

    async def search_logs(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        service: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search logs with filters."""
        results = []

        for log_file in self.log_files:
            file_results = await self._search_file(
                log_file, query, start_time, end_time, level, category, service
            )
            results.extend(file_results)

            if len(results) >= limit:
                break

        # Sort by timestamp and limit
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results[:limit]

    async def _search_file(
        self,
        file_path: Path,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        service: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search individual log file."""
        results = []

        try:
            # Handle compressed files
            if file_path.suffix == ".gz":
                mode = "rt"
            else:
                mode = "r"

            async with aiofiles.open(file_path, mode=mode, encoding="utf-8") as f:
                async for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # Apply filters
                        if not self._matches_filters(
                            log_entry, start_time, end_time, level, category, service
                        ):
                            continue

                        # Apply query filter
                        if query and not self._matches_query(log_entry, query):
                            continue

                        results.append(log_entry)

                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines

        except Exception as e:
            print(f"Error searching log file {file_path}: {e}", file=sys.stderr)

        return results

    def _matches_filters(
        self,
        log_entry: Dict[str, Any],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        category: Optional[LogCategory] = None,
        service: Optional[str] = None,
    ) -> bool:
        """Check if log entry matches filters."""

        # Time filter
        if start_time or end_time:
            try:
                entry_time = datetime.fromisoformat(
                    log_entry["timestamp"].replace("Z", "+00:00")
                )
                if start_time and entry_time < start_time:
                    return False
                if end_time and entry_time > end_time:
                    return False
            except (KeyError, ValueError):
                return False

        # Level filter
        if level and log_entry.get("level") != level.value:
            return False

        # Category filter
        if category and log_entry.get("category") != category.value:
            return False

        # Service filter
        if service and log_entry.get("service") != service:
            return False

        return True

    def _matches_query(self, log_entry: Dict[str, Any], query: str) -> bool:
        """Check if log entry matches search query."""
        query_lower = query.lower()

        # Search in message
        if query_lower in log_entry.get("message", "").lower():
            return True

        # Search in data fields
        data = log_entry.get("data", {})
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, str) and query_lower in value.lower():
                    return True

        # Search in tags
        tags = log_entry.get("tags", [])
        if any(query_lower in tag.lower() for tag in tags):
            return True

        return False

    async def get_log_stats(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get aggregated log statistics."""
        stats = {
            "total_entries": 0,
            "by_level": {},
            "by_category": {},
            "by_service": {},
            "error_rate": 0.0,
            "most_common_errors": [],
            "performance_summary": {},
        }

        for log_file in self.log_files:
            file_stats = await self._analyze_file_stats(log_file, start_time, end_time)

            # Aggregate stats
            stats["total_entries"] += file_stats["total_entries"]

            for level, count in file_stats["by_level"].items():
                stats["by_level"][level] = stats["by_level"].get(level, 0) + count

            for category, count in file_stats["by_category"].items():
                stats["by_category"][category] = (
                    stats["by_category"].get(category, 0) + count
                )

            for service, count in file_stats["by_service"].items():
                stats["by_service"][service] = (
                    stats["by_service"].get(service, 0) + count
                )

        # Calculate error rate
        total_errors = stats["by_level"].get("ERROR", 0) + stats["by_level"].get(
            "CRITICAL", 0
        )
        if stats["total_entries"] > 0:
            stats["error_rate"] = total_errors / stats["total_entries"]

        return stats

    async def _analyze_file_stats(
        self,
        file_path: Path,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Analyze statistics for individual log file."""
        stats = {
            "total_entries": 0,
            "by_level": {},
            "by_category": {},
            "by_service": {},
        }

        try:
            if file_path.suffix == ".gz":
                mode = "rt"
            else:
                mode = "r"

            async with aiofiles.open(file_path, mode=mode, encoding="utf-8") as f:
                async for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # Apply time filter
                        if not self._matches_filters(log_entry, start_time, end_time):
                            continue

                        stats["total_entries"] += 1

                        # Count by level
                        level = log_entry.get("level", "UNKNOWN")
                        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

                        # Count by category
                        category = log_entry.get("category", "unknown")
                        stats["by_category"][category] = (
                            stats["by_category"].get(category, 0) + 1
                        )

                        # Count by service
                        service = log_entry.get("service", "unknown")
                        stats["by_service"][service] = (
                            stats["by_service"].get(service, 0) + 1
                        )

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"Error analyzing log file {file_path}: {e}", file=sys.stderr)

        return stats


class AuditTrailManager:
    """Audit trail management for compliance and security."""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger
        self.audit_events = []
        self.sensitive_fields = {
            "password",
            "api_key",
            "secret",
            "token",
            "private_key",
            "ssn",
            "credit_card",
            "bank_account",
        }

    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Log user action for audit trail."""
        audit_data = {
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "result": result,
            "ip_address": ip_address,
            "session_id": session_id,
            "details": self._sanitize_data(details) if details else None,
        }

        self.logger.audit(
            f"User action: {action} on {resource}",
            data=audit_data,
            user_id=user_id,
            session_id=session_id,
            tags=["audit", "user_action"],
        )

    def log_system_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        severity: LogLevel = LogLevel.INFO,
    ):
        """Log system event for audit trail."""
        audit_data = {
            "event_type": event_type,
            "event_data": self._sanitize_data(event_data),
            "system_user": "automated_system",
        }

        self.logger.audit(
            f"System event: {event_type}",
            data=audit_data,
            tags=["audit", "system_event"],
        )

    def log_trade_event(
        self,
        trade_id: str,
        event_type: str,
        trade_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ):
        """Log trade-related event for audit trail."""
        audit_data = {
            "trade_id": trade_id,
            "event_type": event_type,
            "trade_data": self._sanitize_data(trade_data),
            "automated": user_id is None,
        }

        self.logger.audit(
            f"Trade event: {event_type} for trade {trade_id}",
            data=audit_data,
            user_id=user_id or "system",
            tags=["audit", "trade", event_type],
        )

    def log_configuration_change(
        self,
        config_section: str,
        old_values: Dict[str, Any],
        new_values: Dict[str, Any],
        user_id: str,
        session_id: str,
    ):
        """Log configuration changes for audit trail."""
        audit_data = {
            "config_section": config_section,
            "old_values": self._sanitize_data(old_values),
            "new_values": self._sanitize_data(new_values),
            "change_summary": self._generate_change_summary(old_values, new_values),
        }

        self.logger.audit(
            f"Configuration changed: {config_section}",
            data=audit_data,
            user_id=user_id,
            session_id=session_id,
            tags=["audit", "configuration", "change"],
        )

    def _sanitize_data(self, data: Any) -> Any:
        """Sanitize sensitive data from logs."""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, str):
            # Basic pattern matching for sensitive data
            if any(sensitive in data.lower() for sensitive in self.sensitive_fields):
                return "***REDACTED***"

        return data

    def _generate_change_summary(
        self, old_values: Dict[str, Any], new_values: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable change summary."""
        changes = []

        all_keys = set(old_values.keys()) | set(new_values.keys())

        for key in all_keys:
            old_val = old_values.get(key, "<not set>")
            new_val = new_values.get(key, "<removed>")

            if old_val != new_val:
                changes.append(f"{key}: {old_val} → {new_val}")

        return changes

    def get_audit_summary(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get audit trail summary for time period."""
        return {
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "total_events": len(self.audit_events),
            "events_by_type": self._count_events_by_type(),
            "users_active": self._count_unique_users(),
            "most_active_user": self._get_most_active_user(),
            "security_events": self._count_security_events(),
        }

    def _count_events_by_type(self) -> Dict[str, int]:
        """Count audit events by type."""
        counts = {}
        for event in self.audit_events:
            event_type = event.get("event_type", "unknown")
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def _count_unique_users(self) -> int:
        """Count unique users in audit trail."""
        users = set()
        for event in self.audit_events:
            user_id = event.get("user_id")
            if user_id and user_id != "system":
                users.add(user_id)
        return len(users)

    def _get_most_active_user(self) -> Optional[str]:
        """Get most active user from audit trail."""
        user_counts = {}
        for event in self.audit_events:
            user_id = event.get("user_id")
            if user_id and user_id != "system":
                user_counts[user_id] = user_counts.get(user_id, 0) + 1

        if user_counts:
            return max(user_counts, key=lambda user: user_counts[user])
        return None

    def _count_security_events(self) -> int:
        """Count security-related events."""
        return sum(
            1 for event in self.audit_events if "security" in event.get("tags", [])
        )


class LogRetentionManager:
    """Manage log file retention and archival."""

    def __init__(
        self,
        log_directory: str,
        retention_days: int = 90,
        archive_days: int = 30,
        compression_enabled: bool = True,
    ):
        self.log_directory = Path(log_directory)
        self.retention_days = retention_days
        self.archive_days = archive_days
        self.compression_enabled = compression_enabled

    async def cleanup_old_logs(self):
        """Clean up old log files based on retention policy."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        archive_date = datetime.now() - timedelta(days=self.archive_days)

        for log_file in self.log_directory.glob("**/*.log*"):
            try:
                file_age = datetime.fromtimestamp(log_file.stat().st_mtime)

                if file_age < cutoff_date:
                    # Delete very old files
                    log_file.unlink()
                elif file_age < archive_date and self.compression_enabled:
                    # Compress old files
                    if not str(log_file).endswith(".gz"):
                        await self._compress_file(log_file)

            except Exception as e:
                print(f"Error processing log file {log_file}: {e}", file=sys.stderr)

    async def _compress_file(self, file_path: Path):
        """Compress log file."""
        compressed_path = file_path.with_suffix(file_path.suffix + ".gz")

        try:
            with open(file_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove original file after successful compression
            file_path.unlink()

        except Exception as e:
            print(f"Error compressing file {file_path}: {e}", file=sys.stderr)

    def get_retention_stats(self) -> Dict[str, Any]:
        """Get log retention statistics."""
        stats = {
            "total_files": 0,
            "total_size_mb": 0,
            "compressed_files": 0,
            "uncompressed_files": 0,
            "oldest_file": None,
            "newest_file": None,
        }

        oldest_time = None
        newest_time = None

        for log_file in self.log_directory.glob("**/*.log*"):
            try:
                file_stat = log_file.stat()
                stats["total_files"] += 1
                stats["total_size_mb"] += file_stat.st_size / (1024 * 1024)

                if str(log_file).endswith(".gz"):
                    stats["compressed_files"] += 1
                else:
                    stats["uncompressed_files"] += 1

                file_time = datetime.fromtimestamp(file_stat.st_mtime)
                if oldest_time is None or file_time < oldest_time:
                    oldest_time = file_time
                    stats["oldest_file"] = str(log_file)

                if newest_time is None or file_time > newest_time:
                    newest_time = file_time
                    stats["newest_file"] = str(log_file)

            except Exception:
                continue

        return stats


class PerformanceLoggingMixin:
    """Mixin for performance-aware logging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_tracking = {}

    @contextmanager
    def log_performance(self, operation_name: str, threshold_ms: float = 1000.0):
        """Context manager for performance logging."""
        start_time = time.time()
        operation_id = str(uuid.uuid4())[:8]

        try:
            yield operation_id
        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Log if above threshold or if tracking is enabled
            if (
                duration_ms > threshold_ms
                or operation_name in self.performance_tracking
            ):
                level_str = "warning" if duration_ms > threshold_ms else "info"

                # Try different ways to log performance data
                log_data = {
                    "operation_id": operation_id,
                    "operation_name": operation_name,
                    "duration_ms": duration_ms,
                    "threshold_ms": threshold_ms,
                    "above_threshold": duration_ms > threshold_ms,
                }

                # Check if we have structured logging methods
                if hasattr(self, "_log") and callable(getattr(self, "_log")):  # type: ignore
                    level = (
                        LogLevel.WARNING
                        if duration_ms > threshold_ms
                        else LogLevel.INFO
                    )
                    self._log(  # type: ignore
                        level,
                        f"Performance: {operation_name}",
                        data=log_data,
                        category=LogCategory.PERFORMANCE,
                        tags=["performance"],
                    )
                elif hasattr(self, level_str):
                    # Use standard logging methods
                    log_method = getattr(self, level_str)
                    log_method(f"Performance: {operation_name}", data=log_data)
                elif hasattr(self, "logger") and self.logger is not None:  # type: ignore
                    # Use logger attribute if available
                    logger_method = getattr(self.logger, level_str, None)  # type: ignore
                    if logger_method and callable(logger_method):
                        logger_method(
                            f"Performance: {operation_name} - {duration_ms:.2f}ms"
                        )
                else:
                    # Fallback to print
                    print(
                        f"Performance: {operation_name} - {duration_ms:.2f}ms (threshold: {threshold_ms}ms)"
                    )

            # Update performance tracking
            if not hasattr(self, "performance_tracking"):
                self.performance_tracking = {}

            if operation_name not in self.performance_tracking:
                self.performance_tracking[operation_name] = []

            self.performance_tracking[operation_name].append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": duration_ms,
                    "operation_id": operation_id,
                }
            )

            # Keep only recent measurements
            if len(self.performance_tracking[operation_name]) > 100:
                self.performance_tracking[operation_name] = self.performance_tracking[
                    operation_name
                ][-100:]


class LoggingConfig:
    """Centralized logging configuration."""

    @staticmethod
    def create_trading_logger(
        service_name: str,
        log_level: LogLevel = LogLevel.INFO,
        enable_file_logging: bool = True,
        enable_elasticsearch: bool = False,
        console_output: bool = True,
    ) -> TradingLogger:
        """Create configured trading logger."""

        # Determine log file path
        log_file = None
        if enable_file_logging:
            log_dir = os.getenv("LOG_DIRECTORY", "/app/logs")
            log_file = f"{log_dir}/{service_name}.log"

        # Elasticsearch configuration
        elasticsearch_config = None
        if enable_elasticsearch:
            elasticsearch_config = {
                "host": os.getenv("ELASTICSEARCH_HOST", "localhost"),
                "port": int(os.getenv("ELASTICSEARCH_PORT", "9200")),
                "index_prefix": f"trading-{service_name}",
            }

        return TradingLogger(
            service_name=service_name,
            log_level=log_level,
            log_file=log_file,
            elasticsearch_config=elasticsearch_config,
            console_output=console_output,
        )

    @staticmethod
    def setup_service_logging(service_name: str) -> TradingLogger:
        """Setup logging for a trading service."""
        # Get configuration from environment
        log_level_str = os.getenv("LOG_LEVEL", "INFO")
        log_level = LogLevel(log_level_str.upper())

        enable_file_logging = os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
        enable_elasticsearch = (
            os.getenv("ENABLE_ELASTICSEARCH_LOGGING", "false").lower() == "true"
        )
        console_output = os.getenv("CONSOLE_LOGGING", "true").lower() == "true"

        logger = LoggingConfig.create_trading_logger(
            service_name=service_name,
            log_level=log_level,
            enable_file_logging=enable_file_logging,
            enable_elasticsearch=enable_elasticsearch,
            console_output=console_output,
        )

        # Set service-specific context
        logger.set_context(session_id=str(uuid.uuid4()), request_id=str(uuid.uuid4()))

        return logger


class LogSearchEngine:
    """Advanced log search and analysis engine."""

    def __init__(self, aggregator: LogAggregator):
        self.aggregator = aggregator
        self.search_cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def search_with_aggregation(
        self,
        query: str,
        time_range: Dict[str, datetime],
        group_by: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Advanced search with aggregation capabilities."""

        # Check cache first
        cache_key = self._generate_cache_key(query, time_range, group_by, filters)
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["results"]

        # Perform search
        results = await self.aggregator.search_logs(
            query=query,
            start_time=time_range.get("start"),
            end_time=time_range.get("end"),
            level=filters.get("level") if filters else None,
            category=filters.get("category") if filters else None,
            service=filters.get("service") if filters else None,
            limit=filters.get("limit", 1000) if filters else 1000,
        )

        # Apply aggregation
        aggregated_results = {"total_matches": len(results), "results": results}

        if group_by:
            aggregated_results["aggregations"] = self._aggregate_results(
                results, group_by
            )

        # Cache results
        self.search_cache[cache_key] = {
            "results": aggregated_results,
            "timestamp": time.time(),
        }

        return aggregated_results

    def _generate_cache_key(
        self,
        query: str,
        time_range: Dict[str, datetime],
        group_by: Optional[str],
        filters: Optional[Dict[str, Any]],
    ) -> str:
        """Generate cache key for search."""
        key_data = {
            "query": query,
            "time_range": {
                k: v.isoformat() if v else None for k, v in time_range.items()
            },
            "group_by": group_by,
            "filters": filters or {},
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))

    def _aggregate_results(
        self, results: List[Dict[str, Any]], group_by: str
    ) -> Dict[str, Any]:
        """Aggregate search results by specified field."""
        aggregations = {}

        for result in results:
            group_value = result.get(group_by, "unknown")
            if group_value not in aggregations:
                aggregations[group_value] = {
                    "count": 0,
                    "first_occurrence": result.get("timestamp"),
                    "last_occurrence": result.get("timestamp"),
                    "levels": {},
                }

            agg = aggregations[group_value]
            agg["count"] += 1

            # Update time range
            timestamp = result.get("timestamp")
            if timestamp:
                if not agg["first_occurrence"] or timestamp < agg["first_occurrence"]:
                    agg["first_occurrence"] = timestamp
                if not agg["last_occurrence"] or timestamp > agg["last_occurrence"]:
                    agg["last_occurrence"] = timestamp

            # Count by level
            level = result.get("level", "UNKNOWN")
            agg["levels"][level] = agg["levels"].get(level, 0) + 1

        return aggregations

    async def analyze_error_patterns(
        self, time_range: Dict[str, datetime]
    ) -> Dict[str, Any]:
        """Analyze error patterns in logs."""
        error_results = await self.aggregator.search_logs(
            query="",
            start_time=time_range.get("start"),
            end_time=time_range.get("end"),
            level=LogLevel.ERROR,
            limit=5000,
        )

        analysis = {
            "total_errors": len(error_results),
            "error_frequency": self._calculate_error_frequency(
                error_results, time_range
            ),
            "common_errors": self._find_common_errors(error_results),
            "error_sources": self._analyze_error_sources(error_results),
            "error_timeline": self._create_error_timeline(error_results),
        }

        return analysis

    def _calculate_error_frequency(
        self, errors: List[Dict[str, Any]], time_range: Dict[str, datetime]
    ) -> float:
        """Calculate error frequency per hour."""
        if not time_range.get("start") or not time_range.get("end"):
            return 0.0

        duration_hours = (
            time_range["end"] - time_range["start"]
        ).total_seconds() / 3600
        return len(errors) / duration_hours if duration_hours > 0 else 0.0

    def _find_common_errors(self, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find most common error messages."""
        error_counts = {}

        for error in errors:
            message = error.get("message", "Unknown error")
            # Normalize error message (remove specific IDs, timestamps, etc.)
            normalized_message = self._normalize_error_message(message)
            error_counts[normalized_message] = (
                error_counts.get(normalized_message, 0) + 1
            )

        # Sort by frequency
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        return [
            {"message": msg, "count": count, "percentage": count / len(errors) * 100}
            for msg, count in common_errors[:10]  # Top 10
        ]

    def _normalize_error_message(self, message: str) -> str:
        """Normalize error message for pattern detection."""
        import re

        # Remove UUIDs
        message = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "<UUID>",
            message,
            flags=re.IGNORECASE,
        )

        # Remove timestamps
        message = re.sub(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TIMESTAMP>", message
        )

        # Remove specific numbers
        message = re.sub(r"\b\d+\.\d+\b", "<NUMBER>", message)
        message = re.sub(r"\b\d+\b", "<NUMBER>", message)

        # Remove file paths
        message = re.sub(r"/[\w/.-]+", "<PATH>", message)

        return message

    def _analyze_error_sources(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze error sources and services."""
        sources = {}

        for error in errors:
            service = error.get("service", "unknown")
            if service not in sources:
                sources[service] = {"count": 0, "percentage": 0, "categories": {}}

            sources[service]["count"] += 1

            category = error.get("category", "unknown")
            sources[service]["categories"][category] = (
                sources[service]["categories"].get(category, 0) + 1
            )

        # Calculate percentages
        total_errors = len(errors)
        for service_data in sources.values():
            service_data["percentage"] = service_data["count"] / total_errors * 100

        return sources

    def _create_error_timeline(
        self, errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create error timeline for visualization."""
        if not errors:
            return []

        # Group errors by hour
        hourly_counts = {}

        for error in errors:
            timestamp_str = error.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    hour_key = timestamp.replace(minute=0, second=0, microsecond=0)
                    hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1
                except ValueError:
                    continue

        # Convert to timeline format
        timeline = [
            {"timestamp": hour.isoformat(), "error_count": count}
            for hour, count in sorted(hourly_counts.items())
        ]

        return timeline


class SecurityLogAnalyzer:
    """Analyze logs for security events and anomalies."""

    def __init__(self, log_aggregator: LogAggregator):
        self.aggregator = log_aggregator
        self.security_patterns = [
            r"failed.*login.*attempt",
            r"unauthorized.*access",
            r"invalid.*api.*key",
            r"brute.*force.*attack",
            r"sql.*injection.*attempt",
            r"xss.*attempt",
            r"suspicious.*activity",
            r"rate.*limit.*exceeded",
        ]

    async def detect_security_anomalies(
        self, time_range: Dict[str, datetime]
    ) -> Dict[str, Any]:
        """Detect security anomalies in logs."""

        # Search for security-related logs
        security_logs = await self.aggregator.search_logs(
            query="",
            start_time=time_range.get("start"),
            end_time=time_range.get("end"),
            category=LogCategory.SECURITY,
            limit=10000,
        )

        analysis = {
            "total_security_events": len(security_logs),
            "threat_indicators": self._find_threat_indicators(security_logs),
            "suspicious_ips": self._analyze_suspicious_ips(security_logs),
            "failed_authentications": self._count_failed_auth(security_logs),
            "anomalous_patterns": self._detect_anomalous_patterns(security_logs),
            "risk_score": self._calculate_security_risk_score(security_logs),
        }

        return analysis

    def _find_threat_indicators(
        self, logs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find threat indicators in security logs."""
        threats = []

        for log in logs:
            message = log.get("message", "").lower()

            for pattern in self.security_patterns:
                import re

                if re.search(pattern, message, re.IGNORECASE):
                    threats.append(
                        {
                            "timestamp": log.get("timestamp"),
                            "pattern": pattern,
                            "message": log.get("message"),
                            "service": log.get("service"),
                            "data": log.get("data", {}),
                        }
                    )
                    break

        return threats

    def _analyze_suspicious_ips(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze suspicious IP addresses."""
        ip_activity = {}

        for log in logs:
            data = log.get("data", {})
            ip_address = data.get("ip_address") or data.get("source_ip")

            if ip_address:
                if ip_address not in ip_activity:
                    ip_activity[ip_address] = {
                        "total_events": 0,
                        "failed_attempts": 0,
                        "first_seen": log.get("timestamp"),
                        "last_seen": log.get("timestamp"),
                        "event_types": [],
                    }

                activity = ip_activity[ip_address]
                activity["total_events"] += 1
                activity["last_seen"] = log.get("timestamp")

                if "failed" in log.get("message", "").lower():
                    activity["failed_attempts"] += 1

                event_type = data.get("event_type", "unknown")
                if event_type not in activity["event_types"]:
                    activity["event_types"].append(event_type)

        # Identify suspicious IPs
        suspicious_ips = {}
        for ip, activity in ip_activity.items():
            if (
                activity["failed_attempts"] > 10
                or activity["total_events"] > 100
                or len(activity["event_types"]) > 5
            ):
                suspicious_ips[ip] = activity

        return {
            "total_unique_ips": len(ip_activity),
            "suspicious_ips": suspicious_ips,
            "most_active_ip": (
                max(ip_activity.items(), key=lambda x: x[1]["total_events"])[0]
                if ip_activity
                else None
            ),
        }

    def _count_failed_auth(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Count failed authentication attempts."""
        failed_auth = {
            "total_attempts": 0,
            "by_service": {},
            "by_hour": {},
            "common_failure_reasons": {},
        }

        for log in logs:
            message = log.get("message", "").lower()
            if "authentication" in message and "failed" in message:
                failed_auth["total_attempts"] += 1

                # Count by service
                service = log.get("service", "unknown")
                failed_auth["by_service"][service] = (
                    failed_auth["by_service"].get(service, 0) + 1
                )

                # Count by hour
                timestamp_str = log.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(
                            timestamp_str.replace("Z", "+00:00")
                        )
                        hour_key = timestamp.strftime("%Y-%m-%d %H:00")
                        failed_auth["by_hour"][hour_key] = (
                            failed_auth["by_hour"].get(hour_key, 0) + 1
                        )
                    except ValueError:
                        pass

                # Extract failure reason
                data = log.get("data", {})
                reason = data.get("failure_reason", "unknown")
                failed_auth["common_failure_reasons"][reason] = (
                    failed_auth["common_failure_reasons"].get(reason, 0) + 1
                )

        return failed_auth

    def _detect_anomalous_patterns(
        self, logs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalous patterns in security logs."""
        patterns = []

        # Detect time-based anomalies
        time_pattern = self._detect_time_anomalies(logs)
        if time_pattern:
            patterns.append(time_pattern)

        # Detect volume anomalies
        volume_pattern = self._detect_volume_anomalies(logs)
        if volume_pattern:
            patterns.append(volume_pattern)

        # Detect geographic anomalies
        geo_pattern = self._detect_geographic_anomalies(logs)
        if geo_pattern:
            patterns.append(geo_pattern)

        return patterns

    def _detect_time_anomalies(
        self, logs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect time-based anomalies."""
        if len(logs) < 10:
            return None

        # Analyze activity by hour
        hourly_activity = {}
        for log in logs:
            timestamp_str = log.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    hour = timestamp.hour
                    hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
                except ValueError:
                    continue

        if not hourly_activity:
            return None

        # Calculate statistics
        counts = list(hourly_activity.values())
        mean_activity = np.mean(counts)
        std_activity = np.std(counts)

        # Find anomalous hours (more than 2 standard deviations)
        anomalous_hours = []
        for hour, count in hourly_activity.items():
            if abs(count - mean_activity) > 2 * std_activity:
                anomalous_hours.append(
                    {
                        "hour": hour,
                        "activity_count": count,
                        "deviation": abs(count - mean_activity) / std_activity,
                    }
                )

        if anomalous_hours:
            return {
                "type": "time_anomaly",
                "description": "Unusual activity patterns detected",
                "anomalous_hours": anomalous_hours,
                "severity": "high" if len(anomalous_hours) > 3 else "medium",
            }

        return None

    def _detect_volume_anomalies(
        self, logs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect volume-based anomalies."""
        # Group logs by 10-minute intervals
        interval_counts = {}

        for log in logs:
            timestamp_str = log.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    # Round to 10-minute intervals
                    interval = timestamp.replace(
                        minute=(timestamp.minute // 10) * 10, second=0, microsecond=0
                    )
                    interval_counts[interval] = interval_counts.get(interval, 0) + 1
                except ValueError:
                    continue

        if len(interval_counts) < 5:
            return None

        counts = list(interval_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        # Find intervals with unusual volume
        anomalous_intervals = []
        for interval, count in interval_counts.items():
            if count > mean_count + 3 * std_count:  # 3 sigma threshold
                anomalous_intervals.append(
                    {
                        "interval": interval.isoformat(),
                        "event_count": count,
                        "normal_range": f"{mean_count:.1f} ± {std_count:.1f}",
                    }
                )

        if anomalous_intervals:
            return {
                "type": "volume_anomaly",
                "description": "Unusual activity volume detected",
                "anomalous_intervals": anomalous_intervals,
                "severity": "high" if len(anomalous_intervals) > 3 else "medium",
            }

        return None

    def _detect_geographic_anomalies(
        self, logs: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect geographic anomalies in access patterns."""
        # This would require IP geolocation data
        # For now, return placeholder
        return None

    def _calculate_security_risk_score(self, logs: List[Dict[str, Any]]) -> float:
        """Calculate overall security risk score."""
        if not logs:
            return 0.0

        risk_factors = 0
        total_weight = 0

        # Weight different types of security events
        for log in logs:
            message = log.get("message", "").lower()
            weight = 1

            if "failed" in message and "login" in message:
                risk_factors += 2 * weight
            elif "unauthorized" in message:
                risk_factors += 3 * weight
            elif "injection" in message or "xss" in message:
                risk_factors += 5 * weight
            elif "brute force" in message:
                risk_factors += 4 * weight
            else:
                risk_factors += 1 * weight

            total_weight += weight

        # Normalize to 0-1 scale
        raw_score = risk_factors / total_weight if total_weight > 0 else 0
        return min(raw_score / 10, 1.0)  # Cap at 1.0


class LogDashboard:
    """Real-time log dashboard for monitoring."""

    def __init__(self, search_engine: LogSearchEngine):
        self.search_engine = search_engine
        self.dashboard_data = {}
        self.refresh_interval = 30  # seconds

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data."""
        now = datetime.now(timezone.utc)
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)

        # Recent activity
        recent_logs = await self.search_engine.aggregator.search_logs(
            query="", start_time=last_hour, end_time=now, limit=100
        )

        # Error analysis
        error_analysis = await self.search_engine.analyze_error_patterns(
            {"start": last_day, "end": now}
        )

        # Service health indicators
        service_health = await self._get_service_health_indicators(last_hour, now)

        # Performance metrics
        performance_metrics = await self._get_performance_metrics(last_hour, now)

        return {
            "timestamp": now.isoformat(),
            "recent_activity": {
                "total_logs": len(recent_logs),
                "by_level": self._count_by_level(recent_logs),
                "by_service": self._count_by_service(recent_logs),
                "latest_entries": recent_logs[:10],
            },
            "error_analysis": error_analysis,
            "service_health": service_health,
            "performance_metrics": performance_metrics,
            "alerts": await self._generate_alerts(recent_logs),
        }

    def _count_by_level(self, logs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count logs by level."""
        counts = {}
        for log in logs:
            level = log.get("level", "UNKNOWN")
            counts[level] = counts.get(level, 0) + 1
        return counts

    def _count_by_service(self, logs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count logs by service."""
        counts = {}
        for log in logs:
            service = log.get("service", "unknown")
            counts[service] = counts.get(service, 0) + 1
        return counts

    async def _get_service_health_indicators(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get service health indicators from logs."""
        health_logs = await self.search_engine.aggregator.search_logs(
            query="health", start_time=start_time, end_time=end_time, limit=500
        )

        service_status = {}
        for log in health_logs:
            service = log.get("service", "unknown")
            if service not in service_status:
                service_status[service] = {
                    "status": "unknown",
                    "last_heartbeat": None,
                    "error_count": 0,
                    "warning_count": 0,
                }

            level = log.get("level", "INFO")
            if level == "ERROR":
                service_status[service]["error_count"] += 1
            elif level == "WARNING":
                service_status[service]["warning_count"] += 1

            # Update last heartbeat
            timestamp = log.get("timestamp")
            if timestamp:
                if (
                    not service_status[service]["last_heartbeat"]
                    or timestamp > service_status[service]["last_heartbeat"]
                ):
                    service_status[service]["last_heartbeat"] = timestamp

        # Determine overall status
        for service, status in service_status.items():
            if status["error_count"] > 5:
                status["status"] = "unhealthy"
            elif status["warning_count"] > 10:
                status["status"] = "degraded"
            else:
                status["status"] = "healthy"

        return service_status

    async def _get_performance_metrics(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get performance metrics from logs."""
        perf_logs = await self.search_engine.aggregator.search_logs(
            query="",
            start_time=start_time,
            end_time=end_time,
            category=LogCategory.PERFORMANCE,
            limit=1000,
        )

        metrics = {
            "total_operations": len(perf_logs),
            "avg_response_time": 0,
            "slow_operations": [],
            "operation_counts": {},
        }

        response_times = []
        for log in perf_logs:
            data = log.get("data", {})
            duration_ms = data.get("duration_ms")

            if duration_ms:
                response_times.append(duration_ms)

                # Track slow operations
                if duration_ms > 1000:  # Slower than 1 second
                    metrics["slow_operations"].append(
                        {
                            "operation": data.get("operation"),
                            "duration_ms": duration_ms,
                            "timestamp": log.get("timestamp"),
                        }
                    )

                # Count operations
                operation = data.get("operation", "unknown")
                metrics["operation_counts"][operation] = (
                    metrics["operation_counts"].get(operation, 0) + 1
                )

        if response_times:
            metrics["avg_response_time"] = np.mean(response_times)
            metrics["p95_response_time"] = np.percentile(response_times, 95)
            metrics["p99_response_time"] = np.percentile(response_times, 99)

        return metrics

    async def _generate_alerts(
        self, recent_logs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate alerts based on recent log activity."""
        alerts = []

        # High error rate alert
        error_count = sum(
            1 for log in recent_logs if log.get("level") in ["ERROR", "CRITICAL"]
        )
        if error_count > len(recent_logs) * 0.1:  # More than 10% errors
            alerts.append(
                {
                    "type": "high_error_rate",
                    "severity": "warning",
                    "message": f"High error rate detected: {error_count}/{len(recent_logs)} logs",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Service unavailability alert
        services_with_errors = set()
        for log in recent_logs:
            if log.get("level") == "ERROR":
                services_with_errors.add(log.get("service", "unknown"))

        if len(services_with_errors) > 2:
            alerts.append(
                {
                    "type": "multiple_service_errors",
                    "severity": "critical",
                    "message": f'Multiple services reporting errors: {", ".join(services_with_errors)}',
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        # Security alert
        security_events = [
            log for log in recent_logs if log.get("category") == "security"
        ]
        if len(security_events) > 5:
            alerts.append(
                {
                    "type": "security_activity",
                    "severity": "warning",
                    "message": f"Elevated security event activity: {len(security_events)} events",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

        return alerts


# Performance monitoring decorator
def log_performance(operation_name: str, threshold_ms: float = 1000.0):
    """Decorator for automatic performance logging."""

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                # Try to get logger from instance
                logger = None
                if args and hasattr(args[0], "logger"):
                    logger = args[0].logger

                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    if logger and duration_ms > threshold_ms:
                        logger.performance_metric(
                            {
                                "operation": operation_name,
                                "duration_ms": duration_ms,
                                "threshold_ms": threshold_ms,
                                "function": func.__name__,
                                "module": func.__module__,
                            }
                        )

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    if logger:
                        logger.error(
                            f"Performance tracked operation failed: {operation_name}",
                            data={
                                "operation": operation_name,
                                "duration_ms": duration_ms,
                                "function": func.__name__,
                                "error": str(e),
                            },
                            exception=e,
                            category=LogCategory.PERFORMANCE,
                        )
                    raise

            return async_wrapper
        else:

            def sync_wrapper(*args, **kwargs):
                logger = None
                if args and hasattr(args[0], "logger"):
                    logger = args[0].logger

                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    if logger and duration_ms > threshold_ms:
                        logger.performance_metric(
                            {
                                "operation": operation_name,
                                "duration_ms": duration_ms,
                                "threshold_ms": threshold_ms,
                                "function": func.__name__,
                                "module": func.__module__,
                            }
                        )

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    if logger:
                        logger.error(
                            f"Performance tracked operation failed: {operation_name}",
                            data={
                                "operation": operation_name,
                                "duration_ms": duration_ms,
                                "function": func.__name__,
                                "error": str(e),
                            },
                            exception=e,
                            category=LogCategory.PERFORMANCE,
                        )
                    raise

            return sync_wrapper

    return decorator


# Utility functions
def create_service_logger(service_name: str) -> TradingLogger:
    """Create a logger for a trading service."""
    return LoggingConfig.setup_service_logging(service_name)


def setup_log_aggregation(log_directory: str) -> LogAggregator:
    """Setup log aggregation for the system."""
    return LogAggregator(log_directory)


def create_audit_manager(logger: StructuredLogger) -> AuditTrailManager:
    """Create audit trail manager."""
    return AuditTrailManager(logger)


# Context managers for common logging patterns
@contextmanager
def trade_logging_context(
    logger: TradingLogger, trade_id: str, symbol: str, user_id: Optional[str] = None
):
    """Context manager for trade-specific logging."""
    original_context = dict(logger.context.__dict__)

    try:
        logger.set_context(correlation_id=trade_id, user_id=user_id)

        logger.info(
            f"Starting trade operation for {symbol}",
            data={"trade_id": trade_id, "symbol": symbol},
            category=LogCategory.TRADING,
            tags=["trade", "start"],
        )

        yield

        logger.info(
            f"Completed trade operation for {symbol}",
            data={"trade_id": trade_id, "symbol": symbol},
            category=LogCategory.TRADING,
            tags=["trade", "complete"],
        )

    except Exception as e:
        logger.error(
            f"Trade operation failed for {symbol}",
            data={"trade_id": trade_id, "symbol": symbol},
            category=LogCategory.TRADING,
            tags=["trade", "error"],
            exception=e,
        )
        raise

    finally:
        # Restore original context
        for key, value in original_context.items():
            setattr(logger.context, key, value)


@contextmanager
def api_logging_context(
    logger: TradingLogger, api_name: str, endpoint: str, method: str = "GET"
):
    """Context manager for API request logging."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    try:
        logger.info(
            f"API request: {method} {endpoint}",
            data={
                "request_id": request_id,
                "api_name": api_name,
                "endpoint": endpoint,
                "method": method,
            },
            category=LogCategory.API,
            tags=["api", "request", "start"],
        )

        yield request_id

        duration_ms = (time.time() - start_time) * 1000
        logger.api_request(
            {
                "request_id": request_id,
                "api_name": api_name,
                "endpoint": endpoint,
                "method": method,
                "status": "success",
            },
            duration_ms,
        )

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"API request failed: {method} {endpoint}",
            data={
                "request_id": request_id,
                "api_name": api_name,
                "endpoint": endpoint,
                "method": method,
                "duration_ms": duration_ms,
            },
            category=LogCategory.API,
            tags=["api", "request", "error"],
            exception=e,
        )
        raise


# Configuration templates
LOGGING_CONFIGS = {
    "development": {
        "log_level": LogLevel.DEBUG,
        "console_output": True,
        "file_logging": True,
        "elasticsearch": False,
        "performance_tracking": True,
    },
    "staging": {
        "log_level": LogLevel.INFO,
        "console_output": True,
        "file_logging": True,
        "elasticsearch": True,
        "performance_tracking": True,
    },
    "production": {
        "log_level": LogLevel.INFO,
        "console_output": False,
        "file_logging": True,
        "elasticsearch": True,
        "performance_tracking": False,
    },
}


def get_logging_config(environment: str) -> Dict[str, Any]:
    """Get logging configuration for environment."""
    return LOGGING_CONFIGS.get(environment, LOGGING_CONFIGS["development"])


# Export main classes and functions
__all__ = [
    "LogLevel",
    "LogCategory",
    "LogContext",
    "TradeLogEntry",
    "StructuredLogger",
    "TradingLogger",
    "LogAggregator",
    "AuditTrailManager",
    "LogRetentionManager",
    "SecurityLogAnalyzer",
    "LogSearchEngine",
    "LogDashboard",
    "LoggingConfig",
    "log_performance",
    "create_service_logger",
    "setup_log_aggregation",
    "create_audit_manager",
    "trade_logging_context",
    "api_logging_context",
    "get_logging_config",
]
