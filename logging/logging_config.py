import json
import logging
import logging.config
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from pythonjsonlogger.json import JsonFormatter


class TradingSystemProcessor:
    """Custom structlog processor for trading system"""

    def __init__(self, service_name: str, environment: str = "development"):
        self.service_name = service_name
        self.environment = environment

    def __call__(
        self, logger: Any, method_name: str, event_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process log event"""
        # Add service metadata
        event_dict["service"] = self.service_name
        event_dict["environment"] = self.environment
        event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        event_dict["level"] = method_name.upper()

        # Add correlation ID if available
        try:
            # Get context from structlog using proper API
            bound_logger = structlog.get_logger()
            context = getattr(bound_logger, "_context", {})

            if "correlation_id" in context:
                event_dict["correlation_id"] = context["correlation_id"]
        except (AttributeError, TypeError):
            pass

        return event_dict


class TradingJSONFormatter(JsonFormatter):
    """Custom JSON formatter for trading system logs"""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Ensure timestamp is present
        if not log_record.get("timestamp"):
            log_record["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add service metadata if not present
        if not log_record.get("service"):
            log_record["service"] = getattr(record, "service", "unknown")

        # Add log level
        log_record["level"] = record.levelname

        # Add source location
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno

        # Add process and thread info
        log_record["process_id"] = record.process
        log_record["thread_id"] = record.thread

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }


class TradeAuditFormatter(JsonFormatter):
    """Special formatter for trade audit logs"""

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Add audit-specific fields
        log_record["audit_type"] = "trade_event"
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_record["service"] = getattr(record, "service", "trade_executor")

        # Extract trading-specific information
        if hasattr(record, "trade_data"):
            log_record.update(getattr(record, "trade_data", {}))


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs"""

    SENSITIVE_FIELDS = [
        "password",
        "api_key",
        "secret_key",
        "token",
        "auth",
        "authorization",
        "x-api-key",
        "private_key",
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive information"""
        if hasattr(record, "msg") and isinstance(record.msg, str):
            message = record.msg
            for field in self.SENSITIVE_FIELDS:
                if field in message.lower():
                    # Replace with masked value
                    record.msg = self._mask_sensitive_data(message)

        # Filter args as well
        if hasattr(record, "args") and record.args:
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, (str, dict)):
                    filtered_args.append(self._mask_sensitive_data(str(arg)))
                else:
                    filtered_args.append(str(arg))
            record.args = tuple(filtered_args)

        return True

    def _mask_sensitive_data(self, text: str) -> str:
        """Mask sensitive data in text"""
        import re

        # Pattern for API keys and tokens
        patterns = [
            (
                r'api[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{10,})',
                r'api_key: "***MASKED***"',
            ),
            (
                r'secret[_-]?key["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{10,})',
                r'secret_key: "***MASKED***"',
            ),
            (
                r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{10,})',
                r'token: "***MASKED***"',
            ),
            (
                r'password["\']?\s*[:=]\s*["\']?([a-zA-Z0-9]{8,})',
                r'password: "***MASKED***"',
            ),
        ]

        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result


def setup_logging(
    service_name: str, log_level: str = "INFO", environment: str = "development"
) -> None:
    """Setup comprehensive logging for trading system"""

    # Create logs directory
    log_dir = Path("/app/data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            TradingSystemProcessor(service_name, environment),
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": TradingJSONFormatter,
                "format": "%(timestamp)s %(level)s %(service)s %(logger)s %(message)s",
            },
            "trade_audit": {
                "()": TradeAuditFormatter,
                "format": "%(timestamp)s %(audit_type)s %(message)s",
            },
            "console": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "filters": {"security_filter": {"()": SecurityFilter}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
                "level": log_level,
                "stream": sys.stdout,
                "filters": ["security_filter"],
            },
            "main_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_dir}/trading_system.log",
                "formatter": "json",
                "level": log_level,
                "maxBytes": 50 * 1024 * 1024,  # 50MB
                "backupCount": 10,
                "filters": ["security_filter"],
            },
            "service_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_dir}/{service_name}.log",
                "formatter": "json",
                "level": log_level,
                "maxBytes": 20 * 1024 * 1024,  # 20MB
                "backupCount": 5,
                "filters": ["security_filter"],
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_dir}/error.log",
                "formatter": "json",
                "level": "ERROR",
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 10,
                "filters": ["security_filter"],
            },
            "trade_audit_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_dir}/trade_audit.log",
                "formatter": "trade_audit",
                "level": "INFO",
                "maxBytes": 100 * 1024 * 1024,  # 100MB
                "backupCount": 20,
                "filters": ["security_filter"],
            },
            "risk_events_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_dir}/risk_events.log",
                "formatter": "json",
                "level": "WARNING",
                "maxBytes": 20 * 1024 * 1024,  # 20MB
                "backupCount": 10,
                "filters": ["security_filter"],
            },
            "performance_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": f"{log_dir}/performance.log",
                "formatter": "json",
                "level": "INFO",
                "maxBytes": 30 * 1024 * 1024,  # 30MB
                "backupCount": 5,
                "filters": ["security_filter"],
            },
        },
        "loggers": {
            # Root logger
            "": {
                "handlers": ["console", "main_file", "error_file"],
                "level": log_level,
                "propagate": False,
            },
            # Service-specific logger
            service_name: {
                "handlers": ["console", "service_file", "error_file"],
                "level": log_level,
                "propagate": False,
            },
            # Trade audit logger
            "trade_audit": {
                "handlers": ["trade_audit_file"],
                "level": "INFO",
                "propagate": False,
            },
            # Risk events logger
            "risk_events": {
                "handlers": ["risk_events_file", "error_file"],
                "level": "WARNING",
                "propagate": False,
            },
            # Performance logger
            "performance": {
                "handlers": ["performance_file"],
                "level": "INFO",
                "propagate": False,
            },
            # External libraries
            "httpx": {"level": "WARNING", "propagate": True},
            "urllib3": {"level": "WARNING", "propagate": True},
            "asyncio": {"level": "WARNING", "propagate": True},
        },
    }

    # Apply configuration
    logging.config.dictConfig(config)

    # Set up structlog logger
    logger = structlog.get_logger(service_name)
    logger.info(
        "Logging system initialized",
        service=service_name,
        environment=environment,
        log_level=log_level,
    )


class TradingLogger:
    """Enhanced logger for trading system with specialized methods"""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = structlog.get_logger(service_name)
        self.audit_logger = logging.getLogger("trade_audit")
        self.risk_logger = logging.getLogger("risk_events")
        self.performance_logger = logging.getLogger("performance")

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with context"""
        self.logger.critical(message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context"""
        self.logger.debug(message, **kwargs)

    def log_trade_execution(self, trade_data: Dict[str, Any]) -> None:
        """Log trade execution for audit trail"""
        audit_data = {
            "event_type": "trade_execution",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "trade_id": trade_data.get("trade_id"),
            "order_id": trade_data.get("order_id"),
            "symbol": trade_data.get("symbol"),
            "side": trade_data.get("side"),
            "quantity": trade_data.get("quantity"),
            "price": trade_data.get("price"),
            "value": trade_data.get("value"),
            "commission": trade_data.get("commission"),
            "strategy": trade_data.get("strategy"),
            "execution_latency": trade_data.get("execution_latency"),
            "slippage": trade_data.get("slippage"),
        }

        # Log to audit file
        self.audit_logger.info("Trade executed", extra={"trade_data": audit_data})

        # Log to main logger
        self.logger.info(
            "Trade executed",
            trade_id=trade_data.get("trade_id"),
            symbol=trade_data.get("symbol"),
            side=trade_data.get("side"),
            quantity=trade_data.get("quantity"),
            price=trade_data.get("price"),
            strategy=trade_data.get("strategy"),
        )

    def log_signal_generation(self, signal_data: Dict[str, Any]) -> None:
        """Log signal generation"""
        self.logger.info(
            "Trading signal generated",
            symbol=signal_data.get("symbol"),
            signal_type=signal_data.get("signal_type"),
            confidence=signal_data.get("confidence"),
            strategy=signal_data.get("strategy"),
            price=signal_data.get("price"),
            metadata=signal_data.get("metadata", {}),
        )

    def log_risk_event(self, risk_event_data: Dict[str, Any]) -> None:
        """Log risk management event"""
        self.risk_logger.warning(
            f"Risk event: {risk_event_data.get('event_type', 'unknown')}",
            extra={
                "event_type": risk_event_data.get("event_type"),
                "severity": risk_event_data.get("severity"),
                "affected_symbols": risk_event_data.get("affected_symbols", []),
                "message": risk_event_data.get("message"),
                "recommended_action": risk_event_data.get("recommended_action"),
                "portfolio_impact": risk_event_data.get("portfolio_impact"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Also log to main logger based on severity
        if risk_event_data.get("severity") == "critical":
            self.logger.critical(
                "Critical risk event",
                event_type=risk_event_data.get("event_type"),
                message=risk_event_data.get("message"),
            )
        else:
            self.logger.warning(
                "Risk event",
                event_type=risk_event_data.get("event_type"),
                message=risk_event_data.get("message"),
            )

    def log_portfolio_update(self, portfolio_data: Dict[str, Any]) -> None:
        """Log portfolio state updates"""
        self.logger.info(
            "Portfolio updated",
            total_value=portfolio_data.get("total_value"),
            cash=portfolio_data.get("cash"),
            positions_count=portfolio_data.get("positions_count"),
            daily_pnl=portfolio_data.get("daily_pnl"),
            total_pnl=portfolio_data.get("total_pnl"),
            drawdown=portfolio_data.get("drawdown"),
        )

        # Log to performance logger
        self.performance_logger.info(
            "Portfolio performance update", extra=portfolio_data
        )

    def log_market_data_collection(self, collection_data: Dict[str, Any]) -> None:
        """Log market data collection events"""
        self.logger.debug(
            "Market data collected",
            symbol=collection_data.get("symbol"),
            timeframe=collection_data.get("timeframe"),
            source=collection_data.get("source"),
            data_points=collection_data.get("data_points", 0),
            collection_time=collection_data.get("collection_time"),
        )

    def log_api_request(self, api_data: Dict[str, Any]) -> None:
        """Log external API requests"""
        self.logger.debug(
            "External API request",
            api=api_data.get("api"),
            endpoint=api_data.get("endpoint"),
            method=api_data.get("method", "GET"),
            status_code=api_data.get("status_code"),
            response_time=api_data.get("response_time"),
            rate_limited=api_data.get("rate_limited", False),
        )

    def log_database_event(self, db_data: Dict[str, Any]) -> None:
        """Log database operations"""
        self.logger.debug(
            "Database operation",
            operation=db_data.get("operation"),
            table=db_data.get("table"),
            query_time=db_data.get("query_time"),
            rows_affected=db_data.get("rows_affected"),
            success=db_data.get("success", True),
        )

    def log_system_event(self, event_data: Dict[str, Any]) -> None:
        """Log system-level events"""
        level = event_data.get("level", "info").lower()

        if level == "critical":
            self.logger.critical(
                event_data.get("message", "System event"),
                **{k: v for k, v in event_data.items() if k != "message"},
            )
        elif level == "error":
            self.logger.error(
                event_data.get("message", "System event"),
                **{k: v for k, v in event_data.items() if k != "message"},
            )
        elif level == "warning":
            self.logger.warning(
                event_data.get("message", "System event"),
                **{k: v for k, v in event_data.items() if k != "message"},
            )
        else:
            self.logger.info(
                event_data.get("message", "System event"),
                **{k: v for k, v in event_data.items() if k != "message"},
            )

    def log_strategy_performance(self, strategy_data: Dict[str, Any]) -> None:
        """Log strategy performance metrics"""
        self.performance_logger.info(
            f"Strategy performance: {strategy_data.get('strategy_name')}",
            extra={
                "event_type": "strategy_performance",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **strategy_data,
            },
        )

        self.logger.info(
            "Strategy performance update",
            strategy=strategy_data.get("strategy_name"),
            win_rate=strategy_data.get("win_rate"),
            sharpe_ratio=strategy_data.get("sharpe_ratio"),
            total_return=strategy_data.get("total_return"),
            total_trades=strategy_data.get("total_trades"),
        )

    def log_order_lifecycle(self, order_data: Dict[str, Any], event_type: str) -> None:
        """Log order lifecycle events"""
        self.audit_logger.info(
            f"Order {event_type}",
            extra={
                "trade_data": {
                    "event_type": f"order_{event_type}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_data.get("order_id"),
                    "symbol": order_data.get("symbol"),
                    "side": order_data.get("side"),
                    "order_type": order_data.get("order_type"),
                    "quantity": order_data.get("quantity"),
                    "price": order_data.get("price"),
                    "status": order_data.get("status"),
                    "filled_quantity": order_data.get("filled_quantity", 0),
                    "filled_price": order_data.get("filled_price"),
                }
            },
        )

    def log_error_with_context(
        self, error: Exception, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log error with full context"""
        self.logger.error(
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            exc_info=True,
        )

    def log_performance_metric(
        self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metrics"""
        self.performance_logger.info(
            f"Performance metric: {metric_name}",
            extra={
                "event_type": "performance_metric",
                "metric_name": metric_name,
                "metric_value": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            },
        )

    def log_business_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log business-level events"""
        self.logger.info(
            f"Business event: {event_type}", event_type=event_type, **event_data
        )

    def with_context(self, **context: Any) -> "TradingLogger":
        """Create logger with additional context"""
        # Add context to structlog
        structlog.contextvars.bind_contextvars(**context)
        return self

    def set_correlation_id(self, correlation_id: str) -> None:
        """Add correlation ID to log context"""
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

    def clear_context(self) -> None:
        """Clear log context"""
        structlog.contextvars.clear_contextvars()


class LogAnalyzer:
    """Analyze logs for patterns and anomalies"""

    def __init__(self, log_directory: str = "/app/data/logs"):
        self.log_directory = Path(log_directory)
        self.logger = logging.getLogger(__name__)

    def analyze_error_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze error patterns in logs"""
        error_log_path = self.log_directory / "error.log"

        if not error_log_path.exists():
            return {"error": "Error log file not found"}

        error_counts: Dict[str, int] = {}
        recent_errors: List[Dict[str, Any]] = []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            with open(error_log_path, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        timestamp = datetime.fromisoformat(
                            log_entry.get("timestamp", "")
                        )

                        if timestamp >= cutoff_time:
                            error_type = log_entry.get("exception", {}).get(
                                "type", "Unknown"
                            )
                            error_message = log_entry.get("message", "")

                            # Count error types
                            error_counts[error_type] = (
                                error_counts.get(error_type, 0) + 1
                            )

                            # Store recent errors
                            recent_errors.append(
                                {
                                    "timestamp": log_entry.get("timestamp"),
                                    "service": log_entry.get("service"),
                                    "error_type": error_type,
                                    "message": error_message,
                                    "module": log_entry.get("module"),
                                }
                            )

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        except Exception as e:
            self.logger.error(f"Failed to analyze error patterns: {e}")
            return {"error": str(e)}

        return {
            "analysis_period_hours": hours,
            "total_errors": len(recent_errors),
            "error_counts_by_type": error_counts,
            "most_common_errors": sorted(
                error_counts.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "recent_errors": recent_errors[-20:],  # Last 20 errors
        }

    def analyze_trade_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze trading patterns in audit logs"""
        audit_log_path = self.log_directory / "trade_audit.log"

        if not audit_log_path.exists():
            return {"error": "Trade audit log file not found"}

        trade_stats: Dict[str, Any] = {
            "total_trades": 0,
            "by_symbol": {},
            "by_strategy": {},
            "by_side": {"buy": 0, "sell": 0},
            "execution_times": [],
            "trade_values": [],
            "hourly_distribution": {},
        }

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            with open(audit_log_path, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        timestamp = datetime.fromisoformat(
                            log_entry.get("timestamp", "")
                        )

                        if (
                            timestamp >= cutoff_time
                            and log_entry.get("event_type") == "trade_execution"
                        ):
                            trade_stats["total_trades"] += 1

                            # By symbol
                            symbol = log_entry.get("symbol", "Unknown")
                            trade_stats["by_symbol"][symbol] = (
                                trade_stats["by_symbol"].get(symbol, 0) + 1
                            )

                            # By strategy
                            strategy = log_entry.get("strategy", "Unknown")
                            trade_stats["by_strategy"][strategy] = (
                                trade_stats["by_strategy"].get(strategy, 0) + 1
                            )

                            # By side
                            side = log_entry.get("side", "").lower()
                            if side in trade_stats["by_side"]:
                                trade_stats["by_side"][side] += 1

                            # Execution times
                            exec_time = log_entry.get("execution_latency")
                            if exec_time:
                                trade_stats["execution_times"].append(exec_time)

                            # Trade values
                            value = log_entry.get("value")
                            if value:
                                trade_stats["trade_values"].append(value)

                            # Hourly distribution
                            hour = timestamp.hour
                            trade_stats["hourly_distribution"][hour] = (
                                trade_stats["hourly_distribution"].get(hour, 0) + 1
                            )

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        except Exception as e:
            self.logger.error(f"Failed to analyze trade patterns: {e}")
            return {"error": str(e)}

        # Calculate statistics
        if trade_stats["execution_times"]:
            trade_stats["avg_execution_time"] = sum(
                trade_stats["execution_times"]
            ) / len(trade_stats["execution_times"])
            trade_stats["max_execution_time"] = max(trade_stats["execution_times"])
            trade_stats["min_execution_time"] = min(trade_stats["execution_times"])

        if trade_stats["trade_values"]:
            trade_stats["avg_trade_value"] = sum(trade_stats["trade_values"]) / len(
                trade_stats["trade_values"]
            )
            trade_stats["total_trade_value"] = sum(trade_stats["trade_values"])

        return trade_stats

    def analyze_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze performance trends from logs"""
        performance_log_path = self.log_directory / "performance.log"

        if not performance_log_path.exists():
            return {"error": "Performance log file not found"}

        performance_data: Dict[str, Any] = {
            "daily_returns": [],
            "sharpe_ratios": [],
            "drawdowns": [],
            "portfolio_values": [],
            "strategy_performance": {},
        }

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            with open(performance_log_path, "r") as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)
                        timestamp = datetime.fromisoformat(
                            log_entry.get("timestamp", "")
                        )

                        if timestamp >= cutoff_time:
                            event_type = log_entry.get("event_type")

                            if event_type == "portfolio_performance":
                                performance_data["daily_returns"].append(
                                    log_entry.get("daily_return", 0)
                                )
                                performance_data["sharpe_ratios"].append(
                                    log_entry.get("sharpe_ratio", 0)
                                )
                                performance_data["drawdowns"].append(
                                    log_entry.get("drawdown", 0)
                                )
                                performance_data["portfolio_values"].append(
                                    log_entry.get("total_value", 0)
                                )

                            elif event_type == "strategy_performance":
                                strategy = log_entry.get("strategy_name")
                                if strategy:
                                    if (
                                        strategy
                                        not in performance_data["strategy_performance"]
                                    ):
                                        performance_data["strategy_performance"][
                                            strategy
                                        ] = []
                                    performance_data["strategy_performance"][
                                        strategy
                                    ].append(
                                        {
                                            "timestamp": log_entry.get("timestamp"),
                                            "win_rate": log_entry.get("win_rate"),
                                            "sharpe_ratio": log_entry.get(
                                                "sharpe_ratio"
                                            ),
                                            "total_return": log_entry.get(
                                                "total_return"
                                            ),
                                        }
                                    )

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        except Exception as e:
            self.logger.error(f"Failed to analyze performance trends: {e}")
            return {"error": str(e)}

        # Calculate trend statistics
        trends = {}
        if performance_data["portfolio_values"]:
            values = performance_data["portfolio_values"]
            trends["portfolio_trend"] = (
                "increasing" if values[-1] > values[0] else "decreasing"
            )
            trends["portfolio_volatility"] = str(
                float(np.std(performance_data["daily_returns"]))
                if performance_data["daily_returns"]
                else 0.0
            )

        return {
            "performance_data": performance_data,
            "trends": trends,
            "analysis_period_days": days,
        }
