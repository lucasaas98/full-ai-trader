"""
Shared utilities for the automated trading system.

This module provides common utility functions including logging setup,
date/time helpers, and other shared functionality.
"""

import hashlib
import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .config import Config


def setup_logging(config: Config, service_name: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration for the service.

    Args:
        config: Configuration object
        service_name: Name of the service (used in log messages)

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(config.logging.file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging format
    formatter = logging.Formatter(config.logging.format)

    # Create logger
    logger_name = service_name or config.service_name
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, config.logging.level))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if config.logging.enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.logging.level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if config.logging.enable_file:
        # Use service-specific log file if service name is provided
        log_file = config.logging.file_path
        if service_name:
            log_dir = Path(config.logging.file_path).parent
            log_file = str(log_dir / f"{service_name}.log")

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=config.logging.max_file_size,
            backupCount=config.logging.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, config.logging.level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """
    Convert datetime to UTC.

    Args:
        dt: Datetime to convert

    Returns:
        UTC datetime
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_currency(amount: Union[Decimal, float, int], currency: str = "USD") -> str:
    """
    Format amount as currency string.

    Args:
        amount: Amount to format
        currency: Currency code

    Returns:
        Formatted currency string
    """
    if isinstance(amount, (int, float)):
        amount = Decimal(str(amount))

    # Round to 2 decimal places
    amount = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return f"{currency} {amount:,.2f}"


def format_percentage(value: Union[Decimal, float], decimal_places: int = 2) -> str:
    """
    Format value as percentage string.

    Args:
        value: Value to format (e.g., 0.05 for 5%)
        decimal_places: Number of decimal places

    Returns:
        Formatted percentage string
    """
    if isinstance(value, Decimal):
        percentage = value * 100
    else:
        percentage = Decimal(str(value)) * 100

    format_str = f"{{:.{decimal_places}f}}%"
    return format_str.format(float(percentage))


def calculate_hash(data: str) -> str:
    """
    Calculate SHA256 hash of string data.

    Args:
        data: String data to hash

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def safe_divide(
    numerator: Union[Decimal, float], denominator: Union[Decimal, float]
) -> Optional[Decimal]:
    """
    Safely divide two numbers, returning None if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator

    Returns:
        Division result or None if denominator is zero
    """
    if denominator == 0:
        return None

    if isinstance(numerator, float):
        numerator = Decimal(str(numerator))
    if isinstance(denominator, float):
        denominator = Decimal(str(denominator))

    result = numerator / denominator
    return Decimal(str(result)) if not isinstance(result, Decimal) else result


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 1.0):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Backoff multiplier

    Returns:
        Decorated function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    # Calculate backoff delay
                    delay = backoff_factor * (2**attempt)
                    time.sleep(delay)

            # Re-raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError(
                    "All retry attempts failed but no exception was captured"
                )

        return wrapper

    return decorator


def validate_symbol(symbol: str) -> bool:
    """
    Validate trading symbol format.

    Args:
        symbol: Symbol to validate

    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False

    # Basic validation - alphanumeric and common separators
    allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-/")
    return all(c in allowed_chars for c in symbol.upper())


def chunks(lst: list, n: int):
    """
    Yield successive n-sized chunks from list.

    Args:
        lst: List to chunk
        n: Chunk size

    Yields:
        List chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def flatten_dict(
    d: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """
    Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator between keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def serialize_for_json(obj: Any) -> Any:
    """
    Serialize object for JSON encoding.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, Decimal):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, "dict"):
        # Pydantic model
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(" .")

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    return filename


def calculate_price_change(
    current_price: Decimal, previous_price: Decimal
) -> Dict[str, Decimal]:
    """
    Calculate price change and percentage change.

    Args:
        current_price: Current price
        previous_price: Previous price

    Returns:
        Dictionary with 'change' and 'change_percent' keys
    """
    change = current_price - previous_price
    change_percent = safe_divide(change, previous_price) or Decimal("0")

    return {"change": change, "change_percent": change_percent}


def is_market_hours(config: Config, dt: Optional[datetime] = None) -> bool:
    """
    Check if current time is within market hours.

    Args:
        config: Configuration object
        dt: Datetime to check (defaults to current time)

    Returns:
        True if within market hours, False otherwise
    """
    if dt is None:
        dt = utc_now()

    # Convert to market timezone
    # Note: This is a simplified implementation
    # In production, you'd want to use proper timezone handling
    # and account for market holidays

    # Check if weekend (simplified)
    if dt.weekday() >= 5 and not config.scheduler.trade_weekends:
        return False

    # For simplicity, assuming ET market hours in UTC
    # Real implementation should use proper timezone conversion
    market_open_hour = 14  # 9:30 AM ET in UTC (approximate)
    market_close_hour = 21  # 4:00 PM ET in UTC (approximate)

    current_hour = dt.hour
    return market_open_hour <= current_hour < market_close_hour


def round_to_tick_size(price: Decimal, tick_size: Decimal = Decimal("0.01")) -> Decimal:
    """
    Round price to valid tick size.

    Args:
        price: Price to round
        tick_size: Minimum tick size

    Returns:
        Rounded price
    """
    return (price / tick_size).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    ) * tick_size


def calculate_position_size(
    portfolio_value: Decimal,
    risk_per_trade: Decimal,
    entry_price: Decimal,
    stop_loss_price: Decimal,
) -> int:
    """
    Calculate position size based on risk management rules.

    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Risk per trade as percentage (e.g., 0.02 for 2%)
        entry_price: Entry price
        stop_loss_price: Stop loss price

    Returns:
        Position size in shares
    """
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0

    risk_amount = portfolio_value * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)

    if price_risk <= 0:
        return 0

    position_size = risk_amount / price_risk
    return int(position_size)


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes, 0 if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0


def ensure_directory(path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json_file(file_path: str, default: Any = None) -> Any:
    """
    Load JSON file with error handling.

    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, IOError):
        return default


def save_json_file(data: Any, file_path: str, indent: int = 2) -> bool:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        file_path: Path to save file
        indent: JSON indentation

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        ensure_directory(str(Path(file_path).parent))

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=serialize_for_json)
        return True
    except (IOError, TypeError) as e:
        logger = get_logger(__name__)
        logger.error(f"Failed to save JSON file {file_path}: {e}")
        return False


class RateLimiter:
    """Simple rate limiter implementation."""

    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    def is_allowed(self) -> bool:
        """
        Check if request is allowed under rate limit.

        Returns:
            True if allowed, False otherwise
        """
        now = time.time()

        # Remove old requests outside time window
        self.requests = [
            req_time for req_time in self.requests if now - req_time < self.time_window
        ]

        # Check if we can make a new request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True

        return False

    def wait_time(self) -> float:
        """
        Get time to wait before next request is allowed.

        Returns:
            Wait time in seconds
        """
        if not self.requests:
            return 0.0

        oldest_request = min(self.requests)
        return max(0.0, self.time_window - (time.time() - oldest_request))


def timing_decorator(func):
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            logger = get_logger(func.__module__)
            logger.debug(
                f"{func.__name__} executed in {end_time - start_time:.4f} seconds"
            )

    return wrapper


def validate_environment_variables(required_vars: List[str]) -> Dict[str, str]:
    """
    Validate that required environment variables are set.

    Args:
        required_vars: List of required environment variable names

    Returns:
        Dictionary of environment variables

    Raises:
        ValueError: If any required variables are missing
    """
    env_vars = {}
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
        else:
            env_vars[var] = value

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    return env_vars


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def parse_timeframe_to_seconds(timeframe: str) -> int:
    """
    Parse timeframe string to seconds.

    Args:
        timeframe: Timeframe string (e.g., "1min", "1h", "1day")

    Returns:
        Seconds
    """
    timeframe_map = {
        "1min": 60,
        "5min": 300,
        "15min": 900,
        "30min": 1800,
        "1h": 3600,
        "4h": 14400,
        "1day": 86400,
        "1week": 604800,
        "1month": 2592000,  # Approximate
    }

    return timeframe_map.get(timeframe, 60)  # Default to 1 minute


def health_check_service(service_url: str, timeout: int = 5) -> bool:
    """
    Perform basic health check on a service.

    Args:
        service_url: URL to check
        timeout: Request timeout in seconds

    Returns:
        True if service is healthy, False otherwise
    """
    try:
        import requests

        response = requests.get(f"{service_url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def get_memory_usage() -> Dict[str, Union[float, str]]:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory usage information
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


def get_cpu_usage() -> float:
    """
    Get current CPU usage percentage.

    Returns:
        CPU usage percentage
    """
    try:
        import psutil

        return psutil.cpu_percent(interval=1)
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Timeout in seconds before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, func, *args, **kwargs):
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.timeout
            ):
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
