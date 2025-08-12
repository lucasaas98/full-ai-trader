"""
Centralized configuration management for the automated trading system.

This module provides configuration management using environment variables
with sensible defaults and validation.
"""

import os
from decimal import Decimal
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    host: str = Field(default="localhost", alias="DB_HOST")
    port: int = Field(default=5432, alias="DB_PORT")
    database: str = Field(default="trading_system", alias="DB_NAME")
    username: str = Field(default="trader", alias="DB_USER")
    password: str = Field(default="", alias="DB_PASSWORD")
    pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseSettings):
    """Redis configuration."""

    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    database: int = Field(default=0, alias="REDIS_DB")
    password: Optional[str] = Field(None, alias="REDIS_PASSWORD")
    max_connections: int = Field(default=20, alias="REDIS_MAX_CONNECTIONS")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

    @property
    def url(self) -> str:
        """Get Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"


class AlpacaConfig(BaseSettings):
    """Alpaca API configuration."""

    api_key: str = Field(default="", alias="ALPACA_API_KEY")
    secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    base_url: str = Field(default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL")
    data_url: str = Field(default="https://data.alpaca.markets", alias="ALPACA_DATA_URL")
    paper_trading: bool = Field(default=True, alias="ALPACA_PAPER_TRADING")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v):
        """Validate Alpaca base URL."""
        valid_urls = [
            "https://api.alpaca.markets",  # Live trading
            "https://paper-api.alpaca.markets"  # Paper trading
        ]
        if v not in valid_urls:
            raise ValueError(f"Invalid Alpaca base URL. Must be one of: {valid_urls}")
        return v


class TwelveDataConfig(BaseSettings):
    """TwelveData API configuration."""

    api_key: str = Field(default="", alias="TWELVE_DATA_API_KEY")
    base_url: str = Field(default="https://api.twelvedata.com", alias="TWELVE_DATA_BASE_URL")
    rate_limit_requests: int = Field(default=800, alias="TWELVE_DATA_RATE_LIMIT")
    rate_limit_period: int = Field(default=60, alias="TWELVE_DATA_RATE_PERIOD")  # seconds
    timeout: int = Field(default=30, alias="TWELVE_DATA_TIMEOUT")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )


class FinVizConfig(BaseSettings):
    """FinViz configuration."""

    api_key: Optional[str] = Field(None, alias="FINVIZ_API_KEY")
    base_url: str = Field(default="https://finviz.com", alias="FINVIZ_BASE_URL")
    timeout: int = Field(default=30, alias="FINVIZ_TIMEOUT")
    rate_limit_delay: float = Field(default=1.0, alias="FINVIZ_RATE_LIMIT_DELAY")  # seconds between requests

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )


class JWTConfig(BaseSettings):
    """JWT authentication configuration."""

    secret_key: str = Field(default="default-jwt-secret-change-in-production", alias="JWT_SECRET")
    algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    issuer: str = Field(default="trading-system", alias="JWT_ISSUER")
    audience: str = Field(default="trading-api", alias="JWT_AUDIENCE")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v):
        """Validate JWT secret key strength."""
        if not v:
            # Allow empty for configuration loading, but warn
            return v
        if len(v) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
        return v


class NotificationConfig(BaseSettings):
    """Notification configuration."""

    gotify_url: Optional[str] = Field(None, alias="GOTIFY_URL")
    gotify_token: Optional[str] = Field(None, alias="GOTIFY_TOKEN")
    slack_webhook_url: Optional[str] = Field(None, alias="SLACK_WEBHOOK_URL")
    email_smtp_host: Optional[str] = Field(None, alias="EMAIL_SMTP_HOST")
    email_smtp_port: int = Field(default=587, alias="EMAIL_SMTP_PORT")
    email_username: Optional[str] = Field(None, alias="EMAIL_USERNAME")
    email_password: Optional[str] = Field(None, alias="EMAIL_PASSWORD")
    email_from: Optional[str] = Field(None, alias="EMAIL_FROM")
    email_to: Optional[List[str]] = Field(None, alias="EMAIL_TO")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

    @field_validator('email_to', mode='before')
    @classmethod
    def parse_email_list(cls, v):
        """Parse comma-separated email list."""
        if isinstance(v, str):
            return [email.strip() for email in v.split(',') if email.strip()]
        return v


class RiskConfig(BaseSettings):
    """Risk management configuration."""

    max_position_size: Decimal = Field(default=Decimal("0.05"), alias="RISK_MAX_POSITION_SIZE")
    max_portfolio_risk: Decimal = Field(default=Decimal("0.02"), alias="RISK_MAX_PORTFOLIO_RISK")
    max_correlation: float = Field(default=0.7, alias="RISK_MAX_CORRELATION")
    stop_loss_percentage: Decimal = Field(default=Decimal("0.02"), alias="RISK_STOP_LOSS_PCT")
    take_profit_percentage: Decimal = Field(default=Decimal("0.06"), alias="RISK_TAKE_PROFIT_PCT")
    max_daily_trades: int = Field(default=10, alias="RISK_MAX_DAILY_TRADES")
    min_trade_amount: Decimal = Field(default=Decimal("100"), alias="RISK_MIN_TRADE_AMOUNT")
    max_trade_amount: Decimal = Field(default=Decimal("10000"), alias="RISK_MAX_TRADE_AMOUNT")
    drawdown_limit: Decimal = Field(default=Decimal("0.15"), alias="RISK_DRAWDOWN_LIMIT")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

    @field_validator('max_portfolio_risk', 'max_position_size', 'drawdown_limit')
    @classmethod
    def validate_percentages(cls, v):
        """Validate percentage values are between 0 and 1."""
        if not (0 < v <= 1):
            raise ValueError("Percentage values must be between 0 and 1")
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", alias="LOG_LEVEL")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        alias="LOG_FORMAT"
    )
    file_path: str = Field(default="data/logs/trading_system.log", alias="LOG_FILE_PATH")
    max_file_size: int = Field(default=10485760, alias="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")
    enable_console: bool = Field(default=True, alias="LOG_ENABLE_CONSOLE")
    enable_file: bool = Field(default=True, alias="LOG_ENABLE_FILE")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()


class SchedulerConfig(BaseSettings):
    """Scheduler configuration."""

    # Market data collection intervals
    market_data_interval: int = Field(default=60, alias="SCHEDULER_MARKET_DATA_INTERVAL")  # seconds
    finviz_scan_interval: int = Field(default=3600, alias="SCHEDULER_FINVIZ_SCAN_INTERVAL")  # seconds

    # Strategy execution intervals
    strategy_execution_interval: int = Field(default=300, alias="SCHEDULER_STRATEGY_INTERVAL")  # seconds
    risk_check_interval: int = Field(default=60, alias="SCHEDULER_RISK_CHECK_INTERVAL")  # seconds

    # Portfolio monitoring
    portfolio_sync_interval: int = Field(default=300, alias="SCHEDULER_PORTFOLIO_SYNC_INTERVAL")  # seconds
    health_check_interval: int = Field(default=60, alias="SCHEDULER_HEALTH_CHECK_INTERVAL")  # seconds

    # Trading hours
    trading_start_time: str = Field(default="09:30", alias="TRADING_START_TIME")
    trading_end_time: str = Field(default="16:00", alias="TRADING_END_TIME")
    timezone: str = Field(default="America/New_York", alias="TRADING_TIMEZONE")

    # Weekend and holiday trading
    trade_weekends: bool = Field(default=False, alias="TRADE_WEEKENDS")
    trade_holidays: bool = Field(default=False, alias="TRADE_HOLIDAYS")

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )


class DataConfig(BaseSettings):
    """Data management configuration."""

    parquet_path: str = Field(default="data/parquet", alias="DATA_PARQUET_PATH")
    retention_days: int = Field(default=365, alias="DATA_RETENTION_DAYS")
    compression: str = Field(default="snappy", alias="DATA_COMPRESSION")
    batch_size: int = Field(default=1000, alias="DATA_BATCH_SIZE")

    # Data sources priority
    primary_data_source: str = Field(default="twelvedata", alias="PRIMARY_DATA_SOURCE")
    fallback_data_sources: List[str] = Field(
        default=["alpaca"],
        alias="FALLBACK_DATA_SOURCES"
    )

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False
    )

    @field_validator('fallback_data_sources', mode='before')
    @classmethod
    def parse_data_sources(cls, v):
        """Parse comma-separated data sources list."""
        if isinstance(v, str):
            return [source.strip() for source in v.split(',') if source.strip()]
        return v


class Config(BaseSettings):
    """Main configuration class."""

    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        return DatabaseConfig(**{})

    @property
    def redis(self) -> RedisConfig:
        """Get Redis configuration."""
        return RedisConfig(**{})

    @property
    def alpaca(self) -> AlpacaConfig:
        """Get Alpaca configuration."""
        return AlpacaConfig(**{})

    @property
    def twelvedata(self) -> TwelveDataConfig:
        """Get TwelveData configuration."""
        return TwelveDataConfig(**{})

    @property
    def finviz(self) -> FinVizConfig:
        """Get FinViz configuration."""
        return FinVizConfig(**{})

    @property
    def notifications(self) -> NotificationConfig:
        """Get notification configuration."""
        return NotificationConfig(**{})

    @property
    def risk(self) -> RiskConfig:
        """Get risk configuration."""
        return RiskConfig(**{})

    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration."""
        return LoggingConfig(**{})

    @property
    def scheduler(self) -> SchedulerConfig:
        """Get scheduler configuration."""
        return SchedulerConfig(**{})

    @property
    def data(self) -> DataConfig:
        """Get data configuration."""
        return DataConfig(**{})

    @property
    def jwt(self) -> JWTConfig:
        """Get JWT configuration."""
        return JWTConfig(**{})

    # Service-specific settings
    service_name: str = Field(default="trading_system", alias="SERVICE_NAME")
    service_port: int = Field(default=8000, alias="SERVICE_PORT")

    # API rate limiting
    api_rate_limit_enabled: bool = Field(default=True, alias="API_RATE_LIMIT_ENABLED")
    api_rate_limit_requests: int = Field(default=100, alias="API_RATE_LIMIT_REQUESTS")
    api_rate_limit_period: int = Field(default=60, alias="API_RATE_LIMIT_PERIOD")

    # Monitoring and health checks
    health_check_enabled: bool = Field(default=True, alias="HEALTH_CHECK_ENABLED")
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "testing", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Invalid environment. Must be one of: {valid_envs}")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config() -> Config:
    """Reload configuration from environment."""
    global config
    config = Config()
    return config
