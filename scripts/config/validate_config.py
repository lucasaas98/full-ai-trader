#!/usr/bin/env python3
"""
Configuration Validation Script for AI Trading System

This script validates environment configurations across different environments
(development, staging, production) to ensure all required settings are present
and properly formatted.

Usage:
    python validate_config.py --env development
    python validate_config.py --env staging --strict
    python validate_config.py --env production --strict --check-secrets
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ValidationLevel(Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationRule:
    """Defines a configuration validation rule"""
    key: str
    required: bool = True
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    allowed_values: Optional[List[str]] = None
    env_specific: Optional[List[Environment]] = None
    validation_level: ValidationLevel = ValidationLevel.ERROR
    description: str = ""


@dataclass
class ValidationResult:
    """Results of configuration validation"""
    key: str
    status: str  # "pass", "warning", "error", "critical"
    message: str
    level: ValidationLevel


class ConfigValidator:
    """Configuration validator for AI Trading System"""

    def __init__(self, environment: Environment, strict_mode: bool = False):
        self.environment = environment
        self.strict_mode = strict_mode
        self.config = {}
        self.results: List[ValidationResult] = []

        # Define validation rules
        self.rules = self._define_validation_rules()

    def _define_validation_rules(self) -> List[ValidationRule]:
        """Define all validation rules for configuration"""
        return [
            # Environment Settings
            ValidationRule(
                key="ENVIRONMENT",
                required=True,
                allowed_values=["development", "staging", "production"],
                description="Environment name must be valid"
            ),
            ValidationRule(
                key="DEBUG",
                required=True,
                allowed_values=["true", "false"],
                description="Debug flag must be boolean string"
            ),
            ValidationRule(
                key="LOG_LEVEL",
                required=True,
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                description="Log level must be valid Python logging level"
            ),

            # Database Configuration
            ValidationRule(
                key="DB_HOST",
                required=True,
                min_length=1,
                description="Database host must be specified"
            ),
            ValidationRule(
                key="DB_PORT",
                required=True,
                pattern=r"^\d{1,5}$",
                description="Database port must be valid port number"
            ),
            ValidationRule(
                key="DB_NAME",
                required=True,
                min_length=1,
                pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$",
                description="Database name must be valid identifier"
            ),
            ValidationRule(
                key="DB_USER",
                required=True,
                min_length=1,
                description="Database user must be specified"
            ),
            ValidationRule(
                key="DB_PASSWORD",
                required=True,
                min_length=8,
                env_specific=[Environment.STAGING, Environment.PRODUCTION],
                validation_level=ValidationLevel.CRITICAL,
                description="Database password must be at least 8 characters"
            ),

            # Redis Configuration
            ValidationRule(
                key="REDIS_HOST",
                required=True,
                min_length=1,
                description="Redis host must be specified"
            ),
            ValidationRule(
                key="REDIS_PORT",
                required=True,
                pattern=r"^\d{1,5}$",
                description="Redis port must be valid port number"
            ),
            ValidationRule(
                key="REDIS_PASSWORD",
                required=True,
                min_length=8,
                env_specific=[Environment.STAGING, Environment.PRODUCTION],
                validation_level=ValidationLevel.CRITICAL,
                description="Redis password must be at least 8 characters"
            ),

            # API Keys
            ValidationRule(
                key="ALPACA_API_KEY",
                required=True,
                min_length=20,
                validation_level=ValidationLevel.CRITICAL,
                description="Alpaca API key is required for trading"
            ),
            ValidationRule(
                key="ALPACA_SECRET_KEY",
                required=True,
                min_length=20,
                validation_level=ValidationLevel.CRITICAL,
                description="Alpaca secret key is required for trading"
            ),
            ValidationRule(
                key="TWELVE_DATA_API_KEY",
                required=True,
                min_length=10,
                description="Twelve Data API key is required for market data"
            ),

            # Security Configuration
            ValidationRule(
                key="JWT_SECRET_KEY",
                required=True,
                min_length=32,
                validation_level=ValidationLevel.CRITICAL,
                description="JWT secret key must be at least 32 characters"
            ),
            ValidationRule(
                key="ENCRYPTION_KEY",
                required=True,
                min_length=32,
                max_length=32,
                validation_level=ValidationLevel.CRITICAL,
                description="Encryption key must be exactly 32 characters for AES-256"
            ),
            ValidationRule(
                key="BACKUP_ENCRYPTION_KEY",
                required=True,
                min_length=32,
                max_length=32,
                validation_level=ValidationLevel.CRITICAL,
                description="Backup encryption key must be exactly 32 characters"
            ),

            # Trading Configuration
            ValidationRule(
                key="RISK_MAX_POSITION_SIZE",
                required=True,
                pattern=r"^0\.\d+$",
                description="Risk max position size must be decimal between 0 and 1"
            ),
            ValidationRule(
                key="RISK_MAX_PORTFOLIO_RISK",
                required=True,
                pattern=r"^0\.\d+$",
                description="Risk max portfolio risk must be decimal between 0 and 1"
            ),
            ValidationRule(
                key="TRADING_DRY_RUN",
                required=True,
                allowed_values=["true", "false"],
                description="Trading dry run must be boolean string"
            ),

            # Production-specific validations
            ValidationRule(
                key="SSL_ENABLED",
                required=True,
                allowed_values=["true"],
                env_specific=[Environment.PRODUCTION],
                validation_level=ValidationLevel.CRITICAL,
                description="SSL must be enabled in production"
            ),
            ValidationRule(
                key="TRADING_PAPER_MODE",
                required=True,
                allowed_values=["false"],
                env_specific=[Environment.PRODUCTION],
                validation_level=ValidationLevel.CRITICAL,
                description="Paper mode must be disabled in production for live trading"
            ),
        ]

    def load_config(self, config_path: Path) -> None:
        """Load configuration from environment file"""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Handle quoted values
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Handle environment variable substitution
                    if value.startswith('${') and value.endswith('}'):
                        env_var = value[2:-1]
                        value = os.getenv(env_var, value)

                    self.config[key] = value
                else:
                    logger.warning(f"Invalid line format at line {line_num}: {line}")

    def validate_rule(self, rule: ValidationRule) -> ValidationResult:
        """Validate a single configuration rule"""
        key = rule.key
        value = self.config.get(key)

        # Check if rule applies to current environment
        if rule.env_specific and self.environment not in rule.env_specific:
            return ValidationResult(
                key=key,
                status="skip",
                message=f"Rule not applicable to {self.environment.value} environment",
                level=ValidationLevel.WARNING
            )

        # Check if required key is missing
        if rule.required and not value:
            return ValidationResult(
                key=key,
                status="error" if rule.validation_level != ValidationLevel.CRITICAL else "critical",
                message=f"Required configuration key '{key}' is missing",
                level=rule.validation_level
            )

        # Skip validation if key is not required and not present
        if not rule.required and not value:
            return ValidationResult(
                key=key,
                status="pass",
                message=f"Optional key '{key}' not set",
                level=ValidationLevel.WARNING
            )

        # Validate pattern
        if rule.pattern and value is not None and not re.match(rule.pattern, value):
            return ValidationResult(
                key=key,
                status="error",
                message=f"Value '{value}' does not match required pattern: {rule.pattern}",
                level=rule.validation_level
            )

        # Validate length
        if rule.min_length and value is not None and len(value) < rule.min_length:
            return ValidationResult(
                key=key,
                status="error",
                message=f"Value length {len(value)} is less than minimum {rule.min_length}",
                level=rule.validation_level
            )

        if rule.max_length and value is not None and len(value) > rule.max_length:
            return ValidationResult(
                key=key,
                status="error",
                message=f"Value length {len(value)} exceeds maximum {rule.max_length}",
                level=rule.validation_level
            )

        # Validate allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            return ValidationResult(
                key=key,
                status="error",
                message=f"Value '{value}' not in allowed values: {rule.allowed_values}",
                level=rule.validation_level
            )

        # All validations passed
        return ValidationResult(
            key=key,
            status="pass",
            message=f"Configuration valid: {rule.description}",
            level=ValidationLevel.WARNING
        )

    def validate_custom_rules(self) -> List[ValidationResult]:
        """Validate custom business logic rules"""
        custom_results = []

        # Check trading safety in production
        if self.environment == Environment.PRODUCTION:
            trading_dry_run = self.config.get("TRADING_DRY_RUN", "true").lower()
            trading_paper_mode = self.config.get("TRADING_PAPER_MODE", "true").lower()

            if trading_dry_run == "false" and trading_paper_mode == "false":
                custom_results.append(ValidationResult(
                    key="TRADING_SAFETY",
                    status="critical",
                    message="CRITICAL: Live trading enabled in production! Ensure this is intentional.",
                    level=ValidationLevel.CRITICAL
                ))

        # Validate risk parameters make sense
        try:
            max_position = float(self.config.get("RISK_MAX_POSITION_SIZE", "0"))
            max_portfolio = float(self.config.get("RISK_MAX_PORTFOLIO_RISK", "0"))

            if max_position > 0.2:  # 20% position size warning
                custom_results.append(ValidationResult(
                    key="RISK_POSITION_SIZE",
                    status="warning",
                    message=f"Position size {max_position} is quite high (>20%)",
                    level=ValidationLevel.WARNING
                ))

            if max_portfolio > 0.1:  # 10% portfolio risk warning
                custom_results.append(ValidationResult(
                    key="RISK_PORTFOLIO_RISK",
                    status="warning",
                    message=f"Portfolio risk {max_portfolio} is quite high (>10%)",
                    level=ValidationLevel.WARNING
                ))

        except (ValueError, TypeError):
            custom_results.append(ValidationResult(
                key="RISK_PARAMETERS",
                status="error",
                message="Risk parameters must be valid decimal numbers",
                level=ValidationLevel.ERROR
            ))

        # Validate URL formats
        url_keys = ["ALPACA_BASE_URL", "ALPACA_DATA_URL", "GRAFANA_ROOT_URL"]
        url_pattern = re.compile(r'^https?://[^\s/$.?#].[^\s]*$')

        for url_key in url_keys:
            url_value = self.config.get(url_key)
            if url_value and not url_pattern.match(url_value):
                custom_results.append(ValidationResult(
                    key=url_key,
                    status="error",
                    message=f"Invalid URL format: {url_value}",
                    level=ValidationLevel.ERROR
                ))

        # Validate port numbers
        port_keys = [
            "DB_PORT", "REDIS_PORT", "DATA_COLLECTOR_PORT", "STRATEGY_ENGINE_PORT",
            "RISK_MANAGER_PORT", "TRADE_EXECUTOR_PORT", "SCHEDULER_PORT"
        ]

        for port_key in port_keys:
            port_value = self.config.get(port_key)
            if port_value:
                try:
                    port_num = int(port_value)
                    if not (1 <= port_num <= 65535):
                        custom_results.append(ValidationResult(
                            key=port_key,
                            status="error",
                            message=f"Port {port_num} is not in valid range (1-65535)",
                            level=ValidationLevel.ERROR
                        ))
                except ValueError:
                    custom_results.append(ValidationResult(
                        key=port_key,
                        status="error",
                        message=f"Port value '{port_value}' is not a valid integer",
                        level=ValidationLevel.ERROR
                    ))

        # Check for default/weak passwords in non-development environments
        if self.environment != Environment.DEVELOPMENT:
            weak_passwords = [
                "password", "123456", "admin", "root", "changeme", "default",
                "dev_password", "test_password"
            ]

            password_keys = ["DB_PASSWORD", "REDIS_PASSWORD", "GRAFANA_PASSWORD"]
            for pwd_key in password_keys:
                pwd_value = self.config.get(pwd_key, "").lower()
                if any(weak in pwd_value for weak in weak_passwords):
                    custom_results.append(ValidationResult(
                        key=pwd_key,
                        status="critical",
                        message=f"Weak/default password detected in {pwd_key}",
                        level=ValidationLevel.CRITICAL
                    ))

        # Validate encryption key formats (should be base64 or hex)
        encryption_keys = ["ENCRYPTION_KEY", "BACKUP_ENCRYPTION_KEY", "EXPORT_ENCRYPTION_KEY"]
        for enc_key in encryption_keys:
            enc_value = self.config.get(enc_key)
            if enc_value and len(enc_value) == 32:
                # Check if it's a valid hex or base64-like string
                if not re.match(r'^[A-Fa-f0-9]{32}$|^[A-Za-z0-9+/]{32}$', enc_value):
                    custom_results.append(ValidationResult(
                        key=enc_key,
                        status="warning",
                        message="Encryption key format may be invalid",
                        level=ValidationLevel.WARNING
                    ))

        return custom_results

    def validate_dependencies(self) -> List[ValidationResult]:
        """Validate interdependent configuration values"""
        dep_results = []

        # Check SSL configuration consistency
        ssl_enabled = self.config.get("SSL_ENABLED", "false").lower() == "true"
        if ssl_enabled:
            ssl_cert = self.config.get("SSL_CERT_PATH")
            ssl_key = self.config.get("SSL_KEY_PATH")

            if not ssl_cert or not ssl_key:
                dep_results.append(ValidationResult(
                    key="SSL_CONFIG",
                    status="error",
                    message="SSL enabled but certificate paths not configured",
                    level=ValidationLevel.ERROR
                ))

        # Check backup configuration
        backup_enabled = self.config.get("BACKUP_ENCRYPTION_ENABLED", "false").lower() == "true"
        if backup_enabled and not self.config.get("BACKUP_ENCRYPTION_KEY"):
            dep_results.append(ValidationResult(
                key="BACKUP_CONFIG",
                status="error",
                message="Backup encryption enabled but encryption key not set",
                level=ValidationLevel.ERROR
            ))

        # Check S3 backup configuration
        s3_bucket = self.config.get("S3_BACKUP_BUCKET")
        if s3_bucket:
            aws_key = self.config.get("AWS_ACCESS_KEY_ID")
            aws_secret = self.config.get("AWS_SECRET_ACCESS_KEY")

            if not aws_key or not aws_secret:
                dep_results.append(ValidationResult(
                    key="S3_BACKUP_CONFIG",
                    status="error",
                    message="S3 bucket configured but AWS credentials missing",
                    level=ValidationLevel.ERROR
                ))

        # Check trading mode consistency
        if self.environment == Environment.PRODUCTION:
            dry_run = self.config.get("TRADING_DRY_RUN", "true").lower() == "true"
            paper_mode = self.config.get("TRADING_PAPER_MODE", "true").lower() == "true"
            alpaca_url = self.config.get("ALPACA_BASE_URL", "")

            if not dry_run and not paper_mode and "paper" in alpaca_url:
                dep_results.append(ValidationResult(
                    key="TRADING_MODE_MISMATCH",
                    status="critical",
                    message="Live trading mode but using paper trading URL",
                    level=ValidationLevel.CRITICAL
                ))

        return dep_results

    def check_secrets_security(self) -> List[ValidationResult]:
        """Check for potential security issues with secrets"""
        security_results = []

        # Check for placeholder values
        placeholder_patterns = [
            r"your_.*_key",
            r"your_.*_password",
            r"your_.*_token",
            r"change_me",
            r"replace_with",
            r"placeholder"
        ]

        for key, value in self.config.items():
            if any(re.search(pattern, value.lower()) for pattern in placeholder_patterns):
                security_results.append(ValidationResult(
                    key=key,
                    status="error",
                    message=f"Placeholder value detected: {value}",
                    level=ValidationLevel.ERROR
                ))

        # Check for hardcoded secrets in non-development environments
        if self.environment != Environment.DEVELOPMENT:
            secret_keys = [
                "PASSWORD", "SECRET", "KEY", "TOKEN", "API_KEY"
            ]

            for key, value in self.config.items():
                if any(secret in key.upper() for secret in secret_keys):
                    if len(value) < 16:
                        security_results.append(ValidationResult(
                            key=key,
                            status="warning",
                            message=f"Secret appears to be too short: {len(value)} characters",
                            level=ValidationLevel.WARNING
                        ))

        return security_results

    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validations and return results"""
        all_results = []

        # Validate individual rules
        for rule in self.rules:
            result = self.validate_rule(rule)
            all_results.append(result)

        # Validate custom business rules
        all_results.extend(self.validate_custom_rules())

        # Validate dependencies
        all_results.extend(self.validate_dependencies())

        # Check secrets security if in strict mode
        if self.strict_mode:
            all_results.extend(self.check_secrets_security())

        self.results = all_results

        # Determine overall success
        has_critical = any(r.status == "critical" for r in all_results)
        has_error = any(r.status == "error" for r in all_results)

        success = not (has_critical or has_error)

        return success, all_results

    def print_results(self) -> None:
        """Print validation results in a formatted way"""
        if not self.results:
            logger.info("No validation results to display")
            return

        # Group results by status
        critical_results = [r for r in self.results if r.status == "critical"]
        error_results = [r for r in self.results if r.status == "error"]
        warning_results = [r for r in self.results if r.status == "warning"]
        pass_results = [r for r in self.results if r.status == "pass"]

        print(f"\n🔍 Configuration Validation Results for {self.environment.value.upper()}")
        print("=" * 70)

        if critical_results:
            print(f"\n🚨 CRITICAL ISSUES ({len(critical_results)}):")
            for result in critical_results:
                print(f"  ❌ {result.key}: {result.message}")

        if error_results:
            print(f"\n❌ ERRORS ({len(error_results)}):")
            for result in error_results:
                print(f"  ❌ {result.key}: {result.message}")

        if warning_results:
            print(f"\n⚠️  WARNINGS ({len(warning_results)}):")
            for result in warning_results:
                print(f"  ⚠️  {result.key}: {result.message}")

        if pass_results and not self.strict_mode:
            print(f"\n✅ PASSED ({len(pass_results)})")
        elif self.strict_mode:
            print(f"\n✅ PASSED VALIDATIONS: {len(pass_results)}")

        # Summary
        total = len(self.results)
        passed = len(pass_results)
        print(f"\n📊 Summary: {passed}/{total} validations passed")

        if critical_results or error_results:
            print("❌ Configuration validation FAILED")
        else:
            print("✅ Configuration validation PASSED")

    def export_results(self, output_path: Path) -> None:
        """Export validation results to JSON file"""
        results_data = {
            "environment": self.environment.value,
            "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
            "strict_mode": self.strict_mode,
            "summary": {
                "total": len(self.results),
                "passed": len([r for r in self.results if r.status == "pass"]),
                "warnings": len([r for r in self.results if r.status == "warning"]),
                "errors": len([r for r in self.results if r.status == "error"]),
                "critical": len([r for r in self.results if r.status == "critical"])
            },
            "results": [
                {
                    "key": r.key,
                    "status": r.status,
                    "message": r.message,
                    "level": r.level.value
                }
                for r in self.results
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"Validation results exported to: {output_path}")


def main():
    """Main entry point for configuration validation"""
    parser = argparse.ArgumentParser(
        description="Validate AI Trading System configuration"
    )
    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        required=True,
        help="Environment to validate"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation mode"
    )
    parser.add_argument(
        "--check-secrets",
        action="store_true",
        help="Perform additional security checks on secrets"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Custom path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Export results to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except errors"
    )

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Determine environment and config path
    environment = Environment(args.env)

    if args.config_path:
        config_path = Path(args.config_path)
    else:
        # Default config path based on environment
        config_path = Path(f"config/environments/.env.{args.env}")

    try:
        # Initialize validator
        validator = ConfigValidator(
            environment=environment,
            strict_mode=args.strict or args.check_secrets
        )

        # Load configuration
        validator.load_config(config_path)

        # Run validation
        success, results = validator.validate_all()

        # Print results unless quiet mode
        if not args.quiet:
            validator.print_results()

        # Export results if requested
        if args.output:
            validator.export_results(Path(args.output))

        # Exit with appropriate code
        if not success:
            logger.error("Configuration validation failed!")
            sys.exit(1)
        else:
            if not args.quiet:
                logger.info("Configuration validation passed!")
            sys.exit(0)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
