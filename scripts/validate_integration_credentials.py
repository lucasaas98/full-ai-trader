#!/usr/bin/env python3
"""
Integration Test Credential Validation Script

This script validates that integration test credentials are different from
production credentials and safe for testing. It checks:
- Alpaca API credentials are different
- Database names indicate testing
- Redis configuration is separate
- No production data can be accidentally accessed
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CredentialValidator:
    """Validates integration test credentials against production credentials."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.prod_env_file = self.project_root / ".env"
        self.integration_env_file = self.project_root / ".env.integration"

        self.validation_rules = {
            "ALPACA_API_KEY": self._validate_alpaca_api_key,
            "ALPACA_SECRET_KEY": self._validate_alpaca_secret_key,
            "DB_NAME": self._validate_database_name,
            "DB_USER": self._validate_database_user,
            "DB_PASSWORD": self._validate_database_password,
            "DB_PORT": self._validate_database_port,
            "REDIS_PORT": self._validate_redis_port,
            "REDIS_PASSWORD": self._validate_redis_password,
            "JWT_SECRET_KEY": self._validate_jwt_secret,
            "ENCRYPTION_KEY": self._validate_encryption_key,
        }

    def validate_all(self) -> Tuple[bool, List[str]]:
        """Validate all credentials and return result with error messages."""
        logger.info("🔍 Starting comprehensive credential validation...")

        errors = []

        # Check if files exist
        if not self.prod_env_file.exists():
            errors.append(f"Production .env file not found: {self.prod_env_file}")

        if not self.integration_env_file.exists():
            errors.append(
                f"Integration .env.integration file not found: {self.integration_env_file}"
            )
            return False, errors

        # Load environment variables
        prod_env = self._load_env_file(self.prod_env_file)
        integration_env = self._load_env_file(self.integration_env_file)

        if not prod_env:
            errors.append("Failed to load production environment variables")

        if not integration_env:
            errors.append("Failed to load integration environment variables")
            return False, errors

        # Validate each credential type
        for key, validator in self.validation_rules.items():
            try:
                is_valid, error_msg = validator(
                    prod_env.get(key), integration_env.get(key)
                )

                if not is_valid:
                    errors.append(f"{key}: {error_msg}")
                else:
                    logger.debug(f"✅ {key} validation passed")

            except Exception as e:
                errors.append(f"{key}: Validation error - {str(e)}")

        # Additional safety checks
        safety_errors = self._perform_safety_checks(prod_env, integration_env)
        errors.extend(safety_errors)

        success = len(errors) == 0

        if success:
            logger.info("✅ All credential validations passed!")
        else:
            logger.error(f"❌ {len(errors)} validation errors found")

        return success, errors

    def _load_env_file(self, file_path: Path) -> Dict[str, str]:
        """Load environment variables from a .env file."""
        env_vars = {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Parse key=value pairs
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        env_vars[key] = value
                    else:
                        logger.warning(
                            f"Invalid line format in {file_path}:{line_num}: {line}"
                        )

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {}

        logger.debug(f"Loaded {len(env_vars)} variables from {file_path}")
        return env_vars

    def _validate_alpaca_api_key(
        self, prod_key: Optional[str], integration_key: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate Alpaca API key is different and safe for testing."""
        if not integration_key:
            return False, "Integration Alpaca API key is missing"

        if not prod_key:
            logger.warning("Production Alpaca API key not found - cannot compare")

        # Check if keys are the same
        if prod_key and prod_key == integration_key:
            return (
                False,
                "Integration Alpaca API key is identical to production key - DANGEROUS!",
            )

        # Check for test indicators
        test_indicators = ["test", "integration", "pkt", "paper", "demo", "sandbox"]
        has_test_indicator = any(
            indicator in integration_key.lower() for indicator in test_indicators
        )

        if not has_test_indicator:
            return (
                False,
                f"Integration Alpaca API key should contain test indicators like: {', '.join(test_indicators)}",
            )

        # Check it's not obviously a production key
        prod_indicators = ["prod", "live", "real"]
        has_prod_indicator = any(
            indicator in integration_key.lower() for indicator in prod_indicators
        )

        if has_prod_indicator:
            return (
                False,
                "Integration Alpaca API key contains production indicators - DANGEROUS!",
            )

        return True, "Valid"

    def _validate_alpaca_secret_key(
        self, prod_secret: Optional[str], integration_secret: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate Alpaca secret key is different."""
        if not integration_secret:
            return False, "Integration Alpaca secret key is missing"

        if prod_secret and prod_secret == integration_secret:
            return (
                False,
                "Integration Alpaca secret key is identical to production key - DANGEROUS!",
            )

        # Check for test indicators
        test_indicators = ["test", "integration", "paper", "demo", "sandbox"]
        has_test_indicator = any(
            indicator in integration_secret.lower() for indicator in test_indicators
        )

        if not has_test_indicator:
            return False, "Integration Alpaca secret key should contain test indicators"

        return True, "Valid"

    def _validate_database_name(
        self, prod_name: Optional[str], integration_name: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate database name indicates testing."""
        if not integration_name:
            return False, "Integration database name is missing"

        if prod_name and prod_name == integration_name:
            return (
                False,
                "Integration database name is identical to production - DANGEROUS!",
            )

        # Check for test indicators
        test_indicators = ["test", "integration", "dev", "staging"]
        has_test_indicator = any(
            indicator in integration_name.lower() for indicator in test_indicators
        )

        if not has_test_indicator:
            return (
                False,
                f"Integration database name should contain test indicators: {', '.join(test_indicators)}",
            )

        return True, "Valid"

    def _validate_database_user(
        self, prod_user: Optional[str], integration_user: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate database user is appropriate for testing."""
        if not integration_user:
            return False, "Integration database user is missing"

        if (
            prod_user
            and prod_user == integration_user
            and "test" not in integration_user.lower()
        ):
            return (
                False,
                "Integration database user should be different from production or contain 'test'",
            )

        return True, "Valid"

    def _validate_database_password(
        self, prod_password: Optional[str], integration_password: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate database password is different."""
        if not integration_password:
            return False, "Integration database password is missing"

        if prod_password and prod_password == integration_password:
            return (
                False,
                "Integration database password is identical to production - DANGEROUS!",
            )

        return True, "Valid"

    def _validate_database_port(
        self, prod_port: Optional[str], integration_port: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate database port is different to avoid conflicts."""
        if not integration_port:
            return False, "Integration database port is missing"

        if prod_port and prod_port == integration_port:
            return (
                False,
                "Integration database port should be different from production to avoid conflicts",
            )

        try:
            port_num = int(integration_port)
            if port_num < 1024 or port_num > 65535:
                return False, f"Invalid port number: {port_num}"
        except ValueError:
            return False, f"Invalid port format: {integration_port}"

        return True, "Valid"

    def _validate_redis_port(
        self, prod_port: Optional[str], integration_port: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate Redis port is different."""
        if not integration_port:
            return False, "Integration Redis port is missing"

        if prod_port and prod_port == integration_port:
            return False, "Integration Redis port should be different from production"

        try:
            port_num = int(integration_port)
            if port_num < 1024 or port_num > 65535:
                return False, f"Invalid port number: {port_num}"
        except ValueError:
            return False, f"Invalid port format: {integration_port}"

        return True, "Valid"

    def _validate_redis_password(
        self, prod_password: Optional[str], integration_password: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate Redis password is different."""
        if not integration_password:
            return False, "Integration Redis password is missing"

        if prod_password and prod_password == integration_password:
            return (
                False,
                "Integration Redis password should be different from production",
            )

        return True, "Valid"

    def _validate_jwt_secret(
        self, prod_secret: Optional[str], integration_secret: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate JWT secret is different."""
        if not integration_secret:
            return False, "Integration JWT secret is missing"

        if prod_secret and prod_secret == integration_secret:
            return False, "Integration JWT secret should be different from production"

        if len(integration_secret) < 16:
            return False, "Integration JWT secret should be at least 16 characters long"

        return True, "Valid"

    def _validate_encryption_key(
        self, prod_key: Optional[str], integration_key: Optional[str]
    ) -> Tuple[bool, str]:
        """Validate encryption key is different."""
        if not integration_key:
            return False, "Integration encryption key is missing"

        if prod_key and prod_key == integration_key:
            return (
                False,
                "Integration encryption key should be different from production",
            )

        if len(integration_key) < 32:
            return (
                False,
                "Integration encryption key should be at least 32 characters long",
            )

        return True, "Valid"

    def _perform_safety_checks(
        self, prod_env: Dict[str, str], integration_env: Dict[str, str]
    ) -> List[str]:
        """Perform additional safety checks."""
        errors = []

        # Check environment setting
        integration_environment = integration_env.get("ENVIRONMENT", "")
        if (
            "test" not in integration_environment.lower()
            and "integration" not in integration_environment.lower()
        ):
            errors.append(
                "ENVIRONMENT should indicate testing (e.g., 'integration_test')"
            )

        # Check testing flag
        testing_flag = integration_env.get("TESTING", "").lower()
        if testing_flag != "true":
            errors.append("TESTING flag should be set to 'true' for integration tests")

        # Check trading settings
        dry_run = integration_env.get("TRADING_DRY_RUN", "").lower()
        if dry_run != "true":
            errors.append("TRADING_DRY_RUN should be 'true' for integration tests")

        paper_mode = integration_env.get("TRADING_PAPER_MODE", "").lower()
        if paper_mode != "true":
            errors.append("TRADING_PAPER_MODE should be 'true' for integration tests")

        # Check Alpaca URL
        alpaca_url = integration_env.get("ALPACA_BASE_URL", "")
        if "paper-api" not in alpaca_url:
            errors.append("ALPACA_BASE_URL should point to paper trading API")

        # Check AI testing mode
        ai_testing = integration_env.get("AI_TESTING_MODE", "").lower()
        if ai_testing != "true":
            errors.append("AI_TESTING_MODE should be 'true' to avoid API costs")

        return errors

    def create_integration_env_template(self):
        """Create a template .env.integration file with safe defaults."""
        template_path = self.project_root / ".env.integration.template"

        template_content = """# Integration Testing Environment Configuration
# This file contains SAFE configuration for integration tests
# DO NOT use production credentials here!

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================
ENVIRONMENT=integration_test
DEBUG=true
LOG_LEVEL=DEBUG
TESTING=true

# =============================================================================
# DATABASE CONFIGURATION (SEPARATE FROM PRODUCTION)
# =============================================================================
DB_HOST=localhost
DB_PORT=5434
DB_NAME=trading_system_integration_test
DB_USER=trader_integration
DB_PASSWORD=integration_test_password_2024
DATABASE_URL=postgresql://trader_integration:integration_test_password_2024@localhost:5434/trading_system_integration_test

# =============================================================================
# REDIS CONFIGURATION (SEPARATE FROM PRODUCTION)
# =============================================================================
REDIS_HOST=localhost
REDIS_PORT=6381
REDIS_PASSWORD=integration_redis_password_2024
REDIS_URL=redis://:integration_redis_password_2024@localhost:6381/0

# =============================================================================
# API KEYS - INTEGRATION TEST CREDENTIALS
# =============================================================================
# IMPORTANT: Use DIFFERENT credentials from production!
ALPACA_API_KEY=PKT_INTEGRATION_TEST_KEY_CHANGE_ME
ALPACA_SECRET_KEY=integration_test_secret_key_CHANGE_ME
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Market Data APIs - Test credentials
TWELVE_DATA_API_KEY=integration_test_twelve_data_key
FINVIZ_API_KEY=integration-test-finviz-key

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================
JWT_SECRET_KEY=integration_test_jwt_secret_key_2024_CHANGE_ME
ENCRYPTION_KEY=integration_test_encryption_key_32b_CHANGE_ME

# =============================================================================
# TRADING CONFIGURATION - SAFE FOR TESTING
# =============================================================================
TRADING_DRY_RUN=true
TRADING_PAPER_MODE=true
TRADING_MAX_ORDERS_PER_DAY=10
TRADING_MIN_ORDER_VALUE=50
TRADING_MAX_ORDER_VALUE=1000

# =============================================================================
# AI CONFIGURATION - AVOID COSTS
# =============================================================================
AI_TESTING_MODE=true
AI_USE_OLLAMA=true
AI_CONSENSUS_ENABLED=false
AI_MAX_COST_PER_DAY=0.50

# =============================================================================
# INTEGRATION TEST SPECIFIC SETTINGS
# =============================================================================
INTEGRATION_TEST_MODE=true
USE_MOCK_DATA_COLLECTOR=true
USE_HISTORICAL_DATA=true
MOCK_EXTERNAL_APIS=true
"""

        try:
            with open(template_path, "w") as f:
                f.write(template_content)
            logger.info(f"✅ Created integration environment template: {template_path}")
            logger.info(
                "⚠️  Please copy this to .env.integration and customize the credentials"
            )
        except Exception as e:
            logger.error(f"Failed to create template: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate integration test credentials"
    )

    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create .env.integration template file",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--fix", action="store_true", help="Attempt to fix common issues automatically"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validator = CredentialValidator()

    if args.create_template:
        validator.create_integration_env_template()
        return

    # Validate credentials
    success, errors = validator.validate_all()

    if not success:
        print("\n❌ CREDENTIAL VALIDATION FAILED")
        print("=" * 50)
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        print("=" * 50)
        print("\nTo fix these issues:")
        print(
            "1. Run: python scripts/validate_integration_credentials.py --create-template"
        )
        print("2. Copy .env.integration.template to .env.integration")
        print("3. Update all credentials marked with CHANGE_ME")
        print("4. Ensure all credentials are different from production")
        print("5. Run this script again to validate")

        sys.exit(1)
    else:
        print("\n✅ ALL CREDENTIAL VALIDATIONS PASSED!")
        print("=" * 50)
        print("Your integration test environment is safe to use.")
        print("All credentials are properly separated from production.")

        sys.exit(0)


if __name__ == "__main__":
    main()
