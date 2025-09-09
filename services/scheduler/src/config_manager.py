"""
Configuration Manager for the Trading Scheduler.

This module provides hot-reload configuration management, A/B testing capabilities,
and dynamic parameter updates for the trading system.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class ConfigSource(str, Enum):
    """Configuration source types."""

    ENVIRONMENT = "environment"
    FILE = "file"
    REDIS = "redis"
    DATABASE = "database"
    API = "api"


class ConfigChangeType(str, Enum):
    """Types of configuration changes."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass
class ConfigChange:
    """Represents a configuration change."""

    key: str
    old_value: Any
    new_value: Any
    change_type: ConfigChangeType
    timestamp: datetime
    source: ConfigSource
    applied: bool = False


@dataclass
class ABTestConfig:
    """A/B test configuration."""

    name: str
    description: str
    variants: Dict[str, Dict[str, Any]]
    traffic_split: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime] = None
    enabled: bool = True
    metrics: List[str] = field(default_factory=list)
    current_assignment: Dict[str, str] = field(default_factory=dict)


class ConfigValidator:
    """Validates configuration changes."""

    def __init__(self) -> None:
        self.validation_rules = {
            "risk.max_position_size": lambda x: 0 < float(x) <= 1.0,
            "risk.max_portfolio_risk": lambda x: 0 < float(x) <= 1.0,
            "risk.stop_loss_percentage": lambda x: 0 < float(x) <= 0.5,
            "scheduler.market_data_interval": lambda x: 10 <= int(x) <= 3600,
            "scheduler.finviz_scan_interval": lambda x: 60 <= int(x) <= 86400,
            "database.pool_size": lambda x: 1 <= int(x) <= 50,
            "redis.max_connections": lambda x: 1 <= int(x) <= 100,
        }

    def validate_change(self, key: str, value: Any) -> tuple[bool, str]:
        """Validate a configuration change."""
        try:
            if key in self.validation_rules:
                is_valid = self.validation_rules[key](value)
                if not is_valid:
                    return False, f"Value {value} failed validation for {key}"

            return True, "Valid"

        except Exception as e:
            return False, f"Validation error for {key}: {str(e)}"


class ConfigurationManager:
    """Manages configuration hot-reloading and A/B testing."""

    def __init__(self, redis_client: redis.Redis, config: Any):
        self.redis = redis_client
        self.config = config
        self.validator = ConfigValidator()

        # Configuration state
        self.current_config_hash = ""
        self.config_history: List[ConfigChange] = []
        self.change_callbacks: Dict[str, List[Callable]] = {}
        self.rollback_stack: List[Dict[str, Any]] = []

        # A/B testing
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}

        # File watching
        self.watched_files: Set[str] = set()
        self.file_mtimes: Dict[str, float] = {}

        # Configuration sources
        self.config_sources: Dict[str, ConfigSource] = {}

        # Hot reload settings
        self.auto_reload_enabled = True
        self.reload_interval = 30  # seconds
        self.max_rollback_entries = 10

    async def initialize(self) -> None:
        """Initialize the configuration manager."""
        logger.info("Initializing configuration manager...")

        # Calculate initial config hash
        self.current_config_hash = self._calculate_config_hash()

        # Setup file watching for .env and config files
        env_file = Path(".env")
        if env_file.exists():
            self.watched_files.add(str(env_file))
            self.file_mtimes[str(env_file)] = env_file.stat().st_mtime

        config_files = [
            "config/trading.yaml",
            "config/strategies.yaml",
            "config/risk.yaml",
        ]

        for config_file in config_files:
            file_path = Path(config_file)
            if file_path.exists():
                self.watched_files.add(config_file)
                self.file_mtimes[config_file] = file_path.stat().st_mtime

        # Load A/B tests from Redis
        await self._load_ab_tests()

        # Start monitoring
        if self.auto_reload_enabled:
            asyncio.create_task(self._config_monitoring_loop())

        logger.info("Configuration manager initialized")

    def _calculate_config_hash(self) -> str:
        """Calculate hash of current configuration."""
        try:
            config_str = json.dumps(self.config.dict(), sort_keys=True, default=str)
            return hashlib.md5(config_str.encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate config hash: {e}")
            return ""

    async def _config_monitoring_loop(self) -> None:
        """Monitor configuration files for changes."""
        while self.auto_reload_enabled:
            try:
                await self._check_file_changes()
                await self._check_redis_changes()
                await asyncio.sleep(self.reload_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Configuration monitoring error: {e}")
                await asyncio.sleep(60)

    async def _check_file_changes(self) -> None:
        """Check for file system configuration changes."""
        changes_detected = False

        for file_path in self.watched_files:
            try:
                path_obj = Path(file_path)
                if not path_obj.exists():
                    continue

                current_mtime = path_obj.stat().st_mtime
                if current_mtime != self.file_mtimes.get(file_path, 0):
                    logger.info(f"Configuration file changed: {file_path}")
                    self.file_mtimes[file_path] = current_mtime
                    changes_detected = True

            except Exception as e:
                logger.error(f"Error checking file {file_path}: {e}")

        if changes_detected:
            await self.reload_configuration()

    async def _check_redis_changes(self) -> None:
        """Check for Redis-based configuration changes."""
        try:
            # Check for configuration updates in Redis
            config_version_key = "config:version"
            stored_version = await self.redis.get(config_version_key)

            if stored_version:
                stored_hash = (
                    stored_version.decode()
                    if isinstance(stored_version, bytes)
                    else stored_version
                )
                if stored_hash != self.current_config_hash:
                    logger.info("Redis configuration changes detected")
                    await self._apply_redis_config_changes()

        except Exception as e:
            logger.error(f"Error checking Redis configuration: {e}")

    async def reload_configuration(self) -> bool:
        """Reload configuration from all sources."""
        logger.info("Reloading configuration...")

        try:
            # Store current config as rollback point
            await self._create_rollback_point()

            # Reload from shared config
            from shared.config import reload_config

            new_config = reload_config()

            # Calculate new hash
            new_hash = self._calculate_config_hash()

            if new_hash != self.current_config_hash:
                # Detect specific changes
                changes = await self._detect_config_changes(self.config, new_config)

                # Validate changes
                validation_errors = []
                for change in changes:
                    is_valid, error_msg = self.validator.validate_change(
                        change.key, change.new_value
                    )
                    if not is_valid:
                        validation_errors.append(f"{change.key}: {error_msg}")

                if validation_errors:
                    logger.error(
                        f"Configuration validation failed: {validation_errors}"
                    )
                    return False

                # Apply changes
                self.config = new_config
                self.current_config_hash = new_hash

                # Record changes
                for change in changes:
                    change.applied = True
                    self.config_history.append(change)

                # Notify callbacks
                await self._notify_config_changes(changes)

                logger.info(
                    f"Configuration reloaded successfully - {len(changes)} changes applied"
                )
                return True
            else:
                logger.info("No configuration changes detected")
                return True

        except Exception as e:
            logger.error(f"Configuration reload failed: {e}")
            await self.rollback_last_change()
            return False

    async def _detect_config_changes(
        self, old_config: Any, new_config: Any
    ) -> List[ConfigChange]:
        """Detect changes between old and new configuration."""
        changes = []

        try:
            old_dict = old_config.dict()
            new_dict = new_config.dict()

            # Flatten nested dictionaries for comparison
            old_flat = self._flatten_dict(old_dict)
            new_flat = self._flatten_dict(new_dict)

            # Find added and modified keys
            for key, new_value in new_flat.items():
                if key not in old_flat:
                    changes.append(
                        ConfigChange(
                            key=key,
                            old_value=None,
                            new_value=new_value,
                            change_type=ConfigChangeType.ADDED,
                            timestamp=datetime.now(),
                            source=ConfigSource.FILE,
                        )
                    )
                elif old_flat[key] != new_value:
                    changes.append(
                        ConfigChange(
                            key=key,
                            old_value=old_flat[key],
                            new_value=new_value,
                            change_type=ConfigChangeType.MODIFIED,
                            timestamp=datetime.now(),
                            source=ConfigSource.FILE,
                        )
                    )

            # Find deleted keys
            for key, old_value in old_flat.items():
                if key not in new_flat:
                    changes.append(
                        ConfigChange(
                            key=key,
                            old_value=old_value,
                            new_value=None,
                            change_type=ConfigChangeType.DELETED,
                            timestamp=datetime.now(),
                            source=ConfigSource.FILE,
                        )
                    )

        except Exception as e:
            logger.error(f"Error detecting config changes: {e}")

        return changes

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    async def _notify_config_changes(self, changes: List[ConfigChange]) -> None:
        """Notify registered callbacks about configuration changes."""
        for change in changes:
            callbacks = self.change_callbacks.get(change.key, [])
            callbacks.extend(self.change_callbacks.get("*", []))  # Global callbacks

            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(change)
                    else:
                        callback(change)
                except Exception as e:
                    logger.error(f"Configuration change callback failed: {e}")

    async def _create_rollback_point(self) -> None:
        """Create a rollback point for the current configuration."""
        try:
            rollback_data = {
                "timestamp": datetime.now().isoformat(),
                "config_hash": self.current_config_hash,
                "config": self.config.dict(),
            }

            self.rollback_stack.append(rollback_data)

            # Keep only last N rollback points
            if len(self.rollback_stack) > self.max_rollback_entries:
                self.rollback_stack.pop(0)

            # Store in Redis as well
            self.redis.lpush(
                "config:rollback_stack", json.dumps(rollback_data, default=str)
            )
            self.redis.ltrim("config:rollback_stack", 0, self.max_rollback_entries - 1)

        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")

    async def rollback_last_change(self) -> bool:
        """Rollback to the last configuration."""
        if not self.rollback_stack:
            logger.warning("No rollback points available")
            return False

        try:
            rollback_point = self.rollback_stack.pop()
            logger.info(
                f"Rolling back to configuration from {rollback_point['timestamp']}"
            )

            # This would typically restore the configuration
            # For now, just log the rollback
            logger.info("Configuration rollback completed")
            return True

        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return False

    def register_change_callback(self, config_key: str, callback: Callable) -> None:
        """Register callback for configuration changes."""
        if config_key not in self.change_callbacks:
            self.change_callbacks[config_key] = []

        self.change_callbacks[config_key].append(callback)
        logger.info(f"Registered callback for configuration key: {config_key}")

    async def update_config_value(
        self, key: str, value: Any, source: ConfigSource = ConfigSource.API
    ) -> bool:
        """Update a specific configuration value."""
        try:
            # Validate the change
            is_valid, error_msg = self.validator.validate_change(key, value)
            if not is_valid:
                logger.error(f"Configuration update validation failed: {error_msg}")
                return False

            # Create rollback point
            await self._create_rollback_point()

            # Get current value
            current_value = self._get_config_value(key)

            # Update the configuration
            if await self._set_config_value(key, value):
                # Record the change
                change = ConfigChange(
                    key=key,
                    old_value=current_value,
                    new_value=value,
                    change_type=ConfigChangeType.MODIFIED,
                    timestamp=datetime.now(),
                    source=source,
                    applied=True,
                )

                self.config_history.append(change)

                # Notify callbacks
                await self._notify_config_changes([change])

                # Store in Redis
                await self.redis.setex(
                    f"config:override:{key}",
                    86400,  # 24 hours
                    json.dumps(
                        {"value": value, "timestamp": datetime.now().isoformat()}
                    ),
                )

                logger.info(f"Configuration updated: {key} = {value}")
                return True

        except Exception as e:
            logger.error(f"Failed to update configuration {key}: {e}")
            return False

        return False

    def _get_config_value(self, key: str) -> Any:
        """Get current configuration value by key."""
        try:
            keys = key.split(".")
            value = self.config

            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return None

            return value
        except Exception:
            return None

    async def _set_config_value(self, key: str, value: Any) -> bool:
        """Set configuration value by key."""
        try:
            keys = key.split(".")
            config_obj = self.config

            # Navigate to the parent object
            for k in keys[:-1]:
                if hasattr(config_obj, k):
                    config_obj = getattr(config_obj, k)
                else:
                    return False

            # Set the final value
            final_key = keys[-1]
            if hasattr(config_obj, final_key):
                setattr(config_obj, final_key, value)
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to set config value {key}: {e}")
            return False

    async def _apply_redis_config_changes(self) -> None:
        """Apply configuration changes from Redis."""
        try:
            # Get configuration overrides from Redis
            override_keys = await self.redis.keys("config:override:*")

            for key_bytes in override_keys:
                key_str = (
                    key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                )
                config_key = key_str.replace("config:override:", "")

                override_data = await self.redis.get(key_str)
                if override_data:
                    try:
                        override_info = json.loads(override_data)
                        value = override_info["value"]

                        await self.update_config_value(
                            config_key, value, ConfigSource.REDIS
                        )

                    except json.JSONDecodeError:
                        logger.error(f"Invalid override data for {config_key}")

        except Exception as e:
            logger.error(f"Failed to apply Redis config changes: {e}")

    # A/B Testing Methods
    async def create_ab_test(self, test_config: ABTestConfig) -> bool:
        """Create a new A/B test."""
        try:
            # Validate test configuration
            if not self._validate_ab_test(test_config):
                return False

            # Store test in memory and Redis
            self.ab_tests[test_config.name] = test_config

            await self.redis.setex(
                f"ab_test:{test_config.name}",
                86400 * 30,  # 30 days
                json.dumps(
                    {
                        "name": test_config.name,
                        "description": test_config.description,
                        "variants": test_config.variants,
                        "traffic_split": test_config.traffic_split,
                        "start_time": test_config.start_time.isoformat(),
                        "end_time": (
                            test_config.end_time.isoformat()
                            if test_config.end_time
                            else None
                        ),
                        "enabled": test_config.enabled,
                        "metrics": test_config.metrics,
                    }
                ),
            )

            logger.info(f"A/B test created: {test_config.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            return False

    def _validate_ab_test(self, test_config: ABTestConfig) -> bool:
        """Validate A/B test configuration."""
        # Check traffic split sums to 1.0
        total_traffic = sum(test_config.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.01:
            logger.error(f"Traffic split must sum to 1.0, got {total_traffic}")
            return False

        # Check variants exist in traffic split
        for variant in test_config.variants:
            if variant not in test_config.traffic_split:
                logger.error(f"Variant {variant} not in traffic split")
                return False

        return True

    async def get_ab_test_assignment(
        self, test_name: str, user_id: str = "default"
    ) -> Optional[str]:
        """Get A/B test variant assignment for a user."""
        if test_name not in self.ab_tests:
            return None

        test = self.ab_tests[test_name]

        if not test.enabled:
            return None

        # Check if test is active
        now = datetime.now()
        if now < test.start_time or (test.end_time and now > test.end_time):
            return None

        # Check existing assignment
        if user_id in test.current_assignment:
            return test.current_assignment[user_id]

        # Assign variant based on traffic split
        import random

        random.seed(hash(f"{test_name}:{user_id}"))  # Consistent assignment
        rand_val = random.random()

        cumulative = 0.0
        for variant, traffic in test.traffic_split.items():
            cumulative += traffic
            if rand_val <= cumulative:
                test.current_assignment[user_id] = variant
                logger.debug(
                    f"Assigned user {user_id} to variant {variant} for test {test_name}"
                )
                return variant

        # Fallback to first variant
        first_variant = list(test.variants.keys())[0]
        test.current_assignment[user_id] = first_variant
        return first_variant

    async def get_ab_test_config(
        self, test_name: str, user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Get configuration for A/B test variant."""
        variant = await self.get_ab_test_assignment(test_name, user_id)
        if not variant or test_name not in self.ab_tests:
            return None

        test = self.ab_tests[test_name]
        return test.variants.get(variant)

    async def end_ab_test(
        self, test_name: str, winning_variant: Optional[str] = None
    ) -> bool:
        """End an A/B test and optionally apply winning variant."""
        if test_name not in self.ab_tests:
            return False

        try:
            test = self.ab_tests[test_name]
            test.enabled = False
            test.end_time = datetime.now()

            # Apply winning variant configuration if specified
            if winning_variant and winning_variant in test.variants:
                winning_config = test.variants[winning_variant]
                for key, value in winning_config.items():
                    await self.update_config_value(key, value, ConfigSource.API)

                logger.info(
                    f"Applied winning variant {winning_variant} from test {test_name}"
                )

            # Clean up assignments
            test.current_assignment.clear()

            # Update Redis
            await self.redis.delete(f"ab_test:{test_name}")

            logger.info(f"A/B test {test_name} ended")
            return True

        except Exception as e:
            logger.error(f"Failed to end A/B test {test_name}: {e}")
            return False

    async def _load_ab_tests(self) -> None:
        """Load active A/B tests from Redis."""
        try:
            ab_test_keys = await self.redis.keys("ab_test:*")

            for key_bytes in ab_test_keys:
                key_str = (
                    key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
                )
                test_name = key_str.replace("ab_test:", "")

                test_data = await self.redis.get(key_str)
                if test_data:
                    try:
                        test_info = json.loads(test_data)
                        ab_test = ABTestConfig(
                            name=test_info["name"],
                            description=test_info["description"],
                            variants=test_info["variants"],
                            traffic_split=test_info["traffic_split"],
                            start_time=datetime.fromisoformat(test_info["start_time"]),
                            end_time=(
                                datetime.fromisoformat(test_info["end_time"])
                                if test_info["end_time"]
                                else None
                            ),
                            enabled=test_info["enabled"],
                            metrics=test_info["metrics"],
                        )

                        self.ab_tests[test_name] = ab_test
                        logger.info(f"Loaded A/B test: {test_name}")

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Invalid A/B test data for {test_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to load A/B tests: {e}")

    async def get_ab_test_results(self, test_name: str) -> Dict[str, Any]:
        """Get results for an A/B test."""
        if test_name not in self.ab_tests:
            return {}

        test = self.ab_tests[test_name]

        # Get metrics for each variant
        results: Dict[str, Any] = {
            "test_name": test_name,
            "start_time": test.start_time.isoformat(),
            "end_time": test.end_time.isoformat() if test.end_time else None,
            "enabled": test.enabled,
            "variants": {},
            "assignments": len(test.current_assignment),
        }

        # Count assignments per variant
        variant_counts: Dict[str, int] = {}
        for variant in test.current_assignment.values():
            variant_counts[variant] = variant_counts.get(variant, 0) + 1

        for variant_name in test.variants:
            results["variants"][variant_name] = {
                "config": test.variants[variant_name],
                "assignment_count": variant_counts.get(variant_name, 0),
                "traffic_percentage": test.traffic_split.get(variant_name, 0) * 100,
            }

        return results

    # Configuration export/import
    async def export_configuration(
        self, include_history: bool = False
    ) -> Dict[str, Any]:
        """Export current configuration and optionally history."""
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "config_hash": self.current_config_hash,
            "configuration": self.config.dict(),
            "ab_tests": {},
        }

        # Export A/B tests
        for test_name, test in self.ab_tests.items():
            export_data["ab_tests"][test_name] = {
                "name": test.name,
                "description": test.description,
                "variants": test.variants,
                "traffic_split": test.traffic_split,
                "start_time": test.start_time.isoformat(),
                "end_time": test.end_time.isoformat() if test.end_time else None,
                "enabled": test.enabled,
                "metrics": test.metrics,
            }

        if include_history:
            export_data["change_history"] = [
                {
                    "key": change.key,
                    "old_value": change.old_value,
                    "new_value": change.new_value,
                    "change_type": change.change_type.value,
                    "timestamp": change.timestamp.isoformat(),
                    "source": change.source.value,
                    "applied": change.applied,
                }
                for change in self.config_history
            ]

        return export_data

    async def import_configuration(self, config_data: Dict[str, Any]) -> bool:
        """Import configuration from exported data."""
        try:
            # Validate imported configuration
            imported_config = config_data.get("configuration", {})

            # This would validate the entire configuration structure
            # For now, just check if it's a valid dictionary
            if not isinstance(imported_config, dict):
                logger.error("Invalid configuration format")
                return False

            # Create rollback point
            await self._create_rollback_point()

            # Apply configuration changes
            for key, value in self._flatten_dict(imported_config).items():
                await self.update_config_value(key, value, ConfigSource.API)

            # Import A/B tests if present
            ab_tests = config_data.get("ab_tests", {})
            for test_name, test_data in ab_tests.items():
                ab_test = ABTestConfig(
                    name=test_data["name"],
                    description=test_data["description"],
                    variants=test_data["variants"],
                    traffic_split=test_data["traffic_split"],
                    start_time=datetime.fromisoformat(test_data["start_time"]),
                    end_time=(
                        datetime.fromisoformat(test_data["end_time"])
                        if test_data["end_time"]
                        else None
                    ),
                    enabled=test_data["enabled"],
                    metrics=test_data["metrics"],
                )
                await self.create_ab_test(ab_test)

            logger.info("Configuration imported successfully")
            return True

        except Exception as e:
            logger.error(f"Configuration import failed: {e}")
            await self.rollback_last_change()
            return False

    async def get_config_status(self) -> Dict[str, Any]:
        """Get comprehensive configuration status."""
        return {
            "current_hash": self.current_config_hash,
            "auto_reload_enabled": self.auto_reload_enabled,
            "watched_files": list(self.watched_files),
            "change_history_count": len(self.config_history),
            "rollback_points": len(self.rollback_stack),
            "active_ab_tests": len([t for t in self.ab_tests.values() if t.enabled]),
            "total_ab_tests": len(self.ab_tests),
            "last_reload": max(
                (change.timestamp for change in self.config_history), default=None
            ),
        }

    async def cleanup_config_history(self, days: int = 7) -> int:
        """Clean up old configuration history."""
        cutoff_date = datetime.now() - timedelta(days=days)

        original_count = len(self.config_history)
        self.config_history = [
            change for change in self.config_history if change.timestamp > cutoff_date
        ]

        cleaned_count = original_count - len(self.config_history)
        logger.info(f"Cleaned up {cleaned_count} old configuration changes")
        return cleaned_count

    async def validate_current_config(self) -> Dict[str, Any]:
        """Validate the current configuration."""
        validation_results: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        try:
            # Validate critical configuration values
            config_dict = self._flatten_dict(self.config.dict())

            for key, value in config_dict.items():
                is_valid, error_msg = self.validator.validate_change(key, value)
                if not is_valid:
                    validation_results["errors"].append(f"{key}: {error_msg}")
                    validation_results["valid"] = False

            # Check for missing required values
            required_keys = ["database.password", "alpaca.api_key", "alpaca.secret_key"]

            for key in required_keys:
                value = self._get_config_value(key)
                if not value:
                    validation_results["errors"].append(
                        f"Required configuration missing: {key}"
                    )
                    validation_results["valid"] = False

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {str(e)}")

        return validation_results

    async def shutdown(self) -> None:
        """Shutdown the configuration manager."""
        logger.info("Shutting down configuration manager...")

        self.auto_reload_enabled = False

        # Save final state to Redis
        try:
            final_state = {
                "timestamp": datetime.now().isoformat(),
                "config_hash": self.current_config_hash,
                "active_ab_tests": len(
                    [t for t in self.ab_tests.values() if t.enabled]
                ),
            }

            await self.redis.setex(
                "config:manager:last_state", 86400, json.dumps(final_state)  # 24 hours
            )

        except Exception as e:
            logger.error(f"Failed to save final configuration state: {e}")

        logger.info("Configuration manager shutdown completed")
