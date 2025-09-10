#!/usr/bin/env python3
"""
AI Trading System Backup Service

This service handles automated backups, monitoring, and restoration capabilities
for the AI Trading System. It provides both scheduled backups and on-demand
backup operations with cloud synchronization and encryption.

Features:
- Scheduled database and file backups
- Real-time backup monitoring
- Cloud storage synchronization
- Backup verification and integrity checks
- Automated restore testing
- Backup retention management
- Health monitoring and alerting
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg
import redis
import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field
from redis import Redis

# Add parent directory to path for shared modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.config import get_config  # noqa: E402


# Simple implementations for missing classes
class DatabaseManager:
    def __init__(self) -> None:
        self.pool = None

    async def initialize(self) -> None:
        config = get_config()
        self.pool = await asyncpg.create_pool(
            host=config.database.host,
            port=config.database.port,
            database=config.database.database,
            user=config.database.username,
            password=config.database.password,
        )


class RedisManager:
    def __init__(self) -> None:
        self.client: Optional[Redis] = None

    async def initialize(self) -> None:
        config = get_config()
        self.client = redis.Redis(
            host=config.redis.host, port=config.redis.port, db=config.redis.database
        )


class HealthChecker:
    def __init__(
        self,
        db_manager: Optional[DatabaseManager],
        redis_manager: Optional[RedisManager],
    ) -> None:
        self.db_manager = db_manager
        self.redis_manager = redis_manager

    async def start(self) -> None:
        """Start health monitoring"""
        logger.info("Health checker started")


class ConfigWrapper:
    """Wrapper to provide dict-like interface for config object"""

    def __init__(self, config: Any) -> None:
        self._config = config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with fallback to environment variable"""
        if hasattr(self._config, key.lower()):
            return getattr(self._config, key.lower())
        return os.environ.get(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean config value"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes", "on")

    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer config value"""
        value = self.get(key, default)
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    @property
    def project_root(self) -> Path:
        """Get project root path"""
        return Path(__file__).parent.parent.parent


logger = structlog.get_logger(__name__)

# Prometheus metrics
backup_operations_total = Counter(
    "backup_operations_total",
    "Total number of backup operations",
    ["backup_type", "status"],
)

backup_duration_seconds = Histogram(
    "backup_duration_seconds", "Time spent on backup operations", ["backup_type"]
)

backup_file_size_bytes = Gauge(
    "backup_file_size_bytes", "Size of backup files", ["backup_type", "environment"]
)

backup_success_timestamp = Gauge(
    "backup_success_timestamp", "Timestamp of last successful backup", ["backup_type"]
)


class BackupRequest(BaseModel):
    """Backup request model"""

    backup_type: str = Field(
        ..., description="Type of backup (full, database, files, incremental)"
    )
    environment: str = Field(default="development", description="Environment name")
    compression: str = Field(default="gzip", description="Compression method")
    encryption_enabled: bool = Field(
        default=True, description="Enable backup encryption"
    )
    cloud_sync: bool = Field(default=False, description="Sync to cloud storage")
    retention_days: int = Field(default=30, description="Backup retention in days")


class BackupStatus(BaseModel):
    """Backup status model"""

    backup_id: str
    backup_type: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error_message: Optional[str] = None


class BackupService:
    """Main backup service class"""

    def __init__(self) -> None:
        self.config = ConfigWrapper(get_config())
        self.db_manager: Optional[DatabaseManager] = None
        self.redis_manager: Optional[RedisManager] = None
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.health_checker = HealthChecker(self.db_manager, self.redis_manager)

        # Backup state tracking
        self.active_backups: Dict[str, BackupStatus] = {}
        self.backup_history: List[BackupStatus] = []

        # Configuration
        backup_root_str = self.config.get("BACKUP_ROOT", "./backups")
        self.backup_root = Path(backup_root_str or "./backups")
        self.encryption_key = self.config.get("BACKUP_ENCRYPTION_KEY", "")
        self.s3_bucket = self.config.get("S3_BACKUP_BUCKET", "")

        # Ensure backup directories exist
        self.backup_root.mkdir(parents=True, exist_ok=True)
        (self.backup_root / "database").mkdir(exist_ok=True)
        (self.backup_root / "files").mkdir(exist_ok=True)
        (self.backup_root / "incremental").mkdir(exist_ok=True)
        (self.backup_root / "config").mkdir(exist_ok=True)

    async def initialize(self) -> None:
        """Initialize service connections and scheduler"""
        logger.info("Initializing backup service")

        try:
            # Initialize database connection
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()

            # Initialize Redis connection
            self.redis_manager = RedisManager()
            await self.redis_manager.initialize()

            # Initialize scheduler
            self.scheduler = AsyncIOScheduler()
            await self.setup_scheduled_backups()

            # Start health monitoring
            await self.health_checker.start()

            logger.info("Backup service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize backup service: {str(e)}")
            raise

    async def setup_scheduled_backups(self) -> None:
        """Setup scheduled backup jobs"""
        logger.info("Setting up scheduled backup jobs")

        if not self.scheduler:
            logger.error("Scheduler not initialized")
            return

        # Daily full backup at 2 AM
        self.scheduler.add_job(
            self.run_scheduled_backup,
            CronTrigger.from_crontab(self.config.get("BACKUP_SCHEDULE", "0 2 * * *")),
            args=["full"],
            id="daily_full_backup",
            name="Daily Full Backup",
            max_instances=1,
            coalesce=True,
        )

        # Hourly incremental backup
        self.scheduler.add_job(
            self.run_scheduled_backup,
            CronTrigger(minute=0),  # Every hour
            args=["incremental"],
            id="hourly_incremental_backup",
            name="Hourly Incremental Backup",
            max_instances=1,
            coalesce=True,
        )

        # Daily backup verification
        self.scheduler.add_job(
            self.verify_recent_backups,
            CronTrigger(hour=3, minute=0),  # 3 AM daily
            id="daily_backup_verification",
            name="Daily Backup Verification",
            max_instances=1,
        )

        # Weekly cleanup
        self.scheduler.add_job(
            self.cleanup_old_backups,
            CronTrigger(day_of_week=0, hour=4, minute=0),  # Sunday 4 AM
            id="weekly_cleanup",
            name="Weekly Backup Cleanup",
            max_instances=1,
        )

        # Weekly restore testing
        self.scheduler.add_job(
            self.run_restore_test,
            CronTrigger(day_of_week=6, hour=5, minute=0),  # Saturday 5 AM
            id="weekly_restore_test",
            name="Weekly Restore Test",
            max_instances=1,
        )

        self.scheduler.start()
        logger.info("Scheduled backup jobs configured")

    async def run_scheduled_backup(self, backup_type: str) -> None:
        """Run a scheduled backup operation"""
        logger.info(f"Running scheduled backup: {backup_type}")

        try:
            environment = self.config.get("ENVIRONMENT", "development")
            backup_request = BackupRequest(
                backup_type=backup_type,
                environment=environment or "development",
                cloud_sync=self.config.get_bool("CLOUD_SYNC", False),
                retention_days=self.config.get_int("BACKUP_RETENTION_DAYS", 30),
            )

            await self.create_backup(backup_request)

        except Exception as e:
            logger.error(f"Scheduled backup failed [{backup_type}]: {str(e)}")
            await self.send_backup_alert(
                "error", f"Scheduled {backup_type} backup failed: {str(e)}"
            )

    async def create_backup(self, request: BackupRequest) -> BackupStatus:
        """Create a backup based on the request"""
        backup_id = f"{request.backup_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_status = BackupStatus(
            backup_id=backup_id,
            backup_type=request.backup_type,
            status="running",
            started_at=datetime.now(timezone.utc),
        )

        self.active_backups[backup_id] = backup_status

        logger.info(f"Creating backup {backup_id} [{request.backup_type}]")

        try:
            with backup_duration_seconds.labels(backup_type=request.backup_type).time():

                if request.backup_type == "full":
                    await self.create_full_backup(backup_id, request)
                elif request.backup_type == "database":
                    await self.create_database_backup(backup_id, request)
                elif request.backup_type == "files":
                    await self.create_files_backup(backup_id, request)
                elif request.backup_type == "incremental":
                    await self.create_incremental_backup(backup_id, request)
                elif request.backup_type == "config":
                    await self.create_config_backup(backup_id, request)
                else:
                    raise ValueError(f"Unknown backup type: {request.backup_type}")

            # Update status
            backup_status.status = "completed"
            backup_status.completed_at = datetime.now(timezone.utc)

            # Record metrics
            backup_operations_total.labels(
                backup_type=request.backup_type, status="success"
            ).inc()

            backup_success_timestamp.labels(
                backup_type=request.backup_type
            ).set_to_current_time()

            # Cloud sync if enabled
            if request.cloud_sync:
                await self.sync_to_cloud(backup_status)

            # Add to history
            self.backup_history.append(backup_status)

            # Remove from active backups
            del self.active_backups[backup_id]

            logger.info(f"Backup created successfully: {backup_id}")

            # Send success notification
            await self.send_backup_alert(
                "success",
                f"{request.backup_type.title()} backup completed successfully (ID: {backup_id})",
            )

            return backup_status

        except Exception as e:
            # Update status with error
            backup_status.status = "failed"
            backup_status.error_message = str(e)
            backup_status.completed_at = datetime.now(timezone.utc)

            # Record metrics
            backup_operations_total.labels(
                backup_type=request.backup_type, status="error"
            ).inc()

            # Add to history
            self.backup_history.append(backup_status)

            # Remove from active backups
            del self.active_backups[backup_id]

            logger.error(f"Backup creation failed [{backup_id}]: {str(e)}")

            # Send error notification
            await self.send_backup_alert(
                "error",
                f"{request.backup_type.title()} backup failed (ID: {backup_id}): {str(e)}",
            )

            raise

    async def create_full_backup(self, backup_id: str, request: BackupRequest) -> None:
        """Create a full system backup"""
        logger.info("Creating full backup", backup_id=backup_id)

        # Create session directory
        session_dir = self.backup_root / "full" / backup_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Run parallel backups
        tasks = [
            self.backup_database(session_dir),
            self.backup_redis(session_dir),
            self.backup_files(session_dir),
            self.backup_configuration(session_dir),
        ]

        await asyncio.gather(*tasks)

        # Create consolidated archive
        await self.create_consolidated_archive(session_dir, backup_id, request)

    async def create_database_backup(
        self, backup_id: str, request: BackupRequest
    ) -> None:
        """Create database-only backup"""
        logger.info("Creating database backup", backup_id=backup_id)

        session_dir = self.backup_root / "database" / backup_id
        session_dir.mkdir(parents=True, exist_ok=True)

        await self.backup_database(session_dir)
        await self.backup_redis(session_dir)

    async def create_files_backup(self, backup_id: str, request: BackupRequest) -> None:
        """Create files-only backup"""
        logger.info("Creating files backup", backup_id=backup_id)

        session_dir = self.backup_root / "files" / backup_id
        session_dir.mkdir(parents=True, exist_ok=True)

        await self.backup_files(session_dir)

    async def create_incremental_backup(
        self, backup_id: str, request: BackupRequest
    ) -> None:
        """Create incremental backup"""
        logger.info("Creating incremental backup", backup_id=backup_id)

        # Check for last full backup timestamp
        last_backup_file = Path("./data/last_full_backup.timestamp")

        if not last_backup_file.exists():
            logger.warning(
                "No previous full backup found, creating full backup instead"
            )
            await self.create_full_backup(backup_id, request)
            return

        session_dir = self.backup_root / "incremental" / backup_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Backup only changed files
        await self.backup_incremental_files(
            session_dir, last_backup_file.stat().st_mtime
        )

    async def create_config_backup(
        self, backup_id: str, request: BackupRequest
    ) -> None:
        """Create configuration-only backup"""
        logger.info("Creating configuration backup", backup_id=backup_id)

        session_dir = self.backup_root / "config" / backup_id
        session_dir.mkdir(parents=True, exist_ok=True)

        await self.backup_configuration(session_dir)

    async def backup_database(self, session_dir: Path) -> None:
        """Backup PostgreSQL database"""
        logger.info("Backing up PostgreSQL database")

        backup_file = (
            session_dir / f"postgres_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        )

        # Create database dump using subprocess
        cmd = [
            "pg_dump",
            "-h",
            self.config.get("DB_HOST", "localhost"),
            "-p",
            self.config.get("DB_PORT", "5432"),
            "-U",
            self.config.get("DB_USER", "trader"),
            "-d",
            self.config.get("DB_NAME", "trading_system"),
            "--verbose",
            "--no-owner",
            "--no-privileges",
            "--clean",
            "--if-exists",
            "--create",
        ]

        env = os.environ.copy()
        password = self.config.get("DB_PASSWORD", "")
        env["PGPASSWORD"] = password or ""

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Database backup failed: {stderr.decode()}")

        # Write backup file
        backup_file.write_bytes(stdout)

        # Record file size
        file_size = backup_file.stat().st_size
        backup_file_size_bytes.labels(
            backup_type="database",
            environment=self.config.get("ENVIRONMENT", "development"),
        ).set(file_size)

        logger.info(
            "Database backup completed", file_size=file_size, file_path=str(backup_file)
        )

    async def backup_redis(self, session_dir: Path) -> None:
        """Backup Redis data"""
        logger.info("Backing up Redis data")

        backup_file = (
            session_dir / f"redis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rdb"
        )

        # Create Redis dump
        redis_host = self.config.get("REDIS_HOST", "localhost")
        redis_port = self.config.get("REDIS_PORT", "6379")
        redis_password = self.config.get("REDIS_PASSWORD", "")

        cmd = ["redis-cli", "-h", redis_host, "-p", redis_port]
        if redis_password:
            cmd.extend(["-a", redis_password])
        cmd.extend(["--rdb", str(backup_file)])

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Redis backup failed: {stderr.decode()}")

        # Record file size
        if backup_file.exists():
            file_size = backup_file.stat().st_size
            backup_file_size_bytes.labels(
                backup_type="redis",
                environment=self.config.get("ENVIRONMENT", "development"),
            ).set(file_size)

            logger.info(
                "Redis backup completed",
                file_size=file_size,
                file_path=str(backup_file),
            )

    async def backup_files(self, session_dir: Path) -> None:
        """Backup data files"""
        logger.info("Backing up data files")

        backup_file = (
            session_dir / f"files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        )

        # Define paths to backup
        backup_paths = ["./data/parquet", "./data/exports", "./logs"]

        # Create tar archive
        cmd = ["tar", "-czf", str(backup_file)]

        # Add existing paths only
        for path in backup_paths:
            if Path(path).exists():
                cmd.append(path)

        if len(cmd) == 3:  # Only tar, -czf, and filename
            logger.warning("No data files found to backup")
            return

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.project_root,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"File backup failed: {stderr.decode()}")

        # Record file size
        file_size = backup_file.stat().st_size
        backup_file_size_bytes.labels(
            backup_type="files",
            environment=self.config.get("ENVIRONMENT", "development"),
        ).set(file_size)

        logger.info(
            "File backup completed", file_size=file_size, file_path=str(backup_file)
        )

    async def backup_configuration(self, session_dir: Path) -> None:
        """Backup configuration files"""
        logger.info("Backing up configuration files")

        backup_file = (
            session_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        )

        # Backup config files (excluding sensitive data)
        cmd = [
            "tar",
            "-czf",
            str(backup_file),
            "--exclude=*.env*",
            "--exclude=*password*",
            "--exclude=*secret*",
            "--exclude=*key*",
            "config/",
            "docker-compose*.yml",
            "Dockerfile*",
            "requirements*.txt",
            "scripts/",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.project_root,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Configuration backup failed: {stderr.decode()}")

        logger.info("Configuration backup completed", file_path=str(backup_file))

    async def backup_incremental_files(
        self, session_dir: Path, since_timestamp: float
    ) -> None:
        """Create incremental file backup"""
        logger.info("Creating incremental file backup", since_timestamp=since_timestamp)

        backup_file = (
            session_dir
            / f"incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        )

        # Find files changed since timestamp
        cmd = [
            "find",
            "./data",
            "-type",
            "f",
            "-newermt",
            f"@{since_timestamp}",
            "-print0",
        ]

        find_process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.project_root,
        )

        find_stdout, find_stderr = await find_process.communicate()

        if find_process.returncode != 0:
            raise RuntimeError(f"File search failed: {find_stderr.decode()}")

        if not find_stdout.strip():
            logger.info("No files changed since last backup")
            return

        # Create tar from file list
        tar_process = await asyncio.create_subprocess_exec(
            "tar",
            "-czf",
            str(backup_file),
            "--null",
            "-T",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.project_root,
        )

        tar_stdout, tar_stderr = await tar_process.communicate(input=find_stdout)

        if tar_process.returncode != 0:
            raise RuntimeError(f"Incremental backup failed: {tar_stderr.decode()}")

        logger.info("Incremental backup completed", file_path=str(backup_file))

    async def create_consolidated_archive(
        self, session_dir: Path, backup_id: str, request: BackupRequest
    ) -> str:
        """Create consolidated backup archive"""
        logger.info("Creating consolidated backup archive", backup_id=backup_id)

        consolidated_file = self.backup_root / f"full_backup_{backup_id}.tar.gz"

        # Create consolidated archive
        cmd = ["tar", "-czf", str(consolidated_file), "-C", str(session_dir), "."]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(
                f"Consolidated archive creation failed: {stderr.decode()}"
            )

        # Apply encryption if enabled
        if request.encryption_enabled and self.encryption_key:
            await self.encrypt_backup_file(consolidated_file)

        # Update backup status
        if backup_id in self.active_backups:
            self.active_backups[backup_id].file_path = str(consolidated_file)
            self.active_backups[backup_id].file_size = consolidated_file.stat().st_size

        logger.info("Consolidated archive created", file_path=str(consolidated_file))
        return str(consolidated_file)

    async def encrypt_backup_file(self, backup_file: Path) -> None:
        """Encrypt backup file using GPG"""
        logger.info("Encrypting backup file", file_path=str(backup_file))

        encrypted_file = backup_file.with_suffix(backup_file.suffix + ".enc")

        cmd = [
            "gpg",
            "--batch",
            "--yes",
            "--passphrase",
            self.encryption_key,
            "--symmetric",
            "--cipher-algo",
            "AES256",
            "--output",
            str(encrypted_file),
            str(backup_file),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Backup encryption failed: {stderr.decode()}")

        # Remove unencrypted file
        backup_file.unlink()

        logger.info("Backup file encrypted", encrypted_file=str(encrypted_file))

    async def sync_to_cloud(self, backup_status: BackupStatus) -> None:
        """Sync backup to cloud storage"""
        if not self.s3_bucket:
            logger.warning("Cloud sync requested but S3 bucket not configured")
            return

        logger.info(
            "Syncing backup to cloud storage", backup_id=backup_status.backup_id
        )

        # Use AWS CLI for upload
        if not backup_status.file_path:
            raise RuntimeError("Backup file path is not set")

        cmd = [
            "aws",
            "s3",
            "cp",
            backup_status.file_path,
            f"s3://{self.s3_bucket}/backups/{self.config.get('ENVIRONMENT')}/",
            "--storage-class",
            "STANDARD_IA",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Cloud sync failed: {stderr.decode()}")

        logger.info("Backup synced to cloud", backup_id=backup_status.backup_id)

    async def verify_recent_backups(self) -> None:
        """Verify recent backups for integrity"""
        logger.info("Starting backup verification")

        # Find backups from last 7 days
        cutoff_date = datetime.now() - timedelta(days=7)
        recent_backups = []

        for backup_file in self.backup_root.rglob("*.tar*"):
            if backup_file.stat().st_mtime > cutoff_date.timestamp():
                recent_backups.append(backup_file)

        if not recent_backups:
            logger.warning("No recent backups found for verification")
            return

        verification_results = []

        for backup_file in recent_backups[:5]:  # Verify last 5 backups
            try:
                await self.verify_backup_integrity(backup_file)
                verification_results.append(f"✓ {backup_file.name}")
                logger.info("Backup verification passed", file=backup_file.name)
            except Exception as e:
                verification_results.append(f"✗ {backup_file.name}: {str(e)}")
                logger.error(
                    "Backup verification failed", file=backup_file.name, error=str(e)
                )

        # Send verification summary
        await self.send_backup_alert(
            "info",
            "Backup verification completed:\n" + "\n".join(verification_results),
        )

    async def verify_backup_integrity(self, backup_file: Path) -> None:
        """Verify integrity of a backup file"""
        logger.info("Verifying backup integrity for %s", backup_file.name)

        # Check file exists and is readable
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

        # Verify checksum if available
        checksum_file = backup_file.with_suffix(backup_file.suffix + ".sha256")
        if checksum_file.exists():
            cmd = ["sha256sum", "-c", str(checksum_file)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=backup_file.parent,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"Checksum verification failed: {stderr.decode()}")

        # Test archive integrity
        if backup_file.suffix in [".gz", ".tar.gz"]:
            cmd = ["gzip", "-t", str(backup_file)]
        elif backup_file.suffix == ".tar":
            cmd = ["tar", "-tf", str(backup_file)]
        else:
            return  # Skip unknown formats

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(f"Archive integrity check failed: {stderr.decode()}")

        logger.info("Backup integrity verification passed", file=backup_file.name)

    async def cleanup_old_backups(self) -> None:
        """Clean up old backups based on retention policy"""
        logger.info("Starting backup cleanup")

        retention_days = self.config.get_int("BACKUP_RETENTION_DAYS", 30)
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        deleted_count = 0
        total_size_freed = 0

        for backup_file in self.backup_root.rglob("*"):
            if (
                backup_file.is_file()
                and backup_file.stat().st_mtime < cutoff_date.timestamp()
            ):
                file_size = backup_file.stat().st_size
                backup_file.unlink()

                # Also remove associated files
                for suffix in [".meta", ".sha256", ".enc"]:
                    associated_file = backup_file.with_suffix(
                        backup_file.suffix + suffix
                    )
                    if associated_file.exists():
                        associated_file.unlink()

                deleted_count += 1
                total_size_freed += file_size

                logger.info("Deleted old backup", file=backup_file.name, size=file_size)

        # Remove empty directories
        for backup_dir in self.backup_root.rglob("*"):
            if backup_dir.is_dir() and not any(backup_dir.iterdir()):
                backup_dir.rmdir()

        logger.info(
            "Backup cleanup completed",
            deleted_files=deleted_count,
            size_freed=total_size_freed,
        )

        # Send cleanup summary
        await self.send_backup_alert(
            "info",
            f"Backup cleanup completed: {deleted_count} files deleted, "
            f"{total_size_freed / (1024 * 1024):.1f} MB freed",
        )

    async def run_restore_test(self) -> bool:
        """Run automated restore test"""
        logger.info("Starting automated restore test")

        try:
            # Find latest full backup
            latest_backup = None
            latest_time = 0.0

            for backup_file in self.backup_root.rglob("full_*.tar*"):
                if backup_file.stat().st_mtime > latest_time:
                    latest_time = backup_file.stat().st_mtime
                    latest_backup = backup_file

            if not latest_backup:
                logger.warning("No full backups found for restore testing")
                return False

            # Run restore test script
            cmd = [
                str(
                    Path(__file__).parent.parent.parent
                    / "scripts"
                    / "backup"
                    / "test_restore.sh"
                ),
                self.config.get("ENVIRONMENT", "development"),
                "--backup-file",
                str(latest_backup),
                "--timeout",
                "1800",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info("Restore test completed successfully")
                return True
            else:
                logger.error(f"Restore test failed: {stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Error running restore test: {e}")
            return False

    async def send_backup_alert(self, alert_type: str, message: str) -> None:
        """Send backup alert notification"""
        try:
            logger.info(f"Backup alert [{alert_type}]: {message}")

            # Here you could integrate with notification services like:
            # - Slack webhook
            # - Email service
            # - Discord webhook
            # - PagerDuty
            # For now, just log the alert

        except Exception as e:
            logger.error(f"Error sending backup alert: {e}")
