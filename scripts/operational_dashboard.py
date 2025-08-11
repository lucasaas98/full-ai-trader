#!/usr/bin/env python3
"""
AI Trading System Operational Dashboard
Comprehensive operational status and management interface
Usage: python operational_dashboard.py [options]
"""

import os
import sys
import json
import time
import subprocess
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import docker
    import psutil
    import requests
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    from rich.text import Text
    pass  # yaml import removed as unused
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Install with: pip install docker psutil requests rich pyyaml")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'operations' / 'dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Console for rich output
console = Console()


class ServiceStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"


class HealthStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ServiceInfo:
    name: str
    status: ServiceStatus
    health: HealthStatus
    port: Optional[int] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    uptime: str = "0s"
    last_restart: Optional[datetime] = None
    error_count: int = 0


@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_rx: int
    network_tx: int
    load_average: Tuple[float, float, float]
    uptime: str
    timestamp: datetime


@dataclass
class TradingMetrics:
    active_positions: int
    daily_trades: int
    daily_pnl: float
    portfolio_value: float
    risk_score: float
    strategy_performance: Dict[str, float]
    last_trade_time: Optional[datetime]
    system_status: str


class OperationalDashboard:
    def __init__(self, environment: str = "development", refresh_interval: int = 30):
        self.environment = environment
        self.refresh_interval = refresh_interval
        self.project_root = PROJECT_ROOT
        self.docker_client = None
        self.compose_file = self._get_compose_file()

        # Service definitions
        self.core_services = [
            "postgres", "redis", "data_collector", "strategy_engine",
            "risk_manager", "trade_executor", "scheduler"
        ]
        self.support_services = ["export_service", "maintenance_service"]
        self.monitoring_services = [
            "prometheus", "grafana", "alertmanager",
            "elasticsearch", "kibana", "logstash"
        ]

        self.all_services = (
            self.core_services + self.support_services + self.monitoring_services
        )

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")

    def _get_compose_file(self) -> str:
        """Get appropriate Docker Compose file for environment"""
        compose_files = {
            "production": "docker-compose.prod.yml",
            "staging": "docker-compose.staging.yml",
            "development": "docker-compose.yml"
        }
        return compose_files.get(self.environment, "docker-compose.yml")

    def get_service_info(self, service_name: str) -> ServiceInfo:
        """Get detailed information about a service"""
        try:
            # Get container info
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "ps", "-q", service_name],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )

            if not result.stdout.strip():
                return ServiceInfo(
                    name=service_name,
                    status=ServiceStatus.STOPPED,
                    health=HealthStatus.UNKNOWN
                )

            container_id = result.stdout.strip()
            if self.docker_client is None:
                return ServiceInfo(
                    name=service_name,
                    status=ServiceStatus.UNKNOWN,
                    health=HealthStatus.UNKNOWN
                )
            container = self.docker_client.containers.get(container_id)

            # Determine status
            status = ServiceStatus.UNKNOWN
            if container.status == "running":
                status = ServiceStatus.RUNNING
            elif container.status == "exited":
                status = ServiceStatus.STOPPED
            elif container.status in ["created", "restarting"]:
                status = ServiceStatus.STARTING

            # Get resource usage
            stats = container.stats(stream=False)
            cpu_usage = self._calculate_cpu_usage(stats)
            memory_usage = self._calculate_memory_usage(stats)

            # Get health status
            health = self._check_service_health(service_name, container)

            # Get uptime
            started_at = datetime.fromisoformat(
                container.attrs['State']['StartedAt'].replace('Z', '+00:00')
            )
            uptime = str(datetime.now(started_at.tzinfo) - started_at).split('.')[0]

            return ServiceInfo(
                name=service_name,
                status=status,
                health=health,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                uptime=uptime,
                last_restart=started_at
            )

        except Exception as e:
            logger.error(f"Failed to get info for service {service_name}: {e}")
            return ServiceInfo(
                name=service_name,
                status=ServiceStatus.ERROR,
                health=HealthStatus.UNKNOWN
            )

    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats"""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"] -
                stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"] -
                stats["precpu_stats"]["system_cpu_usage"]
            )
            online_cpus = stats["cpu_stats"]["online_cpus"]

            if system_delta > 0 and cpu_delta > 0:
                return (cpu_delta / system_delta) * online_cpus * 100.0
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0

    def _calculate_memory_usage(self, stats: Dict) -> float:
        """Calculate memory usage percentage from Docker stats"""
        try:
            usage = stats["memory_stats"]["usage"]
            limit = stats["memory_stats"]["limit"]
            return (usage / limit) * 100.0
        except (KeyError, ZeroDivisionError):
            return 0.0

    def _check_service_health(self, service_name: str, container) -> HealthStatus:
        """Check health status of a service"""
        try:
            if service_name == "postgres":
                # Check PostgreSQL health
                result = subprocess.run(
                    ["docker-compose", "-f", self.compose_file, "exec", "-T",
                     "postgres", "pg_isready", "-U", "trading_user", "-d", "trading_db"],
                    cwd=self.project_root,
                    capture_output=True,
                    timeout=10
                )
                return HealthStatus.HEALTHY if result.returncode == 0 else HealthStatus.UNHEALTHY

            elif service_name == "redis":
                # Check Redis health
                result = subprocess.run(
                    ["docker-compose", "-f", self.compose_file, "exec", "-T",
                     "redis", "redis-cli", "ping"],
                    cwd=self.project_root,
                    capture_output=True,
                    timeout=10
                )
                return HealthStatus.HEALTHY if result.returncode == 0 else HealthStatus.UNHEALTHY

            else:
                # Check API health endpoint
                try:
                    port_result = subprocess.run(
                        ["docker-compose", "-f", self.compose_file, "port", service_name, "8000"],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if port_result.returncode == 0:
                        port = port_result.stdout.strip().split(':')[-1]
                        response = requests.get(
                            f"http://localhost:{port}/health",
                            timeout=5
                        )
                        return HealthStatus.HEALTHY if response.status_code == 200 else HealthStatus.UNHEALTHY
                except:
                    pass

                # Fallback to container status
                return HealthStatus.HEALTHY if container.status == "running" else HealthStatus.UNHEALTHY

        except Exception as e:
            logger.debug(f"Health check failed for {service_name}: {e}")
            return HealthStatus.UNKNOWN

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU and memory
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage(str(self.project_root))
            disk_usage = (disk.used / disk.total) * 100

            # Network
            network = psutil.net_io_counters()

            # Load average
            load_avg = os.getloadavg()

            # Uptime
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            uptime = str(timedelta(seconds=int(uptime_seconds)))

            return SystemMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk_usage,
                network_rx=network.bytes_recv,
                network_tx=network.bytes_sent,
                load_average=load_avg,
                uptime=uptime,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_rx=0,
                network_tx=0,
                load_average=(0.0, 0.0, 0.0),
                uptime="unknown",
                timestamp=datetime.now()
            )

    def get_trading_metrics(self) -> TradingMetrics:
        """Get trading-specific metrics"""
        try:
            # Default values
            active_positions = 0
            daily_trades = 0
            daily_pnl = 0.0
            portfolio_value = 0.0
            risk_score = 0.0
            strategy_performance = {}
            last_trade_time = None
            system_status = "unknown"

            # Try to get metrics from database
            try:
                result = subprocess.run(
                    ["docker-compose", "-f", self.compose_file, "exec", "-T", "postgres",
                     "psql", "-U", "trading_user", "-d", "trading_db", "-t", "-c",
                     "SELECT COUNT(*) FROM positions WHERE status = 'OPEN';"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    active_positions = int(result.stdout.strip() or 0)
            except:
                pass

            try:
                result = subprocess.run(
                    ["docker-compose", "-f", self.compose_file, "exec", "-T", "postgres",
                     "psql", "-U", "trading_user", "-d", "trading_db", "-t", "-c",
                     "SELECT COUNT(*) FROM trades WHERE DATE(created_at) = CURRENT_DATE;"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    daily_trades = int(result.stdout.strip() or 0)
            except:
                pass

            try:
                result = subprocess.run(
                    ["docker-compose", "-f", self.compose_file, "exec", "-T", "postgres",
                     "psql", "-U", "trading_user", "-d", "trading_db", "-t", "-c",
                     "SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE DATE(created_at) = CURRENT_DATE;"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    daily_pnl = float(result.stdout.strip() or 0)
            except:
                pass

            # Try to get system status from maintenance service
            try:
                response = requests.get(
                    "http://localhost:8007/status",
                    timeout=5
                )
                if response.status_code == 200:
                    status_data = response.json()
                    system_status = "maintenance" if status_data.get("maintenance_mode") else "operational"
            except:
                system_status = "unknown"

            return TradingMetrics(
                active_positions=active_positions,
                daily_trades=daily_trades,
                daily_pnl=daily_pnl,
                portfolio_value=portfolio_value,
                risk_score=risk_score,
                strategy_performance=strategy_performance,
                last_trade_time=last_trade_time,
                system_status=system_status
            )

        except Exception as e:
            logger.error(f"Failed to get trading metrics: {e}")
            return TradingMetrics(
                active_positions=0,
                daily_trades=0,
                daily_pnl=0.0,
                portfolio_value=0.0,
                risk_score=0.0,
                strategy_performance={},
                last_trade_time=None,
                system_status="error"
            )

    def create_services_table(self, services: List[ServiceInfo]) -> Table:
        """Create a table showing service status"""
        table = Table(title="Service Status", box=box.ROUNDED)

        table.add_column("Service", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Health", justify="center")
        table.add_column("CPU %", justify="right")
        table.add_column("Memory %", justify="right")
        table.add_column("Uptime", justify="center")

        for service in services:
            # Status styling
            status_style = "green" if service.status == ServiceStatus.RUNNING else "red"
            status_text = f"[{status_style}]{service.status.value.upper()}[/{status_style}]"

            # Health styling
            health_style = {
                HealthStatus.HEALTHY: "green",
                HealthStatus.UNHEALTHY: "red",
                HealthStatus.DEGRADED: "yellow",
                HealthStatus.UNKNOWN: "dim"
            }.get(service.health, "dim")
            health_text = f"[{health_style}]{service.health.value.upper()}[/{health_style}]"

            # Resource usage styling
            cpu_style = "red" if service.cpu_usage > 80 else "yellow" if service.cpu_usage > 60 else "green"
            memory_style = "red" if service.memory_usage > 80 else "yellow" if service.memory_usage > 60 else "green"

            table.add_row(
                service.name,
                status_text,
                health_text,
                f"[{cpu_style}]{service.cpu_usage:.1f}[/{cpu_style}]",
                f"[{memory_style}]{service.memory_usage:.1f}[/{memory_style}]",
                service.uptime
            )

        return table

    def create_system_metrics_panel(self, metrics: SystemMetrics) -> Panel:
        """Create system metrics panel"""
        # CPU styling
        cpu_style = "red" if metrics.cpu_usage > 80 else "yellow" if metrics.cpu_usage > 60 else "green"

        # Memory styling
        memory_style = "red" if metrics.memory_usage > 80 else "yellow" if metrics.memory_usage > 60 else "green"

        # Disk styling
        disk_style = "red" if metrics.disk_usage > 90 else "yellow" if metrics.disk_usage > 75 else "green"

        content = f"""
[bold]System Resources[/bold]

CPU Usage:    [{cpu_style}]{metrics.cpu_usage:.1f}%[/{cpu_style}]
Memory Usage: [{memory_style}]{metrics.memory_usage:.1f}%[/{memory_style}]
Disk Usage:   [{disk_style}]{metrics.disk_usage:.1f}%[/{disk_style}]

Load Average: {metrics.load_average[0]:.2f}, {metrics.load_average[1]:.2f}, {metrics.load_average[2]:.2f}
Uptime:       {metrics.uptime}

Network:
  RX: {metrics.network_rx / 1024 / 1024:.1f} MB
  TX: {metrics.network_tx / 1024 / 1024:.1f} MB

Last Updated: {metrics.timestamp.strftime('%H:%M:%S')}
        """.strip()

        return Panel(content, title="System Metrics", border_style="blue")

    def create_trading_metrics_panel(self, metrics: TradingMetrics) -> Panel:
        """Create trading metrics panel"""
        # PnL styling
        pnl_style = "green" if metrics.daily_pnl > 0 else "red" if metrics.daily_pnl < 0 else "white"

        # Risk styling
        risk_style = "red" if metrics.risk_score > 80 else "yellow" if metrics.risk_score > 60 else "green"

        # Status styling
        status_style = {
            "operational": "green",
            "maintenance": "yellow",
            "error": "red",
            "unknown": "dim"
        }.get(metrics.system_status, "dim")

        content = f"""
[bold]Trading Performance[/bold]

System Status:    [{status_style}]{metrics.system_status.upper()}[/{status_style}]
Active Positions: {metrics.active_positions}
Daily Trades:     {metrics.daily_trades}
Daily P&L:        [{pnl_style}]${metrics.daily_pnl:,.2f}[/{pnl_style}]
Portfolio Value:  ${metrics.portfolio_value:,.2f}
Risk Score:       [{risk_style}]{metrics.risk_score:.1f}/100[/{risk_style}]

Last Trade: {metrics.last_trade_time.strftime('%H:%M:%S') if metrics.last_trade_time else 'N/A'}

Strategy Performance:
""".strip()

        for strategy, performance in metrics.strategy_performance.items():
            perf_style = "green" if performance > 0 else "red"
            content += f"\n  {strategy}: [{perf_style}]{performance:.2f}%[/{perf_style}]"

        return Panel(content, title="Trading Metrics", border_style="green")

    def create_alerts_panel(self) -> Panel:
        """Create alerts and notifications panel"""
        content = "[bold]Recent Alerts[/bold]\n\n"

        # Read recent alerts
        alert_file = self.project_root / "logs" / "monitoring" / "alerts.log"
        if alert_file.exists():
            try:
                with open(alert_file, 'r') as f:
                    lines = f.readlines()
                    recent_alerts = lines[-10:] if len(lines) > 10 else lines

                    if recent_alerts:
                        for alert in recent_alerts:
                            if "ERROR" in alert or "CRITICAL" in alert:
                                content += f"[red]🚨 {alert.strip()}[/red]\n"
                            elif "WARNING" in alert:
                                content += f"[yellow]⚠️  {alert.strip()}[/yellow]\n"
                            else:
                                content += f"[dim]ℹ️  {alert.strip()}[/dim]\n"
                    else:
                        content += "[green]✅ No recent alerts[/green]"
            except Exception:
                content += "[dim]Unable to read alerts[/dim]"
        else:
            content += "[dim]No alert log found[/dim]"

        return Panel(content, title="System Alerts", border_style="yellow")

    def create_operations_panel(self) -> Panel:
        """Create operations and maintenance panel"""
        content = "[bold]Operations Status[/bold]\n\n"

        # Check backup status
        backup_dir = self.project_root / "data" / "backups"
        if backup_dir.exists():
            backups = list(backup_dir.glob("*.tar.gz"))
            if backups:
                latest_backup = max(backups, key=lambda x: x.stat().st_mtime)
                backup_age = datetime.now() - datetime.fromtimestamp(latest_backup.stat().st_mtime)

                if backup_age.days == 0:
                    content += f"[green]✅ Latest Backup: {latest_backup.name} ({backup_age.seconds // 3600}h ago)[/green]\n"
                elif backup_age.days <= 1:
                    content += f"[yellow]⚠️  Latest Backup: {latest_backup.name} ({backup_age.days}d ago)[/yellow]\n"
                else:
                    content += f"[red]❌ Latest Backup: {latest_backup.name} ({backup_age.days}d ago)[/red]\n"
            else:
                content += "[red]❌ No backups found[/red]\n"
        else:
            content += "[red]❌ Backup directory not found[/red]\n"

        # Check log file sizes
        log_dir = self.project_root / "logs"
        if log_dir.exists():
            total_log_size = sum(f.stat().st_size for f in log_dir.rglob("*.log") if f.is_file())
            log_size_mb = total_log_size / (1024 * 1024)

            if log_size_mb > 1000:  # > 1GB
                content += f"[yellow]⚠️  Log Size: {log_size_mb:.1f} MB (cleanup recommended)[/yellow]\n"
            else:
                content += f"[green]✅ Log Size: {log_size_mb:.1f} MB[/green]\n"

        # Check disk space
        disk = psutil.disk_usage(str(self.project_root))
        free_space_gb = disk.free / (1024 ** 3)

        if free_space_gb < 5:
            content += f"[red]❌ Free Space: {free_space_gb:.1f} GB (critical)[/red]\n"
        elif free_space_gb < 20:
            content += f"[yellow]⚠️  Free Space: {free_space_gb:.1f} GB (low)[/yellow]\n"
        else:
            content += f"[green]✅ Free Space: {free_space_gb:.1f} GB[/green]\n"

        # Check for maintenance mode
        try:
            response = requests.get("http://localhost:8007/status", timeout=5)
            if response.status_code == 200:
                status = response.json()
                if status.get("maintenance_mode"):
                    content += "[yellow]🔧 System in maintenance mode[/yellow]\n"
                else:
                    content += "[green]✅ System operational[/green]\n"
        except:
            content += "[dim]❓ Maintenance status unknown[/dim]\n"

        return Panel(content, title="Operations", border_style="purple")

    def create_main_layout(self) -> Layout:
        """Create the main dashboard layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(name="services"),
            Layout(name="trading", size=12)
        )

        layout["right"].split_column(
            Layout(name="system", size=12),
            Layout(name="alerts"),
            Layout(name="operations", size=8)
        )

        return layout

    async def update_dashboard(self, layout: Layout):
        """Update dashboard with current data"""
        # Header
        header_text = Text()
        header_text.append("🚀 AI Trading System - Operational Dashboard", style="bold blue")
        header_text.append(f" | Environment: {self.environment.upper()}", style="bold")
        header_text.append(f" | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim")
        layout["header"].update(Panel(header_text, border_style="blue"))

        # Get data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Updating dashboard...", total=100)

            # Get service information
            progress.update(task, advance=20, description="Checking services...")
            services = [self.get_service_info(service) for service in self.all_services]

            # Get system metrics
            progress.update(task, advance=20, description="Collecting system metrics...")
            system_metrics = self.get_system_metrics()

            # Get trading metrics
            progress.update(task, advance=20, description="Collecting trading metrics...")
            trading_metrics = self.get_trading_metrics()

            progress.update(task, advance=40, description="Rendering dashboard...")

        # Update layout sections
        layout["services"].update(self.create_services_table(services))
        layout["system"].update(self.create_system_metrics_panel(system_metrics))
        layout["trading"].update(self.create_trading_metrics_panel(trading_metrics))
        layout["alerts"].update(self.create_alerts_panel())
        layout["operations"].update(self.create_operations_panel())

        # Footer with controls
        footer_text = Text()
        footer_text.append("Controls: ", style="bold")
        footer_text.append("[q]uit ", style="red")
        footer_text.append("[r]efresh ", style="green")
        footer_text.append("[s]ervices ", style="blue")
        footer_text.append("[b]ackup ", style="yellow")
        footer_text.append("[m]aintenance ", style="purple")
        footer_text.append(f"| Auto-refresh: {self.refresh_interval}s", style="dim")

        layout["footer"].update(Panel(footer_text, border_style="white"))

    def run_interactive_dashboard(self):
        """Run interactive dashboard with live updates"""
        layout = self.create_main_layout()

        with Live(layout, refresh_per_second=1, screen=True) as live:
            while True:
                try:
                    # Update dashboard
                    asyncio.run(self.update_dashboard(layout))

                    # Wait for refresh interval or user input
                    for _ in range(self.refresh_interval * 10):  # Check every 100ms
                        time.sleep(0.1)

                        # Check for user input (simplified)
                        # In a real implementation, would use proper async input handling

                except KeyboardInterrupt:
                    console.print("\n[yellow]Dashboard stopped by user[/yellow]")
                    break
                except Exception as e:
                    logger.error(f"Dashboard update error: {e}")
                    time.sleep(5)  # Wait before retrying

    def run_single_status_check(self):
        """Run a single status check and display results"""
        console.print("[bold blue]AI Trading System Status Report[/bold blue]")
        console.print(f"Environment: {self.environment}")
        console.print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        console.print()

        # Get and display service status
        console.print("[bold]Checking services...[/bold]")
        services = [self.get_service_info(service) for service in self.all_services]
        console.print(self.create_services_table(services))
        console.print()

        # Get and display system metrics
        system_metrics = self.get_system_metrics()
        console.print(self.create_system_metrics_panel(system_metrics))
        console.print()

        # Get and display trading metrics
        trading_metrics = self.get_trading_metrics()
        console.print(self.create_trading_metrics_panel(trading_metrics))
        console.print()

        # Show alerts
        console.print(self.create_alerts_panel())
        console.print()

        # Show operations status
        console.print(self.create_operations_panel())

        # Summary
        running_services = sum(1 for s in services if s.status == ServiceStatus.RUNNING)
        healthy_services = sum(1 for s in services if s.health == HealthStatus.HEALTHY)

        if running_services == len(services) and healthy_services == len(services):
            console.print("\n[bold green]✅ System Status: ALL SYSTEMS OPERATIONAL[/bold green]")
        elif running_services == len(services):
            console.print("\n[bold yellow]⚠️  System Status: RUNNING WITH ISSUES[/bold yellow]")
        else:
            console.print(f"\n[bold red]❌ System Status: DEGRADED ({running_services}/{len(services)} services running)[/bold red]")

    def generate_status_report(self, output_file: str):
        """Generate detailed status report"""
        console.print("[blue]Generating status report...[/blue]")

        # Collect all data
        services = [self.get_service_info(service) for service in self.all_services]
        system_metrics = self.get_system_metrics()
        trading_metrics = self.get_trading_metrics()

        # Create comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "host": os.uname().nodename,
            "system_metrics": asdict(system_metrics),
            "trading_metrics": asdict(trading_metrics),
            "services": [asdict(service) for service in services],
            "summary": {
                "total_services": len(services),
                "running_services": sum(1 for s in services if s.status == ServiceStatus.RUNNING),
                "healthy_services": sum(1 for s in services if s.health == HealthStatus.HEALTHY),
                "system_health_score": self._calculate_health_score(services, system_metrics)
            }
        }

        # Write report
        with open(output_file, 'w') as f:
            if output_file.endswith('.json'):
                json.dump(report, f, indent=2, default=str)
            else:
                # Text format
                f.write("AI Trading System Status Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated: {report['timestamp']}\n")
                f.write(f"Environment: {report['environment']}\n")
                f.write(f"Host: {report['host']}\n\n")

                f.write("System Metrics:\n")
                f.write(f"  CPU Usage: {system_metrics.cpu_usage:.1f}%\n")
                f.write(f"  Memory Usage: {system_metrics.memory_usage:.1f}%\n")
                f.write(f"  Disk Usage: {system_metrics.disk_usage:.1f}%\n")
                f.write(f"  Uptime: {system_metrics.uptime}\n\n")

                f.write("Trading Metrics:\n")
                f.write(f"  Active Positions: {trading_metrics.active_positions}\n")
                f.write(f"  Daily Trades: {trading_metrics.daily_trades}\n")
                f.write(f"  Daily P&L: ${trading_metrics.daily_pnl:,.2f}\n")
                f.write(f"  System Status: {trading_metrics.system_status}\n\n")

                f.write("Service Status:\n")
                for service in services:
                    f.write(f"  {service.name}: {service.status.value} ({service.health.value})\n")

        console.print(f"[green]Status report saved to: {output_file}[/green]")

    def _calculate_health_score(self, services: List[ServiceInfo], metrics: SystemMetrics) -> float:
        """Calculate overall system health score (0-100)"""
        # Service health (40% of score)
        running_services = sum(1 for s in services if s.status == ServiceStatus.RUNNING)
        healthy_services = sum(1 for s in services if s.health == HealthStatus.HEALTHY)
        service_score = (running_services + healthy_services) / (2 * len(services)) * 40

        # System resource health (30% of score)
        resource_score = 30.0
        if metrics.cpu_usage > 90:
            resource_score -= 10
        elif metrics.cpu_usage > 80:
            resource_score -= 5

        if metrics.memory_usage > 90:
            resource_score -= 10
        elif metrics.memory_usage > 80:
            resource_score -= 5

        if metrics.disk_usage > 95:
            resource_score -= 10
        elif metrics.disk_usage > 85:
            resource_score -= 5

        # Performance health (30% of score)
        performance_score = 30.0
        cpu_count = psutil.cpu_count()
        if cpu_count is not None and metrics.load_average[0] > cpu_count * 2:
            performance_score -= 15
        elif cpu_count is not None and metrics.load_average[0] > cpu_count:
            performance_score -= 5

        # Calculate total score
        total_score = service_score + resource_score + performance_score

        return max(0.0, min(100.0, total_score))



    def execute_operation(self, operation: str, **kwargs):
        """Execute operational commands"""
        console.print(f"[blue]Executing operation: {operation}[/blue]")

        try:
            if operation == "restart_service":
                service = kwargs.get("service")
                if service:
                    result = subprocess.run(
                        ["docker-compose", "-f", self.compose_file, "restart", service],
                        cwd=self.project_root,
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        console.print(f"[green]✅ Service {service} restarted successfully[/green]")
                    else:
                        console.print(f"[red]❌ Failed to restart {service}: {result.stderr}[/red]")

            elif operation == "create_backup":
                result = subprocess.run(
                    ["bash", "scripts/backup/backup.sh", "--type", "manual"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    console.print("[green]✅ Backup created successfully[/green]")
                else:
                    console.print(f"[red]❌ Backup failed: {result.stderr}[/red]")

            elif operation == "enter_maintenance":
                try:
                    response = requests.post(
                        "http://localhost:8007/maintenance/enter",
                        json={"message": "Maintenance via dashboard", "initiated_by": os.getlogin()},
                        timeout=10
                    )
                    if response.status_code == 200:
                        console.print("[yellow]🔧 Maintenance mode activated[/yellow]")
                    else:
                        console.print("[red]❌ Failed to enter maintenance mode[/red]")
                except:
                    console.print("[red]❌ Maintenance service not available[/red]")

            elif operation == "exit_maintenance":
                try:
                    response = requests.post(
                        "http://localhost:8007/maintenance/exit",
                        timeout=10
                    )
                    if response.status_code == 200:
                        console.print("[green]✅ Maintenance mode deactivated[/green]")
                    else:
                        console.print("[red]❌ Failed to exit maintenance mode[/red]")
                except:
                    console.print("[red]❌ Maintenance service not available[/red]")

        except Exception as e:
            console.print(f"[red]❌ Operation failed: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="AI Trading System Operational Dashboard")
    parser.add_argument("--env", choices=["development", "staging", "production"],
                       default="development", help="Environment")
    parser.add_argument("--refresh-interval", type=int, default=30,
                       help="Dashboard refresh interval in seconds")
    parser.add_argument("--mode", choices=["interactive", "status", "report"],
                       default="interactive", help="Dashboard mode")
    parser.add_argument("--output", help="Output file for report mode")
    parser.add_argument("--format", choices=["json", "text"], default="text",
                       help="Report format")

    args = parser.parse_args()

    # Create dashboard instance
    dashboard = OperationalDashboard(
        environment=args.env,
        refresh_interval=args.refresh_interval
    )

    try:
        if args.mode == "interactive":
            console.print("[bold green]Starting AI Trading System Dashboard...[/bold green]")
            dashboard.run_interactive_dashboard()

        elif args.mode == "status":
            dashboard.run_single_status_check()

        elif args.mode == "report":
            output_file = args.output or f"status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{args.format}"
            dashboard.generate_status_report(output_file)

    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Dashboard error: {e}[/red]")
        logger.error(f"Dashboard error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
