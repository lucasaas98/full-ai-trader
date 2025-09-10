"""
Command Line Interface for the Trading Scheduler.

This module provides a comprehensive CLI for controlling and monitoring
the trading system scheduler and all its components.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import httpx
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(help="Trading Scheduler CLI")
console = Console()

# Default scheduler URL
SCHEDULER_URL = "http://localhost:8000"


class SchedulerClient:
    """Client for communicating with the scheduler service."""

    def __init__(self, base_url: str = SCHEDULER_URL):
        self.base_url = base_url.rstrip("/")

    async def _make_request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Make HTTP request to scheduler API."""
        url = f"{self.base_url}{endpoint}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            console.print(
                f"[red]HTTP Error {e.response.status_code}: {e.response.text}[/red]"
            )
            raise typer.Exit(1)
        except httpx.RequestError as e:
            console.print(f"[red]Request Error: {e}[/red]")
            raise typer.Exit(1)

    async def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return await self._make_request("GET", "/status")

    async def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return await self._make_request("GET", "/performance")

    async def trigger_task(
        self, task_id: str, priority: str = "normal"
    ) -> Dict[str, Any]:
        """Trigger a task."""
        return await self._make_request(
            "POST", f"/tasks/{task_id}/trigger", json={"priority": priority}
        )

    async def pause_task(self, task_id: str) -> Dict[str, Any]:
        """Pause a task."""
        return await self._make_request("POST", f"/tasks/{task_id}/pause")

    async def resume_task(self, task_id: str) -> Dict[str, Any]:
        """Resume a task."""
        return await self._make_request("POST", f"/tasks/{task_id}/resume")

    async def restart_service(self, service_name: str) -> Dict[str, Any]:
        """Restart a service."""
        return await self._make_request("POST", f"/services/{service_name}/restart")

    async def set_maintenance_mode(self, enabled: bool) -> Dict[str, Any]:
        """Set maintenance mode."""
        return await self._make_request(
            "POST", "/system/maintenance", json={"enabled": enabled}
        )

    async def emergency_stop(self) -> Dict[str, Any]:
        """Trigger emergency stop."""
        return await self._make_request("POST", "/system/emergency-stop")

    async def resume_trading(self) -> Dict[str, Any]:
        """Resume trading."""
        return await self._make_request("POST", "/system/resume-trading")

    async def trigger_pipeline(self, reason: str = "manual") -> Dict[str, Any]:
        """Trigger data pipeline."""
        return await self._make_request(
            "POST", "/pipeline/trigger", json={"reason": reason}
        )

    async def get_logs(self, lines: int = 100) -> Dict[str, Any]:
        """Get recent logs."""
        return await self._make_request("GET", f"/logs?lines={lines}")


def run_async(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to run async functions in typer commands."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


@app.command()
@run_async
async def status(
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed status"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Show system status."""
    client = SchedulerClient(url)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching system status...", total=None)
        status_data = await client.get_status()
        progress.remove_task(task)

    if json_output:
        console.print(json.dumps(status_data, indent=2))
        return

    # Display status in a nice format
    scheduler_info = status_data.get("scheduler", {})
    market_info = status_data.get("market", {})
    services = status_data.get("services", {})
    tasks = status_data.get("tasks", {})
    metrics = status_data.get("metrics", {})

    # System overview
    console.print(
        Panel.fit(
            f"[bold green]Trading System Status[/bold green]\n"
            f"Scheduler Running: [{'green' if scheduler_info.get('running') else 'red'}]{scheduler_info.get('running')}[/]\n"
            f"Maintenance Mode: [{'yellow' if scheduler_info.get('maintenance_mode') else 'green'}]{scheduler_info.get('maintenance_mode')}[/]\n"
            f"Emergency Stop: [{'red' if scheduler_info.get('emergency_stop') else 'green'}]{scheduler_info.get('emergency_stop')}[/]",
            title="System Overview",
        )
    )

    # Market information
    console.print(
        Panel.fit(
            f"Market Session: [bold]{market_info.get('session', 'unknown').title()}[/bold]\n"
            f"Trading Day: [{'green' if market_info.get('is_trading_day') else 'red'}]{market_info.get('is_trading_day')}[/]\n"
            f"Next Open: {market_info.get('next_open', 'N/A')}\n"
            f"Next Close: {market_info.get('next_close', 'N/A')}",
            title="Market Information",
        )
    )

    # Services status
    services_table = Table(title="Services Status")
    services_table.add_column("Service", style="cyan")
    services_table.add_column("Status", style="magenta")
    services_table.add_column("Errors", style="red")
    services_table.add_column("Restarts", style="yellow")
    services_table.add_column("Last Check", style="dim")

    for name, service in services.items():
        status_color = {
            "running": "green",
            "stopped": "red",
            "error": "red",
            "starting": "yellow",
            "stopping": "yellow",
        }.get(service.get("status"), "white")

        services_table.add_row(
            name,
            f"[{status_color}]{service.get('status', 'unknown')}[/]",
            str(service.get("error_count", 0)),
            str(service.get("restart_count", 0)),
            service.get("last_check", "Never"),
        )

    console.print(services_table)

    if detailed:
        # Tasks status
        tasks_table = Table(title="Tasks Status")
        tasks_table.add_column("Task", style="cyan")
        tasks_table.add_column("Enabled", style="magenta")
        tasks_table.add_column("Errors", style="red")
        tasks_table.add_column("Last Run", style="dim")
        tasks_table.add_column("Last Success", style="green")

        for task_id, task in tasks.items():
            tasks_table.add_row(
                task_id,
                f"[{'green' if task.get('enabled') else 'red'}]{task.get('enabled')}[/]",
                str(task.get("error_count", 0)),
                task.get("last_run", "Never"),
                task.get("last_success", "Never"),
            )

        console.print(tasks_table)

        # System metrics
        if metrics:
            console.print(
                Panel.fit(
                    f"CPU: {metrics.get('cpu_percent', 0):.1f}%\n"
                    f"Memory: {metrics.get('memory', {}).get('percent', 0):.1f}%\n"
                    f"Disk: {metrics.get('disk', {}).get('percent', 0):.1f}%\n"
                    f"Redis Clients: {metrics.get('redis', {}).get('connected_clients', 0)}",
                    title="System Metrics",
                )
            )


@app.command()
@run_async
async def tasks(
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL")
) -> None:
    """List all tasks."""
    client = SchedulerClient(url)
    status_data = await client.get_status()
    tasks = status_data.get("tasks", {})

    table = Table(title="Scheduled Tasks")
    table.add_column("Task ID", style="cyan")
    table.add_column("Enabled", style="magenta")
    table.add_column("Last Run", style="dim")
    table.add_column("Last Success", style="green")
    table.add_column("Error Count", style="red")
    table.add_column("Retry Count", style="yellow")

    for task_id, task in tasks.items():
        table.add_row(
            task_id,
            f"[{'green' if task.get('enabled') else 'red'}]{task.get('enabled')}[/]",
            task.get("last_run", "Never"),
            task.get("last_success", "Never"),
            str(task.get("error_count", 0)),
            str(task.get("retry_count", 0)),
        )

    console.print(table)


@app.command()
@run_async
async def trigger(
    task_id: str = typer.Argument(..., help="Task ID to trigger"),
    priority: str = typer.Option("normal", "--priority", "-p", help="Task priority"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Manually trigger a task."""
    client = SchedulerClient(url)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Triggering task {task_id}...", total=None)
        result = await client.trigger_task(task_id, priority)
        progress.remove_task(task)

    if result.get("status") == "success":
        console.print(f"[green]✓[/green] {result.get('message')}")
    else:
        console.print(f"[red]✗[/red] {result.get('message')}")


@app.command()
@run_async
async def pause(
    task_id: str = typer.Argument(..., help="Task ID to pause"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Pause a scheduled task."""
    client = SchedulerClient(url)
    result = await client.pause_task(task_id)

    if result.get("status") == "success":
        console.print(f"[green]✓[/green] {result.get('message')}")
    else:
        console.print(f"[red]✗[/red] {result.get('message')}")


@app.command()
@run_async
async def resume(
    task_id: str = typer.Argument(..., help="Task ID to resume"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Resume a paused task."""
    client = SchedulerClient(url)
    result = await client.resume_task(task_id)

    if result.get("status") == "success":
        console.print(f"[green]✓[/green] {result.get('message')}")
    else:
        console.print(f"[red]✗[/red] {result.get('message')}")


@app.command()
@run_async
async def services(
    action: str = typer.Argument(None, help="Action: list, restart <service>"),
    service_name: str = typer.Argument(None, help="Service name for restart"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Manage services."""
    client = SchedulerClient(url)

    if action == "restart" and service_name:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Restarting service {service_name}...", total=None
            )
            result = await client.restart_service(service_name)
            progress.remove_task(task)

        if result.get("status") == "success":
            console.print(f"[green]✓[/green] {result.get('message')}")
        else:
            console.print(f"[red]✗[/red] {result.get('message')}")
    else:
        # List services
        status_data = await client.get_status()
        services_data = status_data.get("services", {})

        table = Table(title="Services")
        table.add_column("Service", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Error Count", style="red")
        table.add_column("Restart Count", style="yellow")
        table.add_column("Last Check", style="dim")

        for name, service in services_data.items():
            status_color = {
                "running": "green",
                "stopped": "red",
                "error": "red",
                "starting": "yellow",
                "stopping": "yellow",
            }.get(service.get("status"), "white")

            table.add_row(
                name,
                f"[{status_color}]{service.get('status', 'unknown')}[/]",
                str(service.get("error_count", 0)),
                str(service.get("restart_count", 0)),
                service.get("last_check", "Never"),
            )

        console.print(table)


@app.command()
@run_async
async def maintenance(
    enable: bool = typer.Option(
        None, "--enable/--disable", help="Enable or disable maintenance mode"
    ),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Manage maintenance mode."""
    if enable is None:
        # Show current status
        client = SchedulerClient(url)
        status_data = await client.get_status()
        maintenance_mode = status_data.get("scheduler", {}).get(
            "maintenance_mode", False
        )

        status_text = (
            "[yellow]ENABLED[/yellow]"
            if maintenance_mode
            else "[green]DISABLED[/green]"
        )
        console.print(f"Maintenance Mode: {status_text}")
        return

    client = SchedulerClient(url)
    result = await client.set_maintenance_mode(enable)

    if result.get("status") == "success":
        console.print(f"[green]✓[/green] {result.get('message')}")
    else:
        console.print(f"[red]✗[/red] {result.get('message')}")


@app.command()
@run_async
async def emergency(
    action: str = typer.Argument(..., help="Action: stop, resume"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Emergency trading controls."""
    client = SchedulerClient(url)

    if action == "stop":
        console.print("[red]⚠️  EMERGENCY STOP - Are you sure?[/red]")
        confirm = typer.confirm("This will halt all trading activities!")

        if confirm:
            result = await client.emergency_stop()
            if result.get("status") == "success":
                console.print(f"[red]🛑 {result.get('message')}[/red]")
            else:
                console.print(f"[red]✗[/red] {result.get('message')}")

    elif action == "resume":
        result = await client.resume_trading()
        if result.get("status") == "success":
            console.print(f"[green]✓[/green] {result.get('message')}")
        else:
            console.print(f"[red]✗[/red] {result.get('message')}")

    else:
        console.print("[red]Invalid action. Use 'stop' or 'resume'[/red]")


@app.command()
@run_async
async def pipeline(
    action: str = typer.Argument("trigger", help="Action: trigger"),
    reason: str = typer.Option(
        "manual", "--reason", "-r", help="Reason for triggering"
    ),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Control data pipeline."""
    client = SchedulerClient(url)

    if action == "trigger":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Triggering data pipeline...", total=None)
            result = await client.trigger_pipeline(reason)
            progress.remove_task(task)

        if result.get("status") == "success":
            console.print(f"[green]✓[/green] {result.get('message')}")
        else:
            console.print(f"[red]✗[/red] {result.get('message')}")
    else:
        console.print("[red]Invalid action. Use 'trigger'[/red]")


@app.command()
@run_async
async def metrics(
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch metrics in real-time"
    ),
    interval: int = typer.Option(
        5, "--interval", "-i", help="Update interval for watch mode"
    ),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Show system metrics."""
    client = SchedulerClient(url)

    async def show_metrics() -> None:
        performance_data = await client.get_performance()
        status_data = await client.get_status()
        metrics_data = status_data.get("metrics", {})

        # System resources
        console.print(
            Panel.fit(
                f"CPU: {metrics_data.get('cpu_percent', 0):.1f}%\n"
                f"Memory: {metrics_data.get('memory', {}).get('percent', 0):.1f}% "
                f"({metrics_data.get('memory', {}).get('used', 0) / 1024**3:.1f}GB / "
                f"{metrics_data.get('memory', {}).get('total', 0) / 1024**3:.1f}GB)\n"
                f"Disk: {metrics_data.get('disk', {}).get('percent', 0):.1f}%",
                title="System Resources",
            )
        )

        # Redis metrics
        redis_metrics = metrics_data.get("redis", {})
        console.print(
            Panel.fit(
                f"Connected Clients: {redis_metrics.get('connected_clients', 0)}\n"
                f"Memory Used: {redis_metrics.get('used_memory_human', '0')}\n"
                f"Keyspace Hits: {redis_metrics.get('keyspace_hits', 0)}\n"
                f"Keyspace Misses: {redis_metrics.get('keyspace_misses', 0)}",
                title="Redis Metrics",
            )
        )

        # Task execution times
        exec_times = performance_data.get("task_execution_times", {})
        if exec_times:
            table = Table(title="Task Performance")
            table.add_column("Task", style="cyan")
            table.add_column("Last Execution Time", style="green")

            for task_id, exec_time in exec_times.items():
                table.add_row(task_id, f"{exec_time:.2f}s")

            console.print(table)

        # Queue lengths
        queues = status_data.get("queues", {})
        if queues:
            console.print(
                Panel.fit(
                    "\n".join(
                        [f"{queue}: {length}" for queue, length in queues.items()]
                    ),
                    title="Task Queue Lengths",
                )
            )

    if watch:
        try:
            while True:
                console.clear()
                console.print(
                    f"[dim]Last updated: {datetime.now().strftime('%H:%M:%S')}[/dim]"
                )
                await show_metrics()
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped watching metrics[/yellow]")
    else:
        await show_metrics()


@app.command()
@run_async
async def logs(
    lines: int = typer.Option(
        100, "--lines", "-n", help="Number of recent log lines to show"
    ),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Show system logs."""
    client = SchedulerClient(url)

    if follow:
        console.print("[yellow]Following logs (Ctrl+C to stop)...[/yellow]")
        try:
            while True:
                logs_data = await client.get_logs(lines=50)  # Get recent logs
                logs = logs_data.get("logs", [])

                for log_entry in logs[-10:]:  # Show last 10 entries
                    console.print(log_entry)

                await asyncio.sleep(2)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped following logs[/yellow]")
    else:
        logs_data = await client.get_logs(lines)
        logs = logs_data.get("logs", [])

        for log_entry in logs:
            console.print(log_entry)


@app.command()
@run_async
async def positions(
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL")
) -> None:
    """Show current positions."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url.replace('8000', '9104')}/positions")
            response.raise_for_status()
            positions_data = response.json()

        if not positions_data.get("positions"):
            console.print("[yellow]No open positions[/yellow]")
            return

        table = Table(title="Current Positions")
        table.add_column("Symbol", style="cyan")
        table.add_column("Side", style="magenta")
        table.add_column("Quantity", style="green")
        table.add_column("Entry Price", style="blue")
        table.add_column("Current Price", style="blue")
        table.add_column("Unrealized P&L", style="red")
        table.add_column("Unrealized %", style="red")

        for position in positions_data.get("positions", []):
            pnl_color = "green" if position.get("unrealized_pnl", 0) >= 0 else "red"
            pnl_pct_color = (
                "green" if position.get("unrealized_pnl_percent", 0) >= 0 else "red"
            )

            table.add_row(
                position.get("symbol", ""),
                position.get("side", ""),
                str(position.get("quantity", 0)),
                f"${position.get('entry_price', 0):.2f}",
                f"${position.get('current_price', 0):.2f}",
                f"[{pnl_color}]${position.get('unrealized_pnl', 0):.2f}[/]",
                f"[{pnl_pct_color}]{position.get('unrealized_pnl_percent', 0):.2f}%[/]",
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error fetching positions: {e}[/red]")


@app.command()
@run_async
async def portfolio(
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL")
) -> None:
    """Show portfolio summary."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{url.replace('8000', '9104')}/portfolio")
            response.raise_for_status()
            portfolio_data = response.json()

        console.print(
            Panel.fit(
                f"Total Value: [green]${portfolio_data.get('total_value', 0):.2f}[/green]\n"
                f"Cash: [blue]${portfolio_data.get('cash', 0):.2f}[/blue]\n"
                f"Day P&L: [{'green' if portfolio_data.get('day_pnl', 0) >= 0 else 'red'}]"
                f"${portfolio_data.get('day_pnl', 0):.2f}[/]\n"
                f"Total P&L: [{'green' if portfolio_data.get('total_pnl', 0) >= 0 else 'red'}]"
                f"${portfolio_data.get('total_pnl', 0):.2f}[/]",
                title="Portfolio Summary",
            )
        )

    except Exception as e:
        console.print(f"[red]Error fetching portfolio: {e}[/red]")


@app.command()
@run_async
async def export(
    format: str = typer.Option(
        "json", "--format", "-f", help="Export format: json, csv, yaml"
    ),
    output: str = typer.Option(None, "--output", "-o", help="Output file path"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Export trading data for TradeNote or other analysis."""
    try:
        # Get trades data
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f"{url.replace('8000', '9104')}/trades/export"
            )
            response.raise_for_status()
            trades_data = response.json()

        if not trades_data.get("trades"):
            console.print("[yellow]No trades to export[/yellow]")
            return

        # Format data based on requested format
        if format.lower() == "json":
            export_data = json.dumps(trades_data, indent=2)
        elif format.lower() == "yaml":
            export_data = yaml.dump(trades_data, default_flow_style=False)
        elif format.lower() == "csv":
            import pandas as pd

            df = pd.DataFrame(trades_data.get("trades", []))
            export_data = df.to_csv(index=False)
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            return

        if output:
            with open(output, "w") as f:
                f.write(export_data)
            console.print(f"[green]✓[/green] Data exported to {output}")
        else:
            console.print(export_data)

    except Exception as e:
        console.print(f"[red]Export failed: {e}[/red]")


@app.command()
@run_async
async def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    reload: bool = typer.Option(False, "--reload", help="Reload configuration"),
    set_param: str = typer.Option(None, "--set", help="Set parameter: key=value"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Manage configuration."""
    if show:
        try:
            async with httpx.AsyncClient() as client_http:
                response = await client_http.get(f"{url}/config")
                response.raise_for_status()
                config_data = response.json()

            # Display configuration as a tree
            tree = Tree("Configuration")

            def add_to_tree(
                parent: Tree, data: Dict[str, Any], prefix: str = ""
            ) -> None:
                for key, value in data.items():
                    if isinstance(value, dict):
                        branch = parent.add(f"[cyan]{key}[/cyan]")
                        add_to_tree(branch, value, f"{prefix}{key}.")
                    else:
                        parent.add(f"[cyan]{key}[/cyan]: [green]{value}[/green]")

            add_to_tree(tree, config_data)
            console.print(tree)

        except Exception as e:
            console.print(f"[red]Error fetching configuration: {e}[/red]")

    elif reload:
        try:
            async with httpx.AsyncClient() as client_http:
                response = await client_http.post(f"{url}/config/reload")
                response.raise_for_status()
                result = response.json()

            if result.get("status") == "success":
                console.print(f"[green]✓[/green] {result.get('message')}")
            else:
                console.print(f"[red]✗[/red] {result.get('message')}")

        except Exception as e:
            console.print(f"[red]Config reload failed: {e}[/red]")

    elif set_param:
        try:
            if "=" not in set_param:
                console.print("[red]Invalid format. Use key=value[/red]")
                return

            key, value = set_param.split("=", 1)

            async with httpx.AsyncClient() as client_http:
                response = await client_http.post(
                    f"{url}/config/update", json={key: value}
                )
                response.raise_for_status()
                result = response.json()

            if result.get("status") == "success":
                console.print(f"[green]✓[/green] Updated {key} = {value}")
            else:
                console.print(f"[red]✗[/red] {result.get('message')}")

        except Exception as e:
            console.print(f"[red]Config update failed: {e}[/red]")


@app.command()
@run_async
async def monitor(
    interval: int = typer.Option(
        5, "--interval", "-i", help="Update interval in seconds"
    ),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Real-time system monitoring dashboard."""
    client = SchedulerClient(url)

    try:
        while True:
            console.clear()
            console.print(
                f"[bold blue]Trading System Monitor[/bold blue] - {datetime.now().strftime('%H:%M:%S')}"
            )
            console.print()

            # Get status and metrics
            status_data = await client.get_status()

            # Scheduler status
            scheduler_info = status_data.get("scheduler", {})
            market_info = status_data.get("market", {})

            # Status indicators
            status_indicators = []
            if scheduler_info.get("running"):
                status_indicators.append("[green]●[/green] Running")
            else:
                status_indicators.append("[red]●[/red] Stopped")

            if scheduler_info.get("maintenance_mode"):
                status_indicators.append("[yellow]🔧[/yellow] Maintenance")

            if scheduler_info.get("emergency_stop"):
                status_indicators.append("[red]🛑[/red] Emergency Stop")

            console.print(" ".join(status_indicators))
            console.print(
                f"Market: [bold]{market_info.get('session', 'unknown').title()}[/bold]"
            )
            console.print()

            # Services grid
            services = status_data.get("services", {})
            services_table = Table(
                title="Services", show_header=True, header_style="bold magenta"
            )
            services_table.add_column("Service", style="cyan", width=20)
            services_table.add_column("Status", style="white", width=10)
            services_table.add_column("Errors", style="red", width=8)

            for name, service in services.items():
                status_emoji = {
                    "running": "[green]✓[/green]",
                    "stopped": "[red]✗[/red]",
                    "error": "[red]⚠[/red]",
                    "starting": "[yellow]⟳[/yellow]",
                    "stopping": "[yellow]⟳[/yellow]",
                }.get(service.get("status"), "[white]?[/white]")

                services_table.add_row(
                    name, status_emoji, str(service.get("error_count", 0))
                )

            console.print(services_table)

            # Quick metrics
            metrics_data = status_data.get("metrics", {})
            console.print(
                Panel.fit(
                    f"CPU: {metrics_data.get('cpu_percent', 0):.1f}% | "
                    f"Memory: {metrics_data.get('memory', {}).get('percent', 0):.1f}% | "
                    f"Disk: {metrics_data.get('disk', {}).get('percent', 0):.1f}%",
                    title="System Resources",
                )
            )

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]Monitoring error: {e}[/red]")


@app.command()
@run_async
async def strategy(
    action: str = typer.Argument(
        ..., help="Action: list, enable <strategy>, disable <strategy>"
    ),
    strategy_name: str = typer.Argument(None, help="Strategy name"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Manage trading strategies."""
    try:
        async with httpx.AsyncClient() as client:
            if action == "list":
                response = await client.get(f"{url.replace('8000', '9102')}/strategies")
                response.raise_for_status()
                strategies_data = response.json()

                table = Table(title="Trading Strategies")
                table.add_column("Strategy", style="cyan")
                table.add_column("Enabled", style="magenta")
                table.add_column("Performance", style="green")
                table.add_column("Active Signals", style="blue")

                for strategy in strategies_data.get("strategies", []):
                    enabled_color = "green" if strategy.get("enabled") else "red"
                    performance = strategy.get("performance", {})
                    pnl = performance.get("total_pnl", 0)
                    pnl_color = "green" if pnl >= 0 else "red"

                    table.add_row(
                        strategy.get("name", ""),
                        f"[{enabled_color}]{strategy.get('enabled')}[/]",
                        f"[{pnl_color}]{pnl:.2f}%[/]",
                        str(strategy.get("active_signals", 0)),
                    )

                console.print(table)

            elif action in ["enable", "disable"] and strategy_name:
                enabled = action == "enable"
                response = await client.post(
                    f"{url.replace('8000', '9102')}/strategies/{strategy_name}/toggle",
                    json={"enabled": enabled},
                )
                response.raise_for_status()
                result = response.json()

                if result.get("status") == "success":
                    console.print(
                        f"[green]✓[/green] Strategy {strategy_name} {action}d"
                    )
                else:
                    console.print(f"[red]✗[/red] {result.get('message')}")

            else:
                console.print("[red]Invalid action or missing strategy name[/red]")

    except Exception as e:
        console.print(f"[red]Strategy management error: {e}[/red]")


@app.command()
@run_async
async def risk(
    action: str = typer.Argument("status", help="Action: status, limits, alerts"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Risk management information."""
    try:
        async with httpx.AsyncClient() as client:
            if action == "status":
                response = await client.get(
                    f"{url.replace('8000', '9103')}/risk/status"
                )
                response.raise_for_status()
                risk_data = response.json()

                console.print(
                    Panel.fit(
                        f"Portfolio Risk: [{'red' if risk_data.get('portfolio_risk', 0) > 0.8 else 'green'}]"
                        f"{risk_data.get('portfolio_risk', 0):.1%}[/]\n"
                        f"Daily Trades: {risk_data.get('daily_trades', 0)}\n"
                        f"Max Drawdown: [{'red' if risk_data.get('max_drawdown', 0) > 0.1 else 'green'}]"
                        f"{risk_data.get('max_drawdown', 0):.1%}[/]\n"
                        f"Risk Score: [{'red' if risk_data.get('risk_score', 0) > 70 else 'green'}]"
                        f"{risk_data.get('risk_score', 0):.0f}/100[/]",
                        title="Risk Status",
                    )
                )

            elif action == "limits":
                response = await client.get(
                    f"{url.replace('8000', '9103')}/risk/limits"
                )
                response.raise_for_status()
                limits_data = response.json()

                table = Table(title="Risk Limits")
                table.add_column("Parameter", style="cyan")
                table.add_column("Current", style="blue")
                table.add_column("Limit", style="red")
                table.add_column("Status", style="green")

                for limit in limits_data.get("limits", []):
                    current = limit.get("current", 0)
                    max_limit = limit.get("limit", 0)
                    status = (
                        "OK"
                        if current < max_limit * 0.8
                        else "WARNING" if current < max_limit else "BREACH"
                    )
                    status_color = {
                        "OK": "green",
                        "WARNING": "yellow",
                        "BREACH": "red",
                    }.get(status, "white")

                    table.add_row(
                        limit.get("name", ""),
                        f"{current:.2f}",
                        f"{max_limit:.2f}",
                        f"[{status_color}]{status}[/]",
                    )

                console.print(table)

            elif action == "alerts":
                response = await client.get(
                    f"{url.replace('8000', '9103')}/risk/alerts"
                )
                response.raise_for_status()
                alerts_data = response.json()

                if not alerts_data.get("alerts"):
                    console.print("[green]No active risk alerts[/green]")
                    return

                table = Table(title="Risk Alerts")
                table.add_column("Time", style="dim")
                table.add_column("Severity", style="red")
                table.add_column("Message", style="white")

                for alert in alerts_data.get("alerts", []):
                    severity_color = {
                        "low": "green",
                        "medium": "yellow",
                        "high": "red",
                        "critical": "red bold",
                    }.get(alert.get("severity"), "white")

                    table.add_row(
                        alert.get("timestamp", ""),
                        f"[{severity_color}]{alert.get('severity', '').upper()}[/]",
                        alert.get("message", ""),
                    )

                console.print(table)

    except Exception as e:
        console.print(f"[red]Risk information error: {e}[/red]")


@app.command()
@run_async
async def backtest(
    strategy: str = typer.Argument(..., help="Strategy name to backtest"),
    start_date: str = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    symbols: str = typer.Option(None, "--symbols", help="Comma-separated symbols"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Run strategy backtests."""
    try:
        backtest_params = {"strategy": strategy}
        if start_date:
            backtest_params["start_date"] = start_date
        if end_date:
            backtest_params["end_date"] = end_date
        if symbols:
            backtest_params["symbols"] = symbols

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running backtest...", total=None)

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{url.replace('8000', '9102')}/backtest",
                    json=backtest_params,
                    timeout=300.0,
                )
                response.raise_for_status()
                backtest_data = response.json()

            progress.remove_task(task)

        # Display results
        results = backtest_data.get("results", {})
        console.print(
            Panel.fit(
                f"Total Return: [{'green' if results.get('total_return', 0) >= 0 else 'red'}]"
                f"{results.get('total_return', 0):.2f}%[/]\n"
                f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}\n"
                f"Max Drawdown: [red]{results.get('max_drawdown', 0):.2f}%[/red]\n"
                f"Win Rate: {results.get('win_rate', 0):.1f}%\n"
                f"Total Trades: {results.get('total_trades', 0)}",
                title=f"Backtest Results - {strategy}",
            )
        )

    except Exception as e:
        console.print(f"[red]Backtest error: {e}[/red]")


@app.command()
@run_async
async def alerts(
    action: str = typer.Argument("list", help="Action: list, clear"),
    severity: str = typer.Option(None, "--severity", help="Filter by severity"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Manage system alerts."""
    try:
        async with httpx.AsyncClient() as client:
            if action == "list":
                response = await client.get(f"{url}/alerts")
                if severity:
                    response = await client.get(f"{url}/alerts?severity={severity}")
                response.raise_for_status()
                alerts_data = response.json()

                if not alerts_data.get("alerts"):
                    console.print("[green]No alerts[/green]")
                    return

                table = Table(title="System Alerts")
                table.add_column("Time", style="dim")
                table.add_column("Service", style="cyan")
                table.add_column("Severity", style="red")
                table.add_column("Message", style="white")

                for alert in alerts_data.get("alerts", []):
                    severity_color = {
                        "low": "green",
                        "medium": "yellow",
                        "high": "red",
                        "critical": "red bold",
                    }.get(alert.get("severity"), "white")

                    table.add_row(
                        alert.get("timestamp", ""),
                        alert.get("service", ""),
                        f"[{severity_color}]{alert.get('severity', '').upper()}[/]",
                        alert.get("message", ""),
                    )

                console.print(table)

            elif action == "clear":
                response = await client.delete(f"{url}/alerts")
                response.raise_for_status()
                console.print("[green]✓[/green] Alerts cleared")

    except Exception as e:
        console.print(f"[red]Alerts error: {e}[/red]")


@app.command()
@run_async
async def health(
    service: str = typer.Argument(None, help="Specific service to check"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Check service health."""
    client = SchedulerClient(url)
    status_data = await client.get_status()

    if service:
        # Check specific service
        services = status_data.get("services", {})
        if service not in services:
            console.print(f"[red]Service '{service}' not found[/red]")
            return

        service_info = services[service]
        status = service_info.get("status", "unknown")
        status_color = {
            "running": "green",
            "stopped": "red",
            "error": "red",
            "starting": "yellow",
            "stopping": "yellow",
        }.get(status, "white")

        console.print(
            Panel.fit(
                f"Status: [{status_color}]{status.title()}[/]\n"
                f"Error Count: {service_info.get('error_count', 0)}\n"
                f"Restart Count: {service_info.get('restart_count', 0)}\n"
                f"Last Check: {service_info.get('last_check', 'Never')}",
                title=f"Service Health - {service}",
            )
        )
    else:
        # Show all services health
        services = status_data.get("services", {})
        healthy_count = sum(
            1 for s in services.values() if s.get("status") == "running"
        )
        total_count = len(services)

        health_color = "green" if healthy_count == total_count else "red"
        console.print(
            f"System Health: [{health_color}]{healthy_count}/{total_count} services healthy[/]"
        )

        for name, service in services.items():
            if isinstance(service, dict):
                status = service.get("status", "unknown")
            else:
                status = str(service) if service else "unknown"
            status_emoji = {
                "running": "[green]✓[/green]",
                "stopped": "[red]✗[/red]",
                "error": "[red]⚠[/red]",
            }.get(status, "[white]?[/white]")

            console.print(f"  {status_emoji} {name}")


@app.command()
def dashboard(
    refresh: int = typer.Option(
        3, "--refresh", "-r", help="Refresh interval in seconds"
    ),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Launch comprehensive interactive dashboard."""

    async def _run_dashboard() -> None:
        """Internal async function to run the dashboard."""
        try:
            from rich.align import Align
            from rich.layout import Layout
            from rich.live import Live

            console.print(
                "[bold blue]🚀 Starting AI Trading System Dashboard[/bold blue]"
            )
            console.print("Press Ctrl+C to exit")
            console.print()

            client = SchedulerClient(url)

            def create_dashboard_layout() -> Layout:
                """Create the main dashboard layout."""
                layout = Layout()

                # Split into header, body, and footer
                layout.split_column(
                    Layout(name="header", size=3),
                    Layout(name="body"),
                    Layout(name="footer", size=5),
                )

                # Split body into left and right columns
                layout["body"].split_row(Layout(name="left"), Layout(name="right"))

                # Split left column into system and services
                layout["left"].split_column(
                    Layout(name="system", size=12), Layout(name="services")
                )

                # Split right column into trading and portfolio
                layout["right"].split_column(
                    Layout(name="trading", size=15), Layout(name="portfolio")
                )

                return layout

            async def get_dashboard_data() -> dict:
                """Fetch all dashboard data."""
                try:
                    # Core system data
                    status_data = await client.get_status()
                    performance_data = await client.get_performance()

                    # Additional service data
                    trading_metrics = {}
                    portfolio_data = {}
                    risk_data = {}

                    # Try to get additional data from other services
                    try:
                        async with httpx.AsyncClient(timeout=2.0) as http_client:
                            # Trading metrics from trade executor
                            try:
                                response = await http_client.get(
                                    "http://localhost:9104/metrics"
                                )
                                if response.status_code == 200:
                                    trading_metrics = response.json()
                            except Exception:
                                pass

                            # Portfolio data
                            try:
                                response = await http_client.get(
                                    "http://localhost:9104/portfolio"
                                )
                                if response.status_code == 200:
                                    portfolio_data = response.json()
                            except Exception:
                                pass

                            # Risk data
                            try:
                                response = await http_client.get(
                                    "http://localhost:9103/risk/status"
                                )
                                if response.status_code == 200:
                                    risk_data = response.json()
                            except Exception:
                                pass

                    except Exception:
                        pass

                    return {
                        "status": status_data,
                        "performance": performance_data,
                        "trading": trading_metrics,
                        "portfolio": portfolio_data,
                        "risk": risk_data,
                    }
                except Exception as e:
                    return {
                        "status": {"error": str(e)},
                        "performance": {},
                        "trading": {},
                        "portfolio": {},
                        "risk": {},
                    }

            def render_header(data: Dict[str, Any]) -> Panel:
                """Render dashboard header."""
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status_data = data.get("status", {})

                # System status indicators
                indicators = []
                scheduler_info = status_data.get("scheduler", {})

                if scheduler_info.get("running"):
                    indicators.append("[green]●[/green] Running")
                else:
                    indicators.append("[red]●[/red] Stopped")

                if scheduler_info.get("maintenance_mode"):
                    indicators.append("[yellow]🔧[/yellow] Maintenance")

                if scheduler_info.get("emergency_stop"):
                    indicators.append("[red]🛑[/red] Emergency Stop")

                market_info = status_data.get("market", {})
                market_session = market_info.get("session", "unknown").title()

                header_text = f"[bold blue]🚀 AI Trading System Dashboard[/bold blue] | {' '.join(indicators)} | Market: [bold]{market_session}[/bold] | {timestamp}"

                return Panel(Align.center(header_text), border_style="blue")

            def render_system_panel(data: Dict[str, Any]) -> Panel:
                """Render system metrics panel."""
                status_data = data.get("status", {})
                metrics = status_data.get("metrics", {})

                content = []

                # System resources
                cpu_percent = metrics.get("cpu_percent", 0)
                memory = metrics.get("memory", {})
                disk = metrics.get("disk", {})

                content.append("[bold]System Resources[/bold]")
                content.append(
                    f"CPU Usage: [{'red' if cpu_percent > 80 else 'yellow' if cpu_percent > 60 else 'green'}]{cpu_percent:.1f}%[/]"
                )
                content.append(
                    f"Memory: [{'red' if memory.get('percent', 0) > 80 else 'yellow' if memory.get('percent', 0) > 60 else 'green'}]{memory.get('percent', 0):.1f}%[/]"
                )
                content.append(
                    f"Disk: [{'red' if disk.get('percent', 0) > 80 else 'yellow' if disk.get('percent', 0) > 60 else 'green'}]{disk.get('percent', 0):.1f}%[/]"
                )
                content.append("")

                # Uptime and load
                uptime = status_data.get("uptime", "Unknown")
                content.append(f"Uptime: {uptime}")

                if "load_average" in metrics:
                    load = metrics["load_average"]
                    content.append(
                        f"Load Avg: {load[0]:.2f}, {load[1]:.2f}, {load[2]:.2f}"
                    )

                return Panel(
                    "\n".join(content),
                    title="[bold cyan]System Status[/bold cyan]",
                    border_style="cyan",
                )

            def render_services_panel(data: Dict[str, Any]) -> Panel:
                """Render services status panel."""
                status_data = data.get("status", {})
                services = status_data.get("services", {})

                if not services:
                    return Panel(
                        "No service data available",
                        title="[bold green]Services[/bold green]",
                        border_style="green",
                    )

                # Create services table
                services_table = Table(
                    show_header=True, header_style="bold green", box=None
                )
                services_table.add_column("Service", style="cyan", width=15)
                services_table.add_column("Status", width=8)
                services_table.add_column("Errors", width=6)
                services_table.add_column("Health", width=8)

                for name, service in services.items():
                    status = service.get("status", "unknown")
                    error_count = service.get("error_count", 0)
                    health = service.get("health", "unknown")

                    status_emoji = {
                        "running": "[green]✓[/green]",
                        "stopped": "[red]✗[/red]",
                        "error": "[red]⚠[/red]",
                        "starting": "[yellow]⟳[/yellow]",
                        "stopping": "[yellow]⟳[/yellow]",
                    }.get(status, "[dim]?[/dim]")

                    health_emoji = {
                        "healthy": "[green]●[/green]",
                        "unhealthy": "[red]●[/red]",
                        "degraded": "[yellow]●[/yellow]",
                    }.get(health, "[dim]●[/dim]")

                    error_style = "red" if error_count > 0 else "dim"

                    services_table.add_row(
                        name,
                        status_emoji,
                        f"[{error_style}]{error_count}[/{error_style}]",
                        health_emoji,
                    )

                return Panel(
                    services_table,
                    title="[bold green]Services[/bold green]",
                    border_style="green",
                )

            def render_trading_panel(data: Dict[str, Any]) -> Panel:
                """Render trading metrics panel."""
                trading_data = data.get("trading", {})
                _ = data.get("portfolio", {})
                risk_data = data.get("risk", {})

                content = []

                # Trading metrics
                content.append("[bold]Trading Metrics[/bold]")

                if trading_data:
                    total_trades = trading_data.get("total_trades", 0)
                    successful_trades = trading_data.get("successful_trades", 0)
                    win_rate = (
                        (successful_trades / total_trades * 100)
                        if total_trades > 0
                        else 0
                    )

                    content.append(f"Total Trades: {total_trades}")
                    content.append(
                        f"Win Rate: [{'green' if win_rate >= 60 else 'yellow' if win_rate >= 40 else 'red'}]{win_rate:.1f}%[/]"
                    )

                    if "daily_pnl" in trading_data:
                        daily_pnl = trading_data["daily_pnl"]
                        pnl_color = "green" if daily_pnl >= 0 else "red"
                        content.append(
                            f"Daily P&L: [{pnl_color}]${daily_pnl:,.2f}[/{pnl_color}]"
                        )
                else:
                    content.append("[dim]No trading data available[/dim]")

                content.append("")

                # Risk metrics
                content.append("[bold]Risk Status[/bold]")
                if risk_data:
                    risk_level = risk_data.get("risk_level", "unknown")
                    risk_color = {
                        "low": "green",
                        "medium": "yellow",
                        "high": "red",
                        "critical": "red bold",
                    }.get(risk_level.lower(), "dim")

                    content.append(
                        f"Risk Level: [{risk_color}]{risk_level.title()}[/{risk_color}]"
                    )

                    if "max_drawdown" in risk_data:
                        drawdown = risk_data["max_drawdown"]
                        dd_color = (
                            "red"
                            if drawdown > 0.1
                            else "yellow" if drawdown > 0.05 else "green"
                        )
                        content.append(
                            f"Max Drawdown: [{dd_color}]{drawdown:.1%}[/{dd_color}]"
                        )
                else:
                    content.append("[dim]Risk data unavailable[/dim]")

                return Panel(
                    "\n".join(content),
                    title="[bold yellow]Trading & Risk[/bold yellow]",
                    border_style="yellow",
                )

            def render_portfolio_panel(data: Dict[str, Any]) -> Panel:
                """Render portfolio information panel."""
                portfolio_data = data.get("portfolio", {})

                content = []

                if portfolio_data:
                    # Portfolio summary
                    total_value = portfolio_data.get("total_value", 0)
                    cash = portfolio_data.get("cash", 0)
                    positions_value = portfolio_data.get("positions_value", 0)

                    content.append("[bold]Portfolio Summary[/bold]")
                    content.append(f"Total Value: [bold]${total_value:,.2f}[/bold]")
                    content.append(f"Cash: ${cash:,.2f}")
                    content.append(f"Positions: ${positions_value:,.2f}")
                    content.append("")

                    # Active positions
                    positions = portfolio_data.get("positions", [])
                    if positions:
                        content.append("[bold]Top Positions[/bold]")
                        for i, pos in enumerate(positions[:5]):  # Show top 5
                            symbol = pos.get("symbol", "Unknown")
                            quantity = pos.get("quantity", 0)
                            value = pos.get("market_value", 0)
                            pnl = pos.get("unrealized_pnl", 0)
                            pnl_color = "green" if pnl >= 0 else "red"

                            content.append(
                                f"{symbol}: {quantity} shares (${value:,.0f}) [{pnl_color}]{pnl:+.0f}[/{pnl_color}]"
                            )
                    else:
                        content.append("[dim]No active positions[/dim]")
                else:
                    content.append("[dim]Portfolio data unavailable[/dim]")

                return Panel(
                    "\n".join(content),
                    title="[bold magenta]Portfolio[/bold magenta]",
                    border_style="magenta",
                )

            def render_footer(data: Dict[str, Any]) -> Panel:
                """Render dashboard footer with controls."""
                controls = [
                    "[bold blue]Controls:[/bold blue]",
                    "[cyan]Ctrl+C[/cyan] Exit",
                    "[cyan]Space[/cyan] Refresh",
                    "[cyan]M[/cyan] Maintenance Mode",
                    "[cyan]E[/cyan] Emergency Stop",
                    "[cyan]R[/cyan] Resume Trading",
                ]

                footer_text = " | ".join(controls)
                return Panel(Align.center(footer_text), border_style="white")

            def render_dashboard(data: Dict[str, Any]) -> Layout:
                """Render complete dashboard."""
                layout = create_dashboard_layout()

                layout["header"].update(render_header(data))
                layout["system"].update(render_system_panel(data))
                layout["services"].update(render_services_panel(data))
                layout["trading"].update(render_trading_panel(data))
                layout["portfolio"].update(render_portfolio_panel(data))
                layout["footer"].update(render_footer(data))

                return layout

            # Initialize dashboard
            try:
                initial_data = await get_dashboard_data()
            except Exception as e:
                console.print(f"[red]Failed to initialize dashboard: {e}[/red]")
                return

            # Run live dashboard
            with Live(
                render_dashboard(initial_data),
                refresh_per_second=1 / refresh,
                screen=True,
            ) as live:
                try:
                    while True:
                        await asyncio.sleep(refresh)
                        try:
                            updated_data = await get_dashboard_data()
                            live.update(render_dashboard(updated_data))
                        except Exception:
                            # Continue with last known data on error
                            pass

                except KeyboardInterrupt:
                    console.print("\n[yellow]Dashboard stopped by user[/yellow]")

        except ImportError:
            console.print(
                "[red]Dashboard requires additional dependencies. Install with:[/red]"
            )
            console.print("pip install rich")
        except Exception as e:
            console.print(f"[red]Dashboard error: {e}[/red]")

    # Run the async dashboard function
    asyncio.run(_run_dashboard())


@app.command()
@run_async
async def queue(
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL")
) -> None:
    """Show task queue status."""
    client = SchedulerClient(url)
    status_data = await client.get_status()
    queues = status_data.get("queues", {})

    if not queues or all(length == 0 for length in queues.values()):
        console.print("[green]All queues empty[/green]")
        return

    table = Table(title="Task Queues")
    table.add_column("Priority", style="cyan")
    table.add_column("Queue Length", style="magenta")
    table.add_column("Status", style="green")

    total_queued = 0
    for priority, length in queues.items():
        total_queued += length
        status = (
            "[green]OK[/green]"
            if length < 10
            else "[yellow]HIGH[/yellow]" if length < 50 else "[red]CRITICAL[/red]"
        )
        table.add_row(priority.title(), str(length), status)

    console.print(table)
    console.print(f"Total Queued Tasks: [bold]{total_queued}[/bold]")


@app.command()
@run_async
async def trade(
    action: str = typer.Argument(..., help="Action: status, history, execute"),
    symbol: str = typer.Argument(None, help="Symbol for trade execution"),
    side: str = typer.Option(None, "--side", help="Trade side: buy, sell"),
    quantity: int = typer.Option(None, "--quantity", help="Trade quantity"),
    url: str = typer.Option(SCHEDULER_URL, "--url", help="Scheduler service URL"),
) -> None:
    """Trading operations."""
    try:
        if action == "status":
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url.replace('8000', '9104')}/status")
                response.raise_for_status()
                trade_data = response.json()

            console.print(
                Panel.fit(
                    f"Trading Enabled: [{'green' if trade_data.get('enabled') else 'red'}]{trade_data.get('enabled')}[/]\n"
                    f"Orders Today: {trade_data.get('orders_today', 0)}\n"
                    f"Active Orders: {trade_data.get('active_orders', 0)}\n"
                    f"Buying Power: [green]${trade_data.get('buying_power', 0):.2f}[/green]",
                    title="Trading Status",
                )
            )

        elif action == "history":
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{url.replace('8000', '9104')}/trades/recent"
                )
                response.raise_for_status()
                trades_data = response.json()

            if not trades_data.get("trades"):
                console.print("[yellow]No recent trades[/yellow]")
                return

            table = Table(title="Recent Trades")
            table.add_column("Time", style="dim")
            table.add_column("Symbol", style="cyan")
            table.add_column("Side", style="magenta")
            table.add_column("Quantity", style="green")
            table.add_column("Price", style="blue")
            table.add_column("P&L", style="red")

            for trade in trades_data.get("trades", [])[:20]:  # Show last 20
                pnl = trade.get("pnl", 0)
                pnl_color = "green" if pnl >= 0 else "red"
                side_color = "green" if trade.get("side") == "buy" else "red"

                table.add_row(
                    trade.get("timestamp", ""),
                    trade.get("symbol", ""),
                    f"[{side_color}]{trade.get('side', '').upper()}[/]",
                    str(trade.get("quantity", 0)),
                    f"${trade.get('price', 0):.2f}",
                    f"[{pnl_color}]${pnl:.2f}[/]",
                )

            console.print(table)

        elif action == "execute" and symbol and side and quantity:
            # Manual trade execution
            console.print(
                f"[yellow]⚠️  Manual trade execution: {side.upper()} {quantity} shares of {symbol}[/yellow]"
            )
            confirm = typer.confirm("Are you sure you want to execute this trade?")

            if confirm:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{url.replace('8000', '9104')}/trades/manual",
                        json={"symbol": symbol, "side": side, "quantity": quantity},
                    )
                    response.raise_for_status()
                    result = response.json()

                if result.get("status") == "success":
                    console.print(
                        f"[green]✓[/green] Trade executed: {result.get('order_id')}"
                    )
                else:
                    console.print(f"[red]✗[/red] Trade failed: {result.get('message')}")

        else:
            console.print("[red]Invalid action or missing parameters[/red]")

    except Exception as e:
        console.print(f"[red]Trading operation error: {e}[/red]")


# Utility functions
def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format."""
    size = float(bytes_value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}PB"


def get_status_emoji(status: str) -> str:
    """Get emoji for status."""
    return {
        "running": "🟢",
        "stopped": "🔴",
        "error": "🔴",
        "starting": "🟡",
        "stopping": "🟡",
        "maintenance": "🔧",
    }.get(status.lower(), "⚪")


@app.command()
def version() -> None:
    """Show version information."""
    console.print("[bold blue]Trading Scheduler CLI[/bold blue]")
    console.print("Version: 1.0.0")
    console.print("Built for: Full AI Trader System")


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug output"),
) -> None:
    """
    Trading Scheduler CLI - Control and monitor your automated trading system.

    This CLI provides comprehensive control over the trading scheduler and all
    its components, including real-time monitoring, task management, and system control.
    """
    if debug:
        console.print("[dim]Debug mode enabled[/dim]")
    elif verbose:
        console.print("[dim]Verbose mode enabled[/dim]")


@app.command()
@run_async
async def maintenance_manage(
    action: str = typer.Argument(
        ..., help="Maintenance action: status, run, run-all, history, report, schedule"
    ),
    task_name: Optional[str] = typer.Option(
        None, "--task", help="Specific task name for 'run' action"
    ),
    report_type: str = typer.Option(
        "daily", "--type", help="Report type: daily, weekly, monthly"
    ),
    format_type: str = typer.Option(
        "json", "--format", help="Export format: json, csv, html"
    ),
    limit: int = typer.Option(50, "--limit", help="Number of history records"),
    url: str = typer.Option(
        "http://localhost:8000", "--url", help="Scheduler service URL"
    ),
) -> None:
    """Maintenance system management."""
    try:
        async with httpx.AsyncClient() as client:
            if action == "status":
                # Show maintenance system status
                response = await client.get(f"{url}/maintenance/status")
                response.raise_for_status()
                status = response.json()

                console.print("[bold blue]🔧 Maintenance System Status[/bold blue]")
                console.print(
                    f"Running: {'🟢 Yes' if status.get('is_running') else '🔴 No'}"
                )
                console.print(f"Current Task: {status.get('current_task', 'None')}")
                console.print(
                    f"Registered Tasks: {len(status.get('registered_tasks', []))}"
                )

                if status.get("last_maintenance_cycle"):
                    last_cycle = status["last_maintenance_cycle"]
                    console.print(
                        f"Last Cycle: {last_cycle.get('timestamp', 'Unknown')}"
                    )
                    console.print(
                        f"Last Result: {'✅ Success' if last_cycle.get('success') else '❌ Failed'}"
                    )

                console.print("\n[bold]Available Tasks:[/bold]")
                for task in status.get("registered_tasks", []):
                    console.print(f"  • {task}")

            elif action == "run":
                if not task_name:
                    console.print(
                        "[red]Task name required for 'run' action. Use --task option.[/red]"
                    )
                    return

                console.print(f"[yellow]Running maintenance task: {task_name}[/yellow]")
                response = await client.post(
                    f"{url}/maintenance/tasks/{task_name}/run", json={}
                )
                response.raise_for_status()
                result = response.json()

                if result.get("success"):
                    console.print(
                        f"[green]✓[/green] Task completed: {result.get('message')}"
                    )
                    if result.get("duration"):
                        console.print(
                            f"Duration: {format_duration(result['duration'])}"
                        )
                    if result.get("bytes_freed"):
                        console.print(
                            f"Space freed: {format_bytes(result['bytes_freed'])}"
                        )
                else:
                    console.print(f"[red]✗[/red] Task failed: {result.get('message')}")

            elif action == "run-all":
                console.print("[yellow]Running all maintenance tasks...[/yellow]")
                response = await client.post(f"{url}/maintenance/run-all", json={})
                response.raise_for_status()
                results = response.json()["results"]

                console.print("\n[bold]Maintenance Results:[/bold]")
                successful = 0
                total = len(results)

                for task_name_result, result in results.items():
                    if result.get("success"):
                        console.print(
                            f"[green]✓[/green] {task_name_result}: {result.get('message')}"
                        )
                        successful += 1
                    else:
                        console.print(
                            f"[red]✗[/red] {task_name_result}: {result.get('message')}"
                        )

                    if result.get("duration"):
                        console.print(
                            f"  Duration: {format_duration(result['duration'])}"
                        )

                console.print(
                    f"\n[bold]Summary: {successful}/{total} tasks successful[/bold]"
                )

            elif action == "history":
                response = await client.get(f"{url}/maintenance/history?limit={limit}")
                response.raise_for_status()
                history = response.json()["history"]

                if not history:
                    console.print("[yellow]No maintenance history found[/yellow]")
                    return

                table = Table()
                table.add_column("Timestamp", style="dim")
                table.add_column("Task", style="bold")
                table.add_column("Status")
                table.add_column("Duration")
                table.add_column("Files Processed")
                table.add_column("Bytes Freed")

                for record in history:
                    task_name_record = record.get("task_name", "Unknown")
                    success = record.get("success", False)
                    status_emoji = "[green]✓[/green]" if success else "[red]✗[/red]"
                    timestamp = record.get("timestamp", "Unknown")

                    # Format timestamp if available
                    try:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        timestamp = dt.strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        pass

                    table.add_row(
                        timestamp,
                        task_name_record,
                        status_emoji,
                        format_duration(record.get("duration", 0)),
                        str(record.get("files_processed", 0)),
                        format_bytes(record.get("bytes_freed", 0)),
                    )

                console.print(table)

            elif action == "report":
                console.print(
                    f"[yellow]Generating {report_type} maintenance report...[/yellow]"
                )
                response = await client.post(
                    f"{url}/maintenance/reports/generate",
                    json={
                        "report_type": report_type,
                        "format_type": format_type,
                        "include_details": True,
                    },
                )
                response.raise_for_status()
                report_data = response.json()["report"]

                if format_type == "json":
                    # Display summary in console
                    summary = report_data.get("summary", {})
                    console.print(
                        f"\n[bold]{report_type.title()} Maintenance Report[/bold]"
                    )
                    console.print(f"Date: {report_data.get('date', 'Unknown')}")
                    console.print(f"Tasks Run: {summary.get('total_tasks_run', 0)}")
                    console.print(
                        f"Success Rate: {summary.get('success_rate', 0):.1f}%"
                    )
                    console.print(
                        f"Total Duration: {format_duration(summary.get('total_duration', 0))}"
                    )
                    console.print(
                        f"Space Freed: {format_bytes(summary.get('total_bytes_freed', 0))}"
                    )

                    if summary.get("recommendations"):
                        console.print("\n[bold]Recommendations:[/bold]")
                        for rec in summary["recommendations"]:
                            console.print(f"  • {rec}")

                else:
                    # File export - show confirmation
                    export_path = report_data.get("export_path", "Unknown")
                    console.print(f"[green]✓[/green] Report exported to: {export_path}")

            elif action == "schedule":
                response = await client.get(f"{url}/maintenance/schedule")
                response.raise_for_status()
                schedule_data = response.json()

                console.print("[bold blue]📅 Maintenance Schedule[/bold blue]")

                scheduled_tasks = schedule_data.get("scheduled_tasks", {})
                if scheduled_tasks:
                    table = Table()
                    table.add_column("Schedule ID", style="bold")
                    table.add_column("Task Name")
                    table.add_column("Schedule")
                    table.add_column("Next Run")
                    table.add_column("Enabled")

                    for schedule_id, task_info in scheduled_tasks.items():
                        enabled = "✅" if task_info.get("enabled", True) else "❌"

                        table.add_row(
                            schedule_id,
                            task_info.get("task_name", "Unknown"),
                            task_info.get("schedule", "Unknown"),
                            task_info.get("next_run", "Unknown"),
                            enabled,
                        )

                    console.print(table)

                    # Show next few scheduled tasks
                    next_tasks = schedule_data.get("next_24_hours", [])
                    if next_tasks:
                        console.print(
                            f"\n[bold]Next 24 Hours ({len(next_tasks)} tasks):[/bold]"
                        )
                        for task in next_tasks:
                            console.print(
                                f"  • {task['description']} - {task.get('estimated_next_run', 'TBD')}"
                            )
                else:
                    console.print(
                        "[yellow]No scheduled maintenance tasks found[/yellow]"
                    )

            else:
                console.print(f"[red]Unknown action: {action}[/red]")
                console.print(
                    "Available actions: status, run, run-all, history, report, schedule"
                )

    except httpx.HTTPError as e:
        console.print(f"[red]HTTP error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Maintenance error: {e}[/red]")


@app.command()
@run_async
async def maintenance_dashboard(
    refresh: int = typer.Option(5, "--refresh", help="Refresh interval in seconds"),
    url: str = typer.Option(
        "http://localhost:8000", "--url", help="Scheduler service URL"
    ),
) -> None:
    """Maintenance dashboard with real-time updates."""
    try:
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel

        console.print("[bold blue]🎛️  Starting Maintenance Dashboard[/bold blue]")
        console.print("Press Ctrl+C to exit")

        def create_dashboard() -> Layout:
            layout = Layout()

            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=3),
            )

            layout["main"].split_row(Layout(name="left"), Layout(name="right"))

            layout["left"].split_column(
                Layout(name="status", size=10), Layout(name="metrics")
            )

            layout["right"].split_column(Layout(name="tasks"), Layout(name="alerts"))

            return layout

        async def update_dashboard_data() -> dict:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get maintenance dashboard data
                dashboard_response = await client.get(f"{url}/maintenance/dashboard")
                dashboard_response.raise_for_status()
                return dashboard_response.json()["dashboard"]

        def render_dashboard(data: Dict[str, Any]) -> Layout:
            layout = create_dashboard()

            # Header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            layout["header"].update(
                Panel(
                    f"[bold blue]🔧 Maintenance Dashboard[/bold blue] - {timestamp}",
                    style="blue",
                )
            )

            # System Health
            health = data.get("system_health", {})
            health_text = ""
            if health and "error" not in health:
                cpu = health.get("cpu_percent", 0)
                memory = health.get("memory_percent", 0)
                disk = health.get("disk_percent", 0)

                cpu_color = "red" if cpu > 80 else "yellow" if cpu > 60 else "green"
                memory_color = (
                    "red" if memory > 85 else "yellow" if memory > 70 else "green"
                )
                disk_color = "red" if disk > 90 else "yellow" if disk > 80 else "green"

                health_text = f"""[{cpu_color}]CPU: {cpu:.1f}%[/]
[{memory_color}]Memory: {memory:.1f}%[/]
[{disk_color}]Disk: {disk:.1f}%[/]"""

                if "load_average" in health:
                    load_avg = health["load_average"]
                    health_text += f"\nLoad: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}"
            else:
                health_text = "[red]Unable to collect system health[/]"

            layout["status"].update(Panel(health_text, title="System Health"))

            # Recent Tasks
            recent_tasks = data.get("recent_tasks", [])
            task_text = ""
            for task in recent_tasks[:5]:
                status_emoji = "✅" if task.get("success") else "❌"
                timestamp = task.get("timestamp", "")[:16]  # YYYY-MM-DD HH:MM
                task_text += f"{status_emoji} {timestamp} {task.get('task_name', '')}\n"

            if not task_text:
                task_text = "[dim]No recent tasks[/]"

            layout["tasks"].update(Panel(task_text, title="Recent Tasks"))

            # Alerts
            alerts = data.get("alerts", [])
            alert_text = ""
            for alert in alerts[:5]:
                alert_type = alert.get("type", "unknown")
                timestamp = alert.get("timestamp", "")[:16]

                emoji = {
                    "maintenance_failure": "🔴",
                    "maintenance_slow": "🟡",
                    "maintenance_low_impact": "🟠",
                }.get(alert_type, "⚪")

                alert_text += (
                    f"{emoji} {timestamp} {alert.get('message', alert_type)}\n"
                )

            if not alert_text:
                alert_text = "[green]No active alerts[/]"

            layout["alerts"].update(Panel(alert_text, title="Alerts"))

            # Performance Metrics
            metrics = data.get("performance_metrics", {})
            metric_text = ""
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        if "bytes" in key.lower():
                            metric_text += f"{key}: {format_bytes(int(value))}\n"
                        elif "percent" in key.lower():
                            metric_text += f"{key}: {value:.1f}%\n"
                        else:
                            metric_text += f"{key}: {value}\n"
                    else:
                        metric_text += f"{key}: {value}\n"

            if not metric_text:
                metric_text = "[dim]No performance data[/]"

            layout["metrics"].update(Panel(metric_text, title="Performance"))

            # Footer
            layout["footer"].update(
                Panel(
                    f"[dim]Refresh rate: {refresh}s | Press Ctrl+C to exit[/dim]",
                    style="dim",
                )
            )

            return layout

        # Initial data load
        try:
            initial_data = await update_dashboard_data()
        except Exception as e:
            console.print(f"[red]Failed to connect to scheduler service: {e}[/red]")
            return

        # Live dashboard
        with Live(
            render_dashboard(initial_data), refresh_per_second=1 / refresh, screen=True
        ) as live:
            try:
                while True:
                    await asyncio.sleep(refresh)
                    try:
                        updated_data = await update_dashboard_data()
                        live.update(render_dashboard(updated_data))
                    except Exception:
                        # Continue running even if updates fail
                        pass
            except KeyboardInterrupt:
                console.print("\n[yellow]Dashboard stopped[/yellow]")

    except Exception as e:
        console.print(f"[red]Dashboard error: {e}[/red]")


if __name__ == "__main__":
    app()
