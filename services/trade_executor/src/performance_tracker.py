"""
Performance Tracker for Trade Execution Service.

This module tracks and calculates comprehensive trading performance metrics
including win/loss ratios, profit factors, expectancy, Sharpe ratios, and
provides TradeNote-compatible export functionality.
"""

import asyncio
import logging
from datetime import datetime, date, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID
import math

import asyncpg
import redis.asyncio as redis
import pandas as pd

from shared.config import get_config


logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Comprehensive performance tracking and analysis system.

    Calculates and stores:
    - Win/loss ratios and streaks
    - Profit factors and expectancy
    - Risk-adjusted returns (Sharpe, Sortino)
    - Drawdown analysis
    - Strategy-specific performance
    - TradeNote export compatibility
    """

    def __init__(self):
        """Initialize performance tracker."""
        self.config = get_config()
        self._db_pool = None
        self._redis = None
        self._performance_cache: Dict[str, Any] = {}
        self._running = False

    async def initialize(self):
        """Initialize database and Redis connections."""
        try:
            self._db_pool = await asyncpg.create_pool(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                user=self.config.database.username,
                password=self.config.database.password,
                min_size=2,
                max_size=5,
                command_timeout=30
            )

            self._redis = redis.from_url(
                self.config.redis.url,
                max_connections=5,
                retry_on_timeout=True
            )

            logger.info("PerformanceTracker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PerformanceTracker: {e}")
            raise

    async def cleanup(self):
        """Clean up resources."""
        try:
            self._running = False
            if self._db_pool:
                await self._db_pool.close()
            if self._redis:
                await self._redis.close()
            logger.info("PerformanceTracker cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def calculate_daily_performance(self, target_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive daily performance metrics.

        Args:
            target_date: Date to calculate metrics for (default: today)

        Returns:
            Dictionary with daily performance metrics
        """
        try:
            from datetime import date
            if target_date is None:
                target_date = date.today()

            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                # Get daily trade statistics
                daily_stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                        SUM(pnl) as gross_pnl,
                        SUM(pnl - commission) as net_pnl,
                        SUM(commission) as total_commission,
                        MAX(pnl) as largest_win,
                        MIN(pnl) as largest_loss,
                        AVG(pnl) as avg_pnl,
                        STDDEV(pnl) as pnl_stddev
                    FROM trading.positions
                    WHERE DATE(exit_time) = $1 AND status = 'closed'
                """, target_date)

                if not daily_stats or daily_stats['total_trades'] == 0:
                    return self._empty_daily_metrics(target_date)

                # Calculate derived metrics
                win_rate = Decimal(daily_stats['winning_trades']) / Decimal(daily_stats['total_trades'])

                # Calculate profit factor
                gross_profit = await conn.fetchval("""
                    SELECT SUM(pnl) FROM trading.positions
                    WHERE DATE(exit_time) = $1 AND status = 'closed' AND pnl > 0
                """, target_date) or Decimal("0")

                gross_loss = await conn.fetchval("""
                    SELECT SUM(ABS(pnl)) FROM trading.positions
                    WHERE DATE(exit_time) = $1 AND status = 'closed' AND pnl < 0
                """, target_date) or Decimal("0")

                profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

                # Calculate expectancy
                avg_win = gross_profit / Decimal(daily_stats['winning_trades']) if daily_stats['winning_trades'] > 0 else Decimal("0")
                avg_loss = gross_loss / Decimal(daily_stats['losing_trades']) if daily_stats['losing_trades'] > 0 else Decimal("0")
                expectancy = (win_rate * avg_win) - ((Decimal("1") - win_rate) * avg_loss)

                # Get portfolio value for the day
                portfolio_value = await self._get_portfolio_value_for_date(target_date)

                # Calculate Sharpe ratio (simplified daily)
                sharpe_ratio = None
                if daily_stats['pnl_stddev'] and daily_stats['pnl_stddev'] > 0:
                    risk_free_rate = Decimal("0.02") / Decimal("365")  # Assume 2% annual risk-free rate
                    excess_return = Decimal(str(daily_stats['avg_pnl'])) - risk_free_rate
                    sharpe_ratio = excess_return / Decimal(str(daily_stats['pnl_stddev']))

                metrics = {
                    'date': target_date,
                    'total_trades': daily_stats['total_trades'],
                    'winning_trades': daily_stats['winning_trades'],
                    'losing_trades': daily_stats['losing_trades'],
                    'gross_pnl': float(daily_stats['gross_pnl']),
                    'net_pnl': float(daily_stats['net_pnl']),
                    'commission_total': float(daily_stats['total_commission']),
                    'largest_win': float(daily_stats['largest_win']),
                    'largest_loss': float(daily_stats['largest_loss']),
                    'win_rate': float(win_rate),
                    'profit_factor': float(profit_factor) if profit_factor else None,
                    'expectancy': float(expectancy),
                    'sharpe_ratio': float(sharpe_ratio) if sharpe_ratio else None,
                    'avg_win': float(avg_win),
                    'avg_loss': float(avg_loss),
                    'portfolio_value': float(portfolio_value),
                    'return_pct': float(Decimal(str(daily_stats['net_pnl'])) / Decimal(str(portfolio_value)) * Decimal("100")) if portfolio_value > 0 else 0
                }

                # Store in daily performance table
                await self._store_daily_performance(target_date, metrics)

                return metrics

        except Exception as e:
            from datetime import date
            logger.error(f"Failed to calculate daily performance for {target_date}: {e}")
            return self._empty_daily_metrics(target_date if target_date else date.today())

    def _empty_daily_metrics(self, target_date: date) -> Dict[str, Any]:
        """Return empty daily metrics structure."""
        return {
            'date': target_date,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'gross_pnl': 0.0,
            'net_pnl': 0.0,
            'commission_total': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': None,
            'expectancy': 0.0,
            'sharpe_ratio': None,
            'portfolio_value': 0.0,
            'return_pct': 0.0
        }

    async def _store_daily_performance(self, target_date: date, metrics: Dict[str, Any]):
        """Store daily performance metrics in database."""
        try:
            if not self._db_pool:
                return 0.0
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trading.daily_performance (
                        date, total_trades, winning_trades, losing_trades,
                        gross_pnl, net_pnl, commission_total, largest_win, largest_loss,
                        win_rate, profit_factor, expectancy, sharpe_ratio,
                        portfolio_value
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (date) DO UPDATE SET
                        total_trades = EXCLUDED.total_trades,
                        winning_trades = EXCLUDED.winning_trades,
                        losing_trades = EXCLUDED.losing_trades,
                        gross_pnl = EXCLUDED.gross_pnl,
                        net_pnl = EXCLUDED.net_pnl,
                        commission_total = EXCLUDED.commission_total,
                        largest_win = EXCLUDED.largest_win,
                        largest_loss = EXCLUDED.largest_loss,
                        win_rate = EXCLUDED.win_rate,
                        profit_factor = EXCLUDED.profit_factor,
                        expectancy = EXCLUDED.expectancy,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        portfolio_value = EXCLUDED.portfolio_value,
                        updated_at = NOW()
                """,
                    target_date, metrics['total_trades'], metrics['winning_trades'], metrics['losing_trades'],
                    metrics['gross_pnl'], metrics['net_pnl'], metrics['commission_total'],
                    metrics['largest_win'], metrics['largest_loss'], metrics['win_rate'],
                    metrics['profit_factor'], metrics['expectancy'], metrics['sharpe_ratio'],
                    metrics['portfolio_value']
                )

        except Exception as e:
            logger.error(f"Failed to store daily performance: {e}")

    async def _get_portfolio_value_for_date(self, target_date: date) -> Decimal:
        """Get portfolio value for specific date."""
        try:
            if not self._db_pool:
                return Decimal("0")
            async with self._db_pool.acquire() as conn:
                # Try to get from account snapshots first
                portfolio_value = await conn.fetchval("""
                    SELECT total_equity FROM trading.account_snapshots
                    WHERE DATE(timestamp) = $1
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, target_date)

                if portfolio_value:
                    return Decimal(str(portfolio_value))

                # Fallback: calculate from positions
                total_value = await conn.fetchval("""
                    SELECT SUM(ABS(quantity * current_price))
                    FROM trading.positions
                    WHERE DATE(entry_time) <= $1
                    AND (DATE(exit_time) > $1 OR exit_time IS NULL)
                """, target_date)

                return Decimal(str(total_value)) if total_value else Decimal("100000")  # Default value

        except Exception as e:
            logger.error(f"Failed to get portfolio value for {target_date}: {e}")
            return Decimal("100000")

    async def calculate_strategy_performance(self, strategy_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Calculate performance metrics for a specific strategy.

        Args:
            strategy_name: Name of the strategy
            days: Number of days to analyze

        Returns:
            Strategy performance metrics
        """
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                strategy_stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade,
                        STDDEV(pnl) as pnl_stddev,
                        SUM(commission) as total_commission,
                        AVG(EXTRACT(EPOCH FROM (exit_time - entry_time))/3600) as avg_holding_hours
                    FROM trading.positions
                    WHERE strategy_type = $1
                    AND status = 'closed'
                    AND entry_time >= NOW() - INTERVAL '1 day' * $2
                """, strategy_name, days)

                if not strategy_stats or strategy_stats['total_trades'] == 0:
                    return {
                        'strategy_name': strategy_name,
                        'period_days': days,
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'total_pnl': 0.0,
                        'profit_factor': None,
                        'expectancy': 0.0
                    }

                # Calculate detailed metrics
                win_rate = strategy_stats['winning_trades'] / strategy_stats['total_trades']

                # Get profit/loss breakdown
                gross_profit = await conn.fetchval("""
                    SELECT SUM(pnl) FROM trading.positions
                    WHERE strategy_type = $1 AND pnl > 0 AND status = 'closed'
                    AND entry_time >= NOW() - INTERVAL '1 day' * $2
                """, strategy_name, days) or Decimal("0")

                gross_loss = await conn.fetchval("""
                    SELECT SUM(ABS(pnl)) FROM trading.positions
                    WHERE strategy_type = $1 AND pnl < 0 AND status = 'closed'
                    AND entry_time >= NOW() - INTERVAL '1 day' * $2
                """, strategy_name, days) or Decimal("0")

                profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

                # Calculate expectancy
                avg_win = gross_profit / strategy_stats['winning_trades'] if strategy_stats['winning_trades'] > 0 else Decimal("0")
                avg_loss = gross_loss / strategy_stats['losing_trades'] if strategy_stats['losing_trades'] > 0 else Decimal("0")
                expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

                # Calculate Sharpe ratio
                sharpe_ratio = None
                if strategy_stats['pnl_stddev'] and strategy_stats['pnl_stddev'] > 0:
                    risk_free_rate = Decimal("0.02") / 365  # Daily risk-free rate
                    excess_return = strategy_stats['avg_pnl'] - risk_free_rate
                    sharpe_ratio = excess_return / strategy_stats['pnl_stddev'] * math.sqrt(252)  # Annualized

                # Get consecutive wins/losses
                win_loss_streak = await self._calculate_win_loss_streaks(strategy_name, days)

                return {
                    'strategy_name': strategy_name,
                    'period_days': days,
                    'total_trades': strategy_stats['total_trades'],
                    'winning_trades': strategy_stats['winning_trades'],
                    'losing_trades': strategy_stats['losing_trades'],
                    'win_rate': float(win_rate),
                    'total_pnl': float(strategy_stats['total_pnl']),
                    'avg_pnl': float(strategy_stats['avg_pnl']),
                    'best_trade': float(strategy_stats['best_trade']),
                    'worst_trade': float(strategy_stats['worst_trade']),
                    'profit_factor': float(profit_factor) if profit_factor else None,
                    'expectancy': float(expectancy),
                    'sharpe_ratio': float(sharpe_ratio) if sharpe_ratio else None,
                    'avg_win': float(avg_win),
                    'avg_loss': float(avg_loss),
                    'total_commission': float(strategy_stats['total_commission']),
                    'avg_holding_hours': float(strategy_stats['avg_holding_hours'] or 0),
                    'max_consecutive_wins': win_loss_streak['max_consecutive_wins'],
                    'max_consecutive_losses': win_loss_streak['max_consecutive_losses'],
                    'current_streak': win_loss_streak['current_streak'],
                    'current_streak_type': win_loss_streak['current_streak_type']
                }

        except Exception as e:
            logger.error(f"Failed to calculate strategy performance for {strategy_name}: {e}")
            return {}

    async def _calculate_win_loss_streaks(self, strategy_name: str, days: int) -> Dict[str, Any]:
        """Calculate win/loss streaks for a strategy."""
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                trades = await conn.fetch("""
                    SELECT pnl, exit_time
                    FROM trading.positions
                    WHERE strategy_type = $1 AND status = 'closed'
                    AND entry_time >= NOW() - INTERVAL '1 day' * $2
                    ORDER BY exit_time ASC
                """, strategy_name, days)

                if not trades:
                    return {
                        'max_consecutive_wins': 0,
                        'max_consecutive_losses': 0,
                        'current_streak': 0,
                        'current_streak_type': 'none'
                    }

                # Analyze streaks
                max_wins = 0
                max_losses = 0
                current_wins = 0
                current_losses = 0

                for trade in trades:
                    if trade['pnl'] > 0:
                        current_wins += 1
                        current_losses = 0
                        max_wins = max(max_wins, current_wins)
                    else:
                        current_losses += 1
                        current_wins = 0
                        max_losses = max(max_losses, current_losses)

                # Determine current streak
                if current_wins > 0:
                    current_streak = current_wins
                    current_streak_type = 'wins'
                elif current_losses > 0:
                    current_streak = current_losses
                    current_streak_type = 'losses'
                else:
                    current_streak = 0
                    current_streak_type = 'none'

                return {
                    'max_consecutive_wins': max_wins,
                    'max_consecutive_losses': max_losses,
                    'current_streak': current_streak,
                    'current_streak_type': current_streak_type
                }

        except Exception as e:
            logger.error(f"Failed to calculate streaks for {strategy_name}: {e}")
            return {
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'current_streak': 0,
                'current_streak_type': 'none'
            }

    async def calculate_risk_metrics(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics.

        Args:
            days: Number of days to analyze

        Returns:
            Risk metrics
        """
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                # Get daily returns
                returns_data = await conn.fetch("""
                    SELECT
                        date,
                        net_pnl,
                        portfolio_value,
                        net_pnl / NULLIF(portfolio_value, 0) as daily_return
                    FROM trading.daily_performance
                    WHERE date >= CURRENT_DATE - INTERVAL '1 day' * $1
                    ORDER BY date
                """, days)

                if len(returns_data) < 2:
                    return {'error': 'Insufficient data for risk calculation'}

                returns = [float(r['daily_return']) for r in returns_data if r['daily_return'] is not None]

                if not returns:
                    return {'error': 'No valid returns data'}

                # Calculate risk metrics
                avg_return = sum(returns) / len(returns)
                return_std = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns)) if len(returns) > 1 else 0

                # Sharpe ratio (annualized)
                risk_free_rate = 0.02 / 252  # Daily risk-free rate
                excess_return = avg_return - risk_free_rate
                sharpe_ratio = (excess_return / return_std * math.sqrt(252)) if return_std > 0 else 0

                # Sortino ratio (downside deviation)
                downside_returns = [r for r in returns if r < 0]
                downside_std = math.sqrt(sum(r ** 2 for r in downside_returns) / len(downside_returns)) if downside_returns else 0
                sortino_ratio = (excess_return / downside_std * math.sqrt(252)) if downside_std > 0 else 0

                # Maximum drawdown
                cumulative_returns = []
                cumulative = 1.0
                for ret in returns:
                    cumulative *= (1 + ret)
                    cumulative_returns.append(cumulative)

                peak = cumulative_returns[0]
                max_drawdown = 0
                for value in cumulative_returns:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)

                # Value at Risk (95% confidence)
                sorted_returns = sorted(returns)
                var_95_index = int(len(sorted_returns) * 0.05)
                var_95 = sorted_returns[var_95_index] if var_95_index < len(sorted_returns) else sorted_returns[0]

                return {
                    'period_days': days,
                    'avg_daily_return': avg_return,
                    'daily_volatility': return_std,
                    'annual_volatility': return_std * math.sqrt(252),
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': max_drawdown,
                    'value_at_risk_95': var_95,
                    'total_return': cumulative_returns[-1] - 1 if cumulative_returns else 0,
                    'calmar_ratio': (cumulative_returns[-1] - 1) / max_drawdown if max_drawdown > 0 else 0
                }

        except Exception as e:
            logger.error(f"Failed to calculate risk metrics: {e}")
            return {}

    async def calculate_monthly_performance(self, year: int, month: int) -> Dict[str, Any]:
        """
        Calculate monthly performance summary.

        Args:
            year: Year
            month: Month (1-12)

        Returns:
            Monthly performance metrics
        """
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                monthly_stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade,
                        SUM(commission) as total_commission
                    FROM trading.positions
                    WHERE EXTRACT(YEAR FROM exit_time) = $1
                    AND EXTRACT(MONTH FROM exit_time) = $2
                    AND status = 'closed'
                """, year, month)

                if not monthly_stats or monthly_stats['total_trades'] == 0:
                    return {
                        'year': year,
                        'month': month,
                        'total_trades': 0,
                        'total_pnl': 0.0,
                        'win_rate': 0.0
                    }

                win_rate = monthly_stats['winning_trades'] / monthly_stats['total_trades']

                # Get daily performance for the month
                daily_data = await conn.fetch("""
                    SELECT net_pnl, portfolio_value
                    FROM trading.daily_performance
                    WHERE EXTRACT(YEAR FROM date) = $1
                    AND EXTRACT(MONTH FROM date) = $2
                    ORDER BY date
                """, year, month)

                # Calculate monthly return
                if daily_data:
                    monthly_return = sum(float(d['net_pnl']) for d in daily_data)
                    avg_portfolio_value = sum(float(d['portfolio_value']) for d in daily_data) / len(daily_data)
                    monthly_return_pct = (monthly_return / avg_portfolio_value * 100) if avg_portfolio_value > 0 else 0
                else:
                    monthly_return_pct = 0

                return {
                    'year': year,
                    'month': month,
                    'total_trades': monthly_stats['total_trades'],
                    'winning_trades': monthly_stats['winning_trades'],
                    'total_pnl': float(monthly_stats['total_pnl']),
                    'avg_pnl': float(monthly_stats['avg_pnl']),
                    'best_trade': float(monthly_stats['best_trade']),
                    'worst_trade': float(monthly_stats['worst_trade']),
                    'win_rate': float(win_rate),
                    'total_commission': float(monthly_stats['total_commission']),
                    'monthly_return_pct': monthly_return_pct,
                    'trading_days': len(daily_data)
                }

        except Exception as e:
            logger.error(f"Failed to calculate monthly performance for {year}-{month}: {e}")
            return {}

    async def export_for_tradenote(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        strategy_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Export trade data in TradeNote compatible format.

        Args:
            start_date: Start date for export
            end_date: End date for export
            strategy_name: Optional strategy filter

        Returns:
            List of trades in TradeNote format
        """
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                conditions = ["status = 'closed'", "exit_time IS NOT NULL"]
                params = []

                if start_date:
                    params.append(start_date)
                    conditions.append(f"DATE(entry_time) >= ${len(params)}")

                if end_date:
                    params.append(end_date)
                    conditions.append(f"DATE(exit_time) <= ${len(params)}")

                if strategy_name:
                    params.append(strategy_name)
                    conditions.append(f"strategy_type = ${len(params)}")

                where_clause = " WHERE " + " AND ".join(conditions)

                query = f"""
                    SELECT
                        p.*,
                        tp.gross_pnl,
                        tp.net_pnl,
                        tp.commission_total,
                        tp.return_percentage,
                        tp.holding_period_hours,
                        tp.max_favorable_excursion,
                        tp.max_adverse_excursion
                    FROM trading.positions p
                    LEFT JOIN trading.trade_performance tp ON p.id = tp.position_id
                    {where_clause}
                    ORDER BY p.exit_time DESC
                """

                rows = await conn.fetch(query, *params)

                tradenote_trades = []
                for row in rows:
                    # TradeNote format requirements
                    trade_data = {
                        # Required fields
                        'symbol': row['ticker'],
                        'side': 'Long' if row['side'] == 'long' else 'Short',
                        'entry_date': row['entry_time'].strftime('%Y-%m-%d'),
                        'entry_time': row['entry_time'].strftime('%H:%M:%S'),
                        'quantity': abs(row['quantity']),
                        'entry_price': float(row['entry_price']),

                        # Exit fields (if available)
                        'exit_date': row['exit_time'].strftime('%Y-%m-%d') if row['exit_time'] else None,
                        'exit_time': row['exit_time'].strftime('%H:%M:%S') if row['exit_time'] else None,
                        'exit_price': float(row['exit_price']) if row['exit_price'] else None,

                        # P&L fields
                        'gross_pnl': float(row['gross_pnl']) if row['gross_pnl'] else float(row['pnl']),
                        'net_pnl': float(row['net_pnl']) if row['net_pnl'] else float(row['pnl']),
                        'commission': float(row['commission_total']) if row['commission_total'] else float(row['commission']),
                        'fees': 0.0,  # Alpaca doesn't charge fees

                        # Strategy and tags
                        'strategy': row['strategy_type'],
                        'setup': row['strategy_type'],
                        'tags': f"automated,{row['strategy_type']}",

                        # Additional metrics
                        'return_pct': float(row['return_percentage']) if row['return_percentage'] else None,
                        'duration_hours': float(row['holding_period_hours']) if row['holding_period_hours'] else None,
                        'mfe': float(row['max_favorable_excursion']) if row['max_favorable_excursion'] else None,
                        'mae': float(row['max_adverse_excursion']) if row['max_adverse_excursion'] else None,

                        # Stop loss and take profit
                        'stop_loss': float(row['stop_loss']) if row['stop_loss'] else None,
                        'take_profit': float(row['take_profit']) if row['take_profit'] else None,

                        # Notes
                        'notes': f"Entry: {row['entry_time']}, Strategy: {row['strategy_type']}, Signal ID: {row['signal_id']}"
                    }

                    tradenote_trades.append(trade_data)

                logger.info(f"Exported {len(tradenote_trades)} trades for TradeNote")
                return tradenote_trades

        except Exception as e:
            logger.error(f"Failed to export TradeNote data: {e}")
            return []

    async def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            days: Number of days to analyze

        Returns:
            Comprehensive performance report
        """
        try:
            # Get overall performance
            overall_perf = await self.calculate_daily_performance()

            # Get risk metrics
            risk_metrics = await self.calculate_risk_metrics(days)

            # Get strategy breakdown
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                strategies = await conn.fetch("""
                    SELECT DISTINCT strategy_type
                    FROM trading.positions
                    WHERE entry_time >= NOW() - INTERVAL '1 day' * $1
                    AND status = 'closed'
                """, days)

            strategy_performance = {}
            for strategy in strategies:
                strategy_name = strategy['strategy_type']
                strategy_performance[strategy_name] = await self.calculate_strategy_performance(strategy_name, days)

            # Get monthly performance
            current_date = date.today()
            monthly_perf = await self.calculate_monthly_performance(current_date.year, current_date.month)

            # Get symbol performance breakdown
            symbol_performance = await self._get_symbol_performance(days)

            # Get recent trades
            recent_trades = await self._get_recent_trades(days, limit=10)

            # Generate report
            report = {
                'report_date': current_date.isoformat(),
                'period_days': days,
                'overall_performance': overall_perf,
                'risk_metrics': risk_metrics,
                'monthly_performance': monthly_perf,
                'strategy_performance': strategy_performance,
                'symbol_performance': symbol_performance,
                'recent_trades': recent_trades,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {}

    async def _get_symbol_performance(self, days: int) -> List[Dict[str, Any]]:
        """Get performance breakdown by symbol."""
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                symbol_stats = await conn.fetch("""
                    SELECT
                        ticker as symbol,
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade
                    FROM trading.positions
                    WHERE status = 'closed'
                    AND entry_time >= NOW() - INTERVAL '1 day' * $1
                    GROUP BY ticker
                    ORDER BY total_pnl DESC
                    LIMIT 20
                """, days)

                symbol_performance = []
                for row in symbol_stats:
                    win_rate = row['winning_trades'] / row['total_trades'] if row['total_trades'] > 0 else 0
                    symbol_performance.append({
                        'symbol': row['symbol'],
                        'total_trades': row['total_trades'],
                        'winning_trades': row['winning_trades'],
                        'win_rate': float(win_rate),
                        'total_pnl': float(row['total_pnl']),
                        'avg_pnl': float(row['avg_pnl']),
                        'best_trade': float(row['best_trade']),
                        'worst_trade': float(row['worst_trade'])
                    })

                return symbol_performance

        except Exception as e:
            logger.error(f"Failed to get symbol performance: {e}")
            return []

    async def _get_recent_trades(self, days: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent closed trades."""
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT
                        ticker as symbol,
                        side,
                        quantity,
                        entry_price,
                        exit_price,
                        pnl,
                        entry_time,
                        exit_time,
                        strategy_type
                    FROM trading.positions
                    WHERE status = 'closed'
                    AND exit_time >= NOW() - INTERVAL '1 day' * $1
                    ORDER BY exit_time DESC
                    LIMIT $2
                """, days, limit)

                recent_trades = []
                for row in rows:
                    trade_data = {
                        'symbol': row['symbol'],
                        'side': row['side'],
                        'quantity': row['quantity'],
                        'entry_price': float(row['entry_price']),
                        'exit_price': float(row['exit_price']) if row['exit_price'] else None,
                        'pnl': float(row['pnl']) if row['pnl'] else None,
                        'entry_time': row['entry_time'].isoformat(),
                        'exit_time': row['exit_time'].isoformat() if row['exit_time'] else None,
                        'strategy': row['strategy_type'],
                        'duration_hours': (row['exit_time'] - row['entry_time']).total_seconds() / 3600 if row['exit_time'] else None
                    }
                    recent_trades.append(trade_data)

                return recent_trades

        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    async def update_daily_metrics(self, target_date: Optional[date] = None):
        """
        Update daily performance metrics.

        Args:
            target_date: Date to update metrics for (default: today)
        """
        try:
            from datetime import date
            if target_date is None:
                target_date = date.today()
            metrics = await self.calculate_daily_performance(target_date)

            # Publish metrics update
            if self._redis:
                import json as json_module
                await self._redis.publish("performance_report", json_module.dumps({
                'date': target_date.isoformat(),
                'metrics': metrics,
                'timestamp': datetime.now(timezone.utc).isoformat()
                }, default=str))

            logger.info(f"Daily metrics updated for {target_date}")

        except Exception as e:
            logger.error(f"Failed to update daily metrics for {target_date}: {e}")

    async def calculate_trade_quality_score(self, position_id: UUID) -> float:
        """
        Calculate execution quality score for a trade.

        Args:
            position_id: Position ID

        Returns:
            Quality score (0-100)
        """
        try:
            if not self._db_pool:
                return 0.0
            async with self._db_pool.acquire() as conn:
                # Get position and performance data
                trade_data = await conn.fetchrow("""
                    SELECT
                        p.*,
                        tp.slippage,
                        tp.execution_quality_score,
                        tp.max_favorable_excursion,
                        tp.max_adverse_excursion
                    FROM trading.positions p
                    LEFT JOIN trading.trade_performance tp ON p.id = tp.position_id
                    WHERE p.id = $1
                """, position_id)

                if not trade_data:
                    return 0.0

                score = 100.0  # Start with perfect score

                # Penalize for slippage
                if trade_data['slippage']:
                    slippage_penalty = min(float(trade_data['slippage']) * 1000, 20)  # Max 20 point penalty
                    score -= slippage_penalty

                # Reward for achieving targets
                if trade_data['pnl'] and trade_data['pnl'] > 0:
                    # Check if near take profit target
                    if trade_data['take_profit'] and trade_data['exit_price']:
                        target_achievement = abs(trade_data['exit_price'] - trade_data['take_profit']) / trade_data['take_profit']
                        if target_achievement < 0.02:  # Within 2% of target
                            score += 10

                # Penalize for poor timing (high MAE relative to final profit)
                if trade_data['max_adverse_excursion'] and trade_data['pnl']:
                    mae_ratio = abs(float(trade_data['max_adverse_excursion']) / float(trade_data['pnl']))
                    if mae_ratio > 2:  # MAE more than 2x final profit
                        score -= min(mae_ratio * 5, 25)

                # Penalize for leaving money on table (high MFE vs final profit)
                if trade_data['max_favorable_excursion'] and trade_data['pnl']:
                    mfe_ratio = float(trade_data['max_favorable_excursion']) / float(trade_data['pnl'])
                    if mfe_ratio > 2:  # Left more than 2x final profit on table
                        score -= min((mfe_ratio - 2) * 10, 20)

                # Normalize score
                score = max(0.0, min(100.0, score))

                # Update in database
                await conn.execute("""
                    UPDATE trading.trade_performance
                    SET execution_quality_score = $2
                    WHERE position_id = $1
                """, position_id, score)

                return score

        except Exception as e:
            logger.error(f"Failed to calculate trade quality score for {position_id}: {e}")
            return 0.0

    async def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get high-level performance summary.

        Args:
            days: Number of days to analyze

        Returns:
            Performance summary
        """
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                # Basic stats
                summary = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade,
                        SUM(commission) as total_commission
                    FROM trading.positions
                    WHERE status = 'closed'
                    AND entry_time >= NOW() - INTERVAL '1 day' * $1
                """, days)

                if not summary or summary['total_trades'] == 0:
                    return {
                        'period_days': days,
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'total_pnl': 0.0,
                        'avg_daily_pnl': 0.0
                    }

                # Calculate key metrics
                win_rate = summary['winning_trades'] / summary['total_trades']
                avg_daily_pnl = summary['total_pnl'] / days

                # Get current portfolio metrics
                portfolio_metrics = await self._get_current_portfolio_metrics()

                return {
                    'period_days': days,
                    'total_trades': summary['total_trades'],
                    'winning_trades': summary['winning_trades'],
                    'win_rate': float(win_rate),
                    'total_pnl': float(summary['total_pnl']),
                    'avg_pnl': float(summary['avg_pnl']),
                    'avg_daily_pnl': float(avg_daily_pnl),
                    'best_trade': float(summary['best_trade']),
                    'worst_trade': float(summary['worst_trade']),
                    'total_commission': float(summary['total_commission']),
                    'current_portfolio_value': portfolio_metrics.get('total_equity', 0),
                    'current_positions': portfolio_metrics.get('position_count', 0),
                    'unrealized_pnl': portfolio_metrics.get('unrealized_pnl', 0)
                }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

    async def _get_current_portfolio_metrics(self) -> Dict[str, Any]:
        """Get current portfolio metrics."""
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                metrics = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as position_count,
                        SUM(market_value) as total_market_value,
                        SUM(unrealized_pnl) as total_unrealized_pnl
                    FROM trading.positions
                    WHERE status = 'open'
                """)

                # Get latest account snapshot
                account_snapshot = await conn.fetchrow("""
                    SELECT total_equity, cash, buying_power
                    FROM trading.account_snapshots
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)

                return {
                    'position_count': metrics['position_count'] if metrics else 0,
                    'total_market_value': float(metrics['total_market_value']) if metrics and metrics['total_market_value'] else 0,
                    'unrealized_pnl': float(metrics['total_unrealized_pnl']) if metrics and metrics['total_unrealized_pnl'] else 0,
                    'total_equity': float(account_snapshot['total_equity']) if account_snapshot else 0,
                    'cash': float(account_snapshot['cash']) if account_snapshot else 0,
                    'buying_power': float(account_snapshot['buying_power']) if account_snapshot else 0
                }

        except Exception as e:
            logger.error(f"Failed to get current portfolio metrics: {e}")
            return {}

    async def start_performance_monitoring(self):
        """Start background performance monitoring and updates."""
        self._running = True

        async def performance_monitor():
            """Background task for performance updates."""
            while self._running:
                try:
                    # Update daily metrics
                    await self.update_daily_metrics()

                    # Clean up old data
                    await self._cleanup_old_performance_data()

                    # Sleep for an hour
                    await asyncio.sleep(3600)

                except Exception as e:
                    logger.error(f"Error in performance monitoring: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes on error

        # Start monitoring task
        asyncio.create_task(performance_monitor())
        logger.info("Performance monitoring started")

    async def _cleanup_old_performance_data(self, days_to_keep: int = 365):
        """Clean up old performance data."""
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                # Clean up old trade performance records
                deleted_count = await conn.fetchval("""
                    DELETE FROM trading.trade_performance
                    WHERE entry_date < CURRENT_DATE - INTERVAL '1 day' * $1
                """, days_to_keep)

                if deleted_count and deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old trade performance records")

        except Exception as e:
            logger.error(f"Failed to cleanup old performance data: {e}")

    async def export_csv_report(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        strategy_name: Optional[str] = None
    ) -> str:
        """
        Export performance data as CSV format.

        Args:
            start_date: Start date for export
            end_date: End date for export
            strategy_name: Optional strategy filter

        Returns:
            CSV data as string
        """
        try:
            trades_data = await self.export_for_tradenote(start_date, end_date, strategy_name)

            if not trades_data:
                return "No trades found for export"

            # Convert to DataFrame for CSV export
            df = pd.DataFrame(trades_data)

            # Reorder columns for better readability
            column_order = [
                'symbol', 'side', 'entry_date', 'entry_time', 'exit_date', 'exit_time',
                'quantity', 'entry_price', 'exit_price', 'gross_pnl', 'net_pnl',
                'commission', 'return_pct', 'duration_hours', 'strategy', 'notes'
            ]

            # Only include columns that exist
            available_columns = [col for col in column_order if col in df.columns]
            df_ordered = df[available_columns]

            return df_ordered.to_csv(index=False)

        except Exception as e:
            logger.error(f"Failed to export CSV report: {e}")
            return f"Error exporting data: {e}"

    async def get_win_loss_distribution(self, days: int = 30) -> Dict[str, Any]:
        """
        Get win/loss distribution analysis.

        Args:
            days: Number of days to analyze

        Returns:
            Distribution analysis
        """
        try:
            if not self._db_pool:
                return {}
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                # Get P&L distribution
                pnl_data = await conn.fetch("""
                    SELECT pnl
                    FROM trading.positions
                    WHERE status = 'closed'
                    AND entry_time >= NOW() - INTERVAL '1 day' * $1
                    AND pnl IS NOT NULL
                    ORDER BY pnl
                """, days)

                if not pnl_data:
                    return {'error': 'No trade data available'}

                pnls = [float(row['pnl']) for row in pnl_data]

                # Calculate distribution metrics
                wins = [pnl for pnl in pnls if pnl > 0]
                losses = [pnl for pnl in pnls if pnl < 0]

                # Percentiles
                percentiles = [10, 25, 50, 75, 90, 95, 99]
                pnl_percentiles = {}
                for p in percentiles:
                    idx = int(len(pnls) * p / 100)
                    if idx < len(pnls):
                        pnl_percentiles[f'p{p}'] = pnls[idx]

                return {
                    'total_trades': len(pnls),
                    'winning_trades': len(wins),
                    'losing_trades': len(losses),
                    'win_rate': len(wins) / len(pnls) if pnls else 0,
                    'avg_win': sum(wins) / len(wins) if wins else 0,
                    'avg_loss': sum(losses) / len(losses) if losses else 0,
                    'largest_win': max(wins) if wins else 0,
                    'largest_loss': min(losses) if losses else 0,
                    'percentiles': pnl_percentiles,
                    'win_loss_ratio': (sum(wins) / len(wins)) / abs(sum(losses) / len(losses)) if wins and losses else None
                }

        except Exception as e:
            logger.error(f"Failed to get win/loss distribution: {e}")
            return {}

    async def calculate_monthly_returns(self, months: int = 12) -> List[Dict[str, Any]]:
        """
        Calculate monthly returns for the specified number of months.

        Args:
            months: Number of months to calculate

        Returns:
            List of monthly return data
        """
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                monthly_data = await conn.fetch("""
                    SELECT
                        EXTRACT(YEAR FROM date) as year,
                        EXTRACT(MONTH FROM date) as month,
                        SUM(net_pnl) as monthly_pnl,
                        AVG(portfolio_value) as avg_portfolio_value,
                        COUNT(DISTINCT date) as trading_days
                    FROM trading.daily_performance
                    WHERE date >= CURRENT_DATE - INTERVAL '1 month' * $1
                    GROUP BY EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date)
                    ORDER BY year DESC, month DESC
                """, months)

                monthly_returns = []
                for row in monthly_data:
                    month_return_pct = (float(row['monthly_pnl']) / float(row['avg_portfolio_value']) * 100) if row['avg_portfolio_value'] > 0 else 0

                    monthly_returns.append({
                        'year': int(row['year']),
                        'month': int(row['month']),
                        'month_name': datetime(int(row['year']), int(row['month']), 1).strftime('%B'),
                        'monthly_pnl': float(row['monthly_pnl']),
                        'monthly_return_pct': month_return_pct,
                        'trading_days': row['trading_days']
                    })

                return monthly_returns

        except Exception as e:
            logger.error(f"Failed to calculate monthly returns: {e}")
            return []

    async def get_strategy_comparison(self, days: int = 30) -> Dict[str, Any]:
        """
        Compare performance across different strategies.

        Args:
            days: Number of days to analyze

        Returns:
            Strategy comparison data
        """
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                strategies = await conn.fetch("""
                    SELECT DISTINCT strategy_type
                    FROM trading.positions
                    WHERE entry_time >= NOW() - INTERVAL '1 day' * $1
                    AND status = 'closed'
                """, days)

                strategy_comparison = []
                for strategy_row in strategies:
                    strategy_name = strategy_row['strategy_type']
                    performance = await self.calculate_strategy_performance(strategy_name, days)
                    strategy_comparison.append(performance)

                # Sort by total P&L
                strategy_comparison.sort(key=lambda x: x.get('total_pnl', 0), reverse=True)

                return {
                    'period_days': days,
                    'strategies': strategy_comparison,
                    'best_strategy': strategy_comparison[0] if strategy_comparison else None,
                    'total_strategies': len(strategy_comparison)
                }

        except Exception as e:
            logger.error(f"Failed to get strategy comparison: {e}")
            return {}

    async def calculate_correlation_matrix(self, symbols: List[str], days: int = 30) -> Dict[str, Any]:
        """
        Calculate correlation matrix for given symbols.

        Args:
            symbols: List of symbols to analyze
            days: Number of days for correlation calculation

        Returns:
            Correlation matrix and analysis
        """
        try:
            correlation_matrix = {}

            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        correlation_matrix[symbol1][symbol2] = 1.0
                    elif symbol2 in correlation_matrix and symbol1 in correlation_matrix[symbol2]:
                        # Use already calculated correlation
                        correlation_matrix[symbol1][symbol2] = correlation_matrix[symbol2][symbol1]
                    else:
                        # Calculate correlation
                        corr = await self._calculate_symbol_correlation(symbol1, symbol2, days)
                        correlation_matrix[symbol1][symbol2] = corr

            # Find highest correlations
            high_correlations = []
            for symbol1 in symbols:
                for symbol2 in symbols:
                    if symbol1 < symbol2:  # Avoid duplicates
                        corr = correlation_matrix[symbol1][symbol2]
                        if abs(corr) > 0.5:  # High correlation threshold
                            high_correlations.append({
                                'symbol1': symbol1,
                                'symbol2': symbol2,
                                'correlation': corr
                            })

            high_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

            return {
                'correlation_matrix': correlation_matrix,
                'high_correlations': high_correlations,
                'avg_correlation': self._calculate_avg_correlation(correlation_matrix),
                'analysis_period_days': days
            }

        except Exception as e:
            logger.error(f"Failed to calculate correlation matrix: {e}")
            return {}

    async def _calculate_symbol_correlation(self, symbol1: str, symbol2: str, days: int) -> float:
        """Calculate correlation between two symbols."""
        try:
            if not self._db_pool:
                return 0
            async with self._db_pool.acquire() as conn:
                # Get overlapping position returns
                correlation = await conn.fetchval("""
                    WITH symbol_returns AS (
                        SELECT
                            DATE(exit_time) as trade_date,
                            p1.pnl / p1.cost_basis as return1,
                            p2.pnl / p2.cost_basis as return2
                        FROM trading.positions p1
                        JOIN trading.positions p2 ON DATE(p1.exit_time) = DATE(p2.exit_time)
                        WHERE p1.ticker = $1 AND p2.ticker = $2
                        AND p1.status = 'closed' AND p2.status = 'closed'
                        AND p1.entry_time >= NOW() - INTERVAL '1 day' * $3
                        AND p1.cost_basis > 0 AND p2.cost_basis > 0
                    )
                    SELECT CORR(return1, return2)
                    FROM symbol_returns
                    WHERE return1 IS NOT NULL AND return2 IS NOT NULL
                """, symbol1, symbol2, days)

                return float(correlation) if correlation is not None else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate correlation between {symbol1} and {symbol2}: {e}")
            return 0.0

    def _calculate_avg_correlation(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate average correlation from matrix."""
        try:
            correlations = []
            symbols = list(correlation_matrix.keys())

            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:  # Avoid duplicates and self-correlation
                        corr = correlation_matrix[symbol1][symbol2]
                        correlations.append(abs(corr))

            return sum(correlations) / len(correlations) if correlations else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate average correlation: {e}")
            return 0.0
