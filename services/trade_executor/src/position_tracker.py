"""
Position Tracker for Trade Execution Service.

This module handles real-time position monitoring, state management,
and position lifecycle tracking with comprehensive risk metrics.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

import asyncpg
import redis.asyncio as redis
# from shared.models import Position as PositionModel  # Not used
from shared.config import get_config
from .alpaca_client import AlpacaClient


logger = logging.getLogger(__name__)


class PositionStatus:
    """Position status constants."""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    STOPPED_OUT = "stopped_out"
    PROFIT_TAKEN = "profit_taken"


class PositionTracker:
    """
    Real-time position tracking and management system.

    Handles:
    - Position state synchronization
    - Real-time P&L tracking
    - Risk metric calculation
    - Position lifecycle management
    - Stop loss and take profit monitoring
    """

    def __init__(self, alpaca_client: AlpacaClient):
        """
        Initialize position tracker.

        Args:
            alpaca_client: Alpaca API client instance
        """
        self.config = get_config()
        self.alpaca = alpaca_client
        self._db_pool = None
        self._redis = None
        self._positions_cache: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._price_callbacks: Dict[str, List] = {}

    async def initialize(self):
        """Initialize database and Redis connections."""
        try:
            # Initialize database connection pool
            self._db_pool = await asyncpg.create_pool(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                user=self.config.database.username,
                password=self.config.database.password,
                min_size=3,
                max_size=10,
                command_timeout=30
            )

            # Initialize Redis connection
            self._redis = redis.from_url(
                self.config.redis.url,
                max_connections=10,
                retry_on_timeout=True
            )

            # Load existing positions into cache
            await self._load_positions_cache()

            logger.info("PositionTracker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PositionTracker: {e}")
            raise

    async def cleanup(self):
        """Clean up resources."""
        try:
            self._running = False
            if self._db_pool:
                await self._db_pool.close()
            if self._redis:
                await self._redis.close()
            logger.info("PositionTracker cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _load_positions_cache(self):
        """Load active positions into cache."""
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM trading.positions
                    WHERE status = 'open'
                    ORDER BY entry_time DESC
                """)

                self._positions_cache.clear()
                for row in rows:
                    position_data = dict(row)
                    self._positions_cache[position_data['ticker']] = position_data

                logger.info(f"Loaded {len(self._positions_cache)} active positions into cache")

        except Exception as e:
            logger.error(f"Failed to load positions cache: {e}")

    async def create_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: Decimal,
        strategy_type: str,
        signal_id: Optional[UUID] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None
    ) -> UUID:
        """
        Create a new position record.

        Args:
            symbol: Trading symbol
            quantity: Position size (positive for long, negative for short)
            entry_price: Entry price
            strategy_type: Strategy that created this position
            signal_id: Associated signal ID
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Position ID
        """
        try:
            position_id = uuid4()
            side = "long" if quantity > 0 else "short"
            cost_basis = abs(Decimal(str(quantity))) * entry_price

            # Get account ID
            account = await self.alpaca.get_account()
            account_id = account.id

            if not self._db_pool:
                return uuid4()
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO trading.positions (
                        id, ticker, entry_time, entry_price, quantity,
                        stop_loss, take_profit, status, strategy_type,
                        signal_id, account_id, side, cost_basis,
                        current_price, market_value, unrealized_pnl
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """,
                    position_id, symbol, datetime.now(timezone.utc), entry_price, quantity,
                    stop_loss, take_profit, PositionStatus.OPEN, strategy_type,
                    signal_id, account_id, side, cost_basis,
                    entry_price, cost_basis, Decimal("0")
                )

            # Add to cache
            position_data = {
                'id': position_id,
                'ticker': symbol,
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': PositionStatus.OPEN,
                'strategy_type': strategy_type,
                'cost_basis': cost_basis,
                'market_value': cost_basis,
                'unrealized_pnl': Decimal("0"),
                'entry_time': datetime.now(timezone.utc)
            }
            self._positions_cache[symbol] = position_data

            # Publish position creation
            await self._publish_position_update(symbol, position_data, 'created')

            logger.info(f"Position created: {position_id} for {quantity} {symbol} @ {entry_price}")
            return position_id

        except Exception as e:
            logger.error(f"Failed to create position for {symbol}: {e}")
            raise

    async def update_position_price(self, symbol: str, current_price: Decimal, quote_data: Optional[Dict] = None):
        """
        Update position with current price and recalculate metrics.

        Args:
            symbol: Trading symbol
            current_price: Current market price
            quote_data: Optional quote data (bid/ask/volume)
        """
        try:
            if symbol not in self._positions_cache:
                return

            position = self._positions_cache[symbol]

            # Calculate new metrics
            quantity = position['quantity']
            entry_price = position['entry_price']

            # Calculate unrealized P&L
            if quantity > 0:  # Long position
                unrealized_pnl = (current_price - entry_price) * abs(quantity)
            else:  # Short position
                unrealized_pnl = (entry_price - current_price) * abs(quantity)

            market_value = current_price * abs(quantity)
            unrealized_pnl_pct = unrealized_pnl / position['cost_basis'] if position['cost_basis'] > 0 else Decimal("0")

            # Calculate distances to stop/profit levels
            distance_to_stop = None
            distance_to_profit = None

            if position['stop_loss'] and current_price:
                distance_to_stop = abs(current_price - position['stop_loss']) / current_price

            if position['take_profit'] and current_price:
                distance_to_profit = abs(current_price - position['take_profit']) / current_price

            # Update cache
            position.update({
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_percentage': unrealized_pnl_pct,
                'distance_to_stop_loss': distance_to_stop,
                'distance_to_take_profit': distance_to_profit,
                'last_updated': datetime.now(timezone.utc)
            })

            # Update database using stored function
            await self._update_position_snapshot(
                position['id'],
                current_price,
                quote_data.get('bid') if quote_data else None,
                quote_data.get('ask') if quote_data else None,
                quote_data.get('volume') if quote_data else None
            )

            # Check for stop loss/take profit triggers
            await self._check_exit_conditions(symbol, position, current_price)

            # Publish update
            await self._publish_position_update(symbol, position, 'updated')

        except Exception as e:
            logger.error(f"Failed to update position price for {symbol}: {e}")

    async def _update_position_snapshot(
        self,
        position_id: UUID,
        current_price: Decimal,
        bid_price: Optional[Decimal] = None,
        ask_price: Optional[Decimal] = None,
        volume: Optional[int] = None
    ):
        """Update position snapshot using database function."""
        try:
            if not self._db_pool:
                return
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    SELECT trading.update_position_snapshot($1, $2, $3, $4, $5)
                """, position_id, current_price, bid_price, ask_price, volume)

        except Exception as e:
            logger.error(f"Failed to update position snapshot: {e}")

    async def _check_exit_conditions(self, symbol: str, position: Dict[str, Any], current_price: Decimal):
        """Check if position should be exited based on stop loss or take profit."""
        try:
            quantity = position['quantity']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']

            should_exit = False
            exit_reason = None

            if quantity > 0:  # Long position
                if stop_loss and current_price <= stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif take_profit and current_price >= take_profit:
                    should_exit = True
                    exit_reason = "take_profit"
            else:  # Short position
                if stop_loss and current_price >= stop_loss:
                    should_exit = True
                    exit_reason = "stop_loss"
                elif take_profit and current_price <= take_profit:
                    should_exit = True
                    exit_reason = "take_profit"

            if should_exit:
                logger.info(f"Exit condition triggered for {symbol}: {exit_reason} at {current_price}")
                await self._trigger_position_exit(symbol, position, exit_reason or "unknown", current_price)

        except Exception as e:
            logger.error(f"Failed to check exit conditions for {symbol}: {e}")

    async def _trigger_position_exit(self, symbol: str, position: Dict[str, Any], reason: str, current_price: Decimal):
        """Trigger position exit."""
        try:
            # Update position status to closing
            await self._update_position_status(position['id'], PositionStatus.CLOSING)

            # Publish exit signal
            exit_signal = {
                'position_id': str(position['id']),
                'symbol': symbol,
                'reason': reason,
                'trigger_price': str(current_price),
                'quantity': position['quantity'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            if self._redis:
                import json
                await self._redis.publish(f"position_exits:{symbol}", json.dumps(exit_signal, default=str))
            logger.info(f"Position exit triggered for {symbol}: {reason}")

        except Exception as e:
            logger.error(f"Failed to trigger position exit for {symbol}: {e}")

    async def close_position(
        self,
        position_id: UUID,
        exit_price: Decimal,
        exit_time: Optional[datetime] = None,
        reason: str = "manual"
    ) -> Decimal:
        """
        Close a position and calculate final P&L.

        Args:
            position_id: Position ID to close
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for closing

        Returns:
            Final realized P&L
        """
        try:
            exit_time = exit_time or datetime.now(timezone.utc)

            # Use database function to close position
            if not self._db_pool:
                return Decimal("0")
            async with self._db_pool.acquire() as conn:
                final_pnl = await conn.fetchval("""
                    SELECT trading.close_position($1, $2, $3)
                """, position_id, exit_price, exit_time)

            # Get position details for cache update
            position = await self._get_position_by_id(position_id)
            if position and position['ticker'] in self._positions_cache:
                # Update cache
                self._positions_cache[position['ticker']].update({
                    'status': PositionStatus.CLOSED,
                    'exit_price': exit_price,
                    'exit_time': exit_time,
                    'pnl': final_pnl
                })

                # Publish closure
                await self._publish_position_update(
                    position['ticker'],
                    self._positions_cache[position['ticker']],
                    'closed'
                )

                # Remove from cache after publishing
                del self._positions_cache[position['ticker']]

            logger.info(f"Position {position_id} closed with P&L: {final_pnl}")
            return final_pnl

        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {e}")
            raise

    async def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position data or None if no position
        """
        try:
            # Check cache first
            if symbol in self._positions_cache:
                return self._positions_cache[symbol].copy()

            # Check database
            if not self._db_pool:
                return None
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM trading.positions
                    WHERE ticker = $1 AND status = 'open'
                    ORDER BY entry_time DESC
                    LIMIT 1
                """, symbol)

                if row:
                    position_data = dict(row)
                    self._positions_cache[symbol] = position_data
                    return position_data

                return None

        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    async def get_all_positions(self, include_closed: bool = False) -> List[Dict[str, Any]]:
        """
        Get all positions.

        Args:
            include_closed: Whether to include closed positions

        Returns:
            List of positions
        """
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                if include_closed:
                    query = """
                        SELECT * FROM trading.positions
                        ORDER BY entry_time DESC
                        LIMIT 1000
                    """
                else:
                    query = """
                        SELECT * FROM trading.positions
                        WHERE status = 'open'
                        ORDER BY entry_time DESC
                    """

                rows = await conn.fetch(query)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get all positions: {e}")
            return []

    async def sync_with_alpaca(self) -> Dict[str, Any]:
        """
        Synchronize positions with Alpaca broker.

        Returns:
            Sync result with any discrepancies
        """
        try:
            # Get positions from Alpaca
            alpaca_positions = await self.alpaca.get_positions(use_cache=False)

            # Get positions from database
            db_positions = await self.get_all_positions(include_closed=False)

            discrepancies = []
            synced_count = 0

            # Check each Alpaca position
            for alpaca_pos in alpaca_positions:
                db_pos = next((p for p in db_positions if p['ticker'] == alpaca_pos.symbol), None)

                if not db_pos:
                    # Position exists in Alpaca but not in DB
                    discrepancies.append({
                        'type': 'missing_in_db',
                        'symbol': alpaca_pos.symbol,
                        'alpaca_quantity': alpaca_pos.quantity,
                        'alpaca_avg_price': alpaca_pos.entry_price
                    })

                    # Create missing position
                    await self.create_position(
                        symbol=alpaca_pos.symbol,
                        quantity=alpaca_pos.quantity,
                        entry_price=alpaca_pos.entry_price,
                        strategy_type="manual_sync"
                    )
                    synced_count += 1

                elif db_pos['quantity'] != alpaca_pos.quantity:
                    # Quantity mismatch
                    discrepancies.append({
                        'type': 'quantity_mismatch',
                        'symbol': alpaca_pos.symbol,
                        'db_quantity': db_pos['quantity'],
                        'alpaca_quantity': alpaca_pos.quantity
                    })

                    # Update quantity in database
                    await self._update_position_quantity(db_pos['id'], alpaca_pos.quantity)
                    synced_count += 1
                else:
                    # Update price for existing position
                    await self.update_position_price(alpaca_pos.symbol, alpaca_pos.current_price)
                    synced_count += 1

            # Check for positions in DB but not in Alpaca
            alpaca_symbols = {pos.symbol for pos in alpaca_positions}
            for db_pos in db_positions:
                if db_pos['ticker'] not in alpaca_symbols and db_pos['status'] == 'open':
                    discrepancies.append({
                        'type': 'missing_in_alpaca',
                        'symbol': db_pos['ticker'],
                        'db_quantity': db_pos['quantity']
                    })

                    # Mark as closed (likely closed outside system)
                    current_price = await self._get_current_price(db_pos['ticker'])
                    await self.close_position(db_pos['id'], current_price, reason="sync_missing")

            logger.info(f"Position sync completed: {synced_count} synced, {len(discrepancies)} discrepancies")

            return {
                'success': True,
                'synced_positions': synced_count,
                'discrepancies': discrepancies,
                'timestamp': datetime.now(timezone.utc)
            }

        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            return {'success': False, 'error': str(e)}

    async def calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio metrics.

        Returns:
            Portfolio risk and performance metrics
        """
        try:
            positions = list(self._positions_cache.values())
            if not positions:
                return {
                    'total_positions': 0,
                    'total_market_value': Decimal("0"),
                    'total_unrealized_pnl': Decimal("0"),
                    'long_exposure': Decimal("0"),
                    'short_exposure': Decimal("0"),
                    'net_exposure': Decimal("0"),
                    'largest_position_pct': Decimal("0"),
                    'concentration_risk': Decimal("0")
                }

            # Calculate basic metrics
            total_market_value = sum(Decimal(str(p.get('market_value', 0))) for p in positions)
            total_unrealized_pnl = sum(Decimal(str(p.get('unrealized_pnl', 0))) for p in positions)

            long_exposure = sum(
                Decimal(str(p.get('market_value', 0)))
                for p in positions if p.get('quantity', 0) > 0
            )

            short_exposure = sum(
                abs(Decimal(str(p.get('market_value', 0))))
                for p in positions if p.get('quantity', 0) < 0
            )

            net_exposure = long_exposure - short_exposure

            # Calculate concentration risk
            largest_position = max(
                (abs(Decimal(str(p.get('market_value', 0)))) for p in positions),
                default=Decimal("0")
            )

            largest_position_pct = (largest_position / total_market_value * 100) if total_market_value > 0 else Decimal("0")

            # Calculate concentration risk (Herfindahl index)
            if total_market_value > 0:
                concentration_risk = sum(
                    (abs(Decimal(str(p.get('market_value', 0)))) / total_market_value) ** 2
                    for p in positions
                )
            else:
                concentration_risk = Decimal("0")

            return {
                'total_positions': len(positions),
                'long_positions': sum(1 for p in positions if p.get('quantity', 0) > 0),
                'short_positions': sum(1 for p in positions if p.get('quantity', 0) < 0),
                'total_market_value': total_market_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'net_exposure': net_exposure,
                'largest_position_pct': largest_position_pct,
                'concentration_risk': concentration_risk,
                'avg_position_size': total_market_value / len(positions) if positions else Decimal("0"),
                'portfolio_return_pct': (total_unrealized_pnl / (total_market_value - total_unrealized_pnl) * 100) if total_market_value > total_unrealized_pnl else Decimal("0")
            }

        except Exception as e:
            logger.error(f"Failed to calculate portfolio metrics: {e}")
            return {}

    async def get_position_history(self, symbol: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get position history.

        Args:
            symbol: Optional symbol filter
            days: Number of days to look back

        Returns:
            List of historical positions
        """
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                if symbol:
                    query = """
                        SELECT * FROM trading.positions
                        WHERE ticker = $1 AND entry_time >= NOW() - INTERVAL '1 day' * $2
                        ORDER BY entry_time DESC
                    """
                    rows = await conn.fetch(query, symbol, days)
                else:
                    query = """
                        SELECT * FROM trading.positions
                        WHERE entry_time >= NOW() - INTERVAL '1 day' * $1
                        ORDER BY entry_time DESC
                        LIMIT 1000
                    """
                    rows = await conn.fetch(query, days)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get position history: {e}")
            return []

    async def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get position performance summary.

        Args:
            days: Number of days to analyze

        Returns:
            Performance summary
        """
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                summary = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_positions,
                        COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_positions,
                        COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_positions,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade,
                        SUM(commission) as total_commission,
                        AVG(EXTRACT(EPOCH FROM (exit_time - entry_time))/3600) as avg_holding_hours
                    FROM trading.positions
                    WHERE status = 'closed'
                    AND entry_time >= NOW() - INTERVAL '1 day' * $1
                """, days)

                if summary and summary['total_positions'] > 0:
                    win_rate = summary['winning_positions'] / summary['total_positions']

                    # Calculate profit factor
                    gross_profit = await conn.fetchval("""
                        SELECT SUM(pnl) FROM trading.positions
                        WHERE status = 'closed' AND pnl > 0
                        AND entry_time >= NOW() - INTERVAL '%s days'
                    """, days) or Decimal("0")

                    gross_loss = await conn.fetchval("""
                        SELECT SUM(ABS(pnl)) FROM trading.positions
                        WHERE status = 'closed' AND pnl < 0
                        AND entry_time >= NOW() - INTERVAL '%s days'
                    """, days) or Decimal("0")

                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

                    # Calculate expectancy
                    avg_win = gross_profit / summary['winning_positions'] if summary['winning_positions'] > 0 else Decimal("0")
                    avg_loss = gross_loss / summary['losing_positions'] if summary['losing_positions'] > 0 else Decimal("0")
                    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

                    return {
                        'period_days': days,
                        'total_positions': summary['total_positions'],
                        'winning_positions': summary['winning_positions'],
                        'losing_positions': summary['losing_positions'],
                        'win_rate': float(win_rate),
                        'total_pnl': float(summary['total_pnl']),
                        'avg_pnl': float(summary['avg_pnl']),
                        'best_trade': float(summary['best_trade']),
                        'worst_trade': float(summary['worst_trade']),
                        'total_commission': float(summary['total_commission']),
                        'avg_holding_hours': float(summary['avg_holding_hours'] or 0),
                        'profit_factor': float(profit_factor) if profit_factor else None,
                        'expectancy': float(expectancy),
                        'gross_profit': float(gross_profit),
                        'gross_loss': float(gross_loss)
                    }
                else:
                    return {
                        'period_days': days,
                        'total_positions': 0,
                        'win_rate': 0.0,
                        'total_pnl': 0.0,
                        'profit_factor': None,
                        'expectancy': 0.0
                    }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}

    async def start_real_time_tracking(self):
        """Start real-time position tracking."""
        self._running = True

        # Get symbols to track
        symbols = list(self._positions_cache.keys())
        if not symbols:
            logger.info("No positions to track")
            return

        async def price_update_callback(data_type: str, data: Dict[str, Any]):
            """Handle real-time price updates."""
            try:
                if data_type == 'quote':
                    symbol = data['symbol']
                    mid_price = (data['bid'] + data['ask']) / 2
                    await self.update_position_price(symbol, mid_price, data)
                elif data_type == 'trade':
                    symbol = data['symbol']
                    await self.update_position_price(symbol, data['price'])

            except Exception as e:
                logger.error(f"Error processing price update: {e}")

        try:
            # Start position streaming
            await self.alpaca.start_position_stream(callback=price_update_callback)
        except Exception as e:
            logger.error(f"Failed to start real-time tracking: {e}")

    async def _publish_position_update(self, symbol: str, position_data: Dict[str, Any], action: str):
        """Publish position update to Redis."""
        try:
            update_message = {
                'action': action,
                'symbol': symbol,
                'position_id': str(position_data['id']),
                'quantity': position_data['quantity'],
                'current_price': str(position_data.get('current_price', 0)),
                'unrealized_pnl': str(position_data.get('unrealized_pnl', 0)),
                'market_value': str(position_data.get('market_value', 0)),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

            # Publish to symbol-specific channel
            if self._redis:
                import json
                await self._redis.publish(f"positions:{symbol}", json.dumps(update_message, default=str))
                await self._redis.publish("positions:all", json.dumps(update_message, default=str))

        except Exception as e:
            logger.error(f"Failed to publish position update for {symbol}: {e}")

    async def _get_position_by_id(self, position_id: UUID) -> Optional[Dict[str, Any]]:
        """Get position by ID from database."""
        try:
            if not self._db_pool:
                return None
            async with self._db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM trading.positions WHERE id = $1
                """, position_id)

                if row:
                    return dict(row)
                return None

        except Exception as e:
            logger.error(f"Failed to get position {position_id}: {e}")
            return None

    async def _update_position_status(self, position_id: UUID, status: str):
        """Update position status."""
        try:
            if not self._db_pool:
                return False
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trading.positions
                    SET status = $2, updated_at = NOW()
                    WHERE id = $1
                """, position_id, status)

        except Exception as e:
            logger.error(f"Failed to update position status {position_id}: {e}")

    async def _update_position_quantity(self, position_id: UUID, new_quantity: int):
        """Update position quantity."""
        try:
            if not self._db_pool:
                return False
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trading.positions
                    SET quantity = $2, side = $3, updated_at = NOW()
                    WHERE id = $1
                """, position_id, new_quantity, "long" if new_quantity > 0 else "short")

        except Exception as e:
            logger.error(f"Failed to update position quantity {position_id}: {e}")

    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current price for symbol."""
        try:
            quote = await self.alpaca.get_latest_quote(symbol)
            if not quote or 'bid' not in quote or 'ask' not in quote:
                logger.warning(f"No quote data available for {symbol}")
                if symbol in self._positions_cache:
                    return self._positions_cache[symbol].get('current_price', Decimal("0"))
                return Decimal("0")
            return (quote['bid'] + quote['ask']) / 2
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            # Fallback to last known price
            if symbol in self._positions_cache:
                return self._positions_cache[symbol].get('current_price', Decimal("0"))
            raise

    async def update_stop_loss(self, position_id: UUID, new_stop_loss: Decimal) -> bool:
        """
        Update stop loss for a position.

        Args:
            position_id: Position ID
            new_stop_loss: New stop loss price

        Returns:
            True if update successful
        """
        try:
            if not self._db_pool:
                return False
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trading.positions
                    SET stop_loss = $2, updated_at = NOW()
                    WHERE id = $1 AND status = 'open'
                """, position_id, new_stop_loss)

            # Update cache
            position = await self._get_position_by_id(position_id)
            if position and position['ticker'] in self._positions_cache:
                self._positions_cache[position['ticker']]['stop_loss'] = new_stop_loss

            logger.info(f"Stop loss updated for position {position_id}: {new_stop_loss}")
            return True

        except Exception as e:
            logger.error(f"Failed to update stop loss for {position_id}: {e}")
            return False

    async def update_take_profit(self, position_id: UUID, new_take_profit: Decimal) -> bool:
        """
        Update take profit for a position.

        Args:
            position_id: Position ID
            new_take_profit: New take profit price

        Returns:
            True if update successful
        """
        try:
            if not self._db_pool:
                return False
            async with self._db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE trading.positions
                    SET take_profit = $2, updated_at = NOW()
                    WHERE id = $1 AND status = 'open'
                """, position_id, new_take_profit)

            # Update cache
            position = await self._get_position_by_id(position_id)
            if position and position['ticker'] in self._positions_cache:
                self._positions_cache[position['ticker']]['take_profit'] = new_take_profit

            logger.info(f"Take profit updated for position {position_id}: {new_take_profit}")
            return True

        except Exception as e:
            logger.error(f"Failed to update take profit for {position_id}: {e}")
            return False

    async def get_positions_at_risk(self, max_loss_pct: float = 0.02) -> List[Dict[str, Any]]:
        """
        Get positions that are at risk (near stop loss).

        Args:
            max_loss_pct: Maximum loss percentage threshold

        Returns:
            List of at-risk positions
        """
        try:
            at_risk_positions = []

            for symbol, position in self._positions_cache.items():
                if position['status'] != PositionStatus.OPEN:
                    continue

                current_price = position.get('current_price')
                entry_price = position.get('entry_price')
                # stop_loss = position.get('stop_loss')  # Not used in current logic

                if not all([current_price, entry_price]):
                    continue

                # Calculate current loss percentage
                if position['quantity'] > 0:  # Long position
                    current_loss_pct = (entry_price - current_price) / entry_price if entry_price else 0
                else:  # Short position
                    current_loss_pct = (current_price - entry_price) / entry_price if entry_price else 0

                # Check if position is at risk
                if current_loss_pct >= max_loss_pct * 0.8:  # 80% of max loss threshold
                    risk_data = {
                        'position_id': position['id'],
                        'symbol': symbol,
                        'current_loss_pct': float(current_loss_pct),
                        'unrealized_pnl': position.get('unrealized_pnl', 0),
                        'distance_to_stop': position.get('distance_to_stop_loss'),
                        'market_value': position.get('market_value', 0)
                    }
                    at_risk_positions.append(risk_data)

            return sorted(at_risk_positions, key=lambda x: x['current_loss_pct'], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get positions at risk: {e}")
            return []

    async def get_position_snapshots(self, position_id: UUID, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get position price snapshots for analysis.

        Args:
            position_id: Position ID
            hours: Number of hours to look back

        Returns:
            List of position snapshots
        """
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM trading.position_snapshots
                    WHERE position_id = $1
                    AND timestamp >= NOW() - INTERVAL '%s hours'
                    ORDER BY timestamp ASC
                """, position_id, hours)

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get position snapshots for {position_id}: {e}")
            return []

    async def calculate_position_metrics(self, position_id: UUID) -> Dict[str, Any]:
        """
        Calculate detailed metrics for a position.

        Args:
            position_id: Position ID

        Returns:
            Position metrics
        """
        try:
            # Get position data
            position = await self._get_position_by_id(position_id)
            if not position:
                return {}

            # Get snapshots for analysis
            snapshots = await self.get_position_snapshots(position_id, hours=24)

            if not snapshots:
                return {'error': 'No snapshot data available'}

            # Calculate metrics
            prices = [Decimal(str(s['current_price'])) for s in snapshots]
            pnls = [Decimal(str(s['unrealized_pnl'])) for s in snapshots]

            # Maximum Favorable Excursion (MFE)
            mfe = max(pnls) if pnls else Decimal("0")

            # Maximum Adverse Excursion (MAE)
            mae = min(pnls) if pnls else Decimal("0")

            # Price volatility
            if len(prices) > 1:
                price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                avg_volatility = sum(price_changes) / len(price_changes)
            else:
                avg_volatility = Decimal("0")

            # Current metrics
            # current_price = position.get('current_price', position['entry_price'])  # Not used in current logic
            unrealized_pnl = position.get('unrealized_pnl', 0)
            return_pct = unrealized_pnl / position['cost_basis'] if position['cost_basis'] > 0 else Decimal("0")

            # Time metrics
            holding_time = datetime.now(timezone.utc) - position['entry_time']

            return {
                'position_id': position_id,
                'symbol': position['ticker'],
                'holding_time_hours': holding_time.total_seconds() / 3600,
                'current_return_pct': float(return_pct),
                'unrealized_pnl': float(unrealized_pnl),
                'mfe': float(mfe),
                'mae': float(mae),
                'avg_volatility': float(avg_volatility),
                'snapshots_count': len(snapshots),
                'distance_to_stop_loss': position.get('distance_to_stop_loss'),
                'distance_to_take_profit': position.get('distance_to_take_profit')
            }

        except Exception as e:
            logger.error(f"Failed to calculate position metrics for {position_id}: {e}")
            return {'error': str(e)}

    async def export_tradenote_format(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Export trade data in TradeNote compatible format.

        Args:
            start_date: Start date for export
            end_date: End date for export

        Returns:
            List of trades in TradeNote format
        """
        try:
            if not self._db_pool:
                return []
            async with self._db_pool.acquire() as conn:
                conditions = ["status = 'closed'"]
                params = []

                if start_date:
                    conditions.append("entry_time >= $1")
                    params.append(start_date)

                if end_date:
                    param_num = len(params) + 1
                    conditions.append(f"exit_time <= ${param_num}")
                    params.append(end_date)

                where_clause = " WHERE " + " AND ".join(conditions)

                query = f"""
                    SELECT
                        ticker as symbol,
                        entry_time,
                        exit_time,
                        side,
                        quantity,
                        entry_price,
                        exit_price,
                        pnl,
                        commission,
                        strategy_type,
                        EXTRACT(EPOCH FROM (exit_time - entry_time))/3600 as duration_hours
                    FROM trading.positions
                    {where_clause}
                    ORDER BY exit_time DESC
                """

                rows = await conn.fetch(query, *params)

                tradenote_data = []
                for row in rows:
                    trade_data = {
                        'symbol': row['symbol'],
                        'side': 'Long' if row['side'] == 'long' else 'Short',
                        'entry_date': row['entry_time'].strftime('%Y-%m-%d'),
                        'entry_time': row['entry_time'].strftime('%H:%M:%S'),
                        'exit_date': row['exit_time'].strftime('%Y-%m-%d') if row['exit_time'] else None,
                        'exit_time': row['exit_time'].strftime('%H:%M:%S') if row['exit_time'] else None,
                        'quantity': abs(row['quantity']),
                        'entry_price': float(row['entry_price']),
                        'exit_price': float(row['exit_price']) if row['exit_price'] else None,
                        'gross_pnl': float(row['pnl'] + row['commission']) if row['pnl'] and row['commission'] else None,
                        'net_pnl': float(row['pnl']) if row['pnl'] else None,
                        'commission': float(row['commission']) if row['commission'] else 0,
                        'strategy': row['strategy_type'],
                        'duration_hours': float(row['duration_hours']) if row['duration_hours'] else None,
                        'notes': f"Strategy: {row['strategy_type']}"
                    }
                    tradenote_data.append(trade_data)

                logger.info(f"Exported {len(tradenote_data)} trades for TradeNote")
                return tradenote_data

        except Exception as e:
            logger.error(f"Failed to export TradeNote data: {e}")
            return []

    async def get_drawdown_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Analyze portfolio drawdown over time.

        Args:
            days: Number of days to analyze

        Returns:
            Drawdown analysis
        """
        try:
            if not self._db_pool:
                return {}
            async with self._db_pool.acquire() as conn:
                # Get daily portfolio values
                rows = await conn.fetch("""
                    SELECT
                        date,
                        portfolio_value,
                        net_pnl,
                        SUM(net_pnl) OVER (ORDER BY date) as cumulative_pnl
                    FROM trading.daily_performance
                    WHERE date >= CURRENT_DATE - INTERVAL '%s days'
                    ORDER BY date
                """, days)

                if not rows:
                    return {'max_drawdown': 0, 'current_drawdown': 0, 'drawdown_duration_days': 0}

                # Calculate drawdown
                peak_value = Decimal("0")
                max_drawdown = Decimal("0")
                current_drawdown = Decimal("0")
                drawdown_start = None
                max_drawdown_duration = 0

                for row in rows:
                    current_value = Decimal(str(row['cumulative_pnl']))

                    # Update peak
                    if current_value > peak_value:
                        peak_value = current_value
                        if drawdown_start:
                            # Drawdown ended
                            duration = (row['date'] - drawdown_start).days
                            max_drawdown_duration = max(max_drawdown_duration, duration)
                            drawdown_start = None

                    # Calculate drawdown
                    if peak_value > 0:
                        drawdown = (peak_value - current_value) / peak_value
                        max_drawdown = max(max_drawdown, drawdown)

                        if drawdown > 0 and not drawdown_start:
                            drawdown_start = row['date']

                        if row == rows[-1]:  # Last row
                            current_drawdown = drawdown

                return {
                    'max_drawdown': float(max_drawdown),
                    'current_drawdown': float(current_drawdown),
                    'max_drawdown_duration_days': max_drawdown_duration,
                    'peak_portfolio_value': float(peak_value),
                    'analysis_period_days': days
                }

        except Exception as e:
            logger.error(f"Failed to analyze drawdown: {e}")
            return {}

    async def monitor_position_risks(self):
        """Monitor positions for risk conditions."""
        while self._running:
            try:
                # Check each position for risk conditions
                for symbol, position in self._positions_cache.items():
                    try:
                        # Check for large unrealized losses
                        unrealized_pnl = position.get('unrealized_pnl', 0)
                        cost_basis = position.get('cost_basis', 1)
                        loss_pct = abs(unrealized_pnl) / cost_basis if unrealized_pnl < 0 else 0

                        if loss_pct > 0.05:  # 5% loss threshold
                            if self._redis:
                                risk_alert = {
                                    'type': 'large_loss',
                                    'position_id': str(position['id']),
                                    'symbol': symbol,
                                    'loss_percentage': float(loss_pct),
                                    'unrealized_pnl': float(unrealized_pnl),
                                    'timestamp': datetime.now(timezone.utc).isoformat()
                                }
                                import json
                                await self._redis.publish(f"risk_alerts:{symbol}", json.dumps(risk_alert, default=str))

                        # Check for stale positions (no price updates)
                        last_updated = position.get('last_updated', datetime.now(timezone.utc))
                        if datetime.now(timezone.utc) - last_updated > timedelta(minutes=10):
                            logger.warning(f"Stale position data for {symbol}: last updated {last_updated}")

                    except Exception as e:
                        logger.error(f"Error monitoring position {symbol}: {e}")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in position risk monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def cleanup_old_snapshots(self, days_to_keep: int = 7):
        """
        Clean up old position snapshots to manage database size.

        Args:
            days_to_keep: Number of days of snapshots to keep
        """
        try:
            if not self._db_pool:
                return 0.0
            async with self._db_pool.acquire() as conn:
                deleted_count = await conn.fetchval("""
                    DELETE FROM trading.position_snapshots
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                    RETURNING COUNT(*)
                """, days_to_keep)

                if deleted_count:
                    logger.info(f"Cleaned up {deleted_count} old position snapshots")

        except Exception as e:
            logger.error(f"Failed to cleanup old snapshots: {e}")

    async def get_position_correlation(self, symbol1: str, symbol2: str, days: int = 30) -> float:
        """
        Calculate correlation between two positions.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            days: Number of days for correlation calculation

        Returns:
            Correlation coefficient (-1 to 1)
        """
        try:
            if not self._db_pool:
                return 0.0
            async with self._db_pool.acquire() as conn:
                # Get price changes for both symbols
                rows = await conn.fetch("""
                    WITH price_changes AS (
                        SELECT
                            ps1.timestamp,
                            (ps1.current_price - LAG(ps1.current_price) OVER (ORDER BY ps1.timestamp)) / LAG(ps1.current_price) OVER (ORDER BY ps1.timestamp) as return1,
                            (ps2.current_price - LAG(ps2.current_price) OVER (ORDER BY ps2.timestamp)) / LAG(ps2.current_price) OVER (ORDER BY ps2.timestamp) as return2
                        FROM trading.position_snapshots ps1
                        JOIN trading.position_snapshots ps2 ON ps1.timestamp = ps2.timestamp
                        JOIN trading.positions p1 ON ps1.position_id = p1.id
                        JOIN trading.positions p2 ON ps2.position_id = p2.id
                        WHERE p1.ticker = $1 AND p2.ticker = $2
                        AND ps1.timestamp >= NOW() - INTERVAL '%s days'
                    )
                    SELECT
                        CORR(return1, return2) as correlation
                    FROM price_changes
                    WHERE return1 IS NOT NULL AND return2 IS NOT NULL
                """, symbol1, symbol2, days)

                correlation = rows[0]['correlation'] if rows and rows[0]['correlation'] else 0.0
                return float(correlation)

        except Exception as e:
            logger.error(f"Failed to calculate correlation between {symbol1} and {symbol2}: {e}")
            return 0.0
