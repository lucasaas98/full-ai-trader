"""
AI Strategy Integration Module

This module handles the integration of the AI strategy engine with the rest of the trading system,
including Redis pub/sub, database persistence, and real-time decision making.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import polars as pl
import redis.asyncio as redis
from ai_models import (
    AIDecisionRecord,
    AIPerformanceMetrics,
    MarketRegimeState,
    create_performance_summary,
    init_database,
)
from ai_strategy import AIDecision, AIStrategyEngine
from base_strategy import Signal, StrategyConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)


class AIStrategyIntegration:
    """
    Integrates AI strategy with the trading system infrastructure.

    Handles:
    - Real-time data subscription from Redis
    - AI decision persistence to PostgreSQL
    - Signal publishing to execution engine
    - Performance monitoring and reporting
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        db_connection_string: str,
        config: Dict[str, Any],
    ):
        """
        Initialize AI strategy integration.

        Args:
            redis_client: Redis client for pub/sub
            db_connection_string: PostgreSQL connection string
            config: Configuration dictionary
        """
        self.redis_client = redis_client
        self.config = config

        # Initialize database
        self.engine = create_async_engine(
            db_connection_string.replace("postgresql://", "postgresql+asyncpg://"),
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
        )
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)

        # Initialize AI strategy
        from base_strategy import StrategyMode

        strategy_config = StrategyConfig(
            name="ai_strategy",
            mode=(
                StrategyMode.LONG_SHORT
                if hasattr(StrategyMode, "LONG_SHORT")
                else StrategyMode.LONG
            ),
            parameters={
                "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY"),
                **config.get("strategy_parameters", {}),
            },
        )
        self.ai_strategy = AIStrategyEngine(strategy_config)

        # Initialize Redis components
        try:
            from redis_integration import RedisDataSubscriber, RedisSignalPublisher

            self.signal_publisher = RedisSignalPublisher(redis_client)
            self.data_subscriber = RedisDataSubscriber(redis_client)
        except ImportError:
            # Fallback if redis_integration is not available
            self.signal_publisher = None
            self.data_subscriber = None

        # Data buffers
        self.price_data_buffer: Dict[str, pl.DataFrame] = {}
        self.finviz_data_buffer: Dict[str, Dict] = {}
        self.market_data_buffer: Dict[str, Any] = {}

        # Active positions tracking
        self.active_positions: Dict[str, Dict] = {}

        # Performance tracking
        self.decision_count = 0
        self.last_performance_update = datetime.now()

        # Callbacks
        self.signal_callbacks: List[Callable] = []

    async def initialize(self) -> None:
        """Initialize all components and start listening for data."""
        logger.info("Initializing AI Strategy Integration")

        # Initialize database tables
        async with self.engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: init_database(str(self.engine.url)))

        # Initialize AI strategy
        self.ai_strategy.initialize()

        # Register data callbacks
        if self.data_subscriber and hasattr(
            self.data_subscriber, "register_price_callback"
        ):
            self.data_subscriber.register_price_callback(self._handle_screener_update)
        if self.data_subscriber and hasattr(
            self.data_subscriber, "register_screener_callback"
        ):
            self.data_subscriber.register_screener_callback(
                self._handle_screener_update
            )

        # Start background tasks
        asyncio.create_task(self._market_regime_monitor())
        asyncio.create_task(self._position_monitor())
        asyncio.create_task(self._position_monitor())

        # Start listening for data
        if self.data_subscriber:
            await self.data_subscriber.start_listening()

        logger.info("AI Strategy Integration initialized successfully")

    async def process_price_data(self, data: Dict[str, Any]) -> None:
        """
        Handle incoming price updates from Redis.

        Args:
            data: Price update data
        """
        try:
            ticker = data.get("ticker")
            if not ticker:
                return

            # Update price buffer
            new_row = pl.DataFrame(
                {
                    "timestamp": [datetime.fromisoformat(data["timestamp"])],
                    "open": [data["open"]],
                    "high": [data["high"]],
                    "low": [data["low"]],
                    "close": [data["close"]],
                    "volume": [data["volume"]],
                }
            )

            if ticker in self.price_data_buffer:
                self.price_data_buffer[ticker] = pl.concat(
                    [self.price_data_buffer[ticker], new_row]
                ).tail(
                    500
                )  # Keep last 500 candles
            else:
                self.price_data_buffer[ticker] = new_row

            # Check if we have enough data to analyze
            if len(self.price_data_buffer[ticker]) >= 50:
                await self._analyze_ticker(ticker)

        except Exception as e:
            logger.error(f"Error handling price update: {e}")

    async def _handle_screener_update(self, data: Dict[str, Any]) -> None:
        """
        Handle incoming screener data updates.

        Args:
            data: Screener data containing stocks and metadata
        """
        try:
            # Extract screener data format: {screener_type, data: [stocks], timestamp, count}
            screener_type = data.get("screener_type", "unknown")
            stocks_data = data.get("data", [])

            logger.info(
                f"Received screener update: {screener_type} with {len(stocks_data)} stocks"
            )

            # Process each stock in the screener results
            for stock in stocks_data:
                symbol = stock.get("symbol")
                if symbol:
                    self.finviz_data_buffer[symbol] = stock
                    logger.debug(f"Updated screener data for {symbol}")
        except Exception as e:
            logger.error(f"Error handling screener update: {e}")

    async def _analyze_ticker(self, ticker: str) -> None:
        """
        Analyze a ticker using AI strategy.

        Args:
            ticker: Stock ticker symbol
        """
        try:
            # Get price data
            price_data = self.price_data_buffer.get(ticker)
            if price_data is None or len(price_data) < 50:
                return

            # Get FinViz data if available
            _ = self.finviz_data_buffer.get(ticker)

            # Run AI analysis
            signal = await self.ai_strategy.analyze(ticker, price_data)

            # Process signal if significant
            if signal.confidence >= self.config.get("min_confidence", 60):
                await self._process_ai_signal(ticker, signal)

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")

    async def _process_ai_signal(self, ticker: str, signal: Signal) -> None:
        """
        Process and publish AI-generated trading signal.

        Args:
            ticker: Stock ticker
            signal: Trading signal from AI
        """
        try:
            # Import RedisChannels if available
            try:
                from redis_integration import RedisChannels
            except ImportError:
                logger.warning("RedisChannels not available")
                return

            # Extract AI decision from signal metadata
            ai_decision = signal.metadata.get("ai_decision")
            if not ai_decision:
                logger.warning(f"No AI decision in signal for {ticker}")
                return

            # Save decision to database
            await self._save_ai_decision(ticker, ai_decision, signal)

            # Check position limits
            if not await self._check_position_limits(ticker, signal):
                logger.info(f"Position limits exceeded for {ticker}, skipping signal")
                return

            # Publish signal to Redis
            if self.signal_publisher and hasattr(RedisChannels, "STRATEGY_SIGNALS"):
                await self.signal_publisher.publish_signal(
                    RedisChannels.STRATEGY_SIGNALS,
                    {
                        "timestamp": datetime.now().isoformat(),
                        "ticker": ticker,
                        "action": signal.action,
                        "confidence": signal.confidence,
                        "entry_price": signal.entry_price,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "position_size": signal.position_size,
                        "reasoning": (
                            ai_decision.reasoning
                            if isinstance(ai_decision, AIDecision)
                            else ""
                        ),
                        "strategy": "ai_strategy",
                        "metadata": signal.metadata,
                    },
                )

            # Track decision
            self.decision_count += 1

            # Execute callbacks
            for callback in self.signal_callbacks:
                await callback(ticker, signal)

            logger.info(
                f"Published AI signal for {ticker}: {signal.action} @ {signal.confidence}% confidence"
            )

        except Exception as e:
            logger.error(f"Error processing AI signal for {ticker}: {e}")

    async def _save_ai_decision(
        self, ticker: str, ai_decision: Any, signal: Signal
    ) -> None:
        """
        Save AI decision to database for tracking and analysis.

        Args:
            ticker: Stock ticker
            ai_decision: AI decision object
            signal: Trading signal
        """
        try:
            async with self.async_session() as session:
                # Create decision record
                decision_record = AIDecisionRecord(
                    timestamp=datetime.now(),
                    ticker=ticker,
                    decision=signal.action,
                    confidence=signal.confidence,
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=signal.position_size,
                    risk_reward_ratio=signal.metadata.get("risk_reward_ratio"),
                    models_used=signal.metadata.get("models_used", []),
                    consensus_details=signal.metadata.get("consensus_details"),
                    total_tokens=signal.metadata.get("total_tokens"),
                    total_cost=signal.metadata.get("total_cost"),
                    market_context=self.market_data_buffer,
                    reasoning=signal.metadata.get("reasoning"),
                    key_risks=signal.metadata.get("key_risks", []),
                    strategy_version=self.config.get("version", "1.0.0"),
                    prompt_versions=signal.metadata.get("prompt_versions", {}),
                    execution_time_ms=signal.metadata.get("execution_time_ms"),
                )

                session.add(decision_record)
                await session.commit()

                logger.debug(f"Saved AI decision for {ticker} to database")

        except Exception as e:
            logger.error(f"Error saving AI decision to database: {e}")

    async def _check_position_limits(self, ticker: str, signal: Signal) -> bool:
        """
        Check if taking this position would exceed risk limits.

        Args:
            ticker: Stock ticker
            signal: Trading signal

        Returns:
            True if position can be taken, False otherwise
        """
        # Check maximum positions
        max_positions = self.config.get("max_positions", 10)
        if (
            len(self.active_positions) >= max_positions
            and ticker not in self.active_positions
        ):
            return False

        # Check position size limits
        max_position_size = self.config.get("max_position_size", 0.1)
        if signal.position_size > max_position_size:
            signal.position_size = max_position_size

        # Check daily loss limit
        daily_loss_limit = self.config.get("daily_loss_limit", -1000)
        current_daily_pnl = sum(
            pos.get("unrealized_pnl", 0) for pos in self.active_positions.values()
        )
        if current_daily_pnl < daily_loss_limit:
            logger.warning(f"Daily loss limit reached: {current_daily_pnl}")
            return False

        return True

    async def _market_regime_monitor(self) -> None:
        """Monitor and update market regime periodically."""
        while True:
            try:
                # Update every 30 minutes
                await asyncio.sleep(1800)

                # Get market data
                market_indices = await self._fetch_market_indices()

                # Update AI strategy's market context
                if hasattr(self.ai_strategy, "update_market_context"):
                    await self.ai_strategy.update_market_context(market_indices)

                # Save market regime to database
                await self._save_market_regime(market_indices)

            except Exception as e:
                logger.error(f"Error in market regime monitor: {e}")
                await asyncio.sleep(60)

    async def _position_monitor(self) -> None:
        """Monitor open positions and generate exit signals."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for ticker, position in self.active_positions.items():
                    # Get latest price
                    if ticker not in self.price_data_buffer:
                        continue

                    latest_price = float(self.price_data_buffer[ticker]["close"][-1])

                    # Calculate unrealized P&L
                    entry_price = position["entry_price"]
                    quantity = position["quantity"]
                    unrealized_pnl = (latest_price - entry_price) * quantity

                    position["unrealized_pnl"] = unrealized_pnl
                    position["current_price"] = latest_price

                    # Check exit conditions
                    should_exit = False
                    exit_reason = None

                    # Stop loss
                    if (
                        position.get("stop_loss")
                        and latest_price <= position["stop_loss"]
                    ):
                        should_exit = True
                        exit_reason = "stop_loss"

                    # Take profit
                    elif (
                        position.get("take_profit")
                        and latest_price >= position["take_profit"]
                    ):
                        should_exit = True
                        exit_reason = "take_profit"

                    # Time-based exit for day trades
                    elif position.get("timeframe") == "day_trade":
                        position_age = (
                            datetime.now() - position["entry_time"]
                        ).total_seconds() / 3600
                        if position_age > 6:  # Exit after 6 hours
                            should_exit = True
                            exit_reason = "time_limit"

                    # AI-based exit signal
                    else:
                        # Query AI for exit decision
                        exit_signal = await self._get_ai_exit_signal(ticker, position)
                        if exit_signal and exit_signal.action == "SELL":
                            should_exit = True
                            exit_reason = "ai_signal"

                    if should_exit and exit_reason:
                        await self._publish_exit_signal(ticker, position, exit_reason)

            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(10)

    async def _get_ai_exit_signal(
        self, ticker: str, position: Dict
    ) -> Optional[Signal]:
        """
        Get AI-based exit signal for a position.

        Args:
            ticker: Stock ticker
            position: Position details

        Returns:
            Exit signal or None
        """
        try:
            # Prepare context for exit optimizer
            price_data = self.price_data_buffer.get(ticker)
            if price_data is None or len(price_data) < 20:
                return None

            # Create position context
            _ = {
                "entry_price": position["entry_price"],
                "current_price": position["current_price"],
                "unrealized_pnl": position["unrealized_pnl"],
                "holding_time": (
                    datetime.now() - position["entry_time"]
                ).total_seconds()
                / 3600,
            }

            # Query AI for exit decision
            # This would use the exit_optimizer prompt
            # For now, return None (would be implemented in ai_strategy.py)
            return None

        except Exception as e:
            logger.error(f"Error getting AI exit signal: {e}")
            return None

    async def _publish_exit_signal(
        self, ticker: str, position: Dict, reason: str
    ) -> None:
        """
        Publish exit signal for a position.

        Args:
            ticker: Stock ticker
            position: Position details
            reason: Exit reason
        """
        try:
            # Import RedisChannels if available
            try:
                from redis_integration import RedisChannels
            except ImportError:
                logger.warning("RedisChannels not available")
                return

            exit_signal = {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "action": "SELL",
                "confidence": 100,  # Exit signals have high confidence
                "entry_price": position["current_price"],
                "position_size": position["quantity"],
                "reasoning": f"Exit signal: {reason}",
                "strategy": "ai_strategy",
                "metadata": {
                    "exit_reason": reason,
                    "entry_price": position["entry_price"],
                    "unrealized_pnl": position["unrealized_pnl"],
                },
            }

            if self.signal_publisher and hasattr(RedisChannels, "STRATEGY_SIGNALS"):
                await self.signal_publisher.publish_signal(
                    RedisChannels.STRATEGY_SIGNALS, exit_signal
                )

            # Remove from active positions
            del self.active_positions[ticker]

            logger.info(f"Published exit signal for {ticker}: {reason}")

        except Exception as e:
            logger.error(f"Error publishing exit signal: {e}")

    async def _performance_tracker(self) -> None:
        """Monitor and report strategy performance."""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour

                # Calculate performance metrics
                async with self.async_session() as session:
                    # Get recent decisions
                    cutoff_time = datetime.now() - timedelta(days=1)
                    result = await session.execute(
                        text("SELECT * FROM ai_decisions WHERE timestamp > :cutoff"),
                        {"cutoff": cutoff_time},
                    )
                    recent_decisions = result.fetchall()

                    # Get executions
                    result = await session.execute(
                        text(
                            "SELECT * FROM ai_trade_executions WHERE executed_at > :cutoff"
                        ),
                        {"cutoff": cutoff_time},
                    )
                    recent_executions = result.fetchall()

                    # Create performance summary
                    if recent_decisions:
                        performance = create_performance_summary(
                            recent_decisions, recent_executions
                        )

                        # Save metrics to database
                        metrics = AIPerformanceMetrics(
                            date=datetime.now(),
                            total_decisions=performance.total_decisions,
                            accuracy_rate=performance.accuracy_rate,
                            win_rate=performance.win_rate,
                            total_pnl=performance.total_pnl,
                            total_api_cost=performance.total_api_cost,
                            avg_cost_per_decision=(
                                performance.total_api_cost / performance.total_decisions
                                if performance.total_decisions > 0
                                else 0
                            ),
                        )

                        session.add(metrics)
                        await session.commit()

                        # Log performance
                        logger.info(
                            f"Performance Update - Decisions: {performance.total_decisions}, "
                            f"Accuracy: {performance.accuracy_rate:.2%}, "
                            f"Win Rate: {performance.win_rate:.2%}, "
                            f"P&L: ${performance.total_pnl:.2f}, "
                            f"API Cost: ${performance.total_api_cost:.2f}"
                        )

            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)

    async def _fetch_market_indices(self) -> Dict[str, Any]:
        """
        Fetch current market indices data.

        Returns:
            Market indices data
        """
        # This would fetch real market data from a data provider
        # For now, return placeholder data
        return {
            "spy_price": 450.00,
            "spy_change": 0.5,
            "qqq_price": 380.00,
            "qqq_change": 0.8,
            "vix_level": 15.5,
            "vix_change": -0.3,
        }

    async def _save_market_regime(self, market_data: Dict) -> None:
        """
        Save market regime assessment to database.

        Args:
            market_data: Current market data
        """
        try:
            async with self.async_session() as session:
                regime_state = MarketRegimeState(
                    timestamp=datetime.now(),
                    regime=market_data.get("regime", "unknown"),
                    strength=market_data.get("strength", 50),
                    risk_level=market_data.get("risk_level", "medium"),
                    spy_price=market_data.get("spy_price"),
                    spy_change=market_data.get("spy_change"),
                    vix_level=market_data.get("vix_level"),
                    vix_change=market_data.get("vix_change"),
                )

                session.add(regime_state)
                await session.commit()

        except Exception as e:
            logger.error(f"Error saving market regime: {e}")

    async def register_signal_callback(self, callback: Callable) -> None:
        """
        Register a callback for when signals are generated.

        Args:
            callback: Async function to call with (ticker, signal)
        """
        self.signal_callbacks.append(callback)

    async def add_position(
        self, ticker: str, entry_price: float, quantity: int, **kwargs: Any
    ) -> None:
        """
        Add a position to track.

        Args:
            ticker: Stock ticker
            entry_price: Entry price
            quantity: Number of shares
            **kwargs: Additional position details
        """
        self.active_positions[ticker] = {
            "entry_price": entry_price,
            "quantity": quantity,
            "entry_time": datetime.now(),
            "current_price": entry_price,
            "unrealized_pnl": 0,
            **kwargs,
        }
        logger.info(f"Added position: {ticker} @ {entry_price} x {quantity}")

    async def remove_position(self, ticker: str) -> None:
        """
        Remove a position from tracking.

        Args:
            ticker: Stock ticker
        """
        if ticker in self.active_positions:
            del self.active_positions[ticker]
            logger.info(f"Removed position: {ticker}")

    async def get_strategy_status(self) -> Dict[str, Any]:
        """
        Get current strategy status and metrics.

        Returns:
            Status dictionary
        """
        return {
            "status": "running",
            "decision_count": self.decision_count,
            "active_positions": len(self.active_positions),
            "total_unrealized_pnl": sum(
                pos.get("unrealized_pnl", 0) for pos in self.active_positions.values()
            ),
            "last_update": datetime.now().isoformat(),
            "market_regime": self.market_data_buffer.get("regime", "unknown"),
            "buffered_tickers": list(self.price_data_buffer.keys()),
        }

    async def stop(self) -> None:
        """Gracefully shutdown the integration."""
        logger.info("Shutting down AI Strategy Integration")

        # Stop data subscription
        if self.data_subscriber:
            await self.data_subscriber.stop_listening()

        # Close database connection
        await self.engine.dispose()

        # Clear buffers
        self.price_data_buffer.clear()
        self.finviz_data_buffer.clear()
        self.market_data_buffer.clear()

        logger.info("AI Strategy Integration shutdown complete")


async def main() -> None:
    """Main entry point for testing."""
    logging.basicConfig(level=logging.INFO)

    # Initialize Redis
    redis_client = await redis.from_url("redis://localhost:6379")

    # Configuration
    config = {
        "min_confidence": 60,
        "max_positions": 10,
        "max_position_size": 0.1,
        "daily_loss_limit": -1000,
        "version": "1.0.0",
        "strategy_parameters": {"use_cache": True, "consensus_min_models": 3},
    }

    # Initialize integration
    integration = AIStrategyIntegration(
        redis_client=redis_client,
        db_connection_string="postgresql://user:password@localhost/trading",
        config=config,
    )

    # Start integration
    await integration.initialize()

    # Run for testing
    try:
        while True:
            status = await integration.get_strategy_status()
            logger.info(f"Strategy Status: {status}")
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        await integration.stop()


if __name__ == "__main__":
    asyncio.run(main())
