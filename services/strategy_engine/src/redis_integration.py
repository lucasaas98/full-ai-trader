"""
Redis Integration Module

This module handles Redis integration for the strategy engine, including:
- Subscribing to price updates from data collector
- Publishing trading signals to channels
- Caching strategy states and analysis results
- Managing real-time signal distribution
"""

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List, Optional

import redis.asyncio as redis

from shared.models import FinVizData, MarketData, SignalType

from .base_strategy import BaseStrategy, Signal
from .hybrid_strategy import HybridSignal, HybridSignalGenerator, HybridStrategy


class RedisChannels:
    """Redis channel names for the trading system."""

    # Input channels (subscribe)
    PRICE_UPDATES = "price_updates"
    MARKET_DATA = "market_data:{symbol}"
    SCREENER_UPDATES = "screener:updates"
    SYSTEM_EVENTS = "system_events"

    # Output channels (publish)
    SIGNALS = "signals:{symbol}"
    STRATEGY_ALERTS = "strategy_alerts"
    BACKTEST_RESULTS = "backtest_results"
    REGIME_UPDATES = "regime_updates"

    # Cache keys
    STRATEGY_STATE = "strategy_state:{strategy_name}"
    ANALYSIS_CACHE = "analysis_cache:{symbol}:{timeframe}"
    REGIME_CACHE = "regime_cache:{symbol}"
    SIGNAL_HISTORY = "signal_history:{symbol}"


class RedisSignalPublisher:
    """Publish trading signals to Redis channels."""

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize signal publisher.

        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client
        self.logger = logging.getLogger("redis_signal_publisher")
        self.signal_generator = HybridSignalGenerator()

    async def publish_signal(
        self, symbol: str, signal: HybridSignal, strategy_type: str = "hybrid"
    ) -> bool:
        """
        Publish trading signal to Redis channel.

        Args:
            symbol: Trading symbol
            signal: Trading signal to publish
            strategy_type: Strategy type for formatting

        Returns:
            True if published successfully
        """
        try:
            # Generate formatted signal
            formatted_signal = self.signal_generator.generate_formatted_signal(
                symbol, signal, strategy_type
            )

            # Add additional metadata
            formatted_signal.update(
                {
                    "published_at": datetime.now(timezone.utc).isoformat() + "Z",
                    "signal_id": f"{symbol}_{int(datetime.now(timezone.utc).timestamp())}",
                    "technical_score": getattr(signal, "technical_score", 0.0),
                    "fundamental_score": getattr(signal, "fundamental_score", 0.0),
                    "combined_score": getattr(signal, "combined_score", 0.0),
                    "signal_strength": str(
                        getattr(signal, "signal_strength", "moderate")
                    ),
                    "regime_adjusted": getattr(signal, "regime_adjusted", False),
                }
            )

            # Publish to symbol-specific channel
            channel = RedisChannels.SIGNALS.format(symbol=symbol)
            await self.redis.publish(channel, json.dumps(formatted_signal, default=str))

            # Also publish to general signals channel for monitoring
            await self.redis.publish(
                "all_signals", json.dumps(formatted_signal, default=str)
            )

            # Cache signal for history
            await self._cache_signal_history(symbol, formatted_signal)

            self.logger.info(
                f"Published {signal.action.value} signal for {symbol} "
                f"(confidence: {signal.confidence:.0f})"
            )

            return True

        except Exception as e:
            self.logger.error(f"Error publishing signal for {symbol}: {e}")
            return False

    async def publish_strategy_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "info",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish strategy alert or notification.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            metadata: Additional alert metadata

        Returns:
            True if published successfully
        """
        try:
            alert = {
                "type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "metadata": metadata or {},
            }

            await self.redis.publish(
                RedisChannels.STRATEGY_ALERTS, json.dumps(alert, default=str)
            )

            self.logger.info(f"Published {severity} alert: {alert_type}")
            return True

        except Exception as e:
            self.logger.error(f"Error publishing alert: {e}")
            return False

    async def publish_regime_update(
        self, symbol: str, regime_info: Dict[str, Any]
    ) -> bool:
        """
        Publish market regime update.

        Args:
            symbol: Trading symbol
            regime_info: Market regime information

        Returns:
            True if published successfully
        """
        try:
            regime_update = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "regime_info": regime_info,
            }

            await self.redis.publish(
                RedisChannels.REGIME_UPDATES, json.dumps(regime_update, default=str)
            )

            # Cache regime state
            cache_key = RedisChannels.REGIME_CACHE.format(symbol=symbol)
            await self.redis.setex(
                cache_key, 3600, json.dumps(regime_update, default=str)
            )  # 1 hour TTL

            return True

        except Exception as e:
            self.logger.error(f"Error publishing regime update for {symbol}: {e}")
            return False

    async def _cache_signal_history(self, symbol: str, signal: Dict[str, Any]) -> None:
        """Cache signal in history for analysis."""
        try:
            history_key = RedisChannels.SIGNAL_HISTORY.format(symbol=symbol)

            # Add to list (keep last 100 signals)
            self.redis.lpush(history_key, json.dumps(signal, default=str))
            self.redis.ltrim(history_key, 0, 99)  # Keep last 100
            self.redis.expire(history_key, 86400 * 7)  # 7 days TTL

        except Exception as e:
            self.logger.error(f"Error caching signal history: {e}")


class RedisDataSubscriber:
    """Subscribe to market data updates from Redis."""

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize data subscriber.

        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client
        self.logger = logging.getLogger("redis_data_subscriber")
        self._subscriptions: Dict[str, Any] = {}
        self._callbacks: Dict[str, Any] = {}
        self._running = False

    async def subscribe_to_price_updates(
        self, symbols: List[str], callback: Callable[[str, Dict], None]
    ) -> None:
        """
        Subscribe to price updates for specific symbols.

        Args:
            symbols: List of symbols to subscribe to
            callback: Callback function for price updates
        """
        try:
            self._callbacks["price_updates"] = callback

            # Subscribe to general price updates channel
            pubsub = self.redis.pubsub()
            await pubsub.subscribe(RedisChannels.PRICE_UPDATES)

            # Subscribe to symbol-specific channels
            for symbol in symbols:
                channel = RedisChannels.MARKET_DATA.format(symbol=symbol)
                await pubsub.subscribe(channel)

            self._subscriptions["price_updates"] = pubsub

            self.logger.info(f"Subscribed to price updates for {len(symbols)} symbols")

        except Exception as e:
            self.logger.error(f"Error subscribing to price updates: {e}")

    async def subscribe_to_screener_updates(
        self, callback: Callable[[Dict], None]
    ) -> None:
        """
        Subscribe to screener data updates.

        Args:
            callback: Callback function for screener updates
        """
        try:
            self._callbacks["screener_updates"] = callback

            pubsub = self.redis.pubsub()
            await pubsub.subscribe(RedisChannels.SCREENER_UPDATES)

            self._subscriptions["screener_updates"] = pubsub

            self.logger.info("Subscribed to screener updates")

        except Exception as e:
            self.logger.error(f"Error subscribing to screener updates: {e}")
            raise

    async def start_listening(self) -> None:
        """Start listening for Redis messages."""
        try:
            self._running = True
            self.logger.info("Started Redis message listener")

            # Create tasks for each subscription
            tasks = []
            for sub_type, pubsub in self._subscriptions.items():
                task = asyncio.create_task(
                    self._listen_to_subscription(sub_type, pubsub)
                )
                tasks.append(task)

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Error in Redis listener: {e}")
        finally:
            self._running = False

    async def _listen_to_subscription(self, sub_type: str, pubsub: Any) -> None:
        """Listen to a specific subscription."""
        try:
            callback = self._callbacks.get(sub_type)
            if not callback:
                return

            async for message in pubsub.listen():
                if not self._running:
                    break

                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])

                        # Route message based on subscription type
                        if sub_type == "price_updates":
                            symbol = data.get("symbol", "")
                            callback(symbol, data)
                        elif sub_type == "screener_updates":
                            callback(data)

                    except Exception as e:
                        self.logger.error(f"Error processing {sub_type} message: {e}")

        except Exception as e:
            self.logger.error(f"Error listening to {sub_type}: {e}")

    async def stop_listening(self) -> None:
        """Stop listening and cleanup subscriptions."""
        try:
            self._running = False

            for pubsub in self._subscriptions.values():
                await pubsub.unsubscribe()
                await pubsub.close()

            self._subscriptions.clear()
            self._callbacks.clear()

            self.logger.info("Stopped Redis listener and cleaned up subscriptions")

        except Exception as e:
            self.logger.error(f"Error stopping Redis listener: {e}")


class RedisStrategyCache:
    """Cache strategy states and analysis results in Redis."""

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize strategy cache.

        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client
        self.logger = logging.getLogger("redis_strategy_cache")

    async def cache_strategy_state(self, strategy: BaseStrategy) -> bool:
        """
        Cache strategy state in Redis.

        Args:
            strategy: Strategy to cache

        Returns:
            True if cached successfully
        """
        try:
            state = strategy.save_state()
            cache_key = RedisChannels.STRATEGY_STATE.format(strategy_name=strategy.name)

            await self.redis.setex(
                cache_key, 3600, json.dumps(state, default=str)  # 1 hour TTL
            )

            self.logger.debug(f"Cached state for strategy {strategy.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error caching strategy state: {e}")
            return False

    async def load_strategy_state(self, strategy: BaseStrategy) -> bool:
        """
        Load strategy state from Redis cache.

        Args:
            strategy: Strategy to load state into

        Returns:
            True if loaded successfully
        """
        try:
            cache_key = RedisChannels.STRATEGY_STATE.format(strategy_name=strategy.name)
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                state = json.loads(cached_data)
                strategy.load_state(state)
                self.logger.debug(f"Loaded cached state for strategy {strategy.name}")
                return True
            else:
                self.logger.debug(f"No cached state found for strategy {strategy.name}")
                return False

        except Exception as e:
            self.logger.error(f"Error loading strategy state: {e}")
            return False

    async def cache_analysis_result(
        self, symbol: str, timeframe: str, analysis: Dict[str, Any], ttl: int = 300
    ) -> bool:
        """
        Cache analysis result for reuse.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            analysis: Analysis results
            ttl: Time to live in seconds

        Returns:
            True if cached successfully
        """
        try:
            cache_key = RedisChannels.ANALYSIS_CACHE.format(
                symbol=symbol, timeframe=timeframe
            )

            cache_data = {
                "analysis": analysis,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
            }

            await self.redis.setex(cache_key, ttl, json.dumps(cache_data, default=str))

            return True

        except Exception as e:
            self.logger.error(f"Error caching analysis result: {e}")
            return False

    async def get_cached_analysis(
        self, symbol: str, timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached analysis result.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe

        Returns:
            Cached analysis if available
        """
        try:
            cache_key = RedisChannels.ANALYSIS_CACHE.format(
                symbol=symbol, timeframe=timeframe
            )
            cached_data = await self.redis.get(cache_key)

            if cached_data:
                cache_obj = json.loads(cached_data)

                # Check if cache is still valid (additional time check)
                cache_time = datetime.fromisoformat(cache_obj["timestamp"])
                if datetime.now(timezone.utc) - cache_time < timedelta(minutes=5):
                    return cache_obj["analysis"]

            return None

        except Exception as e:
            self.logger.error(f"Error getting cached analysis: {e}")
            return None

    async def invalidate_cache(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.

        Args:
            pattern: Redis key pattern to match

        Returns:
            Number of keys deleted
        """
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                deleted = await self.redis.delete(*keys)
                self.logger.info(
                    f"Invalidated {deleted} cache entries matching {pattern}"
                )
                return deleted
            return 0

        except Exception as e:
            self.logger.error(f"Error invalidating cache: {e}")
            return 0


class RedisMarketDataHandler:
    """Handle market data from Redis streams."""

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize market data handler.

        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client
        self.logger = logging.getLogger("redis_market_data")
        self._price_callbacks: Dict[str, Any] = {}
        self._finviz_callbacks: Dict[str, Any] = {}

    def register_price_callback(
        self, symbol: str, callback: Callable[[MarketData], None]
    ) -> None:
        """
        Register callback for price updates.

        Args:
            symbol: Trading symbol
            callback: Callback function
        """
        if symbol not in self._price_callbacks:
            self._price_callbacks[symbol] = []
        self._price_callbacks[symbol].append(callback)

    def register_screener_callback(
        self, callback: Callable[[Dict], Awaitable[None]]
    ) -> None:
        """
        Register callback for screener data updates.

        Args:
            callback: Async callback function
        """
        if "screener" not in self._finviz_callbacks:
            self._finviz_callbacks["screener"] = []
        self._finviz_callbacks["screener"].append(callback)

    async def process_price_update(
        self, symbol: str, price_data: Dict[str, Any]
    ) -> None:
        """
        Process incoming price update.

        Args:
            symbol: Trading symbol
            price_data: Price update data
        """
        try:
            # Convert to MarketData model
            market_data = MarketData(
                symbol=price_data["symbol"],
                timestamp=datetime.fromisoformat(price_data["timestamp"]),
                open=Decimal(str(price_data["open"])),
                high=Decimal(str(price_data["high"])),
                low=Decimal(str(price_data["low"])),
                close=Decimal(str(price_data["close"])),
                volume=int(price_data["volume"]),
                timeframe=price_data.get("timeframe", "1m"),
                adjusted_close=Decimal(
                    str(price_data.get("adjusted_close", price_data["close"]))
                ),
            )

            # Call registered callbacks
            callbacks = self._price_callbacks.get(symbol, [])
            for callback in callbacks:
                try:
                    await callback(market_data)
                except Exception as e:
                    self.logger.error(f"Error in price callback for {symbol}: {e}")

        except Exception as e:
            self.logger.error(f"Error processing price update for {symbol}: {e}")

    async def process_screener_update(self, screener_data: Dict[str, Any]) -> None:
        """
        Process incoming screener update.

        Args:
            screener_data: Screener update data
        """
        try:
            # Process screener data from data collector
            # screener_data contains: screener_type, data (list of stocks), timestamp, count

            screener_type = screener_data.get("screener_type", "unknown")
            stocks_data = screener_data.get("data", [])

            self.logger.info(
                f"Processing screener update: {screener_type} with {len(stocks_data)} stocks"
            )

            # Extract symbols for strategy processing
            _ = [stock.get("symbol") for stock in stocks_data if stock.get("symbol")]

            # Notify all registered screener callbacks
            callbacks = self._finviz_callbacks.get("screener", [])
            for callback in callbacks:
                try:
                    await callback(screener_data)
                except Exception as e:
                    self.logger.error(f"Error in screener callback: {e}")

        except Exception as e:
            self.logger.error(f"Error processing screener update: {e}")


class RedisStrategyEngine:
    """Main Redis-integrated strategy engine."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis strategy engine.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.logger = logging.getLogger("redis_strategy_engine")

        # Redis clients
        self.redis_async = None
        self.redis_sync = None
        self.redis = None  # For compatibility

        # Components
        self.publisher = None
        self.subscriber = None
        self.cache = None
        self.data_handler = None

        # Strategy management
        self.active_strategies: Dict[str, Any] = {}
        self.strategy_tasks: Dict[str, Any] = {}

        # Configuration
        self.config = {
            "max_concurrent_analysis": 10,
            "signal_cache_ttl": 300,  # 5 minutes
            "analysis_cache_ttl": 60,  # 1 minute
            "heartbeat_interval": 30,  # seconds
            "reconnect_attempts": 5,
            "reconnect_delay": 5,  # seconds
        }

    async def initialize(self) -> bool:
        """Initialize Redis connections and components."""
        try:
            # Initialize Redis clients
            self.redis_async = redis.from_url(
                self.redis_url,
                decode_responses=True,
                retry_on_timeout=True,
                health_check_interval=30,
            )

            # Remove sync redis client - not needed for async operations

            # Set redis attribute for compatibility
            self.redis = self.redis_async

            # Test connections
            if self.redis_async:
                await self.redis_async.ping()
            if self.redis_sync:
                self.redis_sync.ping()

            # Initialize components
            if self.redis_async:
                self.publisher = RedisSignalPublisher(self.redis_async)
                self.subscriber = RedisDataSubscriber(self.redis_async)
                self.cache = RedisStrategyCache(self.redis_async)
                self.data_handler = RedisMarketDataHandler(self.redis_async)
            else:
                raise Exception("Redis async client not initialized")

            self.logger.info("Redis strategy engine initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing Redis strategy engine: {e}")
            return False

    async def register_strategy(
        self, strategy: BaseStrategy, symbols: List[str]
    ) -> bool:
        """
        Register strategy for real-time processing.

        Args:
            strategy: Strategy to register
            symbols: Symbols to track for this strategy

        Returns:
            True if registered successfully
        """
        try:
            strategy_id = strategy.name

            # Load cached state if available
            if self.cache:
                await self.cache.load_strategy_state(strategy)

            # Store strategy
            self.active_strategies[strategy_id] = {
                "strategy": strategy,
                "symbols": symbols,
                "last_analysis": {},
                "signal_count": 0,
                "error_count": 0,
                "status": "active",
            }

            # Register price update callbacks
            for symbol in symbols:
                if self.data_handler:

                    def make_price_callback(strat, sym):
                        def callback(data):
                            asyncio.create_task(
                                self._handle_price_update(strat, sym, data)
                            )

                        return callback

                    self.data_handler.register_price_callback(
                        symbol, make_price_callback(strategy, symbol)
                    )

            # Register FinViz callback if strategy supports it
            if isinstance(strategy, HybridStrategy) and self.data_handler:

                def make_screener_callback(strat):
                    async def callback(data):
                        await self._handle_screener_update(strat, data)

                    return callback

                self.data_handler.register_screener_callback(
                    make_screener_callback(strategy)
                )

            self.logger.info(
                f"Registered strategy {strategy_id} for {len(symbols)} symbols"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error registering strategy {strategy.name}: {e}")
            return False

    async def _handle_price_update(
        self, strategy: BaseStrategy, symbol: str, market_data: MarketData
    ) -> None:
        """Handle price update for a strategy."""
        try:
            strategy_id = strategy.name

            if strategy_id not in self.active_strategies:
                return

            strategy_info = self.active_strategies[strategy_id]

            # Check if we need historical data for analysis
            # This would typically fetch from database or cache
            historical_data = await self._get_historical_data(
                symbol, strategy.config.lookback_period
            )

            if historical_data is None:
                self.logger.warning(f"No historical data available for {symbol}")
                return

            # Check cache for recent analysis
            cached_analysis = (
                await self.cache.get_cached_analysis(symbol, "1h")
                if self.cache
                else None
            )

            if cached_analysis:
                self.logger.debug(f"Using cached analysis for {symbol}")
                # Use cached result if recent enough
                # This would extract signal from cached analysis
                return

            # Perform real-time analysis
            try:
                if isinstance(strategy, HybridStrategy):
                    # Get FinViz data for hybrid analysis
                    finviz_data = await self._get_finviz_data(symbol)
                    signal_result = await strategy.analyze(
                        symbol, historical_data, finviz_data
                    )
                    # Convert Signal to HybridSignal if needed
                    if hasattr(signal_result, "signal_type"):
                        # It's already a HybridSignal
                        signal = signal_result
                    else:
                        # Convert Signal to HybridSignal
                        from hybrid_strategy import HybridSignal

                        signal = HybridSignal(
                            action=signal_result.action,
                            confidence=signal_result.confidence,
                            position_size=signal_result.position_size,
                            reasoning=getattr(signal_result, "reasoning", ""),
                            technical_score=0.5,
                            fundamental_score=0.5,
                            ai_score=0.0,
                            risk_score=getattr(signal_result, "risk_score", 0.5),
                            timestamp=getattr(
                                signal_result, "timestamp", datetime.now(timezone.utc)
                            ),
                        )
                else:
                    signal_result = await strategy.analyze(symbol, historical_data)  # type: ignore
                    # Convert Signal to HybridSignal if needed
                    if isinstance(signal_result, HybridSignal):
                        # It's already a HybridSignal
                        hybrid_signal = signal_result
                    else:
                        # Convert Signal to HybridSignal
                        from hybrid_strategy import HybridSignal

                        hybrid_signal = HybridSignal(
                            action=signal_result.action,
                            confidence=signal_result.confidence,
                            position_size=signal_result.position_size,
                            reasoning=getattr(signal_result, "reasoning", ""),
                            technical_score=0.5,
                            fundamental_score=0.5,
                            ai_score=0.0,
                            risk_score=getattr(signal_result, "risk_score", 0.5),
                            timestamp=getattr(
                                signal_result, "timestamp", datetime.now(timezone.utc)
                            ),
                        )

                # Use the appropriate signal (hybrid_signal if converted, otherwise signal_result)
                final_signal = (
                    hybrid_signal if "hybrid_signal" in locals() else signal_result
                )

                # Update strategy info
                strategy_info["last_analysis"][symbol] = {
                    "timestamp": datetime.now(timezone.utc),
                    "signal": final_signal,
                    "confidence": final_signal.confidence,
                }

                # Check if signal should be published
                if final_signal.confidence >= strategy.config.min_confidence:
                    await self._publish_strategy_signal(strategy, symbol, final_signal)
                    strategy_info["signal_count"] += 1

                # Cache analysis result
                if self.cache:
                    await self.cache.cache_analysis_result(
                        symbol,
                        "realtime",
                        {"signal": asdict(signal)},
                        ttl=self.config["analysis_cache_ttl"],
                    )

            except Exception as e:
                self.logger.error(f"Analysis error for {strategy_id}/{symbol}: {e}")
                strategy_info["error_count"] += 1

                # Publish error alert
                if self.publisher:
                    await self.publisher.publish_strategy_alert(
                        "analysis_error",
                        f"Analysis failed for {symbol}: {str(e)}",
                        "error",
                        {"strategy": strategy_id, "symbol": symbol},
                    )

        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")

    async def _handle_screener_update(
        self, strategy: BaseStrategy, screener_data: Dict
    ) -> None:
        """Handle screener data update for hybrid strategies."""
        try:
            _ = screener_data.get("screener_type", "unknown")
            stocks_data = screener_data.get("data", [])

            if not stocks_data:
                return

            strategy_id = strategy.name
            if strategy_id not in self.active_strategies:
                return

            self.logger.info(
                f"Processing {len(stocks_data)} screener stocks for strategy {strategy_id}"
            )

            # Process each stock from the screener
            for stock in stocks_data:
                symbol = stock.get("symbol")
                if not symbol:
                    continue

                # Update screener cache
                screener_key = f"screener_data:{symbol}"
                if self.redis:
                    await self.redis.setex(
                        screener_key, 3600, json.dumps(stock, default=str)  # 1 hour TTL
                    )

            self.logger.debug(f"Updated screener cache for {len(stocks_data)} symbols")

        except Exception as e:
            self.logger.error(f"Error handling screener update: {e}")

    async def _publish_strategy_signal(
        self, strategy: BaseStrategy, symbol: str, signal: Signal
    ) -> None:
        """Publish strategy signal to Redis."""
        try:
            # Publish signal
            if self.publisher:
                # Convert Signal to HybridSignal if needed
                if isinstance(strategy, HybridStrategy) and not isinstance(
                    signal, HybridSignal
                ):
                    hybrid_signal = HybridSignal(
                        action=signal.action,
                        confidence=signal.confidence,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        position_size=signal.position_size,
                        reasoning=signal.reasoning,
                        timestamp=signal.timestamp,
                        metadata=signal.metadata,
                        technical_score=getattr(signal, "technical_score", 0.5),
                        fundamental_score=getattr(signal, "fundamental_score", 0.5),
                        combined_score=getattr(signal, "combined_score", 0.5),
                    )
                    await self.publisher.publish_signal(
                        symbol, hybrid_signal, strategy.name
                    )
                else:
                    # Convert regular Signal to HybridSignal for publisher compatibility
                    hybrid_signal = HybridSignal(
                        action=signal.action,
                        confidence=signal.confidence,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        position_size=signal.position_size,
                        reasoning=signal.reasoning,
                        timestamp=signal.timestamp,
                        metadata=signal.metadata,
                    )
                    await self.publisher.publish_signal(
                        symbol, hybrid_signal, strategy.name
                    )

            # Update signal statistics
            await self._update_signal_statistics(strategy.name, symbol, signal)

        except Exception as e:
            self.logger.error(f"Error publishing strategy signal: {e}")

    async def _update_signal_statistics(
        self, strategy_name: str, symbol: str, signal: Signal
    ) -> None:
        """Update signal statistics in Redis."""
        try:
            stats_key = f"signal_stats:{strategy_name}:{symbol}"

            # Get current stats
            current_stats = await self.redis.get(stats_key) if self.redis else None
            if current_stats:
                stats = json.loads(current_stats)
                # Ensure all stats have proper default values
                stats["total_signals"] = stats.get("total_signals") or 0
                stats["buy_signals"] = stats.get("buy_signals") or 0
                stats["sell_signals"] = stats.get("sell_signals") or 0
                stats["hold_signals"] = stats.get("hold_signals") or 0
                stats["avg_confidence"] = stats.get("avg_confidence") or 0.0
            else:
                stats = {
                    "total_signals": 0,
                    "buy_signals": 0,
                    "sell_signals": 0,
                    "hold_signals": 0,
                    "avg_confidence": 0.0,
                    "last_signal_time": None,
                }

            # Update stats
            stats["total_signals"] = (stats["total_signals"] or 0) + 1
            stats["last_signal_time"] = int(datetime.now(timezone.utc).timestamp())

            if signal.action == SignalType.BUY:
                stats["buy_signals"] = (stats["buy_signals"] or 0) + 1
            elif signal.action == SignalType.SELL:
                stats["sell_signals"] = (stats["sell_signals"] or 0) + 1
            else:
                stats["hold_signals"] = (stats["hold_signals"] or 0) + 1

            # Update average confidence
            total_signals = stats["total_signals"] or 1
            old_avg = stats["avg_confidence"] or 0.0
            stats["avg_confidence"] = (
                float(old_avg) * (total_signals - 1) + signal.confidence
            ) / total_signals

            # Save updated stats
            if self.redis:
                await self.redis.setex(
                    stats_key, 86400, json.dumps(stats, default=str)
                )  # 24 hour TTL

        except Exception as e:
            self.logger.error(f"Error updating signal statistics: {e}")

    async def _get_historical_data(
        self, symbol: str, lookback_period: int
    ) -> Optional[Any]:
        """
        Get historical data for symbol.

        Args:
            symbol: Trading symbol
            lookback_period: Number of periods to retrieve

        Returns:
            Historical data or None
        """
        try:
            # This would typically query the database or cache
            # For now, return None to indicate data needs to be fetched
            # In a real implementation, this would:
            # 1. Check Redis cache for recent data
            # 2. Query database if not in cache
            # 3. Format as Polars DataFrame

            data_key = f"historical_data:{symbol}:{lookback_period}"
            cached_data = await self.redis.get(data_key) if self.redis else None

            if cached_data:
                # Would deserialize Polars DataFrame here
                self.logger.debug(f"Using cached historical data for {symbol}")
                return None  # Placeholder

            return None  # Placeholder - would fetch from database

        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    async def _get_finviz_data(self, symbol: str) -> Optional[FinVizData]:
        """
        Get FinViz data for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            FinViz data or None
        """
        try:
            finviz_key = f"finviz_data:{symbol}"
            cached_data = await self.redis.get(finviz_key) if self.redis else None

            if cached_data:
                # Would convert to FinVizData model here
                # data = json.loads(cached_data)
                return None  # Placeholder

            return None

        except Exception as e:
            self.logger.error(f"Error getting FinViz data for {symbol}: {e}")
            return None

    async def start_real_time_processing(self, symbols: List[str]) -> None:
        """
        Start real-time signal processing.

        Args:
            symbols: Symbols to process
        """
        try:
            self.logger.info(
                f"Starting real-time processing for {len(symbols)} symbols"
            )

            # Subscribe to price updates
            if self.subscriber and self.data_handler:

                def price_callback(symbol: str, price_data: Dict[str, Any]):
                    if self.data_handler:
                        asyncio.create_task(
                            self.data_handler.process_price_update(symbol, price_data)
                        )

                await self.subscriber.subscribe_to_price_updates(
                    symbols, price_callback
                )

                # Subscribe to screener updates
                def screener_callback(screener_data: Dict[str, Any]):
                    if self.data_handler:
                        asyncio.create_task(
                            self.data_handler.process_screener_update(screener_data)
                        )

                await self.subscriber.subscribe_to_screener_updates(screener_callback)

            # Start heartbeat task
            asyncio.create_task(self._heartbeat_loop())

            # Start cache cleanup task
            asyncio.create_task(self._cache_cleanup_loop())

            # Start listening for messages
            if self.subscriber:
                await self.subscriber.start_listening()

        except Exception as e:
            self.logger.error(f"Error starting real-time processing: {e}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat to indicate system health."""
        try:
            while True:
                await asyncio.sleep(self.config["heartbeat_interval"])

                # Publish heartbeat
                heartbeat = {
                    "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                    "active_strategies": len(self.active_strategies),
                    "total_signals": sum(
                        s["signal_count"] for s in self.active_strategies.values()
                    ),
                    "total_errors": sum(
                        s["error_count"] for s in self.active_strategies.values()
                    ),
                    "status": "healthy",
                }

                if self.redis:
                    await self.redis.publish(
                        "strategy_engine_heartbeat", json.dumps(heartbeat)
                    )

        except Exception as e:
            self.logger.error(f"Heartbeat loop error: {e}")

    async def _cache_cleanup_loop(self) -> None:
        """Clean up expired cache entries periodically."""
        try:
            while True:
                await asyncio.sleep(300)  # Clean up every 5 minutes

                # Clean expired analysis cache
                if self.cache:
                    await self.cache.invalidate_cache("analysis_cache:*")

                # Clean old signal history (keep last 7 days)
                await self._cleanup_signal_history()

                self.logger.debug("Cache cleanup completed")

        except Exception as e:
            self.logger.error(f"Cache cleanup loop error: {e}")

    async def _cleanup_signal_history(self) -> None:
        """Clean up old signal history entries."""
        try:
            # Get all signal history keys
            pattern = "signal_history:*"
            keys: List[str] = await self.redis.keys(pattern) if self.redis else []

            for key in keys:
                # Keep only last 100 signals per symbol
                if self.redis:
                    self.redis.ltrim(key, 0, 99)

        except Exception as e:
            self.logger.error(f"Error cleaning signal history: {e}")

    async def get_strategy_status(
        self, strategy_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get status of active strategies.

        Args:
            strategy_name: Specific strategy name, or None for all

        Returns:
            Strategy status information
        """
        try:
            if strategy_name:
                if strategy_name in self.active_strategies:
                    strategy_info = self.active_strategies[strategy_name]
                    return {
                        "name": strategy_name,
                        "symbols": strategy_info["symbols"],
                        "signal_count": strategy_info["signal_count"],
                        "error_count": strategy_info["error_count"],
                        "status": strategy_info["status"],
                        "last_analysis_times": {
                            symbol: info["timestamp"].isoformat()
                            for symbol, info in strategy_info["last_analysis"].items()
                        },
                    }
                else:
                    return {"error": f"Strategy {strategy_name} not found"}
            else:
                # Return all strategies
                return {
                    "total_strategies": len(self.active_strategies),
                    "strategies": [
                        {
                            "name": name,
                            "symbols": info["symbols"],
                            "signal_count": info["signal_count"],
                            "error_count": info["error_count"],
                            "status": info["status"],
                        }
                        for name, info in self.active_strategies.items()
                    ],
                }

        except Exception as e:
            self.logger.error(f"Error getting strategy status: {e}")
            return {"error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown Redis strategy engine and cleanup resources."""
        try:
            self.logger.info("Shutting down Redis strategy engine")

            # Save strategy states
            for strategy_info in self.active_strategies.values():
                strategy = strategy_info["strategy"]
                if self.cache:
                    await self.cache.cache_strategy_state(strategy)

            # Stop subscriber
            if self.subscriber:
                await self.subscriber.stop_listening()

            # Close Redis connections
            if self.redis_async:
                await self.redis_async.close()

            if self.redis_sync:
                self.redis_sync.close()

            # Clear active strategies
            self.active_strategies.clear()

            self.logger.info("Redis strategy engine shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

    async def get_signal_history(
        self, symbol: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get signal history for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of signals to return

        Returns:
            List of historical signals
        """
        try:
            if not self.redis:
                return []
            history_key = RedisChannels.SIGNAL_HISTORY.format(symbol=symbol)
            try:
                signal_history = await self.redis.lrange(history_key, 0, limit - 1)  # type: ignore
            except (TypeError, AttributeError):
                # Fall back to sync operation if async fails
                signal_history = []

            signals = []
            for signal_json in signal_history:
                try:
                    signal = json.loads(signal_json)
                    signals.append(signal)
                except Exception as e:
                    self.logger.warning(f"Error parsing signal from history: {e}")

            return signals

        except Exception as e:
            self.logger.error(f"Error getting signal history for {symbol}: {e}")
            return []

    async def get_active_tickers(self) -> List[str]:
        """
        Get active tickers from Redis (those currently being tracked by the screener).

        Returns:
            List of active ticker symbols from the screener
        """
        try:
            if not self.redis:
                self.logger.warning("Redis not available for getting active tickers")
                return []

            # Get active tickers from Redis set (same key used by data collector)
            tickers = self.redis.smembers("active_tickers")
            if hasattr(tickers, "__await__"):
                tickers = await tickers
            ticker_list = list(tickers) if tickers else []

            self.logger.debug(
                f"Retrieved {len(ticker_list)} active tickers from Redis: {ticker_list}"
            )
            return ticker_list

        except Exception as e:
            self.logger.error(f"Error getting active tickers from Redis: {e}")
            return []

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            # Redis connection info
            if not self.redis:
                return {}
            redis_info = await self.redis.info()

            # Strategy metrics
            total_signals = sum(
                s["signal_count"] for s in self.active_strategies.values()
            )
            total_errors = sum(
                s["error_count"] for s in self.active_strategies.values()
            )

            # Cache metrics
            cache_keys = await self.redis.dbsize()

            return {
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "redis_info": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory": redis_info.get("used_memory_human", "0B"),
                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                },
                "strategy_metrics": {
                    "active_strategies": len(self.active_strategies),
                    "total_signals_generated": total_signals,
                    "total_errors": total_errors,
                    "error_rate": total_errors / max(total_signals, 1),
                    "cache_keys": cache_keys,
                },
                "config": self.config,
            }

        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}


class RedisSignalConsumer:
    """Consumer for trading signals from Redis channels."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize signal consumer.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis = None
        self.pubsub = None
        self.logger = logging.getLogger("redis_signal_consumer")
        self._callbacks: Dict[str, Any] = {}
        self._running = False

    async def initialize(self) -> bool:
        """Initialize Redis connection."""
        try:
            self.redis = redis.from_url(self.redis_url, decode_responses=True)
            if self.redis:
                await self.redis.ping()
            self.logger.info("Redis signal consumer initialized")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing Redis consumer: {e}")
            return False

    def register_signal_callback(
        self, symbol: str, callback: Callable[[Dict], None]
    ) -> None:
        """
        Register callback for trading signals.

        Args:
            symbol: Trading symbol to listen for
            callback: Callback function
        """
        if symbol not in self._callbacks:
            self._callbacks[symbol] = []
        self._callbacks[symbol].append(callback)

    async def start_consuming(self, symbols: List[str]) -> None:
        """
        Start consuming signals for specified symbols.

        Args:
            symbols: Symbols to consume signals for
        """
        try:
            self._running = True
            if not self.redis:
                return
            self.pubsub = self.redis.pubsub()

            # Subscribe to channels
            channels = [
                RedisChannels.SIGNALS.format(symbol=symbol) for symbol in symbols
            ]
            await self.pubsub.subscribe(*channels)
            await self.pubsub.subscribe("all_signals")  # General monitoring

            self.logger.info(f"Started consuming signals for {len(symbols)} symbols")

            async for message in self.pubsub.listen():
                if not self._running:
                    break

                if message["type"] == "message":
                    try:
                        signal_data = json.loads(message["data"])
                        symbol = signal_data.get("ticker", "")

                        # Call registered callbacks
                        callbacks = self._callbacks.get(symbol, [])
                        for callback in callbacks:
                            try:
                                await callback(signal_data)
                            except Exception as e:
                                self.logger.error(
                                    f"Error in signal callback for {symbol}: {e}"
                                )

                    except Exception as e:
                        self.logger.error(f"Error processing signal message: {e}")

        except Exception as e:
            self.logger.error(f"Error in signal consumer: {e}")
        finally:
            self._running = False
            if hasattr(self, "pubsub") and self.pubsub:
                await self.pubsub.close()

    async def stop_consuming(self) -> None:
        """Stop consuming signals."""
        self._running = False
        if self.redis:
            await self.redis.close()
        self.logger.info("Stopped signal consumer")


class RedisBacktestResultsManager:
    """Manage backtest results in Redis."""

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize backtest results manager.

        Args:
            redis_client: Async Redis client
        """
        self.redis = redis_client
        self.logger = logging.getLogger("redis_backtest_manager")

    async def store_backtest_result(
        self, result_id: str, backtest_result: Dict[str, Any]
    ) -> bool:
        """
        Store backtest result in Redis.

        Args:
            result_id: Unique identifier for the result
            backtest_result: Backtest result data

        Returns:
            True if stored successfully
        """
        try:
            # Store main result
            result_key = f"backtest_result:{result_id}"
            await self.redis.setex(
                result_key,
                86400 * 7,  # 7 days TTL
                json.dumps(backtest_result, default=str),
            )

            # Store in results index
            index_key = "backtest_results_index"
            index_entry = {
                "id": result_id,
                "strategy": backtest_result.get("strategy_info", {}).get("name", ""),
                "symbol": backtest_result.get("strategy_info", {}).get("symbol", ""),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_return": backtest_result.get("performance_metrics", {}).get(
                    "total_return", 0.0
                ),
            }

            self.redis.lpush(index_key, json.dumps(index_entry, default=str))
            self.redis.ltrim(index_key, 0, 999)  # Keep last 1000 results

            # Publish result notification
            self.redis.publish(
                RedisChannels.BACKTEST_RESULTS,
                json.dumps(
                    {"result_id": result_id, "summary": index_entry}, default=str
                ),
            )

            self.logger.info(f"Stored backtest result {result_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing backtest result: {e}")
            return False

    async def get_backtest_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        """
        Get backtest result by ID.

        Args:
            result_id: Result identifier

        Returns:
            Backtest result or None
        """
        try:
            result_key = f"backtest_result:{result_id}"
            cached_data = await self.redis.get(result_key)

            if cached_data:
                return json.loads(cached_data)
            return None

        except Exception as e:
            self.logger.error(f"Error getting backtest result {result_id}: {e}")
            return None

    async def list_backtest_results(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent backtest results.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of backtest result summaries
        """
        try:
            index_key = "backtest_results_index"
            try:
                results_data = await self.redis.lrange(index_key, 0, limit - 1)  # type: ignore
            except (TypeError, AttributeError):
                # Fall back to empty list if async fails
                results_data = []

            results = []
            for result_json in results_data:
                try:
                    result = json.loads(result_json)
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Error parsing backtest result: {e}")

            return results

        except Exception as e:
            self.logger.error(f"Error listing backtest results: {e}")
            return []
