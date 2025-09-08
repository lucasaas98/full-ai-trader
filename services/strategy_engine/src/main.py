"""
Strategy Engine Main Service

This is the main entry point for the strategy engine service that orchestrates
all components including technical analysis, fundamental analysis, hybrid strategies,
backtesting, market regime detection, and Redis integration.
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import polars as pl
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, generate_latest
from pydantic import BaseModel, Field

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

from shared.clients.data_collector_client import DataCollectorClient  # noqa: E402
from shared.models import FinVizData, TimeFrame  # noqa: E402

from .backtesting_engine import (  # noqa: E402
    BacktestConfig,
    BacktestingEngine,
    BacktestMode,
)
from .base_strategy import (  # noqa: E402
    BaseStrategy,
    StrategyConfig,
    StrategyMode,
    TimeFrameMapper,
)
from .fundamental_analysis import (  # noqa: E402
    FundamentalAnalysisEngine,
    FundamentalStrategy,
)
from .hybrid_strategy import (  # noqa: E402
    HybridMode,
    HybridSignal,
    HybridSignalGenerator,
    HybridStrategy,
    HybridStrategyFactory,
)
from .market_regime import (  # noqa: E402
    MarketRegimeDetector,
    RegimeAwareStrategyManager,
)
from .redis_integration import RedisStrategyEngine  # noqa: E402
from .technical_analysis import TechnicalAnalysisEngine, TechnicalStrategy  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("strategy_engine_main")


# Pydantic models for API
class StrategyCreateRequest(BaseModel):
    name: str = Field(..., description="Strategy name")
    strategy_type: str = Field(
        ..., description="Strategy type: technical, fundamental, hybrid"
    )
    mode: str = Field(default="swing_trading", description="Trading mode")
    symbols: List[str] = Field(..., description="Symbols to trade")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Strategy parameters"
    )
    # Phase 3: Enhanced Strategy Configuration Support
    primary_timeframe: Optional[str] = Field(
        None, description="Primary timeframe for the strategy (e.g., '1h', '1d')"
    )
    additional_timeframes: Optional[List[str]] = Field(
        None, description="Additional timeframes for multi-timeframe analysis"
    )
    custom_timeframes: Optional[List[str]] = Field(
        None, description="Custom timeframes list (overrides mode-based defaults)"
    )


class SignalGenerationRequest(BaseModel):
    strategy_name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Trading symbol")
    include_analysis: bool = Field(
        default=False, description="Include detailed analysis"
    )


class BacktestRequest(BaseModel):
    strategy_name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Trading symbol")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    mode: str = Field(default="simple", description="Backtest mode")


class ParameterOptimizationRequest(BaseModel):
    strategy_name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Trading symbol")
    optimization_method: str = Field(
        default="grid_search", description="Optimization method"
    )
    objective: str = Field(default="sharpe_ratio", description="Optimization objective")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")


class EODReportRequest(BaseModel):
    date: Optional[str] = Field(
        None, description="Report date (YYYY-MM-DD), defaults to today"
    )
    include_detailed_signals: bool = Field(
        default=False, description="Include detailed signal analysis"
    )
    include_performance_metrics: bool = Field(
        default=True, description="Include strategy performance metrics"
    )


class StrategyEngineService:
    """Main strategy engine service class."""

    def __init__(self):
        """Initialize strategy engine service."""
        self.logger = logging.getLogger("strategy_engine_service")
        self.logger.debug("Initializing StrategyEngineService instance")

        # Core components
        self.redis_engine: Optional[RedisStrategyEngine] = None
        self.backtesting_engine: Optional[BacktestingEngine] = None
        self.regime_manager: Optional[RegimeAwareStrategyManager] = None

        # Data collector client for market data
        self.data_collector_client: Optional[DataCollectorClient] = None

        # Analysis engines
        self.technical_engine = TechnicalAnalysisEngine()
        self.fundamental_engine = FundamentalAnalysisEngine()

        # Strategy management
        self.active_strategies: Dict[str, BaseStrategy] = {}
        # Signal validation handled within strategies

        # Configuration
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis_password = os.getenv("REDIS_PASSWORD", "")
        redis_url = (
            f"redis://:{redis_password}@{redis_host}:{redis_port}"
            if redis_password
            else f"redis://{redis_host}:{redis_port}"
        )

        self.config = {
            "redis_url": os.getenv("REDIS_URL", redis_url),
            "max_concurrent_strategies": int(
                os.getenv("MAX_CONCURRENT_STRATEGIES", "10")
            ),
            "signal_cache_ttl": int(os.getenv("SIGNAL_CACHE_TTL", "300")),
            "enable_real_time": os.getenv("ENABLE_REAL_TIME", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }

    async def initialize(self) -> bool:
        """Initialize all service components."""
        try:
            self.logger.info("Initializing Strategy Engine Service")
            self.logger.debug(f"Service configuration: {self.config}")

            # Initialize data collector client
            data_collector_url = os.getenv(
                "DATA_COLLECTOR_URL", "http://data-collector:8003"
            )
            self.data_collector_client = DataCollectorClient(
                base_url=data_collector_url
            )
            self.logger.info(f"Initialized data collector client: {data_collector_url}")

            # Initialize Redis integration
            if self.config["enable_real_time"]:
                self.logger.debug(
                    f"Initializing Redis engine with URL: {self.config['redis_url']}"
                )
                self.redis_engine = RedisStrategyEngine(str(self.config["redis_url"]))
                if not await self.redis_engine.initialize():
                    self.logger.error("Failed to initialize Redis engine")
                    return False
                self.logger.debug("Redis engine initialized successfully")

            # Initialize backtesting engine
            self.logger.debug("Initializing backtesting engine")
            backtest_config = BacktestConfig(
                initial_capital=100000.0,
                commission_per_trade=1.0,
                commission_percentage=0.001,
                slippage_percentage=0.0005,
            )
            self.backtesting_engine = BacktestingEngine(backtest_config)
            self.logger.debug("Backtesting engine initialized successfully")

            # Initialize regime manager
            self.logger.debug("Initializing regime manager")
            self.regime_manager = RegimeAwareStrategyManager()
            self.logger.debug("Regime manager initialized successfully")

            # Load any persisted strategies
            self.logger.debug("Loading persisted strategies")
            await self._load_persisted_strategies()

            self.logger.info("Strategy Engine Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
            return False

    async def create_strategy(self, request: StrategyCreateRequest) -> Dict[str, Any]:
        """Create and register a new trading strategy."""
        try:
            self.logger.debug(
                f"Creating strategy: {request.name} of type {request.strategy_type}"
            )
            # Validate strategy type
            if request.strategy_type not in ["technical", "fundamental", "hybrid"]:
                raise ValueError(f"Invalid strategy type: {request.strategy_type}")

            # Parse mode
            try:
                mode = StrategyMode(request.mode)
                self.logger.debug(f"Using trading mode: {mode.value}")
            except ValueError:
                self.logger.error(f"Invalid trading mode: {request.mode}")
                raise ValueError(f"Invalid trading mode: {request.mode}")

            # Validate timeframes if provided
            if request.primary_timeframe:
                available, unavailable = TimeFrameMapper.validate_timeframes(
                    [request.primary_timeframe]
                )
                if unavailable:
                    raise ValueError(
                        f"Invalid primary timeframe: {request.primary_timeframe}"
                    )

            if request.additional_timeframes:
                available, unavailable = TimeFrameMapper.validate_timeframes(
                    request.additional_timeframes
                )
                if unavailable:
                    raise ValueError(f"Invalid additional timeframes: {unavailable}")

            if request.custom_timeframes:
                available, unavailable = TimeFrameMapper.validate_timeframes(
                    request.custom_timeframes
                )
                if unavailable:
                    raise ValueError(f"Invalid custom timeframes: {unavailable}")

            # Create strategy configuration
            config = StrategyConfig(
                name=request.name,
                mode=mode,
                lookback_period=request.parameters.get("lookback_period", 50),
                min_confidence=request.parameters.get("min_confidence", 60.0),
                max_position_size=request.parameters.get("max_position_size", 0.20),
                parameters=request.parameters,
                primary_timeframe=request.primary_timeframe,
                additional_timeframes=request.additional_timeframes,
                custom_timeframes=request.custom_timeframes,
            )

            # Create strategy based on type
            self.logger.debug(
                f"Creating {request.strategy_type} strategy with config: {config.__dict__}"
            )
            if request.strategy_type == "technical":
                strategy = TechnicalStrategy(config)
            elif request.strategy_type == "fundamental":
                strategy = FundamentalStrategy(config)  # type: ignore
            elif request.strategy_type == "hybrid":
                hybrid_mode = HybridMode(
                    request.parameters.get("hybrid_mode", "swing_trading")
                )
                self.logger.debug(f"Using hybrid mode: {hybrid_mode.value}")
                strategy = HybridStrategy(config, hybrid_mode)  # type: ignore
            else:
                raise ValueError(f"Unsupported strategy type: {request.strategy_type}")

            # Initialize strategy
            self.logger.debug(f"Initializing strategy: {request.name}")
            strategy.initialize()

            # Register strategy
            self.logger.debug(f"Registering strategy: {request.name}")
            self.active_strategies[request.name] = strategy

            # Register with Redis engine if available
            if self.redis_engine:
                self.logger.debug(
                    f"Registering strategy {request.name} with Redis for symbols: {request.symbols}"
                )
                await self.redis_engine.register_strategy(strategy, request.symbols)

            self.logger.info(f"Created and registered strategy: {request.name}")

            return {
                "status": "success",
                "strategy_name": request.name,
                "strategy_type": request.strategy_type,
                "symbols": request.symbols,
                "strategy_info": strategy.get_strategy_info(),
            }

        except Exception as e:
            self.logger.error(f"Error creating strategy: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def generate_signal(self, request: SignalGenerationRequest) -> Dict[str, Any]:
        """Generate trading signal for a specific strategy and symbol."""
        try:
            self.logger.debug(
                f"Generating signal for {request.symbol} using strategy {request.strategy_name}"
            )
            # Get strategy
            if request.strategy_name not in self.active_strategies:
                raise ValueError(f"Strategy not found: {request.strategy_name}")

            strategy = self.active_strategies[request.strategy_name]
            self.logger.debug(f"Using strategy type: {strategy.__class__.__name__}")

            # Get market data (this would typically come from database)
            self.logger.debug(
                f"Fetching market data for {request.symbol} with lookback period {strategy.config.lookback_period}"
            )

            # Check if strategy supports multi-timeframe analysis
            required_timeframes = strategy.get_required_data_timeframes()
            if len(required_timeframes) > 1:
                self.logger.debug(
                    f"Strategy requires multiple timeframes: {required_timeframes}"
                )
                multi_tf_data = await self._get_multi_timeframe_data(
                    request.symbol, required_timeframes, strategy.config.lookback_period
                )
                if not multi_tf_data:
                    raise ValueError(
                        f"No multi-timeframe data available for {request.symbol}"
                    )

                # Use primary timeframe as main market_data for backward compatibility
                primary_tf = required_timeframes[0]
                market_data = multi_tf_data.get(primary_tf)
                if market_data is None:
                    # Fall back to any available timeframe
                    market_data = (
                        next(iter(multi_tf_data.values())) if multi_tf_data else None
                    )

                if market_data is None:
                    raise ValueError(
                        f"No primary timeframe data available for {request.symbol}"
                    )

                self.logger.debug(
                    f"Retrieved multi-timeframe data: {list(multi_tf_data.keys())}, primary: {market_data.height} points"
                )
            else:
                market_data = await self._get_market_data(
                    request.symbol, strategy.config.lookback_period, strategy
                )
                if market_data is None:
                    raise ValueError(f"No market data available for {request.symbol}")
                self.logger.debug(
                    f"Retrieved {market_data.height} data points for {request.symbol}"
                )
                multi_tf_data = None

            # Get fundamental data for hybrid strategies
            finviz_data = None
            if isinstance(strategy, HybridStrategy):
                self.logger.debug(
                    f"Fetching FinViz data for hybrid strategy and symbol {request.symbol}"
                )
                finviz_data = await self._get_finviz_data(request.symbol)

            # Generate signal
            self.logger.debug(
                f"Analyzing {request.symbol} with strategy {request.strategy_name}"
            )

            # Pass multi-timeframe data if strategy supports it and we have it
            if isinstance(strategy, HybridStrategy):
                # For hybrid strategies, pass multi-timeframe data if available
                if (
                    multi_tf_data
                    and hasattr(strategy, "analyze_multi_timeframe")
                    and callable(getattr(strategy, "analyze_multi_timeframe"))
                ):
                    signal = await strategy.analyze_multi_timeframe(  # type: ignore
                        request.symbol, multi_tf_data, finviz_data
                    )
                else:
                    signal = await strategy.analyze(
                        request.symbol, market_data, finviz_data
                    )
            else:
                # For regular strategies, pass multi-timeframe data if strategy supports it
                if (
                    multi_tf_data
                    and hasattr(strategy, "analyze_multi_timeframe")
                    and callable(getattr(strategy, "analyze_multi_timeframe"))
                ):
                    signal = await strategy.analyze_multi_timeframe(  # type: ignore
                        request.symbol, multi_tf_data
                    )
                else:
                    signal = await strategy.analyze(request.symbol, market_data)  # type: ignore

            if signal:
                self.logger.debug(
                    f"Generated signal for {request.symbol}: action={signal.action}, confidence={signal.confidence}"
                )
            else:
                self.logger.debug(f"No signal generated for {request.symbol}")

            # Format signal for output
            signal_generator = HybridSignalGenerator()
            # Convert to HybridSignal if needed
            hybrid_signal: Optional[HybridSignal] = None
            if not isinstance(signal, HybridSignal) and signal:
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
            elif isinstance(signal, HybridSignal):
                hybrid_signal = signal

            if hybrid_signal:
                formatted_signal = signal_generator.generate_formatted_signal(
                    request.symbol, hybrid_signal, strategy.config.mode.value
                )
            else:
                formatted_signal = {"error": "No signal generated"}

            # Add analysis details if requested
            if request.include_analysis:
                formatted_signal["analysis_details"] = {
                    "technical_score": getattr(signal, "technical_score", 0.0),
                    "fundamental_score": getattr(signal, "fundamental_score", 0.0),
                    "signal_metadata": (
                        signal.metadata if hasattr(signal, "metadata") else {}
                    ),
                    "strategy_info": strategy.get_strategy_info(),
                }

            # Publish signal if Redis is available
            if (
                self.redis_engine
                and self.redis_engine.publisher
                and signal.confidence >= strategy.config.min_confidence
            ):
                self.logger.debug(
                    f"Publishing signal for {request.symbol} to Redis (confidence: {signal.confidence})"
                )
                await self.redis_engine.publisher.publish_signal(
                    request.symbol, hybrid_signal, strategy.name
                )
            elif signal.confidence < strategy.config.min_confidence:
                self.logger.debug(
                    f"Signal confidence {signal.confidence} below threshold {strategy.config.min_confidence}, not publishing"
                )

            return {
                "status": "success",
                "signal": formatted_signal,
                "generated_at": datetime.now(timezone.utc).isoformat() + "Z",
            }

        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def run_backtest(self, request: BacktestRequest) -> Dict[str, Any]:
        """Run backtest for a strategy."""
        try:
            self.logger.debug(
                f"Starting backtest for strategy {request.strategy_name} on {request.symbol}"
            )
            # Get strategy
            if request.strategy_name not in self.active_strategies:
                raise ValueError(f"Strategy not found: {request.strategy_name}")

            strategy = self.active_strategies[request.strategy_name]
            self.logger.debug(f"Running backtest for strategy: {request.strategy_name}")
            self.logger.debug(
                f"Backtest period: {request.start_date} to {request.end_date}"
            )

            # Parse dates
            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)
            self.logger.debug(f"Optimization period: {start_date} to {end_date}")
            self.logger.debug(f"Parsed dates - start: {start_date}, end: {end_date}")

            # Get historical data
            self.logger.debug(f"Fetching historical data for {request.symbol}")
            historical_data = await self._get_historical_data(
                request.symbol, start_date, end_date
            )
            if historical_data is None:
                raise ValueError(f"No historical data available for {request.symbol}")
            self.logger.debug(
                f"Retrieved historical data with {historical_data.height} points"
            )

            # Get FinViz data for hybrid strategies
            finviz_data = None
            if isinstance(strategy, HybridStrategy):
                self.logger.debug("Fetching FinViz data for hybrid strategy backtest")
                finviz_data = await self._get_finviz_data(request.symbol)

            # Parse backtest mode
            try:
                mode = BacktestMode(request.mode)
                self.logger.debug(f"Using backtest mode: {mode.value}")
            except ValueError:
                mode = BacktestMode.SIMPLE
                self.logger.debug(f"Invalid mode {request.mode}, defaulting to SIMPLE")

            # Run backtest
            if self.backtesting_engine:
                self.logger.debug("Running backtest with backtesting engine")
                result = await self.backtesting_engine.backtest_strategy(
                    strategy,
                    request.symbol,
                    historical_data,
                    start_date,
                    end_date,
                    finviz_data,
                    mode,
                )
                self.logger.debug("Backtest execution completed")

                # Generate report
                self.logger.debug("Generating backtest report")
                report = self.backtesting_engine.generate_backtest_report(result)
            else:
                raise HTTPException(
                    status_code=500, detail="Backtesting engine not available"
                )

            # Store result in Redis if available
            result_id = f"{request.strategy_name}_{request.symbol}_{int(datetime.now(timezone.utc).timestamp())}"
            if self.redis_engine and self.redis_engine.redis:
                self.logger.debug(
                    f"Storing backtest result in Redis with ID: {result_id}"
                )
                await self.redis_engine.redis.setex(
                    f"backtest_result:{result_id}",
                    86400 * 7,  # 7 days
                    json.dumps(report, default=str),
                )
            return {
                "status": "success",
                "backtest_id": result_id if "result_id" in locals() else None,
                "report": report,
            }

        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def optimize_parameters(
        self, request: ParameterOptimizationRequest
    ) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        try:
            self.logger.debug(
                f"Starting parameter optimization for strategy {request.strategy_name}"
            )
            # Get strategy
            if request.strategy_name not in self.active_strategies:
                raise ValueError(f"Strategy not found: {request.strategy_name}")

            strategy = self.active_strategies[request.strategy_name]

            # Parse dates
            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)

            # Get historical data
            historical_data = await self._get_historical_data(
                request.symbol, start_date, end_date
            )
            if historical_data is None:
                raise ValueError(f"No historical data available for {request.symbol}")

            # Optimization configuration
            optimization_config = {
                "method": request.optimization_method,
                "objective": request.objective,
                "max_combinations": 50,
            }

            # Get FinViz data for hybrid strategies
            finviz_data = None
            if isinstance(strategy, HybridStrategy):
                finviz_data = await self._get_finviz_data(request.symbol)

            # Run optimization
            if self.backtesting_engine:
                self.logger.debug("Running parameter optimization")
                optimization_result = (
                    await self.backtesting_engine.optimize_strategy_parameters(
                        strategy,
                        request.symbol,
                        historical_data,
                        optimization_config,
                        finviz_data,
                    )
                )
                self.logger.debug("Parameter optimization completed")
            else:
                raise HTTPException(
                    status_code=500, detail="Backtesting engine not available"
                )

            # Update strategy with best parameters if found
            best_params = optimization_result.get("best_parameters", {})
            if best_params:
                self.logger.debug(
                    f"Updating strategy with best parameters: {best_params}"
                )
                strategy.update_config(best_params)
                self.logger.info(
                    f"Updated {request.strategy_name} with optimized parameters"
                )
            else:
                self.logger.debug("No parameter improvements found during optimization")

            return {
                "status": "success",
                "optimization_result": optimization_result,
                "strategy_updated": bool(best_params),
            }

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def get_market_regime(self, symbol: str) -> Dict[str, Any]:
        """Get current market regime for a symbol."""
        try:
            self.logger.debug(f"Analyzing market regime for {symbol}")
            # Get recent market data
            market_data = await self._get_market_data(symbol, 100)
            if market_data is None:
                raise ValueError(f"No market data available for {symbol}")
            self.logger.debug(
                f"Retrieved {market_data.height} data points for regime analysis"
            )

            # Detect regime
            regime_detector = MarketRegimeDetector()
            regime_state = regime_detector.detect_regime(market_data)
            self.logger.debug(
                f"Detected regime for {symbol}: {regime_state.primary_regime.value} (confidence: {regime_state.confidence})"
            )

            # Publish regime update if Redis is available
            if self.redis_engine and self.redis_engine.publisher:
                self.logger.debug(f"Publishing regime update for {symbol} to Redis")
                await self.redis_engine.publisher.publish_regime_update(
                    symbol, regime_state.__dict__
                )

            return {
                "status": "success",
                "symbol": symbol,
                "regime_state": {
                    "primary_regime": regime_state.primary_regime.value,
                    "volatility_regime": regime_state.volatility_regime.value,
                    "confidence": regime_state.confidence,
                    "trend_strength": regime_state.trend_strength,
                    "favorable_for_trading": regime_state.favorable_for_trading,
                    "regime_duration": regime_state.regime_duration,
                    "position_size_multiplier": regime_state.recommended_position_size,
                },
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            }

        except Exception as e:
            self.logger.error(f"Error getting market regime: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def _get_market_data(
        self,
        symbol: str,
        periods: int,
        strategy: Optional[Any] = None,
        timeframes: Optional[List[str]] = None,
    ) -> Optional[pl.DataFrame]:
        """Get market data for analysis using the data collector client.

        Args:
            symbol: Trading symbol
            periods: Number of periods to load
            strategy: Optional strategy instance to get timeframes from
            timeframes: Optional list of data timeframes to try (overrides strategy)
        """
        try:
            if not self.data_collector_client:
                self.logger.error("Data collector client not initialized")
                return None

            self.logger.debug(
                f"Loading market data for {symbol}, requesting {periods} periods"
            )

            # Determine timeframes to try based on strategy or fallback to defaults
            if timeframes:
                # Use provided timeframes directly (assumed to be data collector format)
                timeframes_to_try = timeframes
                self.logger.debug(f"Using provided timeframes: {timeframes_to_try}")
            elif strategy and hasattr(strategy, "get_required_data_timeframes"):
                # Get timeframes from strategy (already in data collector format)
                try:
                    timeframes_to_try = strategy.get_required_data_timeframes()
                    self.logger.debug(f"Using strategy timeframes: {timeframes_to_try}")
                    if not timeframes_to_try:
                        # Fallback if strategy returns empty list
                        timeframes_to_try = [
                            TimeFrame.ONE_DAY.value,
                            TimeFrame.ONE_HOUR.value,
                            TimeFrame.FIFTEEN_MINUTES.value,
                        ]
                        self.logger.debug(
                            "Strategy returned empty timeframes, using defaults"
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Error getting strategy timeframes: {e}, using defaults"
                    )
                    timeframes_to_try = [
                        TimeFrame.ONE_DAY.value,
                        TimeFrame.ONE_HOUR.value,
                        TimeFrame.FIFTEEN_MINUTES.value,
                    ]
            else:
                # Fallback to default timeframes (prefer daily, then hourly, then minutes)
                timeframes_to_try = [
                    TimeFrame.ONE_DAY.value,
                    TimeFrame.ONE_HOUR.value,
                    TimeFrame.FIFTEEN_MINUTES.value,
                ]
                self.logger.debug("Using default timeframes")

            # Try each timeframe until we find data
            for tf_str in timeframes_to_try:
                try:
                    timeframe_enum = TimeFrame(tf_str)
                    self.logger.debug(
                        f"Attempting to load data for timeframe: {tf_str}"
                    )

                    # Use data collector client to load market data
                    df = await self.data_collector_client.load_market_data(
                        ticker=symbol, timeframe=timeframe_enum, limit=periods
                    )

                    if df is not None and not df.is_empty():
                        self.logger.info(
                            f"Loaded {df.height} data points for {symbol} using timeframe {tf_str}"
                        )
                        self.logger.debug(
                            f"Data date range: {str(df['timestamp'].min())} to {str(df['timestamp'].max())}"
                        )
                        return df
                    else:
                        self.logger.debug(
                            f"No data available for {symbol} in timeframe {tf_str}"
                        )

                except ValueError as e:
                    self.logger.warning(f"Invalid timeframe {tf_str}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(
                        f"Error loading data for {symbol} in timeframe {tf_str}: {e}"
                    )
                    continue

            self.logger.warning(f"No market data found for {symbol} in any timeframe")
            return None

        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None

    async def _get_multi_timeframe_data(
        self, symbol: str, timeframes: List[str], periods: int = 100
    ) -> Dict[str, pl.DataFrame]:
        """
        Load data from multiple timeframes for a symbol using data collector client.

        Args:
            symbol: Trading symbol
            timeframes: List of data timeframes to load
            periods: Number of periods to load for each timeframe

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        try:
            if not self.data_collector_client:
                self.logger.error("Data collector client not initialized")
                return {}

            self.logger.debug(
                f"Loading multi-timeframe data for {symbol}: {timeframes}"
            )

            results = {}

            for tf_str in timeframes:
                try:
                    timeframe_enum = TimeFrame(tf_str)
                    self.logger.debug(f"Loading data for timeframe: {tf_str}")

                    # Use data collector client to load market data
                    df = await self.data_collector_client.load_market_data(
                        ticker=symbol, timeframe=timeframe_enum, limit=periods
                    )

                    if df is not None and not df.is_empty():
                        results[tf_str] = df
                        self.logger.debug(f"Loaded {df.height} records for {tf_str}")
                    else:
                        self.logger.debug(
                            f"No data available for {symbol} in timeframe {tf_str}"
                        )

                except ValueError as e:
                    self.logger.warning(f"Invalid timeframe {tf_str}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(
                        f"Error loading data for {symbol} in timeframe {tf_str}: {e}"
                    )
                    continue

            self.logger.info(f"Loaded data for {len(results)} timeframes for {symbol}")
            return results

        except Exception as e:
            self.logger.error(f"Error loading multi-timeframe data for {symbol}: {e}")
            return {}

    async def _align_multi_timeframe_data(
        self, multi_tf_data: Dict[str, pl.DataFrame]
    ) -> Dict[str, pl.DataFrame]:
        """
        Align timestamps across multiple timeframes for consistent analysis.

        Args:
            multi_tf_data: Dictionary of timeframe -> DataFrame

        Returns:
            Aligned multi-timeframe data
        """
        try:
            if not multi_tf_data:
                return {}

            # Find common time range across all timeframes
            min_timestamp = None
            max_timestamp = None

            for tf, df in multi_tf_data.items():
                if df.is_empty():
                    continue

                tf_min = df["timestamp"].min()
                tf_max = df["timestamp"].max()

                if min_timestamp is None:
                    min_timestamp = tf_min
                elif (
                    tf_min is not None
                    and min_timestamp is not None
                    and tf_min > min_timestamp  # type: ignore
                ):
                    min_timestamp = tf_min

                if max_timestamp is None:
                    max_timestamp = tf_max
                elif (
                    tf_max is not None
                    and max_timestamp is not None
                    and tf_max < max_timestamp  # type: ignore
                ):
                    max_timestamp = tf_max

            if min_timestamp is None or max_timestamp is None:
                return multi_tf_data

            # Filter all timeframes to common range
            aligned_data = {}
            for tf, df in multi_tf_data.items():
                if not df.is_empty():
                    aligned_df = df.filter(
                        (pl.col("timestamp") >= min_timestamp)
                        & (pl.col("timestamp") <= max_timestamp)
                    )
                    if not aligned_df.is_empty():
                        aligned_data[tf] = aligned_df

            self.logger.debug(
                f"Aligned {len(aligned_data)} timeframes to range {str(min_timestamp)} - {str(max_timestamp)}"
            )
            return aligned_data

        except Exception as e:
            self.logger.error(f"Error aligning multi-timeframe data: {e}")
            return multi_tf_data

    async def _get_historical_data(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> Optional[pl.DataFrame]:
        """Get historical data for backtesting (placeholder)."""
        try:
            self.logger.debug(
                f"Attempting to retrieve historical data for {symbol} from {start_date} to {end_date}"
            )
            # Placeholder - would query database with date range
            self.logger.warning(
                f"Historical data retrieval not implemented for {symbol}"
            )
            return None

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None

    async def _get_finviz_data(self, symbol: str) -> Optional[FinVizData]:
        """Get FinViz fundamental data (placeholder)."""
        try:
            self.logger.debug(f"Attempting to retrieve FinViz data for {symbol}")
            # Placeholder - would query database or FinViz API
            self.logger.warning(f"FinViz data retrieval not implemented for {symbol}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting FinViz data: {e}")
            return None

    async def _load_persisted_strategies(self) -> None:
        """Load any persisted strategies from previous runs."""
        try:
            self.logger.debug("Loading persisted strategies")
            if not self.redis_engine:
                self.logger.debug(
                    "No Redis engine available, creating default strategies only"
                )
                await self._create_default_strategies()
                return

            # This would load saved strategy configurations
            # For now, create some default strategies
            self.logger.debug("Creating default strategies")
            await self._create_default_strategies()

        except Exception as e:
            self.logger.error(f"Error loading persisted strategies: {e}")

    async def _create_default_strategies(self) -> None:
        """Create default strategies for demonstration."""
        try:
            self.logger.debug("Creating default hybrid strategies")
            # Create default hybrid strategies
            day_trader = HybridStrategyFactory.create_day_trading_strategy(
                "default_day_trader"
            )
            swing_trader = HybridStrategyFactory.create_swing_trading_strategy(
                "default_swing_trader"
            )
            position_trader = HybridStrategyFactory.create_position_trading_strategy(
                "default_position_trader"
            )
            self.logger.debug("Default strategies created successfully")

            # Initialize strategies
            strategies = [day_trader, swing_trader, position_trader]
            for strategy in strategies:
                strategy.initialize()
                self.active_strategies[strategy.name] = strategy

                # Register with Redis if available
                if self.redis_engine:
                    default_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"]
                    await self.redis_engine.register_strategy(strategy, default_symbols)

            self.logger.info(f"Created {len(strategies)} default strategies")

        except Exception as e:
            self.logger.error(f"Error creating default strategies: {e}")

    async def generate_eod_report(
        self, request: Optional[EODReportRequest] = None
    ) -> Dict[str, Any]:
        """Generate End-of-Day trading report."""
        try:
            self.logger.debug("Starting EOD report generation")
            if request is None:
                request = EODReportRequest(date=None)
                self.logger.debug("Using default EOD report request")

            # Parse report date
            if request.date:
                report_date = datetime.fromisoformat(request.date).replace(
                    tzinfo=timezone.utc
                )
                self.logger.debug(f"Using requested report date: {request.date}")
            else:
                report_date = datetime.now(timezone.utc)
                self.logger.debug(
                    f"Using current date for report: {report_date.date()}"
                )

            # start_of_day = report_date.replace(hour=0, minute=0, second=0, microsecond=0)  # Not used in current implementation

            self.logger.info(f"Generating EOD report for {report_date.date()}")

            # Collect strategy performance data
            strategy_performance = {}
            total_signals_generated = 0
            total_high_confidence_signals = 0

            for strategy_name, strategy in self.active_strategies.items():
                try:
                    self.logger.debug(
                        f"Collecting performance data for strategy: {strategy_name}"
                    )
                    # Get strategy info
                    strategy_info = strategy.get_strategy_info()

                    # Count signals generated today (would need Redis to get actual counts)
                    signals_today = 0
                    high_confidence_signals = 0

                    # Try to get signal metrics from Redis if available
                    if self.redis_engine and self.redis_engine.redis:
                        try:
                            self.logger.debug(
                                f"Retrieving signal metrics from Redis for {strategy_name}"
                            )
                            # Get signal count for today
                            signal_key = f"signals:{strategy_name}:{report_date.strftime('%Y-%m-%d')}"
                            signals_today = (
                                await self.redis_engine.redis.get(signal_key) or 0
                            )
                            signals_today = int(signals_today)
                            self.logger.debug(
                                f"Found {signals_today} signals for {strategy_name}"
                            )

                            # Get high confidence signal count
                            hc_signal_key = f"high_confidence_signals:{strategy_name}:{report_date.strftime('%Y-%m-%d')}"
                            high_confidence_signals = (
                                await self.redis_engine.redis.get(hc_signal_key) or 0
                            )
                            high_confidence_signals = int(high_confidence_signals)
                            self.logger.debug(
                                f"Found {high_confidence_signals} high confidence signals for {strategy_name}"
                            )
                        except Exception as redis_error:
                            self.logger.warning(
                                f"Could not retrieve signal metrics from Redis: {redis_error}"
                            )

                    strategy_performance[strategy_name] = {
                        "type": strategy.__class__.__name__,
                        "mode": strategy_info.get("mode", "unknown"),
                        "signals_generated": signals_today,
                        "high_confidence_signals": high_confidence_signals,
                        "min_confidence_threshold": strategy.config.min_confidence,
                        "max_position_size": strategy.config.max_position_size,
                        "lookback_period": strategy.config.lookback_period,
                        "active": True,
                    }

                    total_signals_generated += signals_today
                    total_high_confidence_signals += high_confidence_signals

                except Exception as strategy_error:
                    self.logger.error(
                        f"Error collecting performance data for {strategy_name}: {strategy_error}"
                    )
                    strategy_performance[strategy_name] = {
                        "type": strategy.__class__.__name__,
                        "error": str(strategy_error),
                        "active": False,
                    }

            # Collect market analysis summary
            self.logger.debug("Starting market analysis summary collection")
            symbols_analyzed = []
            technical_analysis_summary = {}

            # Get list of commonly tracked symbols
            tracked_symbols = set()
            for strategy in self.active_strategies.values():
                if hasattr(strategy, "symbols") and getattr(strategy, "symbols", None):
                    tracked_symbols.update(getattr(strategy, "symbols"))

            # Fallback to default symbols if none found
            if not tracked_symbols:
                tracked_symbols = {"AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"}
                self.logger.debug("Using default symbol set for analysis")

            # Perform analysis on a subset of symbols for the report
            analysis_symbols = list(tracked_symbols)[
                :10
            ]  # Limit to 10 symbols for performance
            self.logger.debug(
                f"Analyzing {len(analysis_symbols)} symbols for report: {analysis_symbols}"
            )

            for symbol in analysis_symbols:
                try:
                    self.logger.debug(f"Performing technical analysis for {symbol}")
                    market_data = await self._get_market_data(symbol, 50)
                    if market_data is not None and not market_data.is_empty():
                        # Perform quick technical analysis
                        self.logger.debug(
                            f"Running technical analysis on {market_data.height} data points for {symbol}"
                        )
                        tech_analysis = self.technical_engine.full_analysis(
                            symbol, market_data
                        )
                        if tech_analysis:
                            symbols_analyzed.append(symbol)
                            technical_analysis_summary[symbol] = {
                                "technical_score": tech_analysis.get(
                                    "technical_score", 50.0
                                ),
                                "trend": self._determine_trend(
                                    tech_analysis.get("indicators", {})
                                ),
                                "volatility": self._calculate_volatility(market_data),
                                "data_points": tech_analysis.get("data_points", 0),
                            }
                except Exception as analysis_error:
                    self.logger.warning(
                        f"Analysis failed for {symbol}: {analysis_error}"
                    )
                    self.logger.debug(f"Skipping {symbol} due to analysis error")
                    continue

            # Generate summary statistics
            self.logger.debug("Calculating summary statistics")
            avg_technical_score = 0.0
            if technical_analysis_summary:
                avg_technical_score = sum(
                    data["technical_score"]
                    for data in technical_analysis_summary.values()
                ) / len(technical_analysis_summary)
                self.logger.debug(f"Average technical score: {avg_technical_score}")

            # Create EOD report
            eod_report: Dict[str, Any] = {
                "report_type": "end_of_day",
                "date": report_date.strftime("%Y-%m-%d"),
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "summary": {
                    "active_strategies": len(self.active_strategies),
                    "total_signals_generated": total_signals_generated,
                    "high_confidence_signals": total_high_confidence_signals,
                    "symbols_analyzed": len(symbols_analyzed),
                    "average_technical_score": round(avg_technical_score, 2),
                    "redis_available": self.redis_engine is not None,
                    "backtesting_available": self.backtesting_engine is not None,
                },
                "strategy_performance": strategy_performance,
                "market_analysis": {
                    "symbols_tracked": list(tracked_symbols),
                    "symbols_analyzed": symbols_analyzed,
                    "technical_summary": technical_analysis_summary,
                },
                "system_health": {
                    "strategy_engine_status": "healthy",
                    "redis_connection": (
                        "connected" if self.redis_engine else "disconnected"
                    ),
                    "components_active": {
                        "technical_engine": self.technical_engine is not None,
                        "fundamental_engine": self.fundamental_engine is not None,
                        "backtesting_engine": self.backtesting_engine is not None,
                        "regime_manager": self.regime_manager is not None,
                    },
                },
            }

            # Add detailed signals if requested
            if request.include_detailed_signals:
                eod_report["detailed_signals"] = (
                    await self._get_detailed_signals_for_date(report_date)
                )

            # Store report in Redis if available
            if self.redis_engine and self.redis_engine.redis:
                try:
                    report_key = f"eod_report:{report_date.strftime('%Y-%m-%d')}"
                    self.logger.debug(
                        f"Storing EOD report in Redis with key: {report_key}"
                    )
                    await self.redis_engine.redis.setex(
                        report_key,
                        86400 * 7,  # Keep for 7 days
                        json.dumps(eod_report, default=str),
                    )
                    self.logger.info(
                        f"EOD report stored in Redis with key: {report_key}"
                    )
                except Exception as redis_error:
                    self.logger.warning(
                        f"Could not store EOD report in Redis: {redis_error}"
                    )

            self.logger.info(
                f"EOD report generated successfully for {report_date.date()}"
            )
            return eod_report

        except Exception as e:
            self.logger.error(f"Error generating EOD report: {e}")
            raise HTTPException(
                status_code=500, detail=f"EOD report generation failed: {str(e)}"
            )

    def _determine_trend(self, indicators: Dict[str, Any]) -> str:
        """Determine market trend from technical indicators."""
        try:
            sma_20 = indicators.get("sma_20")
            sma_50 = indicators.get("sma_50")
            rsi = indicators.get("rsi_14")

            if sma_20 and sma_50:
                if sma_20 > sma_50:
                    if rsi and rsi > 60:
                        return "strong_uptrend"
                    return "uptrend"
                elif sma_20 < sma_50:
                    if rsi and rsi < 40:
                        return "strong_downtrend"
                    return "downtrend"

            return "sideways"
        except Exception:
            return "unknown"

    def _calculate_volatility(self, market_data: pl.DataFrame) -> float:
        """Calculate simple volatility measure."""
        try:
            if market_data.height < 2:
                return 0.0

            # Calculate daily returns
            closes = market_data["close"].to_list()
            returns = []
            for i in range(1, len(closes)):
                returns.append((closes[i] - closes[i - 1]) / closes[i - 1])

            if not returns:
                return 0.0

            # Calculate standard deviation
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            volatility = (variance**0.5) * (252**0.5)  # Annualized

            return round(volatility, 4)
        except Exception:
            return 0.0

    async def _get_detailed_signals_for_date(
        self, report_date: datetime
    ) -> List[Dict[str, Any]]:
        """Get detailed signal information for a specific date."""
        detailed_signals: List[Dict[str, Any]] = []

        if not self.redis_engine or not self.redis_engine.redis:
            return detailed_signals

        try:
            # Search for signals from the specified date
            date_key = report_date.strftime("%Y-%m-%d")
            # Search pattern for signals from the specified date
            # pattern = f"signal:*:{date_key}:*"

            # This would require implementing signal storage with date keys
            # For now, return empty list with a note
            detailed_signals.append(
                {
                    "note": "Detailed signal tracking requires Redis signal storage implementation",
                    "date": date_key,
                }
            )

        except Exception as e:
            self.logger.error(f"Error retrieving detailed signals: {e}")

        return detailed_signals

    async def start_real_time_processing(self) -> None:
        """Start real-time signal processing."""
        try:
            self.logger.debug("Attempting to start real-time processing")
            if not self.redis_engine:
                self.logger.warning(
                    "Redis engine not available - real-time processing disabled"
                )
                return

            # Get symbols from screener via Redis
            all_symbols = set()

            # First try to get active tickers from the screener
            if self.redis_engine:
                screener_tickers = await self.redis_engine.get_active_tickers()
                if screener_tickers:
                    all_symbols.update(screener_tickers)
                    self.logger.info(
                        f"Using {len(screener_tickers)} tickers from screener: {screener_tickers[:10]}{'...' if len(screener_tickers) > 10 else ''}"
                    )
                else:
                    self.logger.warning(
                        "No active tickers found in screener, using fallback symbols"
                    )

            # Fallback to default symbols if screener has no tickers
            if not all_symbols:
                fallback_symbols = [
                    "AAPL",
                    "MSFT",
                    "GOOGL",
                    "TSLA",
                    "SPY",
                    "QQQ",
                    "IWM",
                    "XLF",
                    "XLK",
                    "NVDA",
                ]
                all_symbols.update(fallback_symbols)
                self.logger.info(f"Using fallback symbols: {fallback_symbols}")

            # Also include any symbols from active strategies
            for strategy_info in self.active_strategies.values():
                if hasattr(strategy_info, "symbols") and getattr(
                    strategy_info, "symbols", None
                ):
                    all_symbols.update(getattr(strategy_info, "symbols"))

            self.logger.debug(
                f"Starting real-time processing for {len(all_symbols)} symbols: {list(all_symbols)[:10]}{'...' if len(all_symbols) > 10 else ''}"
            )
            # Start real-time processing
            await self.redis_engine.start_real_time_processing(list(all_symbols))

            self.logger.info("Real-time processing started")

        except Exception as e:
            self.logger.error(f"Error starting real-time processing: {e}")

    async def refresh_screener_tickers(self) -> List[str]:
        """
        Refresh and get the latest tickers from the screener.

        Returns:
            List of current active tickers from screener
        """
        try:
            if not self.redis_engine:
                self.logger.warning("Redis engine not available for ticker refresh")
                return []

            screener_tickers = await self.redis_engine.get_active_tickers()
            if screener_tickers:
                self.logger.info(
                    f"Refreshed {len(screener_tickers)} tickers from screener"
                )
                return screener_tickers
            else:
                self.logger.warning(
                    "No active tickers found in screener during refresh"
                )
                return []

        except Exception as e:
            self.logger.error(f"Error refreshing screener tickers: {e}")
            return []

    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        try:
            self.logger.info("Shutting down Strategy Engine Service")
            self.logger.debug("Starting graceful shutdown sequence")

            # Shutdown Redis engine
            if self.redis_engine:
                self.logger.debug("Shutting down Redis engine")
                await self.redis_engine.shutdown()

            # Shutdown data collector client
            if self.data_collector_client:
                self.logger.debug("Shutting down data collector client")
                await self.data_collector_client.close()

            # Save strategy states
            for strategy in self.active_strategies.values():
                if hasattr(strategy, "save_state"):
                    self.logger.debug(f"Saving state for strategy: {strategy.name}")
                    strategy.save_state()
                    # Would save to persistent storage here

            self.logger.info("Strategy Engine Service shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Global service instance
service = StrategyEngineService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.debug("Starting FastAPI application lifespan")
    await service.initialize()
    if service.config["enable_real_time"]:
        logger.debug("Starting real-time processing task")
        asyncio.create_task(service.start_real_time_processing())

    logger.debug("FastAPI application startup complete")
    yield

    # Shutdown
    logger.debug("Starting FastAPI application shutdown")
    await service.shutdown()
    logger.debug("FastAPI application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Strategy Engine Service",
    description="Advanced trading strategy engine with technical and fundamental analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics (initialized lazily to avoid registration conflicts)
strategies_count_gauge = None
signals_generated_counter = None
backtests_run_counter = None
service_health_gauge = None


def _initialize_metrics():
    """Initialize Prometheus metrics if not already done."""
    global strategies_count_gauge, signals_generated_counter, backtests_run_counter, service_health_gauge

    if strategies_count_gauge is None:
        strategies_count_gauge = Gauge(
            "strategy_engine_strategies_total", "Number of active strategies"
        )
        signals_generated_counter = Counter(
            "strategy_engine_signals_total",
            "Total signals generated",
            ["strategy_type"],
        )
        backtests_run_counter = Counter(
            "strategy_engine_backtests_total", "Total backtests run"
        )
        service_health_gauge = Gauge(
            "strategy_engine_service_health",
            "Health status of components",
            ["component"],
        )


# API Endpoints


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check endpoint called")
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "active_strategies": len(service.active_strategies),
        "redis_available": service.redis_engine is not None,
    }


# Phase 5: Timeframe validation and management endpoints
@app.get("/timeframes/available")
async def get_available_timeframes():
    """Get all available timeframes for strategies and data loading."""
    return {
        "status": "success",
        "strategy_timeframes": TimeFrameMapper.get_available_strategy_timeframes(),
        "data_timeframes": TimeFrameMapper.get_available_data_timeframes(),
        "mapping": {
            tf: (
                TimeFrameMapper.strategy_to_data([tf])[0]
                if TimeFrameMapper.strategy_to_data([tf])
                else None
            )
            for tf in TimeFrameMapper.get_available_strategy_timeframes()
        },
    }


@app.post("/timeframes/validate")
async def validate_timeframes(timeframes: List[str]):
    """Validate a list of strategy timeframes."""
    try:
        available, unavailable = TimeFrameMapper.validate_timeframes(timeframes)
        data_timeframes = TimeFrameMapper.strategy_to_data(available)

        return {
            "status": "success",
            "requested": timeframes,
            "available": available,
            "unavailable": unavailable,
            "data_timeframes": data_timeframes,
            "all_valid": len(unavailable) == 0,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/strategies/{strategy_name}/timeframes")
async def get_strategy_timeframes(strategy_name: str):
    """Get timeframe information for a specific strategy."""
    try:
        if strategy_name not in service.active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy = service.active_strategies[strategy_name]
        availability_info = strategy.validate_timeframe_availability()

        return {
            "status": "success",
            "strategy_name": strategy_name,
            "timeframe_info": availability_info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/timeframes/check/{symbol}")
async def check_symbol_timeframes(symbol: str):
    """Check available timeframes for a specific symbol using data collector client."""
    try:
        if not service.data_collector_client:
            raise HTTPException(
                status_code=500, detail="Data collector client not available"
            )

        available_timeframes = []

        # Test each timeframe by attempting to load a small amount of data
        for tf_str in TimeFrameMapper.get_available_data_timeframes():
            try:
                timeframe_enum = TimeFrame(tf_str)
                df = await service.data_collector_client.load_market_data(
                    ticker=symbol, timeframe=timeframe_enum, limit=1
                )

                if df is not None and not df.is_empty():
                    available_timeframes.append(tf_str)

            except Exception as e:
                logger.debug(f"Timeframe {tf_str} not available for {symbol}: {e}")
                continue

        strategy_timeframes = TimeFrameMapper.data_to_strategy(available_timeframes)

        return {
            "status": "success",
            "symbol": symbol,
            "available_data_timeframes": available_timeframes,
            "available_strategy_timeframes": strategy_timeframes,
            "data_available": len(available_timeframes) > 0,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/strategies")
async def list_strategies():
    """List all active strategies."""
    try:
        logger.debug("Listing all active strategies")
        strategies = []
        for name, strategy in service.active_strategies.items():
            logger.debug(f"Including strategy in list: {name}")
            strategies.append(
                {
                    "name": name,
                    "type": strategy.__class__.__name__,
                    "mode": strategy.config.mode.value,
                    "info": strategy.get_strategy_info(),
                }
            )

        logger.debug(f"Returning {len(strategies)} strategies")
        return {
            "status": "success",
            "strategies": strategies,
            "total_count": len(strategies),
        }

    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategies")
async def create_strategy(request: StrategyCreateRequest):
    """Create a new trading strategy."""
    return await service.create_strategy(request)


@app.post("/signals/generate")
async def generate_signal(request: SignalGenerationRequest):
    """Generate trading signal for a strategy."""
    return await service.generate_signal(request)


@app.post("/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run strategy backtest."""
    return await service.run_backtest(request)


@app.post("/optimize")
async def optimize_parameters(request: ParameterOptimizationRequest):
    """Optimize strategy parameters."""
    return await service.optimize_parameters(request)


@app.post("/analyze")
async def analyze_market_data():
    """Analyze market data and calculate technical indicators for all tracked symbols."""
    try:
        logger.debug("Starting market data analysis endpoint")
        # Get tracked symbols from active strategies or default list
        symbols = set()

        # Collect symbols from active strategies
        for strategy in service.active_strategies.values():
            if hasattr(strategy, "symbols") and getattr(strategy, "symbols", None):
                symbols.update(getattr(strategy, "symbols"))

        # Fallback to common symbols if no strategies are active
        if not symbols:
            symbols = {
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "JPM",
                "BAC",
                "GS",
                "MS",
                "WFC",
                "SPY",
                "QQQ",
                "IWM",
                "XLK",
                "XLF",
                "HD",
                "WMT",
                "KO",
                "PEP",
                "JNJ",
                "PFE",
                "UNH",
                "TMO",
                "PG",
                "ABBV",
            }

        analysis_results = []
        indicators_to_save = []

        for symbol in symbols:
            try:
                logger.debug(f"Analyzing market data for symbol: {symbol}")
                # Get market data for the symbol
                market_data = await service._get_market_data(symbol, periods=200)

                if market_data is None or market_data.is_empty():
                    logger.warning(f"No market data available for {symbol}")
                    continue

                # Ensure we have minimum required columns for technical analysis
                required_columns = ["open", "high", "low", "close", "volume"]
                if not all(col in market_data.columns for col in required_columns):
                    logger.warning(f"Missing required columns for {symbol}")
                    continue

                # Perform technical analysis with error handling
                try:
                    tech_analysis = service.technical_engine.full_analysis(
                        symbol, market_data
                    )
                except Exception as analysis_error:
                    logger.error(
                        f"Technical analysis failed for {symbol}: {analysis_error}"
                    )
                    continue

                if tech_analysis and "indicators" in tech_analysis:
                    # Extract current indicators for saving
                    current_indicators = tech_analysis["indicators"]

                    # Create TechnicalIndicators object for storage
                    indicator_data = {
                        "symbol": symbol,
                        "timestamp": tech_analysis.get(
                            "timestamp", datetime.now(timezone.utc)
                        ),
                        "timeframe": "1day",  # Default timeframe
                        "sma_20": current_indicators.get("sma_20"),
                        "sma_50": current_indicators.get("sma_50"),
                        "sma_200": current_indicators.get("sma_200"),
                        "ema_12": current_indicators.get(
                            "ema_10"
                        ),  # Use ema_10 as closest to ema_12
                        "ema_26": current_indicators.get(
                            "ema_20"
                        ),  # Use ema_20 as closest to ema_26
                        "rsi": current_indicators.get("rsi_14"),
                        "macd": current_indicators.get("macd_line"),
                        "macd_signal": current_indicators.get("macd_signal"),
                        "macd_histogram": current_indicators.get("macd_histogram"),
                        "bollinger_upper": current_indicators.get("bb_upper"),
                        "bollinger_middle": current_indicators.get("bb_middle"),
                        "bollinger_lower": current_indicators.get("bb_lower"),
                        "atr": current_indicators.get("atr_14"),
                        "volume_sma": current_indicators.get(
                            "obv"
                        ),  # Use OBV as volume indicator
                    }

                    indicators_to_save.append(indicator_data)

                    analysis_results.append(
                        {
                            "symbol": symbol,
                            "technical_score": tech_analysis.get(
                                "technical_score", 50.0
                            ),
                            "regime": tech_analysis.get("regime", {}),
                            "patterns": tech_analysis.get("patterns", []),
                            "data_points": tech_analysis.get("data_points", 0),
                        }
                    )

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        # Save technical indicators if any were calculated
        saved_count = 0
        if indicators_to_save:
            try:
                logger.debug(f"Saving {len(indicators_to_save)} technical indicators")
                # Direct file saving approach
                from datetime import date
                from pathlib import Path

                base_path = Path("/app/data/parquet/technical_indicators")
                base_path.mkdir(parents=True, exist_ok=True)

                # Group indicators by symbol and save
                grouped_indicators: dict[str, Any] = {}
                for indicator in indicators_to_save:
                    symbol = indicator["symbol"]
                    if symbol not in grouped_indicators:
                        grouped_indicators[symbol] = []
                    grouped_indicators[symbol].append(indicator)

                for symbol, symbol_indicators in grouped_indicators.items():
                    # Create symbol directory
                    symbol_dir = base_path / symbol / "1day"
                    symbol_dir.mkdir(parents=True, exist_ok=True)

                    # Create filename with current date
                    today = date.today()
                    file_path = symbol_dir / f"{today.isoformat()}.parquet"

                    # Convert to DataFrame and save
                    df = pl.DataFrame(symbol_indicators)
                    df.write_parquet(file_path)
                    saved_count += len(symbol_indicators)

                logger.info(f"Saved {saved_count} technical indicators to {base_path}")

            except Exception as e:
                logger.error(f"Error saving technical indicators: {e}")

        return {
            "status": "success",
            "message": f"Analysis completed for {len(analysis_results)} symbols",
            "symbols_analyzed": len(analysis_results),
            "indicators_saved": saved_count,
            "results": analysis_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in market data analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/regime/{symbol}")
async def get_market_regime(symbol: str):
    """Get market regime for a symbol."""
    return await service.get_market_regime(symbol)


@app.get("/strategies/{strategy_name}")
async def get_strategy_details(strategy_name: str):
    """Get detailed information about a strategy."""
    try:
        logger.debug(f"Getting details for strategy: {strategy_name}")
        if strategy_name not in service.active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy = service.active_strategies[strategy_name]

        return {
            "status": "success",
            "strategy": {
                "name": strategy_name,
                "type": strategy.__class__.__name__,
                "config": strategy.get_strategy_info(),
                "state": (
                    strategy.save_state() if hasattr(strategy, "save_state") else {}
                ),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/strategies/{strategy_name}")
async def delete_strategy(strategy_name: str):
    """Delete a strategy."""
    try:
        logger.debug(f"Deleting strategy: {strategy_name}")
        if strategy_name not in service.active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Remove from active strategies
        del service.active_strategies[strategy_name]
        logger.debug(f"Strategy {strategy_name} removed from active strategies")

        # TODO: Unregister from Redis engine

        return {"status": "success", "message": f"Strategy {strategy_name} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/history/{symbol}")
async def get_signal_history(symbol: str, limit: int = 50):
    """Get signal history for a symbol."""
    try:
        logger.debug(f"Getting signal history for {symbol}, limit: {limit}")
        if not service.redis_engine:
            raise HTTPException(status_code=503, detail="Redis not available")

        history = await service.redis_engine.get_signal_history(symbol, limit)

        return {
            "status": "success",
            "symbol": symbol,
            "signals": history,
            "count": len(history),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        logger.debug("Generating Prometheus metrics")
        # Initialize metrics if not already done
        _initialize_metrics()

        # Update metrics before returning them
        if strategies_count_gauge is not None:
            strategies_count_gauge.set(len(service.active_strategies))
        if service_health_gauge is not None:
            service_health_gauge.labels(component="redis").set(
                1 if service.redis_engine else 0
            )
            service_health_gauge.labels(component="service").set(1)

        # Generate Prometheus format
        metrics_output = generate_latest()

        return Response(content=metrics_output, media_type="text/plain; version=0.0.4")

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain; version=0.0.4",
            status_code=500,
        )


@app.get("/metrics/json")
async def get_system_metrics():
    """Get system performance metrics in JSON format."""
    try:
        # Initialize metrics if not already done
        _initialize_metrics()

        metrics = {}

        # Strategy metrics
        metrics["strategies"] = {
            "active_count": len(service.active_strategies),
            "strategy_names": list(service.active_strategies.keys()),
        }

        # Redis metrics if available
        if service.redis_engine:
            redis_metrics = await service.redis_engine.get_system_metrics()
            metrics["redis"] = redis_metrics

        # Service metrics
        metrics["service"] = {
            "uptime": "unknown",  # Would track actual uptime
            "config": service.config,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }

        return {"status": "success", "metrics": metrics}

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategies/{strategy_name}/update")
async def update_strategy(strategy_name: str, parameters: Dict[str, Any]):
    """Update strategy parameters."""
    try:
        logger.debug(f"Updating strategy {strategy_name} with parameters: {parameters}")
        if strategy_name not in service.active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy = service.active_strategies[strategy_name]
        strategy.update_config(parameters)

        # Cache updated state
        if service.redis_engine and service.redis_engine.cache:
            await service.redis_engine.cache.cache_strategy_state(strategy)

        return {
            "status": "success",
            "message": f"Strategy {strategy_name} updated",
            "updated_config": strategy.get_strategy_info(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/screener/tickers")
async def get_screener_tickers():
    """Get current active tickers from the screener."""
    try:
        logger.debug("Getting current screener tickers")
        tickers = await service.refresh_screener_tickers()

        return {
            "status": "success",
            "tickers": tickers,
            "count": len(tickers),
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }

    except Exception as e:
        logger.error(f"Error getting screener tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screener/refresh")
async def refresh_screener_tickers():
    """Refresh tickers from screener and restart real-time processing if needed."""
    try:
        logger.debug("Refreshing screener tickers")
        tickers = await service.refresh_screener_tickers()

        # Optionally restart real-time processing with new tickers
        if service.config.get("enable_real_time", False) and tickers:
            logger.info("Restarting real-time processing with refreshed tickers")
            await service.start_real_time_processing()

        return {
            "status": "success",
            "message": f"Refreshed {len(tickers)} tickers from screener",
            "tickers": tickers,
            "real_time_restarted": service.config.get("enable_real_time", False)
            and len(tickers) > 0,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        }

    except Exception as e:
        logger.error(f"Error refreshing screener tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reports/eod")
async def generate_eod_report(request: Optional[EODReportRequest] = None):
    """Generate End-of-Day trading report."""
    try:
        logger.debug("EOD report generation requested")
        report = await service.generate_eod_report(request)

        # Update Prometheus metrics if available
        try:
            _initialize_metrics()
            if strategies_count_gauge is not None:
                strategies_count_gauge.set(len(service.active_strategies))
        except Exception as metrics_error:
            logger.warning(f"Could not update metrics: {metrics_error}")

        return {"status": "success", "report": report}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating EOD report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/eod")
async def get_latest_eod_report():
    """Get the most recent EOD report."""
    try:
        logger.debug("Getting latest EOD report")
        # Try to get from Redis first
        if service.redis_engine and service.redis_engine.redis:
            today = datetime.now(timezone.utc).date()
            report_key = f"eod_report:{today.isoformat()}"
            logger.debug(f"Looking for cached EOD report with key: {report_key}")

            try:
                stored_report = await service.redis_engine.redis.get(report_key)
                if stored_report:
                    logger.debug("Found cached EOD report")
                    report_data = json.loads(stored_report)
                    return {
                        "status": "success",
                        "report": report_data,
                        "source": "cached",
                    }
            except Exception as redis_error:
                logger.warning(f"Could not retrieve cached report: {redis_error}")

        # Generate new report if no cached version available
        logger.info("No cached EOD report found, generating new one")
        report = await service.generate_eod_report()

        return {"status": "success", "report": report, "source": "generated"}

    except Exception as e:
        logger.error(f"Error getting EOD report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_service_status():
    """Get detailed service status."""
    try:
        logger.debug("Getting service status")
        status = {
            "service": "strategy_engine",
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "version": "1.0.0",
            "components": {
                "redis_engine": service.redis_engine is not None,
                "backtesting_engine": service.backtesting_engine is not None,
                "regime_manager": service.regime_manager is not None,
                "technical_engine": service.technical_engine is not None,
                "fundamental_engine": service.fundamental_engine is not None,
            },
            "strategies": {
                "active_count": len(service.active_strategies),
                "names": list(service.active_strategies.keys()),
            },
            "config": service.config,
        }

        # Add Redis status if available
        if service.redis_engine:
            try:
                logger.debug("Checking Redis connection status")
                if service.redis_engine and service.redis_engine.redis:
                    await service.redis_engine.redis.ping()
                    status["redis_status"] = "connected"
                    logger.debug("Redis connection confirmed")
                else:
                    status["redis_status"] = "disconnected"
                    logger.debug("Redis engine exists but no connection")
            except Exception as redis_error:
                status["redis_status"] = "disconnected"
                logger.debug(f"Redis connection check failed: {redis_error}")

        return status

    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class StrategyEngineApp:
    """Application wrapper for Strategy Engine service for integration testing."""

    def __init__(self):
        """Initialize the Strategy Engine application."""
        self.service = service
        self.app = app
        self._initialized = False
        self.logger = logging.getLogger("strategy_engine_app")
        self.logger.debug("StrategyEngineApp instance created")

    async def initialize(self):
        """Initialize the application."""
        if not self._initialized:
            self.logger.debug("Initializing StrategyEngineApp")
            await self.service.initialize()
            self._initialized = True
            self.logger.debug("StrategyEngineApp initialization complete")

    async def start(self):
        """Start the Strategy Engine service."""
        self.logger.debug("Starting StrategyEngineApp")
        await self.initialize()
        if self.service.config.get("enable_real_time", False):
            self.logger.debug("Real-time processing enabled, starting background task")
            asyncio.create_task(self.service.start_real_time_processing())
        self.logger.debug("StrategyEngineApp start complete")

    async def stop(self):
        """Stop the Strategy Engine service."""
        self.logger.debug("Stopping StrategyEngineApp")
        await self.service.shutdown()
        self._initialized = False
        self.logger.debug("StrategyEngineApp stopped")

    def get_service(self):
        """Get the underlying service instance."""
        self.logger.debug("Returning service instance")
        return self.service

    def get_app(self):
        """Get the FastAPI application instance."""
        self.logger.debug("Returning FastAPI app instance")
        return self.app


if __name__ == "__main__":
    # Development server
    port = int(os.environ.get("SERVICE_PORT", 9102))
    logger.debug(f"Starting development server on port {port}")
    logger.info(f"Strategy Engine Service starting on port {port}")
    uvicorn.run(
        "src.main:app", host="0.0.0.0", port=port, reload=False, log_level="info"
    )
