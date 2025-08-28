"""
Strategy Engine Main Service

This is the main entry point for the strategy engine service that orchestrates
all components including technical analysis, fundamental analysis, hybrid strategies,
backtesting, market regime detection, and Redis integration.
"""

import asyncio
import logging
import os
import sys
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import polars as pl

# Add shared module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))

from .base_strategy import BaseStrategy, StrategyConfig, StrategyMode
from .technical_analysis import TechnicalStrategy, TechnicalAnalysisEngine
from .fundamental_analysis import FundamentalStrategy, FundamentalAnalysisEngine
from .hybrid_strategy import (
    HybridStrategy, HybridMode, HybridSignalGenerator, HybridSignal,
    HybridStrategyFactory
)
from .market_regime import MarketRegimeDetector, RegimeAwareStrategyManager
from .backtesting_engine import BacktestingEngine, BacktestConfig, BacktestMode
from .redis_integration import RedisStrategyEngine

# Import shared models
from shared.models import FinVizData, TechnicalIndicators


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("strategy_engine_main")


# Pydantic models for API
class StrategyCreateRequest(BaseModel):
    name: str = Field(..., description="Strategy name")
    strategy_type: str = Field(..., description="Strategy type: technical, fundamental, hybrid")
    mode: str = Field(default="swing_trading", description="Trading mode")
    symbols: List[str] = Field(..., description="Symbols to trade")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")


class SignalGenerationRequest(BaseModel):
    strategy_name: str = Field(..., description="Strategy name")
    symbol: str = Field(..., description="Trading symbol")
    include_analysis: bool = Field(default=False, description="Include detailed analysis")


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
    optimization_method: str = Field(default="grid_search", description="Optimization method")
    objective: str = Field(default="sharpe_ratio", description="Optimization objective")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")


class StrategyEngineService:
    """Main strategy engine service class."""

    def __init__(self):
        """Initialize strategy engine service."""
        self.logger = logging.getLogger("strategy_engine_service")

        # Core components
        self.redis_engine: Optional[RedisStrategyEngine] = None
        self.backtesting_engine: Optional[BacktestingEngine] = None
        self.regime_manager: Optional[RegimeAwareStrategyManager] = None

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
        redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}" if redis_password else f"redis://{redis_host}:{redis_port}"

        self.config = {
            "redis_url": os.getenv("REDIS_URL", redis_url),
            "max_concurrent_strategies": int(os.getenv("MAX_CONCURRENT_STRATEGIES", "10")),
            "signal_cache_ttl": int(os.getenv("SIGNAL_CACHE_TTL", "300")),
            "enable_real_time": os.getenv("ENABLE_REAL_TIME", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        }

    async def initialize(self) -> bool:
        """Initialize all service components."""
        try:
            self.logger.info("Initializing Strategy Engine Service")

            # Initialize Redis integration
            if self.config["enable_real_time"]:
                self.redis_engine = RedisStrategyEngine(self.config["redis_url"])
                if not await self.redis_engine.initialize():
                    self.logger.error("Failed to initialize Redis engine")
                    return False

            # Initialize backtesting engine
            backtest_config = BacktestConfig(
                initial_capital=100000.0,
                commission_per_trade=1.0,
                commission_percentage=0.001,
                slippage_percentage=0.0005
            )
            self.backtesting_engine = BacktestingEngine(backtest_config)

            # Initialize regime manager
            self.regime_manager = RegimeAwareStrategyManager()

            # Load any persisted strategies
            await self._load_persisted_strategies()

            self.logger.info("Strategy Engine Service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
            return False

    async def create_strategy(self, request: StrategyCreateRequest) -> Dict[str, Any]:
        """Create and register a new trading strategy."""
        try:
            # Validate strategy type
            if request.strategy_type not in ["technical", "fundamental", "hybrid"]:
                raise ValueError(f"Invalid strategy type: {request.strategy_type}")

            # Parse mode
            try:
                mode = StrategyMode(request.mode)
            except ValueError:
                raise ValueError(f"Invalid trading mode: {request.mode}")

            # Create strategy configuration
            config = StrategyConfig(
                name=request.name,
                mode=mode,
                lookback_period=request.parameters.get("lookback_period", 50),
                min_confidence=request.parameters.get("min_confidence", 60.0),
                max_position_size=request.parameters.get("max_position_size", 0.20),
                parameters=request.parameters
            )

            # Create strategy based on type
            if request.strategy_type == "technical":
                strategy = TechnicalStrategy(config)
            elif request.strategy_type == "fundamental":
                strategy = FundamentalStrategy(config)
            elif request.strategy_type == "hybrid":
                hybrid_mode = HybridMode(request.parameters.get("hybrid_mode", "swing_trading"))
                strategy = HybridStrategy(config, hybrid_mode)
            else:
                raise ValueError(f"Unsupported strategy type: {request.strategy_type}")

            # Initialize strategy
            strategy.initialize()

            # Register strategy
            self.active_strategies[request.name] = strategy

            # Register with Redis engine if available
            if self.redis_engine:
                await self.redis_engine.register_strategy(strategy, request.symbols)

            self.logger.info(f"Created and registered strategy: {request.name}")

            return {
                "status": "success",
                "strategy_name": request.name,
                "strategy_type": request.strategy_type,
                "symbols": request.symbols,
                "strategy_info": strategy.get_strategy_info()
            }

        except Exception as e:
            self.logger.error(f"Error creating strategy: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def generate_signal(self, request: SignalGenerationRequest) -> Dict[str, Any]:
        """Generate trading signal for a specific strategy and symbol."""
        try:
            # Get strategy
            if request.strategy_name not in self.active_strategies:
                raise ValueError(f"Strategy not found: {request.strategy_name}")

            strategy = self.active_strategies[request.strategy_name]

            # Get market data (this would typically come from database)
            market_data = await self._get_market_data(request.symbol, strategy.config.lookback_period)
            if market_data is None:
                raise ValueError(f"No market data available for {request.symbol}")

            # Get fundamental data for hybrid strategies
            finviz_data = None
            if isinstance(strategy, HybridStrategy):
                finviz_data = await self._get_finviz_data(request.symbol)

            # Generate signal
            if isinstance(strategy, HybridStrategy):
                signal = await strategy.analyze(request.symbol, market_data, finviz_data)
            else:
                signal = await strategy.analyze(request.symbol, market_data)

            # Format signal for output
            signal_generator = HybridSignalGenerator()
            # Convert to HybridSignal if needed
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
                    metadata=signal.metadata
                )
            else:
                hybrid_signal = signal

            formatted_signal = signal_generator.generate_formatted_signal(
                request.symbol, hybrid_signal, strategy.config.mode.value
            )

            # Add analysis details if requested
            if request.include_analysis:
                formatted_signal["analysis_details"] = {
                    "technical_score": getattr(signal, 'technical_score', 0.0),
                    "fundamental_score": getattr(signal, 'fundamental_score', 0.0),
                    "signal_metadata": signal.metadata if hasattr(signal, 'metadata') else {},
                    "strategy_info": strategy.get_strategy_info()
                }

            # Publish signal if Redis is available
            if self.redis_engine and self.redis_engine.publisher and signal.confidence >= strategy.config.min_confidence:
                await self.redis_engine.publisher.publish_signal(request.symbol, hybrid_signal, strategy.name)

            return {
                "status": "success",
                "signal": formatted_signal,
                "generated_at": datetime.now(timezone.utc).isoformat() + "Z"
            }

        except Exception as e:
            self.logger.error(f"Error generating signal: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def run_backtest(self, request: BacktestRequest) -> Dict[str, Any]:
        """Run backtest for a strategy."""
        try:
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

            # Get FinViz data for hybrid strategies
            finviz_data = None
            if isinstance(strategy, HybridStrategy):
                finviz_data = await self._get_finviz_data(request.symbol)

            # Parse backtest mode
            try:
                mode = BacktestMode(request.mode)
            except ValueError:
                mode = BacktestMode.SIMPLE

            # Run backtest
            if self.backtesting_engine:
                result = await self.backtesting_engine.backtest_strategy(
                    strategy, request.symbol, historical_data, start_date, end_date, finviz_data, mode
                )

                # Generate report
                report = self.backtesting_engine.generate_backtest_report(result)
            else:
                raise HTTPException(status_code=500, detail="Backtesting engine not available")

            # Store result in Redis if available
            result_id = f"{request.strategy_name}_{request.symbol}_{int(datetime.now(timezone.utc).timestamp())}"
            if self.redis_engine and self.redis_engine.redis:
                await self.redis_engine.redis.setex(
                    f"backtest_result:{result_id}",
                    86400 * 7,  # 7 days
                    json.dumps(report, default=str)
                )
            return {
                "status": "success",
                "backtest_id": result_id if 'result_id' in locals() else None,
                "report": report
            }

        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def optimize_parameters(self, request: ParameterOptimizationRequest) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        try:
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
                "max_combinations": 50
            }

            # Get FinViz data for hybrid strategies
            finviz_data = None
            if isinstance(strategy, HybridStrategy):
                finviz_data = await self._get_finviz_data(request.symbol)

            # Run optimization
            if self.backtesting_engine:
                optimization_result = await self.backtesting_engine.optimize_strategy_parameters(
                    strategy, request.symbol, historical_data, optimization_config, finviz_data
                )
            else:
                raise HTTPException(status_code=500, detail="Backtesting engine not available")

            # Update strategy with best parameters if found
            best_params = optimization_result.get("best_parameters", {})
            if best_params:
                strategy.update_config(best_params)
                self.logger.info(f"Updated {request.strategy_name} with optimized parameters")

            return {
                "status": "success",
                "optimization_result": optimization_result,
                "strategy_updated": bool(best_params)
            }

        except Exception as e:
            self.logger.error(f"Error optimizing parameters: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def get_market_regime(self, symbol: str) -> Dict[str, Any]:
        """Get current market regime for a symbol."""
        try:
            # Get recent market data
            market_data = await self._get_market_data(symbol, 100)
            if market_data is None:
                raise ValueError(f"No market data available for {symbol}")

            # Detect regime
            regime_detector = MarketRegimeDetector()
            regime_state = regime_detector.detect_regime(market_data)

            # Publish regime update if Redis is available
            if self.redis_engine and self.redis_engine.publisher:
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
                    "position_size_multiplier": regime_state.recommended_position_size
                },
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
            }

        except Exception as e:
            self.logger.error(f"Error getting market regime: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def _get_market_data(self, symbol: str, periods: int) -> Optional[pl.DataFrame]:
        """Get market data for analysis by loading from parquet files."""
        try:
            from pathlib import Path

            # Direct file access approach - assume default data path
            base_path = Path("/app/data/parquet/market_data") / symbol

            if not base_path.exists():
                self.logger.warning(f"No market data directory found for {symbol} at {base_path}")
                return None

            # Collect data from available timeframes (prefer daily, then hourly, then minutes)
            timeframes_to_try = ["1day", "1h", "15min", "5min", "1min"]
            combined_data = []

            for tf in timeframes_to_try:
                tf_path = base_path / tf
                if tf_path.exists():
                    # Load recent parquet files (last 6 months for better indicators)
                    parquet_files = list(tf_path.glob("*.parquet"))
                    parquet_files.sort()  # Sort by filename (date)

                    # Take the most recent files (up to 60 files for 2 months of daily data)
                    recent_files = parquet_files[-60:] if len(parquet_files) > 60 else parquet_files

                    for file_path in recent_files:
                        try:
                            df = pl.read_parquet(file_path)
                            if not df.is_empty():
                                combined_data.append(df)
                        except Exception as e:
                            self.logger.warning(f"Error reading {file_path}: {e}")
                            continue

                    # If we found data, use this timeframe
                    if combined_data:
                        break

            if not combined_data:
                self.logger.warning(f"No readable market data found for {symbol}")
                return None

            # Combine all dataframes
            full_data = pl.concat(combined_data)

            # Sort by timestamp and take most recent periods
            full_data = full_data.sort("timestamp").tail(periods)

            self.logger.info(f"Loaded {full_data.height} data points for {symbol}")
            return full_data

        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def _get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Optional[pl.DataFrame]:
        """Get historical data for backtesting (placeholder)."""
        try:
            # Placeholder - would query database with date range
            self.logger.warning(f"Historical data retrieval not implemented for {symbol}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None

    async def _get_finviz_data(self, symbol: str) -> Optional[FinVizData]:
        """Get FinViz fundamental data (placeholder)."""
        try:
            # Placeholder - would query database or FinViz API
            self.logger.warning(f"FinViz data retrieval not implemented for {symbol}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting FinViz data: {e}")
            return None

    async def _load_persisted_strategies(self) -> None:
        """Load any persisted strategies from previous runs."""
        try:
            if not self.redis_engine:
                return

            # This would load saved strategy configurations
            # For now, create some default strategies
            await self._create_default_strategies()

        except Exception as e:
            self.logger.error(f"Error loading persisted strategies: {e}")

    async def _create_default_strategies(self) -> None:
        """Create default strategies for demonstration."""
        try:
            # Create default hybrid strategies
            day_trader = HybridStrategyFactory.create_day_trading_strategy("default_day_trader")
            swing_trader = HybridStrategyFactory.create_swing_trading_strategy("default_swing_trader")
            position_trader = HybridStrategyFactory.create_position_trading_strategy("default_position_trader")

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

    async def start_real_time_processing(self) -> None:
        """Start real-time signal processing."""
        try:
            if not self.redis_engine:
                self.logger.warning("Redis engine not available - real-time processing disabled")
                return

            # Get all symbols from active strategies
            all_symbols = set()
            for strategy_info in self.active_strategies.values():
                # This would get symbols from strategy configuration
                # For now, use default symbols
                all_symbols.update(["AAPL", "MSFT", "GOOGL", "TSLA", "SPY"])

            # Start real-time processing
            await self.redis_engine.start_real_time_processing(list(all_symbols))

            self.logger.info("Real-time processing started")

        except Exception as e:
            self.logger.error(f"Error starting real-time processing: {e}")

    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        try:
            self.logger.info("Shutting down Strategy Engine Service")

            # Shutdown Redis engine
            if self.redis_engine:
                await self.redis_engine.shutdown()

            # Save strategy states
            for strategy in self.active_strategies.values():
                if hasattr(strategy, 'save_state'):
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
    await service.initialize()
    if service.config["enable_real_time"]:
        asyncio.create_task(service.start_real_time_processing())

    yield

    # Shutdown
    await service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Strategy Engine Service",
    description="Advanced trading strategy engine with technical and fundamental analysis",
    version="1.0.0",
    lifespan=lifespan
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
        strategies_count_gauge = Gauge('strategy_engine_strategies_total', 'Number of active strategies')
        signals_generated_counter = Counter('strategy_engine_signals_total', 'Total signals generated', ['strategy_type'])
        backtests_run_counter = Counter('strategy_engine_backtests_total', 'Total backtests run')
        service_health_gauge = Gauge('strategy_engine_service_health', 'Health status of components', ['component'])


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
        "active_strategies": len(service.active_strategies),
        "redis_available": service.redis_engine is not None
    }


@app.get("/strategies")
async def list_strategies():
    """List all active strategies."""
    try:
        strategies = []
        for name, strategy in service.active_strategies.items():
            strategies.append({
                "name": name,
                "type": strategy.__class__.__name__,
                "mode": strategy.config.mode.value,
                "info": strategy.get_strategy_info()
            })

        return {
            "status": "success",
            "strategies": strategies,
            "total_count": len(strategies)
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
        # Get tracked symbols from active strategies or default list
        symbols = set()

        # Collect symbols from active strategies
        for strategy in service.active_strategies.values():
            if hasattr(strategy, 'symbols') and getattr(strategy, 'symbols', None):
                symbols.update(getattr(strategy, 'symbols'))

        # Fallback to common symbols if no strategies are active
        if not symbols:
            symbols = {
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                'JPM', 'BAC', 'GS', 'MS', 'WFC', 'SPY', 'QQQ', 'IWM',
                'XLK', 'XLF', 'HD', 'WMT', 'KO', 'PEP', 'JNJ', 'PFE',
                'UNH', 'TMO', 'PG', 'ABBV'
            }

        analysis_results = []
        indicators_to_save = []

        for symbol in symbols:
            try:
                # Get market data for the symbol
                market_data = await service._get_market_data(symbol, periods=200)

                if market_data is None or market_data.is_empty():
                    logger.warning(f"No market data available for {symbol}")
                    continue

                # Ensure we have minimum required columns for technical analysis
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in market_data.columns for col in required_columns):
                    logger.warning(f"Missing required columns for {symbol}")
                    continue

                # Perform technical analysis with error handling
                try:
                    tech_analysis = service.technical_engine.full_analysis(symbol, market_data)
                except Exception as analysis_error:
                    logger.error(f"Technical analysis failed for {symbol}: {analysis_error}")
                    continue

                if tech_analysis and 'indicators' in tech_analysis:
                    # Extract current indicators for saving
                    current_indicators = tech_analysis['indicators']

                    # Create TechnicalIndicators object for storage
                    indicator_data = {
                        'symbol': symbol,
                        'timestamp': tech_analysis.get('timestamp', datetime.now(timezone.utc)),
                        'timeframe': '1day',  # Default timeframe
                        'sma_20': current_indicators.get('sma_20'),
                        'sma_50': current_indicators.get('sma_50'),
                        'sma_200': current_indicators.get('sma_200'),
                        'ema_12': current_indicators.get('ema_10'),  # Use ema_10 as closest to ema_12
                        'ema_26': current_indicators.get('ema_20'),  # Use ema_20 as closest to ema_26
                        'rsi': current_indicators.get('rsi_14'),
                        'macd': current_indicators.get('macd_line'),
                        'macd_signal': current_indicators.get('macd_signal'),
                        'macd_histogram': current_indicators.get('macd_histogram'),
                        'bollinger_upper': current_indicators.get('bb_upper'),
                        'bollinger_middle': current_indicators.get('bb_middle'),
                        'bollinger_lower': current_indicators.get('bb_lower'),
                        'atr': current_indicators.get('atr_14'),
                        'volume_sma': current_indicators.get('obv')  # Use OBV as volume indicator
                    }

                    indicators_to_save.append(indicator_data)

                    analysis_results.append({
                        'symbol': symbol,
                        'technical_score': tech_analysis.get('technical_score', 50.0),
                        'regime': tech_analysis.get('regime', {}),
                        'patterns': tech_analysis.get('patterns', []),
                        'data_points': tech_analysis.get('data_points', 0)
                    })

            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

        # Save technical indicators if any were calculated
        saved_count = 0
        if indicators_to_save:
            try:
                # Direct file saving approach
                from pathlib import Path
                from datetime import date

                base_path = Path("/app/data/parquet/technical_indicators")
                base_path.mkdir(parents=True, exist_ok=True)

                # Group indicators by symbol and save
                grouped_indicators = {}
                for indicator in indicators_to_save:
                    symbol = indicator['symbol']
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
            "timestamp": datetime.now(timezone.utc).isoformat()
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
        if strategy_name not in service.active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        strategy = service.active_strategies[strategy_name]

        return {
            "status": "success",
            "strategy": {
                "name": strategy_name,
                "type": strategy.__class__.__name__,
                "config": strategy.get_strategy_info(),
                "state": strategy.save_state() if hasattr(strategy, 'save_state') else {}
            }
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
        if strategy_name not in service.active_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # Remove from active strategies
        del service.active_strategies[strategy_name]

        # TODO: Unregister from Redis engine

        return {
            "status": "success",
            "message": f"Strategy {strategy_name} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/history/{symbol}")
async def get_signal_history(symbol: str, limit: int = 50):
    """Get signal history for a symbol."""
    try:
        if not service.redis_engine:
            raise HTTPException(status_code=503, detail="Redis not available")

        history = await service.redis_engine.get_signal_history(symbol, limit)

        return {
            "status": "success",
            "symbol": symbol,
            "signals": history,
            "count": len(history)
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
        # Initialize metrics if not already done
        _initialize_metrics()

        # Update metrics before returning them
        strategies_count_gauge.set(len(service.active_strategies))
        service_health_gauge.labels(component='redis').set(1 if service.redis_engine else 0)
        service_health_gauge.labels(component='service').set(1)

        # Generate Prometheus format
        metrics_output = generate_latest()

        return Response(
            content=metrics_output,
            media_type="text/plain; version=0.0.4"
        )

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain; version=0.0.4",
            status_code=500
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
            "strategy_names": list(service.active_strategies.keys())
        }

        # Redis metrics if available
        if service.redis_engine:
            redis_metrics = await service.redis_engine.get_system_metrics()
            metrics["redis"] = redis_metrics

        # Service metrics
        metrics["service"] = {
            "uptime": "unknown",  # Would track actual uptime
            "config": service.config,
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
        }

        return {
            "status": "success",
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategies/{strategy_name}/update")
async def update_strategy(strategy_name: str, parameters: Dict[str, Any]):
    """Update strategy parameters."""
    try:
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
            "updated_config": strategy.get_strategy_info()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_service_status():
    """Get detailed service status."""
    try:
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
                "fundamental_engine": service.fundamental_engine is not None
            },
            "strategies": {
                "active_count": len(service.active_strategies),
                "names": list(service.active_strategies.keys())
            },
            "config": service.config
        }

        # Add Redis status if available
        if service.redis_engine:
            try:
                if service.redis_engine and service.redis_engine.redis:
                    await service.redis_engine.redis.ping()
                    status["redis_status"] = "connected"
                else:
                    status["redis_status"] = "disconnected"
            except Exception:
                status["redis_status"] = "disconnected"

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

    async def initialize(self):
        """Initialize the application."""
        if not self._initialized:
            await self.service.initialize()
            self._initialized = True

    async def start(self):
        """Start the Strategy Engine service."""
        await self.initialize()
        if self.service.config.get("enable_real_time", False):
            asyncio.create_task(self.service.start_real_time_processing())

    async def stop(self):
        """Stop the Strategy Engine service."""
        await self.service.shutdown()
        self._initialized = False

    def get_service(self):
        """Get the underlying service instance."""
        return self.service

    def get_app(self):
        """Get the FastAPI application instance."""
        return self.app


if __name__ == "__main__":
    # Development server
    port = int(os.environ.get('SERVICE_PORT', 9102))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
