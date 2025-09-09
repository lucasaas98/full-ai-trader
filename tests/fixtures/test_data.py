"""
Test fixtures and data generators for comprehensive testing.
This module provides realistic test data for all trading system components.
"""

import random

# Import shared models
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append("/app/shared")
# Models imported but not used in this module - available for test data generation
from typing import Tuple  # noqa: E402


def _safe_float_convert(value: Any) -> float:
    """Safely convert pandas/numpy values to float."""
    if pd.isna(value):
        return 0.0
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _safe_index_to_int(index_value: Any) -> int:
    """Safely convert pandas index value to int."""
    try:
        if hasattr(index_value, "item"):
            return int(index_value.item())
        return int(index_value)
    except (TypeError, ValueError, AttributeError):
        return 0


class TestDataGenerator:
    """Generate realistic test data for various trading scenarios"""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducible tests"""
        np.random.seed(seed)
        random.seed(seed)

    def generate_realistic_price_series(
        self,
        symbol: str,
        start_price: float = 100.0,
        num_days: int = 252,
        volatility: float = 0.2,
        drift: float = 0.05,
        include_gaps: bool = False,
        include_splits: bool = False,
    ) -> pd.DataFrame:
        """Generate realistic price series with various market conditions"""

        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=num_days),
            periods=num_days,
            freq="D",
        )

        # Generate returns using geometric Brownian motion
        dt = 1 / 252  # Daily time step
        returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), num_days)

        # Add occasional volatility clusters
        for i in range(num_days):
            if np.random.random() < 0.05:  # 5% chance of volatility spike
                cluster_size = np.random.randint(3, 10)
                cluster_end = min(i + cluster_size, num_days)
                returns[i:cluster_end] *= 2.0  # Double volatility during cluster

        # Generate prices
        prices = [start_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # Generate OHLCV data
        ohlcv_data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic intraday range
            daily_volatility = volatility / np.sqrt(252)
            high_factor = 1 + abs(np.random.normal(0, daily_volatility / 2))
            low_factor = 1 - abs(np.random.normal(0, daily_volatility / 2))

            if i == 0:
                open_price = start_price
            else:
                # Gap factor for overnight moves
                gap_factor = 1 + np.random.normal(0, daily_volatility / 4)
                open_price = prices[i - 1] * gap_factor

            high = max(open_price, price) * high_factor
            low = min(open_price, price) * low_factor

            # Include price gaps if specified
            if include_gaps and np.random.random() < 0.02:  # 2% chance of gap
                gap_size = np.random.normal(0, volatility / 10)
                open_price *= 1 + gap_size
                high = max(open_price, price * (1 + gap_size))
                low = min(open_price, price * (1 + gap_size))
                price *= 1 + gap_size

            # Include stock splits if specified
            if include_splits and np.random.random() < 0.01:  # 1% chance of split
                split_ratio = random.choice([2, 3])  # 2:1 or 3:1 splits
                open_price /= split_ratio
                high /= split_ratio
                low /= split_ratio
                price /= split_ratio
                prices[i] = price

            # Generate volume with realistic patterns
            base_volume = 1000000
            volume_factor = np.random.lognormal(0, 0.5)  # Log-normal distribution
            volume = int(base_volume * volume_factor)

            # Higher volume on price movements
            if i > 0:
                price_change = abs(returns[i])
                volume *= 1 + price_change * 10  # Volume increases with price movement

            ohlcv_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(price, 2),
                    "volume": int(volume),
                }
            )

        return pd.DataFrame(ohlcv_data)

    def generate_intraday_data(
        self,
        symbol: str,
        date: datetime,
        start_price: float = 100.0,
        volatility: float = 0.2,
        num_minutes: int = 390,
    ) -> pd.DataFrame:
        """Generate realistic intraday minute-by-minute data"""

        # Market hours: 9:30 AM to 4:00 PM EST (390 minutes)
        market_open = date.replace(hour=9, minute=30, second=0, microsecond=0)
        timestamps = [market_open + timedelta(minutes=i) for i in range(num_minutes)]

        # Intraday patterns
        minute_volatility = volatility / np.sqrt(252 * 390)  # Per-minute volatility

        # Higher volatility at open/close
        volatility_pattern = np.ones(num_minutes)
        volatility_pattern[:30] *= 2.0  # First 30 minutes
        volatility_pattern[-30:] *= 1.5  # Last 30 minutes
        volatility_pattern[180:210] *= 1.3  # Lunch time volatility

        prices = [start_price]
        volumes = []

        for i in range(num_minutes):
            # Generate return with time-varying volatility
            vol = minute_volatility * volatility_pattern[i]
            ret = np.random.normal(0, vol)
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)

            # Generate volume with intraday patterns
            base_volume = 1000
            time_factor = 1.0

            # Higher volume at open/close
            if i < 30 or i > 360:
                time_factor = 2.0
            elif 180 <= i <= 210:  # Lunch time
                time_factor = 0.7

            volume = int(base_volume * time_factor * np.random.lognormal(0, 0.3))
            volumes.append(volume)

        # Generate OHLCV for each minute
        intraday_data = []
        for i, (timestamp, volume) in enumerate(zip(timestamps, volumes)):
            open_price = prices[i]
            close_price = prices[i + 1]

            # Generate high/low within reasonable bounds
            price_range = abs(close_price - open_price)
            high = max(open_price, close_price) + price_range * np.random.random() * 0.5
            low = min(open_price, close_price) - price_range * np.random.random() * 0.5

            intraday_data.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close_price, 2),
                    "volume": volume,
                }
            )

        return pd.DataFrame(intraday_data)

    def generate_multi_asset_data(
        self,
        symbols: List[str],
        num_days: int = 252,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Generate correlated multi-asset price data"""

        if correlation_matrix is None:
            # Generate random correlation matrix
            n_assets = len(symbols)
            A = np.random.randn(n_assets, n_assets)
            correlation_matrix = np.corrcoef(A)

        # Generate correlated returns
        mean_returns = np.random.normal(
            0.0005, 0.0002, len(symbols)
        )  # Different expected returns
        volatilities = np.random.normal(
            0.02, 0.005, len(symbols)
        )  # Different volatilities

        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(correlation_matrix)

        # Generate independent random returns
        independent_returns = np.random.normal(0, 1, (num_days, len(symbols)))

        # Apply correlation
        correlated_returns = independent_returns @ L.T

        # Scale by volatility and add drift
        for i, symbol in enumerate(symbols):
            correlated_returns[:, i] = (
                correlated_returns[:, i] * volatilities[i] + mean_returns[i]
            )

        # Generate price data for each asset
        asset_data = {}
        base_prices = np.random.uniform(50, 500, len(symbols))

        for i, symbol in enumerate(symbols):
            asset_returns = correlated_returns[:, i]
            asset_prices = [base_prices[i]]

            for ret in asset_returns:
                asset_prices.append(asset_prices[-1] * (1 + ret))

            # Create DataFrame
            asset_data[symbol] = self.generate_realistic_price_series(
                symbol=symbol,
                start_price=base_prices[i],
                num_days=num_days,
                volatility=volatilities[i],
                drift=mean_returns[i] * 252,
            )

            # Replace close prices with our correlated prices
            asset_data[symbol]["close"] = asset_prices[1:]

        return asset_data

    def generate_market_regime_data(
        self, symbol: str, num_days: int = 252
    ) -> pd.DataFrame:
        """Generate data with different market regimes (bull, bear, sideways)"""

        regime_length = 60  # Average regime length in days
        regimes = []

        current_day = 0
        while current_day < num_days:
            # Choose regime type
            regime_type = np.random.choice(
                ["bull", "bear", "sideways"], p=[0.4, 0.3, 0.3]
            )

            # Regime duration
            duration = max(30, int(np.random.exponential(regime_length)))
            duration = min(duration, num_days - current_day)

            regimes.append(
                {
                    "type": regime_type,
                    "start": current_day,
                    "end": current_day + duration,
                    "duration": duration,
                }
            )

            current_day += duration

        # Generate price data based on regimes
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=num_days),
            periods=num_days,
            freq="D",
        )

        prices = [100.0]  # Starting price
        volumes = []

        for day in range(num_days):
            # Find current regime
            current_regime = None
            for regime in regimes:
                if regime["start"] <= day < regime["end"]:
                    current_regime = regime
                    break

            if current_regime is None:
                current_regime = {"type": "sideways"}

            # Generate return based on regime
            if current_regime["type"] == "bull":
                drift = 0.0008  # Positive drift
                volatility = 0.015  # Moderate volatility
                volume_multiplier = 1.2
            elif current_regime["type"] == "bear":
                drift = -0.0005  # Negative drift
                volatility = 0.025  # Higher volatility
                volume_multiplier = 1.5
            else:  # sideways
                drift = 0.0001  # Minimal drift
                volatility = 0.012  # Lower volatility
                volume_multiplier = 0.8

            # Generate daily return
            daily_return = np.random.normal(drift, volatility)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(new_price)

            # Generate volume
            base_volume = 1000000
            volume = int(base_volume * volume_multiplier * np.random.lognormal(0, 0.3))
            volumes.append(volume)

        # Create OHLCV data
        ohlcv_data = []
        for i, (date, close_price, volume) in enumerate(
            zip(dates, prices[1:], volumes)
        ):
            # Generate OHLC from close prices
            if i == 0:
                open_price = prices[0]
            else:
                gap = np.random.normal(0, 0.002)  # Overnight gap
                open_price = prices[i] * (1 + gap)

            daily_range = abs(close_price - open_price)
            high = max(open_price, close_price) + daily_range * np.random.random() * 0.5
            low = min(open_price, close_price) - daily_range * np.random.random() * 0.5

            ohlcv_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close_price, 2),
                    "volume": volume,
                }
            )

        df = pd.DataFrame(ohlcv_data)
        df["regime"] = "sideways"  # Default

        # Add regime labels
        for regime in regimes:
            mask = (df.index >= regime["start"]) & (df.index < regime["end"])
            df.loc[mask, "regime"] = regime["type"]

        return df

    def generate_earnings_event_data(
        self, symbol: str, base_data: pd.DataFrame, earnings_dates: List[datetime]
    ) -> pd.DataFrame:
        """Generate data with earnings announcement effects"""

        data = base_data.copy()

        for earnings_date in earnings_dates:
            # Find closest trading day
            try:
                earnings_idx = _safe_index_to_int(
                    data[data["timestamp"].dt.date == earnings_date.date()].index[0]
                )
            except IndexError:
                continue

            # Earnings surprise effect
            surprise_direction = np.random.choice([-1, 1])  # Beat or miss
            surprise_magnitude = np.random.uniform(0.02, 0.08)  # 2-8% move

            # Apply earnings gap
            if earnings_idx < len(data) - 1:
                gap_factor = 1 + (surprise_direction * surprise_magnitude)
                data.loc[earnings_idx:, "open"] *= gap_factor
                data.loc[earnings_idx:, "high"] *= gap_factor
                data.loc[earnings_idx:, "low"] *= gap_factor
                data.loc[earnings_idx:, "close"] *= gap_factor

                # Increased volume around earnings
                volume_spike = np.random.uniform(2.0, 5.0)
                data.loc[
                    max(0, earnings_idx - 1) : earnings_idx + 1, "volume"
                ] *= volume_spike

        return data

    def generate_market_crash_scenario(
        self,
        base_data: pd.DataFrame,
        crash_date: datetime,
        crash_severity: float = 0.20,
    ) -> pd.DataFrame:
        """Generate market crash scenario data"""

        data = base_data.copy()

        # Find crash date index
        try:
            crash_idx = _safe_index_to_int(
                data[data["timestamp"].dt.date >= crash_date.date()].index[0]
            )
        except IndexError:
            return data

        # Apply crash
        crash_factor = 1 - crash_severity
        recovery_days = np.random.randint(10, 30)

        # Crash day - significant drop
        data.loc[crash_idx, "open"] = data.loc[max(0, crash_idx - 1), "close"]
        open_val = data.loc[crash_idx, "open"]
        volume_val = data.loc[crash_idx, "volume"]
        data.loc[crash_idx, "high"] = _safe_float_convert(open_val) * 0.98
        data.loc[crash_idx, "low"] = _safe_float_convert(open_val) * crash_factor
        low_val = data.loc[crash_idx, "low"]
        data.loc[crash_idx, "close"] = _safe_float_convert(low_val) * 1.02
        data.loc[crash_idx, "volume"] = (
            _safe_float_convert(volume_val) * 10
        )  # Panic selling volume

        # Recovery period with increased volatility
        recovery_end = min(int(crash_idx + recovery_days), len(data) - 1)
        for i in range(int(crash_idx + 1), int(recovery_end)):
            # Gradual recovery with high volatility
            recovery_progress = (i - int(crash_idx)) / recovery_days
            volatility_multiplier = 3.0 * (1 - recovery_progress) + 1.0

            # Apply to price ranges
            high_val = data.loc[i, "high"]
            low_val = data.loc[i, "low"]
            volume_val = data.loc[i, "volume"]
            data.loc[i, "high"] = _safe_float_convert(high_val) * (
                1 + 0.02 * volatility_multiplier
            )
            data.loc[i, "low"] = _safe_float_convert(low_val) * (
                1 - 0.02 * volatility_multiplier
            )
            data.loc[i, "volume"] = _safe_float_convert(volume_val) * (
                2.0 * (1 - recovery_progress) + 1.0
            )

        return data

    def generate_options_data(
        self,
        underlying_symbol: str,
        underlying_price: float,
        expiration_dates: List[datetime],
        strike_range: Tuple[float, float] = (0.8, 1.2),
    ) -> pd.DataFrame:
        """Generate realistic options chain data"""

        options_data = []

        for exp_date in expiration_dates:
            days_to_expiry = (exp_date - datetime.now(timezone.utc)).days
            time_value = days_to_expiry / 365.25

            # Generate strikes around current price
            min_strike = underlying_price * strike_range[0]
            max_strike = underlying_price * strike_range[1]
            strikes = np.arange(
                int(min_strike / 5) * 5,  # Round to nearest $5
                int(max_strike / 5) * 5 + 5,
                5,
            )

            for strike in strikes:
                # Black-Scholes approximation for testing
                moneyness = underlying_price / strike
                implied_vol = 0.2 + 0.1 * abs(1 - moneyness)  # Volatility smile

                # Simplified option pricing for test data
                intrinsic_call = max(0, underlying_price - strike)
                intrinsic_put = max(0, strike - underlying_price)

                time_premium = time_value * implied_vol * underlying_price * 0.1

                call_price = intrinsic_call + time_premium
                put_price = intrinsic_put + time_premium

                # Generate bid/ask spreads
                spread_pct = 0.02 + 0.01 * abs(1 - moneyness)  # Wider spreads for OTM

                options_data.extend(
                    [
                        {
                            "underlying_symbol": underlying_symbol,
                            "option_symbol": f"{underlying_symbol}{exp_date.strftime('%y%m%d')}C{strike:08.0f}",
                            "option_type": "call",
                            "strike": strike,
                            "expiration": exp_date,
                            "bid": call_price * (1 - spread_pct / 2),
                            "ask": call_price * (1 + spread_pct / 2),
                            "last": call_price,
                            "volume": np.random.randint(0, 1000),
                            "open_interest": np.random.randint(100, 5000),
                            "implied_volatility": implied_vol,
                            "delta": 0.5 + 0.4 * (moneyness - 1),
                            "gamma": 0.02 * np.exp(-0.5 * (moneyness - 1) ** 2),
                            "theta": -call_price * 0.01 * time_value,
                            "vega": underlying_price * 0.01 * np.sqrt(time_value),
                        },
                        {
                            "underlying_symbol": underlying_symbol,
                            "option_symbol": f"{underlying_symbol}{exp_date.strftime('%y%m%d')}P{strike:08.0f}",
                            "option_type": "put",
                            "strike": strike,
                            "expiration": exp_date,
                            "bid": put_price * (1 - spread_pct / 2),
                            "ask": put_price * (1 + spread_pct / 2),
                            "last": put_price,
                            "volume": np.random.randint(0, 1000),
                            "open_interest": np.random.randint(100, 5000),
                            "implied_volatility": implied_vol,
                            "delta": -0.5 + 0.4 * (1 - moneyness),
                            "gamma": 0.02 * np.exp(-0.5 * (moneyness - 1) ** 2),
                            "theta": -put_price * 0.01 * time_value,
                            "vega": underlying_price * 0.01 * np.sqrt(time_value),
                        },
                    ]
                )

        return pd.DataFrame(options_data)

    def generate_news_sentiment_data(
        self,
        symbol: str,
        date_range: Tuple[datetime, datetime],
        correlation_with_returns: float = 0.3,
    ) -> pd.DataFrame:
        """Generate news sentiment data correlated with price movements"""

        start_date, end_date = date_range
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        sentiment_data = []

        for date in dates:
            # Base sentiment (slightly positive bias)
            base_sentiment = np.random.normal(0.1, 0.3)

            # Add some correlation with simulated returns
            if correlation_with_returns > 0:
                price_return = np.random.normal(0.001, 0.02)
                correlated_component = correlation_with_returns * price_return * 10
                base_sentiment += correlated_component

            # Clamp sentiment between -1 and 1
            sentiment_score = np.clip(base_sentiment, -1, 1)

            # Generate news articles count (more news on significant days)
            base_articles = 5
            news_multiplier = 1 + abs(sentiment_score) * 2
            num_articles = int(
                base_articles * news_multiplier * np.random.lognormal(0, 0.5)
            )

            sentiment_data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "sentiment_score": round(sentiment_score, 3),
                    "num_articles": num_articles,
                    "positive_articles": int(
                        num_articles * max(0, sentiment_score + 1) / 2
                    ),
                    "negative_articles": int(
                        num_articles * max(0, 1 - sentiment_score) / 2
                    ),
                    "neutral_articles": num_articles
                    - int(num_articles * abs(sentiment_score)),
                }
            )

        return pd.DataFrame(sentiment_data)

    def generate_economic_indicators_data(
        self, date_range: Tuple[datetime, datetime]
    ) -> pd.DataFrame:
        """Generate economic indicators data"""

        start_date, end_date = date_range

        # Monthly economic data
        months = pd.date_range(start=start_date, end=end_date, freq="MS")  # Month start

        indicators_data = []

        for month in months:
            # Generate correlated economic indicators
            gdp_growth = np.random.normal(0.02, 0.01)  # Quarterly GDP growth
            inflation = np.random.normal(0.025, 0.005)  # Annual inflation
            unemployment = np.random.normal(0.04, 0.01)  # Unemployment rate
            interest_rate = max(0, np.random.normal(0.025, 0.01))  # Fed funds rate

            # Consumer confidence (correlated with GDP)
            consumer_confidence = 50 + gdp_growth * 500 + np.random.normal(0, 5)
            consumer_confidence = np.clip(consumer_confidence, 0, 100)

            # Manufacturing PMI
            pmi = 50 + gdp_growth * 100 + np.random.normal(0, 3)
            pmi = np.clip(pmi, 0, 100)

            indicators_data.append(
                {
                    "date": month,
                    "gdp_growth_rate": round(gdp_growth, 4),
                    "inflation_rate": round(inflation, 4),
                    "unemployment_rate": round(unemployment, 4),
                    "federal_funds_rate": round(interest_rate, 4),
                    "consumer_confidence": round(consumer_confidence, 1),
                    "manufacturing_pmi": round(pmi, 1),
                    "dollar_index": round(100 + np.random.normal(0, 5), 2),
                    "oil_price": round(70 + np.random.normal(0, 10), 2),
                    "gold_price": round(1800 + np.random.normal(0, 100), 2),
                }
            )

        return pd.DataFrame(indicators_data)

    def generate_stress_test_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Generate various stress test scenarios"""

        scenarios = {
            "market_crash_2008": {
                "description": "2008-style financial crisis",
                "market_drop": -0.35,
                "volatility_spike": 3.0,
                "correlation_increase": 0.3,
                "liquidity_crunch": 0.7,
                "duration_days": 180,
            },
            "flash_crash": {
                "description": "Flash crash scenario",
                "market_drop": -0.10,
                "volatility_spike": 5.0,
                "correlation_increase": 0.8,
                "liquidity_crunch": 0.9,
                "duration_days": 1,
                "recovery_days": 5,
            },
            "black_swan": {
                "description": "Black swan event",
                "market_drop": -0.25,
                "volatility_spike": 4.0,
                "correlation_increase": 0.9,
                "liquidity_crunch": 0.8,
                "duration_days": 30,
                "specific_sector_impact": {
                    "financials": -0.45,
                    "energy": -0.30,
                    "technology": -0.15,
                },
            },
            "interest_rate_shock": {
                "description": "Sudden interest rate increase",
                "rate_increase": 0.03,  # 3% increase
                "bond_impact": -0.15,
                "growth_stock_impact": -0.20,
                "value_stock_impact": -0.05,
                "duration_days": 90,
            },
            "geopolitical_crisis": {
                "description": "Geopolitical crisis",
                "market_drop": -0.15,
                "volatility_spike": 2.5,
                "safe_haven_rally": {"gold": 0.10, "bonds": 0.05, "usd": 0.08},
                "commodity_impact": 0.25,
                "duration_days": 45,
            },
            "pandemic_scenario": {
                "description": "Pandemic-style crisis",
                "market_drop": -0.30,
                "volatility_spike": 3.5,
                "sector_specific_impact": {
                    "travel": -0.60,
                    "hospitality": -0.55,
                    "technology": 0.10,
                    "healthcare": 0.15,
                    "retail": -0.25,
                },
                "duration_days": 120,
                "recovery_days": 365,
            },
            "currency_crisis": {
                "description": "Currency devaluation crisis",
                "currency_drop": -0.25,
                "import_sensitive_impact": -0.15,
                "export_sensitive_impact": 0.10,
                "inflation_spike": 0.05,
                "duration_days": 60,
            },
        }

        return scenarios

    def apply_stress_scenario(
        self, base_data: pd.DataFrame, scenario: Dict[str, Any], start_date: datetime
    ) -> pd.DataFrame:
        """Apply stress scenario to base data"""

        data = base_data.copy()

        try:
            start_idx = _safe_index_to_int(
                data[data["timestamp"].dt.date == start_date.date()].index[0]
            )
        except IndexError:
            return data

        duration = scenario.get("duration_days", 30)
        end_idx = min(int(start_idx + duration), len(data) - 1)

        # Apply market drop
        if "market_drop" in scenario:
            drop_factor = 1 + scenario["market_drop"]
            data.loc[start_idx:end_idx, "close"] *= drop_factor
            data.loc[start_idx:end_idx, "high"] *= drop_factor
            data.loc[start_idx:end_idx, "low"] *= drop_factor

        # Apply volatility spike
        if "volatility_spike" in scenario:
            vol_multiplier = scenario["volatility_spike"]
            for i in range(int(start_idx), int(end_idx) + 1):
                high_val = data.loc[i, "high"]
                low_val = data.loc[i, "low"]
                high_float = _safe_float_convert(high_val)
                low_float = _safe_float_convert(low_val)
                daily_range = high_float - low_float
                expanded_range = (
                    daily_range * float(vol_multiplier)
                    if isinstance(vol_multiplier, (int, float))
                    else daily_range
                )
                mid_price = (high_float + low_float) / 2

                data.loc[i, "high"] = mid_price + expanded_range / 2
                data.loc[i, "low"] = mid_price - expanded_range / 2

        # Apply volume changes
        if "liquidity_crunch" in scenario:
            liquidity_factor = 1 - scenario["liquidity_crunch"]
            data.loc[start_idx:end_idx, "volume"] *= liquidity_factor

        return data

    def generate_algorithmic_trading_patterns(
        self, base_data: pd.DataFrame, pattern_type: str = "momentum"
    ) -> pd.DataFrame:
        """Generate data with algorithmic trading patterns"""

        data = base_data.copy()

        if pattern_type == "momentum":
            # Add momentum-driven price acceleration
            for i in range(20, len(data)):
                recent_returns = data["close"].iloc[i - 20 : i].pct_change().mean()

                if abs(recent_returns) > 0.01:  # Significant momentum
                    momentum_factor = 1 + (recent_returns * 0.5)  # Amplify movement
                    close_val = data.loc[i, "close"]
                    data.loc[i, "close"] = (
                        _safe_float_convert(close_val) * momentum_factor
                    )
                    volume_val = data.loc[i, "volume"]
                    data.loc[i, "volume"] = _safe_float_convert(volume_val) * (
                        1 + abs(recent_returns) * 2
                    )

        elif pattern_type == "mean_reversion":
            # Add mean reversion patterns
            close_values = data["close"].tolist()  # Convert to regular Python list

            for i in range(20, len(data)):
                # Calculate simple moving average manually to avoid type issues
                recent_prices = close_values[i - 20 : i]
                sma_value = sum(recent_prices) / len(recent_prices)
                close_val = data.loc[i, "close"]
                current_price = _safe_float_convert(close_val)
                deviation = (current_price - sma_value) / sma_value

                if abs(deviation) > 0.05:  # 5% deviation from mean
                    reversion_factor = 1 - (deviation * 0.3)  # Partial reversion
                    data.loc[i, "close"] = current_price * reversion_factor

        elif pattern_type == "arbitrage":
            # Add small price inefficiencies that can be arbitraged
            for i in range(1, len(data)):
                if i % 100 == 0:  # Periodic inefficiencies
                    noise_factor = 1 + np.random.uniform(-0.002, 0.002)
                    close_val = data.loc[i, "close"]
                    data.loc[i, "close"] = _safe_float_convert(close_val) * noise_factor

        return data
