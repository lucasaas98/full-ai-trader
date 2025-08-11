"""
Data quality validation and anomaly detection service.

This module provides comprehensive data quality validation, anomaly detection,
market event detection, and data cleaning capabilities for the trading system.
"""

import logging
from datetime import datetime, timedelta, time as dt_time, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import statistics

import polars as pl
from pydantic import BaseModel, Field

from shared.models import MarketData, TimeFrame


logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of data anomalies."""
    PRICE_SPIKE = "price_spike"
    VOLUME_SPIKE = "volume_spike"
    MISSING_DATA = "missing_data"
    INVALID_OHLC = "invalid_ohlc"
    SPLIT_DETECTED = "split_detected"
    DIVIDEND_DETECTED = "dividend_detected"
    STALE_DATA = "stale_data"
    EXTREME_VOLATILITY = "extreme_volatility"
    ZERO_VOLUME = "zero_volume"
    PRICE_GAP = "price_gap"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataAnomaly(BaseModel):
    """Data anomaly detection result."""

    symbol: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: datetime
    timeframe: TimeFrame
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    suggested_action: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)


class DataQualityConfig(BaseModel):
    """Configuration for data quality validation."""

    # Price validation thresholds
    max_price_change_percent: float = Field(default=30.0, description="Maximum price change percentage")
    max_intraday_volatility: float = Field(default=50.0, description="Maximum intraday volatility")
    min_price: float = Field(default=0.01, description="Minimum valid price")
    max_price: float = Field(default=10000.0, description="Maximum valid price")

    # Volume validation thresholds
    max_volume_spike_ratio: float = Field(default=10.0, description="Maximum volume spike ratio")
    min_volume_threshold: int = Field(default=100, description="Minimum volume threshold")

    # Data freshness thresholds
    max_data_age_minutes: int = Field(default=60, description="Maximum data age in minutes")
    max_gap_multiplier: float = Field(default=3.0, description="Maximum gap as multiple of expected interval")

    # Split/dividend detection
    min_split_ratio: float = Field(default=1.5, description="Minimum ratio to detect splits")
    max_split_ratio: float = Field(default=10.0, description="Maximum ratio to detect splits")
    min_dividend_yield: float = Field(default=0.001, description="Minimum dividend yield for detection")

    # Statistical thresholds
    outlier_std_threshold: float = Field(default=3.0, description="Standard deviations for outlier detection")
    volatility_lookback_periods: int = Field(default=20, description="Periods for volatility calculation")

    # Market event detection
    enable_market_event_detection: bool = Field(default=True, description="Enable market event detection")
    earnings_impact_threshold: float = Field(default=5.0, description="Earnings impact threshold percentage")


class MarketEventDetector:
    """Detects market events from price and volume data."""

    def __init__(self, config: DataQualityConfig):
        self.config = config

    def detect_stock_split(self, data: pl.DataFrame) -> List[DataAnomaly]:
        """
        Detect potential stock splits from price data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            List of split anomalies detected
        """
        anomalies = []

        if len(data) < 2:
            return anomalies

        try:
            # Calculate day-to-day price ratios
            data_sorted = data.sort("timestamp")

            # Calculate price changes
            close_prices = data_sorted["close"].to_list()

            for i in range(1, len(close_prices)):
                prev_close = close_prices[i-1]
                curr_open = data_sorted["open"][i]

                if prev_close <= 0 or curr_open <= 0:
                    continue

                ratio = prev_close / curr_open

                # Check for potential split
                if self.config.min_split_ratio <= ratio <= self.config.max_split_ratio:
                    # Verify it's likely a split (clean ratio)
                    common_ratios = [2.0, 3.0, 4.0, 5.0, 1.5, 2.5]
                    is_clean_split = any(abs(ratio - r) < 0.1 for r in common_ratios)

                    if is_clean_split:
                        anomaly = DataAnomaly(
                            symbol=data_sorted["symbol"][i],
                            anomaly_type=AnomalyType.SPLIT_DETECTED,
                            severity=AnomalySeverity.HIGH,
                            timestamp=data_sorted["timestamp"][i],
                            timeframe=TimeFrame(data_sorted["timeframe"][i]),
                            description=f"Potential {ratio:.2f}:1 stock split detected",
                            details={
                                "split_ratio": ratio,
                                "previous_close": float(prev_close),
                                "current_open": float(curr_open),
                                "confidence": 0.8 if is_clean_split else 0.6
                            },
                            suggested_action="Verify split and adjust historical data",
                            confidence_score=0.8 if is_clean_split else 0.6
                        )
                        anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Split detection failed: {e}")

        return anomalies

    def detect_dividend_events(self, data: pl.DataFrame) -> List[DataAnomaly]:
        """
        Detect potential dividend events from price gaps.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            List of dividend anomalies detected
        """
        anomalies = []

        if len(data) < 2:
            return anomalies

        try:
            data_sorted = data.sort("timestamp")

            for i in range(1, len(data_sorted)):
                prev_close = data_sorted["close"][i-1]
                curr_open = data_sorted["open"][i]

                if prev_close <= 0 or curr_open <= 0:
                    continue

                # Check for gap down (potential dividend)
                gap_percent = (curr_open - prev_close) / prev_close * 100

                if gap_percent < -0.5:  # Gap down of more than 0.5%
                    # Estimate dividend amount
                    dividend_amount = prev_close - curr_open
                    dividend_yield = dividend_amount / prev_close * 100

                    if dividend_yield >= self.config.min_dividend_yield * 100:
                        anomaly = DataAnomaly(
                            symbol=data_sorted["symbol"][i],
                            anomaly_type=AnomalyType.DIVIDEND_DETECTED,
                            severity=AnomalySeverity.MEDIUM,
                            timestamp=data_sorted["timestamp"][i],
                            timeframe=TimeFrame(data_sorted["timeframe"][i]),
                            description=f"Potential dividend of ${dividend_amount:.2f} ({dividend_yield:.2f}%)",
                            details={
                                "gap_percent": gap_percent,
                                "estimated_dividend": float(dividend_amount),
                                "dividend_yield": dividend_yield,
                                "previous_close": float(prev_close),
                                "current_open": float(curr_open)
                            },
                            suggested_action="Verify dividend payment and adjust data",
                            confidence_score=0.7
                        )
                        anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Dividend detection failed: {e}")

        return anomalies

    def detect_earnings_impact(self, data: pl.DataFrame) -> List[DataAnomaly]:
        """
        Detect potential earnings-related price movements.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            List of earnings-related anomalies
        """
        anomalies = []

        if len(data) < 2:
            return anomalies

        try:
            data_sorted = data.sort("timestamp")

            for i in range(1, len(data_sorted)):
                prev_close = data_sorted["close"][i-1]
                curr_open = data_sorted["open"][i]
                curr_volume = data_sorted["volume"][i]

                if prev_close <= 0 or curr_open <= 0:
                    continue

                # Calculate gap and volume
                gap_percent = abs(curr_open - prev_close) / prev_close * 100

                # Look for significant gaps with high volume
                if gap_percent >= self.config.earnings_impact_threshold:
                    # Calculate average volume from previous periods
                    if i >= 5:
                        avg_volume = statistics.mean(data_sorted["volume"][i-5:i].to_list())
                        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 1

                        if volume_ratio >= 2.0:  # Volume spike
                            anomaly = DataAnomaly(
                                symbol=data_sorted["symbol"][i],
                                anomaly_type=AnomalyType.PRICE_SPIKE,
                                severity=AnomalySeverity.HIGH,
                                timestamp=data_sorted["timestamp"][i],
                                timeframe=TimeFrame(data_sorted["timeframe"][i]),
                                description=f"Potential earnings impact: {gap_percent:.1f}% gap with {volume_ratio:.1f}x volume",
                                details={
                                    "gap_percent": gap_percent,
                                    "volume_ratio": volume_ratio,
                                    "current_volume": curr_volume,
                                    "average_volume": avg_volume
                                },
                                suggested_action="Check for earnings announcement or news",
                                confidence_score=0.8
                            )
                            anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Earnings impact detection failed: {e}")

        return anomalies


class DataQualityValidator:
    """
    Comprehensive data quality validation and anomaly detection.

    Validates market data for consistency, detects anomalies,
    identifies market events, and provides data cleaning recommendations.
    """

    def __init__(self, config: DataQualityConfig):
        self.config = config
        self.event_detector = MarketEventDetector(config)

    async def validate_market_data(
        self,
        data: List[MarketData]
    ) -> Tuple[List[DataAnomaly], List[MarketData]]:
        """
        Validate market data and return anomalies and cleaned data.

        Args:
            data: List of MarketData objects to validate

        Returns:
            Tuple of (anomalies, cleaned_data)
        """
        if not data:
            return [], []

        # Convert to DataFrame for analysis
        df = self._to_dataframe(data)

        # Run all validation checks
        anomalies = []

        # Basic validation
        anomalies.extend(await self._validate_basic_integrity(df))
        anomalies.extend(await self._validate_price_ranges(df))
        anomalies.extend(await self._validate_volume_data(df))
        anomalies.extend(await self._detect_data_gaps(df))

        # Advanced validation
        anomalies.extend(await self._detect_price_anomalies(df))
        anomalies.extend(await self._detect_volume_anomalies(df))
        anomalies.extend(await self._validate_data_freshness(df))

        # Market event detection
        if self.config.enable_market_event_detection:
            anomalies.extend(self.event_detector.detect_stock_split(df))
            anomalies.extend(self.event_detector.detect_dividend_events(df))
            anomalies.extend(self.event_detector.detect_earnings_impact(df))

        # Clean data based on findings
        cleaned_data = await self._clean_data(data, anomalies)

        return anomalies, cleaned_data

    async def _validate_basic_integrity(self, df: pl.DataFrame) -> List[DataAnomaly]:
        """Validate basic data integrity (OHLC relationships, nulls, etc.)."""
        anomalies = []

        try:
            # Check for null values
            null_counts = df.null_count()
            for col, null_count in zip(df.columns, null_counts.row(0)):
                if null_count > 0:
                    anomaly = DataAnomaly(
                        symbol=df["symbol"][0] if len(df) > 0 else "UNKNOWN",
                        anomaly_type=AnomalyType.MISSING_DATA,
                        severity=AnomalySeverity.HIGH,
                        timestamp=datetime.now(timezone.utc),
                        timeframe=TimeFrame(df["timeframe"][0]) if len(df) > 0 else TimeFrame.FIVE_MINUTES,
                        description=f"Missing values in column '{col}': {null_count} records",
                        details={"column": col, "null_count": null_count},
                        suggested_action="Fill missing values or exclude records"
                    )
                    anomalies.append(anomaly)

            # Check OHLC relationships
            invalid_ohlc = df.filter(
                (pl.col("high") < pl.col("low")) |
                (pl.col("high") < pl.col("open")) |
                (pl.col("high") < pl.col("close")) |
                (pl.col("low") > pl.col("open")) |
                (pl.col("low") > pl.col("close"))
            )

            for row in invalid_ohlc.iter_rows(named=True):
                anomaly = DataAnomaly(
                    symbol=row["symbol"],
                    anomaly_type=AnomalyType.INVALID_OHLC,
                    severity=AnomalySeverity.HIGH,
                    timestamp=row["timestamp"],
                    timeframe=TimeFrame(row["timeframe"]),
                    description="Invalid OHLC relationship detected",
                    details={
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"]
                    },
                    suggested_action="Correct OHLC values or exclude record"
                )
                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Basic integrity validation failed: {e}")

        return anomalies

    async def _validate_price_ranges(self, df: pl.DataFrame) -> List[DataAnomaly]:
        """Validate price ranges and detect extreme values."""
        anomalies = []

        try:
            # Check for prices outside valid range
            invalid_prices = df.filter(
                (pl.col("open") < self.config.min_price) |
                (pl.col("high") < self.config.min_price) |
                (pl.col("low") < self.config.min_price) |
                (pl.col("close") < self.config.min_price) |
                (pl.col("open") > self.config.max_price) |
                (pl.col("high") > self.config.max_price) |
                (pl.col("low") > self.config.max_price) |
                (pl.col("close") > self.config.max_price)
            )

            for row in invalid_prices.iter_rows(named=True):
                price_values = [row["open"], row["high"], row["low"], row["close"]]
                min_price = min(price_values)
                max_price = max(price_values)

                severity = AnomalySeverity.CRITICAL if min_price <= 0 else AnomalySeverity.HIGH

                anomaly = DataAnomaly(
                    symbol=row["symbol"],
                    anomaly_type=AnomalyType.PRICE_SPIKE,
                    severity=severity,
                    timestamp=row["timestamp"],
                    timeframe=TimeFrame(row["timeframe"]),
                    description=f"Price outside valid range: ${min_price:.2f} - ${max_price:.2f}",
                    details={
                        "min_price": min_price,
                        "max_price": max_price,
                        "valid_range": f"${self.config.min_price} - ${self.config.max_price}"
                    },
                    suggested_action="Verify price data accuracy"
                )
                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Price range validation failed: {e}")

        return anomalies

    async def _validate_volume_data(self, df: pl.DataFrame) -> List[DataAnomaly]:
        """Validate volume data for anomalies."""
        anomalies = []

        try:
            # Check for zero volume during market hours
            zero_volume = df.filter(pl.col("volume") == 0)

            for row in zero_volume.iter_rows(named=True):
                # Only flag as anomaly if during market hours
                timestamp = row["timestamp"]
                if self._is_market_hours(timestamp):
                    anomaly = DataAnomaly(
                        symbol=row["symbol"],
                        anomaly_type=AnomalyType.ZERO_VOLUME,
                        severity=AnomalySeverity.MEDIUM,
                        timestamp=timestamp,
                        timeframe=TimeFrame(row["timeframe"]),
                        description="Zero volume during market hours",
                        details={"timestamp": timestamp.isoformat()},
                        suggested_action="Verify trading halt or data source issue"
                    )
                    anomalies.append(anomaly)

            # Check for negative volume
            negative_volume = df.filter(pl.col("volume") < 0)

            for row in negative_volume.iter_rows(named=True):
                anomaly = DataAnomaly(
                    symbol=row["symbol"],
                    anomaly_type=AnomalyType.MISSING_DATA,
                    severity=AnomalySeverity.CRITICAL,
                    timestamp=row["timestamp"],
                    timeframe=TimeFrame(row["timeframe"]),
                    description=f"Negative volume detected: {row['volume']}",
                    details={"volume": row["volume"]},
                    suggested_action="Correct volume data"
                )
                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Volume validation failed: {e}")

        return anomalies

    async def _detect_data_gaps(self, df: pl.DataFrame) -> List[DataAnomaly]:
        """Detect missing data gaps."""
        anomalies = []

        if len(df) < 2:
            return anomalies

        try:
            # Group by symbol and timeframe
            for (symbol, timeframe), group_df in df.group_by(["symbol", "timeframe"]):
                symbol_str = str(symbol)
                tf_enum = TimeFrame(timeframe)

                # Expected intervals
                expected_intervals = {
                    TimeFrame.FIVE_MINUTES: timedelta(minutes=5),
                    TimeFrame.FIFTEEN_MINUTES: timedelta(minutes=15),
                    TimeFrame.ONE_HOUR: timedelta(hours=1),
                    TimeFrame.ONE_DAY: timedelta(days=1)
                }

                expected_interval = expected_intervals.get(tf_enum)
                if not expected_interval:
                    continue

                # Sort by timestamp
                sorted_data = group_df.sort("timestamp")
                timestamps = sorted_data["timestamp"].to_list()

                # Check for gaps
                for i in range(1, len(timestamps)):
                    gap = timestamps[i] - timestamps[i-1]
                    max_allowed_gap = expected_interval * self.config.max_gap_multiplier

                    if gap > max_allowed_gap:
                        # Account for weekends and market hours
                        if not self._is_reasonable_gap(timestamps[i-1], timestamps[i], tf_enum):
                            anomaly = DataAnomaly(
                                symbol=symbol_str,
                                anomaly_type=AnomalyType.MISSING_DATA,
                                severity=AnomalySeverity.MEDIUM,
                                timestamp=timestamps[i],
                                timeframe=tf_enum,
                                description=f"Data gap detected: {gap} (expected: {expected_interval})",
                                details={
                                    "gap_duration": str(gap),
                                    "expected_interval": str(expected_interval),
                                    "gap_start": timestamps[i-1].isoformat(),
                                    "gap_end": timestamps[i].isoformat()
                                },
                                suggested_action="Backfill missing data"
                            )
                            anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Data gap detection failed: {e}")

        return anomalies

    async def _detect_price_anomalies(self, df: pl.DataFrame) -> List[DataAnomaly]:
        """Detect price anomalies using statistical methods."""
        anomalies = []

        try:
            # Group by symbol for analysis
            for symbol, group_df in df.group_by("symbol"):
                symbol_str = str(symbol)
                if len(group_df) < self.config.volatility_lookback_periods:
                    continue

                sorted_data = group_df.sort("timestamp")

                # Calculate returns
                returns = sorted_data.with_columns(
                    (pl.col("close").pct_change().alias("returns"))
                ).drop_nulls()

                if len(returns) < 10:
                    continue

                # Statistical analysis
                return_values = returns["returns"].to_list()
                mean_return = statistics.mean(return_values)
                std_return = statistics.stdev(return_values) if len(return_values) > 1 else 0

                # Detect outliers
                outlier_threshold = self.config.outlier_std_threshold

                for i, ret in enumerate(return_values):
                    if abs(ret - mean_return) > outlier_threshold * std_return:
                        timestamp = returns["timestamp"][i]
                        price = returns["close"][i]

                        anomaly = DataAnomaly(
                            symbol=symbol_str,
                            anomaly_type=AnomalyType.PRICE_SPIKE,
                            severity=AnomalySeverity.HIGH,
                            timestamp=timestamp,
                            timeframe=TimeFrame(returns["timeframe"][i]),
                            description=f"Statistical price anomaly: {ret:.1%} return ({outlier_threshold:.1f}σ outlier)",
                            details={
                                "return_percent": ret * 100,
                                "z_score": (ret - mean_return) / std_return if std_return > 0 else 0,
                                "price": float(price),
                                "mean_return": mean_return,
                                "std_return": std_return
                            },
                            suggested_action="Investigate potential market event or data error",
                            confidence_score=min(0.9, abs(ret - mean_return) / (outlier_threshold * std_return))
                        )
                        anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Price anomaly detection failed: {e}")

        return anomalies

    async def _detect_volume_anomalies(self, df: pl.DataFrame) -> List[DataAnomaly]:
        """Detect volume anomalies and spikes."""
        anomalies = []

        try:
            # Group by symbol for analysis
            for symbol, group_df in df.group_by("symbol"):
                symbol_str = str(symbol)
                if len(group_df) < 10:
                    continue

                sorted_data = group_df.sort("timestamp")
                volumes = sorted_data["volume"].to_list()

                # Calculate average volume (excluding zeros)
                non_zero_volumes = [v for v in volumes if v > 0]
                if len(non_zero_volumes) < 5:
                    continue

                avg_volume = statistics.mean(non_zero_volumes)

                # Detect volume spikes
                for i, volume in enumerate(volumes):
                    if volume > 0 and volume > avg_volume * self.config.max_volume_spike_ratio:
                        timestamp = sorted_data["timestamp"][i]

                        anomaly = DataAnomaly(
                            symbol=symbol_str,
                            anomaly_type=AnomalyType.VOLUME_SPIKE,
                            severity=AnomalySeverity.MEDIUM,
                            timestamp=timestamp,
                            timeframe=TimeFrame(sorted_data["timeframe"][i]),
                            description=f"Volume spike: {volume:,} ({volume/avg_volume:.1f}x average)",
                            details={
                                "volume": volume,
                                "average_volume": avg_volume,
                                "spike_ratio": volume / avg_volume
                            },
                            suggested_action="Investigate news or events causing volume spike",
                            confidence_score=min(0.9, (volume / avg_volume) / self.config.max_volume_spike_ratio)
                        )
                        anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Volume anomaly detection failed: {e}")

        return anomalies

    async def _validate_data_freshness(self, df: pl.DataFrame) -> List[DataAnomaly]:
        """Validate data freshness and detect stale data."""
        anomalies = []

        try:
            current_time = datetime.now(timezone.utc)
            freshness_threshold = timedelta(minutes=self.config.max_data_age_minutes)

            # Check each record's age
            for row in df.iter_rows(named=True):
                data_age = current_time - row["timestamp"]

                if data_age > freshness_threshold:
                    # Only flag as stale if it should be fresh (market hours for intraday data)
                    timeframe = TimeFrame(row["timeframe"])

                    if timeframe in [TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES] and self._is_market_hours(current_time):
                        anomaly = DataAnomaly(
                            symbol=row["symbol"],
                            anomaly_type=AnomalyType.STALE_DATA,
                            severity=AnomalySeverity.MEDIUM,
                            timestamp=row["timestamp"],
                            timeframe=timeframe,
                            description=f"Stale data: {data_age} old",
                            details={
                                "data_age_minutes": data_age.total_seconds() / 60,
                                "threshold_minutes": self.config.max_data_age_minutes
                            },
                            suggested_action="Update with fresh data"
                        )
                        anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Data freshness validation failed: {e}")

        return anomalies

    async def _clean_data(
        self,
        original_data: List[MarketData],
        anomalies: List[DataAnomaly]
    ) -> List[MarketData]:
        """
        Clean data based on detected anomalies.

        Args:
            original_data: Original data list
            anomalies: Detected anomalies

        Returns:
            Cleaned data list
        """
        if not anomalies:
            return original_data

        try:
            # Create set of timestamps to exclude
            exclude_timestamps = set()

            for anomaly in anomalies:
                if anomaly.severity in [AnomalySeverity.CRITICAL, AnomalySeverity.HIGH]:
                    if anomaly.anomaly_type in [
                        AnomalyType.INVALID_OHLC,
                        AnomalyType.MISSING_DATA
                    ]:
                        exclude_timestamps.add(anomaly.timestamp)

            # Filter out problematic records
            cleaned_data = [
                data for data in original_data
                if data.timestamp not in exclude_timestamps
            ]

            logger.info(f"Cleaned data: removed {len(original_data) - len(cleaned_data)} problematic records")

            return cleaned_data

        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return original_data

    def _to_dataframe(self, data: List[MarketData]) -> pl.DataFrame:
        """Convert MarketData list to Polars DataFrame."""
        if not data:
            return pl.DataFrame()

        records = []
        for md in data:
            records.append({
                "symbol": md.symbol,
                "timestamp": md.timestamp,
                "timeframe": md.timeframe.value,
                "open": float(md.open),
                "high": float(md.high),
                "low": float(md.low),
                "close": float(md.close),
                "volume": md.volume,
                "asset_type": md.asset_type.value
            })

        return pl.DataFrame(records)

    def _is_market_hours(self, timestamp: datetime) -> bool:
        """Check if timestamp is during market hours."""
        # Simplified market hours check (9:30 AM - 4:00 PM ET, weekdays)
        if timestamp.weekday() >= 5:  # Weekend
            return False

        market_time = timestamp.time()
        return dt_time(9, 30) <= market_time <= dt_time(16, 0)

    def _is_reasonable_gap(
        self,
        start_time: datetime,
        end_time: datetime,
        timeframe: TimeFrame
    ) -> bool:
        """Check if a data gap is reasonable (weekends, holidays, etc.)."""
        gap = end_time - start_time

        # For daily data, gaps over weekends are normal
        if timeframe == TimeFrame.ONE_DAY:
            # Friday to Monday is normal (3 days)
            if start_time.weekday() == 4 and end_time.weekday() == 0:
                return gap <= timedelta(days=4)  # Allow for holidays
            # Normal weekday gap
            return gap <= timedelta(days=2)

        # For intraday data, gaps outside market hours are normal
        if timeframe in [TimeFrame.FIVE_MINUTES, TimeFrame.FIFTEEN_MINUTES, TimeFrame.ONE_HOUR]:
            # If gap spans non-market hours, it's reasonable
            if not self._is_market_hours(start_time) or not self._is_market_hours(end_time):
                return True
            # Weekend gaps are normal
            if start_time.weekday() >= 5 or end_time.weekday() >= 5:
                return True

        return False
