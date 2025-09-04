"""
Simplified Data Store for Backtesting

This module provides a simplified data store implementation for backtesting
that avoids configuration dependencies and focuses only on reading historical
data from parquet files.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List

import polars as pl
from backtest_models import TimeFrame

logger = logging.getLogger(__name__)


class SimpleDataStore:
    """
    Simplified data store for backtesting that reads parquet files
    without complex configuration dependencies.
    """

    def __init__(self, base_path: str = "data/parquet"):
        """
        Initialize the simple data store.

        Args:
            base_path: Base path to the parquet data directory
        """
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(f"{__name__}.SimpleDataStore")

        # Create basic schema definitions
        self.market_data_schema = {
            "symbol": pl.Utf8,
            "timestamp": pl.Datetime,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "adjusted_close": pl.Float64,
            "volume": pl.Int64,
            "timeframe": pl.Utf8,
        }

    def _get_market_data_file_path(
        self, ticker: str, timeframe: TimeFrame, date_obj: date
    ) -> Path:
        """Get file path for market data."""
        timeframe_str = timeframe.value
        ticker_dir = self.base_path / "market_data" / ticker / timeframe_str
        filename = f"{date_obj.strftime('%Y-%m-%d')}.parquet"
        return ticker_dir / filename

    def _get_screener_file_path(self, screener_type: str, date_obj: date) -> Path:
        """Get file path for screener data."""
        screener_dir = self.base_path / "screener_data" / screener_type
        filename = f"{date_obj.strftime('%Y-%m-%d')}.parquet"
        return screener_dir / filename

    async def load_market_data(
        self, ticker: str, timeframe: TimeFrame, start_date: date, end_date: date
    ) -> pl.DataFrame:
        """
        Load market data for a ticker within a date range.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            start_date: Start date
            end_date: End date

        Returns:
            Polars DataFrame with market data
        """
        try:
            # Collect all files in the date range
            all_dataframes = []
            current_date = start_date

            while current_date <= end_date:
                file_path = self._get_market_data_file_path(
                    ticker, timeframe, current_date
                )

                if file_path.exists():
                    try:
                        # Load the parquet file
                        df = await self._load_parquet_file(file_path)
                        if not df.is_empty():
                            all_dataframes.append(df)

                    except Exception as e:
                        self.logger.debug(f"Could not load {file_path}: {e}")

                # Move to next day
                current_date = current_date + timedelta(days=1)

            # Combine all dataframes
            if all_dataframes:
                combined_df = pl.concat(all_dataframes)
                # Sort by timestamp and remove duplicates
                result_df = combined_df.sort("timestamp").unique(
                    subset=["timestamp"], keep="last"
                )
                return result_df
            else:
                # Return empty dataframe with correct schema
                return pl.DataFrame([], schema=self.market_data_schema)

        except Exception as e:
            self.logger.error(f"Error loading market data for {ticker}: {e}")
            return pl.DataFrame([], schema=self.market_data_schema)

    async def load_screener_data(
        self, screener_type: str, start_date: date, end_date: date
    ) -> pl.DataFrame:
        """
        Load screener data for a date range.

        Args:
            screener_type: Type of screener (momentum, breakouts, etc.)
            start_date: Start date
            end_date: End date

        Returns:
            Polars DataFrame with screener data
        """
        try:
            all_dataframes = []
            current_date = start_date

            while current_date <= end_date:
                file_path = self._get_screener_file_path(screener_type, current_date)

                if file_path.exists():
                    try:
                        df = await self._load_parquet_file(file_path)
                        if not df.is_empty():
                            all_dataframes.append(df)

                    except Exception as e:
                        self.logger.debug(
                            f"Could not load screener file {file_path}: {e}"
                        )

                # Move to next day
                current_date = current_date + timedelta(days=1)

            # Combine all dataframes
            if all_dataframes:
                combined_df = pl.concat(all_dataframes)
                return combined_df.sort("timestamp").unique(
                    subset=["symbol", "timestamp"], keep="last"
                )
            else:
                # Return empty dataframe
                return pl.DataFrame([])

        except Exception as e:
            self.logger.error(f"Error loading screener data for {screener_type}: {e}")
            return pl.DataFrame([])

    async def _load_parquet_file(self, file_path: Path) -> pl.DataFrame:
        """Load a parquet file asynchronously."""

        def _load_file():
            try:
                return pl.read_parquet(file_path)
            except Exception as e:
                logger.error(f"Failed to read parquet file {file_path}: {e}")
                return pl.DataFrame([])

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _load_file)

    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in the data store."""
        try:
            market_data_path = self.base_path / "market_data"
            if not market_data_path.exists():
                return []

            symbols = []
            for symbol_dir in market_data_path.iterdir():
                if symbol_dir.is_dir():
                    symbols.append(symbol_dir.name)

            return sorted(symbols)

        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []

    def get_available_timeframes(self, symbol: str) -> List[TimeFrame]:
        """Get list of available timeframes for a symbol."""
        try:
            symbol_path = self.base_path / "market_data" / symbol
            if not symbol_path.exists():
                return []

            timeframes = []
            for tf_dir in symbol_path.iterdir():
                if tf_dir.is_dir():
                    try:
                        # Try to match with TimeFrame enum
                        tf = TimeFrame(tf_dir.name)
                        timeframes.append(tf)
                    except ValueError:
                        continue

            return timeframes

        except Exception as e:
            self.logger.error(f"Error getting timeframes for {symbol}: {e}")
            return []

    def get_date_range_for_symbol(self, symbol: str, timeframe: TimeFrame) -> tuple:
        """Get the available date range for a symbol and timeframe."""
        try:
            symbol_tf_path = self.base_path / "market_data" / symbol / timeframe.value
            if not symbol_tf_path.exists():
                return None, None

            dates = []
            for file_path in symbol_tf_path.glob("*.parquet"):
                try:
                    # Extract date from filename (YYYY-MM-DD.parquet)
                    date_str = file_path.stem
                    file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    dates.append(file_date)
                except ValueError:
                    continue

            if dates:
                return min(dates), max(dates)
            else:
                return None, None

        except Exception as e:
            self.logger.error(f"Error getting date range for {symbol}: {e}")
            return None, None

    async def close(self):
        """Close the data store (placeholder for compatibility)."""
        pass
