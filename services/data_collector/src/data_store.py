"""
Data storage and retrieval using Polars and Parquet files.

This module provides efficient data storage and retrieval capabilities using
Polars for data processing and Parquet for storage format. Organizes data
by ticker, timeframe, and date for optimal performance.
"""

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
from polars import DataType
from pydantic import BaseModel, Field

from shared.models import (
    AssetType,
    FinVizData,
    MarketData,
    TimeFrame,
)

logger = logging.getLogger(__name__)


class DataStoreConfig(BaseModel):
    """Configuration for data storage."""

    base_path: str = Field(
        default="data/parquet", description="Base path for Parquet files"
    )
    compression: str = Field(default="snappy", description="Compression algorithm")
    max_file_size_mb: int = Field(default=100, description="Maximum file size in MB")
    retention_days: int = Field(
        default=730, description="Data retention period in days"
    )
    enable_compression: bool = Field(
        default=True, description="Enable Parquet compression"
    )
    batch_size: int = Field(default=10000, description="Batch size for data operations")
    max_workers: int = Field(default=4, description="Maximum worker threads for I/O")


class DataStore:
    """
    High-performance data storage using Polars and Parquet files.

    Organizes data as: data/parquet/{ticker}/{timeframe}/{date}.parquet
    Provides efficient data updates, retrieval, and management.
    """

    def __init__(self, config: DataStoreConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self._ensure_base_directory()
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._write_lock = threading.Lock()

        # Schema definitions for validation
        self.market_data_schema: Dict[str, DataType] = {
            "symbol": pl.Utf8(),
            "timestamp": pl.Datetime("us"),
            "timeframe": pl.Utf8(),
            "open": pl.Float64(),
            "high": pl.Float64(),
            "low": pl.Float64(),
            "close": pl.Float64(),
            "volume": pl.Int64(),
            "asset_type": pl.Utf8(),
        }

        self.finviz_data_schema: Dict[str, DataType] = {
            "symbol": pl.Utf8(),
            "company": pl.Utf8(),
            "sector": pl.Utf8(),
            "industry": pl.Utf8(),
            "country": pl.Utf8(),
            "market_cap": pl.Utf8(),
            "pe": pl.Float64(),
            "price": pl.Float64(),
            "change": pl.Float64(),
            "volume": pl.Int64(),
        }

        self.technical_indicators_schema: Dict[str, DataType] = {
            "symbol": pl.Utf8(),
            "timestamp": pl.Datetime("us"),
            "timeframe": pl.Utf8(),
            "sma_20": pl.Float64(),
            "sma_50": pl.Float64(),
            "sma_200": pl.Float64(),
            "ema_12": pl.Float64(),
            "ema_26": pl.Float64(),
            "rsi": pl.Float64(),
            "macd": pl.Float64(),
            "macd_signal": pl.Float64(),
            "macd_histogram": pl.Float64(),
            "bollinger_upper": pl.Float64(),
            "bollinger_middle": pl.Float64(),
            "bollinger_lower": pl.Float64(),
            "atr": pl.Float64(),
            "volume_sma": pl.Float64(),
        }

    def _ensure_base_directory(self):
        """Ensure base directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different data types
        (self.base_path / "market_data").mkdir(exist_ok=True)
        (self.base_path / "screener_data").mkdir(exist_ok=True)
        (self.base_path / "technical_indicators").mkdir(exist_ok=True)

    def _get_file_path(
        self,
        ticker: str,
        timeframe: TimeFrame,
        date_obj: date,
        data_type: str = "market_data",
    ) -> Path:
        """
        Get file path for ticker/timeframe/date combination.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            date_obj: Date for the data
            data_type: Type of data (market_data, screener_data, etc.)

        Returns:
            Path object for the Parquet file
        """
        ticker_dir = self.base_path / data_type / ticker.upper() / timeframe.value
        ticker_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{date_obj.strftime('%Y-%m-%d')}.parquet"
        return ticker_dir / filename

    def _get_screener_file_path(
        self, date_obj: date, screener_type: str = "momentum"
    ) -> Path:
        """Get file path for screener data."""
        screener_dir = self.base_path / "screener_data" / screener_type
        screener_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{date_obj.strftime('%Y-%m-%d')}.parquet"
        return screener_dir / filename

    def _market_data_to_dataframe(self, data: List[MarketData]) -> pl.DataFrame:
        """Convert MarketData list to Polars DataFrame."""
        if not data:
            return pl.DataFrame(schema=self.market_data_schema)

        # Convert to list of dictionaries
        records = []
        for md in data:
            records.append(
                {
                    "symbol": md.symbol,
                    "timestamp": md.timestamp,
                    "timeframe": md.timeframe.value,
                    "open": float(md.open),
                    "high": float(md.high),
                    "low": float(md.low),
                    "close": float(md.close),
                    "volume": md.volume,
                    "asset_type": md.asset_type.value,
                }
            )

        return pl.DataFrame(records, schema=self.market_data_schema)

    def _finviz_data_to_dataframe(self, data: List[FinVizData]) -> pl.DataFrame:
        """Convert FinVizData list to Polars DataFrame."""
        if not data:
            return pl.DataFrame(schema=self.finviz_data_schema)

        records = []
        for i, fv in enumerate(data):
            try:
                record = {
                    "symbol": fv.symbol,
                    "company": fv.company or "",
                    "sector": fv.sector or "",
                    "industry": fv.industry or "",
                    "country": fv.country or "",
                    "market_cap": (
                        str(fv.market_cap) if fv.market_cap is not None else ""
                    ),
                    "pe_ratio": float(fv.pe_ratio) if fv.pe_ratio is not None else None,
                    "price": float(fv.price) if fv.price is not None else None,
                    "change": float(fv.change) if fv.change is not None else None,
                    "volume": int(fv.volume) if fv.volume is not None else None,
                    "timestamp": fv.timestamp,
                }
                records.append(record)

                # Log first record for debugging
                if i == 0:
                    logger.debug(
                        f"First record data types: {[(k, type(v).__name__) for k, v in record.items()]}"
                    )
                    logger.debug(f"First record values: {record}")

            except Exception as e:
                logger.error(f"Error processing FinVizData record {i}: {e}")
                logger.error(
                    f"Problem record: symbol={fv.symbol}, price={fv.price}, change={fv.change}, volume={fv.volume}"
                )
                continue

        try:
            logger.info(f"Creating DataFrame with {len(records)} records")
            logger.debug(f"Schema: {self.finviz_data_schema}")

            # Try creating without schema first to see what Polars infers
            df_test = pl.DataFrame(records)
            logger.debug(f"Inferred dtypes: {df_test.dtypes}")

            # Now create with schema
            df = pl.DataFrame(records, schema=self.finviz_data_schema)
            logger.info(f"Successfully created DataFrame with shape {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Failed to create Polars DataFrame: {e}")
            logger.error(f"Number of records: {len(records)}")
            if records:
                logger.error(f"Sample record: {records[0]}")

            # Try without schema as fallback
            try:
                logger.info("Attempting to create DataFrame without schema...")
                df = pl.DataFrame(records)
                logger.info(
                    f"Successfully created DataFrame without schema, shape: {df.shape}"
                )
                return df
            except Exception as e2:
                logger.error(f"Also failed without schema: {e2}")
                raise

    def _technical_indicators_to_dataframe(self, indicators_list: List) -> pl.DataFrame:
        """Convert TechnicalIndicators list to Polars DataFrame."""
        if not indicators_list:
            return pl.DataFrame(schema=self.technical_indicators_schema)

        records = []
        for indicator in indicators_list:
            try:
                # Handle both dict and object formats
                if hasattr(indicator, "__dict__"):
                    # Pydantic model or object
                    data = (
                        indicator.dict()
                        if hasattr(indicator, "dict")
                        else indicator.__dict__
                    )
                else:
                    # Dictionary
                    data = indicator

                record = {
                    "symbol": data.get("symbol", ""),
                    "timestamp": data.get("timestamp"),
                    "timeframe": data.get("timeframe", "1d"),
                    "sma_20": (
                        float(data.get("sma_20"))
                        if data.get("sma_20") is not None
                        else None
                    ),
                    "sma_50": (
                        float(data.get("sma_50"))
                        if data.get("sma_50") is not None
                        else None
                    ),
                    "sma_200": (
                        float(data.get("sma_200"))
                        if data.get("sma_200") is not None
                        else None
                    ),
                    "ema_12": (
                        float(data.get("ema_12"))
                        if data.get("ema_12") is not None
                        else None
                    ),
                    "ema_26": (
                        float(data.get("ema_26"))
                        if data.get("ema_26") is not None
                        else None
                    ),
                    "rsi": (
                        float(data.get("rsi")) if data.get("rsi") is not None else None
                    ),
                    "macd": (
                        float(data.get("macd"))
                        if data.get("macd") is not None
                        else None
                    ),
                    "macd_signal": (
                        float(data.get("macd_signal"))
                        if data.get("macd_signal") is not None
                        else None
                    ),
                    "macd_histogram": (
                        float(data.get("macd_histogram"))
                        if data.get("macd_histogram") is not None
                        else None
                    ),
                    "bollinger_upper": (
                        float(data.get("bollinger_upper"))
                        if data.get("bollinger_upper") is not None
                        else None
                    ),
                    "bollinger_middle": (
                        float(data.get("bollinger_middle"))
                        if data.get("bollinger_middle") is not None
                        else None
                    ),
                    "bollinger_lower": (
                        float(data.get("bollinger_lower"))
                        if data.get("bollinger_lower") is not None
                        else None
                    ),
                    "atr": (
                        float(data.get("atr")) if data.get("atr") is not None else None
                    ),
                    "volume_sma": (
                        float(data.get("volume_sma"))
                        if data.get("volume_sma") is not None
                        else None
                    ),
                }
                records.append(record)

            except Exception as e:
                logger.error(f"Error processing TechnicalIndicators record: {e}")
                logger.error(f"Problem record: {indicator}")
                continue

        try:
            df = pl.DataFrame(records, schema=self.technical_indicators_schema)
            logger.debug(
                f"Successfully created technical indicators DataFrame with shape {df.shape}"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to create technical indicators DataFrame: {e}")
            logger.error(f"Number of records: {len(records)}")
            if records:
                logger.error(f"Sample record: {records[0]}")

            # Try without schema as fallback
            try:
                df = pl.DataFrame(records)
                logger.info(
                    f"Created technical indicators DataFrame without schema, shape: {df.shape}"
                )
                return df
            except Exception as e2:
                logger.error(f"Also failed without schema: {e2}")
                raise

    async def save_market_data(
        self, data: List[MarketData], append: bool = True
    ) -> Dict[str, int]:
        """
        Save market data to Parquet files organized by ticker/timeframe/date.

        Args:
            data: List of MarketData objects
            append: Whether to append to existing files or overwrite

        Returns:
            Dictionary with save statistics
        """
        if not data:
            return {"total_saved": 0, "files_created": 0, "files_updated": 0}

        # Convert to Polars DataFrame
        df = self._market_data_to_dataframe(data)

        # Group by ticker, timeframe, and date
        stats = {"total_saved": 0, "files_created": 0, "files_updated": 0}

        # Create date column for grouping
        df_with_date = df.with_columns(pl.col("timestamp").dt.date().alias("date"))

        # Group by symbol, timeframe, and date
        grouped_data: Dict[Tuple[Any, Any, Any], Any] = {}
        for row in df_with_date.iter_rows(named=True):
            key = (row["symbol"], row["timeframe"], row["date"])
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(row)

        # Process each group
        tasks = []
        for key_tuple, group_rows in grouped_data.items():
            symbol, timeframe_str, date_obj = key_tuple
            try:
                timeframe = TimeFrame(timeframe_str)
            except ValueError:
                logger.warning(f"Unknown timeframe: {timeframe_str}")
                continue

            # Create DataFrame for this group (without date column)
            group_data = []
            for row in group_rows:
                group_data.append(
                    {
                        "symbol": row["symbol"],
                        "timestamp": row["timestamp"],
                        "timeframe": row["timeframe"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "asset_type": row["asset_type"],
                    }
                )

            group_df = pl.DataFrame(group_data, schema=self.market_data_schema)

            # Schedule save task
            task = self._save_dataframe_to_file(
                group_df, symbol, timeframe, date_obj, append, "market_data"
            )
            tasks.append(task)

        # Execute all save tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate statistics
        for result in results:
            if isinstance(result, dict):
                stats["total_saved"] += result.get("records_saved", 0)
                if result.get("file_created", False):
                    stats["files_created"] += 1
                elif result.get("file_updated", False):
                    stats["files_updated"] += 1
            elif isinstance(result, Exception):
                logger.error(f"Save task failed: {result}")

        logger.info(
            f"Saved {stats['total_saved']} records to {stats['files_created']} new files and updated {stats['files_updated']} existing files"
        )
        return stats

    async def save_screener_data(
        self,
        data: List[FinVizData],
        screener_type: str = "momentum",
        append: bool = True,
    ) -> int:
        """
        Save screener data to Parquet files.

        Args:
            data: List of FinVizData objects
            screener_type: Type of screener data
            append: Whether to append to existing files

        Returns:
            Number of records saved
        """
        if not data:
            return 0

        try:
            logger.info(
                f"Attempting to save {len(data)} {screener_type} screener records"
            )

            # Convert to Polars DataFrame
            df = self._finviz_data_to_dataframe(data)

            if df is None or df.is_empty():
                logger.warning("DataFrame is empty after conversion")
                return 0

            # Group by date
            today = date.today()
            file_path = self._get_screener_file_path(today, screener_type)

            await self._save_dataframe_to_parquet(df, file_path, append)

            logger.info(
                f"Successfully saved {len(data)} screener records to {file_path}"
            )
            return len(data)

        except Exception as e:
            logger.error(f"Failed to save screener data: {e}")
            logger.error(f"Data sample: {data[0] if data else 'No data'}")
            # Don't crash the entire service, just skip saving
            return 0

    async def save_technical_indicators(
        self, indicators_list: List, timeframe: str = "1d", append: bool = True
    ) -> int:
        """
        Save technical indicators to Parquet files organized by symbol/timeframe/date.

        Args:
            indicators_list: List of TechnicalIndicators objects or dicts
            timeframe: Timeframe for the indicators
            append: Whether to append to existing files

        Returns:
            Number of records saved
        """
        if not indicators_list:
            return 0

        try:
            logger.info(
                f"Attempting to save {len(indicators_list)} technical indicators records"
            )

            # Convert to Polars DataFrame
            df = self._technical_indicators_to_dataframe(indicators_list)

            if df is None or df.is_empty():
                logger.warning("DataFrame is empty after conversion")
                return 0

            # Add timeframe if not present
            if "timeframe" not in df.columns or df["timeframe"].is_null().all():
                df = df.with_columns(pl.lit(timeframe).alias("timeframe"))

            # Group by symbol and date
            df_with_date = df.with_columns(pl.col("timestamp").dt.date().alias("date"))

            # Group by symbol, timeframe, and date
            grouped_data: Dict[Tuple[Any, Any, Any], Any] = {}
            for row in df_with_date.iter_rows(named=True):
                key = (row["symbol"], row["timeframe"], row["date"])
                if key not in grouped_data:
                    grouped_data[key] = []
                grouped_data[key].append(row)

            total_saved = 0

            # Process each group
            for key_tuple, group_rows in grouped_data.items():
                symbol, tf_str, date_obj = key_tuple
                try:
                    # Create DataFrame for this group (without date column)
                    group_data = []
                    for row in group_rows:
                        group_data.append({k: v for k, v in row.items() if k != "date"})

                    group_df = pl.DataFrame(
                        group_data, schema=self.technical_indicators_schema
                    )

                    # Get file path for technical indicators
                    file_path = self._get_technical_indicators_file_path(
                        symbol, tf_str, date_obj
                    )

                    # Save to file
                    await self._save_dataframe_to_parquet(group_df, file_path, append)
                    total_saved += len(group_data)

                    logger.debug(
                        f"Saved {len(group_data)} indicators for {symbol} {tf_str} {date_obj}"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to save indicators group {symbol} {tf_str} {date_obj}: {e}"
                    )
                    continue

            logger.info(
                f"Successfully saved {total_saved} technical indicators records"
            )
            return total_saved

        except Exception as e:
            logger.error(f"Failed to save technical indicators: {e}")
            return 0

    def _get_technical_indicators_file_path(
        self, symbol: str, timeframe: str, date_obj: date
    ) -> Path:
        """Get file path for technical indicators data."""
        indicators_dir = self.base_path / "technical_indicators" / symbol / timeframe
        indicators_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{date_obj.isoformat()}.parquet"
        return indicators_dir / filename

    async def _save_dataframe_to_file(
        self,
        df: pl.DataFrame,
        ticker: str,
        timeframe: TimeFrame,
        date_obj: date,
        append: bool,
        data_type: str,
    ) -> Dict[str, Any]:
        """
        Save DataFrame to specific file with deduplication.

        Args:
            df: Polars DataFrame to save
            ticker: Stock ticker
            timeframe: Data timeframe
            date_obj: Date for the data
            append: Whether to append or overwrite
            data_type: Type of data being saved

        Returns:
            Save operation statistics
        """
        file_path = self._get_file_path(ticker, timeframe, date_obj, data_type)
        file_existed = file_path.exists()

        if append and file_existed:
            # Load existing data and deduplicate
            try:
                existing_df = await self._load_parquet_file(file_path)

                # Combine and deduplicate based on timestamp
                combined_df = pl.concat([existing_df, df])
                deduplicated_df = combined_df.unique(subset=["timestamp"], keep="last")

                # Sort by timestamp
                final_df = deduplicated_df.sort("timestamp")

            except Exception as e:
                logger.warning(
                    f"Failed to read existing file {file_path}: {e}, overwriting"
                )
                final_df = df.sort("timestamp")
        else:
            final_df = df.sort("timestamp")

        # Save to Parquet
        await self._save_dataframe_to_parquet(final_df, file_path, append=False)

        return {
            "records_saved": len(final_df),
            "file_created": not file_existed,
            "file_updated": file_existed,
            "file_path": str(file_path),
        }

    async def _save_dataframe_to_parquet(
        self, df: pl.DataFrame, file_path: Path, append: bool = False
    ) -> bool:
        """
        Save DataFrame to Parquet file asynchronously.

        Args:
            df: DataFrame to save
            file_path: Target file path
            append: Whether to append (not used, kept for compatibility)

        Returns:
            True if successful
        """

        def _write_file():
            with self._write_lock:
                try:
                    file_path.parent.mkdir(parents=True, exist_ok=True)

                    if self.config.enable_compression:
                        df.write_parquet(
                            file_path,
                            compression=self.config.compression,  # type: ignore
                            use_pyarrow=True,
                        )
                    else:
                        df.write_parquet(file_path, use_pyarrow=True)
                    return True
                except Exception as e:
                    logger.error(f"Failed to write Parquet file {file_path}: {e}")
                    return False

        # Execute in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _write_file)

    async def load_market_data(
        self,
        ticker: str,
        timeframe: TimeFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Load market data from Parquet files.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            start_date: Start date for data range
            end_date: End date for data range
            limit: Maximum number of records to return

        Returns:
            Polars DataFrame with market data
        """
        ticker_dir = self.base_path / "market_data" / ticker.upper() / timeframe.value

        if not ticker_dir.exists():
            logger.warning(f"No data directory found for {ticker} {timeframe}")
            return pl.DataFrame(schema=self.market_data_schema)

        # Get date range
        if start_date is None:
            start_date = date.today() - timedelta(days=30)  # Default 30 days
        if end_date is None:
            end_date = date.today()

        # Find all relevant Parquet files
        file_paths = []
        current_date = start_date

        while current_date <= end_date:
            file_path = ticker_dir / f"{current_date.strftime('%Y-%m-%d')}.parquet"
            if file_path.exists():
                file_paths.append(file_path)
            current_date += timedelta(days=1)

        if not file_paths:
            logger.warning(
                f"No data files found for {ticker} {timeframe} between {start_date} and {end_date}"
            )
            return pl.DataFrame(schema=self.market_data_schema)

        # Load and combine data
        dataframes = []
        for file_path in file_paths:
            try:
                df = await self._load_parquet_file(file_path)
                if not df.is_empty():
                    dataframes.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not dataframes:
            return pl.DataFrame(schema=self.market_data_schema)

        # Combine all DataFrames
        combined_df = pl.concat(dataframes)

        # Filter by date range if needed
        combined_df = combined_df.filter(
            (pl.col("timestamp").dt.date() >= start_date)
            & (pl.col("timestamp").dt.date() <= end_date)
        )

        # Sort by timestamp
        combined_df = combined_df.sort("timestamp")

        # Apply limit if specified
        if limit:
            combined_df = combined_df.head(limit)

        return combined_df

    async def load_multiple_tickers(
        self,
        tickers: List[str],
        timeframe: TimeFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pl.DataFrame:
        """
        Load market data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            timeframe: Data timeframe
            start_date: Start date for data range
            end_date: End date for data range

        Returns:
            Combined Polars DataFrame with all ticker data
        """
        tasks = [
            self.load_market_data(ticker, timeframe, start_date, end_date)
            for ticker in tickers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_dataframes = []
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load data for {ticker}: {result}")
            elif isinstance(result, pl.DataFrame) and not result.is_empty():
                valid_dataframes.append(result)

        if not valid_dataframes:
            return pl.DataFrame(schema=self.market_data_schema)

        return pl.concat(valid_dataframes)

    async def _load_parquet_file(self, file_path: Path) -> pl.DataFrame:
        """Load Parquet file asynchronously."""

        def _read_file():
            try:
                return pl.read_parquet(file_path)
            except Exception as e:
                logger.error(f"Failed to read Parquet file {file_path}: {e}")
                return pl.DataFrame()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _read_file)

    async def get_latest_data(
        self, ticker: str, timeframe: TimeFrame, limit: int = 100
    ) -> pl.DataFrame:
        """
        Get latest data for a ticker and timeframe.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            limit: Maximum number of records to return

        Returns:
            Polars DataFrame with latest data
        """
        # Look for data in the last 7 days
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        df = await self.load_market_data(ticker, timeframe, start_date, end_date)

        if df.is_empty():
            return df

        # Return most recent records
        return df.sort("timestamp", descending=True).head(limit)

    async def get_data_summary(
        self, ticker: Optional[str] = None, timeframe: Optional[TimeFrame] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics about stored data.

        Args:
            ticker: Optional ticker filter
            timeframe: Optional timeframe filter

        Returns:
            Dictionary with data summary
        """

        def _scan_directory():
            summary: Dict[str, Any] = {
                "total_files": 0,
                "total_size_mb": 0,
                "tickers": set(),
                "timeframes": set(),
                "date_range": {"earliest": None, "latest": None},
                "files_by_ticker": {},
                "files_by_timeframe": {},
            }

            market_data_dir = self.base_path / "market_data"
            if not market_data_dir.exists():
                return summary

            for ticker_dir in market_data_dir.iterdir():
                if not ticker_dir.is_dir():
                    continue

                ticker_name = ticker_dir.name
                if ticker and ticker_name != ticker.upper():
                    continue

                ticker_set = summary["tickers"]
                assert isinstance(ticker_set, set), "tickers should be a set"
                ticker_set.add(ticker_name)
                files_by_ticker = summary["files_by_ticker"]
                assert isinstance(
                    files_by_ticker, dict
                ), "files_by_ticker should be a dict"
                files_by_ticker[ticker_name] = 0

                for tf_dir in ticker_dir.iterdir():
                    if not tf_dir.is_dir():
                        continue

                    tf_name = tf_dir.name
                    if timeframe and tf_name != timeframe.value:
                        continue

                    timeframe_set = summary["timeframes"]
                    assert isinstance(timeframe_set, set), "timeframes should be a set"
                    timeframe_set.add(tf_name)
                    files_by_timeframe = summary["files_by_timeframe"]
                    assert isinstance(
                        files_by_timeframe, dict
                    ), "files_by_timeframe should be a dict"
                    if tf_name not in files_by_timeframe:
                        files_by_timeframe[tf_name] = 0

                    for file_path in tf_dir.glob("*.parquet"):
                        total_files = summary["total_files"]
                        assert isinstance(
                            total_files, int
                        ), "total_files should be an int"
                        summary["total_files"] = total_files + 1
                        files_by_ticker = summary["files_by_ticker"]
                        assert isinstance(
                            files_by_ticker, dict
                        ), "files_by_ticker should be a dict"
                        files_by_ticker[ticker_name] = (
                            int(files_by_ticker[ticker_name]) + 1
                        )
                        files_by_timeframe[tf_name] = (
                            int(files_by_timeframe[tf_name]) + 1
                        )

                        # File size
                        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                        total_size_mb = summary["total_size_mb"]
                        assert isinstance(
                            total_size_mb, (int, float)
                        ), "total_size_mb should be numeric"
                        summary["total_size_mb"] = float(total_size_mb) + file_size

                        # Date range
                        try:
                            file_date = datetime.strptime(
                                file_path.stem, "%Y-%m-%d"
                            ).date()
                            date_range = summary["date_range"]
                            assert isinstance(
                                date_range, dict
                            ), "date_range should be a dict"
                            if (
                                date_range["earliest"] is None
                                or file_date < date_range["earliest"]
                            ):
                                date_range["earliest"] = file_date

                            if (
                                date_range["latest"] is None
                                or file_date > date_range["latest"]
                            ):
                                date_range["latest"] = file_date
                        except ValueError:
                            continue

            # Convert sets to lists for JSON serialization
            summary["tickers"] = list(summary["tickers"])
            summary["timeframes"] = list(summary["timeframes"])

            return summary

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _scan_directory)

    async def cleanup_old_data(
        self, older_than_days: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Clean up old data files based on retention policy.

        Args:
            older_than_days: Delete files older than this many days

        Returns:
            Cleanup statistics
        """
        if older_than_days is None:
            older_than_days = self.config.retention_days

        cutoff_date = date.today() - timedelta(days=older_than_days)

        def _cleanup():
            stats = {"files_deleted": 0, "space_freed_mb": 0.0}

            for data_type in ["market_data", "screener_data", "technical_indicators"]:
                data_dir = self.base_path / data_type
                if not data_dir.exists():
                    continue

                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        if not file.endswith(".parquet"):
                            continue

                        file_path = Path(root) / file

                        try:
                            # Extract date from filename
                            file_date = datetime.strptime(
                                file_path.stem, "%Y-%m-%d"
                            ).date()

                            if file_date < cutoff_date:
                                file_size = file_path.stat().st_size / (
                                    1024 * 1024
                                )  # MB
                                file_path.unlink()
                                stats["files_deleted"] += 1
                                stats["space_freed_mb"] += file_size

                        except Exception as e:
                            logger.warning(f"Failed to process file {file_path}: {e}")

            return stats

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._executor, _cleanup)

        logger.info(
            f"Cleanup completed: deleted {result['files_deleted']} files, freed {result['space_freed_mb']:.2f} MB"
        )
        return result

    async def validate_data_integrity(
        self,
        ticker: str,
        timeframe: TimeFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Validate data integrity for a ticker/timeframe combination.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            start_date: Start date for validation
            end_date: End date for validation

        Returns:
            Validation results
        """
        df = await self.load_market_data(ticker, timeframe, start_date, end_date)

        if df.is_empty():
            return {"valid": False, "error": "No data found"}

        validation_results: Dict[str, Any] = {
            "valid": True,
            "total_records": len(df),
            "issues": [],
        }

        try:
            # Check for missing values
            null_counts = df.null_count()
            for col, null_count in zip(df.columns, null_counts.row(0)):
                if null_count > 0:
                    validation_results["issues"].append(
                        f"{col}: {null_count} null values"
                    )

            # Check for invalid OHLC relationships
            invalid_ohlc = df.filter(
                (pl.col("high") < pl.col("low"))
                | (pl.col("high") < pl.col("open"))
                | (pl.col("high") < pl.col("close"))
                | (pl.col("low") > pl.col("open"))
                | (pl.col("low") > pl.col("close"))
            )

            if len(invalid_ohlc) > 0:
                validation_results["issues"].append(
                    f"Invalid OHLC relationships: {len(invalid_ohlc)} records"
                )

            # Check for negative prices
            negative_prices = df.filter(
                (pl.col("open") <= 0)
                | (pl.col("high") <= 0)
                | (pl.col("low") <= 0)
                | (pl.col("close") <= 0)
            )

            if len(negative_prices) > 0:
                validation_results["issues"].append(
                    f"Negative/zero prices: {len(negative_prices)} records"
                )

            # Check for negative volume
            negative_volume = df.filter(pl.col("volume") < 0)
            if len(negative_volume) > 0:
                validation_results["issues"].append(
                    f"Negative volume: {len(negative_volume)} records"
                )

            # Check for duplicates
            duplicates = df.filter(pl.col("timestamp").is_duplicated())
            if len(duplicates) > 0:
                validation_results["issues"].append(
                    f"Duplicate timestamps: {len(duplicates)} records"
                )

            validation_results["valid"] = len(validation_results["issues"]) == 0

        except Exception as e:
            logger.error(f"Data validation failed for {ticker} {timeframe}: {e}")
            validation_results["valid"] = False
            validation_results["error"] = str(e)

        return validation_results

    async def get_available_data_range(
        self, ticker: str, timeframe: TimeFrame
    ) -> Optional[Tuple[date, date]]:
        """
        Get the available date range for a ticker/timeframe.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe

        Returns:
            Tuple of (start_date, end_date) or None if no data
        """
        ticker_dir = self.base_path / "market_data" / ticker.upper() / timeframe.value

        if not ticker_dir.exists():
            return None

        # Find all Parquet files and extract dates
        dates = []
        for file_path in ticker_dir.glob("*.parquet"):
            try:
                file_date = datetime.strptime(file_path.stem, "%Y-%m-%d").date()
                dates.append(file_date)
            except ValueError:
                continue

        if not dates:
            return None

        return min(dates), max(dates)

    async def deduplicate_data(
        self, ticker: str, timeframe: TimeFrame, date_obj: Optional[date] = None
    ) -> int:
        """
        Remove duplicate records from stored data.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            date_obj: Specific date to deduplicate (None for all dates)

        Returns:
            Number of duplicates removed
        """
        if date_obj:
            dates_to_process = [date_obj]
        else:
            # Get all available dates
            date_range = await self.get_available_data_range(ticker, timeframe)
            if not date_range:
                return 0

            start_date, end_date = date_range
            dates_to_process = []
            current_date = start_date
            while current_date <= end_date:
                dates_to_process.append(current_date)
                current_date += timedelta(days=1)

        total_removed = 0

        for process_date in dates_to_process:
            file_path = self._get_file_path(
                ticker, timeframe, process_date, "market_data"
            )

            if not file_path.exists():
                continue

            try:
                df = await self._load_parquet_file(file_path)
                original_count = len(df)

                # Remove duplicates based on timestamp
                deduplicated_df = df.unique(subset=["timestamp"], keep="last")
                new_count = len(deduplicated_df)

                if new_count < original_count:
                    await self._save_dataframe_to_parquet(deduplicated_df, file_path)
                    removed = original_count - new_count
                    total_removed += removed
                    logger.info(f"Removed {removed} duplicates from {file_path}")

            except Exception as e:
                logger.error(f"Failed to deduplicate {file_path}: {e}")

        return total_removed

    async def optimize_storage(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize storage by recompressing files and removing empty files.

        Args:
            ticker: Optional ticker to optimize (None for all tickers)

        Returns:
            Optimization statistics
        """

        def _optimize():
            stats = {
                "files_processed": 0,
                "files_removed": 0,
                "space_saved_mb": 0.0,
                "files_recompressed": 0,
            }

            market_data_dir = self.base_path / "market_data"
            if not market_data_dir.exists():
                return stats

            for ticker_dir in market_data_dir.iterdir():
                if not ticker_dir.is_dir():
                    continue

                ticker_name = ticker_dir.name
                if ticker and ticker_name != ticker.upper():
                    continue

                for tf_dir in ticker_dir.iterdir():
                    if not tf_dir.is_dir():
                        continue

                    for file_path in tf_dir.glob("*.parquet"):
                        stats["files_processed"] += 1
                        original_size = file_path.stat().st_size

                        try:
                            # Load and rewrite with optimal compression
                            df = pl.read_parquet(file_path)

                            if len(df) == 0:
                                # Remove empty files
                                file_path.unlink()
                                stats["files_removed"] += 1
                                stats["space_saved_mb"] += float(
                                    original_size / (1024 * 1024)
                                )
                                continue

                            # Rewrite with optimal settings
                            df.write_parquet(
                                file_path, compression="snappy", use_pyarrow=True
                            )

                            new_size = file_path.stat().st_size
                            if new_size < original_size:
                                space_saved = (original_size - new_size) / (1024 * 1024)
                                stats["space_saved_mb"] += space_saved
                                stats["files_recompressed"] += 1

                        except Exception as e:
                            logger.warning(f"Failed to optimize {file_path}: {e}")

            return stats

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _optimize)

    async def get_data_statistics(
        self,
        ticker: str,
        timeframe: TimeFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Get statistical information about stored data.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Statistical summary
        """
        df = await self.load_market_data(ticker, timeframe, start_date, end_date)

        if df.is_empty():
            return {"error": "No data available"}

        try:
            stats = {
                "ticker": ticker,
                "timeframe": timeframe.value,
                "record_count": len(df),
                "date_range": {
                    "start": df["timestamp"].min(),
                    "end": df["timestamp"].max(),
                },
                "price_statistics": {
                    "open": {
                        "min": df["open"].min(),
                        "max": df["open"].max(),
                        "mean": df["open"].mean(),
                        "std": df["open"].std(),
                    },
                    "high": {
                        "min": df["high"].min(),
                        "max": df["high"].max(),
                        "mean": df["high"].mean(),
                        "std": df["high"].std(),
                    },
                    "low": {
                        "min": df["low"].min(),
                        "max": df["low"].max(),
                        "mean": df["low"].mean(),
                        "std": df["low"].std(),
                    },
                    "close": {
                        "min": df["close"].min(),
                        "max": df["close"].max(),
                        "mean": df["close"].mean(),
                        "std": df["close"].std(),
                    },
                },
                "volume_statistics": {
                    "min": df["volume"].min(),
                    "max": df["volume"].max(),
                    "mean": df["volume"].mean(),
                    "std": df["volume"].std(),
                    "total": df["volume"].sum(),
                },
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate statistics for {ticker}: {e}")
            return {"error": str(e)}

    async def export_data(
        self,
        ticker: str,
        timeframe: TimeFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        format: str = "csv",
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export data to various formats.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            start_date: Start date for export
            end_date: End date for export
            format: Export format ('csv', 'json', 'parquet')
            output_path: Custom output path

        Returns:
            Path to exported file
        """
        df = await self.load_market_data(ticker, timeframe, start_date, end_date)

        if df.is_empty():
            raise ValueError("No data available for export")

        if output_path is None:
            export_dir = self.base_path / "exports"
            export_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                export_dir / f"{ticker}_{timeframe.value}_{timestamp}.{format}"
            )

        def _write_export():
            if format.lower() == "csv":
                df.write_csv(output_path)
            elif format.lower() == "json":
                df.write_json(output_path)
            elif format.lower() == "parquet":
                df.write_parquet(output_path, compression="snappy")
            else:
                raise ValueError(f"Unsupported export format: {format}")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, _write_export)

        logger.info(f"Exported {len(df)} records to {output_path}")
        return str(output_path)

    async def get_missing_data_gaps(
        self,
        ticker: str,
        timeframe: TimeFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identify missing data gaps for a ticker/timeframe.

        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe
            start_date: Start date for gap analysis
            end_date: End date for gap analysis

        Returns:
            List of missing data gaps
        """
        df = await self.load_market_data(ticker, timeframe, start_date, end_date)

        if df.is_empty():
            return []

        # Expected intervals
        interval_map = {
            TimeFrame.FIVE_MINUTES: timedelta(minutes=5),
            TimeFrame.FIFTEEN_MINUTES: timedelta(minutes=15),
            TimeFrame.ONE_HOUR: timedelta(hours=1),
            TimeFrame.ONE_DAY: timedelta(days=1),
        }

        expected_interval = interval_map.get(timeframe)
        if not expected_interval:
            return []

        gaps = []
        timestamps = df["timestamp"].to_list()

        for i in range(1, len(timestamps)):
            current_time = timestamps[i]
            previous_time = timestamps[i - 1]
            actual_gap = current_time - previous_time

            # Account for weekends and market hours for daily data
            if timeframe == TimeFrame.ONE_DAY:
                expected_gap = self._calculate_expected_daily_gap(
                    previous_time, current_time
                )
            else:
                expected_gap = expected_interval

            if actual_gap > expected_gap * 2:  # Allow some tolerance
                gaps.append(
                    {
                        "start_time": previous_time,
                        "end_time": current_time,
                        "gap_duration": str(actual_gap),
                        "expected_duration": str(expected_gap),
                        "gap_ratio": actual_gap / expected_gap,
                    }
                )

        return gaps

    def _calculate_expected_daily_gap(
        self, start_time: datetime, end_time: datetime
    ) -> timedelta:
        """Calculate expected gap between daily data points accounting for weekends."""
        start_weekday = start_time.weekday()
        end_weekday = end_time.weekday()

        # If it's Friday to Monday, expect 3 days
        if start_weekday == 4 and end_weekday == 0:  # Friday to Monday
            return timedelta(days=3)
        # Weekend to Tuesday (Monday holiday)
        elif start_weekday == 4 and end_weekday == 1:
            return timedelta(days=4)
        else:
            return timedelta(days=1)

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)


# Utility functions for data operations
async def batch_save_market_data(
    store: DataStore,
    data_by_symbol: Dict[str, List[MarketData]],
    batch_size: int = 1000,
) -> Dict[str, int]:
    """
    Save market data for multiple symbols in batches.

    Args:
        store: DataStore instance
        data_by_symbol: Dictionary mapping symbols to MarketData lists
        batch_size: Size of batches for processing

    Returns:
        Save statistics
    """
    total_stats = {"total_saved": 0, "files_created": 0, "files_updated": 0}

    # Process symbols in batches
    symbols = list(data_by_symbol.keys())
    symbol_batches = [
        symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
    ]

    for batch in symbol_batches:
        batch_data = []
        for symbol in batch:
            batch_data.extend(data_by_symbol[symbol])

        if batch_data:
            stats = await store.save_market_data(batch_data)
            total_stats["total_saved"] += stats["total_saved"]
            total_stats["files_created"] += stats["files_created"]
            total_stats["files_updated"] += stats["files_updated"]

    return total_stats


async def get_combined_market_data(
    store: DataStore,
    tickers: List[str],
    timeframe: TimeFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pl.DataFrame:
    """
    Get combined market data for multiple tickers.

    Args:
        store: DataStore instance
        tickers: List of ticker symbols
        timeframe: Data timeframe
        start_date: Start date filter
        end_date: End date filter

    Returns:
        Combined DataFrame with all ticker data
    """
    return await store.load_multiple_tickers(tickers, timeframe, start_date, end_date)


async def calculate_returns(
    store: DataStore, ticker: str, timeframe: TimeFrame, periods: int = 30
) -> pl.DataFrame:
    """
    Calculate returns for a ticker.

    Args:
        store: DataStore instance
        ticker: Stock ticker symbol
        timeframe: Data timeframe
        periods: Number of periods for calculation

    Returns:
        DataFrame with returns calculation
    """
    df = await store.get_latest_data(ticker, timeframe, limit=periods + 1)

    if len(df) < 2:
        return pl.DataFrame()

    # Calculate returns
    returns_df = df.with_columns(
        [
            (pl.col("close").pct_change().alias("returns")),
            (pl.col("close").pct_change().cum_sum().alias("cumulative_returns")),
            ((pl.col("high") / pl.col("low") - 1).alias("intraday_returns")),
        ]
    )

    return returns_df.drop_nulls()


# Example usage and testing
if __name__ == "__main__":

    async def main():
        config = DataStoreConfig(
            base_path="test_data/parquet", compression="snappy", retention_days=365
        )

        store = DataStore(config)

        # Create some sample data
        sample_data = [
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now() - timedelta(hours=i),
                timeframe=TimeFrame.FIVE_MINUTES,
                open=Decimal("150.00"),
                high=Decimal("150.50"),
                low=Decimal("149.50"),
                close=Decimal("150.25"),
                volume=1000000,
                adjusted_close=Decimal("150.25"),
                asset_type=AssetType.STOCK,
            )
            for i in range(100)
        ]

        # Save data
        save_stats = await store.save_market_data(sample_data)
        print(f"Save stats: {save_stats}")

        # Load data back
        df = await store.load_market_data("AAPL", TimeFrame.FIVE_MINUTES)
        print(f"Loaded {len(df)} records")

        # Get summary
        summary = await store.get_data_summary()
        print(f"Data summary: {summary}")

        # Validate data
        validation = await store.validate_data_integrity("AAPL", TimeFrame.FIVE_MINUTES)
        print(f"Validation: {validation}")

    asyncio.run(main())
