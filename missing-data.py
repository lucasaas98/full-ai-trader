#!/usr/bin/env python3
"""
Enhanced Data Inspection Tool for Full AI Trader
Analyzes current data completeness, missing data gaps, and ticker status
with detailed timestamp analysis using polars
"""

import asyncio
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

from services.data_collector.src.data_store import DataStore, DataStoreConfig
from services.data_collector.src.redis_client import RedisClient, RedisConfig
from shared.models import TimeFrame

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def get_precise_timestamp_range(
    data_store: DataStore, ticker: str, timeframe: TimeFrame
) -> Optional[Tuple[datetime, datetime, int]]:
    """
    Get precise first and last timestamps from parquet files for a ticker/timeframe.

    Returns:
        Tuple of (first_timestamp, last_timestamp, total_records) or None if no data
    """
    try:
        # Load all data for this ticker/timeframe - use epoch start date to get everything
        start_date = date(1970, 1, 1)  # Unix epoch start
        df = await data_store.load_market_data(ticker, timeframe, start_date=start_date)

        if df.is_empty():
            return None

        # Get min/max timestamps and record count
        first_timestamp = df.select("timestamp").min().item()
        last_timestamp = df.select("timestamp").max().item()
        record_count = len(df)

        return first_timestamp, last_timestamp, record_count

    except Exception as e:
        print(
            f"    Error getting precise timestamps for {ticker} {timeframe.value}: {e}"
        )
        return None


def read_parquet_directly(
    base_path: Path, ticker: str, timeframe: TimeFrame
) -> Optional[Tuple[datetime, datetime, int]]:
    """
    Directly read parquet files to get timestamp ranges when data_store fails.

    Returns:
        Tuple of (first_timestamp, last_timestamp, total_records) or None if no data
    """
    try:
        ticker_dir = base_path / "market_data" / ticker.upper() / timeframe.value

        if not ticker_dir.exists():
            return None

        parquet_files = list(ticker_dir.glob("*.parquet"))
        if not parquet_files:
            return None

        all_dfs = []
        for file_path in parquet_files:
            try:
                df = pl.read_parquet(file_path)
                if not df.is_empty():
                    all_dfs.append(df)
            except Exception as e:
                print(f"      Error reading {file_path}: {e}")
                continue

        if not all_dfs:
            return None

        # Combine all dataframes
        combined_df = pl.concat(all_dfs)

        if combined_df.is_empty():
            return None

        # Get min/max timestamps and record count
        first_timestamp = combined_df.select("timestamp").min().item()
        last_timestamp = combined_df.select("timestamp").max().item()
        record_count = len(combined_df)

        return first_timestamp, last_timestamp, record_count

    except Exception as e:
        print(
            f"    Error reading parquet files directly for {ticker} {timeframe.value}: {e}"
        )
        return None


async def inspect_data_completeness():
    """Comprehensive data inspection with detailed timestamp analysis."""

    print("🔍 ENHANCED FULL AI TRADER DATA INSPECTION")
    print("=" * 60)

    # Initialize data store
    data_store_config = DataStoreConfig()
    data_store = DataStore(data_store_config)

    # Initialize Redis client
    redis_config = RedisConfig(password=None)
    redis_client = RedisClient(redis_config)

    try:
        await redis_client.connect()
        redis_connected = True
    except Exception as e:
        print(f"⚠️ Failed to connect to Redis: {e}")
        redis_connected = False

    try:
        # 1. Get overall data summary
        print("\n📊 OVERALL DATA SUMMARY")
        print("-" * 30)
        summary = await data_store.get_data_summary()

        print(f"Total files: {summary['total_files']:,}")
        print(f"Total size: {summary['total_size_mb']:.2f} MB")
        print(f"Tickers: {len(summary['tickers'])}")
        print(f"Timeframes: {summary['timeframes']}")
        print(
            f"File date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}"
        )

        # 2. Get active tickers from Redis (if connected)
        active_tickers = []
        if redis_connected:
            print("\n🎯 ACTIVE TICKERS FROM REDIS")
            print("-" * 30)
            try:
                active_tickers = await redis_client.get_active_tickers()
                print(f"Active tickers count: {len(active_tickers)}")
                if active_tickers:
                    print(f"Active tickers: {', '.join(sorted(active_tickers))}")
                else:
                    print("⚠️ No active tickers found in Redis!")
            except Exception as e:
                print(f"❌ Error getting active tickers: {e}")
        else:
            print("\n⚠️ REDIS NOT CONNECTED - Cannot check active tickers")
            print("-" * 30)

        # 3. File system vs Redis analysis
        print("\n📁 DATA SOURCE ANALYSIS")
        print("-" * 30)
        fs_tickers = set(summary["tickers"])
        redis_tickers = set(active_tickers) if active_tickers else set()

        print(f"File system tickers: {len(fs_tickers)}")
        print(f"Redis active tickers: {len(redis_tickers)}")

        if redis_tickers:
            only_in_fs = fs_tickers - redis_tickers
            only_in_redis = redis_tickers - fs_tickers
            common_tickers = fs_tickers & redis_tickers

            if only_in_fs:
                print(f"Only in file system: {len(only_in_fs)} tickers")
                if len(only_in_fs) <= 10:
                    print(f"  {', '.join(sorted(only_in_fs))}")
            if only_in_redis:
                print(f"Only in Redis: {len(only_in_redis)} tickers")
                if len(only_in_redis) <= 10:
                    print(f"  {', '.join(sorted(only_in_redis))}")
            print(f"Common tickers: {len(common_tickers)}")

            # Use common tickers if available, otherwise use file system tickers
            tickers_to_analyze = common_tickers if common_tickers else fs_tickers
        else:
            print("⚠️ Redis has no active tickers - analyzing all file system tickers")
            tickers_to_analyze = fs_tickers

        print(f"\n🔍 ANALYZING {len(tickers_to_analyze)} TICKERS")
        print("-" * 30)

        # 4. DETAILED TIMESTAMP ANALYSIS - Limit to reasonable number for display
        print("\n📈 DETAILED DATA TIMESTAMP ANALYSIS")
        print("=" * 60)

        timeframes_to_check = [
            TimeFrame.ONE_MINUTE,
            TimeFrame.FIVE_MINUTES,
            TimeFrame.FIFTEEN_MINUTES,
            TimeFrame.ONE_HOUR,
            TimeFrame.ONE_DAY,
        ]

        # Create a summary table
        data_summary = []

        # Limit analysis to first 20 tickers for detailed display, but analyze all for summary
        display_limit = 20
        tickers_list = sorted(tickers_to_analyze)

        print(
            f"📊 Showing detailed analysis for first {min(display_limit, len(tickers_list))} tickers..."
        )
        print(f"📊 (Full summary table will include all {len(tickers_list)} tickers)")

        for i, ticker in enumerate(tickers_list):
            # Show detailed output only for first N tickers
            show_details = i < display_limit

            if show_details:
                print(f"\n🏷️  {ticker} ({i + 1}/{len(tickers_list)}):")

            for tf in timeframes_to_check:
                if show_details:
                    print(f"  📊 {tf.value}:")

                # First try using data_store method
                timestamp_range = await get_precise_timestamp_range(
                    data_store, ticker, tf
                )

                # If that fails, try direct parquet reading
                if timestamp_range is None:
                    if show_details:
                        print("    Trying direct parquet reading...")
                    timestamp_range = read_parquet_directly(
                        data_store.base_path, ticker, tf
                    )

                if timestamp_range:
                    first_ts, last_ts, record_count = timestamp_range

                    # Calculate time span
                    time_span = last_ts - first_ts

                    if show_details:
                        print(f"    ✅ Records: {record_count:,}")
                        print(
                            f"    📅 First: {first_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        )
                        print(
                            f"    📅 Last:  {last_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                        )
                        print(f"    ⏱️  Span:  {time_span}")

                    # Calculate expected vs actual records
                    if tf == TimeFrame.FIVE_MINUTES:
                        # 5-minute intervals: 12 per hour, ~78 per trading day (6.5 hours)
                        trading_days = max(1, time_span.days)
                        expected_records = trading_days * 78
                    elif tf == TimeFrame.FIFTEEN_MINUTES:
                        # 15-minute intervals: 4 per hour, ~26 per trading day
                        trading_days = max(1, time_span.days)
                        expected_records = trading_days * 26
                    else:
                        expected_records = None

                    completeness = None
                    if expected_records:
                        completeness = (record_count / expected_records) * 100
                        if show_details:
                            print(
                                f"    📊 Completeness: {completeness:.1f}% ({record_count:,}/{expected_records:,} expected)"
                            )

                    # Add to summary
                    data_summary.append(
                        {
                            "ticker": ticker,
                            "timeframe": tf.value,
                            "records": record_count,
                            "first_timestamp": first_ts,
                            "last_timestamp": last_ts,
                            "span_days": time_span.days,
                            "completeness_pct": completeness,
                        }
                    )

                    # Check for data gaps (only for detailed display)
                    if show_details:
                        try:
                            gaps = await data_store.get_missing_data_gaps(ticker, tf)
                            if gaps:
                                print(f"    ⚠️  {len(gaps)} data gaps found")
                                for gap in gaps[:2]:  # Show first 2 gaps
                                    print(
                                        f"      Gap: {gap['start_time'].strftime('%m-%d %H:%M')} to {gap['end_time'].strftime('%m-%d %H:%M')} (ratio: {gap['gap_ratio']:.1f}x)"
                                    )
                        except Exception as e:
                            if show_details:
                                print(f"    Error checking gaps: {e}")

                else:
                    if show_details:
                        print("    ❌ No data found")
                    data_summary.append(
                        {
                            "ticker": ticker,
                            "timeframe": tf.value,
                            "records": 0,
                            "first_timestamp": None,
                            "last_timestamp": None,
                            "span_days": 0,
                            "completeness_pct": 0,
                        }
                    )

            # Show progress for remaining tickers
            if not show_details and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(tickers_list)} tickers...")

        # 5. SUMMARY TABLE
        print(f"\n📋 COMPLETE DATA SUMMARY TABLE ({len(data_summary)} entries)")
        print("=" * 110)
        print(
            f"{'Ticker':<8} {'Timeframe':<12} {'Records':<10} {'First Date':<12} {'Last Date':<12} {'Days':<6} {'Complete%':<10}"
        )
        print("-" * 110)

        for item in data_summary:
            first_date = (
                item["first_timestamp"].strftime("%Y-%m-%d")
                if item["first_timestamp"]
                else "None"
            )
            last_date = (
                item["last_timestamp"].strftime("%Y-%m-%d")
                if item["last_timestamp"]
                else "None"
            )
            completeness_str = (
                f"{item['completeness_pct']:.1f}%"
                if item["completeness_pct"]
                else "N/A"
            )

            print(
                f"{item['ticker']:<8} {item['timeframe']:<12} {item['records']:<10,} {first_date:<12} {last_date:<12} {item['span_days']:<6} {completeness_str:<10}"
            )

        # 6. Check Redis cache status (if connected)
        if redis_connected and active_tickers:
            print("\n💾 REDIS CACHE STATUS")
            print("-" * 30)

            for ticker in sorted(active_tickers)[:5]:  # Check first 5 tickers
                # Get last update times
                for tf in timeframes_to_check:
                    try:
                        last_update = await redis_client.get_last_update_time(
                            ticker, tf
                        )
                        if last_update:
                            time_since_update = datetime.now(timezone.utc) - last_update
                            print(
                                f"{ticker} {tf.value}: Last updated {time_since_update} ago"
                            )
                        else:
                            print(f"{ticker} {tf.value}: No update time recorded")
                    except Exception as e:
                        print(f"{ticker} {tf.value}: Error - {e}")

        # 7. Analysis and Recommendations
        print("\n📊 ANALYSIS SUMMARY")
        print("-" * 30)

        # Count tickers with data
        tickers_with_5m = len(
            [
                item
                for item in data_summary
                if item["timeframe"] == "5min" and item["records"] > 0
            ]
        )
        tickers_with_15m = len(
            [
                item
                for item in data_summary
                if item["timeframe"] == "15min" and item["records"] > 0
            ]
        )
        total_tickers = len(set(item["ticker"] for item in data_summary))

        print(f"Total tickers analyzed: {total_tickers}")
        print(f"Tickers with 5-minute data: {tickers_with_5m}")
        print(f"Tickers with 15-minute data: {tickers_with_15m}")

        # Average completeness
        completeness_5m = [
            item["completeness_pct"]
            for item in data_summary
            if item["timeframe"] == "5min" and item["completeness_pct"]
        ]
        completeness_15m = [
            item["completeness_pct"]
            for item in data_summary
            if item["timeframe"] == "15min" and item["completeness_pct"]
        ]

        if completeness_5m:
            print(
                f"Average 5-minute data completeness: {sum(completeness_5m) / len(completeness_5m):.1f}%"
            )
        if completeness_15m:
            print(
                f"Average 15-minute data completeness: {sum(completeness_15m) / len(completeness_15m):.1f}%"
            )

        # Find tickers with low data counts
        low_data_tickers = [
            item
            for item in data_summary
            if item["records"] < 50 and item["records"] > 0
        ]
        high_data_tickers = [item for item in data_summary if item["records"] > 1000]

        if low_data_tickers:
            print(f"\n⚠️ Tickers with low data counts ({len(low_data_tickers)}):")
            for item in sorted(low_data_tickers, key=lambda x: x["records"]):
                print(
                    f"  {item['ticker']} ({item['timeframe']}): {item['records']} records"
                )

        if high_data_tickers:
            print(f"\n✅ Tickers with high data counts ({len(high_data_tickers)}):")
            for item in sorted(
                high_data_tickers, key=lambda x: x["records"], reverse=True
            )[:5]:
                print(
                    f"  {item['ticker']} ({item['timeframe']}): {item['records']:,} records"
                )

        # 8. System Status Analysis
        print("\n🔧 SYSTEM STATUS ANALYSIS")
        print("-" * 30)
        if not redis_connected:
            print("❌ Redis connection failed - data collection service may be down")
        elif not active_tickers:
            print(
                "⚠️ Redis connected but no active tickers - data collection may be paused"
            )
            print("   - Check if data collection service is running")
            print("   - Check if tickers were properly loaded into Redis")
        else:
            print("✅ Redis connected with active tickers")

        print(f"✅ File system has data for {len(fs_tickers)} tickers")
        print(f"📊 Total data files: {summary['total_files']:,}")
        print(f"💾 Total data size: {summary['total_size_mb']:.2f} MB")

        # 9. Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 30)

        if not redis_connected:
            print("1. 🔧 Fix Redis connection to enable real-time data collection")
        elif not active_tickers:
            print(
                "1. 🔧 Restart data collection service to populate active tickers in Redis"
            )
            print("2. 🔍 Check logs for data collection service startup issues")

        print("3. 📊 Consider data retention policy - you have data spanning full year")
        print("4. 🔄 Implement data cleanup for old files if storage is a concern")

        if low_data_tickers:
            print(
                "5. 🎯 Prioritize re-fetching data for tickers with low record counts"
            )

        print("6. 📈 Focus on 5-minute and 15-minute data for active trading")
        print("7. 🔍 Investigate why some tickers have much more data than others")

    except Exception as e:
        print(f"❌ Error during inspection: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if redis_connected:
            await redis_client.disconnect()


if __name__ == "__main__":
    asyncio.run(inspect_data_completeness())
