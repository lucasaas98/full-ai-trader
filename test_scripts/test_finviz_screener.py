#!/usr/bin/env python3
"""
Test script for FinViz screener functionality.
Tests different screener configurations to see what results are returned.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Set up environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Import the screener modules
from services.data_collector.src.finviz_screener import (  # noqa: E402
    FinVizScreener,
    FinVizScreenerParams,
)

# Set the API key from environment
FINVIZ_API_KEY = os.getenv("FINVIZ_API_KEY", "0b8a5f2f-8cdc-4995-bf37-ee41210d772a")


async def test_basic_screener() -> None:
    """Test basic screener with minimal filters."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Screener (Minimal Filters)")
    print("=" * 60)

    screener = FinVizScreener()

    # Very basic parameters - just market cap
    params = FinVizScreenerParams(
        market_cap_min="1M",
        market_cap_max=None,
        avg_volume_min=None,
        current_volume_min=None,
        price_min=None,
        price_max=None,
        above_sma20=False,
        weekly_volatility_min=None,
    )

    try:
        result = await screener.fetch_screener_data(params, limit=10)
        print(f"Results found: {result.total_count}")

        if result.data:
            print("\nFirst 5 tickers:")
            for i, stock in enumerate(result.data[:5], 1):
                print(
                    f"  {i}. {stock.ticker}: ${stock.price:.2f} | Volume: {stock.volume:,}"
                )
        else:
            print("No results returned!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pass


async def test_momentum_screener() -> None:
    """Test the momentum screener configuration."""
    print("\n" + "=" * 60)
    print("TEST 2: Momentum Screener (Default Parameters)")
    print("=" * 60)

    screener = FinVizScreener()

    try:
        # Use the default momentum parameters
        result = await screener.get_top_momentum_stocks(limit=10)
        print(f"Results found: {result.total_count}")

        if result.data:
            print("\nTop momentum stocks:")
            for i, stock in enumerate(result.data[:5], 1):
                print(
                    f"  {i}. {stock.ticker}: ${stock.price:.2f} | Change: {stock.change:.2f}%"
                )
        else:
            print("No momentum stocks found!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pass


async def test_high_volume_breakouts() -> None:
    """Test high volume breakout screener."""
    print("\n" + "=" * 60)
    print("TEST 3: High Volume Breakouts")
    print("=" * 60)

    screener = FinVizScreener()

    try:
        result = await screener.get_high_volume_breakouts(limit=10)
        print(f"Results found: {result.total_count}")

        if result.data:
            print("\nHigh volume breakouts:")
            for i, stock in enumerate(result.data[:5], 1):
                print(
                    f"  {i}. {stock.ticker}: ${stock.price:.2f} | Volume: {stock.volume:,}"
                )
        else:
            print("No breakout stocks found!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pass


async def test_large_cap_stocks() -> None:
    """Test screener with large cap stocks only."""
    print("\n" + "=" * 60)
    print("TEST 4: Large Cap Stocks (>10B)")
    print("=" * 60)

    screener = FinVizScreener()

    params = FinVizScreenerParams(
        market_cap_min="10B",
        market_cap_max=None,
        avg_volume_min="1M",
        current_volume_min=None,
        price_min=10.0,
        price_max=None,
        above_sma20=False,
        weekly_volatility_min=None,
    )

    try:
        result = await screener.fetch_screener_data(params, limit=10)
        print(f"Results found: {result.total_count}")

        if result.data:
            print("\nLarge cap stocks:")
            for i, stock in enumerate(result.data[:5], 1):
                print(
                    f"  {i}. {stock.ticker}: ${stock.price:.2f} | Market Cap: {stock.market_cap:,}"
                )
        else:
            print("No large cap stocks found!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pass


async def test_raw_api_call() -> None:
    """Test raw API call to FinViz without any processing."""
    print("\n" + "=" * 60)
    print("TEST 5: Raw API Call (Direct HTTP Request)")
    print("=" * 60)

    import aiohttp

    # Build a simple request URL
    base_url = "https://elite.finviz.com/export.ashx"
    params = {
        "auth": FINVIZ_API_KEY,
        "v": "152",  # Version
        "f": "cap_microover",  # Market cap over micro ($50M+)
        "ft": "4",  # Filter type
        "o": "-change",  # Order by change descending
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/csv,application/csv,*/*",
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                base_url, params=params, headers=headers
            ) as response:
                print(f"Status Code: {response.status}")
                print(
                    f"Content Type: {response.headers.get('Content-Type', 'Unknown')}"
                )

                content = await response.text()
                print(f"Response Length: {len(content)} characters")

                if content:
                    lines = content.strip().split("\n")
                    print(f"Number of lines: {len(lines)}")

                    if len(lines) > 1:
                        print("\nFirst 3 lines of response:")
                        for line in lines[:3]:
                            print(f"  {line[:100]}{'...' if len(line) > 100 else ''}")
                    else:
                        print(f"Response content: {content[:500]}")
                else:
                    print("Empty response!")

        except aiohttp.ClientError as e:
            print(f"HTTP Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


async def test_all_screener_types() -> None:
    """Test all 6 screener types that the data collector uses."""
    print("\n" + "=" * 60)
    print("TEST 6: All Screener Types (As Used in Data Collector)")
    print("=" * 60)

    screener = FinVizScreener()

    screener_types = [
        ("Momentum", screener.get_top_momentum_stocks),
        ("High Volume Breakouts", screener.get_high_volume_breakouts),
        ("Gappers", screener.get_gappers),
        ("Stable Growth", screener.get_stable_growth_stocks),
        ("Value Stocks", screener.get_value_stocks),
        ("Dividend Stocks", screener.get_dividend_stocks),
    ]

    results_summary = []

    for name, screener_func in screener_types:
        print(f"\nTesting {name}...")
        try:
            result = await screener_func(limit=5)  # type: ignore
            count = result.total_count if result else 0
            results_summary.append((name, count))

            if count > 0:
                print(f"  ✓ Found {count} stocks")
                if result.data:
                    print(f"    Top ticker: {result.data[0].ticker}")
            else:
                print("  ✗ No results")

            # Small delay between requests
            await asyncio.sleep(1)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results_summary.append((name, "ERROR"))

    pass

    print("\n" + "-" * 40)
    print("SUMMARY:")
    print("-" * 40)
    for name, count in results_summary:
        status = "✓" if (count != "ERROR" and count > 0) else "✗"
        print(f"{status} {name}: {count}")


async def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("FINVIZ SCREENER TEST SUITE")
    print(f"API Key: {FINVIZ_API_KEY[:10]}...")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    # Run tests in sequence to avoid rate limiting
    await test_basic_screener()
    await asyncio.sleep(2)

    await test_momentum_screener()
    await asyncio.sleep(2)

    await test_high_volume_breakouts()
    await asyncio.sleep(2)

    await test_large_cap_stocks()
    await asyncio.sleep(2)

    await test_raw_api_call()
    await asyncio.sleep(2)

    await test_all_screener_types()

    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
