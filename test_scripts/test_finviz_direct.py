#!/usr/bin/env python3
"""
Direct test of FinViz API without complex dependencies.
This script tests the FinViz Elite API directly to understand why we're getting 0 results.
"""

import asyncio
import aiohttp
import csv
import io
from datetime import datetime

# FinViz API configuration
FINVIZ_API_KEY = "0b8a5f2f-8cdc-4995-bf37-ee41210d772a"
FINVIZ_BASE_URL = "https://elite.finviz.com/export.ashx"

async def test_finviz_api(test_name, filters, order_by="-change"):
    """Test FinViz API with specific filters."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    # Build request parameters
    params = {
        'auth': FINVIZ_API_KEY,
        'v': '152',  # Version for detailed view
        'f': filters,  # Filter string
        'ft': '4',  # Filter type
        'o': order_by,  # Order by
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/csv,application/csv,*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    print(f"Filters: {filters}")
    print(f"Order by: {order_by}")
    print(f"URL: {FINVIZ_BASE_URL}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(FINVIZ_BASE_URL, params=params, headers=headers) as response:
                print(f"Status Code: {response.status}")
                print(f"Content Type: {response.headers.get('Content-Type', 'Unknown')}")

                content = await response.text()
                print(f"Response Length: {len(content)} characters")

                if response.status != 200:
                    print(f"Error Response: {content[:500]}")
                    return

                # Parse CSV response
                if content:
                    csv_reader = csv.DictReader(io.StringIO(content))
                    rows = list(csv_reader)

                    print(f"Number of results: {len(rows)}")

                    if rows:
                        print("\nFirst 5 results:")
                        for i, row in enumerate(rows[:5], 1):
                            ticker = row.get('Ticker', 'N/A')
                            price = row.get('Price', 'N/A')
                            change = row.get('Change', 'N/A')
                            volume = row.get('Volume', 'N/A')
                            print(f"  {i}. {ticker}: Price=${price}, Change={change}, Volume={volume}")

                        # Show available columns
                        if rows:
                            print(f"\nAvailable columns: {', '.join(rows[0].keys())}")
                    else:
                        print("CSV parsed but no data rows found")
                        print(f"Raw content preview: {content[:200]}")
                else:
                    print("Empty response received")

        except aiohttp.ClientError as e:
            print(f"HTTP Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Run various tests with different filter combinations."""
    print("="*60)
    print("FINVIZ API DIRECT TESTING")
    print(f"API Key: {FINVIZ_API_KEY[:10]}...")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)

    # Test 1: Very minimal filter - just stocks over $1
    await test_finviz_api(
        "Minimal Filter (Price > $1)",
        "sh_price_o1",
        "-change"
    )
    await asyncio.sleep(2)  # Rate limiting

    # Test 2: Small cap stocks with volume
    await test_finviz_api(
        "Small Cap with Volume",
        "cap_smallover,sh_avgvol_o100",
        "-change"
    )
    await asyncio.sleep(2)

    # Test 3: The exact filters used in our momentum screener
    await test_finviz_api(
        "Our Momentum Screener Filters",
        "cap_microover,sh_avgvol_o500,sh_curvol_o200,ta_sma20_pa",
        "-change"
    )
    await asyncio.sleep(2)

    # Test 4: Very broad filter - any market cap
    await test_finviz_api(
        "Any Market Cap",
        "sh_avgvol_o50",
        "-volume"
    )
    await asyncio.sleep(2)

    # Test 5: Large cap only
    await test_finviz_api(
        "Large Cap Stocks",
        "cap_largeover",
        "-marketcap"
    )
    await asyncio.sleep(2)

    # Test 6: High volume today
    await test_finviz_api(
        "High Volume Today",
        "sh_curvol_o1000",
        "-volume"
    )
    await asyncio.sleep(2)

    # Test 7: No filters at all
    await test_finviz_api(
        "No Filters",
        "",
        "-change"
    )

    print("\n" + "="*60)
    print("TESTING COMPLETED")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
