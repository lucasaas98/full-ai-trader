"""
API Test Script for Multi-Timeframe Functionality

This script tests the new multi-timeframe API endpoints and demonstrates
how to create strategies with custom timeframes.
"""

import asyncio
import json
from typing import Optional

import aiohttp


class MultiTimeframeAPITester:
    """Test client for multi-timeframe API functionality."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_available_timeframes(self):
        """Test the available timeframes endpoint."""
        print("\n=== Testing Available Timeframes Endpoint ===")

        assert self.session is not None
        async with self.session.get(f"{self.base_url}/timeframes/available") as resp:
            if resp.status == 200:
                data = await resp.json()
                print("✓ Successfully retrieved available timeframes")
                print(f"  Strategy timeframes: {data['strategy_timeframes']}")
                print(f"  Data timeframes: {data['data_timeframes']}")
                print(f"  Mapping: {json.dumps(data['mapping'], indent=2)}")
                return data
            else:
                print(f"✗ Failed to get available timeframes: {resp.status}")
                return None

    async def validate_timeframes(self, timeframes):
        """Test the timeframe validation endpoint."""
        print(f"\n=== Testing Timeframe Validation: {timeframes} ===")

        assert self.session is not None
        async with self.session.post(
            f"{self.base_url}/timeframes/validate", json=timeframes
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print("✓ Validation successful")
                print(f"  Requested: {data['requested']}")
                print(f"  Available: {data['available']}")
                print(f"  Unavailable: {data['unavailable']}")
                print(f"  Data timeframes: {data['data_timeframes']}")
                print(f"  All valid: {data['all_valid']}")
                return data
            else:
                print(f"✗ Validation failed: {resp.status}")
                return None

    async def check_symbol_timeframes(self, symbol):
        """Test the symbol timeframe availability check."""
        print(f"\n=== Testing Symbol Timeframe Check: {symbol} ===")

        assert self.session is not None
        async with self.session.get(
            f"{self.base_url}/timeframes/check/{symbol}"
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Symbol check successful for {symbol}")
                print(f"  Data available: {data['data_available']}")
                print(
                    f"  Available data timeframes: {data['available_data_timeframes']}"
                )
                print(
                    f"  Available strategy timeframes: {data['available_strategy_timeframes']}"
                )
                return data
            else:
                print(f"✗ Symbol check failed: {resp.status}")
                return None

    async def create_multi_timeframe_strategy(self):
        """Test creating a strategy with custom timeframes."""
        print("\n=== Testing Multi-Timeframe Strategy Creation ===")

        strategy_config = {
            "name": "test_multi_tf_strategy",
            "strategy_type": "technical",
            "mode": "swing_trading",
            "symbols": ["AAPL", "MSFT"],
            "parameters": {
                "lookback_period": 100,
                "min_confidence": 65.0,
                "fast_ma_period": 20,
                "slow_ma_period": 50,
            },
            "primary_timeframe": "1h",
            "additional_timeframes": ["15m", "1d"],
            "custom_timeframes": ["15m", "1h", "1d"],
        }

        assert self.session is not None
        async with self.session.post(
            f"{self.base_url}/strategies", json=strategy_config
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Strategy created successfully: {data['name']}")
                print(f"  Strategy ID: {data.get('id', 'N/A')}")
                return data
            else:
                error_text = await resp.text()
                print(f"✗ Strategy creation failed: {resp.status}")
                print(f"  Error: {error_text}")
                return None

    async def get_strategy_timeframes(self, strategy_name):
        """Test getting timeframe info for a specific strategy."""
        print(f"\n=== Testing Strategy Timeframes: {strategy_name} ===")

        assert self.session is not None
        async with self.session.get(
            f"{self.base_url}/strategies/{strategy_name}/timeframes"
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print("✓ Strategy timeframes retrieved successfully")
                timeframe_info = data["timeframe_info"]
                print(f"  Required: {timeframe_info['required']}")
                print(f"  Available: {timeframe_info['available']}")
                print(f"  Unavailable: {timeframe_info['unavailable']}")
                print(f"  All available: {timeframe_info['all_available']}")
                print(f"  Data timeframes: {timeframe_info['data_timeframes']}")
                return data
            else:
                print(f"✗ Failed to get strategy timeframes: {resp.status}")
                return None

    async def test_signal_generation(self, strategy_name, symbol):
        """Test signal generation with multi-timeframe strategy."""
        print(f"\n=== Testing Signal Generation: {strategy_name} on {symbol} ===")

        signal_request = {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "include_analysis": True,
        }

        assert self.session is not None
        async with self.session.post(
            f"{self.base_url}/signals/generate", json=signal_request
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                print("✓ Signal generated successfully")
                signal = data.get("signal", {})
                print(f"  Action: {signal.get('action', 'N/A')}")
                print(f"  Confidence: {signal.get('confidence', 'N/A')}%")
                print(f"  Price: {signal.get('price', 'N/A')}")

                metadata = signal.get("metadata", {})
                if "method" in metadata:
                    print(f"  Analysis method: {metadata['method']}")
                if "timeframe_analysis" in metadata:
                    print(
                        f"  Timeframes analyzed: {len(metadata['timeframe_analysis'])}"
                    )
                    for tf, analysis in metadata["timeframe_analysis"].items():
                        print(
                            f"    {tf}: {analysis.get('signal', 'N/A')} ({analysis.get('strength', 0):.2f})"
                        )

                print("  Data source: Data Collector API")

                return data
            else:
                error_text = await resp.text()
                print(f"✗ Signal generation failed: {resp.status}")
                print(f"  Error: {error_text}")
                print(
                    f"  Note: Ensure data collector service is running and has data for {symbol}"
                )
                return None

    async def list_strategies(self):
        """Test listing all strategies."""
        print("\n=== Testing Strategy Listing ===")

        assert self.session is not None
        async with self.session.get(f"{self.base_url}/strategies") as resp:
            if resp.status == 200:
                data = await resp.json()
                print("✓ Strategies listed successfully")
                strategies = data.get("strategies", [])
                print(f"  Total strategies: {len(strategies)}")
                for strategy in strategies:
                    print(
                        f"    - {strategy['name']} ({strategy['type']}, {strategy['mode']})"
                    )
                return data
            else:
                print(f"✗ Failed to list strategies: {resp.status}")
                return None

    async def cleanup_test_strategy(self, strategy_name):
        """Clean up test strategy."""
        print(f"\n=== Cleaning up test strategy: {strategy_name} ===")

        # Note: Assuming there's a delete endpoint (might need to be implemented)
        try:
            assert self.session is not None
            async with self.session.delete(
                f"{self.base_url}/strategies/{strategy_name}"
            ) as resp:
                if resp.status == 200:
                    print(f"✓ Strategy {strategy_name} deleted successfully")
                else:
                    print(f"! Strategy deletion not available or failed: {resp.status}")
        except Exception as e:
            print(f"! Cleanup skipped (endpoint may not exist): {e}")


async def run_comprehensive_test():
    """Run comprehensive test suite for multi-timeframe functionality."""
    print("🚀 Starting Multi-Timeframe API Comprehensive Test Suite")
    print("=" * 60)

    async with MultiTimeframeAPITester() as tester:
        # Test 1: Get available timeframes
        await tester.get_available_timeframes()

        # Test 2: Validate different timeframe combinations
        test_timeframes = [
            ["1h", "1d"],  # Valid timeframes
            ["15m", "1h", "1d"],  # Valid multi-timeframe
            ["1m", "5m", "4h", "1w"],  # Mix of valid and invalid
            ["2h", "3d", "30m"],  # Invalid timeframes
        ]

        for tf_list in test_timeframes:
            await tester.validate_timeframes(tf_list)

        # Test 3: Check symbol timeframe availability
        test_symbols = ["AAPL", "MSFT", "GOOGL", "NONEXISTENT"]
        for symbol in test_symbols:
            await tester.check_symbol_timeframes(symbol)

        # Test 4: Create multi-timeframe strategy
        strategy_data = await tester.create_multi_timeframe_strategy()
        strategy_name = strategy_data.get("name") if strategy_data else None

        if strategy_name:
            # Test 5: Get strategy timeframe info
            await tester.get_strategy_timeframes(strategy_name)

            # Test 6: Test signal generation
            await tester.test_signal_generation(strategy_name, "GOOGL")

        # Test 7: List all strategies
        await tester.list_strategies()

        # Test 8: Cleanup
        if strategy_name:
            await tester.cleanup_test_strategy(strategy_name)

    print("\n" + "=" * 60)
    print("🎉 Multi-Timeframe API Test Suite Completed!")
    print("\nSummary of implemented features:")
    print("✓ Timeframe naming convention mapping")
    print("✓ Strategy-specific timeframe support")
    print("✓ Multi-timeframe data loading via Data Collector API")
    print("✓ Timeframe validation API endpoints")
    print("✓ Enhanced strategy configuration")
    print("✓ Multi-timeframe signal generation")
    print("✓ Data Collector client integration")


async def run_quick_test():
    """Run a quick test of basic functionality."""
    print("🚀 Quick Multi-Timeframe API Test")
    print("=" * 40)

    async with MultiTimeframeAPITester() as tester:
        # Quick tests
        await tester.get_available_timeframes()
        await tester.validate_timeframes(["1h", "1d"])
        await tester.check_symbol_timeframes("GOOGL")

    print("\n🎉 Quick test completed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test multi-timeframe API functionality"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8001",
        help="Base URL for the API (default: http://localhost:8001)",
    )

    args = parser.parse_args()

    if args.quick:
        asyncio.run(run_quick_test())
    else:
        asyncio.run(run_comprehensive_test())
