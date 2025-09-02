#!/usr/bin/env python3
"""
Enable Multi-Timeframe Confirmation Script

Simple script to enable multi-timeframe confirmation for existing strategies
by updating their configuration in Redis.
"""

import asyncio
import json
import sys
from datetime import datetime, timezone

import redis.asyncio as redis

REDIS_URL = "redis://:Xo8uWxU1fmG0P1036pXysH2k4MTNhbmi@localhost:6380/0"

MTF_CONFIG = {
    "enable_mtf_confirmation": True,
    "mtf_min_timeframes": 3,
    "mtf_confidence_boost": 10.0,
    "mtf_confidence_penalty": 20.0,
    "mtf_required_strength": "strong",
    "mtf_min_confidence": 70.0,
}


async def enable_mtf_for_all_strategies():
    """Enable MTF confirmation for all existing strategies."""
    try:
        # Connect to Redis
        redis_client = redis.from_url(REDIS_URL)

        print("🔗 Connecting to Redis...")
        await redis_client.ping()
        print("✅ Connected to Redis successfully")

        # Get all strategy keys
        strategy_keys = await redis_client.keys("strategy_state:*")

        if not strategy_keys:
            print("⚠️  No strategies found in Redis")
            return

        print(f"📊 Found {len(strategy_keys)} strategies to update")

        updated_count = 0

        for key in strategy_keys:
            try:
                # Get existing strategy data
                existing_data = await redis_client.get(key)
                if not existing_data:
                    continue

                strategy_data = json.loads(existing_data)
                strategy_name = key.decode().replace("strategy_state:", "")

                # Initialize config structure if needed
                if "config" not in strategy_data:
                    strategy_data["config"] = {}
                if "parameters" not in strategy_data["config"]:
                    strategy_data["config"]["parameters"] = {}

                # Add MTF configuration
                strategy_data["config"]["parameters"].update(MTF_CONFIG)

                # Add metadata
                strategy_data["mtf_enabled"] = True
                strategy_data["mtf_enabled_at"] = datetime.now(timezone.utc).isoformat()

                # Save updated configuration
                updated_data = json.dumps(strategy_data)
                await redis_client.set(key, updated_data)

                print(f"✅ Enabled MTF for: {strategy_name}")
                updated_count += 1

            except Exception as e:
                strategy_name = key.decode().replace("strategy_state:", "")
                print(f"❌ Failed to update {strategy_name}: {e}")

        print(
            f"\n🎉 Successfully enabled MTF confirmation for {updated_count}/{len(strategy_keys)} strategies"
        )

        # Close Redis connection
        await redis_client.close()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


async def check_mtf_status():
    """Check MTF status for all strategies."""
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()

        strategy_keys = await redis_client.keys("strategy_state:*")

        if not strategy_keys:
            print("No strategies found")
            return

        print("\n" + "=" * 60)
        print("MULTI-TIMEFRAME CONFIRMATION STATUS")
        print("=" * 60)

        for key in strategy_keys:
            try:
                existing_data = await redis_client.get(key)
                if not existing_data:
                    continue

                strategy_data = json.loads(existing_data)
                strategy_name = key.decode().replace("strategy_state:", "")

                # Check MTF status
                parameters = strategy_data.get("config", {}).get("parameters", {})
                mtf_enabled = parameters.get("enable_mtf_confirmation", False)

                status = "ENABLED" if mtf_enabled else "DISABLED"
                mode = strategy_data.get("config", {}).get("mode", "unknown")

                print(f"{strategy_name}: {status} (mode: {mode})")

                if mtf_enabled:
                    confidence_boost = parameters.get("mtf_confidence_boost", "N/A")
                    min_timeframes = parameters.get("mtf_min_timeframes", "N/A")
                    print(f"  • Confidence boost: {confidence_boost}")
                    print(f"  • Min timeframes: {min_timeframes}")

                print()

            except Exception as e:
                print(f"Error reading {key}: {e}")

        await redis_client.close()

    except Exception as e:
        print(f"Error checking status: {e}")


async def main():
    """Main function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "status":
            await check_mtf_status()
        elif command == "enable":
            await enable_mtf_for_all_strategies()
        else:
            print("Usage: python enable_mtf.py [status|enable]")
            sys.exit(1)
    else:
        # Default to enable
        await enable_mtf_for_all_strategies()


if __name__ == "__main__":
    print("🚀 Multi-Timeframe Confirmation Manager")
    print("-" * 40)
    asyncio.run(main())
