#!/usr/bin/env python3
"""
Selective Multi-Timeframe Confirmation Script

Enables MTF confirmation only where it makes sense:
- Day Trading: DISABLED (too restrictive for intraday moves)
- Swing Trading: ENABLED (perfect for 2-7 day holds)
- Position Trading: ENABLED (essential for long-term positions)
"""

import asyncio
import json
import sys
import redis.asyncio as redis
from datetime import datetime, timezone


REDIS_URL = "redis://:Xo8uWxU1fmG0P1036pXysH2k4MTNhbmi@localhost:6380/0"

# Different MTF configs for different strategy types
MTF_CONFIGS = {
    "day_trading": {
        "enable_mtf_confirmation": False,  # DISABLED - too restrictive
        "reasoning": "Day trading needs quick entries, MTF too slow"
    },
    "swing_trading": {
        "enable_mtf_confirmation": True,
        "mtf_min_timeframes": 3,
        "mtf_confidence_boost": 12.0,
        "mtf_confidence_penalty": 18.0,
        "mtf_required_strength": "strong",
        "mtf_min_confidence": 70.0,
        "reasoning": "Swing trading perfect for MTF confirmation"
    },
    "position_trading": {
        "enable_mtf_confirmation": True,
        "mtf_min_timeframes": 4,
        "mtf_confidence_boost": 15.0,
        "mtf_confidence_penalty": 25.0,
        "mtf_required_strength": "strong",
        "mtf_min_confidence": 65.0,
        "reasoning": "Position trading needs strongest confirmation"
    }
}


async def apply_selective_mtf():
    """Apply selective MTF configuration based on strategy mode."""
    try:
        redis_client = redis.from_url(REDIS_URL)

        print("🔗 Connecting to Redis...")
        await redis_client.ping()
        print("✅ Connected successfully")

        # Get all strategies
        strategy_keys = await redis_client.keys("strategy_state:*")

        if not strategy_keys:
            print("⚠️  No strategies found")
            return

        print(f"📊 Found {len(strategy_keys)} strategies")
        print("\n🎯 Applying selective MTF configuration:")

        updated_count = 0

        for key in strategy_keys:
            try:
                existing_data = await redis_client.get(key)
                if not existing_data:
                    continue

                strategy_data = json.loads(existing_data)
                strategy_name = key.decode().replace("strategy_state:", "")

                # Get strategy mode
                mode = strategy_data.get('config', {}).get('mode', 'unknown')

                # Get appropriate MTF config
                mtf_config = MTF_CONFIGS.get(mode, MTF_CONFIGS['swing_trading'])

                # Initialize config structure
                if 'config' not in strategy_data:
                    strategy_data['config'] = {}
                if 'parameters' not in strategy_data['config']:
                    strategy_data['config']['parameters'] = {}

                # Remove old MTF settings first
                old_mtf_keys = [k for k in strategy_data['config']['parameters'].keys()
                              if k.startswith('mtf_') or k == 'enable_mtf_confirmation']
                for old_key in old_mtf_keys:
                    del strategy_data['config']['parameters'][old_key]

                # Apply new MTF configuration
                reasoning = mtf_config.pop('reasoning', '')
                strategy_data['config']['parameters'].update(mtf_config)

                # Add metadata
                strategy_data['selective_mtf_applied'] = True
                strategy_data['selective_mtf_applied_at'] = datetime.now(timezone.utc).isoformat()
                strategy_data['selective_mtf_reasoning'] = reasoning

                # Save updated configuration
                updated_data = json.dumps(strategy_data)
                await redis_client.set(key, updated_data)

                # Show what was applied
                status = "ENABLED" if mtf_config.get('enable_mtf_confirmation', False) else "DISABLED"
                print(f"  {strategy_name} ({mode}): {status}")
                print(f"    → {reasoning}")

                updated_count += 1

            except Exception as e:
                strategy_name = key.decode().replace("strategy_state:", "")
                print(f"❌ Failed to update {strategy_name}: {e}")

        print(f"\n🎉 Successfully configured {updated_count}/{len(strategy_keys)} strategies")

        await redis_client.aclose()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


async def show_selective_status():
    """Show current selective MTF status."""
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()

        strategy_keys = await redis_client.keys("strategy_state:*")

        if not strategy_keys:
            print("No strategies found")
            return

        print("\n" + "="*70)
        print("SELECTIVE MULTI-TIMEFRAME CONFIRMATION STATUS")
        print("="*70)

        for key in strategy_keys:
            try:
                existing_data = await redis_client.get(key)
                if not existing_data:
                    continue

                strategy_data = json.loads(existing_data)
                strategy_name = key.decode().replace("strategy_state:", "")

                # Get strategy details
                mode = strategy_data.get('config', {}).get('mode', 'unknown')
                parameters = strategy_data.get('config', {}).get('parameters', {})
                mtf_enabled = parameters.get('enable_mtf_confirmation', False)

                # Status display
                status = "ENABLED" if mtf_enabled else "DISABLED"

                print(f"\n📈 {strategy_name.upper()}")
                print(f"   Mode: {mode}")
                print(f"   MTF Status: {status}")

                # Show reasoning if available
                reasoning = strategy_data.get('selective_mtf_reasoning', '')
                if reasoning:
                    print(f"   Reasoning: {reasoning}")

                # Show MTF parameters if enabled
                if mtf_enabled:
                    confidence_boost = parameters.get('mtf_confidence_boost', 'N/A')
                    min_timeframes = parameters.get('mtf_min_timeframes', 'N/A')
                    required_strength = parameters.get('mtf_required_strength', 'N/A')
                    print(f"   Config:")
                    print(f"     • Min timeframes: {min_timeframes}")
                    print(f"     • Confidence boost: {confidence_boost}")
                    print(f"     • Required strength: {required_strength}")
                else:
                    print(f"   Config: MTF disabled for optimal {mode} performance")

            except Exception as e:
                print(f"Error reading {key}: {e}")

        print("\n" + "="*70)
        print("STRATEGY RECOMMENDATIONS:")
        print("• Day Trading: MTF disabled - captures intraday momentum")
        print("• Swing Trading: MTF enabled - confirms multi-day trends")
        print("• Position Trading: MTF enabled - validates long-term moves")
        print("="*70)

        await redis_client.aclose()

    except Exception as e:
        print(f"Error checking status: {e}")


async def revert_to_uniform_mtf():
    """Revert all strategies to uniform MTF enabled (original behavior)."""
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()

        UNIFORM_MTF = {
            "enable_mtf_confirmation": True,
            "mtf_min_timeframes": 3,
            "mtf_confidence_boost": 10.0,
            "mtf_confidence_penalty": 20.0,
            "mtf_required_strength": "strong",
            "mtf_min_confidence": 70.0
        }

        strategy_keys = await redis_client.keys("strategy_state:*")
        print(f"🔄 Reverting {len(strategy_keys)} strategies to uniform MTF...")

        for key in strategy_keys:
            existing_data = await redis_client.get(key)
            if not existing_data:
                continue

            strategy_data = json.loads(existing_data)
            strategy_name = key.decode().replace("strategy_state:", "")

            # Apply uniform config
            if 'config' not in strategy_data:
                strategy_data['config'] = {}
            if 'parameters' not in strategy_data['config']:
                strategy_data['config']['parameters'] = {}

            strategy_data['config']['parameters'].update(UNIFORM_MTF)
            strategy_data['uniform_mtf_reverted'] = True
            strategy_data['uniform_mtf_reverted_at'] = datetime.now(timezone.utc).isoformat()

            await redis_client.set(key, json.dumps(strategy_data))
            print(f"✅ Reverted {strategy_name} to uniform MTF")

        await redis_client.aclose()
        print("🎉 All strategies reverted to uniform MTF configuration")

    except Exception as e:
        print(f"❌ Revert failed: {e}")


async def main():
    """Main function."""
    print("🎯 Selective Multi-Timeframe Confirmation Manager")
    print("-" * 50)

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "status":
            await show_selective_status()
        elif command == "apply":
            await apply_selective_mtf()
        elif command == "revert":
            await revert_to_uniform_mtf()
        else:
            print("Usage: python selective_mtf_enable.py [status|apply|revert]")
            print()
            print("Commands:")
            print("  status  - Show current MTF configuration")
            print("  apply   - Apply selective MTF (disable for day trading)")
            print("  revert  - Revert to uniform MTF for all strategies")
            sys.exit(1)
    else:
        print("Available commands:")
        print("  status  - Show current configuration")
        print("  apply   - Apply selective MTF configuration")
        print("  revert  - Revert to uniform MTF")
        print()
        print("Usage: python selective_mtf_enable.py [command]")


if __name__ == "__main__":
    asyncio.run(main())
