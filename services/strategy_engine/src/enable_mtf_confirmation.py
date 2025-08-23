#!/usr/bin/env python3
"""
Enable Multi-Timeframe Confirmation Utility

This script enables multi-timeframe confirmation for existing strategies
running in the trading system. It updates strategy configurations and
provides options to test the new functionality.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))

import redis.asyncio as redis

from base_strategy import StrategyMode
from multi_timeframe_analyzer import create_multi_timeframe_analyzer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MTFConfigurationManager:
    """Manages multi-timeframe confirmation configuration for strategies."""

    def __init__(self, redis_url: str = None):
        """
        Initialize MTF configuration manager.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url or "redis://:Xo8uWxU1fmG0P1036pXysH2k4MTNhbmi@localhost:6380/0"
        self.redis_client = None

        # MTF configuration templates
        self.mtf_configs = {
            StrategyMode.DAY_TRADING: {
                "enable_mtf_confirmation": True,
                "mtf_min_timeframes": 3,
                "mtf_confidence_boost": 8.0,
                "mtf_confidence_penalty": 15.0,
                "mtf_required_strength": "moderate",
                "mtf_min_confidence": 65.0
            },
            StrategyMode.SWING_TRADING: {
                "enable_mtf_confirmation": True,
                "mtf_min_timeframes": 3,
                "mtf_confidence_boost": 10.0,
                "mtf_confidence_penalty": 20.0,
                "mtf_required_strength": "strong",
                "mtf_min_confidence": 70.0
            },
            StrategyMode.POSITION_TRADING: {
                "enable_mtf_confirmation": True,
                "mtf_min_timeframes": 4,
                "mtf_confidence_boost": 12.0,
                "mtf_confidence_penalty": 25.0,
                "mtf_required_strength": "strong",
                "mtf_min_confidence": 60.0
            }
        }

    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True
            )

            # Test connection
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            await self.redis_client.close()

    async def get_existing_strategies(self) -> List[Dict[str, Any]]:
        """Get all existing strategies from Redis."""
        try:
            strategy_keys = await self.redis_client.keys("strategy_state:*")
            strategies = []

            for key in strategy_keys:
                strategy_data = await self.redis_client.get(key)
                if strategy_data:
                    strategy_info = json.loads(strategy_data)
                    strategy_name = key.decode().replace("strategy_state:", "")
                    strategy_info['redis_key'] = key
                    strategy_info['name'] = strategy_name
                    strategies.append(strategy_info)

            return strategies

        except Exception as e:
            logger.error(f"Error getting existing strategies: {e}")
            return []

    async def enable_mtf_for_strategy(self, strategy_name: str, custom_config: Dict[str, Any] = None) -> bool:
        """
        Enable multi-timeframe confirmation for a specific strategy.

        Args:
            strategy_name: Name of the strategy to update
            custom_config: Optional custom MTF configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            strategy_key = f"strategy_state:{strategy_name}"

            # Get existing strategy configuration
            existing_data = await self.redis_client.get(strategy_key)
            if not existing_data:
                logger.error(f"Strategy {strategy_name} not found")
                return False

            strategy_data = json.loads(existing_data)

            # Determine strategy mode
            mode_str = strategy_data.get('config', {}).get('mode', 'swing_trading')
            try:
                strategy_mode = StrategyMode(mode_str)
            except ValueError:
                logger.warning(f"Unknown strategy mode {mode_str}, using SWING_TRADING")
                strategy_mode = StrategyMode.SWING_TRADING

            # Get appropriate MTF configuration
            mtf_config = custom_config or self.mtf_configs.get(
                strategy_mode,
                self.mtf_configs[StrategyMode.SWING_TRADING]
            )

            # Update strategy parameters
            if 'config' not in strategy_data:
                strategy_data['config'] = {}
            if 'parameters' not in strategy_data['config']:
                strategy_data['config']['parameters'] = {}

            # Merge MTF configuration
            strategy_data['config']['parameters'].update(mtf_config)

            # Add MTF metadata
            strategy_data['mtf_enabled'] = True
            strategy_data['mtf_enabled_at'] = datetime.now(timezone.utc).isoformat()
            strategy_data['mtf_config_version'] = "1.0"

            # Save updated configuration
            updated_data = json.dumps(strategy_data)
            await self.redis_client.set(strategy_key, updated_data)

            logger.info(f"Successfully enabled MTF confirmation for {strategy_name}")
            return True

        except Exception as e:
            logger.error(f"Error enabling MTF for strategy {strategy_name}: {e}")
            return False

    async def disable_mtf_for_strategy(self, strategy_name: str) -> bool:
        """
        Disable multi-timeframe confirmation for a specific strategy.

        Args:
            strategy_name: Name of the strategy to update

        Returns:
            True if successful, False otherwise
        """
        try:
            strategy_key = f"strategy_state:{strategy_name}"

            # Get existing strategy configuration
            existing_data = await self.redis_client.get(strategy_key)
            if not existing_data:
                logger.error(f"Strategy {strategy_name} not found")
                return False

            strategy_data = json.loads(existing_data)

            # Disable MTF parameters
            if 'config' in strategy_data and 'parameters' in strategy_data['config']:
                mtf_keys = [k for k in strategy_data['config']['parameters'].keys() if k.startswith('mtf_')]
                for key in mtf_keys:
                    del strategy_data['config']['parameters'][key]

                # Explicitly set enable flag to False
                strategy_data['config']['parameters']['enable_mtf_confirmation'] = False

            # Update MTF metadata
            strategy_data['mtf_enabled'] = False
            strategy_data['mtf_disabled_at'] = datetime.now(timezone.utc).isoformat()

            # Save updated configuration
            updated_data = json.dumps(strategy_data)
            await self.redis_client.set(strategy_key, updated_data)

            logger.info(f"Successfully disabled MTF confirmation for {strategy_name}")
            return True

        except Exception as e:
            logger.error(f"Error disabling MTF for strategy {strategy_name}: {e}")
            return False

    async def get_mtf_status(self) -> Dict[str, Any]:
        """Get MTF status for all strategies."""
        try:
            strategies = await self.get_existing_strategies()

            status = {
                'total_strategies': len(strategies),
                'mtf_enabled_count': 0,
                'mtf_disabled_count': 0,
                'strategies': {}
            }

            for strategy in strategies:
                name = strategy['name']
                config = strategy.get('config', {})
                parameters = config.get('parameters', {})

                mtf_enabled = parameters.get('enable_mtf_confirmation', False)
                mtf_metadata = {
                    'enabled': mtf_enabled,
                    'mode': config.get('mode', 'unknown'),
                    'mtf_enabled_at': strategy.get('mtf_enabled_at'),
                    'mtf_config_version': strategy.get('mtf_config_version')
                }

                if mtf_enabled:
                    status['mtf_enabled_count'] += 1
                    mtf_metadata['config'] = {
                        k: v for k, v in parameters.items()
                        if k.startswith('mtf_') or k == 'enable_mtf_confirmation'
                    }
                else:
                    status['mtf_disabled_count'] += 1

                status['strategies'][name] = mtf_metadata

            return status

        except Exception as e:
            logger.error(f"Error getting MTF status: {e}")
            return {'error': str(e)}

    async def validate_mtf_setup(self) -> Dict[str, Any]:
        """Validate that MTF system is properly set up."""
        try:
            validation_results = {
                'overall_status': 'unknown',
                'checks': {},
                'recommendations': []
            }

            # Check 1: Redis connectivity
            try:
                await self.redis_client.ping()
                validation_results['checks']['redis_connection'] = 'PASS'
            except Exception as e:
                validation_results['checks']['redis_connection'] = f'FAIL: {e}'
                validation_results['recommendations'].append('Fix Redis connection')

            # Check 2: Strategy configurations
            strategies = await self.get_existing_strategies()
            if strategies:
                validation_results['checks']['strategies_found'] = f'PASS: {len(strategies)} strategies'
            else:
                validation_results['checks']['strategies_found'] = 'FAIL: No strategies found'
                validation_results['recommendations'].append('Ensure strategies are properly configured')

            # Check 3: MTF analyzer creation
            try:
                analyzer, enhancer = create_multi_timeframe_analyzer(StrategyMode.SWING_TRADING)
                validation_results['checks']['mtf_analyzer'] = 'PASS'
            except Exception as e:
                validation_results['checks']['mtf_analyzer'] = f'FAIL: {e}'
                validation_results['recommendations'].append('Check MTF analyzer dependencies')

            # Determine overall status
            failed_checks = [k for k, v in validation_results['checks'].items() if v.startswith('FAIL')]
            if not failed_checks:
                validation_results['overall_status'] = 'READY'
            else:
                validation_results['overall_status'] = 'ISSUES_FOUND'
                validation_results['failed_checks'] = failed_checks

            return validation_results

        except Exception as e:
            logger.error(f"Error validating MTF setup: {e}")
            return {
                'overall_status': 'ERROR',
                'error': str(e),
                'checks': {},
                'recommendations': ['Check system configuration and dependencies']
            }


async def main():
    """Main function to handle command line operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enable Multi-Timeframe Confirmation for Trading Strategies"
    )

    parser.add_argument(
        'action',
        choices=['status', 'enable', 'disable', 'enable-all', 'disable-all', 'validate'],
        help='Action to perform'
    )

    parser.add_argument(
        '--strategy',
        help='Strategy name (required for enable/disable actions)'
    )

    parser.add_argument(
        '--redis-url',
        default="redis://:Xo8uWxU1fmG0P1036pXysH2k4MTNhbmi@localhost:6380/0",
        help='Redis connection URL'
    )

    parser.add_argument(
        '--confidence-boost',
        type=float,
        help='Custom confidence boost for MTF confirmation'
    )

    parser.add_argument(
        '--confidence-penalty',
        type=float,
        help='Custom confidence penalty for weak MTF confirmation'
    )

    parser.add_argument(
        '--min-timeframes',
        type=int,
        help='Minimum number of timeframes required for confirmation'
    )

    args = parser.parse_args()

    # Create configuration manager
    manager = MTFConfigurationManager(args.redis_url)

    try:
        await manager.connect()

        if args.action == 'status':
            status = await manager.get_mtf_status()
            print("\n" + "="*60)
            print("MULTI-TIMEFRAME CONFIRMATION STATUS")
            print("="*60)
            print(f"Total Strategies: {status['total_strategies']}")
            print(f"MTF Enabled: {status['mtf_enabled_count']}")
            print(f"MTF Disabled: {status['mtf_disabled_count']}")
            print()

            for name, info in status['strategies'].items():
                status_str = "ENABLED" if info['enabled'] else "DISABLED"
                print(f"  {name}: {status_str} (mode: {info['mode']})")
                if info['enabled'] and 'config' in info:
                    for key, value in info['config'].items():
                        print(f"    {key}: {value}")
                print()

        elif args.action == 'enable':
            if not args.strategy:
                print("Error: --strategy is required for enable action")
                sys.exit(1)

            # Build custom config if provided
            custom_config = {}
            if args.confidence_boost is not None:
                custom_config['mtf_confidence_boost'] = args.confidence_boost
            if args.confidence_penalty is not None:
                custom_config['mtf_confidence_penalty'] = args.confidence_penalty
            if args.min_timeframes is not None:
                custom_config['mtf_min_timeframes'] = args.min_timeframes

            success = await manager.enable_mtf_for_strategy(
                args.strategy,
                custom_config if custom_config else None
            )

            if success:
                print(f"✓ Multi-timeframe confirmation enabled for {args.strategy}")
            else:
                print(f"✗ Failed to enable multi-timeframe confirmation for {args.strategy}")
                sys.exit(1)

        elif args.action == 'disable':
            if not args.strategy:
                print("Error: --strategy is required for disable action")
                sys.exit(1)

            success = await manager.disable_mtf_for_strategy(args.strategy)

            if success:
                print(f"✓ Multi-timeframe confirmation disabled for {args.strategy}")
            else:
                print(f"✗ Failed to disable multi-timeframe confirmation for {args.strategy}")
                sys.exit(1)

        elif args.action == 'enable-all':
            strategies = await manager.get_existing_strategies()

            print(f"Enabling MTF confirmation for {len(strategies)} strategies...")

            success_count = 0
            for strategy in strategies:
                name = strategy['name']
                success = await manager.enable_mtf_for_strategy(name)
                if success:
                    success_count += 1
                    print(f"✓ {name}")
                else:
                    print(f"✗ {name}")

            print(f"\nEnabled MTF for {success_count}/{len(strategies)} strategies")

        elif args.action == 'disable-all':
            strategies = await manager.get_existing_strategies()

            print(f"Disabling MTF confirmation for {len(strategies)} strategies...")

            success_count = 0
            for strategy in strategies:
                name = strategy['name']
                success = await manager.disable_mtf_for_strategy(name)
                if success:
                    success_count += 1
                    print(f"✓ {name}")
                else:
                    print(f"✗ {name}")

            print(f"\nDisabled MTF for {success_count}/{len(strategies)} strategies")

        elif args.action == 'validate':
            print("Validating multi-timeframe confirmation setup...")

            validation = await manager.validate_mtf_setup()

            print("\n" + "="*60)
            print("VALIDATION RESULTS")
            print("="*60)
            print(f"Overall Status: {validation['overall_status']}")
            print()

            print("Checks:")
            for check, result in validation['checks'].items():
                status_symbol = "✓" if result.startswith("PASS") else "✗"
                print(f"  {status_symbol} {check}: {result}")

            if validation.get('recommendations'):
                print("\nRecommendations:")
                for rec in validation['recommendations']:
                    print(f"  • {rec}")

            print()

            if validation['overall_status'] != 'READY':
                sys.exit(1)

    finally:
        await manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
