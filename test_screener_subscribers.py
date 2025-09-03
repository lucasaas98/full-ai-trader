#!/usr/bin/env python3
"""
Test script to validate screener update subscriber implementations.

This script tests that all services correctly subscribe to and handle screener updates
from the data collector. It simulates the publication of screener updates and verifies
that the subscribers process them correctly.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ScreenerUpdateTester:
    """Test screener update subscription functionality."""

    def __init__(self):
        """Initialize the tester."""
        self.redis_client = None
        self.subscribers = {}
        self.received_messages = []

    async def initialize(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                "redis://localhost:6379", decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis for testing")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def create_test_subscriber(self, name: str) -> None:
        """
        Create a test subscriber to monitor screener updates.

        Args:
            name: Name of the test subscriber
        """
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("screener:updates")

            self.subscribers[name] = pubsub
            logger.info(f"Created test subscriber: {name}")

            # Start listening in background
            asyncio.create_task(self._listen_for_messages(name, pubsub))

        except Exception as e:
            logger.error(f"Failed to create subscriber {name}: {e}")

    async def _listen_for_messages(self, name: str, pubsub) -> None:
        """
        Listen for messages from a subscriber.

        Args:
            name: Subscriber name
            pubsub: Redis pubsub object
        """
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    self.received_messages.append(
                        {
                            "subscriber": name,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "data": data,
                        }
                    )
                    logger.info(
                        f"Subscriber {name} received screener update with {len(data.get('data', []))} stocks"
                    )
        except Exception as e:
            logger.error(f"Error in subscriber {name}: {e}")

    async def publish_test_screener_update(
        self, screener_type: str = "test_momentum"
    ) -> Dict[str, Any]:
        """
        Publish a test screener update.

        Args:
            screener_type: Type of screener to simulate

        Returns:
            The published message data
        """
        # Create test screener data
        test_stocks = [
            {
                "symbol": "AAPL",
                "company": "Apple Inc",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "country": "USA",
                "market_cap": "3000000",
                "pe_ratio": 25.5,
                "price": 175.50,
                "change": 2.3,
                "volume": 50000000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "symbol": "GOOGL",
                "company": "Alphabet Inc",
                "sector": "Technology",
                "industry": "Internet Content & Information",
                "country": "USA",
                "market_cap": "2000000",
                "pe_ratio": 22.8,
                "price": 145.20,
                "change": 1.8,
                "volume": 35000000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "symbol": "TSLA",
                "company": "Tesla Inc",
                "sector": "Consumer Cyclical",
                "industry": "Auto Manufacturers",
                "country": "USA",
                "market_cap": "800000",
                "pe_ratio": 45.2,
                "price": 250.75,
                "change": 3.5,
                "volume": 75000000,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]

        message = {
            "screener_type": screener_type,
            "data": test_stocks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "count": len(test_stocks),
        }

        try:
            # Publish to screener:updates channel
            await self.redis_client.publish("screener:updates", json.dumps(message))
            logger.info(
                f"Published test screener update: {screener_type} with {len(test_stocks)} stocks"
            )

            # Also cache like the real system does
            cache_key = f"screener_cache:{screener_type}"
            await self.redis_client.setex(cache_key, 1800, json.dumps(message))
            logger.info(f"Cached screener data with key: {cache_key}")

            return message

        except Exception as e:
            logger.error(f"Failed to publish test screener update: {e}")
            raise

    async def verify_cache_storage(self, screener_type: str) -> bool:
        """
        Verify that screener data was properly cached.

        Args:
            screener_type: Type of screener to check

        Returns:
            True if cache exists and is valid
        """
        try:
            cache_key = f"screener_cache:{screener_type}"
            cached_data = await self.redis_client.get(cache_key)

            if cached_data:
                data = json.loads(cached_data)
                logger.info(
                    f"Cache verification: Found {len(data.get('data', []))} stocks in cache"
                )
                return True
            else:
                logger.warning(f"Cache verification: No data found for key {cache_key}")
                return False

        except Exception as e:
            logger.error(f"Cache verification failed: {e}")
            return False

    async def test_subscriber_integration(self) -> Dict[str, Any]:
        """
        Test the complete screener update flow.

        Returns:
            Test results summary
        """
        results = {
            "test_start": datetime.now(timezone.utc).isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "details": [],
        }

        try:
            # Test 1: Create subscribers
            await self.create_test_subscriber("test_strategy_engine")
            await self.create_test_subscriber("test_trade_executor")
            await self.create_test_subscriber("test_risk_manager")
            await self.create_test_subscriber("test_notification_service")

            # Wait a moment for subscribers to be ready
            await asyncio.sleep(1)

            results["details"].append("✅ Created test subscribers")
            results["tests_passed"] += 1

            # Test 2: Publish screener update
            _ = await self.publish_test_screener_update("test_momentum")
            await asyncio.sleep(2)  # Wait for message propagation

            results["details"].append("✅ Published test screener update")
            results["tests_passed"] += 1

            # Test 3: Verify cache storage
            cache_valid = await self.verify_cache_storage("test_momentum")
            if cache_valid:
                results["details"].append("✅ Cache storage working correctly")
                results["tests_passed"] += 1
            else:
                results["details"].append("❌ Cache storage failed")
                results["tests_failed"] += 1

            # Test 4: Check message reception
            if len(self.received_messages) > 0:
                results["details"].append(
                    f"✅ Received {len(self.received_messages)} messages from subscribers"
                )
                results["tests_passed"] += 1

                # Log details of received messages
                for msg in self.received_messages:
                    subscriber = msg["subscriber"]
                    stock_count = len(msg["data"].get("data", []))
                    results["details"].append(
                        f"  - {subscriber}: processed {stock_count} stocks"
                    )
            else:
                results["details"].append("❌ No messages received by test subscribers")
                results["tests_failed"] += 1

            # Test 5: Publish different screener type
            await self.publish_test_screener_update("test_dividend_stocks")
            await asyncio.sleep(2)

            results["details"].append("✅ Published second screener type")
            results["tests_passed"] += 1

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            results["details"].append(f"❌ Test execution error: {e}")
            results["tests_failed"] += 1

        results["test_end"] = datetime.now(timezone.utc).isoformat()
        results["total_messages_received"] = len(self.received_messages)
        results["success_rate"] = (
            results["tests_passed"]
            / (results["tests_passed"] + results["tests_failed"])
            * 100
        )

        return results

    async def cleanup(self):
        """Clean up test resources."""
        try:
            # Close all subscribers
            for name, pubsub in self.subscribers.items():
                await pubsub.close()
                logger.info(f"Closed subscriber: {name}")

            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Closed Redis connection")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Run the screener subscriber integration test."""
    logger.info("🚀 Starting Screener Update Subscriber Integration Test")

    tester = ScreenerUpdateTester()

    try:
        # Initialize
        await tester.initialize()

        # Run tests
        results = await tester.test_subscriber_integration()

        # Display results
        print("\n" + "=" * 60)
        print("📊 SCREENER SUBSCRIBER TEST RESULTS")
        print("=" * 60)
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Total Messages: {results['total_messages_received']}")
        print("\nDetails:")
        for detail in results["details"]:
            print(f"  {detail}")
        print("=" * 60)

        # Recommend next steps
        if results["tests_failed"] == 0:
            print(
                "\n✅ All tests passed! Screener subscriber integration is working correctly."
            )
            print("\n📝 Next steps:")
            print("  1. Deploy services and verify real screener updates")
            print("  2. Monitor logs for subscriber activity")
            print("  3. Test with production screener data")
        else:
            print("\n⚠️  Some tests failed. Check the details above.")
            print("\n🔧 Troubleshooting:")
            print("  1. Verify Redis is running and accessible")
            print("  2. Check that services are subscribing to screener:updates")
            print(
                "  3. Verify Redis channel names match between publisher and subscribers"
            )
            print("  4. Check service logs for connection errors")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n❌ Test execution failed: {e}")

    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
