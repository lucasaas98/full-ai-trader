#!/usr/bin/env python3
"""
Trigger Daily Summary via Redis Pub/Sub

This script publishes a message to Redis that the notification service
can listen for to trigger a manual daily summary notification.

Usage:
    python scripts/trigger_daily_summary_redis.py

Environment Variables Required:
    - REDIS_URL or REDIS_HOST/REDIS_PORT/REDIS_PASSWORD
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import redis.asyncio as redis

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def publish_daily_summary_trigger():
    """Publish a manual daily summary trigger via Redis."""
    redis_client = None
    try:
        # Get Redis connection info
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6380))
            redis_password = os.getenv('REDIS_PASSWORD', '')
            redis_db = int(os.getenv('REDIS_DB', 0))

            if redis_password:
                redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
            else:
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

        logger.info(f"Connecting to Redis...")
        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Test connection
        await redis_client.ping()
        logger.info("✅ Connected to Redis successfully")

        # Create manual trigger message
        trigger_message = {
            "type": "manual_daily_summary",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "manual_script",
            "message": "Manual daily summary trigger requested",
            "force": True
        }

        # Publish to a system control channel
        channel = "system:control"
        message_json = json.dumps(trigger_message)

        logger.info(f"Publishing manual daily summary trigger to channel: {channel}")
        result = await redis_client.publish(channel, message_json)

        if result > 0:
            logger.info(f"✅ Message published successfully! {result} subscribers received it.")
            print(f"\n🎉 Daily summary trigger sent successfully!")
            print(f"📡 Published to Redis channel: {channel}")
            print(f"👂 {result} subscribers received the message")
            print("\nThe notification service should process this and send a daily summary.")
            return True
        else:
            logger.warning("⚠️ Message published but no subscribers received it")
            print(f"\n⚠️ Message was published but no subscribers are listening to {channel}")
            print("This might mean the notification service is not running or not subscribed to this channel.")
            return False

    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        print(f"\n❌ Failed to connect to Redis: {e}")
        print("Make sure Redis is running and the connection details are correct.")
        return False
    except Exception as e:
        logger.error(f"Error publishing daily summary trigger: {e}")
        print(f"\n❌ Error: {e}")
        return False
    finally:
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                pass


async def send_direct_notification_trigger():
    """Send a direct trigger by publishing to the notification channels."""
    redis_client = None
    try:
        # Get Redis connection info
        redis_url = os.getenv('REDIS_URL')
        if not redis_url:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6380))
            redis_password = os.getenv('REDIS_PASSWORD', '')
            redis_db = int(os.getenv('REDIS_DB', 0))

            if redis_password:
                redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
            else:
                redis_url = f"redis://{redis_host}:{redis_port}/{redis_db}"

        logger.info(f"Connecting to Redis for direct notification trigger...")
        redis_client = redis.from_url(redis_url, decode_responses=True)

        # Test connection
        await redis_client.ping()
        logger.info("✅ Connected to Redis successfully")

        # Create a system status message that will trigger notifications
        status_message = {
            "type": "daily_summary_request",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "manual_trigger",
            "status": "requesting_daily_summary",
            "message": "Manual daily summary requested via Redis trigger",
            "severity": "info",
            "component": "system",
            "force_summary": True
        }

        # Publish to system status channel (which the notification service listens to)
        channel = "system:status"
        message_json = json.dumps(status_message)

        logger.info(f"Publishing daily summary request to channel: {channel}")
        result = await redis_client.publish(channel, message_json)

        if result > 0:
            logger.info(f"✅ Status message published successfully! {result} subscribers received it.")
            print(f"\n🎉 Daily summary request sent successfully!")
            print(f"📡 Published to Redis channel: {channel}")
            print(f"👂 {result} subscribers received the message")
            print("\nThe notification service should detect this system status and send a daily summary.")
            return True
        else:
            logger.warning("⚠️ Message published but no subscribers received it")
            print(f"\n⚠️ Message was published but no subscribers are listening to {channel}")
            print("The notification service might not be running.")
            return False

    except Exception as e:
        logger.error(f"Error sending direct notification trigger: {e}")
        print(f"\n❌ Error: {e}")
        return False
    finally:
        if redis_client:
            try:
                await redis_client.close()
            except Exception:
                pass


def check_environment():
    """Check if required Redis environment variables are set."""
    redis_url = os.getenv('REDIS_URL')
    redis_host = os.getenv('REDIS_HOST')

    if not redis_url and not redis_host:
        print("❌ Missing Redis configuration:")
        print("   Either set REDIS_URL or REDIS_HOST")
        print("   Example: REDIS_URL=redis://:password@localhost:6380/0")
        return False

    print("✅ Redis configuration found")
    return True


def main():
    """Main function."""
    print("🚀 Redis Daily Summary Trigger Script")
    print("=" * 50)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Load environment from .env file if it exists
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"📁 Loading environment from: {env_file}")
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            print("⚠️ python-dotenv not installed, using system environment variables")
    else:
        print("⚠️ No .env file found, using system environment variables")

    print("\n📡 Triggering daily summary via Redis...")

    # Try both methods
    async def run_triggers():
        print("\n1️⃣ Trying system control trigger...")
        success1 = await publish_daily_summary_trigger()

        print("\n2️⃣ Trying system status trigger...")
        success2 = await send_direct_notification_trigger()

        return success1 or success2

    try:
        success = asyncio.run(run_triggers())
        if success:
            print("\n✅ Redis triggers sent successfully!")
            print("\n💡 Next steps:")
            print("   1. Check the notification service logs:")
            print("      docker logs trading_notification_service --tail 20")
            print("   2. Check your Gotify notifications")
            print("   3. Wait a few moments for processing")
            sys.exit(0)
        else:
            print("\n❌ All Redis triggers failed!")
            print("\n🔧 Troubleshooting:")
            print("   1. Ensure the notification service is running:")
            print("      docker ps | grep notification")
            print("   2. Check Redis connectivity:")
            print("      docker logs trading_redis --tail 10")
            print("   3. Verify Redis configuration in .env file")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        logger.exception("Unexpected error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()
