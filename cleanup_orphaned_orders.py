#!/usr/bin/env python3
"""
Cleanup Orphaned Orders Script

This script identifies and cleans up orders in the local database that no longer
exist in the current Alpaca account. This typically happens when switching to a
new paper trading account.

Usage:
    python cleanup_orphaned_orders.py [--dry-run] [--force]

Options:
    --dry-run    Show what would be done without making changes
    --force      Skip confirmation prompts
    --help       Show this help message
"""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import asyncpg
import requests
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrphanedOrdersCleaner:
    """Clean up orders that don't exist in Alpaca but are in our database."""

    def __init__(self):
        self.alpaca_api_key = os.getenv("ALPACA_API_KEY")
        self.alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.alpaca_base_url = os.getenv(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        # Database connection details
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = int(os.getenv("DB_PORT", "5433"))
        self.db_name = os.getenv("DB_NAME", "trading_system_dev")
        self.db_user = os.getenv("DB_USER", "trader_dev")
        self.db_password = os.getenv("DB_PASSWORD")

        if not all([self.alpaca_api_key, self.alpaca_secret_key, self.db_password]):
            raise ValueError("Missing required environment variables")

    async def get_database_connection(self):
        """Get database connection."""
        try:
            conn = await asyncpg.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def get_alpaca_headers(self):
        """Get headers for Alpaca API requests."""
        return {
            "APCA-API-KEY-ID": self.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
            "Content-Type": "application/json",
        }

    async def check_order_exists_in_alpaca(self, order_id: str) -> bool:
        """Check if an order exists in Alpaca."""
        url = f"{self.alpaca_base_url}/v2/orders/{order_id}"
        headers = self.get_alpaca_headers()

        try:
            # Use requests in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, headers=headers)
            )

            if response.status_code == 200:
                return True
            elif response.status_code == 404:
                return False
            else:
                logger.warning(
                    f"Unexpected response for order {order_id}: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Error checking order {order_id}: {e}")
            return False

    async def get_all_database_orders(
        self, conn, days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get all orders from database within the specified time range."""
        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = """
        SELECT
            id,
            broker_order_id,
            symbol,
            status,
            created_at,
            updated_at
        FROM trading.orders
        WHERE created_at >= $1
            AND broker_order_id IS NOT NULL
            AND status IN ('pending', 'new', 'partially_filled', 'accepted')
        ORDER BY created_at DESC
        """

        try:
            rows = await conn.fetch(query, cutoff_date)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching database orders: {e}")
            return []

    async def get_all_alpaca_orders(self) -> List[str]:
        """Get all order IDs from Alpaca account."""
        url = f"{self.alpaca_base_url}/v2/orders"
        headers = self.get_alpaca_headers()
        params = {"status": "all", "limit": 500}  # Maximum allowed

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, headers=headers, params=params)
            )

            if response.status_code == 200:
                orders = response.json()
                return [order["id"] for order in orders]
            else:
                logger.error(f"Failed to fetch Alpaca orders: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Error fetching Alpaca orders: {e}")
            return []

    async def find_orphaned_orders(self, dry_run: bool = True) -> List[Dict[str, Any]]:
        """Find orders in database that don't exist in Alpaca."""
        logger.info("Connecting to database...")
        conn = await self.get_database_connection()

        try:
            logger.info("Fetching database orders...")
            db_orders = await self.get_all_database_orders(conn)
            logger.info(f"Found {len(db_orders)} orders in database")

            if not db_orders:
                logger.info("No orders found in database")
                return []

            logger.info("Fetching Alpaca orders...")
            alpaca_order_ids = await self.get_all_alpaca_orders()
            logger.info(f"Found {len(alpaca_order_ids)} orders in Alpaca account")

            # Convert to set for faster lookup
            alpaca_ids_set = set(alpaca_order_ids)

            orphaned_orders = []
            logger.info("Checking for orphaned orders...")

            for db_order in db_orders:
                broker_id = db_order["broker_order_id"]
                if broker_id not in alpaca_ids_set:
                    # Double-check by making individual API call
                    exists = await self.check_order_exists_in_alpaca(broker_id)
                    if not exists:
                        orphaned_orders.append(db_order)
                        logger.info(
                            f"Found orphaned order: {broker_id} ({db_order['symbol']})"
                        )

            logger.info(f"Found {len(orphaned_orders)} orphaned orders")
            return orphaned_orders

        finally:
            await conn.close()

    async def cleanup_orphaned_orders(
        self, orphaned_orders: List[Dict[str, Any]], dry_run: bool = True
    ) -> int:
        """Clean up orphaned orders from database."""
        if not orphaned_orders:
            logger.info("No orphaned orders to clean up")
            return 0

        logger.info(
            f"{'[DRY RUN] ' if dry_run else ''}Cleaning up {len(orphaned_orders)} orphaned orders..."
        )

        if dry_run:
            for order in orphaned_orders:
                logger.info(
                    f"Would update order {order['broker_order_id']} "
                    f"({order['symbol']}) to 'cancelled' status"
                )
            return len(orphaned_orders)

        conn = await self.get_database_connection()
        try:
            updated_count = 0
            async with conn.transaction():
                for order in orphaned_orders:
                    # Update order status to cancelled and add note
                    update_query = """
                    UPDATE trading.orders
                    SET
                        status = 'cancelled',
                        updated_at = NOW(),
                        error_message = COALESCE(error_message || ' | ', '') || 'Cancelled: Order not found in Alpaca account (orphaned)'
                    WHERE broker_order_id = $1
                    """

                    result = await conn.execute(update_query, order["broker_order_id"])
                    if result == "UPDATE 1":
                        updated_count += 1
                        logger.info(
                            f"Updated order {order['broker_order_id']} to cancelled status"
                        )
                    else:
                        logger.warning(
                            f"Failed to update order {order['broker_order_id']}"
                        )

            logger.info(f"Successfully updated {updated_count} orphaned orders")
            return updated_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
        finally:
            await conn.close()

    async def run_diagnostic(self) -> Dict[str, Any]:
        """Run diagnostic to check system health."""
        logger.info("Running diagnostic...")

        # Test Alpaca connection
        try:
            headers = self.get_alpaca_headers()
            url = f"{self.alpaca_base_url}/v2/account"
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: requests.get(url, headers=headers)
            )
            alpaca_connected = response.status_code == 200
            if alpaca_connected:
                account_data = response.json()
                logger.info(
                    f"Alpaca connection: OK (Account ID: {account_data.get('id', 'Unknown')})"
                )
            else:
                logger.error(f"Alpaca connection: FAILED ({response.status_code})")
        except Exception as e:
            logger.error(f"Alpaca connection: FAILED ({e})")
            alpaca_connected = False

        # Test database connection
        try:
            conn = await self.get_database_connection()
            await conn.fetchval("SELECT 1")
            await conn.close()
            db_connected = True
            logger.info("Database connection: OK")
        except Exception as e:
            logger.error(f"Database connection: FAILED ({e})")
            db_connected = False

        return {
            "alpaca_connected": alpaca_connected,
            "database_connected": db_connected,
            "timestamp": datetime.now().isoformat(),
        }


async def main():
    parser = argparse.ArgumentParser(
        description="Clean up orphaned orders from old Alpaca account"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--diagnostic", action="store_true", help="Run system diagnostic only"
    )

    args = parser.parse_args()

    try:
        cleaner = OrphanedOrdersCleaner()

        if args.diagnostic:
            diagnostic = await cleaner.run_diagnostic()
            logger.info(f"Diagnostic complete: {diagnostic}")
            return

        # Find orphaned orders
        orphaned_orders = await cleaner.find_orphaned_orders()

        if not orphaned_orders:
            logger.info("No orphaned orders found. System is clean!")
            return

        # Show summary
        print(f"\nFound {len(orphaned_orders)} orphaned orders:")
        for order in orphaned_orders[:10]:  # Show first 10
            print(
                f"  - {order['broker_order_id']} ({order['symbol']}) - "
                f"Status: {order['status']} - Created: {order['created_at']}"
            )

        if len(orphaned_orders) > 10:
            print(f"  ... and {len(orphaned_orders) - 10} more")

        # Confirm cleanup unless force flag is used
        if not args.force and not args.dry_run:
            confirm = input(
                f"\nDo you want to mark these {len(orphaned_orders)} orders as cancelled? (y/N): "
            )
            if confirm.lower() != "y":
                logger.info("Cleanup cancelled by user")
                return

        # Perform cleanup
        updated_count = await cleaner.cleanup_orphaned_orders(
            orphaned_orders, args.dry_run
        )

        if args.dry_run:
            logger.info(f"Dry run complete. Would have updated {updated_count} orders")
        else:
            logger.info(f"Cleanup complete. Updated {updated_count} orders")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
