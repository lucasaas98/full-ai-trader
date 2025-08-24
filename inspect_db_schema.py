#!/usr/bin/env python3
"""
Database Schema Inspector

This script inspects the database schema to understand the structure of tables,
particularly the orders table to help with cleanup operations.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv
import asyncpg

# Load environment variables
load_dotenv()

async def inspect_database():
    """Inspect database schema and show relevant table structures."""

    # Database connection details
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = int(os.getenv('DB_PORT', '5433'))
    db_name = os.getenv('DB_NAME', 'trading_system_dev')
    db_user = os.getenv('DB_USER', 'trader_dev')
    db_password = os.getenv('DB_PASSWORD')

    if not db_password:
        print("Error: DB_PASSWORD environment variable not set")
        sys.exit(1)

    try:
        conn = await asyncpg.connect(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )

        print("=== DATABASE SCHEMA INSPECTION ===\n")

        # Check if trading schema exists
        schemas = await conn.fetch("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name IN ('trading', 'public')
        """)

        print("Available schemas:")
        for schema in schemas:
            print(f"  - {schema['schema_name']}")
        print()

        # List all tables in trading and public schemas
        tables = await conn.fetch("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ('trading', 'public')
            AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
        """)

        print("Available tables:")
        for table in tables:
            print(f"  - {table['table_schema']}.{table['table_name']}")
        print()

        # Check for orders-related tables
        order_tables = await conn.fetch("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ('trading', 'public')
            AND table_name ILIKE '%order%'
            AND table_type = 'BASE TABLE'
        """)

        print("Order-related tables:")
        for table in order_tables:
            print(f"  - {table['table_schema']}.{table['table_name']}")
        print()

        # Inspect each order-related table
        for table in order_tables:
            schema_name = table['table_schema']
            table_name = table['table_name']
            full_table_name = f"{schema_name}.{table_name}"

            print(f"=== STRUCTURE OF {full_table_name.upper()} ===")

            # Get column information
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = $2
                ORDER BY ordinal_position
            """, schema_name, table_name)

            if columns:
                print("Columns:")
                for col in columns:
                    nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                    default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                    print(f"  - {col['column_name']}: {col['data_type']} {nullable}{default}")

                # Get sample data count
                try:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {full_table_name}")
                    print(f"\nTotal rows: {count}")

                    if count > 0:
                        # Show sample data
                        sample_data = await conn.fetch(f"SELECT * FROM {full_table_name} LIMIT 3")
                        print("\nSample data (first 3 rows):")
                        for i, row in enumerate(sample_data, 1):
                            print(f"  Row {i}:")
                            for key, value in row.items():
                                # Truncate long values
                                display_value = str(value)
                                if len(display_value) > 50:
                                    display_value = display_value[:47] + "..."
                                print(f"    {key}: {display_value}")
                            print()

                except Exception as e:
                    print(f"Error fetching sample data: {e}")
            else:
                print("No columns found")

            print("-" * 50)
            print()

        # Check for positions table as well (might contain order references)
        positions_exists = await conn.fetchval("""
            SELECT EXISTS(
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'trading' AND table_name = 'positions'
            )
        """)

        if positions_exists:
            print("=== STRUCTURE OF TRADING.POSITIONS ===")
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'trading' AND table_name = 'positions'
                ORDER BY ordinal_position
            """)

            print("Columns:")
            for col in columns:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                print(f"  - {col['column_name']}: {col['data_type']} {nullable}{default}")

            count = await conn.fetchval("SELECT COUNT(*) FROM trading.positions")
            print(f"\nTotal rows: {count}")

            if count > 0:
                # Check for any status patterns
                status_counts = await conn.fetch("""
                    SELECT status, COUNT(*) as count
                    FROM trading.positions
                    GROUP BY status
                    ORDER BY count DESC
                """)

                print("\nStatus distribution:")
                for row in status_counts:
                    print(f"  - {row['status']}: {row['count']}")

        print("\n=== INSPECTION COMPLETE ===")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            await conn.close()

if __name__ == '__main__':
    asyncio.run(inspect_database())
