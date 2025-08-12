-- Database initialization script for trading system
-- This script sets up the initial database structure and users

-- Create extensions if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create database users if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'trader_dev') THEN
        CREATE USER trader_dev WITH PASSWORD '0{pc3TzATw2m>,[l';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'trader') THEN
        CREATE USER trader WITH PASSWORD 'default_password';
    END IF;
END $$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE trading_system TO trader_dev;
GRANT CONNECT ON DATABASE trading_system TO trader;

-- Grant schema permissions
GRANT USAGE ON SCHEMA public TO trader_dev;
GRANT USAGE ON SCHEMA public TO trader;
GRANT CREATE ON SCHEMA public TO trader_dev;
GRANT CREATE ON SCHEMA public TO trader;

-- Grant table permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trader_dev;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO trader;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trader_dev;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO trader;

-- Set default permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trader_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO trader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO trader_dev;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO trader;

-- Create basic tables structure (if not using alembic migrations)
-- These are placeholder tables that will be created by the application

-- Market data tables
CREATE TABLE IF NOT EXISTS market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(10,4) NOT NULL,
    high_price DECIMAL(10,4) NOT NULL,
    low_price DECIMAL(10,4) NOT NULL,
    close_price DECIMAL(10,4) NOT NULL,
    volume BIGINT NOT NULL,
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp, source)
);

-- Portfolio positions
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    average_price DECIMAL(10,4) NOT NULL,
    market_value DECIMAL(18,4) NOT NULL,
    unrealized_pnl DECIMAL(18,4) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(10) NOT NULL CHECK (signal_type IN ('buy', 'sell', 'hold')),
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    price DECIMAL(10,4) NOT NULL,
    quantity DECIMAL(18,8),
    strategy_name VARCHAR(100) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(10,4),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    broker_order_id VARCHAR(100),
    filled_quantity DECIMAL(18,8) DEFAULT 0,
    average_fill_price DECIMAL(10,4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trades
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID REFERENCES orders(id),
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    commission DECIMAL(10,4) DEFAULT 0,
    trade_time TIMESTAMPTZ NOT NULL,
    broker_trade_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Risk metrics
CREATE TABLE IF NOT EXISTS risk_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    portfolio_value DECIMAL(18,4) NOT NULL,
    total_exposure DECIMAL(18,4) NOT NULL,
    max_drawdown DECIMAL(10,4) NOT NULL,
    var_1day DECIMAL(18,4),
    var_5day DECIMAL(18,4),
    calculated_at TIMESTAMPTZ DEFAULT NOW()
);

-- System logs
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_created ON trading_signals(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_trade_time ON trades(symbol, trade_time DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_service_timestamp ON system_logs(service_name, timestamp DESC);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some initial data for development
INSERT INTO system_logs (service_name, level, message, metadata)
VALUES ('database', 'INFO', 'Database initialization completed', '{"version": "1.0", "environment": "development"}')
ON CONFLICT DO NOTHING;

-- Show completion message
SELECT 'Database initialization completed successfully' AS status;
