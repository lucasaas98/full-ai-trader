-- Integration Test Database Initialization Script
-- Creates users, databases, and schema for integration testing

-- Create integration test user
CREATE USER trader_integration WITH PASSWORD 'integration_test_password_2024';

-- Create integration test database
CREATE DATABASE trading_system_integration_test WITH OWNER trader_integration;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE trading_system_integration_test TO trader_integration;

-- Connect to the integration test database
\c trading_system_integration_test;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO trader_integration;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader_integration;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader_integration;

-- Create tables for trading system
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolios (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    initial_value DECIMAL(15, 2) NOT NULL,
    current_value DECIMAL(15, 2) NOT NULL DEFAULT 0,
    cash_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    quantity DECIMAL(15, 8) NOT NULL,
    average_price DECIMAL(15, 8) NOT NULL,
    current_price DECIMAL(15, 8),
    market_value DECIMAL(15, 2),
    unrealized_pnl DECIMAL(15, 2),
    realized_pnl DECIMAL(15, 2) DEFAULT 0,
    position_type VARCHAR(10) DEFAULT 'LONG',
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    order_type VARCHAR(20) NOT NULL,
    quantity DECIMAL(15, 8) NOT NULL,
    price DECIMAL(15, 8),
    stop_price DECIMAL(15, 8),
    time_in_force VARCHAR(10) DEFAULT 'DAY',
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    filled_quantity DECIMAL(15, 8) DEFAULT 0,
    filled_price DECIMAL(15, 8),
    commission DECIMAL(10, 4) DEFAULT 0,
    external_order_id VARCHAR(100),
    placed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(4) NOT NULL,
    quantity DECIMAL(15, 8) NOT NULL,
    price DECIMAL(15, 8) NOT NULL,
    commission DECIMAL(10, 4) DEFAULT 0,
    executed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    snapshot_date DATE NOT NULL,
    total_value DECIMAL(15, 2) NOT NULL,
    cash_balance DECIMAL(15, 2) NOT NULL,
    positions_value DECIMAL(15, 2) NOT NULL,
    daily_pnl DECIMAL(15, 2),
    total_pnl DECIMAL(15, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(portfolio_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    open_price DECIMAL(15, 8) NOT NULL,
    high_price DECIMAL(15, 8) NOT NULL,
    low_price DECIMAL(15, 8) NOT NULL,
    close_price DECIMAL(15, 8) NOT NULL,
    volume BIGINT NOT NULL,
    vwap DECIMAL(15, 8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, timeframe)
);

CREATE TABLE IF NOT EXISTS technical_indicators (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    sma_20 DECIMAL(15, 8),
    sma_50 DECIMAL(15, 8),
    sma_200 DECIMAL(15, 8),
    ema_12 DECIMAL(15, 8),
    ema_26 DECIMAL(15, 8),
    rsi DECIMAL(10, 4),
    macd DECIMAL(15, 8),
    macd_signal DECIMAL(15, 8),
    macd_histogram DECIMAL(15, 8),
    bollinger_upper DECIMAL(15, 8),
    bollinger_middle DECIMAL(15, 8),
    bollinger_lower DECIMAL(15, 8),
    atr DECIMAL(15, 8),
    volume_sma DECIMAL(20, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timestamp, timeframe)
);

CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    parameters JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS strategy_signals (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(4) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    strength DECIMAL(5, 4) NOT NULL,
    confidence DECIMAL(5, 4) NOT NULL,
    price DECIMAL(15, 8),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    var_95 DECIMAL(15, 2),
    var_99 DECIMAL(15, 2),
    expected_shortfall DECIMAL(15, 2),
    sharpe_ratio DECIMAL(10, 6),
    sortino_ratio DECIMAL(10, 6),
    max_drawdown DECIMAL(10, 6),
    volatility DECIMAL(10, 6),
    beta DECIMAL(10, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(portfolio_id, date)
);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'INFO',
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB,
    is_read BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_positions_portfolio_symbol ON positions(portfolio_id, symbol);
CREATE INDEX idx_orders_portfolio_status ON orders(portfolio_id, status);
CREATE INDEX idx_orders_symbol_status ON orders(symbol, status);
CREATE INDEX idx_trades_portfolio_symbol ON trades(portfolio_id, symbol);
CREATE INDEX idx_trades_executed_at ON trades(executed_at);
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_technical_indicators_symbol_timestamp ON technical_indicators(symbol, timestamp);
CREATE INDEX idx_strategy_signals_symbol_timestamp ON strategy_signals(symbol, timestamp);
CREATE INDEX idx_portfolio_snapshots_date ON portfolio_snapshots(snapshot_date);
CREATE INDEX idx_risk_metrics_date ON risk_metrics(date);

-- Grant all privileges on newly created tables to integration user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader_integration;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader_integration;

-- Create a test portfolio for integration tests
INSERT INTO users (username, email, password_hash)
VALUES ('integration_test_user', 'integration@test.com', '$2b$12$test_hash')
ON CONFLICT (username) DO NOTHING;

INSERT INTO portfolios (user_id, name, initial_value, current_value, cash_balance)
SELECT id, 'Integration Test Portfolio', 100000.00, 100000.00, 100000.00
FROM users WHERE username = 'integration_test_user'
ON CONFLICT DO NOTHING;

-- Create test strategy
INSERT INTO strategies (name, description, parameters)
VALUES ('Integration Test Strategy', 'Simple strategy for integration testing', '{"test": true}')
ON CONFLICT (name) DO NOTHING;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Integration test database initialized successfully';
    RAISE NOTICE 'Database: trading_system_integration_test';
    RAISE NOTICE 'User: trader_integration';
    RAISE NOTICE 'Tables created: %', (SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public');
END $$;
