-- Trade Execution Database Schema
-- Comprehensive tables for trade execution, positions, and performance tracking

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create trading schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS trading;

-- Positions table for tracking all position lifecycle
CREATE TABLE IF NOT EXISTS trading.positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ticker VARCHAR(10) NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    entry_price DECIMAL(12,4) NOT NULL,
    quantity INTEGER NOT NULL,
    stop_loss DECIMAL(12,4),
    take_profit DECIMAL(12,4),
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    strategy_type VARCHAR(50) NOT NULL,
    exit_time TIMESTAMPTZ,
    exit_price DECIMAL(12,4),
    pnl DECIMAL(12,4),
    commission DECIMAL(8,4) DEFAULT 0,
    signal_id UUID,
    account_id VARCHAR(50),
    side VARCHAR(10) NOT NULL, -- 'long' or 'short'
    unrealized_pnl DECIMAL(12,4) DEFAULT 0,
    market_value DECIMAL(12,4),
    cost_basis DECIMAL(12,4) NOT NULL,
    current_price DECIMAL(12,4),
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Orders table for comprehensive order tracking
CREATE TABLE IF NOT EXISTS trading.orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_order_id VARCHAR(100) UNIQUE,
    broker_order_id VARCHAR(100),
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop', 'stop_limit'
    quantity INTEGER NOT NULL,
    price DECIMAL(12,4),
    stop_price DECIMAL(12,4),
    filled_quantity INTEGER DEFAULT 0,
    filled_price DECIMAL(12,4),
    remaining_quantity INTEGER,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    time_in_force VARCHAR(10) DEFAULT 'day',
    extended_hours BOOLEAN DEFAULT FALSE,
    submitted_at TIMESTAMPTZ,
    filled_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    rejected_at TIMESTAMPTZ,
    commission DECIMAL(8,4) DEFAULT 0,
    position_id UUID REFERENCES trading.positions(id),
    signal_id UUID,
    strategy_name VARCHAR(50),
    order_class VARCHAR(20), -- 'simple', 'bracket', 'oco', 'oto'
    parent_order_id UUID REFERENCES trading.orders(id),
    legs JSONB, -- For bracket/complex orders
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    last_retry_at TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Fills table for tracking individual trade executions
CREATE TABLE IF NOT EXISTS trading.fills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES trading.orders(id),
    trade_id VARCHAR(100), -- Broker's trade ID
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    commission DECIMAL(8,4) DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL,
    liquidity VARCHAR(20), -- 'maker', 'taker', 'unknown'
    execution_venue VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trade performance tracking
CREATE TABLE IF NOT EXISTS trading.trade_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID NOT NULL REFERENCES trading.positions(id),
    symbol VARCHAR(10) NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    entry_date DATE NOT NULL,
    exit_date DATE,
    holding_period_hours INTEGER,
    entry_price DECIMAL(12,4) NOT NULL,
    exit_price DECIMAL(12,4),
    quantity INTEGER NOT NULL,
    gross_pnl DECIMAL(12,4),
    net_pnl DECIMAL(12,4),
    commission_total DECIMAL(8,4) DEFAULT 0,
    return_percentage DECIMAL(8,6),
    is_winner BOOLEAN,
    max_favorable_excursion DECIMAL(12,4), -- MFE
    max_adverse_excursion DECIMAL(12,4), -- MAE
    slippage DECIMAL(8,4),
    execution_quality_score DECIMAL(4,2),
    market_conditions JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Daily performance summary
CREATE TABLE IF NOT EXISTS trading.daily_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL UNIQUE,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    gross_pnl DECIMAL(12,4) DEFAULT 0,
    net_pnl DECIMAL(12,4) DEFAULT 0,
    commission_total DECIMAL(8,4) DEFAULT 0,
    largest_win DECIMAL(12,4) DEFAULT 0,
    largest_loss DECIMAL(12,4) DEFAULT 0,
    win_rate DECIMAL(6,4),
    profit_factor DECIMAL(8,4),
    expectancy DECIMAL(8,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    portfolio_value DECIMAL(15,8),
    cash_balance DECIMAL(15,8),
    buying_power DECIMAL(15,8),
    positions_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Execution errors and failures for debugging
CREATE TABLE IF NOT EXISTS trading.execution_errors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID REFERENCES trading.orders(id),
    signal_id UUID,
    error_type VARCHAR(50) NOT NULL,
    error_code VARCHAR(20),
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    context JSONB,
    retry_count INTEGER DEFAULT 0,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Bracket order relationships
CREATE TABLE IF NOT EXISTS trading.bracket_orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_order_id UUID NOT NULL REFERENCES trading.orders(id),
    entry_order_id UUID REFERENCES trading.orders(id),
    stop_loss_order_id UUID REFERENCES trading.orders(id),
    take_profit_order_id UUID REFERENCES trading.orders(id),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Execution metrics for TWAP/VWAP tracking
CREATE TABLE IF NOT EXISTS trading.execution_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    order_id UUID NOT NULL REFERENCES trading.orders(id),
    symbol VARCHAR(10) NOT NULL,
    execution_algorithm VARCHAR(20), -- 'TWAP', 'VWAP', 'IMMEDIATE', 'ICEBERG'
    benchmark_price DECIMAL(12,4), -- TWAP/VWAP benchmark
    average_execution_price DECIMAL(12,4),
    price_improvement DECIMAL(8,4),
    slippage DECIMAL(8,4),
    market_impact DECIMAL(8,4),
    timing_risk DECIMAL(8,4),
    bid_ask_spread_avg DECIMAL(8,4),
    volatility_during_execution DECIMAL(8,4),
    execution_duration_seconds INTEGER,
    participation_rate DECIMAL(6,4), -- % of volume
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Real-time position tracking for risk management
CREATE TABLE IF NOT EXISTS trading.position_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID NOT NULL REFERENCES trading.positions(id),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    current_price DECIMAL(12,4) NOT NULL,
    market_value DECIMAL(12,4) NOT NULL,
    unrealized_pnl DECIMAL(12,4) NOT NULL,
    unrealized_pnl_percentage DECIMAL(8,6),
    distance_to_stop_loss DECIMAL(8,6),
    distance_to_take_profit DECIMAL(8,6),
    bid_price DECIMAL(12,4),
    ask_price DECIMAL(12,4),
    bid_ask_spread DECIMAL(8,4),
    volume INTEGER,
    volatility DECIMAL(8,6),
    beta DECIMAL(8,4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trade signals correlation with execution
CREATE TABLE IF NOT EXISTS trading.signal_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signal_id UUID NOT NULL,
    signal_timestamp TIMESTAMPTZ NOT NULL,
    signal_price DECIMAL(12,4),
    signal_confidence DECIMAL(4,3),
    execution_timestamp TIMESTAMPTZ,
    execution_price DECIMAL(12,4),
    execution_delay_seconds INTEGER,
    price_slippage DECIMAL(8,4),
    execution_status VARCHAR(20),
    rejection_reason TEXT,
    position_id UUID REFERENCES trading.positions(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Account state tracking
CREATE TABLE IF NOT EXISTS trading.account_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cash DECIMAL(15,8) NOT NULL,
    buying_power DECIMAL(15,8) NOT NULL,
    total_equity DECIMAL(15,8) NOT NULL,
    total_market_value DECIMAL(15,8) NOT NULL,
    total_unrealized_pnl DECIMAL(15,8) NOT NULL,
    day_trades_count INTEGER DEFAULT 0,
    pattern_day_trader BOOLEAN DEFAULT FALSE,
    maintenance_margin DECIMAL(15,8) DEFAULT 0,
    initial_margin DECIMAL(15,8) DEFAULT 0,
    regt_buying_power DECIMAL(15,8) DEFAULT 0,
    daytrading_buying_power DECIMAL(15,8) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_positions_ticker ON trading.positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_status ON trading.positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON trading.positions(strategy_type);
CREATE INDEX IF NOT EXISTS idx_positions_entry_time ON trading.positions(entry_time);
CREATE INDEX IF NOT EXISTS idx_positions_exit_time ON trading.positions(exit_time);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON trading.orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading.orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_client_order_id ON trading.orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_broker_order_id ON trading.orders(broker_order_id);
CREATE INDEX IF NOT EXISTS idx_orders_submitted_at ON trading.orders(submitted_at);
CREATE INDEX IF NOT EXISTS idx_orders_position_id ON trading.orders(position_id);

CREATE INDEX IF NOT EXISTS idx_fills_order_id ON trading.fills(order_id);
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON trading.fills(symbol);
CREATE INDEX IF NOT EXISTS idx_fills_timestamp ON trading.fills(timestamp);

CREATE INDEX IF NOT EXISTS idx_trade_performance_symbol ON trading.trade_performance(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_performance_strategy ON trading.trade_performance(strategy_name);
CREATE INDEX IF NOT EXISTS idx_trade_performance_entry_date ON trading.trade_performance(entry_date);
CREATE INDEX IF NOT EXISTS idx_trade_performance_exit_date ON trading.trade_performance(exit_date);

CREATE INDEX IF NOT EXISTS idx_daily_performance_date ON trading.daily_performance(date);

CREATE INDEX IF NOT EXISTS idx_execution_errors_order_id ON trading.execution_errors(order_id);
CREATE INDEX IF NOT EXISTS idx_execution_errors_error_type ON trading.execution_errors(error_type);
CREATE INDEX IF NOT EXISTS idx_execution_errors_created_at ON trading.execution_errors(created_at);

CREATE INDEX IF NOT EXISTS idx_bracket_orders_parent ON trading.bracket_orders(parent_order_id);
CREATE INDEX IF NOT EXISTS idx_bracket_orders_status ON trading.bracket_orders(status);

CREATE INDEX IF NOT EXISTS idx_execution_metrics_order_id ON trading.execution_metrics(order_id);
CREATE INDEX IF NOT EXISTS idx_execution_metrics_symbol ON trading.execution_metrics(symbol);

CREATE INDEX IF NOT EXISTS idx_position_snapshots_position_id ON trading.position_snapshots(position_id);
CREATE INDEX IF NOT EXISTS idx_position_snapshots_timestamp ON trading.position_snapshots(timestamp);

CREATE INDEX IF NOT EXISTS idx_signal_executions_signal_id ON trading.signal_executions(signal_id);
CREATE INDEX IF NOT EXISTS idx_signal_executions_position_id ON trading.signal_executions(position_id);

CREATE INDEX IF NOT EXISTS idx_account_snapshots_account_id ON trading.account_snapshots(account_id);
CREATE INDEX IF NOT EXISTS idx_account_snapshots_timestamp ON trading.account_snapshots(timestamp);

-- Views for common queries
CREATE OR REPLACE VIEW trading.active_positions AS
SELECT
    p.*,
    ps.current_price,
    ps.unrealized_pnl,
    ps.unrealized_pnl_percentage,
    ps.distance_to_stop_loss,
    ps.distance_to_take_profit
FROM trading.positions p
LEFT JOIN LATERAL (
    SELECT DISTINCT ON (position_id) *
    FROM trading.position_snapshots
    WHERE position_id = p.id
    ORDER BY position_id, timestamp DESC
) ps ON TRUE
WHERE p.status = 'open';

CREATE OR REPLACE VIEW trading.daily_trade_summary AS
SELECT
    DATE(entry_time) as trade_date,
    strategy_type,
    COUNT(*) as total_trades,
    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
    COUNT(CASE WHEN pnl < 0 THEN 1 END) as losing_trades,
    SUM(pnl) as total_pnl,
    SUM(commission) as total_commission,
    AVG(pnl) as avg_pnl,
    MAX(pnl) as best_trade,
    MIN(pnl) as worst_trade,
    CASE
        WHEN COUNT(*) > 0 THEN
            ROUND(COUNT(CASE WHEN pnl > 0 THEN 1 END)::decimal / COUNT(*)::decimal, 4)
        ELSE 0
    END as win_rate
FROM trading.positions
WHERE status = 'closed' AND exit_time IS NOT NULL
GROUP BY DATE(entry_time), strategy_type
ORDER BY trade_date DESC, strategy_type;

CREATE OR REPLACE VIEW trading.position_risk_summary AS
SELECT
    ticker,
    SUM(CASE WHEN quantity > 0 THEN market_value ELSE 0 END) as long_exposure,
    SUM(CASE WHEN quantity < 0 THEN ABS(market_value) ELSE 0 END) as short_exposure,
    SUM(market_value) as net_exposure,
    COUNT(*) as position_count,
    SUM(unrealized_pnl) as total_unrealized_pnl,
    AVG(unrealized_pnl / NULLIF(cost_basis, 0)) as avg_return_pct
FROM trading.active_positions
GROUP BY ticker
ORDER BY ABS(net_exposure) DESC;

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_positions_updated_at
    BEFORE UPDATE ON trading.positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at
    BEFORE UPDATE ON trading.orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate portfolio metrics
CREATE OR REPLACE FUNCTION trading.calculate_portfolio_metrics(p_date DATE DEFAULT CURRENT_DATE)
RETURNS TABLE (
    date DATE,
    total_positions INTEGER,
    long_positions INTEGER,
    short_positions INTEGER,
    total_market_value DECIMAL(15,8),
    total_unrealized_pnl DECIMAL(15,8),
    largest_position_pct DECIMAL(6,4),
    concentration_risk DECIMAL(6,4)
) AS $$
BEGIN
    RETURN QUERY
    WITH position_stats AS (
        SELECT
            COUNT(*) as total_pos,
            COUNT(CASE WHEN quantity > 0 THEN 1 END) as long_pos,
            COUNT(CASE WHEN quantity < 0 THEN 1 END) as short_pos,
            SUM(market_value) as total_mv,
            SUM(unrealized_pnl) as total_upnl,
            MAX(ABS(market_value)) as largest_pos
        FROM trading.positions
        WHERE status = 'open'
        AND DATE(entry_time) <= p_date
    )
    SELECT
        p_date,
        ps.total_pos::INTEGER,
        ps.long_pos::INTEGER,
        ps.short_pos::INTEGER,
        ps.total_mv,
        ps.total_upnl,
        CASE
            WHEN ps.total_mv > 0 THEN (ps.largest_pos / ps.total_mv * 100)::DECIMAL(6,4)
            ELSE 0::DECIMAL(6,4)
        END,
        CASE
            WHEN ps.total_pos > 0 THEN (ps.largest_pos / ps.total_mv * 100)::DECIMAL(6,4)
            ELSE 0::DECIMAL(6,4)
        END
    FROM position_stats ps;
END;
$$ LANGUAGE plpgsql;

-- Function to update position snapshots
CREATE OR REPLACE FUNCTION trading.update_position_snapshot(
    p_position_id UUID,
    p_current_price DECIMAL(12,4),
    p_bid_price DECIMAL(12,4) DEFAULT NULL,
    p_ask_price DECIMAL(12,4) DEFAULT NULL,
    p_volume INTEGER DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    v_position RECORD;
    v_unrealized_pnl DECIMAL(12,4);
    v_unrealized_pnl_pct DECIMAL(8,6);
    v_market_value DECIMAL(12,4);
    v_distance_stop DECIMAL(8,6);
    v_distance_profit DECIMAL(8,6);
    v_bid_ask_spread DECIMAL(8,4);
BEGIN
    -- Get position details
    SELECT * INTO v_position
    FROM trading.positions
    WHERE id = p_position_id AND status = 'open';

    IF NOT FOUND THEN
        RETURN;
    END IF;

    -- Calculate metrics
    v_market_value := p_current_price * ABS(v_position.quantity);

    IF v_position.quantity > 0 THEN
        -- Long position
        v_unrealized_pnl := (p_current_price - v_position.entry_price) * v_position.quantity;
    ELSE
        -- Short position
        v_unrealized_pnl := (v_position.entry_price - p_current_price) * ABS(v_position.quantity);
    END IF;

    v_unrealized_pnl_pct := v_unrealized_pnl / NULLIF(v_position.cost_basis, 0);

    -- Calculate distances to stop/profit levels
    IF v_position.stop_loss IS NOT NULL THEN
        v_distance_stop := ABS(p_current_price - v_position.stop_loss) / p_current_price;
    END IF;

    IF v_position.take_profit IS NOT NULL THEN
        v_distance_profit := ABS(p_current_price - v_position.take_profit) / p_current_price;
    END IF;

    -- Calculate bid-ask spread
    IF p_bid_price IS NOT NULL AND p_ask_price IS NOT NULL THEN
        v_bid_ask_spread := (p_ask_price - p_bid_price) / ((p_bid_price + p_ask_price) / 2);
    END IF;

    -- Update position
    UPDATE trading.positions
    SET
        current_price = p_current_price,
        market_value = v_market_value,
        unrealized_pnl = v_unrealized_pnl,
        last_updated = NOW()
    WHERE id = p_position_id;

    -- Insert snapshot
    INSERT INTO trading.position_snapshots (
        position_id,
        timestamp,
        current_price,
        market_value,
        unrealized_pnl,
        unrealized_pnl_percentage,
        distance_to_stop_loss,
        distance_to_take_profit,
        bid_price,
        ask_price,
        bid_ask_spread,
        volume
    ) VALUES (
        p_position_id,
        NOW(),
        p_current_price,
        v_market_value,
        v_unrealized_pnl,
        v_unrealized_pnl_pct,
        v_distance_stop,
        v_distance_profit,
        p_bid_price,
        p_ask_price,
        v_bid_ask_spread,
        p_volume
    );
END;
$$ LANGUAGE plpgsql;

-- Function to close position and calculate final PnL
CREATE OR REPLACE FUNCTION trading.close_position(
    p_position_id UUID,
    p_exit_price DECIMAL(12,4),
    p_exit_time TIMESTAMPTZ DEFAULT NOW()
)
RETURNS DECIMAL(12,4) AS $$
DECLARE
    v_position RECORD;
    v_final_pnl DECIMAL(12,4);
    v_commission_total DECIMAL(8,4);
    v_gross_pnl DECIMAL(12,4);
    v_return_pct DECIMAL(8,6);
    v_holding_hours INTEGER;
    v_is_winner BOOLEAN;
BEGIN
    -- Get position details
    SELECT * INTO v_position
    FROM trading.positions
    WHERE id = p_position_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Position not found: %', p_position_id;
    END IF;

    -- Calculate total commission from associated orders
    SELECT COALESCE(SUM(commission), 0) INTO v_commission_total
    FROM trading.orders
    WHERE position_id = p_position_id;

    -- Calculate PnL
    IF v_position.quantity > 0 THEN
        -- Long position
        v_gross_pnl := (p_exit_price - v_position.entry_price) * v_position.quantity;
    ELSE
        -- Short position
        v_gross_pnl := (v_position.entry_price - p_exit_price) * ABS(v_position.quantity);
    END IF;

    v_final_pnl := v_gross_pnl - v_commission_total;
    v_return_pct := v_gross_pnl / NULLIF(v_position.cost_basis, 0);
    v_holding_hours := EXTRACT(EPOCH FROM (p_exit_time - v_position.entry_time)) / 3600;
    v_is_winner := v_final_pnl > 0;

    -- Update position
    UPDATE trading.positions
    SET
        status = 'closed',
        exit_time = p_exit_time,
        exit_price = p_exit_price,
        pnl = v_final_pnl,
        commission = v_commission_total,
        updated_at = NOW()
    WHERE id = p_position_id;

    -- Insert performance record
    INSERT INTO trading.trade_performance (
        position_id,
        symbol,
        strategy_name,
        entry_date,
        exit_date,
        holding_period_hours,
        entry_price,
        exit_price,
        quantity,
        gross_pnl,
        net_pnl,
        commission_total,
        return_percentage,
        is_winner
    ) VALUES (
        p_position_id,
        v_position.ticker,
        v_position.strategy_type,
        DATE(v_position.entry_time),
        DATE(p_exit_time),
        v_holding_hours,
        v_position.entry_price,
        p_exit_price,
        v_position.quantity,
        v_gross_pnl,
        v_final_pnl,
        v_commission_total,
        v_return_pct,
        v_is_winner
    );

    RETURN v_final_pnl;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your user)
GRANT USAGE ON SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA trading TO trader;
