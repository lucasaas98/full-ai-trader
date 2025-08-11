-- Risk Management Database Schema
-- Additional tables for comprehensive risk management functionality

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create risk schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS risk;

-- Enhanced risk events table
CREATE TABLE IF NOT EXISTS risk.risk_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    symbol VARCHAR(20),
    description TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE,
    action_taken VARCHAR(200),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enhanced portfolio snapshots table
CREATE TABLE IF NOT EXISTS risk.portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cash DECIMAL(15,8) NOT NULL,
    buying_power DECIMAL(15,8) NOT NULL,
    total_equity DECIMAL(15,8) NOT NULL,
    total_market_value DECIMAL(15,8) NOT NULL,
    total_unrealized_pnl DECIMAL(15,8) NOT NULL,
    day_trades_count INTEGER DEFAULT 0,
    pattern_day_trader BOOLEAN DEFAULT FALSE,
    positions JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio metrics table for detailed risk tracking
CREATE TABLE IF NOT EXISTS risk.portfolio_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_exposure DECIMAL(15,8) NOT NULL,
    cash_percentage DECIMAL(8,6) NOT NULL,
    position_count INTEGER NOT NULL,
    concentration_risk DECIMAL(8,6) NOT NULL,
    portfolio_beta DECIMAL(8,6) NOT NULL,
    portfolio_correlation DECIMAL(8,6) NOT NULL,
    value_at_risk_1d DECIMAL(15,8) NOT NULL,
    value_at_risk_5d DECIMAL(15,8) NOT NULL,
    expected_shortfall DECIMAL(15,8) NOT NULL,
    sharpe_ratio DECIMAL(8,6) NOT NULL,
    max_drawdown DECIMAL(8,6) NOT NULL,
    current_drawdown DECIMAL(8,6) NOT NULL,
    volatility DECIMAL(8,6) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Position-level risk metrics
CREATE TABLE IF NOT EXISTS risk.position_risks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    symbol VARCHAR(20) NOT NULL,
    position_size DECIMAL(15,8) NOT NULL,
    portfolio_percentage DECIMAL(8,6) NOT NULL,
    volatility DECIMAL(8,6) NOT NULL,
    beta DECIMAL(8,6) NOT NULL,
    var_1d DECIMAL(15,8) NOT NULL,
    expected_return DECIMAL(8,6) NOT NULL,
    sharpe_ratio DECIMAL(8,6) NOT NULL,
    correlation_with_portfolio DECIMAL(8,6) NOT NULL,
    sector VARCHAR(50),
    risk_score DECIMAL(4,2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk alerts management
CREATE TABLE IF NOT EXISTS risk.risk_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    symbol VARCHAR(20),
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(100),
    action_required BOOLEAN DEFAULT FALSE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Daily risk reports
CREATE TABLE IF NOT EXISTS risk.daily_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_date DATE NOT NULL UNIQUE,
    portfolio_value DECIMAL(15,8) NOT NULL,
    daily_pnl DECIMAL(15,8) NOT NULL,
    daily_return DECIMAL(8,6) NOT NULL,
    max_drawdown DECIMAL(8,6) NOT NULL,
    current_drawdown DECIMAL(8,6) NOT NULL,
    volatility DECIMAL(8,6) NOT NULL,
    sharpe_ratio DECIMAL(8,6) NOT NULL,
    var_1d DECIMAL(15,8) NOT NULL,
    total_trades INTEGER NOT NULL,
    winning_trades INTEGER NOT NULL,
    risk_events_count INTEGER NOT NULL,
    compliance_violations JSONB,
    report_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trailing stops configuration
CREATE TABLE IF NOT EXISTS risk.trailing_stops (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL UNIQUE,
    enabled BOOLEAN DEFAULT TRUE,
    trail_percentage DECIMAL(8,6) NOT NULL,
    current_stop_price DECIMAL(15,8) NOT NULL,
    highest_price DECIMAL(15,8) NOT NULL,
    entry_price DECIMAL(15,8) NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Position sizing history
CREATE TABLE IF NOT EXISTS risk.position_sizing_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    signal_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    recommended_shares INTEGER NOT NULL,
    recommended_value DECIMAL(15,8) NOT NULL,
    position_percentage DECIMAL(8,6) NOT NULL,
    confidence_score DECIMAL(4,3),
    volatility_adjustment DECIMAL(4,3) NOT NULL,
    sizing_method VARCHAR(30) NOT NULL,
    max_loss_amount DECIMAL(15,8) NOT NULL,
    risk_reward_ratio DECIMAL(8,4) NOT NULL,
    portfolio_value DECIMAL(15,8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk limits history (for tracking changes)
CREATE TABLE IF NOT EXISTS risk.risk_limits_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    change_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    changed_by VARCHAR(100),
    previous_limits JSONB,
    new_limits JSONB,
    change_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Circuit breaker events
CREATE TABLE IF NOT EXISTS risk.circuit_breaker_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trigger_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    trigger_type VARCHAR(50) NOT NULL,
    trigger_value DECIMAL(15,8),
    threshold_value DECIMAL(15,8),
    duration_minutes INTEGER DEFAULT 15,
    deactivated_at TIMESTAMP WITH TIME ZONE,
    deactivated_by VARCHAR(100),
    portfolio_impact JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance attribution tracking
CREATE TABLE IF NOT EXISTS risk.performance_attribution (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    calculation_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    weight DECIMAL(8,6) NOT NULL,
    return_contribution DECIMAL(8,6) NOT NULL,
    risk_contribution DECIMAL(8,6) NOT NULL,
    alpha DECIMAL(8,6),
    beta DECIMAL(8,6),
    sector VARCHAR(50),
    style_factor VARCHAR(30),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(calculation_date, symbol)
);

-- Create comprehensive indexes for performance
CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk.risk_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_events_type_severity ON risk.risk_events(event_type, severity);
CREATE INDEX IF NOT EXISTS idx_risk_events_symbol ON risk.risk_events(symbol);
CREATE INDEX IF NOT EXISTS idx_risk_events_resolved ON risk.risk_events(resolved_at) WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp ON risk.portfolio_snapshots(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_account ON risk.portfolio_snapshots(account_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_portfolio_metrics_timestamp ON risk.portfolio_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_portfolio_metrics_concentration ON risk.portfolio_metrics(concentration_risk) WHERE concentration_risk > 0.6;
CREATE INDEX IF NOT EXISTS idx_portfolio_metrics_drawdown ON risk.portfolio_metrics(current_drawdown) WHERE current_drawdown > 0.05;

CREATE INDEX IF NOT EXISTS idx_position_risks_timestamp ON risk.position_risks(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_position_risks_symbol ON risk.position_risks(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_position_risks_score ON risk.position_risks(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_position_risks_sector ON risk.position_risks(sector);

CREATE INDEX IF NOT EXISTS idx_risk_alerts_timestamp ON risk.risk_alerts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_severity ON risk.risk_alerts(severity, acknowledged);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_unack ON risk.risk_alerts(acknowledged, timestamp DESC) WHERE acknowledged = FALSE;
CREATE INDEX IF NOT EXISTS idx_risk_alerts_action_req ON risk.risk_alerts(action_required, acknowledged) WHERE action_required = TRUE;

CREATE INDEX IF NOT EXISTS idx_daily_reports_date ON risk.daily_reports(report_date DESC);

CREATE INDEX IF NOT EXISTS idx_trailing_stops_symbol ON risk.trailing_stops(symbol);
CREATE INDEX IF NOT EXISTS idx_trailing_stops_enabled ON risk.trailing_stops(enabled) WHERE enabled = TRUE;

CREATE INDEX IF NOT EXISTS idx_position_sizing_symbol ON risk.position_sizing_history(symbol, signal_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_position_sizing_timestamp ON risk.position_sizing_history(signal_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_risk_limits_timestamp ON risk.risk_limits_history(change_timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_circuit_breaker_timestamp ON risk.circuit_breaker_events(trigger_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_circuit_breaker_active ON risk.circuit_breaker_events(deactivated_at) WHERE deactivated_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_performance_attribution_date ON risk.performance_attribution(calculation_date DESC);
CREATE INDEX IF NOT EXISTS idx_performance_attribution_symbol ON risk.performance_attribution(symbol, calculation_date DESC);

-- Create views for common risk queries
CREATE OR REPLACE VIEW risk.current_portfolio_risk AS
SELECT
    pm.*,
    COUNT(pr.symbol) as position_count_detailed,
    AVG(pr.risk_score) as avg_position_risk_score,
    MAX(pr.risk_score) as max_position_risk_score,
    SUM(CASE WHEN pr.risk_score > 7 THEN 1 ELSE 0 END) as high_risk_positions
FROM risk.portfolio_metrics pm
LEFT JOIN risk.position_risks pr ON DATE_TRUNC('hour', pm.timestamp) = DATE_TRUNC('hour', pr.timestamp)
WHERE pm.timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY pm.id, pm.timestamp, pm.total_exposure, pm.cash_percentage, pm.position_count,
         pm.concentration_risk, pm.portfolio_beta, pm.portfolio_correlation,
         pm.value_at_risk_1d, pm.value_at_risk_5d, pm.expected_shortfall,
         pm.sharpe_ratio, pm.max_drawdown, pm.current_drawdown, pm.volatility, pm.created_at
ORDER BY pm.timestamp DESC
LIMIT 1;

CREATE OR REPLACE VIEW risk.active_alerts AS
SELECT
    ra.*,
    CASE
        WHEN ra.severity = 'critical' THEN 4
        WHEN ra.severity = 'high' THEN 3
        WHEN ra.severity = 'medium' THEN 2
        ELSE 1
    END as priority_score
FROM risk.risk_alerts ra
WHERE ra.acknowledged = FALSE
ORDER BY priority_score DESC, ra.timestamp DESC;

CREATE OR REPLACE VIEW risk.position_risk_summary AS
SELECT
    pr.symbol,
    pr.sector,
    AVG(pr.risk_score) as avg_risk_score,
    MAX(pr.risk_score) as max_risk_score,
    AVG(pr.volatility) as avg_volatility,
    AVG(pr.portfolio_percentage) as avg_portfolio_percentage,
    COUNT(*) as measurement_count,
    MAX(pr.timestamp) as last_measured
FROM risk.position_risks pr
WHERE pr.timestamp >= NOW() - INTERVAL '7 days'
GROUP BY pr.symbol, pr.sector
ORDER BY avg_risk_score DESC;

CREATE OR REPLACE VIEW risk.daily_risk_summary AS
SELECT
    dr.report_date,
    dr.portfolio_value,
    dr.daily_pnl,
    dr.daily_return,
    dr.current_drawdown,
    dr.volatility,
    dr.total_trades,
    dr.winning_trades,
    CASE WHEN dr.total_trades > 0 THEN ROUND((dr.winning_trades::DECIMAL / dr.total_trades) * 100, 2) ELSE 0 END as win_rate_pct,
    dr.risk_events_count,
    CASE
        WHEN dr.current_drawdown > 0.10 THEN 'High Risk'
        WHEN dr.current_drawdown > 0.05 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_level
FROM risk.daily_reports dr
ORDER BY dr.report_date DESC;

-- Create materialized view for performance (refresh daily)
CREATE MATERIALIZED VIEW IF NOT EXISTS risk.portfolio_performance_stats AS
SELECT
    DATE_TRUNC('month', ps.timestamp) as month,
    AVG(ps.total_equity) as avg_portfolio_value,
    STDDEV(ps.total_equity) as portfolio_volatility,
    MIN(ps.total_equity) as min_portfolio_value,
    MAX(ps.total_equity) as max_portfolio_value,
    COUNT(*) as snapshots_count,
    (MAX(ps.total_equity) - MIN(ps.total_equity)) / MIN(ps.total_equity) as month_return,
    (MAX(ps.total_equity) - LAG(MAX(ps.total_equity)) OVER (ORDER BY DATE_TRUNC('month', ps.timestamp))) /
    LAG(MAX(ps.total_equity)) OVER (ORDER BY DATE_TRUNC('month', ps.timestamp)) as monthly_return
FROM risk.portfolio_snapshots ps
WHERE ps.timestamp >= NOW() - INTERVAL '2 years'
GROUP BY DATE_TRUNC('month', ps.timestamp)
ORDER BY month DESC;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_performance_stats_month
ON risk.portfolio_performance_stats(month);

-- Functions for risk calculations
CREATE OR REPLACE FUNCTION risk.calculate_sharpe_ratio(
    returns DECIMAL[],
    risk_free_rate DECIMAL DEFAULT 0.02
) RETURNS DECIMAL AS $$
DECLARE
    avg_return DECIMAL;
    return_stddev DECIMAL;
    excess_return DECIMAL;
BEGIN
    IF array_length(returns, 1) < 2 THEN
        RETURN 0;
    END IF;

    -- Calculate average return
    SELECT AVG(unnest) INTO avg_return FROM unnest(returns);

    -- Calculate standard deviation
    SELECT STDDEV(unnest) INTO return_stddev FROM unnest(returns);

    IF return_stddev = 0 OR return_stddev IS NULL THEN
        RETURN 0;
    END IF;

    -- Calculate Sharpe ratio
    excess_return := avg_return - (risk_free_rate / 252); -- Daily risk-free rate
    RETURN excess_return / return_stddev * SQRT(252); -- Annualized
END;
$$ LANGUAGE plpgsql;

-- Function to calculate maximum drawdown
CREATE OR REPLACE FUNCTION risk.calculate_max_drawdown(
    portfolio_values DECIMAL[]
) RETURNS DECIMAL AS $$
DECLARE
    i INTEGER;
    peak_value DECIMAL := 0;
    max_dd DECIMAL := 0;
    current_dd DECIMAL;
BEGIN
    IF array_length(portfolio_values, 1) < 2 THEN
        RETURN 0;
    END IF;

    peak_value := portfolio_values[1];

    FOR i IN 2..array_length(portfolio_values, 1) LOOP
        IF portfolio_values[i] > peak_value THEN
            peak_value := portfolio_values[i];
        ELSE
            current_dd := (peak_value - portfolio_values[i]) / peak_value;
            IF current_dd > max_dd THEN
                max_dd := current_dd;
            END IF;
        END IF;
    END LOOP;

    RETURN max_dd;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update materialized view
CREATE OR REPLACE FUNCTION risk.refresh_performance_stats() RETURNS TRIGGER AS $$
BEGIN
    -- Refresh materialized view (can be expensive, so do it conditionally)
    IF TG_OP = 'INSERT' AND NEW.timestamp::DATE != (SELECT MAX(timestamp)::DATE FROM risk.portfolio_snapshots WHERE timestamp < NEW.timestamp) THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY risk.portfolio_performance_stats;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for auto-refresh (optional - can be heavy on large datasets)
-- CREATE TRIGGER portfolio_snapshot_refresh_trigger
--     AFTER INSERT ON risk.portfolio_snapshots
--     FOR EACH ROW EXECUTE FUNCTION risk.refresh_performance_stats();

-- Risk monitoring alerts trigger
CREATE OR REPLACE FUNCTION risk.check_alert_conditions() RETURNS TRIGGER AS $$
BEGIN
    -- Auto-acknowledge duplicate alerts within 1 hour
    UPDATE risk.risk_alerts
    SET acknowledged = TRUE,
        acknowledged_by = 'auto_dedup',
        acknowledged_at = NOW()
    WHERE alert_type = NEW.alert_type
      AND symbol = NEW.symbol
      AND acknowledged = FALSE
      AND timestamp < NEW.timestamp - INTERVAL '1 hour';

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER risk_alert_dedup_trigger
    BEFORE INSERT ON risk.risk_alerts
    FOR EACH ROW EXECUTE FUNCTION risk.check_alert_conditions();

-- Performance optimization: Partitioning for large tables (optional)
-- Partition risk_events by month for better performance on large datasets
-- CREATE TABLE risk.risk_events_y2024m01 PARTITION OF risk.risk_events
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON SCHEMA risk TO trader;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA risk TO trader;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA risk TO trader;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA risk TO trader;

-- Comments for documentation
COMMENT ON SCHEMA risk IS 'Risk management schema containing all risk-related tables and functions';

COMMENT ON TABLE risk.risk_events IS 'Log of all risk management events and violations';
COMMENT ON TABLE risk.portfolio_snapshots IS 'Historical portfolio state snapshots for analysis';
COMMENT ON TABLE risk.portfolio_metrics IS 'Calculated portfolio risk metrics over time';
COMMENT ON TABLE risk.position_risks IS 'Individual position risk assessments';
COMMENT ON TABLE risk.risk_alerts IS 'Risk alerts sent to operators and systems';
COMMENT ON TABLE risk.daily_reports IS 'Daily comprehensive risk management reports';
COMMENT ON TABLE risk.trailing_stops IS 'Trailing stop loss configurations for active positions';
COMMENT ON TABLE risk.position_sizing_history IS 'History of position sizing recommendations';
COMMENT ON TABLE risk.risk_limits_history IS 'Audit trail of risk limit changes';
COMMENT ON TABLE risk.circuit_breaker_events IS 'Circuit breaker activation and deactivation events';
COMMENT ON TABLE risk.performance_attribution IS 'Performance attribution analysis by position and factor';

COMMENT ON FUNCTION risk.calculate_sharpe_ratio IS 'Calculate Sharpe ratio from array of returns';
COMMENT ON FUNCTION risk.calculate_max_drawdown IS 'Calculate maximum drawdown from portfolio value series';

-- Sample data cleanup job (run monthly)
-- DELETE FROM risk.risk_events WHERE created_at < NOW() - INTERVAL '1 year';
-- DELETE FROM risk.portfolio_snapshots WHERE created_at < NOW() - INTERVAL '2 years';
-- DELETE FROM risk.portfolio_metrics WHERE created_at < NOW() - INTERVAL '1 year';
-- DELETE FROM risk.position_risks WHERE created_at < NOW() - INTERVAL '6 months';
-- DELETE FROM risk.risk_alerts WHERE created_at < NOW() - INTERVAL '3 months' AND acknowledged = TRUE;

-- Refresh materialized view manually when needed
-- REFRESH MATERIALIZED VIEW CONCURRENTLY risk.portfolio_performance_stats;
