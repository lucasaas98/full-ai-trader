import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from services.risk_manager.src.main import app
from shared.config import Config
from shared.models import (
    OrderRequest,
    OrderSide,
    OrderType,
    PortfolioMetrics,
    PortfolioState,
    Position,
    PositionSizing,
    PositionSizingMethod,
    RiskAlert,
    RiskEvent,
    RiskEventType,
    RiskLimits,
    RiskParameters,
    RiskSeverity,
    SignalType,
    TradeSignal,
)


class TestRiskManagerAPI:
    """Test suite for Risk Manager API endpoints"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Mock(spec=Config)
        config.redis_host = "localhost"
        config.redis_port = 6379
        config.redis_password = None
        config.db_host = "localhost"
        config.db_port = 5432
        config.db_name = "test_db"
        config.db_user = "test_user"
        config.db_password = "test_pass"
        config.risk_max_position_size = 0.05
        config.risk_max_portfolio_risk = 0.02
        config.risk_drawdown_limit = 0.15
        return config

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        redis_mock = AsyncMock()
        redis_mock.publish = AsyncMock()
        redis_mock.set = AsyncMock()
        redis_mock.get = AsyncMock(return_value=None)
        return redis_mock

    @pytest.fixture
    def mock_db_pool(self):
        """Mock database pool"""
        pool_mock = AsyncMock()
        pool_mock.acquire = AsyncMock()
        pool_mock.release = AsyncMock()
        return pool_mock

    @pytest.fixture
    def sample_portfolio_state(self):
        """Sample portfolio state for testing"""
        positions = [
            Position(
                symbol="AAPL",
                quantity=100,
                entry_price=Decimal("150.00"),
                current_price=Decimal("155.00"),
                unrealized_pnl=Decimal("500.00"),
                market_value=Decimal("15500.00"),
                cost_basis=Decimal("15000.00"),
            ),
            Position(
                symbol="GOOGL",
                quantity=50,
                entry_price=Decimal("2800.00"),
                current_price=Decimal("2850.00"),
                unrealized_pnl=Decimal("2500.00"),
                market_value=Decimal("142500.00"),
                cost_basis=Decimal("140000.00"),
            ),
        ]

        return PortfolioState(
            account_id="test_account_123",
            cash=Decimal("50000.00"),
            buying_power=Decimal("100000.00"),
            total_equity=Decimal("208000.00"),
            positions=positions,
        )

    @pytest.fixture
    def sample_risk_parameters(self):
        """Sample risk parameters for testing"""
        return RiskParameters(
            max_position_size=Decimal("0.05"),
            max_daily_loss=Decimal("5000.00"),
            max_total_exposure=Decimal("0.80"),
            stop_loss_percentage=0.02,
            take_profit_percentage=0.06,
            max_correlation=0.7,
            min_trade_amount=Decimal("100.00"),
            max_trade_amount=Decimal("50000.00"),
        )

    @pytest.fixture
    def client(self):
        """FastAPI test client"""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_validate_order_endpoint_success(self, client, sample_portfolio_state):
        """Test successful order validation"""
        order_request = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "order_type": "market",
            "stop_price": 148.0,
            "client_order_id": "test_order_123",
        }

        with patch("src.main.risk_manager") as mock_rm, patch(
            "src.main.portfolio_monitor"
        ) as mock_pm:

            mock_rm.validate_order.return_value = (True, "Order validated successfully")

            response = client.post("/validate-order", json=order_request)

            assert response.status_code == 200
            data = response.json()
            assert data["is_valid"] is True
            assert "validation_message" in data

    def test_validate_order_endpoint_risk_violation(self, client):
        """Test order validation with risk violation"""
        order_request = {
            "symbol": "TSLA",
            "side": "buy",
            "quantity": 1000,
            "order_type": "market",
            "stop_price": 200.0,
            "client_order_id": "test_order_456",
        }

        with patch("src.main.risk_manager") as mock_rm:
            mock_rm.validate_order.return_value = (
                False,
                "Position size exceeds limits",
            )

            response = client.post("/validate-order", json=order_request)

            assert response.status_code == 200
            data = response.json()
            assert data["is_valid"] is False
            assert "exceeds limits" in data["validation_message"]

    def test_portfolio_risk_endpoint(self, client, sample_portfolio_state):
        """Test portfolio risk metrics endpoint"""
        with patch("src.main.portfolio_monitor") as mock_pm:
            mock_metrics = PortfolioMetrics(
                total_exposure=Decimal("158000.00"),
                cash_percentage=Decimal("0.24"),
                position_count=2,
                concentration_risk=0.68,
                portfolio_beta=1.1,
                value_at_risk_1d=Decimal("8500.00"),
                maximum_drawdown=Decimal("0.08"),
                current_drawdown=Decimal("0.02"),
            )

            mock_pm.calculate_portfolio_metrics.return_value = mock_metrics

            response = client.post(
                "/portfolio-risk", json=sample_portfolio_state.model_dump()
            )

            assert response.status_code == 200
            data = response.json()
            assert "total_exposure" in data
            assert "portfolio_beta" in data
            assert "value_at_risk_1d" in data

    def test_position_sizing_endpoint(self, client):
        """Test position sizing endpoint"""
        request_data = {
            "symbol": "MSFT",
            "entry_price": 300.0,
            "portfolio_value": 100000.0,
            "risk_amount": 1000.0,
            "method": "fixed_percentage",
        }

        with patch("src.main.position_sizer") as mock_ps:
            mock_sizing = PositionSizing(
                symbol="MSFT",
                method=PositionSizingMethod.FIXED_PERCENTAGE,
                position_size=33,
                recommended_shares=33,
                recommended_value=Decimal("9900.00"),
                position_percentage=Decimal("0.099"),
            )

            mock_ps.calculate_position_size.return_value = mock_sizing

            response = client.post("/position-sizing", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["recommended_shares"] == 33
            assert "position_percentage" in data

    def test_stress_test_endpoint(self, client):
        """Test stress testing endpoint"""
        request_data = {"scenarios": ["market_crash", "volatility_spike"]}

        with patch("src.main.risk_manager") as mock_rm:
            mock_results = {
                "market_crash": {
                    "portfolio_value": 120000.0,
                    "max_loss": 30000.0,
                    "drawdown_pct": 0.25,
                },
                "volatility_spike": {
                    "portfolio_value": 130000.0,
                    "max_loss": 20000.0,
                    "drawdown_pct": 0.15,
                },
            }

            mock_rm.run_stress_tests.return_value = mock_results

            response = client.post("/stress-test", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert "market_crash" in data
            assert "volatility_spike" in data
            assert data["market_crash"]["drawdown_pct"] == 0.25

    def test_portfolio_monitoring_endpoint(self, client, sample_portfolio_state):
        """Test portfolio monitoring endpoint"""
        with patch("src.main.portfolio_monitor") as mock_pm, patch(
            "src.main.alert_manager"
        ) as mock_am:

            mock_metrics = PortfolioMetrics(
                total_exposure=Decimal("158000.00"),
                position_count=2,
                concentration_risk=0.35,
            )

            mock_alerts = [
                RiskAlert(
                    alert_type=RiskEventType.POSITION_LIMIT_BREACH,
                    severity=RiskSeverity.MEDIUM,
                    symbol="AAPL",
                    title="Position Limit Warning",
                    message="Position approaching size limit",
                    action_required=False,
                )
            ]

            mock_pm.monitor_portfolio.return_value = (mock_metrics, mock_alerts)

            response = client.post(
                "/monitor-portfolio", json=sample_portfolio_state.model_dump()
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "monitoring_complete"
            assert data["alerts_generated"] == 1
            assert "timestamp" in data

    def test_stop_loss_levels_endpoint(self, client):
        """Test stop loss calculation endpoint"""
        with patch("src.main.risk_manager") as mock_rm:
            mock_rm.calculate_stop_loss_take_profit.return_value = (
                Decimal("145.00"),  # stop_loss
                Decimal("165.00"),  # take_profit
            )

            response = client.get("/stop-loss-levels/AAPL?entry_price=150.00&side=buy")

            assert response.status_code == 200
            data = response.json()
            assert data["symbol"] == "AAPL"
            assert data["entry_price"] == "150.00"
            assert data["stop_loss"] == "145.00"
            assert data["take_profit"] == "165.00"

    def test_emergency_stop_endpoint(self, client):
        """Test emergency stop endpoint"""
        request_data = {"reason": "Market crash detected"}

        with patch("src.main.risk_manager") as mock_rm, patch(
            "src.main.alert_manager"
        ) as mock_am:

            mock_rm.activate_emergency_stop.return_value = True

            response = client.post("/emergency-stop", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["emergency_stop_activated"] is True
            assert data["reason"] == "Market crash detected"

    def test_update_trailing_stops_endpoint(self, client, sample_portfolio_state):
        """Test trailing stops update endpoint"""
        market_prices = {"AAPL": 155.00, "GOOGL": 2850.00}

        request_data = {
            "portfolio": sample_portfolio_state.model_dump(),
            "market_prices": market_prices,
        }

        with patch("src.main.risk_manager") as mock_rm:
            mock_events = [
                RiskEvent(
                    event_type=RiskEventType.POSITION_LIMIT_BREACH,
                    severity=RiskSeverity.MEDIUM,
                    symbol="AAPL",
                    description="Trailing stop updated for AAPL",
                    resolved_at=None,
                    action_taken=None,
                )
            ]

            mock_rm.update_trailing_stops.return_value = mock_events

            response = client.post("/update-trailing-stops", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["events_generated"] == 1
            assert data["trailing_stops_updated"] is True

    def test_risk_limits_endpoint(self, client):
        """Test risk limits configuration endpoint"""
        risk_limits = {
            "max_position_percentage": 0.10,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.06,
            "max_daily_loss_percentage": 0.05,
            "max_positions": 5,
            "max_correlation_threshold": 0.7,
            "emergency_stop_percentage": 0.10,
        }

        with patch("src.main.risk_manager") as mock_rm:
            mock_rm.update_risk_limits.return_value = True

            response = client.post("/risk-limits", json=risk_limits)

            assert response.status_code == 200
            data = response.json()
            assert data["limits_updated"] is True

    def test_risk_events_endpoint(self, client):
        """Test risk events history endpoint"""
        with patch("src.main.database_manager") as mock_db:
            mock_events = [
                {
                    "id": "event_123",
                    "event_type": "position_limit_breach",
                    "severity": "medium",
                    "symbol": "AAPL",
                    "description": "Position size exceeded limit",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "resolved_at": None,
                }
            ]

            mock_db.get_recent_risk_events.return_value = mock_events

            response = client.get("/risk-events?limit=10")

            assert response.status_code == 200
            data = response.json()
            assert len(data["events"]) == 1
            assert data["events"][0]["symbol"] == "AAPL"

    def test_risk_dashboard_endpoint(self, client, sample_portfolio_state):
        """Test risk dashboard summary endpoint"""
        with patch("src.main.portfolio_monitor") as mock_pm, patch(
            "src.main.database_manager"
        ) as mock_db:

            mock_metrics = PortfolioMetrics(
                total_exposure=Decimal("158000.00"),
                position_count=2,
                concentration_risk=0.35,
                value_at_risk_1d=Decimal("8500.00"),
                current_drawdown=Decimal("0.02"),
            )

            mock_events = [{"event_type": "position_limit_breach", "count": 1}]

            mock_pm.calculate_portfolio_metrics.return_value = mock_metrics
            mock_db.get_risk_event_summary.return_value = mock_events

            response = client.post(
                "/risk-dashboard", json=sample_portfolio_state.model_dump()
            )

            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            assert "recent_events" in data
            assert "alerts" in data
