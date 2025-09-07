"""
AI Strategy Models and Database Schema

This module defines the data models and database schema for the AI strategy engine,
including decision tracking, performance metrics, and prompt management.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

if TYPE_CHECKING:
    from sqlalchemy.ext.declarative import DeclarativeMeta

    Base = DeclarativeMeta
else:
    Base = declarative_base()


class AIModelType(Enum):
    """Enumeration of available AI models."""

    OPUS = "claude-3-opus"
    SONNET = "claude-3-sonnet"
    HAIKU = "claude-3-haiku"


class DecisionType(Enum):
    """Types of trading decisions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class PromptType(Enum):
    """Types of prompts used in the system."""

    MASTER_ANALYST = "master_analyst"
    MARKET_REGIME = "market_regime"
    RISK_ASSESSMENT = "risk_assessment"
    MOMENTUM_CATALYST = "momentum_catalyst"
    CONTRARIAN = "contrarian"
    EXIT_OPTIMIZER = "exit_optimizer"


# SQLAlchemy Models


class AIDecisionRecord(Base):
    """Records AI trading decisions for audit and performance tracking."""

    __tablename__ = "ai_decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    ticker = Column(String(10), nullable=False)

    # Decision details
    decision = Column(String(10), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    position_size = Column(Float)
    risk_reward_ratio = Column(Float)

    # AI model information
    models_used = Column(JSON, nullable=False)  # List of models used
    consensus_details = Column(JSON)  # Voting details
    total_tokens = Column(Integer)
    total_cost = Column(Float)

    # Context and reasoning
    market_context = Column(JSON)
    reasoning = Column(Text)
    key_risks = Column(JSON)  # List of identified risks

    # Performance tracking
    actual_outcome = Column(Float)  # Actual P&L if position was taken
    accuracy_score = Column(Float)  # Calculated accuracy metric
    execution_time_ms = Column(Integer)  # Time to generate decision

    # Metadata
    strategy_version = Column(String(20))
    prompt_versions = Column(JSON)  # Version of each prompt used

    __table_args__ = (
        Index("ix_ai_decisions_timestamp", "timestamp"),
        Index("ix_ai_decisions_ticker", "ticker"),
        Index("ix_ai_decisions_confidence", "confidence"),
    )


class AIResponseCache(Base):
    """Caches AI responses to reduce API calls."""

    __tablename__ = "ai_response_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key = Column(String(64), nullable=False, unique=True)  # SHA256 hash
    model = Column(String(50), nullable=False)
    prompt_type = Column(String(50), nullable=False)

    # Response data
    response = Column(JSON, nullable=False)
    tokens_used = Column(Integer)
    cost = Column(Float)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    hit_count = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_ai_cache_expires", "expires_at"),
        Index("ix_ai_cache_key_model", "cache_key", "model"),
    )


class PromptVersion(Base):
    """Tracks different versions of prompts for A/B testing."""

    __tablename__ = "prompt_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prompt_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False)

    # Prompt content
    template = Column(Text, nullable=False)
    model_preference = Column(String(50))
    max_tokens = Column(Integer, default=2000)
    temperature = Column(Float, default=0.3)

    # Performance metrics
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float)
    avg_confidence = Column(Float)
    avg_return = Column(Float)

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    is_active = Column(Boolean, default=True)
    notes = Column(Text)

    __table_args__ = (
        UniqueConstraint("prompt_type", "version", name="uq_prompt_version"),
        Index("ix_prompt_active", "is_active"),
    )


class AIPerformanceMetrics(Base):
    """Aggregated performance metrics for AI decisions."""

    __tablename__ = "ai_performance_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date = Column(DateTime, nullable=False)
    ticker = Column(String(10))  # Null for overall metrics

    # Decision metrics
    total_decisions = Column(Integer, default=0)
    buy_decisions = Column(Integer, default=0)
    sell_decisions = Column(Integer, default=0)
    hold_decisions = Column(Integer, default=0)

    # Accuracy metrics
    correct_decisions = Column(Integer, default=0)
    incorrect_decisions = Column(Integer, default=0)
    accuracy_rate = Column(Float)

    # Financial metrics
    total_pnl = Column(Float, default=0)
    win_rate = Column(Float)
    avg_gain = Column(Float)
    avg_loss = Column(Float)
    sharpe_ratio = Column(Float)

    # Cost metrics
    total_api_cost = Column(Float, default=0)
    avg_cost_per_decision = Column(Float)
    roi_on_cost = Column(Float)  # Return per dollar spent on API

    # Model-specific metrics
    model_performance = Column(JSON)  # Performance by model type
    prompt_performance = Column(JSON)  # Performance by prompt version

    __table_args__ = (
        Index("ix_ai_metrics_date", "date"),
        Index("ix_ai_metrics_ticker", "ticker"),
        UniqueConstraint("date", "ticker", name="uq_metrics_date_ticker"),
    )


class MarketRegimeState(Base):
    """Tracks market regime assessments over time."""

    __tablename__ = "market_regime_states"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Regime classification
    regime = Column(
        String(50), nullable=False
    )  # trending_bullish, trending_bearish, etc.
    strength = Column(Float, nullable=False)  # 0-100
    risk_level = Column(String(20), nullable=False)  # low, medium, high, extreme

    # Market data snapshot
    spy_price = Column(Float)
    spy_change = Column(Float)
    vix_level = Column(Float)
    vix_change = Column(Float)

    # Market breadth
    advance_decline_ratio = Column(Float)
    highs_lows_ratio = Column(Float)
    percent_above_sma50 = Column(Float)

    # Trading adjustments
    position_size_multiplier = Column(Float, default=1.0)
    stop_loss_multiplier = Column(Float, default=1.0)
    confidence_threshold_adjustment = Column(Float, default=0)

    # Recommendations
    best_strategy = Column(String(50))
    sectors_to_focus = Column(JSON)

    # AI details
    ai_model_used = Column(String(50))
    tokens_used = Column(Integer)
    response_time_ms = Column(Integer)

    __table_args__ = (
        Index("ix_regime_timestamp", "timestamp"),
        Index("ix_regime_type", "regime"),
    )


class AITradeExecution(Base):
    """Links AI decisions to actual trade executions."""

    __tablename__ = "ai_trade_executions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    decision_id = Column(UUID(as_uuid=True), ForeignKey("ai_decisions.id"))

    # Execution details
    executed_at = Column(DateTime, nullable=False)
    execution_price = Column(Float, nullable=False)
    executed_quantity = Column(Integer, nullable=False)

    # Position tracking
    position_opened_at = Column(DateTime)
    position_closed_at = Column(DateTime)
    exit_price = Column(Float)
    exit_reason = Column(String(50))  # stop_loss, take_profit, ai_signal, manual

    # Performance
    realized_pnl = Column(Float)
    commission = Column(Float)
    slippage = Column(Float)

    # Relationship
    decision = relationship("AIDecisionRecord", backref="executions")

    __table_args__ = (
        Index("ix_execution_decision", "decision_id"),
        Index("ix_execution_timestamp", "executed_at"),
    )


# Dataclass Models for Runtime Use


@dataclass
class AIContext:
    """Runtime context for AI decision making."""

    ticker: str
    current_price: float
    market_regime: str
    risk_level: str
    position_size_multiplier: float = 1.0
    confidence_threshold: float = 60.0
    existing_position: Optional[Dict[str, Any]] = None
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptContext:
    """Context for prompt formatting."""

    template: str
    variables: Dict[str, Any]
    model_preference: AIModelType
    max_tokens: int = 2000
    temperature: float = 0.3
    use_cache: bool = True


@dataclass
class ConsensusResult:
    """Result of consensus building from multiple AI responses."""

    final_decision: str
    consensus_confidence: float
    agreement_rate: float
    dissenting_opinions: List[Dict[str, Any]]
    model_contributions: Dict[str, float]
    combined_reasoning: str
    combined_risks: List[str]


@dataclass
class PerformanceReport:
    """AI strategy performance report."""

    period_start: datetime
    period_end: datetime
    total_decisions: int
    accuracy_rate: float
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    total_api_cost: float
    cost_per_profitable_trade: float
    best_performing_prompt: str
    worst_performing_prompt: str
    model_rankings: Dict[str, float]
    recommendations: List[str]


# Database initialization function


def init_database(connection_string: str):
    """
    Initialize the database with AI strategy tables.

    Args:
        connection_string: PostgreSQL connection string
    """
    engine = create_engine(connection_string)
    Base.metadata.create_all(engine)
    return engine


# Helper functions for data conversion


def decision_to_dict(decision: AIDecisionRecord) -> Dict[str, Any]:
    """Convert AIDecisionRecord to dictionary."""
    return {
        "id": str(decision.id) if decision.id is not None else None,
        "timestamp": (
            decision.timestamp.isoformat() if decision.timestamp is not None else None
        ),
        "ticker": decision.ticker,
        "decision": decision.decision,
        "confidence": decision.confidence,
        "models_used": decision.models_used,
        "consensus_details": decision.consensus_details,
        "total_tokens": decision.total_tokens,
        "total_cost": decision.total_cost,
        "market_context": decision.market_context,
        "reasoning": decision.reasoning,
        "key_risks": decision.key_risks,
        "actual_outcome": decision.actual_outcome,
        "accuracy_score": decision.accuracy_score,
        "execution_time_ms": decision.execution_time_ms,
        "strategy_version": decision.strategy_version,
        "prompt_versions": decision.prompt_versions,
    }


def create_performance_summary(
    decisions: List[AIDecisionRecord], executions: List[AITradeExecution]
) -> PerformanceReport:
    """
    Create a performance summary from decision and execution records.

    Args:
        decisions: List of AI decisions
        executions: List of trade executions

    Returns:
        Performance report
    """
    if not decisions:
        return PerformanceReport(
            period_start=datetime.now(),
            period_end=datetime.now(),
            total_decisions=0,
            accuracy_rate=0,
            win_rate=0,
            total_pnl=0,
            sharpe_ratio=0.0,
            total_api_cost=0,
            cost_per_profitable_trade=0,
            best_performing_prompt="N/A",
            worst_performing_prompt="N/A",
            model_rankings={},
            recommendations=[],
        )

    # Calculate metrics
    timestamps = [
        cast(datetime, d.timestamp)
        for d in decisions
        if d.timestamp is not None and isinstance(d.timestamp, datetime)
    ]
    period_start = min(timestamps) if timestamps else datetime.now()
    period_end = max(timestamps) if timestamps else datetime.now()
    total_decisions = len(decisions)

    # Calculate accuracy (decisions with positive outcomes)
    accurate_decisions = [
        d for d in decisions if (d.actual_outcome is not None and d.actual_outcome > 0)
    ]
    accuracy_rate = (
        len(accurate_decisions) / total_decisions if total_decisions > 0 else 0
    )

    # Calculate win rate from executions
    winning_trades = []
    for e in executions:
        if (
            hasattr(e, "realized_pnl")
            and e.realized_pnl is not None
            and e.realized_pnl > 0
        ):
            winning_trades.append(e)
    win_rate = len(winning_trades) / len(executions) if executions else 0

    # Calculate total P&L
    total_pnl = (
        sum(float(e.realized_pnl) for e in executions if e.realized_pnl is not None)
        or 0.0
    )

    # Calculate API costs
    total_api_cost = (
        sum(float(d.total_cost) for d in decisions if d.total_cost is not None) or 0.0
    )

    # Cost per profitable trade
    cost_per_profitable_trade = (
        float(total_api_cost) / len(winning_trades) if winning_trades else 0.0
    )

    # Model rankings (simplified)
    model_performance: Dict[str, Any] = {}
    for decision in decisions:
        if decision.models_used is not None and isinstance(
            decision.models_used, (list, tuple)
        ):
            for model in decision.models_used:
                if model not in model_performance:
                    model_performance[model] = {"count": 0, "success": 0}
                model_performance[model]["count"] += 1
                if decision.actual_outcome is not None and decision.actual_outcome > 0:
                    model_performance[model]["success"] += 1

    model_rankings = {
        model: (stats["success"] / stats["count"]) if stats["count"] > 0 else 0
        for model, stats in model_performance.items()
    }

    # Generate recommendations
    recommendations = []
    if accuracy_rate < 0.5:
        recommendations.append("Consider adjusting confidence thresholds")
    if total_api_cost > float(total_pnl) * 0.1:
        recommendations.append(
            "API costs are high relative to profits, increase cache usage"
        )
    if win_rate < 0.4 and win_rate > 0:
        recommendations.append("Low win rate, review risk management parameters")

    return PerformanceReport(
        period_start=(
            period_start if isinstance(period_start, datetime) else datetime.now()
        ),
        period_end=period_end if isinstance(period_end, datetime) else datetime.now(),
        total_decisions=total_decisions,
        accuracy_rate=accuracy_rate,
        win_rate=win_rate,
        total_pnl=total_pnl,
        sharpe_ratio=0.0,  # Would need returns series to calculate
        total_api_cost=total_api_cost,
        cost_per_profitable_trade=cost_per_profitable_trade,
        best_performing_prompt="master_analyst",  # Placeholder
        worst_performing_prompt="contrarian",  # Placeholder
        model_rankings={str(k): float(v) for k, v in model_rankings.items()},
        recommendations=recommendations,
    )
