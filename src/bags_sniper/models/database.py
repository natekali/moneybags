"""
SQLAlchemy database models for the Bags Sniper bot.
Tracks deployers, token launches, and trades.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all models."""

    pass


class TradeStatus(str, Enum):
    """Trade lifecycle status."""

    PENDING = "pending"
    EXECUTED = "executed"
    PARTIAL_EXIT = "partial_exit"
    CLOSED = "closed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExitReason(str, Enum):
    """Reason for trade exit."""

    TAKE_PROFIT_T1 = "take_profit_t1"
    TAKE_PROFIT_T2 = "take_profit_t2"
    TAKE_PROFIT_T3 = "take_profit_t3"
    TRAILING_STOP = "trailing_stop"
    STOP_LOSS = "stop_loss"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL = "manual"
    HARD_EXIT = "hard_exit"


class Deployer(Base):
    """
    Tracks token deployers and their historical performance.
    This is the core of the deployer intelligence system.
    """

    __tablename__ = "deployers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wallet_address: Mapped[str] = mapped_column(
        String(44), unique=True, nullable=False, index=True
    )

    # Performance Metrics
    total_launches: Mapped[int] = mapped_column(Integer, default=0)
    graduated_launches: Mapped[int] = mapped_column(Integer, default=0)
    graduation_rate: Mapped[float] = mapped_column(Float, default=0.0)
    rug_count: Mapped[int] = mapped_column(Integer, default=0)  # PRD 5.2.1: Track rugs

    # Profitability
    total_volume_sol: Mapped[Decimal] = mapped_column(
        Numeric(20, 9), default=Decimal("0")
    )
    estimated_profit_sol: Mapped[Decimal] = mapped_column(
        Numeric(20, 9), default=Decimal("0")
    )
    avg_peak_mcap_sol: Mapped[Decimal] = mapped_column(
        Numeric(20, 9), default=Decimal("0")
    )

    # Timing Patterns
    avg_time_to_graduation_minutes: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    launches_last_24h: Mapped[int] = mapped_column(Integer, default=0)
    launches_last_7d: Mapped[int] = mapped_column(Integer, default=0)

    # Scoring
    deployer_score: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    score_updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Metadata
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_launch_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    token_launches: Mapped[list["TokenLaunch"]] = relationship(
        "TokenLaunch", back_populates="deployer", lazy="selectin"
    )

    # Indexes for common queries
    __table_args__ = (
        Index("idx_deployer_score_graduated", "deployer_score", "graduated_launches"),
        Index("idx_deployer_graduation_rate", "graduation_rate"),
    )

    def calculate_score(self) -> float:
        """
        Calculate deployer score using the formula from PRD.
        Score = (grad_rate * 40) + (avg_peak_mcap * 0.3) + (recency * 20) + (consistency * 10) - rug_penalty

        PRD 5.2.1: Deployers with rug history are heavily penalized.
        """
        # PRD 5.2.1: Rug count check - immediate disqualification for serial ruggers
        rug_count = self.rug_count or 0
        if rug_count >= 2:
            self.deployer_score = 0.0
            return 0.0

        # Graduation rate component (40% weight)
        # Baseline graduation rate is 1.4%, we want 3x+ (4.2%+)
        grad_component = min(self.graduation_rate / 0.10, 1.0) * 40

        # Peak market cap component (30% weight)
        # Normalize to reasonable range (0-1000 SOL avg peak = full points)
        mcap_normalized = min(float(self.avg_peak_mcap_sol) / 1000, 1.0)
        mcap_component = mcap_normalized * 30

        # Recency component (20% weight)
        # More recent activity = higher score
        if self.last_launch_at:
            days_since_launch = (
                datetime.utcnow() - self.last_launch_at.replace(tzinfo=None)
            ).days
            recency = max(0, 1 - (days_since_launch / 30))  # Decay over 30 days
        else:
            recency = 0
        recency_component = recency * 20

        # Consistency component (10% weight)
        # Reward deployers with multiple successful launches
        consistency = min(self.graduated_launches / 5, 1.0)
        consistency_component = consistency * 10

        # Rug penalty: -30 points per rug (one rug = heavy penalty)
        rug_penalty = rug_count * 30

        self.deployer_score = max(
            0.0,
            grad_component + mcap_component + recency_component + consistency_component - rug_penalty
        )
        return self.deployer_score


class TokenLaunch(Base):
    """
    Tracks individual token launches by deployers.
    Used for historical analysis and real-time monitoring.
    """

    __tablename__ = "token_launches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mint_address: Mapped[str] = mapped_column(
        String(44), unique=True, nullable=False, index=True
    )
    deployer_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("deployers.id"), nullable=False, index=True
    )

    # Token Info
    name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Launch Metrics
    initial_mcap_sol: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 9), nullable=True
    )
    peak_mcap_sol: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 9), nullable=True
    )
    graduated: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    graduation_mcap_sol: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 9), nullable=True
    )

    # Timing
    launched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    graduated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    time_to_graduation_minutes: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # AI Analysis (async populated)
    narrative_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    narrative_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    deployer: Mapped["Deployer"] = relationship(
        "Deployer", back_populates="token_launches"
    )
    trades: Mapped[list["Trade"]] = relationship(
        "Trade", back_populates="token_launch", lazy="selectin"
    )

    __table_args__ = (
        Index("idx_token_deployer_launched", "deployer_id", "launched_at"),
    )


class Trade(Base):
    """
    Tracks all trades executed by the bot.
    Includes entry, exits, and P&L tracking.
    """

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_launch_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("token_launches.id"), nullable=False, index=True
    )

    # Entry Details
    entry_price_sol: Mapped[Decimal] = mapped_column(Numeric(20, 9), nullable=False)
    entry_amount_sol: Mapped[Decimal] = mapped_column(Numeric(20, 9), nullable=False)
    entry_tokens: Mapped[Decimal] = mapped_column(Numeric(30, 0), nullable=False)
    entry_tx_signature: Mapped[str] = mapped_column(String(88), nullable=False)
    entry_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Position Tracking
    tokens_remaining: Mapped[Decimal] = mapped_column(Numeric(30, 0), nullable=False)
    avg_exit_price_sol: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 9), nullable=True
    )
    total_exit_sol: Mapped[Decimal] = mapped_column(
        Numeric(20, 9), default=Decimal("0")
    )

    # Status
    status: Mapped[TradeStatus] = mapped_column(
        String(20), default=TradeStatus.EXECUTED, index=True
    )

    # Take-Profit Tracking
    tp1_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    tp1_exit_sol: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 9), nullable=True
    )
    tp2_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    tp2_exit_sol: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 9), nullable=True
    )
    tp3_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    tp3_exit_sol: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 9), nullable=True
    )

    # Final Exit
    final_exit_reason: Mapped[Optional[ExitReason]] = mapped_column(
        String(30), nullable=True
    )
    final_exit_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # P&L
    realized_pnl_sol: Mapped[Decimal] = mapped_column(
        Numeric(20, 9), default=Decimal("0")
    )
    realized_pnl_percent: Mapped[float] = mapped_column(Float, default=0.0)
    peak_unrealized_pnl_percent: Mapped[float] = mapped_column(Float, default=0.0)

    # Decision Context (for learning)
    deployer_score_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    entry_reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    token_launch: Mapped["TokenLaunch"] = relationship(
        "TokenLaunch", back_populates="trades"
    )

    __table_args__ = (Index("idx_trade_status_entry", "status", "entry_timestamp"),)

    def calculate_pnl(self) -> tuple[Decimal, float]:
        """Calculate realized P&L."""
        if self.entry_amount_sol == 0:
            return Decimal("0"), 0.0

        self.realized_pnl_sol = self.total_exit_sol - self.entry_amount_sol
        self.realized_pnl_percent = float(
            (self.realized_pnl_sol / self.entry_amount_sol) * 100
        )
        return self.realized_pnl_sol, self.realized_pnl_percent


class DailyStats(Base):
    """
    Daily aggregated statistics for monitoring and circuit breakers.
    """

    __tablename__ = "daily_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), unique=True, nullable=False, index=True
    )

    # Trade Counts
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    winning_trades: Mapped[int] = mapped_column(Integer, default=0)
    losing_trades: Mapped[int] = mapped_column(Integer, default=0)

    # P&L
    total_pnl_sol: Mapped[Decimal] = mapped_column(
        Numeric(20, 9), default=Decimal("0")
    )
    total_volume_sol: Mapped[Decimal] = mapped_column(
        Numeric(20, 9), default=Decimal("0")
    )

    # Drawdown Tracking
    starting_balance_sol: Mapped[Decimal] = mapped_column(Numeric(20, 9), nullable=False)
    ending_balance_sol: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(20, 9), nullable=True
    )
    max_drawdown_percent: Mapped[float] = mapped_column(Float, default=0.0)

    # Circuit Breaker Status
    circuit_breaker_level: Mapped[int] = mapped_column(Integer, default=0)
    circuit_breaker_triggered_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


async def init_database(database_url: str) -> async_sessionmaker:
    """Initialize database connection and create tables."""
    engine = create_async_engine(
        database_url,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return async_sessionmaker(engine, expire_on_commit=False)
