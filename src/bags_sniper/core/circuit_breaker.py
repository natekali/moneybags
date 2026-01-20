"""
Multi-level circuit breaker for risk management.
Automatically reduces exposure and halts trading during drawdowns.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import IntEnum
from typing import Optional

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bags_sniper.core.config import Settings
from bags_sniper.models.database import DailyStats

logger = structlog.get_logger()


class CircuitBreakerLevel(IntEnum):
    """Circuit breaker levels with increasing severity."""

    NORMAL = 0      # Normal operation
    LEVEL_1 = 1     # 3% drawdown - reduce position size 50%
    LEVEL_2 = 2     # 5% drawdown - reduce position size 75%
    LEVEL_3 = 3     # 8% drawdown - pause new entries
    LEVEL_4 = 4     # 10% drawdown - full shutdown, exit all


@dataclass
class CircuitBreakerState:
    """Current state of the circuit breaker."""

    level: CircuitBreakerLevel = CircuitBreakerLevel.NORMAL
    current_drawdown_percent: float = 0.0
    position_size_multiplier: float = 1.0
    new_entries_allowed: bool = True
    triggered_at: Optional[datetime] = None
    auto_reset_at: Optional[datetime] = None
    consecutive_wins_for_reset: int = 0
    consecutive_losses: int = 0  # PRD 5.2.3: Track consecutive losses
    message: str = "Operating normally"


@dataclass
class DayTracker:
    """Tracks daily P&L for circuit breaker calculations."""

    date: datetime = field(default_factory=lambda: datetime.utcnow().date())
    starting_balance_sol: Decimal = Decimal("0")
    current_balance_sol: Decimal = Decimal("0")
    realized_pnl_sol: Decimal = Decimal("0")
    peak_balance_sol: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0


class CircuitBreaker:
    """
    Multi-level circuit breaker that automatically manages risk.

    Levels:
    - L1 (3%): Reduce position size 50%
    - L2 (5%): Reduce position size 75%
    - L3 (8%): Pause new entries, manage existing only
    - L4 (10%): Full shutdown, exit all positions

    Reset conditions:
    - 3 consecutive winning trades, OR
    - New trading day (midnight UTC)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._state = CircuitBreakerState()
        self._day_tracker = DayTracker()
        self._lock = asyncio.Lock()

        # Thresholds from settings
        self._thresholds = {
            CircuitBreakerLevel.LEVEL_1: settings.circuit_breaker_l1_percent,
            CircuitBreakerLevel.LEVEL_2: settings.circuit_breaker_l2_percent,
            CircuitBreakerLevel.LEVEL_3: settings.circuit_breaker_l3_percent,
            CircuitBreakerLevel.LEVEL_4: settings.circuit_breaker_l4_percent,
        }

        # Position size multipliers per level
        self._multipliers = {
            CircuitBreakerLevel.NORMAL: 1.0,
            CircuitBreakerLevel.LEVEL_1: 0.5,
            CircuitBreakerLevel.LEVEL_2: 0.25,
            CircuitBreakerLevel.LEVEL_3: 0.0,  # No new entries
            CircuitBreakerLevel.LEVEL_4: 0.0,  # Full shutdown
        }

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def can_open_position(self) -> bool:
        """Check if new positions are allowed."""
        return (
            self._state.level < CircuitBreakerLevel.LEVEL_3
            and self._state.new_entries_allowed
        )

    @property
    def position_size_multiplier(self) -> float:
        """Get current position size multiplier."""
        return self._state.position_size_multiplier

    async def initialize(
        self,
        session: AsyncSession,
        current_balance_sol: Decimal,
    ):
        """Initialize circuit breaker with current balance."""
        async with self._lock:
            self._day_tracker.starting_balance_sol = current_balance_sol
            self._day_tracker.current_balance_sol = current_balance_sol
            self._day_tracker.peak_balance_sol = current_balance_sol

            # Load today's stats if they exist
            today = datetime.utcnow().date()
            result = await session.execute(
                select(DailyStats).where(
                    DailyStats.date >= datetime.combine(today, datetime.min.time())
                )
            )
            daily_stats = result.scalar_one_or_none()

            if daily_stats:
                self._state.level = CircuitBreakerLevel(daily_stats.circuit_breaker_level)
                if daily_stats.circuit_breaker_triggered_at:
                    self._state.triggered_at = daily_stats.circuit_breaker_triggered_at

            logger.info(
                "circuit_breaker_initialized",
                starting_balance=str(current_balance_sol),
                level=self._state.level.name,
            )

    async def record_trade_result(
        self,
        session: AsyncSession,
        pnl_sol: Decimal,
        is_win: bool,
    ):
        """
        Record a trade result and update circuit breaker state.

        Args:
            session: Database session
            pnl_sol: Realized P&L in SOL
            is_win: Whether the trade was profitable
        """
        async with self._lock:
            # Update day tracker
            self._day_tracker.realized_pnl_sol += pnl_sol
            self._day_tracker.current_balance_sol += pnl_sol
            self._day_tracker.total_trades += 1

            if is_win:
                self._day_tracker.winning_trades += 1
                self._state.consecutive_wins_for_reset += 1
                self._state.consecutive_losses = 0  # Reset loss streak on win
            else:
                self._day_tracker.losing_trades += 1
                self._state.consecutive_wins_for_reset = 0
                self._state.consecutive_losses += 1  # PRD 5.2.3: Track loss streak

                # PRD 5.2.3: 3 consecutive losses triggers level escalation
                if self._state.consecutive_losses >= 3:
                    await self._escalate_from_losses()
                    self._state.consecutive_losses = 0  # Reset after escalation

            # Update peak
            if self._day_tracker.current_balance_sol > self._day_tracker.peak_balance_sol:
                self._day_tracker.peak_balance_sol = self._day_tracker.current_balance_sol

            # Calculate drawdown from peak
            if self._day_tracker.peak_balance_sol > 0:
                drawdown = (
                    self._day_tracker.peak_balance_sol
                    - self._day_tracker.current_balance_sol
                ) / self._day_tracker.peak_balance_sol * 100
                self._state.current_drawdown_percent = float(drawdown)
            else:
                self._state.current_drawdown_percent = 0.0

            # Check for level changes
            await self._evaluate_level()

            # Check for reset conditions
            await self._check_reset_conditions()

            logger.info(
                "trade_recorded",
                pnl_sol=str(pnl_sol),
                is_win=is_win,
                drawdown_percent=f"{self._state.current_drawdown_percent:.2f}%",
                circuit_level=self._state.level.name,
            )

    async def _evaluate_level(self):
        """Evaluate and potentially escalate circuit breaker level."""
        drawdown = self._state.current_drawdown_percent
        old_level = self._state.level

        # Check thresholds from highest to lowest
        if drawdown >= self._thresholds[CircuitBreakerLevel.LEVEL_4]:
            self._state.level = CircuitBreakerLevel.LEVEL_4
        elif drawdown >= self._thresholds[CircuitBreakerLevel.LEVEL_3]:
            self._state.level = CircuitBreakerLevel.LEVEL_3
        elif drawdown >= self._thresholds[CircuitBreakerLevel.LEVEL_2]:
            self._state.level = CircuitBreakerLevel.LEVEL_2
        elif drawdown >= self._thresholds[CircuitBreakerLevel.LEVEL_1]:
            self._state.level = CircuitBreakerLevel.LEVEL_1
        # Note: We don't auto-decrease level here; that requires reset conditions

        if self._state.level != old_level:
            self._state.triggered_at = datetime.utcnow()
            self._state.position_size_multiplier = self._multipliers[self._state.level]
            self._state.new_entries_allowed = (
                self._state.level < CircuitBreakerLevel.LEVEL_3
            )
            self._state.message = self._get_level_message()

            logger.warning(
                "circuit_breaker_triggered",
                old_level=old_level.name,
                new_level=self._state.level.name,
                drawdown=f"{drawdown:.2f}%",
            )

    async def _escalate_from_losses(self):
        """
        PRD 5.2.3: Escalate circuit breaker level due to 3 consecutive losses.
        This is independent of drawdown-based escalation.
        """
        if self._state.level >= CircuitBreakerLevel.LEVEL_3:
            # Already at high level, don't escalate further from losses alone
            logger.warning(
                "consecutive_losses_at_high_level",
                level=self._state.level.name,
                consecutive_losses=3,
            )
            return

        old_level = self._state.level
        new_level = CircuitBreakerLevel(min(self._state.level + 1, CircuitBreakerLevel.LEVEL_3))

        self._state.level = new_level
        self._state.triggered_at = datetime.utcnow()
        self._state.position_size_multiplier = self._multipliers[new_level]
        self._state.new_entries_allowed = new_level < CircuitBreakerLevel.LEVEL_3
        self._state.message = f"3 consecutive losses - escalated to {new_level.name}"

        logger.warning(
            "circuit_breaker_loss_streak",
            old_level=old_level.name,
            new_level=new_level.name,
            reason="3 consecutive losses",
        )

    async def _check_reset_conditions(self):
        """Check if circuit breaker should reset."""
        if self._state.level == CircuitBreakerLevel.NORMAL:
            return

        # Reset condition 1: 3 consecutive wins
        if self._state.consecutive_wins_for_reset >= 3:
            await self._reset_one_level()
            return

        # Reset condition 2: Check if we've recovered above the threshold
        # for the current level
        current_threshold = self._thresholds[self._state.level]
        if self._state.current_drawdown_percent < current_threshold - 1.0:
            # 1% buffer before downgrading
            await self._reset_one_level()

    async def _reset_one_level(self):
        """Reset circuit breaker by one level."""
        if self._state.level == CircuitBreakerLevel.NORMAL:
            return

        old_level = self._state.level
        new_level = CircuitBreakerLevel(max(0, self._state.level - 1))

        self._state.level = new_level
        self._state.position_size_multiplier = self._multipliers[new_level]
        self._state.new_entries_allowed = new_level < CircuitBreakerLevel.LEVEL_3
        self._state.consecutive_wins_for_reset = 0
        self._state.message = self._get_level_message()

        logger.info(
            "circuit_breaker_reset",
            old_level=old_level.name,
            new_level=new_level.name,
            reason="Recovery conditions met",
        )

    async def daily_reset(self, session: AsyncSession, new_balance: Decimal):
        """
        Reset circuit breaker for new trading day.
        Called at midnight UTC.
        """
        async with self._lock:
            # Save yesterday's stats
            await self._save_daily_stats(session)

            # Reset for new day
            self._day_tracker = DayTracker()
            self._day_tracker.starting_balance_sol = new_balance
            self._day_tracker.current_balance_sol = new_balance
            self._day_tracker.peak_balance_sol = new_balance

            self._state = CircuitBreakerState()

            logger.info(
                "circuit_breaker_daily_reset",
                new_balance=str(new_balance),
            )

    async def _save_daily_stats(self, session: AsyncSession):
        """Save daily statistics to database."""
        today = datetime.combine(self._day_tracker.date, datetime.min.time())

        daily_stats = DailyStats(
            date=today,
            total_trades=self._day_tracker.total_trades,
            winning_trades=self._day_tracker.winning_trades,
            losing_trades=self._day_tracker.losing_trades,
            total_pnl_sol=self._day_tracker.realized_pnl_sol,
            starting_balance_sol=self._day_tracker.starting_balance_sol,
            ending_balance_sol=self._day_tracker.current_balance_sol,
            max_drawdown_percent=self._state.current_drawdown_percent,
            circuit_breaker_level=self._state.level,
            circuit_breaker_triggered_at=self._state.triggered_at,
        )
        session.add(daily_stats)

    def _get_level_message(self) -> str:
        """Get human-readable message for current level."""
        messages = {
            CircuitBreakerLevel.NORMAL: "Operating normally",
            CircuitBreakerLevel.LEVEL_1: f"L1 ({self._thresholds[CircuitBreakerLevel.LEVEL_1]}% DD): Position size reduced 50%",
            CircuitBreakerLevel.LEVEL_2: f"L2 ({self._thresholds[CircuitBreakerLevel.LEVEL_2]}% DD): Position size reduced 75%",
            CircuitBreakerLevel.LEVEL_3: f"L3 ({self._thresholds[CircuitBreakerLevel.LEVEL_3]}% DD): New entries PAUSED",
            CircuitBreakerLevel.LEVEL_4: f"L4 ({self._thresholds[CircuitBreakerLevel.LEVEL_4]}% DD): FULL SHUTDOWN - Exiting all positions",
        }
        return messages[self._state.level]

    def get_adjusted_position_size(self, base_size_sol: Decimal) -> Decimal:
        """
        Get position size adjusted for current circuit breaker level.

        Args:
            base_size_sol: Base position size before circuit breaker adjustment

        Returns:
            Adjusted position size
        """
        return base_size_sol * Decimal(str(self._state.position_size_multiplier))

    async def force_level(self, level: CircuitBreakerLevel, reason: str):
        """
        Force circuit breaker to a specific level (for manual override).
        """
        async with self._lock:
            old_level = self._state.level
            self._state.level = level
            self._state.position_size_multiplier = self._multipliers[level]
            self._state.new_entries_allowed = level < CircuitBreakerLevel.LEVEL_3
            self._state.triggered_at = datetime.utcnow()
            self._state.message = f"MANUAL: {reason}"

            logger.warning(
                "circuit_breaker_manual_override",
                old_level=old_level.name,
                new_level=level.name,
                reason=reason,
            )

    def get_status_report(self) -> dict:
        """Get detailed status report for monitoring."""
        return {
            "level": self._state.level.name,
            "level_value": self._state.level.value,
            "message": self._state.message,
            "current_drawdown_percent": round(self._state.current_drawdown_percent, 2),
            "position_size_multiplier": self._state.position_size_multiplier,
            "new_entries_allowed": self._state.new_entries_allowed,
            "triggered_at": (
                self._state.triggered_at.isoformat()
                if self._state.triggered_at
                else None
            ),
            "consecutive_wins_for_reset": self._state.consecutive_wins_for_reset,
            "consecutive_losses": self._state.consecutive_losses,
            "day_stats": {
                "starting_balance": str(self._day_tracker.starting_balance_sol),
                "current_balance": str(self._day_tracker.current_balance_sol),
                "realized_pnl": str(self._day_tracker.realized_pnl_sol),
                "total_trades": self._day_tracker.total_trades,
                "win_rate": (
                    round(
                        self._day_tracker.winning_trades
                        / self._day_tracker.total_trades
                        * 100,
                        1,
                    )
                    if self._day_tracker.total_trades > 0
                    else 0
                ),
            },
            "thresholds": {
                "l1": self._thresholds[CircuitBreakerLevel.LEVEL_1],
                "l2": self._thresholds[CircuitBreakerLevel.LEVEL_2],
                "l3": self._thresholds[CircuitBreakerLevel.LEVEL_3],
                "l4": self._thresholds[CircuitBreakerLevel.LEVEL_4],
            },
        }
