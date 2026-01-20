"""
Quality Gate System for Trade Entry Decisions.

This module implements strict quality controls to ensure the bot only trades
on high-confidence opportunities. Key principles:

1. NEVER trade on low-quality signals
2. Require EITHER strong deployer OR strong narrative
3. Unknown deployers need EXCEPTIONAL narrative to trade
4. Daily capital and loss limits to protect the bankroll
5. Minimum confidence threshold for all trades

PRD References:
- 5.2.1: Deployer score determines entry logic
- 5.2.4: AI should score tokens 0-100
- 3.2: Kill switch criteria (win rate <50%)
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List
import asyncio

import structlog

from bags_sniper.core.config import Settings

logger = structlog.get_logger()


@dataclass
class QualityScore:
    """Combined quality score for trade decisions."""
    deployer_score: float  # 0-100
    narrative_score: float  # 0-100
    combined_score: float  # 0-100 weighted combination
    confidence: float  # 0-1 overall confidence
    quality_tier: str  # "elite", "good", "marginal", "poor", "skip"
    reasoning: str
    should_trade: bool
    max_position_multiplier: float  # How much of base size to use


@dataclass
class DailyStats:
    """Track daily trading statistics for capital protection."""
    date: date = field(default_factory=date.today)
    trades_taken: int = 0
    capital_deployed_sol: Decimal = Decimal("0")
    realized_pnl_sol: Decimal = Decimal("0")
    unrealized_pnl_sol: Decimal = Decimal("0")
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    is_paused: bool = False
    pause_reason: str = ""


class QualityGate:
    """
    Strict quality control for trade entry decisions.

    The bot should be HIGHLY SELECTIVE - better to miss opportunities
    than to lose money on bad trades.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # Quality thresholds - BE STRICT
        self.min_combined_score = 40  # Minimum combined score to trade (lowered for new deployers)
        self.min_confidence = 0.4  # Minimum confidence to trade

        # Deployer-specific thresholds
        self.elite_deployer_score = 85  # Score >85 = elite, bypass filters
        self.good_deployer_score = 65   # Score 65-85 = good, normal filters (lowered from 70)
        self.marginal_deployer_score = 45  # Score 45-65 = marginal, strict filters (lowered from 50)
        # Below 45 = poor/unknown, need exceptional narrative

        # For unknown deployers, require high narrative score
        self.unknown_deployer_min_narrative = 60  # Unknown deployers need AI score >60 (lowered from 70)

        # Daily limits - these are ADAPTIVE based on actual balance
        # Base limits for ~1 SOL balance; scales with balance
        self.base_daily_trades = 10  # Base trades per day
        self.min_daily_trades = 5    # Minimum trades regardless of balance
        self.max_daily_trades = 25   # Maximum trades per day cap
        self.max_daily_capital_pct = 40  # Max 40% of balance per day (increased)
        self.max_daily_loss_pct = 15  # Stop trading if down 15% in a day (increased for more action)
        self.max_consecutive_losses = 5  # Pause after 5 consecutive losses

        # Position size multipliers by quality tier
        self.position_multipliers = {
            "elite": 1.5,      # Elite deployers get larger positions
            "good": 1.0,       # Good deployers get normal positions
            "marginal": 0.6,   # Marginal deployers get 60% positions (increased from 50%)
            "poor": 0.35,      # Poor/unknown get 35% positions (increased from 25%)
            "skip": 0.0,       # Don't trade
        }

        # Daily tracking
        self._daily_stats = DailyStats()
        self._lock = asyncio.Lock()
        self._cached_balance: Optional[Decimal] = None

    def evaluate_quality(
        self,
        deployer_score: float,
        narrative_score: Optional[float],
        deployer_launches: int = 0,
        is_new_deployer: bool = True,
    ) -> QualityScore:
        """
        Evaluate the quality of a trading opportunity.

        This is the main decision point - should we trade or not?
        """

        # Use 0 for missing narrative score
        narr_score = narrative_score if narrative_score is not None else 0.0

        # Calculate combined score with weights
        # For unknown deployers, narrative is MORE important
        # Standard: 60% deployer, 40% narrative
        # Unknown deployer: 30% deployer, 70% narrative (narrative-driven)
        if deployer_score < self.marginal_deployer_score:
            # Unknown deployer - weight narrative more heavily
            combined = deployer_score * 0.3 + narr_score * 0.7
        else:
            # Known deployer - deployer score matters more
            combined = deployer_score * 0.6 + narr_score * 0.4

        # Determine quality tier and whether to trade
        quality_tier = "skip"
        should_trade = False
        max_mult = 0.0
        reasoning = ""
        confidence = 0.0
        narrative_qualified = False  # Flag: was trade approved based on strong narrative?

        # TIER 1: Elite deployers (score >85)
        if deployer_score >= self.elite_deployer_score:
            quality_tier = "elite"
            should_trade = True
            max_mult = self.position_multipliers["elite"]
            confidence = 0.85
            reasoning = f"Elite deployer (score {deployer_score:.0f}) - high confidence entry"

        # TIER 2: Good deployers (score 70-85)
        elif deployer_score >= self.good_deployer_score:
            quality_tier = "good"
            should_trade = True
            max_mult = self.position_multipliers["good"]
            confidence = 0.7
            reasoning = f"Good deployer (score {deployer_score:.0f}) - normal entry"

        # TIER 3: Marginal deployers (score 50-70)
        elif deployer_score >= self.marginal_deployer_score:
            # Marginal deployers need decent narrative
            if narr_score >= 50:
                quality_tier = "marginal"
                should_trade = True
                max_mult = self.position_multipliers["marginal"]
                confidence = 0.5
                reasoning = f"Marginal deployer (score {deployer_score:.0f}) with narrative {narr_score:.0f} - reduced position"
            else:
                quality_tier = "skip"
                should_trade = False
                confidence = 0.3
                reasoning = f"Marginal deployer (score {deployer_score:.0f}) with weak narrative {narr_score:.0f} - SKIP"

        # TIER 4: Poor/Unknown deployers (score <45)
        # Note: With new deployer baseline scoring (~51), most new deployers will be in marginal tier
        # This tier is now mainly for deployers with rug history or known bad track records
        else:
            # Poor deployers need strong narrative to trade
            if narr_score >= self.unknown_deployer_min_narrative:
                quality_tier = "poor"
                should_trade = True
                max_mult = self.position_multipliers["poor"]
                confidence = 0.45
                reasoning = f"Poor/unknown deployer but strong narrative ({narr_score:.0f}) - reduced position"
                narrative_qualified = True  # Bypass combined score check
            elif narr_score >= 50:
                # Decent narrative (50-59) - smaller position
                quality_tier = "poor"
                should_trade = True
                max_mult = self.position_multipliers["poor"] * 0.6  # Slightly reduced
                confidence = 0.42  # Just above minimum to allow trading
                reasoning = f"Poor/unknown deployer with decent narrative ({narr_score:.0f}) - small position"
                narrative_qualified = True  # Bypass combined score check
            elif narr_score >= 40:
                # Minimal narrative but not terrible (40-49) - tiny position
                quality_tier = "poor"
                should_trade = True
                max_mult = self.position_multipliers["poor"] * 0.4  # Very small
                confidence = 0.41
                reasoning = f"Poor/unknown deployer with minimal narrative ({narr_score:.0f}) - tiny position"
                narrative_qualified = True
            else:
                quality_tier = "skip"
                should_trade = False
                confidence = 0.2
                reasoning = f"Poor deployer (score {deployer_score:.0f}) with weak narrative ({narr_score:.0f}) - SKIP"

        # Final check: combined score must meet minimum
        # SKIP this check if the trade was approved based on strong narrative alone
        # (unknown deployers with good narrative should trade regardless of combined score)
        if should_trade and combined < self.min_combined_score and not narrative_qualified:
            should_trade = False
            quality_tier = "skip"
            max_mult = 0.0
            reasoning = f"Combined score ({combined:.0f}) below minimum ({self.min_combined_score}) - SKIP"

        # Final check: confidence must meet minimum
        if should_trade and confidence < self.min_confidence:
            should_trade = False
            quality_tier = "skip"
            max_mult = 0.0
            reasoning = f"Confidence ({confidence:.0%}) below minimum ({self.min_confidence:.0%}) - SKIP"

        return QualityScore(
            deployer_score=deployer_score,
            narrative_score=narr_score,
            combined_score=combined,
            confidence=confidence,
            quality_tier=quality_tier,
            reasoning=reasoning,
            should_trade=should_trade,
            max_position_multiplier=max_mult,
        )

    def _get_adaptive_daily_trade_limit(self, balance_sol: Decimal) -> int:
        """
        Calculate adaptive daily trade limit based on balance.

        Scaling:
        - < 0.5 SOL: min trades (conservative, preserve capital)
        - 0.5-2 SOL: base trades
        - 2-5 SOL: 1.5x base
        - 5-10 SOL: 2x base
        - > 10 SOL: max trades (well-capitalized)
        """
        balance_float = float(balance_sol)

        if balance_float < 0.5:
            return self.min_daily_trades
        elif balance_float < 2.0:
            return self.base_daily_trades
        elif balance_float < 5.0:
            return min(int(self.base_daily_trades * 1.5), self.max_daily_trades)
        elif balance_float < 10.0:
            return min(int(self.base_daily_trades * 2), self.max_daily_trades)
        else:
            return self.max_daily_trades

    def _get_adaptive_position_size(self, balance_sol: Decimal, quality_multiplier: float) -> Decimal:
        """
        Calculate adaptive position size based on balance and quality.

        Small balances = smaller positions to preserve capital
        Larger balances = can afford larger positions
        """
        balance_float = float(balance_sol)

        # Base position as percentage of balance
        if balance_float < 0.5:
            # Very small balance: 3-5% per trade
            base_pct = 0.04
        elif balance_float < 2.0:
            # Small balance: 5-8% per trade
            base_pct = 0.06
        elif balance_float < 5.0:
            # Medium balance: 5-10% per trade
            base_pct = 0.07
        else:
            # Large balance: use configured max
            base_pct = float(self.settings.max_position_size_sol) / balance_float
            base_pct = min(base_pct, 0.10)  # Cap at 10% per trade

        position_sol = balance_sol * Decimal(str(base_pct)) * Decimal(str(quality_multiplier))

        # Enforce minimum and maximum
        min_position = Decimal("0.01")  # Minimum viable trade
        max_position = Decimal(str(self.settings.max_position_size_sol))

        return max(min_position, min(position_sol, max_position))

    async def check_daily_limits(
        self,
        position_size_sol: Decimal,
        total_balance_sol: Decimal,
    ) -> tuple[bool, str]:
        """
        Check if we can take another trade today based on ADAPTIVE daily limits.

        Limits scale with balance:
        - Small balance = fewer trades, preserve capital
        - Large balance = more trades, can afford more action

        Returns:
            (can_trade, reason)
        """
        async with self._lock:
            # Reset daily stats if new day
            today = date.today()
            if self._daily_stats.date != today:
                self._daily_stats = DailyStats(date=today)

            stats = self._daily_stats

            # Cache balance for adaptive calculations
            self._cached_balance = total_balance_sol

            # Check if paused
            if stats.is_paused:
                return False, f"Trading paused: {stats.pause_reason}"

            # ADAPTIVE trade count based on balance
            adaptive_max_trades = self._get_adaptive_daily_trade_limit(total_balance_sol)
            if stats.trades_taken >= adaptive_max_trades:
                return False, f"Daily trade limit reached ({stats.trades_taken}/{adaptive_max_trades} for {total_balance_sol:.2f} SOL balance)"

            # Check capital deployment (adaptive percentage)
            max_daily_capital = total_balance_sol * Decimal(str(self.max_daily_capital_pct / 100))
            if stats.capital_deployed_sol + position_size_sol > max_daily_capital:
                return False, f"Daily capital limit reached ({self.max_daily_capital_pct}% = {max_daily_capital:.3f} SOL)"

            # Check daily loss (adaptive threshold)
            # Smaller accounts have tighter loss limits
            adaptive_loss_pct = self.max_daily_loss_pct
            if float(total_balance_sol) < 1.0:
                adaptive_loss_pct = 10.0  # Tighter for small accounts
            elif float(total_balance_sol) < 3.0:
                adaptive_loss_pct = 12.0

            max_loss = total_balance_sol * Decimal(str(adaptive_loss_pct / 100))
            if stats.realized_pnl_sol < -max_loss:
                stats.is_paused = True
                stats.pause_reason = f"Daily loss limit ({adaptive_loss_pct}% = {max_loss:.3f} SOL) exceeded"
                return False, stats.pause_reason

            # Check consecutive losses
            if stats.consecutive_losses >= self.max_consecutive_losses:
                stats.is_paused = True
                stats.pause_reason = f"Consecutive loss limit ({self.max_consecutive_losses}) reached"
                return False, stats.pause_reason

            logger.debug(
                "daily_limits_check_passed",
                trades_today=stats.trades_taken,
                max_trades=adaptive_max_trades,
                capital_deployed=str(stats.capital_deployed_sol),
                max_capital=str(max_daily_capital),
                balance=str(total_balance_sol),
            )

            return True, "OK"

    async def record_trade_entry(
        self,
        position_size_sol: Decimal,
    ):
        """Record a new trade entry."""
        async with self._lock:
            today = date.today()
            if self._daily_stats.date != today:
                self._daily_stats = DailyStats(date=today)

            self._daily_stats.trades_taken += 1
            self._daily_stats.capital_deployed_sol += position_size_sol

            logger.info(
                "trade_recorded",
                trades_today=self._daily_stats.trades_taken,
                capital_deployed=str(self._daily_stats.capital_deployed_sol),
            )

    async def record_trade_exit(
        self,
        pnl_sol: Decimal,
        is_win: bool,
    ):
        """Record a trade exit."""
        async with self._lock:
            today = date.today()
            if self._daily_stats.date != today:
                self._daily_stats = DailyStats(date=today)

            stats = self._daily_stats
            stats.realized_pnl_sol += pnl_sol

            if is_win:
                stats.wins += 1
                stats.consecutive_losses = 0
            else:
                stats.losses += 1
                stats.consecutive_losses += 1

            logger.info(
                "exit_recorded",
                pnl=str(pnl_sol),
                daily_pnl=str(stats.realized_pnl_sol),
                win_rate=f"{stats.wins}/{stats.wins + stats.losses}",
                consecutive_losses=stats.consecutive_losses,
            )

    def get_daily_stats(self) -> DailyStats:
        """Get current daily statistics."""
        return self._daily_stats

    async def reset_daily_pause(self):
        """Manually reset daily pause (for Telegram command)."""
        async with self._lock:
            self._daily_stats.is_paused = False
            self._daily_stats.pause_reason = ""
            self._daily_stats.consecutive_losses = 0
            logger.info("daily_pause_reset")


def format_quality_decision(quality: QualityScore) -> str:
    """Format quality decision for logging."""
    return (
        f"Quality: {quality.quality_tier.upper()} "
        f"(D:{quality.deployer_score:.0f} N:{quality.narrative_score:.0f} "
        f"C:{quality.combined_score:.0f}) "
        f"[{quality.confidence:.0%}] - "
        f"{'TRADE' if quality.should_trade else 'SKIP'}: {quality.reasoning}"
    )
