"""
Professional Trading Logic Module.

Implements advanced trading strategies like a professional day trader:
1. Fee-aware profitability calculations
2. Smart exit timing based on momentum and volume
3. Opportunity cost analysis (compare new trades vs holding)
4. Position rotation based on better opportunities
5. AI-enhanced decision making

Transaction fees: ~0.015 SOL per trade (entry + exit = ~0.03 SOL round trip)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List
import asyncio

import structlog

from bags_sniper.core.config import Settings

logger = structlog.get_logger()


# Transaction fee constants (Solana + priority fees)
TX_FEE_SOL = Decimal("0.015")  # Approximate fee per transaction
ROUND_TRIP_FEE = TX_FEE_SOL * 2  # Entry + exit fees
MIN_PROFIT_AFTER_FEES = Decimal("0.005")  # Minimum profit to consider selling


@dataclass
class TradingOpportunity:
    """Represents a potential trading opportunity."""
    mint_address: str
    symbol: str
    deployer_score: float
    narrative_score: float
    expected_return: float  # Estimated % return
    confidence: float  # 0-1
    urgency: float  # 0-1 (higher = more time-sensitive)
    volume_momentum: float  # Volume trend indicator
    price_momentum: float  # Price trend indicator
    discovered_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PositionAnalysis:
    """Analysis of a current position."""
    mint_address: str
    symbol: str
    entry_price_sol: Decimal
    current_price_sol: Decimal
    entry_time: datetime
    tokens_held: int

    # Calculated fields
    unrealized_pnl_sol: Decimal = Decimal("0")
    unrealized_pnl_percent: float = 0.0
    net_pnl_after_fees: Decimal = Decimal("0")  # After accounting for exit fee
    hold_time_minutes: float = 0.0

    # Momentum indicators
    price_momentum: float = 0.0  # Positive = rising, negative = falling
    volume_momentum: float = 0.0
    peak_price_sol: Decimal = Decimal("0")
    drawdown_from_peak: float = 0.0

    # Trading signals
    should_sell: bool = False
    sell_urgency: float = 0.0  # 0-1
    sell_reason: str = ""


@dataclass
class TradeDecision:
    """Final trade decision with reasoning."""
    action: str  # "hold", "sell", "sell_partial", "skip"
    confidence: float
    reasoning: str
    suggested_percent: int = 100  # For partial sells
    estimated_pnl_sol: Decimal = Decimal("0")


class ProfessionalTrader:
    """
    Advanced trading logic that thinks like a professional trader.

    Key principles:
    1. Never sell at a loss that doesn't exceed fee savings
    2. Let winners run, cut losers quickly
    3. Rotate into better opportunities
    4. Account for all costs in decisions
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # Take-profit tiers (more aggressive for memecoins)
        self.tp_tiers = [
            {"multiple": 1.5, "percent": 30, "name": "TP1"},  # 50% gain, sell 30%
            {"multiple": 2.0, "percent": 30, "name": "TP2"},  # 100% gain, sell 30%
            {"multiple": 3.0, "percent": 20, "name": "TP3"},  # 200% gain, sell 20%
            {"multiple": 5.0, "percent": 20, "name": "TP4"},  # 400% gain, sell remaining
        ]

        # Stop-loss and trailing stop
        self.stop_loss_percent = -30  # -30% stop loss
        self.trailing_stop_percent = 25  # 25% trailing stop after TP1

        # Momentum thresholds
        self.negative_momentum_threshold = -0.3  # Strong selling pressure
        self.positive_momentum_threshold = 0.3  # Strong buying pressure

    def calculate_net_profit(
        self,
        entry_amount_sol: Decimal,
        exit_amount_sol: Decimal,
        include_entry_fee: bool = True,
    ) -> Decimal:
        """Calculate net profit after all fees."""
        gross_profit = exit_amount_sol - entry_amount_sol

        # Deduct exit fee (always)
        net_profit = gross_profit - TX_FEE_SOL

        # Deduct entry fee if not already accounted for
        if include_entry_fee:
            net_profit -= TX_FEE_SOL

        return net_profit

    def is_profitable_to_sell(
        self,
        entry_amount_sol: Decimal,
        current_value_sol: Decimal,
    ) -> bool:
        """Check if selling would result in profit after fees."""
        net_pnl = self.calculate_net_profit(entry_amount_sol, current_value_sol)
        return net_pnl >= MIN_PROFIT_AFTER_FEES

    def analyze_position(
        self,
        mint_address: str,
        symbol: str,
        entry_price_sol: Decimal,
        current_price_sol: Decimal,
        entry_time: datetime,
        tokens_held: int,
        entry_amount_sol: Decimal,
        peak_price_sol: Optional[Decimal] = None,
        price_history: Optional[List[Decimal]] = None,
        volume_history: Optional[List[Decimal]] = None,
    ) -> PositionAnalysis:
        """Analyze a position for sell signals."""

        # Basic calculations
        current_value = Decimal(str(tokens_held)) * current_price_sol
        unrealized_pnl_sol = current_value - entry_amount_sol
        unrealized_pnl_percent = float(
            (current_price_sol - entry_price_sol) / entry_price_sol * 100
        ) if entry_price_sol > 0 else 0

        hold_time = datetime.utcnow() - entry_time
        hold_time_minutes = hold_time.total_seconds() / 60

        # Net PnL after exit fee
        net_pnl_after_fees = unrealized_pnl_sol - TX_FEE_SOL

        # Peak tracking and drawdown
        if peak_price_sol is None:
            peak_price_sol = current_price_sol
        peak_price_sol = max(peak_price_sol, current_price_sol)

        drawdown_from_peak = 0.0
        if peak_price_sol > 0:
            drawdown_from_peak = float(
                (peak_price_sol - current_price_sol) / peak_price_sol * 100
            )

        # Calculate momentum (if history available)
        price_momentum = 0.0
        volume_momentum = 0.0

        if price_history and len(price_history) >= 3:
            # Simple momentum: compare recent prices to older prices
            recent_avg = sum(price_history[-3:]) / 3
            older_avg = sum(price_history[:3]) / 3 if len(price_history) >= 6 else price_history[0]
            if older_avg > 0:
                price_momentum = float((recent_avg - older_avg) / older_avg)

        if volume_history and len(volume_history) >= 3:
            recent_vol = sum(volume_history[-3:]) / 3
            older_vol = sum(volume_history[:3]) / 3 if len(volume_history) >= 6 else volume_history[0]
            if older_vol > 0:
                volume_momentum = float((recent_vol - older_vol) / older_vol)

        analysis = PositionAnalysis(
            mint_address=mint_address,
            symbol=symbol,
            entry_price_sol=entry_price_sol,
            current_price_sol=current_price_sol,
            entry_time=entry_time,
            tokens_held=tokens_held,
            unrealized_pnl_sol=unrealized_pnl_sol,
            unrealized_pnl_percent=unrealized_pnl_percent,
            net_pnl_after_fees=net_pnl_after_fees,
            hold_time_minutes=hold_time_minutes,
            price_momentum=price_momentum,
            volume_momentum=volume_momentum,
            peak_price_sol=peak_price_sol,
            drawdown_from_peak=drawdown_from_peak,
        )

        # Determine if we should sell
        self._evaluate_sell_signals(analysis)

        return analysis

    def _evaluate_sell_signals(self, analysis: PositionAnalysis):
        """Evaluate all sell signals and set should_sell flag."""

        signals = []
        urgency = 0.0

        price_multiple = float(
            analysis.current_price_sol / analysis.entry_price_sol
        ) if analysis.entry_price_sol > 0 else 1.0

        # Signal 1: Stop loss
        if analysis.unrealized_pnl_percent <= self.stop_loss_percent:
            signals.append(f"Stop loss triggered ({analysis.unrealized_pnl_percent:.1f}%)")
            urgency = max(urgency, 0.9)

        # Signal 2: Severe drawdown from peak (trailing stop logic)
        if analysis.drawdown_from_peak >= self.trailing_stop_percent and price_multiple > 1.3:
            signals.append(f"Trailing stop: {analysis.drawdown_from_peak:.1f}% from peak")
            urgency = max(urgency, 0.8)

        # Signal 3: Strong negative momentum with profit
        if (analysis.price_momentum < self.negative_momentum_threshold and
            analysis.volume_momentum < self.negative_momentum_threshold and
            analysis.net_pnl_after_fees > 0):
            signals.append("Strong negative momentum - secure profits")
            urgency = max(urgency, 0.7)

        # Signal 4: Extended hold time with no significant gains
        if analysis.hold_time_minutes > 15 and analysis.unrealized_pnl_percent < 10:
            signals.append("Extended hold with minimal gains - consider rotation")
            urgency = max(urgency, 0.3)

        # Signal 5: Take profit at 2x+ (always good to secure some)
        if price_multiple >= 2.0 and analysis.net_pnl_after_fees > 0:
            signals.append(f"At {price_multiple:.1f}x - consider taking profit")
            urgency = max(urgency, 0.5)

        # Signal 6: Volume dying with position in profit
        if (analysis.volume_momentum < -0.5 and
            analysis.hold_time_minutes > 5 and
            analysis.net_pnl_after_fees > 0):
            signals.append("Volume dying - liquidity risk increasing")
            urgency = max(urgency, 0.6)

        if signals:
            analysis.should_sell = True
            analysis.sell_urgency = urgency
            analysis.sell_reason = "; ".join(signals)
        else:
            analysis.should_sell = False
            analysis.sell_urgency = 0.0
            analysis.sell_reason = "No sell signals"

    def decide_sell_action(
        self,
        analysis: PositionAnalysis,
        tp1_hit: bool = False,
        tp2_hit: bool = False,
        better_opportunity: Optional[TradingOpportunity] = None,
    ) -> TradeDecision:
        """
        Make final sell decision considering all factors.

        Args:
            analysis: Position analysis
            tp1_hit: Whether TP1 was already taken
            tp2_hit: Whether TP2 was already taken
            better_opportunity: A better opportunity to rotate into
        """

        price_multiple = float(
            analysis.current_price_sol / analysis.entry_price_sol
        ) if analysis.entry_price_sol > 0 else 1.0

        # Emergency sell (stop loss)
        if analysis.unrealized_pnl_percent <= self.stop_loss_percent:
            return TradeDecision(
                action="sell",
                confidence=0.95,
                reasoning=f"Stop loss at {analysis.unrealized_pnl_percent:.1f}%",
                suggested_percent=100,
                estimated_pnl_sol=analysis.net_pnl_after_fees,
            )

        # Trailing stop (after hitting TP1)
        if tp1_hit and analysis.drawdown_from_peak >= self.trailing_stop_percent:
            return TradeDecision(
                action="sell",
                confidence=0.9,
                reasoning=f"Trailing stop: {analysis.drawdown_from_peak:.1f}% drawdown from {analysis.peak_price_sol:.6f}",
                suggested_percent=100,
                estimated_pnl_sol=analysis.net_pnl_after_fees,
            )

        # Take profit tiers
        for tier in self.tp_tiers:
            if price_multiple >= tier["multiple"]:
                # Check if we should take this tier
                tier_name = tier["name"]
                if tier_name == "TP1" and not tp1_hit:
                    return TradeDecision(
                        action="sell_partial",
                        confidence=0.85,
                        reasoning=f"{tier_name}: {price_multiple:.1f}x gain - securing {tier['percent']}%",
                        suggested_percent=tier["percent"],
                        estimated_pnl_sol=analysis.net_pnl_after_fees * Decimal(str(tier["percent"] / 100)),
                    )
                elif tier_name == "TP2" and tp1_hit and not tp2_hit:
                    return TradeDecision(
                        action="sell_partial",
                        confidence=0.85,
                        reasoning=f"{tier_name}: {price_multiple:.1f}x gain - securing {tier['percent']}%",
                        suggested_percent=tier["percent"],
                        estimated_pnl_sol=analysis.net_pnl_after_fees * Decimal(str(tier["percent"] / 100)),
                    )

        # Opportunity rotation: sell to buy something better
        if better_opportunity and analysis.net_pnl_after_fees >= 0:
            opp_score = (
                better_opportunity.deployer_score * 0.4 +
                better_opportunity.narrative_score * 0.3 +
                better_opportunity.expected_return * 0.3
            )
            current_outlook = analysis.unrealized_pnl_percent / 10  # Simplified outlook

            if opp_score > current_outlook + 20:  # Significant improvement needed
                return TradeDecision(
                    action="sell",
                    confidence=0.7,
                    reasoning=f"Rotating to {better_opportunity.symbol} (score: {opp_score:.0f} vs current: {current_outlook:.0f})",
                    suggested_percent=100,
                    estimated_pnl_sol=analysis.net_pnl_after_fees,
                )

        # Momentum-based exit (protect profits)
        if (analysis.price_momentum < self.negative_momentum_threshold and
            analysis.net_pnl_after_fees > 0):
            return TradeDecision(
                action="sell",
                confidence=0.75,
                reasoning=f"Negative momentum ({analysis.price_momentum:.2f}) - protecting {analysis.net_pnl_after_fees:.4f} SOL profit",
                suggested_percent=100,
                estimated_pnl_sol=analysis.net_pnl_after_fees,
            )

        # Default: hold
        return TradeDecision(
            action="hold",
            confidence=0.6,
            reasoning=f"No sell triggers - PnL: {analysis.unrealized_pnl_percent:.1f}%, momentum: {analysis.price_momentum:.2f}",
            suggested_percent=0,
            estimated_pnl_sol=Decimal("0"),
        )

    def should_enter_trade(
        self,
        deployer_score: float,
        narrative_score: float,
        current_positions: int,
        max_positions: int,
        available_balance_sol: Decimal,
        min_position_size_sol: Decimal = Decimal("0.05"),
    ) -> tuple[bool, str]:
        """
        Decide if we should enter a new trade.

        Returns:
            (should_enter, reasoning)
        """

        # Check balance (account for fees)
        min_required = min_position_size_sol + TX_FEE_SOL
        if available_balance_sol < min_required:
            return False, f"Insufficient balance: {available_balance_sol:.4f} SOL < {min_required:.4f} SOL needed"

        # Check position limit
        if current_positions >= max_positions:
            return False, f"At position limit: {current_positions}/{max_positions}"

        # Score thresholds
        combined_score = deployer_score * 0.6 + narrative_score * 0.4

        if combined_score < 30:
            return False, f"Combined score too low: {combined_score:.0f}"

        # Deployer score requirements
        if deployer_score < 20:
            return False, f"Deployer score too low: {deployer_score:.0f}"

        return True, f"Entry approved: combined score {combined_score:.0f}"

    def calculate_optimal_entry_size(
        self,
        available_balance_sol: Decimal,
        deployer_score: float,
        narrative_score: float,
        confidence: float,
        base_size_sol: Decimal = Decimal("0.1"),
    ) -> Decimal:
        """
        Calculate optimal position size accounting for fees.

        Size formula: base * deployer_mult * narrative_mult * confidence
        Then ensure we have enough for fees.
        """

        # Deployer multiplier
        if deployer_score >= 85:
            dep_mult = 1.5
        elif deployer_score >= 70:
            dep_mult = 1.0 + (deployer_score - 70) / 30
        elif deployer_score >= 50:
            dep_mult = 0.5 + (deployer_score - 50) / 40
        else:
            dep_mult = 0.25

        # Narrative multiplier
        if narrative_score >= 80:
            narr_mult = 1.2
        elif narrative_score >= 60:
            narr_mult = 1.0
        else:
            narr_mult = 0.8

        # Confidence adjustment
        conf_mult = 0.5 + (confidence * 0.5)  # 0.5 to 1.0

        # Calculate raw size
        raw_size = base_size_sol * Decimal(str(dep_mult * narr_mult * conf_mult))

        # Ensure we leave enough for exit fee
        max_size = available_balance_sol - TX_FEE_SOL - Decimal("0.01")  # Keep small buffer

        # Apply limits
        min_size = Decimal("0.02")  # Minimum viable trade
        optimal_size = max(min_size, min(raw_size, max_size))

        logger.debug(
            "position_size_calculated",
            base=str(base_size_sol),
            dep_mult=dep_mult,
            narr_mult=narr_mult,
            conf_mult=conf_mult,
            raw_size=str(raw_size),
            optimal_size=str(optimal_size),
        )

        return optimal_size

    def estimate_break_even_price(
        self,
        entry_price_sol: Decimal,
        entry_amount_sol: Decimal,
    ) -> Decimal:
        """Calculate price needed to break even after fees."""
        # Entry cost = entry_amount + entry_fee
        total_cost = entry_amount_sol + TX_FEE_SOL

        # Need to get back: total_cost + exit_fee
        needed_exit_value = total_cost + TX_FEE_SOL

        # Price needed (assuming same quantity)
        price_mult = needed_exit_value / entry_amount_sol
        break_even_price = entry_price_sol * price_mult

        return break_even_price


def format_trade_decision(decision: TradeDecision) -> str:
    """Format trade decision for logging."""
    return (
        f"Action: {decision.action.upper()} "
        f"(confidence: {decision.confidence:.0%}) - "
        f"{decision.reasoning}"
    )
