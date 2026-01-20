"""
Advanced Position Sizing Engine.

Combines multiple signals to determine optimal position size:
1. Deployer score (primary)
2. Narrative strength (secondary)
3. Circuit breaker state
4. Portfolio concentration
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import structlog

from bags_sniper.core.circuit_breaker import CircuitBreaker
from bags_sniper.core.config import Settings
from bags_sniper.core.deployer_intelligence import DeployerProfile
from bags_sniper.services.deepseek_ai import NarrativeAnalysis

logger = structlog.get_logger()


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    recommended_size_sol: Decimal
    base_size_sol: Decimal
    deployer_multiplier: float
    narrative_multiplier: float
    circuit_breaker_multiplier: float
    concentration_multiplier: float
    reasoning: str
    confidence: float  # 0-1 overall confidence in the trade


class PositionSizer:
    """
    Calculates optimal position sizes based on multiple factors.

    Formula:
    size = base_size * deployer_mult * narrative_mult * cb_mult * concentration_mult

    Where:
    - deployer_mult: Based on deployer score (0.5 - 1.5x)
    - narrative_mult: Based on narrative score (0.8 - 1.2x)
    - cb_mult: From circuit breaker (0.25 - 1.0x)
    - concentration_mult: Portfolio risk (0.5 - 1.0x)
    """

    def __init__(
        self,
        settings: Settings,
        circuit_breaker: CircuitBreaker,
    ):
        self.settings = settings
        self.circuit_breaker = circuit_breaker
        self.base_size = Decimal(str(settings.max_position_size_sol))

    def calculate_size(
        self,
        deployer_profile: DeployerProfile,
        narrative_analysis: Optional[NarrativeAnalysis] = None,
        current_positions: int = 0,
        max_positions: int = 5,
    ) -> PositionSizeResult:
        """
        Calculate recommended position size.

        Args:
            deployer_profile: Deployer intelligence analysis
            narrative_analysis: AI narrative analysis (optional)
            current_positions: Number of currently open positions
            max_positions: Maximum allowed concurrent positions

        Returns:
            PositionSizeResult with recommended size and reasoning
        """
        reasoning_parts = []

        # 1. Deployer multiplier (0.5 - 1.5x based on score)
        deployer_mult = self._deployer_multiplier(deployer_profile)
        reasoning_parts.append(
            f"Deployer({deployer_profile.deployer_score:.0f})={deployer_mult:.2f}x"
        )

        # 2. Narrative multiplier (0.8 - 1.2x)
        narrative_mult = self._narrative_multiplier(narrative_analysis)
        if narrative_analysis:
            reasoning_parts.append(
                f"Narrative({narrative_analysis.narrative_score:.0f})={narrative_mult:.2f}x"
            )

        # 3. Circuit breaker multiplier
        cb_mult = self.circuit_breaker.position_size_multiplier
        reasoning_parts.append(f"CB(L{self.circuit_breaker.state.level})={cb_mult:.2f}x")

        # 4. Concentration multiplier (reduce if many positions open)
        concentration_mult = self._concentration_multiplier(
            current_positions, max_positions
        )
        if concentration_mult < 1.0:
            reasoning_parts.append(f"Concentration={concentration_mult:.2f}x")

        # Calculate final size
        total_mult = deployer_mult * narrative_mult * cb_mult * concentration_mult
        recommended_size = self.base_size * Decimal(str(total_mult))

        # Enforce minimums and maximums
        min_size = Decimal("0.01")
        max_size = self.base_size * Decimal("1.5")  # Never more than 1.5x base
        recommended_size = max(min_size, min(recommended_size, max_size))

        # Calculate confidence (weighted average of signal strengths)
        confidence = self._calculate_confidence(
            deployer_profile, narrative_analysis, cb_mult
        )

        return PositionSizeResult(
            recommended_size_sol=recommended_size,
            base_size_sol=self.base_size,
            deployer_multiplier=deployer_mult,
            narrative_multiplier=narrative_mult,
            circuit_breaker_multiplier=cb_mult,
            concentration_multiplier=concentration_mult,
            reasoning=" × ".join(reasoning_parts),
            confidence=confidence,
        )

    def _deployer_multiplier(self, profile: DeployerProfile) -> float:
        """
        Calculate multiplier based on deployer score.

        PRD 5.2.1 Entry Logic:
        - Score >85: 2x position size (capped at 1.5x for safety)
        - Score 70-85: normal position size (1.0x)
        - Score 50-70: 0.5x position size
        - Score <50 or Unknown/New: 0.25x position size

        For new deployers (< 3 launches), always use 0.25x regardless of calculated score.
        """
        score = profile.deployer_score

        # New deployers always get minimum multiplier for safety
        if profile.is_new_deployer:
            logger.debug(
                "new_deployer_min_size",
                launches=profile.total_launches,
                multiplier=0.25,
            )
            return 0.25

        if score >= 85:
            # Elite deployer: 1.5x (capped from 2x for safety)
            return 1.5
        elif score >= 70:
            # Good deployer: 1.0 - 1.5x
            return 1.0 + (score - 70) / 30  # 1.0 at 70, 1.5 at 85
        elif score >= 50:
            # Marginal deployer: 0.5 - 1.0x
            return 0.5 + (score - 50) / 40  # 0.5 at 50, 1.0 at 70
        else:
            # Low score deployer: 0.25x (strictest)
            return 0.25

    def _narrative_multiplier(
        self, analysis: Optional[NarrativeAnalysis]
    ) -> float:
        """
        Calculate multiplier based on narrative strength.

        Score 80+ -> 1.2x (strong narrative)
        Score 60  -> 1.0x (baseline)
        Score 40- -> 0.8x (weak narrative)
        """
        if not analysis:
            return 1.0  # No narrative data, neutral

        score = analysis.narrative_score

        if score >= 80:
            return 1.2
        elif score >= 60:
            return 1.0 + (score - 60) / 100
        elif score >= 40:
            return 0.9 + (score - 40) / 200
        else:
            return 0.8

    def _concentration_multiplier(
        self, current: int, maximum: int
    ) -> float:
        """
        Reduce size when approaching max positions.

        0 positions -> 1.0x
        50% capacity -> 0.9x
        80% capacity -> 0.7x
        100% capacity -> 0.5x
        """
        if maximum <= 0:
            return 1.0

        utilization = current / maximum

        if utilization >= 1.0:
            return 0.5
        elif utilization >= 0.8:
            return 0.7
        elif utilization >= 0.5:
            return 0.9
        else:
            return 1.0

    def _calculate_confidence(
        self,
        deployer: DeployerProfile,
        narrative: Optional[NarrativeAnalysis],
        cb_mult: float,
    ) -> float:
        """
        Calculate overall trade confidence (0-1).

        High confidence = strong signals aligned
        Low confidence = weak or conflicting signals
        """
        confidence = 0.0

        # Deployer confidence (50% weight)
        deployer_conf = deployer.deployer_score / 100
        confidence += deployer_conf * 0.5

        # Narrative confidence (20% weight)
        if narrative:
            narrative_conf = (narrative.narrative_score / 100) * narrative.confidence
            confidence += narrative_conf * 0.2
        else:
            confidence += 0.5 * 0.2  # Neutral if no narrative

        # Circuit breaker confidence (30% weight)
        # Lower CB level = higher confidence in market conditions
        cb_conf = cb_mult  # 1.0 = normal, 0.25 = stressed
        confidence += cb_conf * 0.3

        return min(1.0, confidence)


def format_size_decision(result: PositionSizeResult) -> str:
    """Format position size decision for logging/display."""
    return (
        f"Size: {result.recommended_size_sol:.4f} SOL "
        f"({result.reasoning}) "
        f"[Confidence: {result.confidence:.0%}]"
    )
