"""
Deployer Intelligence Engine.
Core alpha source: tracking deployer wallet success patterns.
Top deployers show 7,700x better performance than average.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select, func, Integer
from sqlalchemy.ext.asyncio import AsyncSession

from bags_sniper.core.config import Settings
from bags_sniper.models.database import Deployer, TokenLaunch
from bags_sniper.services.solana_rpc import SolanaRPCClient

logger = structlog.get_logger()


# Baseline graduation rate from research: 1.4%
BASELINE_GRADUATION_RATE = 0.014

# Score thresholds
SCORE_EXCELLENT = 80.0
SCORE_GOOD = 60.0
SCORE_MARGINAL = 40.0


@dataclass
class DeployerProfile:
    """Complete deployer analysis profile."""

    wallet_address: str
    deployer_score: float
    graduation_rate: float
    total_launches: int
    graduated_launches: int
    avg_peak_mcap_sol: Decimal
    estimated_profit_sol: Decimal
    days_since_last_launch: Optional[int]
    launches_last_7d: int
    is_new_deployer: bool
    score_components: dict[str, float]
    recommendation: str


class DeployerIntelligence:
    """
    Analyzes and scores token deployers based on historical performance.

    The core insight: Top 10 deployers made 365,000 SOL vs 47.3 SOL average.
    That's a 7,700x edge from tracking the right deployers.
    """

    def __init__(
        self,
        settings: Settings,
        rpc_client: SolanaRPCClient,
    ):
        self.settings = settings
        self.rpc = rpc_client

        # Scoring weights (must sum to 100)
        self.weights = {
            "graduation_rate": 40.0,  # Primary factor
            "avg_mcap": 30.0,         # Profitability proxy
            "recency": 20.0,          # Recent activity
            "consistency": 10.0,      # Track record depth
        }

    async def get_or_create_deployer(
        self,
        session: AsyncSession,
        wallet_address: str,
    ) -> Deployer:
        """Get existing deployer or create new one."""
        result = await session.execute(
            select(Deployer).where(Deployer.wallet_address == wallet_address)
        )
        deployer = result.scalar_one_or_none()

        if not deployer:
            deployer = Deployer(wallet_address=wallet_address)
            session.add(deployer)
            await session.flush()
            logger.info("new_deployer_created", wallet=wallet_address)

        return deployer

    async def analyze_deployer(
        self,
        session: AsyncSession,
        wallet_address: str,
        force_refresh: bool = False,
    ) -> DeployerProfile:
        """
        Generate complete deployer analysis profile.

        Args:
            session: Database session
            wallet_address: Deployer's wallet address
            force_refresh: Force recalculation even if recently updated
        """
        deployer = await self.get_or_create_deployer(session, wallet_address)

        # Check if we need to refresh (stale after 1 hour)
        needs_refresh = force_refresh or (
            deployer.score_updated_at is None
            or datetime.utcnow() - deployer.score_updated_at.replace(tzinfo=None)
            > timedelta(hours=1)
        )

        if needs_refresh:
            await self._refresh_deployer_stats(session, deployer)

        # Calculate score components
        score_components = self._calculate_score_components(deployer)
        deployer.deployer_score = sum(score_components.values())
        deployer.score_updated_at = datetime.utcnow()

        await session.commit()

        # Determine recommendation
        recommendation = self._get_recommendation(deployer)

        # Calculate days since last launch
        days_since_last = None
        if deployer.last_launch_at:
            delta = datetime.utcnow() - deployer.last_launch_at.replace(tzinfo=None)
            days_since_last = delta.days

        return DeployerProfile(
            wallet_address=wallet_address,
            deployer_score=deployer.deployer_score,
            graduation_rate=deployer.graduation_rate,
            total_launches=deployer.total_launches,
            graduated_launches=deployer.graduated_launches,
            avg_peak_mcap_sol=deployer.avg_peak_mcap_sol,
            estimated_profit_sol=deployer.estimated_profit_sol,
            days_since_last_launch=days_since_last,
            launches_last_7d=deployer.launches_last_7d,
            is_new_deployer=deployer.total_launches < 3,
            score_components=score_components,
            recommendation=recommendation,
        )

    async def _refresh_deployer_stats(
        self,
        session: AsyncSession,
        deployer: Deployer,
    ):
        """Recalculate deployer statistics from their token launches."""
        # Get aggregated stats
        result = await session.execute(
            select(
                func.count(TokenLaunch.id).label("total"),
                func.sum(
                    func.cast(TokenLaunch.graduated, Integer)
                ).label("graduated"),
                func.avg(TokenLaunch.peak_mcap_sol).label("avg_peak"),
                func.max(TokenLaunch.launched_at).label("last_launch"),
            ).where(TokenLaunch.deployer_id == deployer.id)
        )
        stats = result.one()

        deployer.total_launches = stats.total or 0
        deployer.graduated_launches = int(stats.graduated or 0)
        deployer.avg_peak_mcap_sol = Decimal(str(stats.avg_peak or 0))
        deployer.last_launch_at = stats.last_launch

        # Calculate graduation rate
        if deployer.total_launches > 0:
            deployer.graduation_rate = (
                deployer.graduated_launches / deployer.total_launches
            )
        else:
            deployer.graduation_rate = 0.0

        # Recent activity
        week_ago = datetime.utcnow() - timedelta(days=7)
        day_ago = datetime.utcnow() - timedelta(days=1)

        result = await session.execute(
            select(func.count(TokenLaunch.id)).where(
                TokenLaunch.deployer_id == deployer.id,
                TokenLaunch.launched_at >= week_ago,
            )
        )
        deployer.launches_last_7d = result.scalar() or 0

        result = await session.execute(
            select(func.count(TokenLaunch.id)).where(
                TokenLaunch.deployer_id == deployer.id,
                TokenLaunch.launched_at >= day_ago,
            )
        )
        deployer.launches_last_24h = result.scalar() or 0

        logger.debug(
            "deployer_stats_refreshed",
            wallet=deployer.wallet_address,
            total_launches=deployer.total_launches,
            graduation_rate=f"{deployer.graduation_rate:.2%}",
        )

    def _calculate_score_components(self, deployer: Deployer) -> dict[str, float]:
        """
        Calculate individual score components.
        Returns dict with component scores that sum to total score.

        For NEW DEPLOYERS (< 3 launches): We give a "benefit of the doubt" baseline
        since most memecoin deployers are new. Without this, we'd reject 95%+ of
        opportunities. The AI narrative analysis provides the safety net.

        CRITICAL: The baseline is applied as a MINIMUM for deployers with limited
        history. This ensures they get at least ~50 points instead of being
        stuck at 19.3 (just recency).
        """
        components = {}
        is_new_deployer = deployer.total_launches < 3
        has_no_graduations = deployer.graduated_launches == 0

        # Define baseline values for new deployers (used as MINIMUM values)
        # These ensure a minimum score of ~50 for deployers with no track record
        BASELINE_GRAD_RATIO = 0.5       # 50% = 20 points
        BASELINE_MCAP_RATIO = 0.35      # 35% = 10.5 points
        BASELINE_RECENCY_RATIO = 1.0    # 100% = 20 points (brand new)
        BASELINE_CONSISTENCY_RATIO = 0.25  # 25% = 2.5 points
        # Total baseline: ~53 points

        # 1. Graduation Rate Component (40 points max)
        target_grad_rate = BASELINE_GRADUATION_RATE * 3  # 4.2%
        if deployer.graduation_rate > 0 and target_grad_rate > 0:
            grad_ratio = min(deployer.graduation_rate / target_grad_rate, 1.0)
        else:
            grad_ratio = 0

        # CRITICAL FIX: Apply baseline as MINIMUM for new deployers with no graduations
        if is_new_deployer and has_no_graduations:
            grad_ratio = max(grad_ratio, BASELINE_GRAD_RATIO)

        components["graduation_rate"] = grad_ratio * self.weights["graduation_rate"]

        # 2. Average Peak MCAP Component (30 points max)
        if deployer.avg_peak_mcap_sol > 0:
            mcap_normalized = min(float(deployer.avg_peak_mcap_sol) / 1000, 1.0)
        else:
            mcap_normalized = 0

        # Apply baseline as MINIMUM for new deployers
        if is_new_deployer:
            mcap_normalized = max(mcap_normalized, BASELINE_MCAP_RATIO)

        components["avg_mcap"] = mcap_normalized * self.weights["avg_mcap"]

        # 3. Recency Component (20 points max)
        if deployer.last_launch_at:
            days_since = (
                datetime.utcnow() - deployer.last_launch_at.replace(tzinfo=None)
            ).days
            recency_ratio = max(0, 1 - (days_since / 30))
        else:
            recency_ratio = BASELINE_RECENCY_RATIO

        # New deployers get at least 80% recency (16 points)
        if is_new_deployer:
            recency_ratio = max(recency_ratio, 0.8)

        components["recency"] = recency_ratio * self.weights["recency"]

        # 4. Consistency Component (10 points max)
        if deployer.graduated_launches > 0:
            consistency_ratio = min(deployer.graduated_launches / 5, 1.0)
        else:
            consistency_ratio = 0

        # Apply baseline as MINIMUM for new deployers
        if is_new_deployer:
            consistency_ratio = max(consistency_ratio, BASELINE_CONSISTENCY_RATIO)

        components["consistency"] = consistency_ratio * self.weights["consistency"]

        # Log components for debugging (at INFO level to see in output)
        logger.info(
            "deployer_score_components",
            wallet=deployer.wallet_address[:8] if deployer.wallet_address else "unknown",
            is_new=is_new_deployer,
            total_launches=deployer.total_launches,
            graduated=deployer.graduated_launches,
            components={k: round(v, 1) for k, v in components.items()},
            total=round(sum(components.values()), 1),
        )

        return components

    def _get_recommendation(self, deployer: Deployer) -> str:
        """Generate trading recommendation based on score."""
        score = deployer.deployer_score

        if score >= SCORE_EXCELLENT:
            return "STRONG BUY - Elite deployer with excellent track record"
        elif score >= SCORE_GOOD:
            return "BUY - Good deployer, meets minimum criteria"
        elif score >= SCORE_MARGINAL:
            return "CAUTION - Below threshold, consider smaller position"
        else:
            return "SKIP - Poor track record, do not trade"

    async def score_meets_threshold(
        self,
        session: AsyncSession,
        wallet_address: str,
    ) -> tuple[bool, float]:
        """
        Quick check if deployer meets minimum score threshold.
        Returns (meets_threshold, score).
        """
        profile = await self.analyze_deployer(session, wallet_address)
        meets = profile.deployer_score >= self.settings.min_deployer_score
        return meets, profile.deployer_score

    async def get_top_deployers(
        self,
        session: AsyncSession,
        limit: int = 20,
        min_launches: int = 3,
    ) -> list[Deployer]:
        """Get top scoring deployers for watchlist."""
        result = await session.execute(
            select(Deployer)
            .where(Deployer.total_launches >= min_launches)
            .order_by(Deployer.deployer_score.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def record_token_launch(
        self,
        session: AsyncSession,
        deployer_wallet: str,
        mint_address: str,
        token_name: Optional[str] = None,
        token_symbol: Optional[str] = None,
        initial_mcap_sol: Optional[Decimal] = None,
    ) -> TokenLaunch:
        """Record a new token launch by a deployer."""
        deployer = await self.get_or_create_deployer(session, deployer_wallet)

        launch = TokenLaunch(
            mint_address=mint_address,
            deployer_id=deployer.id,
            name=token_name,
            symbol=token_symbol,
            initial_mcap_sol=initial_mcap_sol,
            launched_at=datetime.utcnow(),
        )
        session.add(launch)
        await session.flush()

        # Update deployer stats
        deployer.total_launches += 1
        deployer.last_launch_at = datetime.utcnow()
        deployer.launches_last_24h += 1
        deployer.launches_last_7d += 1

        logger.info(
            "token_launch_recorded",
            deployer=deployer_wallet[:8],
            mint=mint_address[:8],
            token=token_symbol or token_name,
        )

        return launch

    async def record_graduation(
        self,
        session: AsyncSession,
        mint_address: str,
        graduation_mcap_sol: Decimal,
    ):
        """Record when a token graduates (hits $69K MCAP threshold)."""
        result = await session.execute(
            select(TokenLaunch).where(TokenLaunch.mint_address == mint_address)
        )
        launch = result.scalar_one_or_none()

        if not launch:
            logger.warning("graduation_unknown_token", mint=mint_address)
            return

        if launch.graduated:
            return  # Already recorded

        launch.graduated = True
        launch.graduated_at = datetime.utcnow()
        launch.graduation_mcap_sol = graduation_mcap_sol

        if launch.launched_at:
            delta = datetime.utcnow() - launch.launched_at.replace(tzinfo=None)
            launch.time_to_graduation_minutes = delta.total_seconds() / 60

        # Update deployer stats
        result = await session.execute(
            select(Deployer).where(Deployer.id == launch.deployer_id)
        )
        deployer = result.scalar_one()
        deployer.graduated_launches += 1
        deployer.graduation_rate = deployer.graduated_launches / max(
            deployer.total_launches, 1
        )

        logger.info(
            "token_graduated",
            mint=mint_address[:8],
            deployer=deployer.wallet_address[:8],
            time_minutes=launch.time_to_graduation_minutes,
            deployer_grad_rate=f"{deployer.graduation_rate:.2%}",
        )

        await session.commit()
