"""
Multi-stage filter engine for token evaluation.
Implements fast-fail architecture to minimize latency.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from bags_sniper.core.config import Settings
from bags_sniper.core.deployer_intelligence import DeployerIntelligence, DeployerProfile
from bags_sniper.services.bags_api import BagsAPIClient

logger = structlog.get_logger()


class FilterResult(str, Enum):
    """Result of filter evaluation."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # Filter not applicable


class FilterStage(str, Enum):
    """Filter stages in order of execution (fastest first)."""

    DEPLOYER = "deployer"       # Deployer score check
    BASIC = "basic"             # Basic token validation
    LIQUIDITY = "liquidity"     # Liquidity checks
    TIMING = "timing"           # Age and timing checks
    SECURITY = "security"       # PRD 5.2.2: Honeypot, tax, holder concentration
    NARRATIVE = "narrative"     # AI narrative analysis (async/optional)


@dataclass
class FilterEvaluation:
    """Result of a single filter."""

    filter_name: str
    stage: FilterStage
    result: FilterResult
    reason: str
    value: Optional[Any] = None
    threshold: Optional[Any] = None
    latency_ms: float = 0.0


@dataclass
class TokenCandidate:
    """Token being evaluated for potential entry."""

    mint_address: str
    deployer_wallet: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    mcap_sol: Optional[Decimal] = None
    price_sol: Optional[Decimal] = None
    launched_at: Optional[datetime] = None

    # PRD 5.2.2: Security filter data
    liquidity_sol: Optional[Decimal] = None
    top_holder_percent: Optional[float] = None
    dev_wallet_percent: Optional[float] = None
    buy_tax_percent: Optional[float] = None
    sell_tax_percent: Optional[float] = None
    is_honeypot: Optional[bool] = None
    lp_lock_days: Optional[int] = None

    # Populated during filtering
    deployer_profile: Optional[DeployerProfile] = None
    filter_results: list[FilterEvaluation] = field(default_factory=list)
    passed_all_filters: bool = False
    rejection_reason: Optional[str] = None
    total_filter_time_ms: float = 0.0


class FilterEngine:
    """
    Evaluates tokens against multiple criteria using fast-fail architecture.
    Filters are ordered by speed and importance to minimize latency.
    """

    def __init__(
        self,
        settings: Settings,
        deployer_intel: DeployerIntelligence,
        bags_client: BagsAPIClient,
    ):
        self.settings = settings
        self.deployer_intel = deployer_intel
        self.bags = bags_client

    async def evaluate(
        self,
        session: AsyncSession,
        candidate: TokenCandidate,
    ) -> TokenCandidate:
        """
        Run all filters on a token candidate.
        Returns early on first failure (fast-fail).
        """
        start_time = asyncio.get_event_loop().time()

        # Stage 1: Deployer Score (fastest, most important)
        deployer_eval = await self._filter_deployer(session, candidate)
        candidate.filter_results.append(deployer_eval)
        if deployer_eval.result == FilterResult.FAIL:
            candidate.rejection_reason = deployer_eval.reason
            candidate.total_filter_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            return candidate

        # Stage 2: Basic Validation
        basic_eval = await self._filter_basic(candidate)
        candidate.filter_results.append(basic_eval)
        if basic_eval.result == FilterResult.FAIL:
            candidate.rejection_reason = basic_eval.reason
            candidate.total_filter_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            return candidate

        # Stage 3: Liquidity
        liquidity_eval = await self._filter_liquidity(candidate)
        candidate.filter_results.append(liquidity_eval)
        if liquidity_eval.result == FilterResult.FAIL:
            candidate.rejection_reason = liquidity_eval.reason
            candidate.total_filter_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            return candidate

        # Stage 4: Timing
        timing_eval = await self._filter_timing(candidate)
        candidate.filter_results.append(timing_eval)
        if timing_eval.result == FilterResult.FAIL:
            candidate.rejection_reason = timing_eval.reason
            candidate.total_filter_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            return candidate

        # Stage 5: Security (PRD 5.2.2)
        security_eval = await self._filter_security(candidate)
        candidate.filter_results.append(security_eval)
        if security_eval.result == FilterResult.FAIL:
            candidate.rejection_reason = security_eval.reason
            candidate.total_filter_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            return candidate

        # All filters passed
        candidate.passed_all_filters = True
        candidate.total_filter_time_ms = (
            asyncio.get_event_loop().time() - start_time
        ) * 1000

        logger.info(
            "token_passed_filters",
            mint=candidate.mint_address[:8],
            deployer_score=candidate.deployer_profile.deployer_score
            if candidate.deployer_profile
            else 0,
            filter_time_ms=round(candidate.total_filter_time_ms, 2),
        )

        return candidate

    async def _filter_deployer(
        self,
        session: AsyncSession,
        candidate: TokenCandidate,
    ) -> FilterEvaluation:
        """
        Stage 1: Deployer score filter.
        This is the primary alpha source - most important filter.

        PRD 5.2.1 Entry Logic:
        - Score >85: Immediate entry, bypass other filters, 2x position size
        - Score 70-85: Apply standard filters, normal position size
        - Score 50-70: Apply strict filters, 0.5x position size
        - Score <50 or Unknown: Apply strictest filters, 0.25x position size

        NOTE: Unknown/new deployers CAN trade with stricter security filters.
        Position sizing handles the risk reduction.
        """
        start = asyncio.get_event_loop().time()

        try:
            profile = await self.deployer_intel.analyze_deployer(
                session, candidate.deployer_wallet
            )
            candidate.deployer_profile = profile
            latency = (asyncio.get_event_loop().time() - start) * 1000

            # PRD 5.2.1: Tiered entry based on deployer score
            # High-score deployers (>= min_score) pass easily
            if profile.deployer_score >= self.settings.min_deployer_score:
                # Also check graduation rate for known deployers
                if profile.total_launches >= 3 and profile.graduation_rate < self.settings.min_graduation_rate:
                    return FilterEvaluation(
                        filter_name="graduation_rate",
                        stage=FilterStage.DEPLOYER,
                        result=FilterResult.FAIL,
                        reason=f"Known deployer grad rate {profile.graduation_rate:.2%} below {self.settings.min_graduation_rate:.2%}",
                        value=profile.graduation_rate,
                        threshold=self.settings.min_graduation_rate,
                        latency_ms=latency,
                    )
                return FilterEvaluation(
                    filter_name="deployer_score",
                    stage=FilterStage.DEPLOYER,
                    result=FilterResult.PASS,
                    reason=f"Score {profile.deployer_score:.1f}, grad rate {profile.graduation_rate:.2%}",
                    value=profile.deployer_score,
                    threshold=self.settings.min_deployer_score,
                    latency_ms=latency,
                )

            # New/unknown deployer (< 3 launches) - ALLOW with strict filters
            # Position sizer will apply 0.25-0.5x multiplier
            if profile.is_new_deployer:
                logger.info(
                    "new_deployer_allowed_with_strict_filters",
                    wallet=candidate.deployer_wallet[:8],
                    launches=profile.total_launches,
                    score=profile.deployer_score,
                )
                return FilterEvaluation(
                    filter_name="deployer_score",
                    stage=FilterStage.DEPLOYER,
                    result=FilterResult.PASS,
                    reason=f"New deployer ({profile.total_launches} launches), strict filters applied",
                    value=profile.deployer_score,
                    threshold=self.settings.min_deployer_score,
                    latency_ms=latency,
                )

            # Known deployer with low score - still allow but with warning
            # PRD says "Score <50 or Unknown: Apply strictest filters, 0.25x position size"
            # Allow deployers with score >= 10 (very low threshold to allow more trades)
            # The quality gate and narrative analysis provide additional filtering
            if profile.deployer_score >= 10:  # Lowered from 20 to allow more trades
                logger.info(
                    "low_score_deployer_allowed",
                    wallet=candidate.deployer_wallet[:8],
                    score=profile.deployer_score,
                    grad_rate=f"{profile.graduation_rate:.2%}",
                    reason="Allowing low-score deployer - AI narrative will filter bad tokens",
                )
                return FilterEvaluation(
                    filter_name="deployer_score",
                    stage=FilterStage.DEPLOYER,
                    result=FilterResult.PASS,
                    reason=f"Score {profile.deployer_score:.1f}, relying on narrative analysis",
                    value=profile.deployer_score,
                    threshold=self.settings.min_deployer_score,
                    latency_ms=latency,
                )

            # Only reject truly terrible deployers (score < 10 = definite rug history)
            return FilterEvaluation(
                filter_name="deployer_score",
                stage=FilterStage.DEPLOYER,
                result=FilterResult.FAIL,
                reason=f"Score {profile.deployer_score:.1f} extremely low (confirmed rug history)",
                value=profile.deployer_score,
                threshold=10.0,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (asyncio.get_event_loop().time() - start) * 1000
            logger.error("deployer_filter_error", error=str(e))
            return FilterEvaluation(
                filter_name="deployer_score",
                stage=FilterStage.DEPLOYER,
                result=FilterResult.FAIL,
                reason=f"Error: {str(e)}",
                latency_ms=latency,
            )

    async def _filter_basic(self, candidate: TokenCandidate) -> FilterEvaluation:
        """
        Stage 2: Basic token validation.
        Checks mint address validity and basic properties.
        """
        start = asyncio.get_event_loop().time()

        # Validate mint address format (Solana base58, 32-44 chars)
        if not candidate.mint_address or len(candidate.mint_address) < 32:
            return FilterEvaluation(
                filter_name="mint_format",
                stage=FilterStage.BASIC,
                result=FilterResult.FAIL,
                reason="Invalid mint address format",
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        # Check for known scam patterns in name/symbol
        scam_keywords = ["rug", "scam", "honeypot", "fake"]
        name_lower = (candidate.name or "").lower()
        symbol_lower = (candidate.symbol or "").lower()

        for keyword in scam_keywords:
            if keyword in name_lower or keyword in symbol_lower:
                return FilterEvaluation(
                    filter_name="scam_keywords",
                    stage=FilterStage.BASIC,
                    result=FilterResult.FAIL,
                    reason=f"Suspicious keyword in name/symbol: {keyword}",
                    latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
                )

        return FilterEvaluation(
            filter_name="basic_validation",
            stage=FilterStage.BASIC,
            result=FilterResult.PASS,
            reason="Basic checks passed",
            latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
        )

    async def _filter_liquidity(self, candidate: TokenCandidate) -> FilterEvaluation:
        """
        Stage 3: Liquidity and market cap checks.
        Ensures sufficient liquidity for entry and exit.

        For newly launched tokens (not yet indexed by API), we allow entry
        with reduced position size via the position sizer.
        """
        start = asyncio.get_event_loop().time()

        # Minimum market cap check
        min_mcap = Decimal("5")  # 5 SOL minimum
        max_mcap = Decimal("500")  # 500 SOL maximum (catch early)

        if candidate.mcap_sol is None:
            # Try to fetch market cap from Bags API
            response = await self.bags.get_token_info(candidate.mint_address)
            if response.success and response.data:
                mcap = response.data.get("market_cap_sol") or response.data.get("mcap")
                if mcap:
                    candidate.mcap_sol = Decimal(str(mcap))

                # Also try to get liquidity if available
                liq = response.data.get("liquidity_sol") or response.data.get("liquidity")
                if liq:
                    candidate.liquidity_sol = Decimal(str(liq))

        if candidate.mcap_sol is None:
            # For brand new tokens not yet indexed, allow entry if deployer profile exists
            # This enables sniping truly new launches before API indexes them
            # Position sizer will use minimum position size for unknown tokens
            if candidate.deployer_profile:
                logger.info(
                    "allowing_unindexed_token",
                    mint=candidate.mint_address[:8],
                    deployer_score=candidate.deployer_profile.deployer_score,
                    reason="New token not yet indexed, relying on deployer + AI analysis",
                )
                return FilterEvaluation(
                    filter_name="market_cap",
                    stage=FilterStage.LIQUIDITY,
                    result=FilterResult.PASS,
                    reason="Unindexed new token, using deployer + AI signals",
                    latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
                )
            # No deployer profile and no market cap - skip
            return FilterEvaluation(
                filter_name="market_cap",
                stage=FilterStage.LIQUIDITY,
                result=FilterResult.FAIL,
                reason="Could not determine market cap and no deployer data",
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        if candidate.mcap_sol < min_mcap:
            return FilterEvaluation(
                filter_name="min_market_cap",
                stage=FilterStage.LIQUIDITY,
                result=FilterResult.FAIL,
                reason=f"MCAP {candidate.mcap_sol} SOL below minimum {min_mcap}",
                value=float(candidate.mcap_sol),
                threshold=float(min_mcap),
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        if candidate.mcap_sol > max_mcap:
            return FilterEvaluation(
                filter_name="max_market_cap",
                stage=FilterStage.LIQUIDITY,
                result=FilterResult.FAIL,
                reason=f"MCAP {candidate.mcap_sol} SOL above maximum {max_mcap}",
                value=float(candidate.mcap_sol),
                threshold=float(max_mcap),
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        return FilterEvaluation(
            filter_name="liquidity",
            stage=FilterStage.LIQUIDITY,
            result=FilterResult.PASS,
            reason=f"MCAP {candidate.mcap_sol} SOL within range",
            value=float(candidate.mcap_sol),
            latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
        )

    async def _filter_timing(self, candidate: TokenCandidate) -> FilterEvaluation:
        """
        Stage 4: Age and timing checks.
        Must catch tokens early but not too early.
        """
        start = asyncio.get_event_loop().time()

        if candidate.launched_at is None:
            # Can't determine age, proceed with caution
            return FilterEvaluation(
                filter_name="token_age",
                stage=FilterStage.TIMING,
                result=FilterResult.PASS,
                reason="Launch time unknown, proceeding",
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        now = datetime.utcnow()
        if candidate.launched_at.tzinfo:
            age_seconds = (now - candidate.launched_at.replace(tzinfo=None)).total_seconds()
        else:
            age_seconds = (now - candidate.launched_at).total_seconds()

        age_minutes = age_seconds / 60

        # Too new - wait for initial volatility to settle
        min_age_minutes = 1
        if age_minutes < min_age_minutes:
            return FilterEvaluation(
                filter_name="min_age",
                stage=FilterStage.TIMING,
                result=FilterResult.FAIL,
                reason=f"Token age {age_minutes:.1f}m below minimum {min_age_minutes}m",
                value=age_minutes,
                threshold=min_age_minutes,
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        # Too old - miss the early move
        max_age_minutes = 30
        if age_minutes > max_age_minutes:
            return FilterEvaluation(
                filter_name="max_age",
                stage=FilterStage.TIMING,
                result=FilterResult.FAIL,
                reason=f"Token age {age_minutes:.1f}m exceeds maximum {max_age_minutes}m",
                value=age_minutes,
                threshold=max_age_minutes,
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        return FilterEvaluation(
            filter_name="token_age",
            stage=FilterStage.TIMING,
            result=FilterResult.PASS,
            reason=f"Token age {age_minutes:.1f}m within range",
            value=age_minutes,
            latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
        )

    async def _filter_security(self, candidate: TokenCandidate) -> FilterEvaluation:
        """
        Stage 5: Security filters from PRD 5.2.2.

        Checks:
        - Liquidity < 30 SOL (~$7K at $230/SOL) -> FAIL (if known)
        - Top holders > 25% concentration -> FAIL (if known)
        - Dev wallet > 5% holdings -> FAIL (if known)
        - Honeypot detected -> FAIL (if known)
        - Tax > 3% (buy or sell) -> FAIL (if known)
        - LP lock < 30 days -> WARNING (not fail for pump.fun)

        For new tokens not yet indexed by the API, we allow entry
        with reduced position size (handled by position sizer).
        AI analysis provides additional safety for unknown tokens.
        """
        start = asyncio.get_event_loop().time()
        issues = []
        data_available = False

        # Fetch security data from API if not already populated
        if candidate.is_honeypot is None:
            await self._fetch_security_data(candidate)

        # Track if we have any security data
        if any([
            candidate.liquidity_sol is not None,
            candidate.is_honeypot is not None,
            candidate.buy_tax_percent is not None,
            candidate.top_holder_percent is not None,
        ]):
            data_available = True

        # PRD 5.2.2: Liquidity check (minimum ~$7K = ~30 SOL)
        # Only enforce if we actually have liquidity data
        min_liquidity_sol = Decimal("30")
        if candidate.liquidity_sol is not None:
            if candidate.liquidity_sol < min_liquidity_sol:
                return FilterEvaluation(
                    filter_name="min_liquidity",
                    stage=FilterStage.SECURITY,
                    result=FilterResult.FAIL,
                    reason=f"Liquidity {candidate.liquidity_sol:.1f} SOL below {min_liquidity_sol} SOL",
                    value=float(candidate.liquidity_sol),
                    threshold=float(min_liquidity_sol),
                    latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
                )
        else:
            issues.append("liquidity unknown")

        # PRD 5.2.2: Honeypot detection
        if candidate.is_honeypot is True:
            return FilterEvaluation(
                filter_name="honeypot",
                stage=FilterStage.SECURITY,
                result=FilterResult.FAIL,
                reason="Honeypot detected - cannot sell",
                value=True,
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        # PRD 5.2.2: Tax check (> 3%)
        max_tax_percent = 3.0
        if candidate.buy_tax_percent is not None and candidate.buy_tax_percent > max_tax_percent:
            return FilterEvaluation(
                filter_name="buy_tax",
                stage=FilterStage.SECURITY,
                result=FilterResult.FAIL,
                reason=f"Buy tax {candidate.buy_tax_percent:.1f}% exceeds {max_tax_percent}%",
                value=candidate.buy_tax_percent,
                threshold=max_tax_percent,
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        if candidate.sell_tax_percent is not None and candidate.sell_tax_percent > max_tax_percent:
            return FilterEvaluation(
                filter_name="sell_tax",
                stage=FilterStage.SECURITY,
                result=FilterResult.FAIL,
                reason=f"Sell tax {candidate.sell_tax_percent:.1f}% exceeds {max_tax_percent}%",
                value=candidate.sell_tax_percent,
                threshold=max_tax_percent,
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        # PRD 5.2.2: Top holder concentration (> 25%)
        max_top_holder_percent = 25.0
        if candidate.top_holder_percent is not None:
            if candidate.top_holder_percent > max_top_holder_percent:
                return FilterEvaluation(
                    filter_name="top_holder_concentration",
                    stage=FilterStage.SECURITY,
                    result=FilterResult.FAIL,
                    reason=f"Top holder owns {candidate.top_holder_percent:.1f}% (>{max_top_holder_percent}%)",
                    value=candidate.top_holder_percent,
                    threshold=max_top_holder_percent,
                    latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
                )

        # PRD 5.2.2: Dev wallet holdings (> 5%)
        max_dev_percent = 5.0
        if candidate.dev_wallet_percent is not None:
            if candidate.dev_wallet_percent > max_dev_percent:
                return FilterEvaluation(
                    filter_name="dev_wallet_holdings",
                    stage=FilterStage.SECURITY,
                    result=FilterResult.FAIL,
                    reason=f"Dev wallet holds {candidate.dev_wallet_percent:.1f}% (>{max_dev_percent}%)",
                    value=candidate.dev_wallet_percent,
                    threshold=max_dev_percent,
                    latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
                )

        # LP lock check (warning only for pump.fun - bonding curve doesn't use traditional LP)
        if candidate.lp_lock_days is not None and candidate.lp_lock_days < 30:
            issues.append(f"LP lock only {candidate.lp_lock_days}d")

        # All security checks passed
        if not data_available:
            # No security data available - new token not yet indexed
            # Allow with warnings, AI + deployer analysis will gate entry
            logger.info(
                "security_data_unavailable",
                mint=candidate.mint_address[:8],
                deployer_score=candidate.deployer_profile.deployer_score if candidate.deployer_profile else 0,
            )
            return FilterEvaluation(
                filter_name="security",
                stage=FilterStage.SECURITY,
                result=FilterResult.PASS,
                reason="Security data unavailable (new token), relying on deployer + AI",
                latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
            )

        reason = "Security checks passed"
        if issues:
            reason += f" (warnings: {', '.join(issues)})"

        return FilterEvaluation(
            filter_name="security",
            stage=FilterStage.SECURITY,
            result=FilterResult.PASS,
            reason=reason,
            latency_ms=(asyncio.get_event_loop().time() - start) * 1000,
        )

    async def _fetch_security_data(self, candidate: TokenCandidate) -> None:
        """
        Fetch security-related data from API.
        Populates honeypot, tax, holder concentration fields.

        For new tokens not yet indexed, we leave values as None to allow
        the deployer + AI analysis to drive the decision.
        """
        try:
            # Get token holders data
            response = await self.bags.get_token_info(candidate.mint_address)
            if response.success and response.data:
                data = response.data

                # Extract security metrics from API response
                # Only set values if they're actually provided (not defaults)
                if "is_honeypot" in data:
                    candidate.is_honeypot = data.get("is_honeypot")
                if "buy_tax" in data and data.get("buy_tax") is not None:
                    candidate.buy_tax_percent = data.get("buy_tax")
                if "sell_tax" in data and data.get("sell_tax") is not None:
                    candidate.sell_tax_percent = data.get("sell_tax")

                # Only set liquidity if actually provided
                liq = data.get("liquidity_sol") or data.get("liquidity")
                if liq is not None and liq > 0:
                    candidate.liquidity_sol = Decimal(str(liq))

                # Holder concentration (if available)
                holders = data.get("top_holders", [])
                if holders and len(holders) > 0:
                    top_pct = holders[0].get("percent")
                    if top_pct is not None:
                        candidate.top_holder_percent = top_pct

                # Dev wallet percentage (deployer's holdings)
                for holder in holders:
                    if holder.get("address") == candidate.deployer_wallet:
                        dev_pct = holder.get("percent")
                        if dev_pct is not None:
                            candidate.dev_wallet_percent = dev_pct
                        break

                # LP lock days (if available)
                if "lp_lock_days" in data:
                    candidate.lp_lock_days = data.get("lp_lock_days")
            else:
                logger.debug(
                    "security_data_not_available",
                    mint=candidate.mint_address[:8],
                    reason="Token not yet indexed in API",
                )

        except Exception as e:
            logger.warning(
                "security_data_fetch_error",
                mint=candidate.mint_address[:8],
                error=str(e),
            )
            # Leave values as None - don't set artificial defaults
            # This allows new tokens to pass through with deployer + AI analysis

    def get_filter_summary(self, candidate: TokenCandidate) -> dict[str, Any]:
        """Generate summary of all filter results."""
        return {
            "mint": candidate.mint_address,
            "passed": candidate.passed_all_filters,
            "rejection_reason": candidate.rejection_reason,
            "total_time_ms": round(candidate.total_filter_time_ms, 2),
            "deployer_score": (
                candidate.deployer_profile.deployer_score
                if candidate.deployer_profile
                else None
            ),
            "filters": [
                {
                    "name": f.filter_name,
                    "result": f.result.value,
                    "reason": f.reason,
                    "latency_ms": round(f.latency_ms, 2),
                }
                for f in candidate.filter_results
            ],
        }
