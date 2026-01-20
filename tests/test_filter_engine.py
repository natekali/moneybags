"""Tests for the multi-stage filter engine."""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from bags_sniper.core.filter_engine import (
    FilterEngine,
    FilterResult,
    FilterStage,
    TokenCandidate,
)


class TestFilterStages:
    """Test individual filter stages."""

    @pytest.mark.asyncio
    async def test_deployer_filter_passes_good_deployer(
        self, mock_settings, db_session, elite_deployer
    ):
        """Good deployer should pass deployer filter."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet=elite_deployer.wallet_address,
        )

        result = await filter_engine._filter_deployer(db_session, candidate)

        assert result.result == FilterResult.PASS
        assert result.stage == FilterStage.DEPLOYER

    @pytest.mark.asyncio
    async def test_deployer_filter_fails_poor_deployer(
        self, mock_settings, db_session, poor_deployer
    ):
        """Poor deployer should fail deployer filter."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet=poor_deployer.wallet_address,
        )

        result = await filter_engine._filter_deployer(db_session, candidate)

        assert result.result == FilterResult.FAIL

    @pytest.mark.asyncio
    async def test_basic_filter_catches_scam_keywords(self, mock_settings):
        """Basic filter should catch scam keywords."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet="Deployer12345678901234567890123456789012345",
            name="RUGPULL Token",
            symbol="SCAM",
        )

        result = await filter_engine._filter_basic(candidate)

        assert result.result == FilterResult.FAIL
        assert "scam" in result.reason.lower() or "rug" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_basic_filter_passes_normal_token(self, mock_settings):
        """Normal token should pass basic filter."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet="Deployer12345678901234567890123456789012345",
            name="Cool Meme Token",
            symbol="COOL",
        )

        result = await filter_engine._filter_basic(candidate)

        assert result.result == FilterResult.PASS


class TestLiquidityFilter:
    """Test liquidity and market cap filtering."""

    @pytest.mark.asyncio
    async def test_rejects_too_low_mcap(self, mock_settings):
        """Should reject tokens with MCAP below minimum."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()
        bags_mock.get_token_info = AsyncMock(return_value=MagicMock(
            success=True,
            data={"market_cap_sol": 2}  # Below 5 SOL minimum
        ))

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet="Deployer12345678901234567890123456789012345",
            mcap_sol=Decimal("2"),
        )

        result = await filter_engine._filter_liquidity(candidate)

        assert result.result == FilterResult.FAIL
        assert "below minimum" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_rejects_too_high_mcap(self, mock_settings):
        """Should reject tokens with MCAP above maximum (too late)."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet="Deployer12345678901234567890123456789012345",
            mcap_sol=Decimal("600"),  # Above 500 SOL maximum
        )

        result = await filter_engine._filter_liquidity(candidate)

        assert result.result == FilterResult.FAIL
        assert "above maximum" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_passes_good_mcap(self, mock_settings):
        """Should pass tokens with MCAP in range."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet="Deployer12345678901234567890123456789012345",
            mcap_sol=Decimal("50"),  # Within 5-500 range
        )

        result = await filter_engine._filter_liquidity(candidate)

        assert result.result == FilterResult.PASS


class TestTimingFilter:
    """Test timing and age filtering."""

    @pytest.mark.asyncio
    async def test_rejects_too_new(self, mock_settings):
        """Should reject tokens launched less than 1 minute ago."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet="Deployer12345678901234567890123456789012345",
            launched_at=datetime.utcnow() - timedelta(seconds=30),
        )

        result = await filter_engine._filter_timing(candidate)

        assert result.result == FilterResult.FAIL
        assert "below minimum" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_rejects_too_old(self, mock_settings):
        """Should reject tokens launched more than 30 minutes ago."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet="Deployer12345678901234567890123456789012345",
            launched_at=datetime.utcnow() - timedelta(minutes=45),
        )

        result = await filter_engine._filter_timing(candidate)

        assert result.result == FilterResult.FAIL
        assert "exceeds maximum" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_passes_good_timing(self, mock_settings):
        """Should pass tokens with age in range (1-30 minutes)."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet="Deployer12345678901234567890123456789012345",
            launched_at=datetime.utcnow() - timedelta(minutes=10),
        )

        result = await filter_engine._filter_timing(candidate)

        assert result.result == FilterResult.PASS


class TestFullFilterPipeline:
    """Test the complete filter pipeline."""

    @pytest.mark.asyncio
    async def test_fast_fail_on_deployer(
        self, mock_settings, db_session, poor_deployer
    ):
        """Should fail fast on deployer check without running other filters."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()
        bags_mock.get_token_info = AsyncMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet=poor_deployer.wallet_address,
            name="Good Token",
            symbol="GOOD",
            mcap_sol=Decimal("50"),
            launched_at=datetime.utcnow() - timedelta(minutes=5),
        )

        result = await filter_engine.evaluate(db_session, candidate)

        assert result.passed_all_filters is False
        # Should only have deployer filter result (fast fail)
        assert len(result.filter_results) == 1
        assert result.filter_results[0].stage == FilterStage.DEPLOYER
        # Should not have called bags API since we failed early
        bags_mock.get_token_info.assert_not_called()

    @pytest.mark.asyncio
    async def test_passes_all_filters(
        self, mock_settings, db_session, elite_deployer
    ):
        """Good candidate should pass all filters."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet=elite_deployer.wallet_address,
            name="Elite Token",
            symbol="ELITE",
            mcap_sol=Decimal("50"),
            launched_at=datetime.utcnow() - timedelta(minutes=5),
        )

        result = await filter_engine.evaluate(db_session, candidate)

        assert result.passed_all_filters is True
        assert result.rejection_reason is None
        assert result.deployer_profile is not None
        # Should have all 5 filter results (deployer, basic, liquidity, timing, security)
        assert len(result.filter_results) == 5

    @pytest.mark.asyncio
    async def test_filter_summary_format(
        self, mock_settings, db_session, elite_deployer
    ):
        """Filter summary should contain required fields."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        deployer_intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()

        filter_engine = FilterEngine(mock_settings, deployer_intel, bags_mock)

        candidate = TokenCandidate(
            mint_address="TokenMint123456789012345678901234567890123456",
            deployer_wallet=elite_deployer.wallet_address,
            mcap_sol=Decimal("50"),
            launched_at=datetime.utcnow() - timedelta(minutes=5),
        )

        await filter_engine.evaluate(db_session, candidate)
        summary = filter_engine.get_filter_summary(candidate)

        assert "mint" in summary
        assert "passed" in summary
        assert "total_time_ms" in summary
        assert "deployer_score" in summary
        assert "filters" in summary
