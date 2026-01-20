"""Tests for the deployer intelligence scoring system."""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bags_sniper.models.database import Deployer


class TestDeployerScoring:
    """Test deployer score calculation."""

    def test_score_components_sum_to_100_max(self):
        """Score components should sum to 100 for perfect deployer."""
        deployer = Deployer(
            wallet_address="PerfectDeployer12345678901234567890123456789",
            total_launches=10,
            graduated_launches=5,  # 50% grad rate (well above 4.2% target)
            graduation_rate=0.5,
            avg_peak_mcap_sol=Decimal("1000"),  # Max points
            last_launch_at=datetime.utcnow(),  # Recent
        )
        deployer.graduated_launches = 5  # For consistency calculation

        score = deployer.calculate_score()
        # Should be close to 100 for perfect deployer
        assert score >= 95

    def test_zero_score_for_empty_deployer(self):
        """New deployer with no data should have low score."""
        deployer = Deployer(
            wallet_address="NewDeployer123456789012345678901234567890123",
            total_launches=0,
            graduated_launches=0,
            graduation_rate=0.0,
            avg_peak_mcap_sol=Decimal("0"),
        )

        score = deployer.calculate_score()
        assert score == 0

    def test_graduation_rate_weighted_40_percent(self):
        """Graduation rate should contribute up to 40 points."""
        # Deployer with only graduation rate
        deployer = Deployer(
            wallet_address="GradOnlyDeployer1234567890123456789012345678",
            total_launches=10,
            graduated_launches=1,
            graduation_rate=0.1,  # 10% = target rate for full points
            avg_peak_mcap_sol=Decimal("0"),
        )

        score = deployer.calculate_score()
        # Should be around 40 (graduation component only)
        assert 38 <= score <= 42

    def test_recency_decay(self):
        """Score should decay based on time since last launch."""
        # Recent deployer
        recent = Deployer(
            wallet_address="RecentDeployer12345678901234567890123456789",
            total_launches=10,
            graduated_launches=5,
            graduation_rate=0.5,
            avg_peak_mcap_sol=Decimal("500"),
            last_launch_at=datetime.utcnow(),
        )
        recent.graduated_launches = 5
        recent_score = recent.calculate_score()

        # Old deployer (30+ days ago)
        old = Deployer(
            wallet_address="OldDeployer1234567890123456789012345678901234",
            total_launches=10,
            graduated_launches=5,
            graduation_rate=0.5,
            avg_peak_mcap_sol=Decimal("500"),
            last_launch_at=datetime.utcnow() - timedelta(days=31),
        )
        old.graduated_launches = 5
        old_score = old.calculate_score()

        # Recent should score higher due to recency component (20 points)
        assert recent_score > old_score
        assert recent_score - old_score >= 15  # At least 15 points difference

    def test_consistency_rewards_multiple_graduates(self):
        """Deployers with 5+ graduates should get full consistency points."""
        # 1 graduate
        single = Deployer(
            wallet_address="SingleGrad123456789012345678901234567890123",
            total_launches=5,
            graduated_launches=1,
            graduation_rate=0.2,
            avg_peak_mcap_sol=Decimal("500"),
            last_launch_at=datetime.utcnow(),
        )
        single_score = single.calculate_score()

        # 5 graduates
        multi = Deployer(
            wallet_address="MultiGrad1234567890123456789012345678901234",
            total_launches=10,
            graduated_launches=5,
            graduation_rate=0.5,
            avg_peak_mcap_sol=Decimal("500"),
            last_launch_at=datetime.utcnow(),
        )
        multi_score = multi.calculate_score()

        # Multi should have higher score (both grad rate and consistency)
        assert multi_score > single_score


class TestDeployerThresholds:
    """Test deployer filtering thresholds."""

    def test_baseline_graduation_rate(self):
        """1.4% is the baseline graduation rate."""
        # Market average deployer (1.4% grad rate)
        baseline = Deployer(
            wallet_address="BaselineDeployer12345678901234567890123456",
            total_launches=100,
            graduated_launches=1,  # ~1%
            graduation_rate=0.014,
            avg_peak_mcap_sol=Decimal("50"),
        )
        baseline_score = baseline.calculate_score()

        # 3x baseline (4.2% - our target minimum)
        target = Deployer(
            wallet_address="TargetDeployer123456789012345678901234567890",
            total_launches=100,
            graduated_launches=4,  # ~4%
            graduation_rate=0.042,
            avg_peak_mcap_sol=Decimal("50"),
        )
        target_score = target.calculate_score()

        # Target should score meaningfully higher
        assert target_score > baseline_score

    def test_elite_deployer_identification(self, elite_deployer):
        """Elite deployers should score 80+."""
        elite_deployer.calculate_score()
        assert elite_deployer.deployer_score >= 80

    def test_poor_deployer_filtered(self, poor_deployer):
        """Poor deployers should score below threshold."""
        poor_deployer.calculate_score()
        assert poor_deployer.deployer_score < 60  # Below default threshold


class TestDeployerRecommendations:
    """Test recommendation generation."""

    @pytest.mark.asyncio
    async def test_elite_gets_strong_buy(self, mock_settings, db_session, elite_deployer):
        """Elite deployer should get STRONG BUY recommendation."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        intel = DeployerIntelligence(mock_settings, None)
        profile = await intel.analyze_deployer(
            db_session, elite_deployer.wallet_address
        )

        assert "STRONG BUY" in profile.recommendation

    @pytest.mark.asyncio
    async def test_poor_gets_skip(self, mock_settings, db_session, poor_deployer):
        """Poor deployer should get SKIP recommendation."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        intel = DeployerIntelligence(mock_settings, None)
        profile = await intel.analyze_deployer(
            db_session, poor_deployer.wallet_address
        )

        assert "SKIP" in profile.recommendation


class TestNewDeployerHandling:
    """Test handling of new/unknown deployers."""

    @pytest.mark.asyncio
    async def test_new_deployer_flagged(self, mock_settings, db_session):
        """Deployers with < 3 launches should be flagged as new."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        intel = DeployerIntelligence(mock_settings, None)

        # Create new deployer with 2 launches
        new_deployer = Deployer(
            wallet_address="NewDeployer123456789012345678901234567890123",
            total_launches=2,
            graduated_launches=1,
            graduation_rate=0.5,
        )
        db_session.add(new_deployer)
        await db_session.commit()

        profile = await intel.analyze_deployer(
            db_session, new_deployer.wallet_address
        )

        assert profile.is_new_deployer is True

    @pytest.mark.asyncio
    async def test_established_deployer_not_flagged(
        self, mock_settings, db_session, sample_deployer
    ):
        """Deployers with 3+ launches should not be flagged as new."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        intel = DeployerIntelligence(mock_settings, None)
        profile = await intel.analyze_deployer(
            db_session, sample_deployer.wallet_address
        )

        assert profile.is_new_deployer is False
