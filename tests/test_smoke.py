"""
Smoke Tests - Verify core functionality works end-to-end.

These tests use in-memory SQLite and mocked external APIs
to verify the system integrates correctly without real credentials.

Run with: pytest tests/test_smoke.py -v
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bags_sniper.core.config import Settings
from bags_sniper.models.database import Base, Deployer, TokenLaunch


@pytest.fixture
def mock_settings():
    """Create mock settings that don't require real API keys."""
    return Settings(
        bags_api_key="test_key",
        bags_api_base_url="https://test.bags.fm/api/v1",
        helius_api_key="test_helius",
        deepseek_api_key="test_deepseek",
        telegram_bot_token="test_telegram",
        telegram_chat_id="123456789",
        wallet_private_key="test_wallet",
        database_url="sqlite+aiosqlite:///:memory:",
        dry_run=True,
        min_deployer_score=60.0,
        min_graduation_rate=0.042,
    )


@pytest.fixture
async def session_factory():
    """Create in-memory database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
async def populated_db(session_factory):
    """Populate database with test data."""
    async with session_factory() as session:
        # Create deployers with varying performance
        elite_deployer = Deployer(
            wallet_address="EliteDeployer12345678901234567890123456789012",
            total_launches=20,
            graduated_launches=8,
            graduation_rate=0.4,
            avg_peak_mcap_sol=Decimal("2000"),
            deployer_score=85.0,
            last_launch_at=datetime.utcnow() - timedelta(days=1),
        )

        good_deployer = Deployer(
            wallet_address="GoodDeployer123456789012345678901234567890123",
            total_launches=10,
            graduated_launches=2,
            graduation_rate=0.2,
            avg_peak_mcap_sol=Decimal("500"),
            deployer_score=65.0,
            last_launch_at=datetime.utcnow() - timedelta(days=3),
        )

        poor_deployer = Deployer(
            wallet_address="PoorDeployer123456789012345678901234567890123",
            total_launches=15,
            graduated_launches=0,
            graduation_rate=0.0,
            avg_peak_mcap_sol=Decimal("10"),
            deployer_score=10.0,
            last_launch_at=datetime.utcnow() - timedelta(days=7),
        )

        session.add_all([elite_deployer, good_deployer, poor_deployer])
        await session.flush()

        # Create token launches
        for i, deployer in enumerate([elite_deployer, good_deployer, poor_deployer]):
            for j in range(3):
                token = TokenLaunch(
                    mint_address=f"Token{deployer.id}_{j}_{'x'*32}"[:44],
                    deployer_id=deployer.id,
                    name=f"Test Token {deployer.id}-{j}",
                    symbol=f"TT{deployer.id}{j}",
                    launched_at=datetime.utcnow() - timedelta(days=j + 1),
                    graduated=(deployer == elite_deployer and j < 2),
                    initial_mcap_sol=Decimal("50"),
                    peak_mcap_sol=Decimal("200") if j == 0 else Decimal("30"),
                )
                session.add(token)

        await session.commit()

    return session_factory


class TestDeployerIntelligenceSmoke:
    """Smoke tests for deployer intelligence system."""

    @pytest.mark.asyncio
    async def test_score_calculation_works(self, mock_settings, populated_db):
        """Verify deployer scoring produces reasonable results."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        intel = DeployerIntelligence(mock_settings, None)

        async with populated_db() as session:
            # Check elite deployer
            profile = await intel.analyze_deployer(
                session, "EliteDeployer12345678901234567890123456789012"
            )

            assert profile.deployer_score > 60  # Score based on recalculated stats
            assert profile.graduation_rate > 0.3
            assert "BUY" in profile.recommendation  # May be STRONG BUY or BUY depending on score

            # Check poor deployer
            poor_profile = await intel.analyze_deployer(
                session, "PoorDeployer123456789012345678901234567890123"
            )

            assert poor_profile.deployer_score < 30
            assert "SKIP" in poor_profile.recommendation

    @pytest.mark.asyncio
    async def test_new_deployer_creation(self, mock_settings, session_factory):
        """Verify new deployers are created correctly."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence

        intel = DeployerIntelligence(mock_settings, None)

        async with session_factory() as session:
            deployer = await intel.get_or_create_deployer(
                session, "NewWallet123456789012345678901234567890123456"
            )

            assert deployer is not None
            assert deployer.wallet_address == "NewWallet123456789012345678901234567890123456"
            assert deployer.total_launches == 0


class TestFilterEngineSmoke:
    """Smoke tests for filter engine."""

    @pytest.mark.asyncio
    async def test_elite_deployer_passes_filters(self, mock_settings, populated_db):
        """Elite deployer's tokens should pass filters."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence
        from bags_sniper.core.filter_engine import FilterEngine, TokenCandidate

        intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()
        filter_engine = FilterEngine(mock_settings, intel, bags_mock)

        async with populated_db() as session:
            candidate = TokenCandidate(
                mint_address="NewToken1234567890123456789012345678901234567",
                deployer_wallet="EliteDeployer12345678901234567890123456789012",
                name="Elite Token",
                symbol="ELITE",
                mcap_sol=Decimal("50"),
                launched_at=datetime.utcnow() - timedelta(minutes=5),
            )

            result = await filter_engine.evaluate(session, candidate)

            assert result.passed_all_filters is True
            assert result.deployer_profile is not None
            assert result.deployer_profile.deployer_score > 60

    @pytest.mark.asyncio
    async def test_poor_deployer_fails_filters(self, mock_settings, populated_db):
        """Poor deployer's tokens should fail filters."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence
        from bags_sniper.core.filter_engine import FilterEngine, TokenCandidate

        intel = DeployerIntelligence(mock_settings, None)
        bags_mock = MagicMock()
        filter_engine = FilterEngine(mock_settings, intel, bags_mock)

        async with populated_db() as session:
            candidate = TokenCandidate(
                mint_address="BadToken12345678901234567890123456789012345678",
                deployer_wallet="PoorDeployer123456789012345678901234567890123",
                name="Bad Token",
                symbol="BAD",
                mcap_sol=Decimal("50"),
                launched_at=datetime.utcnow() - timedelta(minutes=5),
            )

            result = await filter_engine.evaluate(session, candidate)

            assert result.passed_all_filters is False
            assert "score" in result.rejection_reason.lower() or "below" in result.rejection_reason.lower()


class TestCircuitBreakerSmoke:
    """Smoke tests for circuit breaker."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_levels(self, mock_settings, session_factory):
        """Verify circuit breaker escalates through levels."""
        from bags_sniper.core.circuit_breaker import CircuitBreaker, CircuitBreakerLevel

        cb = CircuitBreaker(mock_settings)

        async with session_factory() as session:
            await cb.initialize(session, Decimal("100"))

            # Start at normal
            assert cb.state.level == CircuitBreakerLevel.NORMAL
            assert cb.can_open_position is True

            # 3% loss -> L1
            await cb.record_trade_result(session, Decimal("-3"), is_win=False)
            assert cb.state.level == CircuitBreakerLevel.LEVEL_1
            assert cb.position_size_multiplier == 0.5

            # Additional 3% loss (6% total) -> L2
            await cb.record_trade_result(session, Decimal("-3"), is_win=False)
            assert cb.state.level == CircuitBreakerLevel.LEVEL_2
            assert cb.position_size_multiplier == 0.25

    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_on_wins(self, mock_settings, session_factory):
        """Verify circuit breaker resets after consecutive wins."""
        from bags_sniper.core.circuit_breaker import CircuitBreaker, CircuitBreakerLevel

        cb = CircuitBreaker(mock_settings)

        async with session_factory() as session:
            await cb.initialize(session, Decimal("100"))

            # Trigger L1
            await cb.record_trade_result(session, Decimal("-3"), is_win=False)
            assert cb.state.level == CircuitBreakerLevel.LEVEL_1

            # 3 consecutive wins should reset
            await cb.record_trade_result(session, Decimal("1"), is_win=True)
            await cb.record_trade_result(session, Decimal("1"), is_win=True)
            await cb.record_trade_result(session, Decimal("1"), is_win=True)

            assert cb.state.level == CircuitBreakerLevel.NORMAL


class TestBacktestSmoke:
    """Smoke tests for backtest engine."""

    @pytest.mark.asyncio
    async def test_backtest_runs_without_error(self, mock_settings, populated_db):
        """Verify backtest engine runs to completion."""
        from bags_sniper.backtest.engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            initial_balance_sol=Decimal("10"),
            position_size_sol=Decimal("0.3"),
            min_deployer_score=60.0,
        )

        engine = BacktestEngine(config, populated_db)
        state = await engine.run()

        # Should complete without error
        assert state is not None
        assert state.tokens_seen >= 0
        assert state.balance_sol > 0

    @pytest.mark.asyncio
    async def test_analytics_calculates_metrics(self, mock_settings, populated_db):
        """Verify analytics produces valid metrics."""
        from bags_sniper.backtest.analytics import PerformanceAnalytics
        from bags_sniper.backtest.engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            initial_balance_sol=Decimal("10"),
        )

        engine = BacktestEngine(config, populated_db)
        state = await engine.run()

        analytics = PerformanceAnalytics(state, config)
        metrics = analytics.calculate_metrics()

        # Should produce valid metrics structure
        assert hasattr(metrics, "total_trades")
        assert hasattr(metrics, "win_rate_pct")
        assert hasattr(metrics, "sharpe_ratio")
        assert hasattr(metrics, "max_drawdown_pct")

    @pytest.mark.asyncio
    async def test_report_generation(self, mock_settings, populated_db):
        """Verify report generation works."""
        from bags_sniper.backtest.analytics import PerformanceAnalytics
        from bags_sniper.backtest.engine import BacktestConfig, BacktestEngine

        config = BacktestConfig(
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
        )

        engine = BacktestEngine(config, populated_db)
        state = await engine.run()

        analytics = PerformanceAnalytics(state, config)
        report = analytics.generate_report()

        assert isinstance(report, str)
        assert "PERFORMANCE REPORT" in report
        assert "Win Rate" in report


class TestEndToEndFlow:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_token_evaluation_flow(self, mock_settings, populated_db):
        """Test complete flow: token appears -> evaluated -> decision made."""
        from bags_sniper.core.deployer_intelligence import DeployerIntelligence
        from bags_sniper.core.filter_engine import FilterEngine, TokenCandidate
        from bags_sniper.core.circuit_breaker import CircuitBreaker

        async with populated_db() as session:
            # Initialize components
            intel = DeployerIntelligence(mock_settings, None)
            bags_mock = MagicMock()
            filter_engine = FilterEngine(mock_settings, intel, bags_mock)
            circuit_breaker = CircuitBreaker(mock_settings)
            await circuit_breaker.initialize(session, Decimal("10"))

            # Simulate new token from elite deployer
            candidate = TokenCandidate(
                mint_address="SimToken1234567890123456789012345678901234567",
                deployer_wallet="EliteDeployer12345678901234567890123456789012",
                name="Simulated Token",
                symbol="SIM",
                mcap_sol=Decimal("50"),
                launched_at=datetime.utcnow() - timedelta(minutes=5),
            )

            # Evaluate
            result = await filter_engine.evaluate(session, candidate)

            # Verify decision
            if result.passed_all_filters:
                # Should be able to trade
                assert circuit_breaker.can_open_position is True
                position_size = circuit_breaker.get_adjusted_position_size(Decimal("0.3"))
                assert position_size > 0
            else:
                # If filtered out, should have valid reason
                assert result.rejection_reason is not None
