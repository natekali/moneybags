"""Tests for the circuit breaker system."""

from decimal import Decimal

import pytest

from bags_sniper.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerLevel,
    CircuitBreakerState,
)


class TestCircuitBreakerLevels:
    """Test circuit breaker level transitions."""

    @pytest.mark.asyncio
    async def test_initial_state_is_normal(self, mock_settings, db_session):
        """Circuit breaker should start at NORMAL level."""
        cb = CircuitBreaker(mock_settings)
        assert cb.state.level == CircuitBreakerLevel.NORMAL
        assert cb.can_open_position is True
        assert cb.position_size_multiplier == 1.0

    @pytest.mark.asyncio
    async def test_l1_triggered_at_3_percent(self, mock_settings, db_session):
        """L1 should trigger at 3% drawdown."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))

        # Simulate 3% loss
        await cb.record_trade_result(db_session, Decimal("-3"), is_win=False)

        assert cb.state.level == CircuitBreakerLevel.LEVEL_1
        assert cb.position_size_multiplier == 0.5
        assert cb.can_open_position is True

    @pytest.mark.asyncio
    async def test_l2_triggered_at_5_percent(self, mock_settings, db_session):
        """L2 should trigger at 5% drawdown."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))

        # Simulate 5% loss
        await cb.record_trade_result(db_session, Decimal("-5"), is_win=False)

        assert cb.state.level == CircuitBreakerLevel.LEVEL_2
        assert cb.position_size_multiplier == 0.25
        assert cb.can_open_position is True

    @pytest.mark.asyncio
    async def test_l3_pauses_new_entries(self, mock_settings, db_session):
        """L3 should pause new entries at 8% drawdown."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))

        # Simulate 8% loss
        await cb.record_trade_result(db_session, Decimal("-8"), is_win=False)

        assert cb.state.level == CircuitBreakerLevel.LEVEL_3
        assert cb.can_open_position is False
        assert cb.position_size_multiplier == 0.0

    @pytest.mark.asyncio
    async def test_l4_full_shutdown(self, mock_settings, db_session):
        """L4 should trigger full shutdown at 10% drawdown."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))

        # Simulate 10% loss
        await cb.record_trade_result(db_session, Decimal("-10"), is_win=False)

        assert cb.state.level == CircuitBreakerLevel.LEVEL_4
        assert cb.can_open_position is False


class TestCircuitBreakerReset:
    """Test circuit breaker reset conditions."""

    @pytest.mark.asyncio
    async def test_reset_after_3_consecutive_wins(self, mock_settings, db_session):
        """Should reset one level after 3 consecutive wins."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))

        # Trigger L1
        await cb.record_trade_result(db_session, Decimal("-3"), is_win=False)
        assert cb.state.level == CircuitBreakerLevel.LEVEL_1

        # 3 consecutive wins
        await cb.record_trade_result(db_session, Decimal("1"), is_win=True)
        await cb.record_trade_result(db_session, Decimal("1"), is_win=True)
        await cb.record_trade_result(db_session, Decimal("1"), is_win=True)

        assert cb.state.level == CircuitBreakerLevel.NORMAL

    @pytest.mark.asyncio
    async def test_win_streak_breaks_on_loss(self, mock_settings, db_session):
        """Win streak should reset on any loss."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))

        # Trigger L2 with larger loss (need to stay above L1 threshold during wins)
        await cb.record_trade_result(db_session, Decimal("-5"), is_win=False)
        assert cb.state.level == CircuitBreakerLevel.LEVEL_2

        # 2 small wins (not enough to drop below L2 threshold)
        await cb.record_trade_result(db_session, Decimal("0.5"), is_win=True)
        await cb.record_trade_result(db_session, Decimal("0.5"), is_win=True)
        assert cb.state.consecutive_wins_for_reset == 2

        # Loss resets win streak
        await cb.record_trade_result(db_session, Decimal("-0.5"), is_win=False)
        assert cb.state.consecutive_wins_for_reset == 0


class TestPositionSizeAdjustment:
    """Test position size calculations under circuit breaker."""

    @pytest.mark.asyncio
    async def test_normal_full_size(self, mock_settings, db_session):
        """Normal level should allow full position size."""
        cb = CircuitBreaker(mock_settings)
        base_size = Decimal("0.3")

        adjusted = cb.get_adjusted_position_size(base_size)
        assert adjusted == Decimal("0.3")

    @pytest.mark.asyncio
    async def test_l1_half_size(self, mock_settings, db_session):
        """L1 should reduce position to 50%."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))
        await cb.record_trade_result(db_session, Decimal("-3"), is_win=False)

        base_size = Decimal("0.3")
        adjusted = cb.get_adjusted_position_size(base_size)
        assert adjusted == Decimal("0.15")

    @pytest.mark.asyncio
    async def test_l2_quarter_size(self, mock_settings, db_session):
        """L2 should reduce position to 25%."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))
        await cb.record_trade_result(db_session, Decimal("-5"), is_win=False)

        base_size = Decimal("0.3")
        adjusted = cb.get_adjusted_position_size(base_size)
        assert adjusted == Decimal("0.075")


class TestDrawdownCalculation:
    """Test drawdown tracking and calculation."""

    @pytest.mark.asyncio
    async def test_drawdown_from_peak(self, mock_settings, db_session):
        """Drawdown should be calculated from peak balance."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))

        # Gain then loss
        await cb.record_trade_result(db_session, Decimal("10"), is_win=True)
        # Peak is now 110

        await cb.record_trade_result(db_session, Decimal("-5"), is_win=False)
        # Current is 105, drawdown from 110 = 4.5%

        assert cb._day_tracker.peak_balance_sol == Decimal("110")
        assert cb._day_tracker.current_balance_sol == Decimal("105")
        # Drawdown = (110 - 105) / 110 * 100 = 4.55%
        assert 4.5 <= cb.state.current_drawdown_percent <= 4.6

    @pytest.mark.asyncio
    async def test_status_report(self, mock_settings, db_session):
        """Status report should contain all required fields."""
        cb = CircuitBreaker(mock_settings)
        await cb.initialize(db_session, Decimal("100"))

        report = cb.get_status_report()

        assert "level" in report
        assert "message" in report
        assert "current_drawdown_percent" in report
        assert "position_size_multiplier" in report
        assert "new_entries_allowed" in report
        assert "day_stats" in report
        assert "thresholds" in report
