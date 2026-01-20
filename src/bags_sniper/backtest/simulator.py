"""
Trade Simulator - High-level interface for running backtests.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import structlog

from bags_sniper.backtest.analytics import PerformanceAnalytics, PerformanceMetrics
from bags_sniper.backtest.engine import BacktestConfig, BacktestEngine, BacktestState
from bags_sniper.core.config import get_settings
from bags_sniper.models.database import init_database

logger = structlog.get_logger()


class TradeSimulator:
    """
    High-level simulator for running backtests with various configurations.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_settings().database_url.get_secret_value()
        self._session_factory = None

    async def initialize(self):
        """Initialize database connection."""
        self._session_factory = await init_database(self.database_url)

    async def run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        initial_balance: Decimal = Decimal("10.0"),
        position_size: Decimal = Decimal("0.3"),
        min_score: float = 60.0,
        min_grad_rate: float = 0.042,
    ) -> tuple[BacktestState, PerformanceMetrics]:
        """
        Run a single backtest with specified parameters.

        Returns:
            Tuple of (final_state, performance_metrics)
        """
        if not self._session_factory:
            await self.initialize()

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_balance_sol=initial_balance,
            position_size_sol=position_size,
            min_deployer_score=min_score,
            min_graduation_rate=min_grad_rate,
        )

        engine = BacktestEngine(config, self._session_factory)
        state = await engine.run()

        analytics = PerformanceAnalytics(state, config)
        metrics = analytics.calculate_metrics()

        return state, metrics

    async def parameter_sweep(
        self,
        start_date: datetime,
        end_date: datetime,
        score_thresholds: list[float] = None,
        grad_rate_thresholds: list[float] = None,
    ) -> list[dict]:
        """
        Run multiple backtests with different parameter combinations.

        Useful for finding optimal thresholds.
        """
        if score_thresholds is None:
            score_thresholds = [40.0, 50.0, 60.0, 70.0, 80.0]
        if grad_rate_thresholds is None:
            grad_rate_thresholds = [0.02, 0.03, 0.042, 0.05, 0.06]

        results = []

        for score in score_thresholds:
            for grad_rate in grad_rate_thresholds:
                logger.info(
                    "running_sweep_iteration",
                    min_score=score,
                    min_grad_rate=grad_rate,
                )

                state, metrics = await self.run_backtest(
                    start_date=start_date,
                    end_date=end_date,
                    min_score=score,
                    min_grad_rate=grad_rate,
                )

                results.append({
                    "min_score": score,
                    "min_grad_rate": grad_rate,
                    "total_trades": metrics.total_trades,
                    "win_rate": metrics.win_rate_pct,
                    "total_return_pct": metrics.total_return_pct,
                    "max_drawdown_pct": metrics.max_drawdown_pct,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "profit_factor": metrics.profit_factor,
                })

        return results

    async def walk_forward_analysis(
        self,
        start_date: datetime,
        end_date: datetime,
        train_days: int = 14,
        test_days: int = 7,
    ) -> list[dict]:
        """
        Walk-forward analysis to test parameter stability.

        1. Train on train_days
        2. Test on test_days
        3. Roll forward
        """
        results = []
        current_start = start_date

        while current_start + timedelta(days=train_days + test_days) <= end_date:
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)

            logger.info(
                "walk_forward_period",
                train_start=current_start.isoformat(),
                train_end=train_end.isoformat(),
                test_end=test_end.isoformat(),
            )

            # Train: find best parameters on training period
            train_results = await self.parameter_sweep(
                start_date=current_start,
                end_date=train_end,
                score_thresholds=[50.0, 60.0, 70.0],
                grad_rate_thresholds=[0.03, 0.042, 0.05],
            )

            # Find best by Sharpe ratio
            best_train = max(train_results, key=lambda x: x["sharpe_ratio"])

            # Test: apply best parameters to test period
            test_state, test_metrics = await self.run_backtest(
                start_date=train_end,
                end_date=test_end,
                min_score=best_train["min_score"],
                min_grad_rate=best_train["min_grad_rate"],
            )

            results.append({
                "train_start": current_start,
                "train_end": train_end,
                "test_end": test_end,
                "best_params": {
                    "min_score": best_train["min_score"],
                    "min_grad_rate": best_train["min_grad_rate"],
                },
                "train_sharpe": best_train["sharpe_ratio"],
                "test_sharpe": test_metrics.sharpe_ratio,
                "test_return": test_metrics.total_return_pct,
                "test_trades": test_metrics.total_trades,
            })

            # Roll forward
            current_start = train_end

        return results


async def run_quick_backtest(days_back: int = 7) -> str:
    """
    Run a quick backtest for the last N days.

    Returns performance report as string.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    simulator = TradeSimulator()
    await simulator.initialize()

    state, metrics = await simulator.run_backtest(
        start_date=start_date,
        end_date=end_date,
    )

    analytics = PerformanceAnalytics(state)
    return analytics.generate_report()
