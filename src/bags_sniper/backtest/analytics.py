"""
Performance Analytics for Backtest Results.
Calculates key metrics like Sharpe, Sortino, Win Rate, Max Drawdown.
"""

import math
from dataclasses import dataclass
from decimal import Decimal

from bags_sniper.backtest.engine import BacktestState


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics from backtest."""

    # Return metrics
    total_return_pct: float
    total_return_sol: Decimal
    annualized_return_pct: float

    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_sol: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float

    # P&L distribution
    avg_win_sol: Decimal
    avg_loss_sol: Decimal
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float  # Gross profit / Gross loss
    expectancy_sol: Decimal  # Avg profit per trade

    # Time metrics
    avg_hold_time_minutes: float
    avg_win_hold_minutes: float
    avg_loss_hold_minutes: float

    # Deployer analysis
    avg_deployer_score: float
    trades_by_score_bucket: dict[str, int]
    win_rate_by_score_bucket: dict[str, float]

    # Exit analysis
    exits_by_reason: dict[str, int]
    pnl_by_exit_reason: dict[str, float]

    # Peak analysis
    avg_peak_multiple: float
    trades_hit_2x: int
    trades_hit_5x: int
    trades_hit_10x: int


class PerformanceAnalytics:
    """
    Analyzes backtest results and generates performance metrics.
    """

    def __init__(self, state: BacktestState, config=None):
        self.state = state
        self.config = config
        self.trades = state.completed_trades

    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        if not self.trades:
            return self._empty_metrics()

        # Separate wins and losses
        wins = [t for t in self.trades if t.pnl_sol > 0]
        losses = [t for t in self.trades if t.pnl_sol <= 0]

        # Basic trade stats
        total_trades = len(self.trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # Return calculations
        initial_balance = self.config.initial_balance_sol if self.config else Decimal("10")
        total_return_sol = self.state.balance_sol - initial_balance
        total_return_pct = float(total_return_sol / initial_balance * 100)

        # Annualized return (assuming backtest period)
        if self.trades:
            start_time = min(t.entry_time for t in self.trades)
            end_time = max(t.exit_time for t in self.trades)
            days = max((end_time - start_time).days, 1)
            annualized = ((1 + total_return_pct / 100) ** (365 / days) - 1) * 100
        else:
            annualized = 0

        # P&L stats
        gross_profit = sum(t.pnl_sol for t in wins) if wins else Decimal("0")
        gross_loss = abs(sum(t.pnl_sol for t in losses)) if losses else Decimal("1")
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0

        avg_win_sol = gross_profit / len(wins) if wins else Decimal("0")
        avg_loss_sol = gross_loss / len(losses) if losses else Decimal("0")
        avg_win_pct = sum(t.pnl_percent for t in wins) / len(wins) if wins else 0
        avg_loss_pct = sum(t.pnl_percent for t in losses) / len(losses) if losses else 0

        expectancy = total_return_sol / total_trades if total_trades > 0 else Decimal("0")

        # Time stats
        avg_hold = sum(t.hold_time_minutes for t in self.trades) / total_trades
        avg_win_hold = sum(t.hold_time_minutes for t in wins) / len(wins) if wins else 0
        avg_loss_hold = sum(t.hold_time_minutes for t in losses) / len(losses) if losses else 0

        # Risk metrics
        max_dd_pct, max_dd_sol = self._calculate_max_drawdown()
        sharpe = self._calculate_sharpe_ratio()
        sortino = self._calculate_sortino_ratio()
        calmar = annualized / max_dd_pct if max_dd_pct > 0 else 0

        # Deployer analysis
        avg_score = sum(t.deployer_score for t in self.trades) / total_trades
        score_buckets = self._analyze_by_score_bucket()

        # Exit analysis
        exits = self._analyze_exits()

        # Peak analysis
        avg_peak = sum(t.peak_multiple for t in self.trades) / total_trades
        hit_2x = sum(1 for t in self.trades if t.peak_multiple >= 2)
        hit_5x = sum(1 for t in self.trades if t.peak_multiple >= 5)
        hit_10x = sum(1 for t in self.trades if t.peak_multiple >= 10)

        return PerformanceMetrics(
            total_return_pct=total_return_pct,
            total_return_sol=total_return_sol,
            annualized_return_pct=annualized,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_sol=max_dd_sol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate,
            avg_win_sol=avg_win_sol,
            avg_loss_sol=avg_loss_sol,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            profit_factor=profit_factor,
            expectancy_sol=expectancy,
            avg_hold_time_minutes=avg_hold,
            avg_win_hold_minutes=avg_win_hold,
            avg_loss_hold_minutes=avg_loss_hold,
            avg_deployer_score=avg_score,
            trades_by_score_bucket=score_buckets["trades"],
            win_rate_by_score_bucket=score_buckets["win_rates"],
            exits_by_reason=exits["counts"],
            pnl_by_exit_reason=exits["pnl"],
            avg_peak_multiple=avg_peak,
            trades_hit_2x=hit_2x,
            trades_hit_5x=hit_5x,
            trades_hit_10x=hit_10x,
        )

    def _calculate_max_drawdown(self) -> tuple[float, Decimal]:
        """Calculate maximum drawdown from trade sequence."""
        if not self.trades:
            return 0.0, Decimal("0")

        # Sort trades by exit time
        sorted_trades = sorted(self.trades, key=lambda t: t.exit_time)

        initial = self.config.initial_balance_sol if self.config else Decimal("10")
        balance = initial
        peak = initial
        max_dd_pct = 0.0
        max_dd_sol = Decimal("0")

        for trade in sorted_trades:
            balance += trade.pnl_sol

            if balance > peak:
                peak = balance

            dd_sol = peak - balance
            dd_pct = float(dd_sol / peak * 100) if peak > 0 else 0

            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_sol = dd_sol

        return max_dd_pct, max_dd_sol

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sharpe ratio.
        Sharpe = (Return - Risk-Free) / StdDev(Returns)
        """
        if len(self.trades) < 2:
            return 0.0

        returns = [float(t.pnl_percent) for t in self.trades]
        avg_return = sum(returns) / len(returns)
        std_dev = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns))

        if std_dev == 0:
            return 0.0

        # Annualize (assuming ~250 trading days)
        daily_rf = risk_free_rate / 365
        sharpe = (avg_return - daily_rf) / std_dev

        # Rough annualization
        return sharpe * math.sqrt(365)

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sortino ratio (only considers downside deviation).
        """
        if len(self.trades) < 2:
            return 0.0

        returns = [float(t.pnl_percent) for t in self.trades]
        avg_return = sum(returns) / len(returns)

        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return float("inf")  # No losses

        downside_dev = math.sqrt(
            sum(r ** 2 for r in negative_returns) / len(negative_returns)
        )

        if downside_dev == 0:
            return 0.0

        daily_rf = risk_free_rate / 365
        sortino = (avg_return - daily_rf) / downside_dev

        return sortino * math.sqrt(365)

    def _analyze_by_score_bucket(self) -> dict:
        """Analyze trade performance by deployer score bucket."""
        buckets = {
            "0-40": [],
            "40-60": [],
            "60-80": [],
            "80-100": [],
        }

        for trade in self.trades:
            score = trade.deployer_score
            if score < 40:
                buckets["0-40"].append(trade)
            elif score < 60:
                buckets["40-60"].append(trade)
            elif score < 80:
                buckets["60-80"].append(trade)
            else:
                buckets["80-100"].append(trade)

        trades_by_bucket = {k: len(v) for k, v in buckets.items()}
        win_rates = {}
        for bucket, trades in buckets.items():
            if trades:
                wins = sum(1 for t in trades if t.pnl_sol > 0)
                win_rates[bucket] = wins / len(trades) * 100
            else:
                win_rates[bucket] = 0.0

        return {"trades": trades_by_bucket, "win_rates": win_rates}

    def _analyze_exits(self) -> dict:
        """Analyze trade exits by reason."""
        exit_counts: dict[str, int] = {}
        exit_pnl: dict[str, float] = {}

        for trade in self.trades:
            reason = trade.exit_reason
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
            exit_pnl[reason] = exit_pnl.get(reason, 0.0) + float(trade.pnl_sol)

        return {"counts": exit_counts, "pnl": exit_pnl}

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades."""
        return PerformanceMetrics(
            total_return_pct=0.0,
            total_return_sol=Decimal("0"),
            annualized_return_pct=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_sol=Decimal("0"),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate_pct=0.0,
            avg_win_sol=Decimal("0"),
            avg_loss_sol=Decimal("0"),
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            profit_factor=0.0,
            expectancy_sol=Decimal("0"),
            avg_hold_time_minutes=0.0,
            avg_win_hold_minutes=0.0,
            avg_loss_hold_minutes=0.0,
            avg_deployer_score=0.0,
            trades_by_score_bucket={},
            win_rate_by_score_bucket={},
            exits_by_reason={},
            pnl_by_exit_reason={},
            avg_peak_multiple=0.0,
            trades_hit_2x=0,
            trades_hit_5x=0,
            trades_hit_10x=0,
        )

    def generate_report(self) -> str:
        """Generate human-readable performance report."""
        metrics = self.calculate_metrics()

        report = f"""
{'='*70}
                    BACKTEST PERFORMANCE REPORT
{'='*70}

SUMMARY
{'-'*70}
Total Return:        {metrics.total_return_pct:>10.2f}% ({metrics.total_return_sol:>10.4f} SOL)
Annualized Return:   {metrics.annualized_return_pct:>10.2f}%
Max Drawdown:        {metrics.max_drawdown_pct:>10.2f}% ({metrics.max_drawdown_sol:>10.4f} SOL)

RISK METRICS
{'-'*70}
Sharpe Ratio:        {metrics.sharpe_ratio:>10.2f}
Sortino Ratio:       {metrics.sortino_ratio:>10.2f}
Calmar Ratio:        {metrics.calmar_ratio:>10.2f}

TRADE STATISTICS
{'-'*70}
Total Trades:        {metrics.total_trades:>10}
Winning Trades:      {metrics.winning_trades:>10}
Losing Trades:       {metrics.losing_trades:>10}
Win Rate:            {metrics.win_rate_pct:>10.1f}%

P&L ANALYSIS
{'-'*70}
Avg Win:             {float(metrics.avg_win_sol):>10.4f} SOL ({metrics.avg_win_pct:>6.1f}%)
Avg Loss:            {float(metrics.avg_loss_sol):>10.4f} SOL ({metrics.avg_loss_pct:>6.1f}%)
Profit Factor:       {metrics.profit_factor:>10.2f}
Expectancy:          {float(metrics.expectancy_sol):>10.4f} SOL/trade

TIMING
{'-'*70}
Avg Hold Time:       {metrics.avg_hold_time_minutes:>10.1f} minutes
Avg Win Hold:        {metrics.avg_win_hold_minutes:>10.1f} minutes
Avg Loss Hold:       {metrics.avg_loss_hold_minutes:>10.1f} minutes

PEAK MULTIPLES
{'-'*70}
Avg Peak Multiple:   {metrics.avg_peak_multiple:>10.2f}x
Trades Hit 2x:       {metrics.trades_hit_2x:>10} ({metrics.trades_hit_2x/max(metrics.total_trades,1)*100:.1f}%)
Trades Hit 5x:       {metrics.trades_hit_5x:>10} ({metrics.trades_hit_5x/max(metrics.total_trades,1)*100:.1f}%)
Trades Hit 10x:      {metrics.trades_hit_10x:>10} ({metrics.trades_hit_10x/max(metrics.total_trades,1)*100:.1f}%)

DEPLOYER SCORE ANALYSIS
{'-'*70}
Avg Deployer Score:  {metrics.avg_deployer_score:>10.1f}

{'Score Bucket':<15} {'Trades':>10} {'Win Rate':>12}
{'-'*37}"""

        for bucket in ["0-40", "40-60", "60-80", "80-100"]:
            trades = metrics.trades_by_score_bucket.get(bucket, 0)
            win_rate = metrics.win_rate_by_score_bucket.get(bucket, 0)
            report += f"\n{bucket:<15} {trades:>10} {win_rate:>11.1f}%"

        report += f"""

EXIT REASON ANALYSIS
{'-'*70}
{'Exit Reason':<20} {'Count':>10} {'Total P&L':>15}
{'-'*45}"""

        for reason, count in sorted(
            metrics.exits_by_reason.items(), key=lambda x: x[1], reverse=True
        ):
            pnl = metrics.pnl_by_exit_reason.get(reason, 0)
            report += f"\n{reason:<20} {count:>10} {pnl:>14.4f} SOL"

        report += f"""

{'='*70}
"""
        return report

    def export_trades_csv(self, filepath: str):
        """Export all trades to CSV for external analysis."""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "mint_address",
                "symbol",
                "deployer_wallet",
                "deployer_score",
                "deployer_grad_rate",
                "entry_time",
                "entry_price",
                "entry_amount_sol",
                "exit_time",
                "exit_price",
                "exit_reason",
                "pnl_sol",
                "pnl_percent",
                "hold_time_minutes",
                "peak_multiple",
                "graduated",
            ])

            for t in self.trades:
                writer.writerow([
                    t.mint_address,
                    t.symbol,
                    t.deployer_wallet,
                    t.deployer_score,
                    t.deployer_grad_rate,
                    t.entry_time.isoformat(),
                    str(t.entry_price),
                    str(t.entry_amount_sol),
                    t.exit_time.isoformat(),
                    str(t.exit_price),
                    t.exit_reason,
                    str(t.pnl_sol),
                    t.pnl_percent,
                    t.hold_time_minutes,
                    t.peak_multiple,
                    t.graduated,
                ])
