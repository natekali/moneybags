"""
Prometheus Metrics for Monitoring.

Exposes key trading metrics for monitoring via Prometheus/Grafana.
"""

from typing import Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from bags_sniper.core.config import Settings

logger = structlog.get_logger()


class TradingMetrics:
    """
    Prometheus metrics for the trading bot.

    Categories:
    - Trading: Entry/exit counts, P&L, position sizes
    - Risk: Circuit breaker levels, drawdowns
    - Performance: Latency, API response times
    - System: Health checks, queue sizes
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._initialized = False

        # =========================================================================
        # TRADING METRICS
        # =========================================================================

        # Trade counters
        self.trades_total = Counter(
            "bags_sniper_trades_total",
            "Total number of trades executed",
            ["direction", "reason"],  # direction: buy/sell, reason: entry/tp1/tp2/tp3/stop_loss/etc
        )

        self.trades_won = Counter(
            "bags_sniper_trades_won_total",
            "Total winning trades",
        )

        self.trades_lost = Counter(
            "bags_sniper_trades_lost_total",
            "Total losing trades",
        )

        # P&L tracking
        self.pnl_sol = Gauge(
            "bags_sniper_pnl_sol",
            "Cumulative P&L in SOL",
        )

        self.daily_pnl_sol = Gauge(
            "bags_sniper_daily_pnl_sol",
            "Daily P&L in SOL",
        )

        self.trade_pnl_sol = Histogram(
            "bags_sniper_trade_pnl_sol",
            "Distribution of trade P&L in SOL",
            buckets=[-1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        self.trade_pnl_percent = Histogram(
            "bags_sniper_trade_pnl_percent",
            "Distribution of trade P&L percentage",
            buckets=[-50, -30, -10, 0, 10, 50, 100, 200, 500, 1000],
        )

        # Position metrics
        self.position_size_sol = Histogram(
            "bags_sniper_position_size_sol",
            "Distribution of position sizes in SOL",
            buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
        )

        self.active_positions = Gauge(
            "bags_sniper_active_positions",
            "Number of currently active positions",
        )

        self.portfolio_value_sol = Gauge(
            "bags_sniper_portfolio_value_sol",
            "Total portfolio value in SOL",
        )

        # =========================================================================
        # DEPLOYER INTELLIGENCE METRICS
        # =========================================================================

        self.deployer_score_at_entry = Histogram(
            "bags_sniper_deployer_score_at_entry",
            "Distribution of deployer scores at trade entry",
            buckets=[40, 50, 60, 70, 80, 90, 100],
        )

        self.tokens_filtered = Counter(
            "bags_sniper_tokens_filtered_total",
            "Tokens filtered at each stage",
            ["stage", "reason"],  # stage: deployer/basic/liquidity/timing/security
        )

        self.tokens_passed = Counter(
            "bags_sniper_tokens_passed_total",
            "Tokens that passed all filters",
        )

        # =========================================================================
        # RISK METRICS
        # =========================================================================

        self.circuit_breaker_level = Gauge(
            "bags_sniper_circuit_breaker_level",
            "Current circuit breaker level (0=normal, 1-4=triggered)",
        )

        self.current_drawdown_percent = Gauge(
            "bags_sniper_current_drawdown_percent",
            "Current drawdown from peak",
        )

        self.consecutive_losses = Gauge(
            "bags_sniper_consecutive_losses",
            "Current consecutive losing streak",
        )

        self.circuit_breaker_triggers = Counter(
            "bags_sniper_circuit_breaker_triggers_total",
            "Circuit breaker trigger events",
            ["level", "reason"],  # reason: drawdown/consecutive_losses/manual
        )

        # =========================================================================
        # PERFORMANCE METRICS
        # =========================================================================

        self.filter_latency_ms = Histogram(
            "bags_sniper_filter_latency_ms",
            "Time to evaluate token through all filters",
            buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000],
        )

        self.api_latency_ms = Histogram(
            "bags_sniper_api_latency_ms",
            "API response latency",
            ["endpoint"],  # endpoint: bags/rpc/deepseek/jito
            buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
        )

        self.api_errors = Counter(
            "bags_sniper_api_errors_total",
            "API error count",
            ["endpoint", "error_type"],
        )

        # =========================================================================
        # SYSTEM METRICS
        # =========================================================================

        self.narrative_queue_size = Gauge(
            "bags_sniper_narrative_queue_size",
            "Number of tokens pending narrative analysis",
        )

        self.bot_status = Gauge(
            "bags_sniper_bot_status",
            "Bot operational status (1=running, 0=stopped)",
        )

        self.last_trade_timestamp = Gauge(
            "bags_sniper_last_trade_timestamp",
            "Unix timestamp of last trade",
        )

    async def start(self, port: int = 9090):
        """Start the Prometheus metrics HTTP server."""
        if self._initialized:
            return

        try:
            start_http_server(port)
            self._initialized = True
            self.bot_status.set(1)
            logger.info("metrics_server_started", port=port)
        except Exception as e:
            logger.error("metrics_server_start_failed", error=str(e))

    def stop(self):
        """Mark bot as stopped."""
        self.bot_status.set(0)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def record_entry(self, size_sol: float, deployer_score: float):
        """Record a trade entry."""
        self.trades_total.labels(direction="buy", reason="entry").inc()
        self.position_size_sol.observe(size_sol)
        self.deployer_score_at_entry.observe(deployer_score)
        self.last_trade_timestamp.set_to_current_time()

    def record_exit(
        self,
        reason: str,
        pnl_sol: float,
        pnl_percent: float,
        is_win: bool,
    ):
        """Record a trade exit."""
        self.trades_total.labels(direction="sell", reason=reason).inc()
        self.trade_pnl_sol.observe(pnl_sol)
        self.trade_pnl_percent.observe(pnl_percent)

        if is_win:
            self.trades_won.inc()
        else:
            self.trades_lost.inc()

        self.last_trade_timestamp.set_to_current_time()

    def update_positions(
        self,
        active_count: int,
        portfolio_value: float,
        daily_pnl: float,
        total_pnl: float,
    ):
        """Update position-related metrics."""
        self.active_positions.set(active_count)
        self.portfolio_value_sol.set(portfolio_value)
        self.daily_pnl_sol.set(daily_pnl)
        self.pnl_sol.set(total_pnl)

    def update_circuit_breaker(
        self,
        level: int,
        drawdown: float,
        consecutive_losses: int,
    ):
        """Update circuit breaker metrics."""
        self.circuit_breaker_level.set(level)
        self.current_drawdown_percent.set(drawdown)
        self.consecutive_losses.set(consecutive_losses)

    def record_filter_result(self, stage: str, reason: str):
        """Record a filter rejection."""
        self.tokens_filtered.labels(stage=stage, reason=reason).inc()

    def record_filter_pass(self, latency_ms: float):
        """Record a token passing all filters."""
        self.tokens_passed.inc()
        self.filter_latency_ms.observe(latency_ms)

    def record_api_call(self, endpoint: str, latency_ms: float, error: Optional[str] = None):
        """Record an API call."""
        self.api_latency_ms.labels(endpoint=endpoint).observe(latency_ms)
        if error:
            self.api_errors.labels(endpoint=endpoint, error_type=error).inc()


# Global metrics instance (singleton)
_metrics: Optional[TradingMetrics] = None


def get_metrics(settings: Optional[Settings] = None) -> TradingMetrics:
    """Get or create the global metrics instance."""
    global _metrics
    if _metrics is None:
        if settings is None:
            from bags_sniper.core.config import get_settings
            settings = get_settings()
        _metrics = TradingMetrics(settings)
    return _metrics
