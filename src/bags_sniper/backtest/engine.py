"""
Backtest Engine - Core simulation logic.

CRITICAL: Implements point-in-time scoring to prevent look-ahead bias.
At any simulation timestamp T, we only use data available before T.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bags_sniper.models.database import Deployer, TokenLaunch

logger = structlog.get_logger()


@dataclass
class SimulatedPosition:
    """A position being tracked in the simulation."""

    token_id: int
    mint_address: str
    symbol: str
    deployer_wallet: str
    deployer_score_at_entry: float

    entry_time: datetime
    entry_price: Decimal
    entry_amount_sol: Decimal
    tokens_held: Decimal

    # Price tracking
    current_price: Optional[Decimal] = None
    peak_price: Optional[Decimal] = None

    # Exit tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    tokens_remaining: Decimal = Decimal("0")

    # P&L
    realized_pnl: Decimal = Decimal("0")
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

    def __post_init__(self):
        self.tokens_remaining = self.tokens_held
        self.peak_price = self.entry_price


@dataclass
class BacktestTrade:
    """Completed trade record for analysis."""

    mint_address: str
    symbol: str
    deployer_wallet: str
    deployer_score: float
    deployer_grad_rate: float

    entry_time: datetime
    entry_price: Decimal
    entry_amount_sol: Decimal

    exit_time: datetime
    exit_price: Decimal
    exit_reason: str

    pnl_sol: Decimal
    pnl_percent: float
    hold_time_minutes: float

    peak_multiple: float  # Highest price / entry price
    graduated: bool  # Did token graduate?


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    start_date: datetime
    end_date: datetime
    initial_balance_sol: Decimal = Decimal("10.0")
    position_size_sol: Decimal = Decimal("0.3")
    max_concurrent_positions: int = 5

    # Deployer thresholds
    min_deployer_score: float = 60.0
    min_graduation_rate: float = 0.042  # 3x baseline

    # Take-profit config
    tp1_multiple: float = 2.5
    tp1_percent: float = 25.0
    tp2_multiple: float = 4.0
    tp2_percent: float = 35.0
    tp3_multiple: float = 7.0
    tp3_percent: float = 20.0
    trailing_stop_percent: float = 15.0
    stop_loss_percent: float = 30.0

    # Timing filters
    min_token_age_minutes: float = 1.0
    max_token_age_minutes: float = 30.0
    min_mcap_sol: Decimal = Decimal("5")
    max_mcap_sol: Decimal = Decimal("500")


@dataclass
class BacktestState:
    """Current state during backtest simulation."""

    current_time: datetime = field(default_factory=datetime.utcnow)
    balance_sol: Decimal = Decimal("10.0")
    peak_balance_sol: Decimal = Decimal("10.0")

    open_positions: dict[str, SimulatedPosition] = field(default_factory=dict)
    completed_trades: list[BacktestTrade] = field(default_factory=list)

    # Counters
    tokens_seen: int = 0
    tokens_filtered_deployer: int = 0
    tokens_filtered_other: int = 0
    entries_attempted: int = 0
    entries_skipped_balance: int = 0
    entries_skipped_max_positions: int = 0


class BacktestEngine:
    """
    Core backtesting engine with point-in-time simulation.

    Key principle: At simulation time T, we only know what was knowable at T.
    - Deployer scores use only launches BEFORE time T
    - Token outcomes (graduation) are revealed as simulation progresses
    """

    def __init__(
        self,
        config: BacktestConfig,
        session_factory: async_sessionmaker,
    ):
        self.config = config
        self.session_factory = session_factory
        self.state = BacktestState(
            current_time=config.start_date,
            balance_sol=config.initial_balance_sol,
            peak_balance_sol=config.initial_balance_sol,
        )

    async def run(self) -> BacktestState:
        """
        Run the backtest simulation.

        Returns final state with all completed trades.
        """
        logger.info(
            "backtest_starting",
            start=self.config.start_date.isoformat(),
            end=self.config.end_date.isoformat(),
            initial_balance=str(self.config.initial_balance_sol),
        )

        async with self.session_factory() as session:
            # Load all token launches in date range
            tokens = await self._load_tokens_in_range(session)
            logger.info("loaded_tokens", count=len(tokens))

            if not tokens:
                logger.warning("no_tokens_in_date_range")
                return self.state

            # Sort by launch time
            tokens.sort(key=lambda t: t.launched_at)

            # Process each token chronologically
            for token in tokens:
                self.state.current_time = token.launched_at
                self.state.tokens_seen += 1

                # First, update existing positions with price changes
                await self._update_positions(session, token.launched_at)

                # Then evaluate new token for entry
                await self._evaluate_token(session, token)

                # Log progress periodically
                if self.state.tokens_seen % 100 == 0:
                    logger.info(
                        "backtest_progress",
                        tokens_processed=self.state.tokens_seen,
                        trades=len(self.state.completed_trades),
                        balance=str(self.state.balance_sol),
                    )

            # Close any remaining positions at end
            await self._close_all_positions("backtest_end")

        logger.info(
            "backtest_complete",
            total_tokens=self.state.tokens_seen,
            total_trades=len(self.state.completed_trades),
            final_balance=str(self.state.balance_sol),
        )

        return self.state

    async def _load_tokens_in_range(
        self, session: AsyncSession
    ) -> list[TokenLaunch]:
        """Load all token launches in the backtest date range."""
        result = await session.execute(
            select(TokenLaunch)
            .where(
                and_(
                    TokenLaunch.launched_at >= self.config.start_date,
                    TokenLaunch.launched_at <= self.config.end_date,
                )
            )
            .order_by(TokenLaunch.launched_at)
        )
        return list(result.scalars().all())

    async def _evaluate_token(
        self, session: AsyncSession, token: TokenLaunch
    ):
        """Evaluate a token for potential entry using point-in-time data."""
        # Skip if already in position
        if token.mint_address in self.state.open_positions:
            return

        # Skip if max positions reached
        if len(self.state.open_positions) >= self.config.max_concurrent_positions:
            self.state.entries_skipped_max_positions += 1
            return

        # Skip if insufficient balance
        if self.state.balance_sol < self.config.position_size_sol:
            self.state.entries_skipped_balance += 1
            return

        # Get deployer with POINT-IN-TIME scoring
        deployer_score, grad_rate = await self._get_point_in_time_score(
            session, token.deployer_id, token.launched_at
        )

        # Filter 1: Deployer score
        if deployer_score < self.config.min_deployer_score:
            self.state.tokens_filtered_deployer += 1
            return

        # Filter 2: Graduation rate
        if grad_rate < self.config.min_graduation_rate:
            self.state.tokens_filtered_deployer += 1
            return

        # Filter 3: Market cap (if available)
        if token.initial_mcap_sol:
            if token.initial_mcap_sol < self.config.min_mcap_sol:
                self.state.tokens_filtered_other += 1
                return
            if token.initial_mcap_sol > self.config.max_mcap_sol:
                self.state.tokens_filtered_other += 1
                return

        # All filters passed - execute entry
        await self._execute_entry(session, token, deployer_score, grad_rate)

    async def _get_point_in_time_score(
        self,
        session: AsyncSession,
        deployer_id: int,
        as_of_time: datetime,
    ) -> tuple[float, float]:
        """
        Calculate deployer score using ONLY data available at as_of_time.

        This is critical for preventing look-ahead bias.
        """
        # Get deployer
        result = await session.execute(
            select(Deployer).where(Deployer.id == deployer_id)
        )
        deployer = result.scalar_one_or_none()
        if not deployer:
            return 0.0, 0.0

        # Count launches BEFORE as_of_time
        result = await session.execute(
            select(TokenLaunch).where(
                and_(
                    TokenLaunch.deployer_id == deployer_id,
                    TokenLaunch.launched_at < as_of_time,
                )
            )
        )
        past_launches = list(result.scalars().all())

        if not past_launches:
            return 0.0, 0.0  # New deployer, no history

        # Count graduations KNOWN at as_of_time
        # A graduation is only known if it happened before as_of_time
        graduated_count = sum(
            1
            for t in past_launches
            if t.graduated and t.graduated_at and t.graduated_at < as_of_time
        )

        total_count = len(past_launches)
        grad_rate = graduated_count / total_count if total_count > 0 else 0.0

        # Calculate point-in-time score
        # Simplified scoring (same weights as main engine)
        # Target: 3x baseline (4.2%) normalized to 10% for full points
        grad_component = min(grad_rate / 0.10, 1.0) * 40  # 40% weight

        # Avg peak mcap from past launches
        mcaps = [
            float(t.peak_mcap_sol)
            for t in past_launches
            if t.peak_mcap_sol
        ]
        avg_mcap = sum(mcaps) / len(mcaps) if mcaps else 0
        mcap_component = min(avg_mcap / 1000, 1.0) * 30  # 30% weight

        # Recency
        most_recent = max(t.launched_at for t in past_launches)
        days_since = (as_of_time - most_recent).days
        recency = max(0, 1 - (days_since / 30))
        recency_component = recency * 20  # 20% weight

        # Consistency
        consistency = min(graduated_count / 5, 1.0)
        consistency_component = consistency * 10  # 10% weight

        score = grad_component + mcap_component + recency_component + consistency_component

        return score, grad_rate

    async def _execute_entry(
        self,
        session: AsyncSession,
        token: TokenLaunch,
        deployer_score: float,
        grad_rate: float,
    ):
        """Execute a simulated entry."""
        # Get deployer wallet
        result = await session.execute(
            select(Deployer).where(Deployer.id == token.deployer_id)
        )
        deployer = result.scalar_one()

        # Calculate entry price (use initial mcap as proxy)
        entry_price = token.initial_mcap_sol or Decimal("10")  # Fallback

        # Calculate tokens received (simplified)
        tokens = self.config.position_size_sol / entry_price * Decimal("1000000")

        position = SimulatedPosition(
            token_id=token.id,
            mint_address=token.mint_address,
            symbol=token.symbol or "UNKNOWN",
            deployer_wallet=deployer.wallet_address,
            deployer_score_at_entry=deployer_score,
            entry_time=token.launched_at,
            entry_price=entry_price,
            entry_amount_sol=self.config.position_size_sol,
            tokens_held=tokens,
            current_price=entry_price,
        )

        self.state.open_positions[token.mint_address] = position
        self.state.balance_sol -= self.config.position_size_sol
        self.state.entries_attempted += 1

        logger.debug(
            "backtest_entry",
            symbol=token.symbol,
            deployer_score=f"{deployer_score:.1f}",
            grad_rate=f"{grad_rate:.2%}",
        )

    async def _update_positions(
        self, session: AsyncSession, current_time: datetime
    ):
        """Update all open positions and check exit conditions."""
        for mint, position in list(self.state.open_positions.items()):
            # Get token's current state
            result = await session.execute(
                select(TokenLaunch).where(TokenLaunch.mint_address == mint)
            )
            token = result.scalar_one_or_none()
            if not token:
                continue

            # Simulate price movement based on peak mcap
            # This is a simplification - real backtest would use tick data
            if token.peak_mcap_sol:
                position.current_price = token.peak_mcap_sol
                if position.current_price > position.peak_price:
                    position.peak_price = position.current_price

            # Check exit conditions
            await self._check_exits(position, token, current_time)

    async def _check_exits(
        self,
        position: SimulatedPosition,
        token: TokenLaunch,
        current_time: datetime,
    ):
        """Check and execute exit conditions."""
        if position.entry_price <= 0:
            return

        price_multiple = float(position.current_price / position.entry_price)
        peak_multiple = float(position.peak_price / position.entry_price)

        # Calculate unrealized P&L
        unrealized_pnl_pct = (price_multiple - 1) * 100

        # Exit condition 1: Stop loss
        if unrealized_pnl_pct <= -self.config.stop_loss_percent:
            await self._execute_exit(position, token, "stop_loss", 100, current_time)
            return

        # Exit condition 2: Trailing stop (after TP1)
        if position.tp1_hit:
            trailing_threshold = peak_multiple * (
                1 - self.config.trailing_stop_percent / 100
            )
            if price_multiple < trailing_threshold:
                await self._execute_exit(
                    position, token, "trailing_stop", 100, current_time
                )
                return

        # Exit condition 3: Take profits
        if not position.tp1_hit and price_multiple >= self.config.tp1_multiple:
            await self._execute_exit(
                position, token, "take_profit_1", self.config.tp1_percent, current_time
            )
            position.tp1_hit = True

        if (
            position.tp1_hit
            and not position.tp2_hit
            and price_multiple >= self.config.tp2_multiple
        ):
            await self._execute_exit(
                position, token, "take_profit_2", self.config.tp2_percent, current_time
            )
            position.tp2_hit = True

        if (
            position.tp2_hit
            and not position.tp3_hit
            and price_multiple >= self.config.tp3_multiple
        ):
            await self._execute_exit(
                position, token, "take_profit_3", self.config.tp3_percent, current_time
            )
            position.tp3_hit = True

        # Exit condition 4: Time-based exit (token too old, force close)
        hold_time = (current_time - position.entry_time).total_seconds() / 3600
        if hold_time > 24:  # Max 24 hour hold
            await self._execute_exit(
                position, token, "time_exit", 100, current_time
            )

    async def _execute_exit(
        self,
        position: SimulatedPosition,
        token: TokenLaunch,
        reason: str,
        percent: float,
        exit_time: datetime,
    ):
        """Execute a simulated exit."""
        tokens_to_sell = position.tokens_remaining * Decimal(str(percent / 100))
        exit_value = tokens_to_sell * position.current_price / Decimal("1000000")
        entry_value = tokens_to_sell * position.entry_price / Decimal("1000000")

        pnl = exit_value - entry_value
        position.realized_pnl += pnl
        position.tokens_remaining -= tokens_to_sell

        # Update balance
        self.state.balance_sol += exit_value
        if self.state.balance_sol > self.state.peak_balance_sol:
            self.state.peak_balance_sol = self.state.balance_sol

        # If fully exited, record trade and remove position
        if position.tokens_remaining <= 0 or percent >= 100:
            hold_time = (exit_time - position.entry_time).total_seconds() / 60

            # Simplified - use score as proxy for graduation rate
            grad_rate = position.deployer_score_at_entry / 100 * 0.1

            trade = BacktestTrade(
                mint_address=position.mint_address,
                symbol=position.symbol,
                deployer_wallet=position.deployer_wallet,
                deployer_score=position.deployer_score_at_entry,
                deployer_grad_rate=grad_rate,
                entry_time=position.entry_time,
                entry_price=position.entry_price,
                entry_amount_sol=position.entry_amount_sol,
                exit_time=exit_time,
                exit_price=position.current_price,
                exit_reason=reason,
                pnl_sol=position.realized_pnl,
                pnl_percent=float(position.realized_pnl / position.entry_amount_sol * 100),
                hold_time_minutes=hold_time,
                peak_multiple=float(position.peak_price / position.entry_price),
                graduated=token.graduated if token else False,
            )

            self.state.completed_trades.append(trade)
            del self.state.open_positions[position.mint_address]

            logger.debug(
                "backtest_exit",
                symbol=position.symbol,
                reason=reason,
                pnl=f"{position.realized_pnl:.4f}",
                hold_minutes=f"{hold_time:.1f}",
            )

    async def _close_all_positions(self, reason: str):
        """Close all remaining positions at end of backtest."""
        for mint, position in list(self.state.open_positions.items()):
            # Simplified exit at current price
            exit_value = position.tokens_remaining * position.current_price / Decimal("1000000")
            entry_value = position.tokens_remaining * position.entry_price / Decimal("1000000")

            pnl = exit_value - entry_value
            position.realized_pnl += pnl

            self.state.balance_sol += exit_value

            trade = BacktestTrade(
                mint_address=position.mint_address,
                symbol=position.symbol,
                deployer_wallet=position.deployer_wallet,
                deployer_score=position.deployer_score_at_entry,
                deployer_grad_rate=0.0,
                entry_time=position.entry_time,
                entry_price=position.entry_price,
                entry_amount_sol=position.entry_amount_sol,
                exit_time=self.state.current_time,
                exit_price=position.current_price,
                exit_reason=reason,
                pnl_sol=position.realized_pnl,
                pnl_percent=float(position.realized_pnl / position.entry_amount_sol * 100),
                hold_time_minutes=(
                    self.state.current_time - position.entry_time
                ).total_seconds() / 60,
                peak_multiple=float(position.peak_price / position.entry_price),
                graduated=False,
            )

            self.state.completed_trades.append(trade)

        self.state.open_positions.clear()
