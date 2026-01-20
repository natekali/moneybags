"""
Main trading engine that orchestrates all components.
Implements the core trading loop with deployer intelligence.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bags_sniper.core.circuit_breaker import CircuitBreaker, CircuitBreakerLevel
from bags_sniper.core.config import Settings
from bags_sniper.core.deployer_intelligence import DeployerIntelligence
from bags_sniper.core.filter_engine import FilterEngine, TokenCandidate
from bags_sniper.core.position_sizer import PositionSizer, format_size_decision
from bags_sniper.core.professional_trader import ProfessionalTrader, TX_FEE_SOL, format_trade_decision
from bags_sniper.core.quality_gate import QualityGate, QualityScore, format_quality_decision
from bags_sniper.models.database import Trade, TradeStatus, ExitReason
from bags_sniper.services.bags_api import BagsAPIClient
from bags_sniper.services.deepseek_ai import DeepseekAIService
from bags_sniper.services.solana_rpc import SolanaRPCClient
from bags_sniper.services.telegram_bot import TelegramBot, AlertPriority
from bags_sniper.services.token_discovery import TokenDiscoveryService

logger = structlog.get_logger()


@dataclass
class Position:
    """Active position being managed."""

    trade_id: int
    mint_address: str
    symbol: str
    entry_price_sol: Decimal
    entry_amount_sol: Decimal
    tokens_held: int
    entry_time: datetime
    deployer_score: float
    current_price_sol: Optional[Decimal] = None
    unrealized_pnl_percent: float = 0.0
    peak_price_sol: Optional[Decimal] = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False

    # PRD 5.2.6: Hard exit trigger tracking
    entry_volume_sol: Optional[Decimal] = None  # Volume at entry for comparison
    current_volume_sol: Optional[Decimal] = None
    peak_volume_sol: Optional[Decimal] = None
    entry_top_holder_percent: Optional[float] = None  # Top holder % at entry
    current_top_holder_percent: Optional[float] = None


class TradingEngine:
    """
    Main trading engine that coordinates all components.

    Flow:
    1. Monitor for new token launches
    2. Filter candidates using deployer intelligence
    3. Execute entries for qualifying tokens
    4. Manage positions with tiered take-profits
    5. Apply circuit breaker protections
    """

    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker,
        bags_client: BagsAPIClient,
        rpc_client: SolanaRPCClient,
        telegram: TelegramBot,
    ):
        self.settings = settings
        self.session_factory = session_factory
        self.bags = bags_client
        self.rpc = rpc_client
        self.telegram = telegram

        # Initialize components
        self.circuit_breaker = CircuitBreaker(settings)
        self.deployer_intel = DeployerIntelligence(settings, rpc_client)
        self.filter_engine = FilterEngine(settings, self.deployer_intel, bags_client)
        self.position_sizer = PositionSizer(settings, self.circuit_breaker)
        self.narrative_ai = DeepseekAIService(settings)
        self.token_discovery = TokenDiscoveryService(settings, rpc_client)
        self.professional_trader = ProfessionalTrader(settings)
        self.quality_gate = QualityGate(settings)  # Strict quality control

        # Price history for momentum analysis
        self._price_history: dict[str, list[Decimal]] = {}  # mint -> [prices]
        self._volume_history: dict[str, list[Decimal]] = {}  # mint -> [volumes]

        # State
        self._running = False
        self._paused = False
        self._positions: dict[str, Position] = {}  # mint -> Position
        self._max_positions = 5

        # Register telegram callbacks and set circuit breaker reference
        telegram.circuit_breaker = self.circuit_breaker
        telegram.register_callbacks(
            on_pause=self._handle_pause,
            on_resume=self._handle_resume,
            on_exit_all=self._handle_exit_all,
            get_positions=self._get_positions_for_telegram,
            get_balance=self._get_balance_for_telegram,
        )

        # Metrics tracking
        self._tokens_analyzed = 0
        self._tokens_filtered = 0

    async def start(self):
        """Start the trading engine."""
        logger.info("trading_engine_starting", dry_run=self.settings.dry_run)

        # Initialize circuit breaker with current balance
        balance = await self.bags.get_wallet_balance()
        if balance:
            async with self.session_factory() as session:
                await self.circuit_breaker.initialize(session, balance)

        # Start narrative AI service (async, non-blocking)
        await self.narrative_ai.start()

        self._running = True

        # Start background tasks
        await asyncio.gather(
            self._token_discovery_loop(),
            self._position_management_loop(),
            self._health_check_loop(),
        )

    async def stop(self):
        """Stop the trading engine gracefully."""
        logger.info("trading_engine_stopping")
        self._running = False

        # Stop narrative AI service
        await self.narrative_ai.stop()

        # Exit all positions if not dry run
        if not self.settings.dry_run:
            await self._exit_all_positions("Engine shutdown")

    async def _token_discovery_loop(self):
        """Main loop for discovering and evaluating new tokens."""
        logger.info("token_discovery_loop_started")

        while self._running:
            try:
                if self._paused or not self.circuit_breaker.can_open_position:
                    await asyncio.sleep(5)
                    continue

                # Discover new tokens via Helius RPC (monitors Bags program)
                discovered_tokens = await self.token_discovery.discover_new_tokens(limit=20)

                if not discovered_tokens:
                    # No new tokens found, wait before next poll
                    await asyncio.sleep(2)
                    continue

                logger.debug(
                    "processing_discovered_tokens",
                    count=len(discovered_tokens),
                )

                async with self.session_factory() as session:
                    for token in discovered_tokens:
                        # Track tokens analyzed
                        self._tokens_analyzed += 1

                        # Skip if already in position
                        if token.mint_address in self._positions:
                            continue

                        # Enrich token with metadata if missing (for AI analysis)
                        if not token.name or not token.symbol:
                            token = await self.token_discovery.enrich_token_metadata(token)

                        # Create candidate from discovered token
                        candidate = TokenCandidate(
                            mint_address=token.mint_address,
                            deployer_wallet=token.deployer_wallet,
                            name=token.name,
                            symbol=token.symbol,
                            mcap_sol=Decimal(str(token.mcap_sol))
                            if token.mcap_sol
                            else None,
                        )

                        # Evaluate through filters
                        candidate = await self.filter_engine.evaluate(session, candidate)

                        if candidate.passed_all_filters:
                            # Get AI analysis for entry decision (with timeout)
                            narrative = await self._get_ai_analysis_for_entry(candidate)
                            narrative_score = narrative.narrative_score if narrative else 0.0

                            # QUALITY GATE: Strict quality control
                            # This is the main filter - only trade high-quality opportunities
                            quality = self.quality_gate.evaluate_quality(
                                deployer_score=candidate.deployer_profile.deployer_score,
                                narrative_score=narrative_score,
                                deployer_launches=candidate.deployer_profile.total_launches,
                                is_new_deployer=candidate.deployer_profile.is_new_deployer,
                            )

                            logger.info(
                                "quality_gate_decision",
                                mint=candidate.mint_address[:8],
                                symbol=candidate.symbol or "UNKNOWN",
                                decision=format_quality_decision(quality),
                            )

                            if not quality.should_trade:
                                logger.info(
                                    "quality_gate_rejected",
                                    mint=candidate.mint_address[:8],
                                    symbol=candidate.symbol or "UNKNOWN",
                                    tier=quality.quality_tier,
                                    reason=quality.reasoning,
                                )
                                self._tokens_filtered += 1
                                self._log_to_telegram(
                                    "INFO",
                                    f"SKIP: {candidate.symbol or 'UNKNOWN'} - {quality.reasoning[:50]}"
                                )
                                continue

                            # Check daily limits before trading
                            # Get balance first for adaptive calculations
                            balance = await self.bags.get_wallet_balance() or Decimal("0")

                            # Use adaptive position sizing based on balance and quality
                            prelim_size = self.quality_gate._get_adaptive_position_size(
                                balance, quality.max_position_multiplier
                            )

                            can_trade, limit_reason = await self.quality_gate.check_daily_limits(
                                position_size_sol=prelim_size,
                                total_balance_sol=balance,
                            )

                            if not can_trade:
                                logger.warning(
                                    "daily_limit_reached",
                                    mint=candidate.mint_address[:8],
                                    reason=limit_reason,
                                )
                                self._log_to_telegram("WARNING", f"Daily limit: {limit_reason}")
                                continue

                            await self._execute_entry(session, candidate, quality)
                            self._log_to_telegram(
                                "INFO",
                                f"Entry: {candidate.symbol or 'UNKNOWN'} "
                                f"(D:{quality.deployer_score:.0f} N:{quality.narrative_score:.0f} "
                                f"Tier:{quality.quality_tier})"
                            )
                        else:
                            # Track filtered tokens with reason
                            self._tokens_filtered += 1
                            logger.info(
                                "token_filtered",
                                mint=candidate.mint_address[:8],
                                symbol=candidate.symbol or "UNKNOWN",
                                reason=candidate.rejection_reason or "Unknown",
                                deployer_score=candidate.deployer_profile.deployer_score if candidate.deployer_profile else 0,
                            )

                await asyncio.sleep(2)  # Poll every 2 seconds

            except Exception as e:
                logger.error("token_discovery_error", error=str(e))
                await asyncio.sleep(5)

    async def _position_management_loop(self):
        """Loop for managing active positions (take-profits, stops)."""
        logger.info("position_management_loop_started")

        while self._running:
            try:
                if not self._positions:
                    await asyncio.sleep(1)
                    continue

                for mint, position in list(self._positions.items()):
                    await self._update_position(position)
                    await self._check_exit_conditions(position)

                await asyncio.sleep(0.5)  # Check positions every 500ms

            except Exception as e:
                logger.error("position_management_error", error=str(e))
                await asyncio.sleep(1)

    async def _health_check_loop(self):
        """Periodic health checks and status updates."""
        import time

        while self._running:
            try:
                # Check API health
                bags_healthy = await self.bags.health_check()

                # Check RPC health and measure latency
                start_time = time.monotonic()
                rpc_health_result = await self.rpc.health_check()
                rpc_latency = (time.monotonic() - start_time) * 1000

                # Determine if RPC is healthy
                rpc_healthy = any(
                    ep.get('healthy', False)
                    for ep in rpc_health_result.values()
                ) if isinstance(rpc_health_result, dict) else bool(rpc_health_result)

                if not bags_healthy:
                    logger.warning("bags_api_unhealthy")
                    self._log_to_telegram("WARNING", "Bags API unhealthy")

                if not rpc_healthy:
                    logger.warning("rpc_api_unhealthy")
                    self._log_to_telegram("WARNING", "RPC unhealthy")

                # Update Telegram with metrics
                self.telegram.update_system_metrics(
                    rpc_latency_ms=rpc_latency,
                    tokens_analyzed=self._tokens_analyzed,
                    tokens_filtered=self._tokens_filtered,
                )

                # Update positions in Telegram
                self.telegram.update_positions(self._get_positions_for_telegram())

                # Log status periodically
                logger.info(
                    "health_check",
                    positions=len(self._positions),
                    circuit_level=self.circuit_breaker.state.level.name,
                    paused=self._paused,
                    rpc_latency_ms=round(rpc_latency, 1),
                )

                # Add to Telegram logs
                self._log_to_telegram(
                    "INFO",
                    f"Health check: {len(self._positions)} positions, "
                    f"CB={self.circuit_breaker.state.level.name}"
                )

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error("health_check_error", error=str(e))
                self._log_to_telegram("ERROR", f"Health check error: {str(e)[:50]}")
                await asyncio.sleep(60)

    async def _execute_entry(
        self,
        session: AsyncSession,
        candidate: TokenCandidate,
        quality: Optional[QualityScore] = None,
    ):
        """
        Execute entry for a qualifying token.

        Args:
            session: Database session
            candidate: Token candidate
            quality: Quality gate result (used to cap position size)
        """
        if not candidate.deployer_profile:
            return

        # Queue narrative analysis (async, non-blocking)
        await self.narrative_ai.analyze_token(
            name=candidate.name or "",
            symbol=candidate.symbol or "",
            description="",
            wait=False,  # Don't block on AI
        )

        # Get any cached narrative analysis
        narrative = await self.narrative_ai.get_cached_analysis(
            symbol=candidate.symbol or "",
            name=candidate.name or "",
        )

        # Calculate position size using advanced sizer
        size_result = self.position_sizer.calculate_size(
            deployer_profile=candidate.deployer_profile,
            narrative_analysis=narrative,
            current_positions=len(self._positions),
            max_positions=self._max_positions,
        )

        if size_result.recommended_size_sol <= 0:
            logger.info("position_size_zero", mint=candidate.mint_address[:8])
            return

        # Apply adaptive position sizing based on wallet balance
        # This ensures position sizes scale with available capital
        current_balance = await self.bags.get_wallet_balance() or Decimal("0")

        if current_balance > 0 and quality:
            # Use adaptive sizing from quality gate
            adaptive_size = self.quality_gate._get_adaptive_position_size(
                current_balance, quality.max_position_multiplier
            )
            # Take the smaller of adaptive size and position sizer recommendation
            quality_adjusted_size = min(adaptive_size, size_result.recommended_size_sol)
        elif quality and quality.max_position_multiplier < 1.0:
            quality_adjusted_size = size_result.recommended_size_sol * Decimal(str(quality.max_position_multiplier))
        else:
            quality_adjusted_size = size_result.recommended_size_sol

        # Ensure minimum viable position (after fees)
        min_viable_size = TX_FEE_SOL * Decimal("3")  # At least 3x the fee
        adjusted_size = max(min_viable_size, quality_adjusted_size)

        logger.info(
            "executing_entry",
            mint=candidate.mint_address[:8],
            symbol=candidate.symbol,
            base_size=str(size_result.recommended_size_sol),
            quality_adjusted=str(adjusted_size),
            quality_tier=quality.quality_tier if quality else "N/A",
            deployer_score=candidate.deployer_profile.deployer_score,
            narrative_score=narrative.narrative_score if narrative else None,
            dry_run=self.settings.dry_run,
        )

        if self.settings.dry_run:
            # Simulate entry
            await self._record_simulated_entry(session, candidate, adjusted_size)
            # Record to quality gate
            await self.quality_gate.record_trade_entry(adjusted_size)
        else:
            # Real entry
            response = await self.bags.buy_token(
                mint_address=candidate.mint_address,
                amount_sol=adjusted_size,
            )

            if response.success:
                await self._record_entry(session, candidate, adjusted_size, response.data)
                # Record to quality gate
                await self.quality_gate.record_trade_entry(adjusted_size)
            else:
                logger.error(
                    "entry_failed",
                    mint=candidate.mint_address[:8],
                    error=response.error,
                )

    async def _record_simulated_entry(
        self,
        session: AsyncSession,
        candidate: TokenCandidate,
        amount_sol: Decimal,
    ):
        """Record simulated entry for dry run."""
        # Use current price or estimate
        price = candidate.price_sol or Decimal("0.000001")
        tokens = int(amount_sol / price) if price > 0 else 0

        # Create position tracker
        self._positions[candidate.mint_address] = Position(
            trade_id=0,  # No DB record in dry run
            mint_address=candidate.mint_address,
            symbol=candidate.symbol or "UNKNOWN",
            entry_price_sol=price,
            entry_amount_sol=amount_sol,
            tokens_held=tokens,
            entry_time=datetime.utcnow(),
            deployer_score=candidate.deployer_profile.deployer_score,
            current_price_sol=price,
            peak_price_sol=price,
        )

        await self.telegram.send_trade_entry(
            token_symbol=candidate.symbol or "UNKNOWN",
            mint_address=candidate.mint_address,
            amount_sol=amount_sol,
            price_sol=price,
            deployer_score=candidate.deployer_profile.deployer_score,
        )

    async def _record_entry(
        self,
        session: AsyncSession,
        candidate: TokenCandidate,
        amount_sol: Decimal,
        response_data: dict,
    ):
        """Record real entry to database."""
        # Extract details from response
        price = Decimal(str(response_data.get("price", 0)))
        tokens = int(response_data.get("tokens_received", 0))
        tx_sig = response_data.get("signature", "")

        # Record to database
        from bags_sniper.models.database import TokenLaunch
        from sqlalchemy import select

        result = await session.execute(
            select(TokenLaunch).where(TokenLaunch.mint_address == candidate.mint_address)
        )
        token_launch = result.scalar_one_or_none()

        if not token_launch:
            # Create token launch record
            token_launch = await self.deployer_intel.record_token_launch(
                session,
                candidate.deployer_wallet,
                candidate.mint_address,
                candidate.name,
                candidate.symbol,
            )

        trade = Trade(
            token_launch_id=token_launch.id,
            entry_price_sol=price,
            entry_amount_sol=amount_sol,
            entry_tokens=tokens,
            entry_tx_signature=tx_sig,
            entry_timestamp=datetime.utcnow(),
            tokens_remaining=tokens,
            status=TradeStatus.EXECUTED,
            deployer_score_at_entry=candidate.deployer_profile.deployer_score,
        )
        session.add(trade)
        await session.commit()

        # Create position tracker
        self._positions[candidate.mint_address] = Position(
            trade_id=trade.id,
            mint_address=candidate.mint_address,
            symbol=candidate.symbol or "UNKNOWN",
            entry_price_sol=price,
            entry_amount_sol=amount_sol,
            tokens_held=tokens,
            entry_time=datetime.utcnow(),
            deployer_score=candidate.deployer_profile.deployer_score,
            current_price_sol=price,
            peak_price_sol=price,
        )

        await self.telegram.send_trade_entry(
            token_symbol=candidate.symbol or "UNKNOWN",
            mint_address=candidate.mint_address,
            amount_sol=amount_sol,
            price_sol=price,
            deployer_score=candidate.deployer_profile.deployer_score,
        )

    async def _update_position(self, position: Position):
        """Update position with current price and track history for momentum."""
        price = await self.bags.get_token_price(position.mint_address)
        if price:
            position.current_price_sol = price

            # Update peak
            if not position.peak_price_sol or price > position.peak_price_sol:
                position.peak_price_sol = price

            # Calculate unrealized P&L
            if position.entry_price_sol > 0:
                position.unrealized_pnl_percent = float(
                    (price - position.entry_price_sol) / position.entry_price_sol * 100
                )

            # Track price history for momentum analysis (keep last 20 samples)
            mint = position.mint_address
            if mint not in self._price_history:
                self._price_history[mint] = []
            self._price_history[mint].append(price)
            if len(self._price_history[mint]) > 20:
                self._price_history[mint] = self._price_history[mint][-20:]

        # Also track volume if available
        try:
            response = await self.bags.get_token_info(position.mint_address)
            if response.success and response.data:
                volume = response.data.get("volume_24h_sol") or response.data.get("volume")
                if volume:
                    mint = position.mint_address
                    if mint not in self._volume_history:
                        self._volume_history[mint] = []
                    self._volume_history[mint].append(Decimal(str(volume)))
                    if len(self._volume_history[mint]) > 20:
                        self._volume_history[mint] = self._volume_history[mint][-20:]
        except Exception:
            pass  # Volume tracking is optional

    async def _check_exit_conditions(self, position: Position):
        """
        Check and execute exit conditions using professional trader logic.

        Uses momentum analysis, fee awareness, and smart timing to maximize profits.
        """
        if not position.current_price_sol or not position.entry_price_sol:
            return

        price_multiple = float(position.current_price_sol / position.entry_price_sol)

        # Exit precedence: Hard Exit > Stop Loss > Trailing > Professional Trader Logic > Take-Profit

        # 1. Circuit breaker hard exit (L4)
        if self.circuit_breaker.state.level >= CircuitBreakerLevel.LEVEL_4:
            await self._execute_exit(position, ExitReason.CIRCUIT_BREAKER, percent=100)
            return

        # 2. PRD 5.2.6: Volume drop 60% hard exit
        if await self._check_volume_drop_trigger(position):
            logger.warning(
                "hard_exit_volume_drop",
                mint=position.mint_address[:8],
                symbol=position.symbol,
            )
            await self._execute_exit(position, ExitReason.HARD_EXIT, percent=100)
            return

        # 3. PRD 5.2.6: Top holder sells >8% hard exit
        if await self._check_top_holder_dump_trigger(position):
            logger.warning(
                "hard_exit_top_holder_dump",
                mint=position.mint_address[:8],
                symbol=position.symbol,
            )
            await self._execute_exit(position, ExitReason.HARD_EXIT, percent=100)
            return

        # 4. Use professional trader for smart exit decisions
        mint = position.mint_address
        price_history = list(self._price_history.get(mint, []))
        volume_history = list(self._volume_history.get(mint, []))

        analysis = self.professional_trader.analyze_position(
            mint_address=mint,
            symbol=position.symbol,
            entry_price_sol=position.entry_price_sol,
            current_price_sol=position.current_price_sol,
            entry_time=position.entry_time,
            tokens_held=position.tokens_held,
            entry_amount_sol=position.entry_amount_sol,
            peak_price_sol=position.peak_price_sol,
            price_history=price_history,
            volume_history=volume_history,
        )

        decision = self.professional_trader.decide_sell_action(
            analysis=analysis,
            tp1_hit=position.tp1_hit,
            tp2_hit=position.tp2_hit,
        )

        # Log trading decision
        if decision.action != "hold":
            logger.info(
                "professional_trader_decision",
                mint=mint[:8],
                symbol=position.symbol,
                action=decision.action,
                confidence=f"{decision.confidence:.0%}",
                reasoning=decision.reasoning[:100],
                pnl_estimate=str(decision.estimated_pnl_sol),
            )

        # Execute based on decision
        if decision.action == "sell":
            if "Stop loss" in decision.reasoning:
                await self._execute_exit(position, ExitReason.STOP_LOSS, percent=100)
            elif "Trailing stop" in decision.reasoning:
                await self._execute_exit(position, ExitReason.TRAILING_STOP, percent=100)
            elif "momentum" in decision.reasoning.lower():
                await self._execute_exit(position, ExitReason.HARD_EXIT, percent=100)
            else:
                await self._execute_exit(position, ExitReason.TAKE_PROFIT_T1, percent=100)
            return

        elif decision.action == "sell_partial":
            if "TP1" in decision.reasoning:
                await self._execute_exit(
                    position,
                    ExitReason.TAKE_PROFIT_T1,
                    percent=decision.suggested_percent,
                )
                position.tp1_hit = True
            elif "TP2" in decision.reasoning:
                await self._execute_exit(
                    position,
                    ExitReason.TAKE_PROFIT_T2,
                    percent=decision.suggested_percent,
                )
                position.tp2_hit = True
            elif "TP3" in decision.reasoning:
                await self._execute_exit(
                    position,
                    ExitReason.TAKE_PROFIT_T3,
                    percent=decision.suggested_percent,
                )
                position.tp3_hit = True
            return

        # Fallback to classic take-profit tiers if professional trader says hold
        # but we've hit classic TP levels
        if not position.tp1_hit and price_multiple >= self.settings.tp_tier1_multiple:
            await self._execute_exit(
                position,
                ExitReason.TAKE_PROFIT_T1,
                percent=int(self.settings.tp_tier1_percent),
            )
            position.tp1_hit = True

        elif (
            position.tp1_hit
            and not position.tp2_hit
            and price_multiple >= self.settings.tp_tier2_multiple
        ):
            await self._execute_exit(
                position,
                ExitReason.TAKE_PROFIT_T2,
                percent=int(self.settings.tp_tier2_percent),
            )
            position.tp2_hit = True

        elif (
            position.tp2_hit
            and not position.tp3_hit
            and price_multiple >= self.settings.tp_tier3_multiple
        ):
            await self._execute_exit(
                position,
                ExitReason.TAKE_PROFIT_T3,
                percent=int(self.settings.tp_tier3_percent),
            )
            position.tp3_hit = True

    async def _check_volume_drop_trigger(self, position: Position) -> bool:
        """
        PRD 5.2.6: Check if volume has dropped 60% from peak.
        This indicates loss of interest and potential dump incoming.
        """
        try:
            # Fetch current volume
            response = await self.bags.get_token_info(position.mint_address)
            if not response.success or not response.data:
                return False

            current_volume = response.data.get("volume_24h_sol")
            if current_volume is None:
                return False

            current_volume = Decimal(str(current_volume))
            position.current_volume_sol = current_volume

            # Initialize peak volume if not set
            if position.peak_volume_sol is None:
                position.peak_volume_sol = current_volume
                return False

            # Update peak volume
            if current_volume > position.peak_volume_sol:
                position.peak_volume_sol = current_volume
                return False

            # Check for 60% drop from peak
            if position.peak_volume_sol > 0:
                volume_drop_percent = float(
                    (position.peak_volume_sol - current_volume) / position.peak_volume_sol * 100
                )
                if volume_drop_percent >= 60:
                    logger.warning(
                        "volume_drop_detected",
                        mint=position.mint_address[:8],
                        peak_volume=str(position.peak_volume_sol),
                        current_volume=str(current_volume),
                        drop_percent=f"{volume_drop_percent:.1f}%",
                    )
                    return True

            return False

        except Exception as e:
            logger.error("volume_check_error", error=str(e))
            return False

    async def _check_top_holder_dump_trigger(self, position: Position) -> bool:
        """
        PRD 5.2.6: Check if top holder has sold >8% of their holdings.
        Large holder dumps often precede major price drops.
        """
        try:
            # Fetch current holder data
            response = await self.bags.get_token_info(position.mint_address)
            if not response.success or not response.data:
                return False

            top_holders = response.data.get("top_holders", [])
            if not top_holders:
                return False

            current_top_percent = top_holders[0].get("percent", 0)
            position.current_top_holder_percent = current_top_percent

            # Initialize entry top holder percent if not set
            if position.entry_top_holder_percent is None:
                position.entry_top_holder_percent = current_top_percent
                return False

            # Check if top holder has sold >8%
            if position.entry_top_holder_percent > 0:
                percent_sold = position.entry_top_holder_percent - current_top_percent

                if percent_sold >= 8:
                    logger.warning(
                        "top_holder_dump_detected",
                        mint=position.mint_address[:8],
                        entry_percent=f"{position.entry_top_holder_percent:.1f}%",
                        current_percent=f"{current_top_percent:.1f}%",
                        sold_percent=f"{percent_sold:.1f}%",
                    )
                    return True

            return False

        except Exception as e:
            logger.error("holder_check_error", error=str(e))
            return False

    async def _execute_exit(
        self,
        position: Position,
        reason: ExitReason,
        percent: int,
    ):
        """Execute an exit for a position."""
        logger.info(
            "executing_exit",
            mint=position.mint_address[:8],
            reason=reason.value,
            percent=percent,
            dry_run=self.settings.dry_run,
        )

        if self.settings.dry_run:
            await self._record_simulated_exit(position, reason, percent)
        else:
            response = await self.bags.sell_token(
                mint_address=position.mint_address,
                percent=percent,
            )

            if response.success:
                await self._record_exit(position, reason, percent, response.data)
            else:
                logger.error(
                    "exit_failed",
                    mint=position.mint_address[:8],
                    error=response.error,
                )

    async def _record_simulated_exit(
        self,
        position: Position,
        reason: ExitReason,
        percent: int,
    ):
        """Record simulated exit for dry run."""
        tokens_to_sell = int(position.tokens_held * percent / 100)
        exit_value = Decimal(str(tokens_to_sell)) * (
            position.current_price_sol or position.entry_price_sol
        )
        entry_value = Decimal(str(tokens_to_sell)) * position.entry_price_sol

        pnl_sol = exit_value - entry_value
        pnl_percent = float(pnl_sol / entry_value * 100) if entry_value > 0 else 0

        # Update position
        position.tokens_held -= tokens_to_sell

        # Record circuit breaker
        async with self.session_factory() as session:
            await self.circuit_breaker.record_trade_result(
                session, pnl_sol, pnl_sol > 0
            )

        # Remove position if fully exited
        if position.tokens_held <= 0 or percent >= 100:
            del self._positions[position.mint_address]

        hold_time = (datetime.utcnow() - position.entry_time).total_seconds() / 60

        await self.telegram.send_trade_exit(
            token_symbol=position.symbol,
            exit_reason=reason.value,
            pnl_sol=pnl_sol,
            pnl_percent=pnl_percent,
            hold_time_minutes=hold_time,
        )

    async def _record_exit(
        self,
        position: Position,
        reason: ExitReason,
        percent: int,
        response_data: dict,
    ):
        """Record real exit to database."""
        exit_sol = Decimal(str(response_data.get("sol_received", 0)))
        tokens_sold = int(response_data.get("tokens_sold", 0))

        # Calculate P&L for this exit
        entry_value = Decimal(str(tokens_sold)) * position.entry_price_sol
        pnl_sol = exit_sol - entry_value
        pnl_percent = float(pnl_sol / entry_value * 100) if entry_value > 0 else 0

        # Update database
        async with self.session_factory() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(Trade).where(Trade.id == position.trade_id)
            )
            trade = result.scalar_one()

            trade.tokens_remaining -= tokens_sold
            trade.total_exit_sol += exit_sol
            trade.realized_pnl_sol += pnl_sol

            # Update TP flags
            if reason == ExitReason.TAKE_PROFIT_T1:
                trade.tp1_hit = True
                trade.tp1_exit_sol = exit_sol
            elif reason == ExitReason.TAKE_PROFIT_T2:
                trade.tp2_hit = True
                trade.tp2_exit_sol = exit_sol
            elif reason == ExitReason.TAKE_PROFIT_T3:
                trade.tp3_hit = True
                trade.tp3_exit_sol = exit_sol

            if trade.tokens_remaining <= 0 or percent >= 100:
                trade.status = TradeStatus.CLOSED
                trade.final_exit_reason = reason
                trade.final_exit_timestamp = datetime.utcnow()
                trade.calculate_pnl()

            await session.commit()

            # Record circuit breaker
            await self.circuit_breaker.record_trade_result(session, pnl_sol, pnl_sol > 0)

        # Update position
        position.tokens_held -= tokens_sold
        if position.tokens_held <= 0 or percent >= 100:
            del self._positions[position.mint_address]

        hold_time = (datetime.utcnow() - position.entry_time).total_seconds() / 60

        await self.telegram.send_trade_exit(
            token_symbol=position.symbol,
            exit_reason=reason.value,
            pnl_sol=pnl_sol,
            pnl_percent=pnl_percent,
            hold_time_minutes=hold_time,
        )

    async def _exit_all_positions(self, reason: str):
        """Exit all positions (emergency or shutdown)."""
        logger.warning("exiting_all_positions", reason=reason)

        for mint, position in list(self._positions.items()):
            await self._execute_exit(position, ExitReason.HARD_EXIT, percent=100)

        await self.telegram.send_alert(
            f"All positions exited: {reason}",
            priority=AlertPriority.CRITICAL,
        )

    # Telegram integration methods
    def _get_positions_for_telegram(self) -> dict[str, dict]:
        """Get positions data formatted for Telegram display."""
        return {
            mint: {
                'symbol': pos.symbol,
                'entry_price': pos.entry_price_sol,
                'current_price': pos.current_price_sol or pos.entry_price_sol,
                'amount_sol': pos.entry_amount_sol,
                'unrealized_pnl_percent': pos.unrealized_pnl_percent,
                'deployer_score': pos.deployer_score,
                'entry_time': pos.entry_time,
            }
            for mint, pos in self._positions.items()
        }

    async def _get_balance_for_telegram(self) -> Optional[Decimal]:
        """Get wallet balance for Telegram display."""
        return await self.bags.get_wallet_balance()

    def _update_telegram_metrics(self):
        """Update Telegram bot with current metrics."""
        # Update positions
        self.telegram.update_positions(self._get_positions_for_telegram())

        # Update system metrics
        self.telegram.update_system_metrics(
            tokens_analyzed=self._tokens_analyzed,
            tokens_filtered=self._tokens_filtered,
        )

    def _log_to_telegram(self, level: str, event: str, details: str = ""):
        """Send a log entry to Telegram buffer."""
        self.telegram.add_log(level, event, details)

    async def _get_ai_analysis_for_entry(
        self, candidate: TokenCandidate
    ) -> Optional["NarrativeAnalysis"]:
        """
        Get AI narrative analysis for entry decision.

        PRD 5.2.4: AI should score tokens 0-100 for narrative strength.
        For new/unknown deployers, AI analysis helps filter out poor narratives.

        Returns:
            NarrativeAnalysis if available within timeout, None otherwise
        """
        from bags_sniper.services.deepseek_ai import NarrativeAnalysis

        if not self.narrative_ai:
            return None

        try:
            # Strategy: Use heuristic immediately, AI is too slow (8-19s first token latency)
            # Per benchmarks, DeepSeek has 8-19 second first token latency
            # For memecoin sniping, we need instant decisions

            name = candidate.name or "Unknown"
            symbol = candidate.symbol or "UNKNOWN"

            # First check if we have cached AI analysis (from previous background run)
            cached = await self.narrative_ai.get_cached_analysis(symbol, name)
            if cached:
                logger.info(
                    "using_cached_ai_analysis",
                    mint=candidate.mint_address[:8],
                    symbol=symbol,
                    score=cached.narrative_score,
                )
                return cached

            # Use fast heuristic for immediate decision
            analysis = self.narrative_ai._heuristic_analysis(name, symbol, "")

            logger.info(
                "ai_analysis_complete",
                mint=candidate.mint_address[:8],
                symbol=symbol,
                narrative_score=analysis.narrative_score,
                category=analysis.category.value,
                confidence=analysis.confidence,
                reasoning=analysis.reasoning[:50] if analysis.reasoning else "",
            )

            # Queue AI analysis in background for cache warming (future calls)
            await self.narrative_ai.analyze_token(
                name=name,
                symbol=symbol,
                description="",
                wait=False,  # Non-blocking background queue
            )

            return analysis

        except Exception as e:
            logger.error("ai_analysis_error", error=str(e))
            # Return heuristic analysis as final fallback
            return self.narrative_ai._heuristic_analysis(
                candidate.name or "Unknown",
                candidate.symbol or "UNKNOWN",
                "",
            )

    # Telegram command handlers
    async def _handle_pause(self):
        """Handle pause command from Telegram."""
        self._paused = True
        logger.info("trading_paused_via_telegram")
        self._log_to_telegram("INFO", "Trading paused via Telegram")

    async def _handle_resume(self):
        """Handle resume command from Telegram."""
        self._paused = False
        logger.info("trading_resumed_via_telegram")
        self._log_to_telegram("INFO", "Trading resumed via Telegram")

    async def _handle_exit_all(self):
        """Handle exit all command from Telegram."""
        self._log_to_telegram("WARNING", "Exit all triggered via Telegram")
        await self._exit_all_positions("Manual exit via Telegram")
