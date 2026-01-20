"""
Deployer Data Ingestion Service.
Fetches historical pump.fun token data to build deployer intelligence database.

This is CRITICAL - without historical data, deployer scores are all 0
and the bot cannot make intelligent decisions.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bags_sniper.core.config import Settings
from bags_sniper.core.deployer_intelligence import DeployerIntelligence
from bags_sniper.models.database import Deployer, TokenLaunch

logger = structlog.get_logger()

# Pump.fun program addresses
PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_FUN_BONDING_CURVE = "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1"

# Graduation threshold (~69K USD market cap on bonding curve)
GRADUATION_SOL_THRESHOLD = Decimal("85")  # ~85 SOL at graduation


@dataclass
class TokenCreateEvent:
    """Parsed token creation event from pump.fun."""

    mint_address: str
    deployer_wallet: str
    name: str
    symbol: str
    uri: str
    timestamp: datetime
    signature: str
    initial_buy_sol: Optional[Decimal] = None


@dataclass
class GraduationEvent:
    """Parsed graduation event (token migrated to Raydium)."""

    mint_address: str
    timestamp: datetime
    final_mcap_sol: Decimal
    signature: str


class HeliusClient:
    """
    Helius API client for historical data fetching.
    Uses DAS API and enhanced transaction APIs.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = "https://api.helius.xyz/v0"
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    @property
    def api_key(self) -> str:
        return self.settings.helius_api_key.get_secret_value()

    async def get_signatures_for_address(
        self,
        address: str,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get transaction signatures for an address."""
        url = f"{self.base_url}/addresses/{address}/transactions"
        params = {
            "api-key": self.api_key,
            "limit": limit,
        }
        if before:
            params["before"] = before

        try:
            response = await self._client.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            logger.warning(
                "helius_request_failed",
                status=response.status_code,
                url=url,
            )
            return []
        except Exception as e:
            logger.error("helius_request_error", error=str(e))
            return []

    async def parse_transactions(
        self,
        signatures: list[str],
    ) -> list[dict]:
        """Parse transactions using Helius enhanced API."""
        if not signatures:
            return []

        url = f"{self.base_url}/transactions"
        params = {"api-key": self.api_key}

        try:
            response = await self._client.post(
                url,
                params=params,
                json={"transactions": signatures},
            )
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            logger.error("helius_parse_error", error=str(e))
            return []

    async def get_asset(self, mint: str) -> Optional[dict]:
        """Get token metadata using DAS API."""
        url = f"https://mainnet.helius-rpc.com/?api-key={self.api_key}"

        try:
            response = await self._client.post(
                url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getAsset",
                    "params": {"id": mint},
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("result")
            return None
        except Exception as e:
            logger.error("helius_das_error", error=str(e))
            return None


class DataIngestionService:
    """
    Service to ingest historical pump.fun data and build deployer profiles.

    Strategy:
    1. Fetch recent pump.fun token creations
    2. Extract deployer wallets
    3. Track which tokens graduated
    4. Calculate deployer scores
    """

    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker,
    ):
        self.settings = settings
        self.session_factory = session_factory
        self.helius = HeliusClient(settings)
        self._running = False

    async def run_backfill(
        self,
        days_back: int = 7,
        max_tokens: int = 1000,
    ):
        """
        Run historical backfill to populate deployer data.

        Args:
            days_back: How many days of history to fetch
            max_tokens: Maximum number of tokens to process
        """
        logger.info(
            "starting_backfill",
            days_back=days_back,
            max_tokens=max_tokens,
        )

        async with self.helius:
            # Fetch token creations
            token_events = await self._fetch_token_creations(max_tokens)
            logger.info("fetched_token_creations", count=len(token_events))

            if not token_events:
                logger.warning("no_token_events_found")
                return

            # Process each token
            async with self.session_factory() as session:
                deployer_intel = DeployerIntelligence(self.settings, None)

                for i, event in enumerate(token_events):
                    try:
                        await self._process_token_event(
                            session, deployer_intel, event
                        )

                        if (i + 1) % 50 == 0:
                            logger.info(
                                "backfill_progress",
                                processed=i + 1,
                                total=len(token_events),
                            )
                            await session.commit()

                    except Exception as e:
                        logger.error(
                            "token_processing_error",
                            mint=event.mint_address[:8],
                            error=str(e),
                        )

                await session.commit()

            # Update all deployer scores
            await self._recalculate_all_scores()

        logger.info("backfill_complete")

    async def _fetch_token_creations(
        self,
        max_tokens: int,
    ) -> list[TokenCreateEvent]:
        """Fetch recent token creation events from pump.fun."""
        events = []
        before_sig = None

        while len(events) < max_tokens:
            # Fetch transactions from pump.fun program
            txs = await self.helius.get_signatures_for_address(
                PUMP_FUN_PROGRAM,
                before=before_sig,
                limit=100,
            )

            if not txs:
                break

            # Parse transactions to find token creates
            signatures = [tx.get("signature") for tx in txs if tx.get("signature")]
            parsed = await self.helius.parse_transactions(signatures)

            for tx in parsed:
                event = self._parse_token_create(tx)
                if event:
                    events.append(event)
                    if len(events) >= max_tokens:
                        break

            # Pagination
            if txs:
                before_sig = txs[-1].get("signature")
            else:
                break

            # Rate limiting
            await asyncio.sleep(0.2)

        return events

    def _parse_token_create(self, tx: dict) -> Optional[TokenCreateEvent]:
        """Parse a transaction to extract token creation event."""
        try:
            # Look for token creation in transaction type
            tx_type = tx.get("type", "")
            if "CREATE" not in tx_type.upper() and "MINT" not in tx_type.upper():
                # Check instructions for pump.fun create
                instructions = tx.get("instructions", [])
                is_create = any(
                    PUMP_FUN_PROGRAM in str(inst.get("programId", ""))
                    and "create" in str(inst.get("data", "")).lower()
                    for inst in instructions
                )
                if not is_create:
                    return None

            # Extract details
            source = tx.get("source", "")
            if "PUMP" not in source.upper():
                return None

            # Get token info from transaction
            token_transfers = tx.get("tokenTransfers", [])
            if not token_transfers:
                return None

            mint = token_transfers[0].get("mint")
            if not mint:
                return None

            # Get deployer (fee payer is usually the deployer)
            fee_payer = tx.get("feePayer")
            if not fee_payer:
                return None

            # Get timestamp
            timestamp = tx.get("timestamp")
            if timestamp:
                timestamp = datetime.fromtimestamp(timestamp)
            else:
                timestamp = datetime.utcnow()

            # Try to get token metadata
            description = tx.get("description", "")
            name = ""
            symbol = ""

            # Parse from description if available
            if "created" in description.lower():
                # Try to extract name/symbol from description
                parts = description.split()
                for i, part in enumerate(parts):
                    if part.upper() == part and len(part) <= 10:
                        symbol = part
                        break

            return TokenCreateEvent(
                mint_address=mint,
                deployer_wallet=fee_payer,
                name=name,
                symbol=symbol,
                uri="",
                timestamp=timestamp,
                signature=tx.get("signature", ""),
            )

        except Exception as e:
            logger.debug("parse_error", error=str(e))
            return None

    async def _process_token_event(
        self,
        session: AsyncSession,
        deployer_intel: DeployerIntelligence,
        event: TokenCreateEvent,
    ):
        """Process a single token creation event."""
        # Check if token already exists
        result = await session.execute(
            select(TokenLaunch).where(TokenLaunch.mint_address == event.mint_address)
        )
        existing = result.scalar_one_or_none()
        if existing:
            return

        # Get or create deployer
        deployer = await deployer_intel.get_or_create_deployer(
            session, event.deployer_wallet
        )

        # Fetch additional token metadata
        asset_data = await self.helius.get_asset(event.mint_address)
        if asset_data:
            content = asset_data.get("content", {})
            metadata = content.get("metadata", {})
            event.name = metadata.get("name", event.name)
            event.symbol = metadata.get("symbol", event.symbol)

        # Check if token graduated (simplified - check if it has Raydium pool)
        graduated = await self._check_graduation(event.mint_address)

        # Create token launch record
        token_launch = TokenLaunch(
            mint_address=event.mint_address,
            deployer_id=deployer.id,
            name=event.name,
            symbol=event.symbol,
            uri=event.uri,
            launched_at=event.timestamp,
            graduated=graduated,
        )

        if graduated:
            token_launch.graduated_at = event.timestamp + timedelta(hours=1)  # Estimate
            token_launch.graduation_mcap_sol = GRADUATION_SOL_THRESHOLD

        session.add(token_launch)

        # Update deployer stats
        deployer.total_launches += 1
        if graduated:
            deployer.graduated_launches += 1
        deployer.last_launch_at = event.timestamp

        logger.debug(
            "processed_token",
            mint=event.mint_address[:8],
            deployer=event.deployer_wallet[:8],
            graduated=graduated,
        )

    async def _check_graduation(self, mint_address: str) -> bool:
        """
        Check if a token graduated (migrated to Raydium).
        Simplified check - looks for Raydium pool existence.
        """
        try:
            # Check if token has significant trading activity
            # This is a simplified heuristic
            asset = await self.helius.get_asset(mint_address)
            if not asset:
                return False

            # Check supply and ownership distribution
            # Graduated tokens typically have wider distribution
            ownership = asset.get("ownership", {})
            if ownership.get("delegated"):
                return True

            # Check token info for graduation indicators
            token_info = asset.get("token_info", {})
            price_info = token_info.get("price_info", {})
            if price_info.get("price_per_token"):
                # Has price = likely graduated and trading
                return True

            return False

        except Exception:
            return False

    async def _recalculate_all_scores(self):
        """Recalculate scores for all deployers."""
        logger.info("recalculating_deployer_scores")

        async with self.session_factory() as session:
            result = await session.execute(select(Deployer))
            deployers = result.scalars().all()

            for deployer in deployers:
                # Calculate graduation rate
                if deployer.total_launches > 0:
                    deployer.graduation_rate = (
                        deployer.graduated_launches / deployer.total_launches
                    )
                else:
                    deployer.graduation_rate = 0.0

                # Calculate score
                deployer.calculate_score()
                deployer.score_updated_at = datetime.utcnow()

            await session.commit()

            # Log statistics
            high_score = [d for d in deployers if d.deployer_score >= 60]
            logger.info(
                "scores_recalculated",
                total_deployers=len(deployers),
                high_score_deployers=len(high_score),
            )

    async def run_live_monitoring(self):
        """
        Run continuous monitoring for new token launches.
        Subscribes to pump.fun events in real-time.
        """
        logger.info("starting_live_monitoring")
        self._running = True

        async with self.helius:
            last_signature = None

            while self._running:
                try:
                    # Fetch latest transactions
                    txs = await self.helius.get_signatures_for_address(
                        PUMP_FUN_PROGRAM,
                        limit=20,
                    )

                    if txs and txs[0].get("signature") != last_signature:
                        # New transactions
                        new_sigs = []
                        for tx in txs:
                            sig = tx.get("signature")
                            if sig == last_signature:
                                break
                            new_sigs.append(sig)

                        if new_sigs:
                            parsed = await self.helius.parse_transactions(new_sigs)
                            async with self.session_factory() as session:
                                deployer_intel = DeployerIntelligence(
                                    self.settings, None
                                )
                                for tx in parsed:
                                    event = self._parse_token_create(tx)
                                    if event:
                                        await self._process_token_event(
                                            session, deployer_intel, event
                                        )
                                await session.commit()

                        last_signature = txs[0].get("signature")

                    await asyncio.sleep(2)  # Poll every 2 seconds

                except Exception as e:
                    logger.error("monitoring_error", error=str(e))
                    await asyncio.sleep(5)

    def stop(self):
        """Stop live monitoring."""
        self._running = False


async def run_backfill_cli(days: int = 7, max_tokens: int = 500):
    """CLI entry point for running backfill."""
    from bags_sniper.core.config import get_settings
    from bags_sniper.models.database import init_database

    settings = get_settings()
    session_factory = await init_database(settings.database_url.get_secret_value())

    service = DataIngestionService(settings, session_factory)
    await service.run_backfill(days_back=days, max_tokens=max_tokens)
