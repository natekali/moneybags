"""
Token discovery service using Helius RPC.
Monitors the Bags program for new token launches instead of using Bags API.

The Bags.fm API doesn't have a /tokens/new endpoint, so we use Solana RPC
to monitor the Bags token launch program directly.
"""

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog
from solders.pubkey import Pubkey

from bags_sniper.core.config import Settings
from bags_sniper.services.solana_rpc import SolanaRPCClient

logger = structlog.get_logger()

# Native SOL mint address - must be filtered at discovery to avoid buying SOL with SOL
SOL_MINT = "So11111111111111111111111111111111111111112"

# Other common non-tradeable mints to filter
EXCLUDED_MINTS = {
    SOL_MINT,
    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
}

# Bags.fm / Meteora program IDs (from docs.bags.fm/principles/program-ids)
# Meteora DBC - handles token creation and bonding curves (new launches)
METEORA_DBC_PROGRAM_ID = "dbcij3LWUppWqq96dh6gJWwBifmcGfLSB5D4DuSMaqN"
# Meteora DAMM v2 - post-launch token trading
METEORA_DAMM_PROGRAM_ID = "cpamdpZCGKUy5JxQXB4dcpGPiikHawvSWAd6mEn1sGG"
# We monitor DBC for new token launches
BAGS_PROGRAM_ID = METEORA_DBC_PROGRAM_ID


@dataclass
class DiscoveredToken:
    """Represents a newly discovered token from the Bags program."""

    mint_address: str
    deployer_wallet: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    signature: str = ""
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    slot: int = 0
    mcap_sol: Optional[float] = None


class TokenDiscoveryService:
    """
    Discovers new token launches by monitoring the Bags program on Solana.

    Uses Helius RPC to:
    1. Get recent transaction signatures for the Bags program
    2. Parse transactions to extract new token launch data
    3. Track seen signatures to avoid duplicates
    """

    def __init__(self, settings: Settings, rpc_client: SolanaRPCClient):
        self.settings = settings
        self.rpc = rpc_client

        # Track seen signatures to avoid duplicates
        self._seen_signatures: set[str] = set()
        self._max_seen_signatures = 10000  # Limit memory usage

        # Last processed signature for pagination
        self._last_signature: Optional[str] = None

        # Cache discovered tokens briefly to handle race conditions
        self._recent_tokens: dict[str, DiscoveredToken] = {}
        self._token_cache_ttl = timedelta(minutes=5)

    async def discover_new_tokens(self, limit: int = 20) -> list[DiscoveredToken]:
        """
        Discover new token launches from the Bags program.

        Returns:
            List of newly discovered tokens (not seen before)
        """
        try:
            # Get recent transaction signatures for Bags program
            response = await self.rpc.get_signatures_for_address(
                BAGS_PROGRAM_ID,
                limit=limit,
            )

            if not response.success:
                logger.warning(
                    "failed_to_fetch_program_signatures",
                    error=response.error,
                )
                return []

            signatures_data = response.result or []
            new_tokens: list[DiscoveredToken] = []

            for sig_info in signatures_data:
                signature = sig_info.get("signature", "")

                # Skip if already seen
                if signature in self._seen_signatures:
                    continue

                # Skip failed transactions
                if sig_info.get("err") is not None:
                    self._seen_signatures.add(signature)
                    continue

                # Parse the transaction
                token = await self._parse_token_launch_tx(signature, sig_info)
                if token:
                    new_tokens.append(token)
                    self._recent_tokens[token.mint_address] = token

                self._seen_signatures.add(signature)

            # Update last signature for pagination
            if signatures_data:
                self._last_signature = signatures_data[-1].get("signature")

            # Clean up old seen signatures if needed
            if len(self._seen_signatures) > self._max_seen_signatures:
                # Keep most recent half
                self._seen_signatures = set(
                    list(self._seen_signatures)[self._max_seen_signatures // 2:]
                )

            # Clean up old cached tokens
            self._cleanup_token_cache()

            if new_tokens:
                logger.info(
                    "discovered_new_tokens",
                    count=len(new_tokens),
                    mints=[t.mint_address[:8] for t in new_tokens],
                )

            return new_tokens

        except Exception as e:
            logger.error("token_discovery_error", error=str(e))
            return []

    async def _parse_token_launch_tx(
        self,
        signature: str,
        sig_info: dict,
    ) -> Optional[DiscoveredToken]:
        """
        Parse a transaction to extract token launch information.

        The Bags program creates tokens with specific instruction patterns.
        We look for token creation instructions and extract the mint/deployer.
        """
        try:
            # Get full transaction details
            response = await self.rpc.get_transaction(signature)

            if not response.success or not response.result:
                return None

            tx = response.result
            meta = tx.get("meta", {})

            # Skip failed transactions
            if meta.get("err") is not None:
                return None

            # Get the transaction message
            message = tx.get("transaction", {}).get("message", {})
            account_keys = message.get("accountKeys", [])

            # For parsed transactions
            if not account_keys and "transaction" in tx:
                parsed_tx = tx.get("transaction", {})
                if "message" in parsed_tx:
                    message = parsed_tx["message"]
                    account_keys = message.get("accountKeys", [])

            if not account_keys:
                return None

            # Extract account addresses (handle both string and dict formats)
            accounts: list[str] = []
            for key in account_keys:
                if isinstance(key, str):
                    accounts.append(key)
                elif isinstance(key, dict):
                    accounts.append(key.get("pubkey", ""))

            # Find potential mint address from post token balances
            post_token_balances = meta.get("postTokenBalances", [])

            # Look for new token mints (tokens that didn't exist before)
            pre_token_balances = meta.get("preTokenBalances", [])
            pre_mints = {b.get("mint") for b in pre_token_balances if b.get("mint")}

            new_mint = None
            for balance in post_token_balances:
                mint = balance.get("mint")
                if mint and mint not in pre_mints:
                    new_mint = mint
                    break

            if not new_mint:
                # Try to find from inner instructions or logs
                new_mint = self._extract_mint_from_logs(meta.get("logMessages", []))

            if not new_mint:
                return None

            # Filter out excluded mints (SOL, USDC, USDT, etc.) at discovery stage
            # This prevents wasted processing and API calls for non-tradeable tokens
            if new_mint in EXCLUDED_MINTS:
                logger.debug(
                    "excluded_mint_filtered_at_discovery",
                    mint=new_mint[:8],
                    reason="Non-tradeable token (SOL/USDC/USDT)",
                )
                return None

            # The deployer is typically the fee payer (first account)
            deployer = accounts[0] if accounts else ""

            # Extract token name/symbol from logs if available
            name, symbol = self._extract_token_metadata_from_logs(
                meta.get("logMessages", [])
            )

            return DiscoveredToken(
                mint_address=new_mint,
                deployer_wallet=deployer,
                name=name,
                symbol=symbol,
                signature=signature,
                slot=sig_info.get("slot", 0),
            )

        except Exception as e:
            logger.debug(
                "failed_to_parse_tx",
                signature=signature[:16],
                error=str(e),
            )
            return None

    def _extract_mint_from_logs(self, logs: list[str]) -> Optional[str]:
        """Extract mint address from transaction logs."""
        for log in logs:
            # Look for patterns like "mint: <address>" or "Token: <address>"
            if "mint" in log.lower():
                # Try to extract base58 address (32-44 chars, alphanumeric)
                parts = log.split()
                for part in parts:
                    # Clean up the part
                    clean = part.strip(",:;'\"")
                    if len(clean) >= 32 and len(clean) <= 44 and clean.isalnum():
                        return clean
        return None

    def _extract_token_metadata_from_logs(
        self,
        logs: list[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract token name and symbol from transaction logs."""
        name = None
        symbol = None

        for log in logs:
            log_lower = log.lower()

            # Look for name patterns
            if "name:" in log_lower or "token name" in log_lower:
                parts = log.split(":")
                if len(parts) > 1:
                    name = parts[-1].strip().strip('"\'')[:50]  # Limit length

            # Look for symbol patterns
            if "symbol:" in log_lower or "token symbol" in log_lower:
                parts = log.split(":")
                if len(parts) > 1:
                    symbol = parts[-1].strip().strip('"\'')[:10]  # Limit length

        return name, symbol

    def _cleanup_token_cache(self):
        """Remove tokens older than TTL from cache."""
        now = datetime.utcnow()
        expired = [
            mint for mint, token in self._recent_tokens.items()
            if now - token.discovered_at > self._token_cache_ttl
        ]
        for mint in expired:
            del self._recent_tokens[mint]

    def get_cached_token(self, mint_address: str) -> Optional[DiscoveredToken]:
        """Get a recently discovered token from cache."""
        return self._recent_tokens.get(mint_address)

    async def get_token_metadata_from_chain(
        self,
        mint_address: str,
    ) -> Optional[dict[str, Any]]:
        """
        Fetch token metadata from the blockchain.

        Tries multiple sources:
        1. Metaplex token metadata PDA
        2. Token account data
        3. Cached transaction logs (from discovery)
        """
        try:
            # First check if we have cached metadata from discovery
            cached_token = self.get_cached_token(mint_address)
            if cached_token and (cached_token.name or cached_token.symbol):
                return {
                    "mint": mint_address,
                    "name": cached_token.name,
                    "symbol": cached_token.symbol,
                }

            # Try to derive Metaplex metadata PDA and fetch it
            metadata = await self._fetch_metaplex_metadata(mint_address)
            if metadata:
                return metadata

            # Fallback: Get basic account info
            response = await self.rpc.get_account_info(mint_address)

            if not response.success or not response.result:
                return None

            account = response.result.get("value")
            if not account:
                return None

            # Basic mint account info
            return {
                "mint": mint_address,
                "owner": account.get("owner"),
                "lamports": account.get("lamports"),
            }

        except Exception as e:
            logger.debug(
                "failed_to_fetch_token_metadata",
                mint=mint_address[:8],
                error=str(e),
            )
            return None

    async def _fetch_metaplex_metadata(
        self,
        mint_address: str,
    ) -> Optional[dict[str, Any]]:
        """
        Fetch token metadata using Helius DAS API (preferred) or Metaplex fallback.

        The DAS API is more reliable and returns structured metadata directly.
        Docs: https://www.helius.dev/docs/das-api
        """
        # Try Helius DAS API first (more reliable)
        try:
            response = await self.rpc.get_asset(mint_address)

            if response.success and response.result:
                content = response.result.get("content", {})
                metadata = content.get("metadata", {})

                name = metadata.get("name", "")
                symbol = metadata.get("symbol", "")

                # Clean up null bytes if present
                if name:
                    name = name.replace('\x00', '').strip()
                if symbol:
                    symbol = symbol.replace('\x00', '').strip()

                if name or symbol:
                    logger.debug(
                        "das_api_metadata_success",
                        mint=mint_address[:8],
                        name=name[:20] if name else None,
                        symbol=symbol,
                    )
                    return {
                        "mint": mint_address,
                        "name": name if name else None,
                        "symbol": symbol if symbol else None,
                    }

        except Exception as e:
            logger.debug(
                "das_api_metadata_error",
                mint=mint_address[:8],
                error=str(e),
            )

        # Fallback to manual Metaplex PDA derivation (less reliable)
        try:
            METADATA_PROGRAM_ID = "metaqbxxUerdq28cj1RbAWkYQm3ybzjb6a8bt518x1s"

            from solders.pubkey import Pubkey
            import hashlib

            mint_pubkey = Pubkey.from_string(mint_address)
            metadata_program = Pubkey.from_string(METADATA_PROGRAM_ID)

            seeds = [
                b"metadata",
                bytes(metadata_program),
                bytes(mint_pubkey),
            ]

            for bump in range(255, 250, -1):
                try:
                    seed_with_bump = seeds + [bytes([bump])]
                    hasher = hashlib.sha256()
                    for seed in seed_with_bump:
                        hasher.update(seed)
                    hasher.update(bytes(metadata_program))
                    hasher.update(b"ProgramDerivedAddress")
                    derived = hasher.digest()
                    metadata_pda = Pubkey.from_bytes(derived)
                    break
                except Exception:
                    continue
            else:
                return None

            response = await self.rpc.get_account_info(str(metadata_pda))

            if not response.success or not response.result:
                return None

            account = response.result.get("value")
            if not account:
                return None

            data = account.get("data", [])
            if isinstance(data, list) and len(data) > 0:
                import base64
                raw_data = base64.b64decode(data[0])

                if len(raw_data) > 100:
                    name_start = 65
                    name_bytes = raw_data[name_start:name_start + 32]
                    name = name_bytes.decode('utf-8', errors='ignore').strip('\x00').strip()

                    symbol_start = name_start + 32 + 4
                    symbol_bytes = raw_data[symbol_start:symbol_start + 10]
                    symbol = symbol_bytes.decode('utf-8', errors='ignore').strip('\x00').strip()

                    if name or symbol:
                        return {
                            "mint": mint_address,
                            "name": name if name else None,
                            "symbol": symbol if symbol else None,
                        }

            return None

        except Exception as e:
            logger.debug(
                "metaplex_metadata_fetch_error",
                mint=mint_address[:8],
                error=str(e),
            )
            return None

    async def enrich_token_metadata(
        self,
        token: "DiscoveredToken",
    ) -> "DiscoveredToken":
        """
        Enrich a discovered token with metadata if not already present.
        """
        if token.name and token.symbol:
            return token

        metadata = await self.get_token_metadata_from_chain(token.mint_address)
        if metadata:
            if not token.name and metadata.get("name"):
                token.name = metadata["name"]
            if not token.symbol and metadata.get("symbol"):
                token.symbol = metadata["symbol"]

        return token
