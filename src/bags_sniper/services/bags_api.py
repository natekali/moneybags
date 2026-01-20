"""
Bags.fm API client with rate limiting and error handling.
Handles all interactions with the Bags.fm trading platform.

Rate Limits (from https://docs.bags.fm/principles/rate-limits):
- 1,000 requests per hour per user (~0.28 RPS)
- Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
- 429 response when limit exceeded
"""

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import httpx
import structlog

from bags_sniper.core.config import Settings
from bags_sniper.services.rate_limiter import (
    RateLimiter,
    RateLimitService,
    get_rate_limiter,
)

logger = structlog.get_logger()


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderPriority(int, Enum):
    """Request priority for rate limiting."""

    CRITICAL = 1  # Exits, stop losses
    HIGH = 2  # Entries
    NORMAL = 3  # Data fetching
    LOW = 4  # Background updates


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter with priority queue support.
    Allows burst traffic while maintaining average rate.
    """

    capacity: int  # Max tokens
    refill_rate: float  # Tokens per second
    tokens: float = 0.0
    last_refill: float = 0.0

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, priority: OrderPriority = OrderPriority.NORMAL) -> bool:
        """
        Attempt to acquire a token. Returns True if successful.
        Higher priority requests get preference.
        """
        async with self._lock:
            self._refill()

            # Priority boost: critical requests can overdraw slightly
            min_tokens = 0.0
            if priority == OrderPriority.CRITICAL:
                min_tokens = -5.0  # Allow 5 token overdraw for critical
            elif priority == OrderPriority.HIGH:
                min_tokens = -2.0

            if self.tokens >= min_tokens:
                self.tokens -= 1
                return True
            return False

    async def wait_for_token(
        self, priority: OrderPriority = OrderPriority.NORMAL, timeout: float = 30.0
    ) -> bool:
        """Wait until a token is available or timeout."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if await self.acquire(priority):
                return True
            # Wait based on refill rate
            await asyncio.sleep(1.0 / self.refill_rate)
        return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


@dataclass
class BagsAPIResponse:
    """Standardized API response."""

    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    status_code: int = 0


class BagsAPIClient:
    """
    Async client for Bags.fm API.
    Implements rate limiting, retries, and error handling.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.bags_api_base_url.rstrip("/")

        # Token bucket: 1000 requests/hour = ~0.278 requests/second
        # Allow burst of up to 50 requests
        self.rate_limiter = TokenBucket(
            capacity=50,
            refill_rate=settings.bags_rate_limit_per_hour / 3600,
        )

        self._client: Optional[httpx.AsyncClient] = None
        self._global_rate_limiter: Optional[RateLimiter] = None

        # Track rate limit status from headers
        self._rate_limit_remaining: int = 1000
        self._rate_limit_reset: float = 0.0

    async def __aenter__(self) -> "BagsAPIClient":
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def connect(self):
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.settings.bags_api_key.get_secret_value(),
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        self._global_rate_limiter = await get_rate_limiter()
        logger.info("bags_api_connected", base_url=self.base_url)

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        priority: OrderPriority = OrderPriority.NORMAL,
        retries: int = 3,
        **kwargs,
    ) -> BagsAPIResponse:
        """
        Make rate-limited API request with retries.

        Handles Bags.fm rate limits:
        - 1,000 requests/hour per user
        - X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset headers
        - 429 response with exponential backoff
        """
        if not self._client:
            await self.connect()

        # Use global rate limiter first
        if self._global_rate_limiter:
            wait_time = await self._global_rate_limiter.acquire(RateLimitService.BAGS_FM)
            if wait_time > 0:
                logger.debug(
                    "bags_rate_limit_wait",
                    wait_seconds=wait_time,
                    endpoint=endpoint,
                )
                await asyncio.sleep(wait_time)

        # Then use local token bucket for burst control
        if not await self.rate_limiter.wait_for_token(priority, timeout=30.0):
            logger.warning("rate_limit_timeout", endpoint=endpoint)
            return BagsAPIResponse(
                success=False,
                error="Rate limit timeout",
                status_code=429,
            )

        last_error = None
        for attempt in range(retries):
            try:
                response = await self._client.request(method, endpoint, **kwargs)

                # Extract rate limit headers for tracking
                self._update_rate_limit_from_headers(response.headers)

                if response.status_code == 429:
                    # Rate limited by server - use headers for backoff
                    retry_after = None
                    headers_dict = dict(response.headers)

                    # Check for X-RateLimit-Reset header (Unix timestamp)
                    if "X-RateLimit-Reset" in headers_dict:
                        reset_time = float(headers_dict["X-RateLimit-Reset"])
                        retry_after = max(0, reset_time - time.time())
                    elif "Retry-After" in headers_dict:
                        retry_after = float(headers_dict["Retry-After"])

                    # Report to global rate limiter
                    if self._global_rate_limiter:
                        backoff = await self._global_rate_limiter.report_rate_limit(
                            RateLimitService.BAGS_FM,
                            retry_after=retry_after,
                            headers=headers_dict,
                        )
                    else:
                        backoff = retry_after or (2 ** (attempt + 1))

                    logger.warning(
                        "bags_rate_limited",
                        endpoint=endpoint,
                        backoff_seconds=backoff,
                        remaining=self._rate_limit_remaining,
                        attempt=attempt,
                    )
                    await asyncio.sleep(backoff)
                    continue

                if response.status_code >= 500:
                    # Server error, retry with exponential backoff
                    wait_time = 2**attempt
                    logger.warning(
                        "server_error",
                        endpoint=endpoint,
                        status=response.status_code,
                        attempt=attempt,
                        response_body=response.text[:500] if response.text else None,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code >= 400:
                    return BagsAPIResponse(
                        success=False,
                        error=response.text,
                        status_code=response.status_code,
                    )

                # Success - report to rate limiter
                if self._global_rate_limiter:
                    await self._global_rate_limiter.report_success(RateLimitService.BAGS_FM)

                return BagsAPIResponse(
                    success=True,
                    data=response.json(),
                    status_code=response.status_code,
                )

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                logger.warning("request_timeout", endpoint=endpoint, attempt=attempt)
                await asyncio.sleep(2**attempt)
            except httpx.RequestError as e:
                last_error = f"Request error: {e}"
                logger.error("request_error", endpoint=endpoint, error=str(e))
                await asyncio.sleep(2**attempt)

        return BagsAPIResponse(
            success=False,
            error=last_error or "Max retries exceeded",
            status_code=0,
        )

    def _update_rate_limit_from_headers(self, headers: httpx.Headers):
        """Update rate limit tracking from response headers."""
        try:
            if "X-RateLimit-Remaining" in headers:
                self._rate_limit_remaining = int(headers["X-RateLimit-Remaining"])
            if "X-RateLimit-Reset" in headers:
                self._rate_limit_reset = float(headers["X-RateLimit-Reset"])

            # Log warning if running low
            if self._rate_limit_remaining < 100:
                logger.warning(
                    "bags_rate_limit_low",
                    remaining=self._rate_limit_remaining,
                    reset_in=max(0, self._rate_limit_reset - time.time()),
                )
        except (ValueError, TypeError):
            pass  # Ignore header parsing errors

    # =========================================================================
    # Trading Endpoints (Two-step: Quote -> Swap)
    # =========================================================================

    # SOL mint address (native SOL wrapper)
    SOL_MINT = "So11111111111111111111111111111111111111112"

    def _get_wallet_public_key(self) -> str:
        """Get wallet public key from private key."""
        keypair = self._get_keypair()
        return str(keypair.pubkey())

    def _get_keypair(self):
        """Get Solana keypair from private key in various formats."""
        from solders.keypair import Keypair
        import base58
        import base64
        import json

        private_key_str = self.settings.wallet_private_key.get_secret_value().strip()

        # Try JSON array format first (Phantom exports as [n1, n2, ...])
        if private_key_str.startswith('['):
            try:
                key_array = json.loads(private_key_str)
                if isinstance(key_array, list) and len(key_array) == 64:
                    keypair = Keypair.from_bytes(bytes(key_array))
                    logger.debug("keypair_from_json_array", pubkey=str(keypair.pubkey())[:8])
                    return keypair
            except json.JSONDecodeError:
                pass

        # Handle different private key string formats
        private_key_bytes = None

        # Base58-encoded 64-byte keypair is typically 87-88 characters
        # Base64-encoded 64-byte keypair is 88 characters but uses = padding
        # Hex-encoded 64-byte keypair is 128 characters

        if len(private_key_str) == 128:
            # Hex encoded (64 bytes = 128 hex chars)
            try:
                private_key_bytes = bytes.fromhex(private_key_str)
            except ValueError:
                pass

        if private_key_bytes is None and len(private_key_str) == 64:
            # Could be hex encoded seed (32 bytes = 64 hex chars)
            try:
                private_key_bytes = bytes.fromhex(private_key_str)
            except ValueError:
                pass

        if private_key_bytes is None:
            # Try base58 first (most common for Solana wallets)
            try:
                private_key_bytes = base58.b58decode(private_key_str)
            except Exception:
                pass

        if private_key_bytes is None:
            # Try base64 as last resort
            try:
                private_key_bytes = base64.b64decode(private_key_str)
            except Exception:
                pass

        if private_key_bytes is None:
            raise ValueError("Could not decode private key (tried hex, base58, base64)")

        logger.debug(
            "private_key_decoded",
            str_len=len(private_key_str),
            bytes_len=len(private_key_bytes),
            first_bytes=private_key_bytes[:4].hex() if len(private_key_bytes) >= 4 else None,
        )

        # Handle different key lengths
        keypair = None

        if len(private_key_bytes) == 64:
            # Full keypair (32 private + 32 public)
            keypair = Keypair.from_bytes(private_key_bytes)
        elif len(private_key_bytes) == 32:
            # Just the seed - generate keypair from seed
            keypair = Keypair.from_seed(private_key_bytes)
        else:
            # Try different interpretations for unusual lengths
            # For 66 bytes, common formats are:
            # - [1 byte version][64 byte keypair][1 byte checksum]
            # - [64 byte keypair][2 byte checksum]
            # - [2 byte prefix][64 byte keypair]
            attempts = []

            # Try bytes[1:65] - version prefix format
            if len(private_key_bytes) >= 65:
                try:
                    kp = Keypair.from_bytes(private_key_bytes[1:65])
                    attempts.append(("bytes[1:65]", kp))
                except Exception as e:
                    pass

            # Try bytes[0:64] - suffix checksum format
            if len(private_key_bytes) >= 64:
                try:
                    kp = Keypair.from_bytes(private_key_bytes[0:64])
                    attempts.append(("bytes[0:64]", kp))
                except Exception as e:
                    pass

            # Try bytes[2:66] - 2-byte prefix format
            if len(private_key_bytes) >= 66:
                try:
                    kp = Keypair.from_bytes(private_key_bytes[2:66])
                    attempts.append(("bytes[2:66]", kp))
                except Exception as e:
                    pass

            # Try bytes[-64:] - last 64 bytes
            if len(private_key_bytes) >= 64:
                try:
                    kp = Keypair.from_bytes(private_key_bytes[-64:])
                    attempts.append(("bytes[-64:]", kp))
                except Exception as e:
                    pass

            if attempts:
                # Log all valid keypairs found for debugging
                for method, kp in attempts:
                    logger.info(
                        "keypair_candidate",
                        method=method,
                        pubkey=str(kp.pubkey()),
                    )
                # Use the first valid one
                keypair = attempts[0][1]
            else:
                raise ValueError(
                    f"Could not parse private key ({len(private_key_bytes)} bytes). "
                    "No valid 64-byte keypair slice found."
                )

        logger.info("keypair_created", pubkey=str(keypair.pubkey()))
        return keypair

    async def get_trade_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,  # Amount in smallest unit (lamports for SOL)
        slippage_bps: int = 500,
    ) -> BagsAPIResponse:
        """
        Get a trade quote from the Bags.fm API.

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in smallest unit (e.g., lamports for SOL)
            slippage_bps: Slippage tolerance in basis points
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": amount,
            "slippageMode": "manual",
            "slippageBps": slippage_bps,
        }

        return await self._request(
            "GET",
            "/trade/quote",
            priority=OrderPriority.HIGH,
            params=params,
        )

    async def create_swap_transaction(
        self,
        quote_response: dict,
        user_public_key: str,
    ) -> BagsAPIResponse:
        """
        Create a swap transaction from a quote.

        Args:
            quote_response: The quote response from get_trade_quote
            user_public_key: User's wallet public key
        """
        return await self._request(
            "POST",
            "/trade/swap",
            priority=OrderPriority.HIGH,
            json={
                "quoteResponse": quote_response,
                "userPublicKey": user_public_key,
            },
        )

    async def sign_and_submit_transaction(
        self,
        serialized_tx: str,
    ) -> BagsAPIResponse:
        """
        Sign and submit a serialized transaction to the Solana network.

        Args:
            serialized_tx: Base64 or base58 encoded serialized transaction
        """
        import base64
        from solders.keypair import Keypair
        from solders.transaction import VersionedTransaction, Transaction
        from solders.message import Message
        import base58

        try:
            # Get keypair using the centralized method
            keypair = self._get_keypair()

            logger.info(
                "wallet_signing_transaction",
                wallet_pubkey=str(keypair.pubkey()),
            )

            # Try to decode the transaction - could be base58 or base64
            # Bags.fm API typically returns base58, but we handle both
            tx_bytes = None
            decode_method = None

            # Detect likely encoding based on character set
            # Base58 uses: 123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz
            # Base64 uses: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=
            has_base64_chars = '+' in serialized_tx or '/' in serialized_tx or serialized_tx.endswith('=')
            has_base58_invalid = '0' in serialized_tx or 'O' in serialized_tx or 'I' in serialized_tx or 'l' in serialized_tx

            # Try base58 first if it looks like base58
            if not has_base64_chars or not has_base58_invalid:
                try:
                    decoded = base58.b58decode(serialized_tx)
                    # Valid Solana transactions are typically 200-1500 bytes
                    if 100 < len(decoded) < 2000:
                        tx_bytes = decoded
                        decode_method = "base58"
                        logger.debug(
                            "tx_decoded_base58",
                            tx_length=len(tx_bytes),
                        )
                except Exception as e:
                    logger.debug("base58_decode_failed", error=str(e))

            # Try base64 if base58 failed or looks like base64
            if tx_bytes is None:
                try:
                    # Handle padding if needed
                    padded_tx = serialized_tx
                    padding_needed = len(serialized_tx) % 4
                    if padding_needed:
                        padded_tx = serialized_tx + "=" * (4 - padding_needed)
                    decoded = base64.b64decode(padded_tx)
                    if 100 < len(decoded) < 2000:
                        tx_bytes = decoded
                        decode_method = "base64"
                        logger.debug(
                            "tx_decoded_base64",
                            tx_length=len(tx_bytes),
                        )
                except Exception as e:
                    logger.debug("base64_decode_failed", error=str(e))

            # Try URL-safe base64 variant
            if tx_bytes is None:
                try:
                    decoded = base64.urlsafe_b64decode(serialized_tx + '==')
                    if 100 < len(decoded) < 2000:
                        tx_bytes = decoded
                        decode_method = "base64_urlsafe"
                        logger.debug(
                            "tx_decoded_base64_urlsafe",
                            tx_length=len(tx_bytes),
                        )
                except Exception as e:
                    logger.debug("base64_urlsafe_decode_failed", error=str(e))

            if tx_bytes is None:
                logger.error(
                    "tx_decode_failed_all_methods",
                    tx_preview=serialized_tx[:100] if serialized_tx else None,
                    tx_length=len(serialized_tx) if serialized_tx else 0,
                )
                return BagsAPIResponse(
                    success=False,
                    error="Could not decode transaction (tried base58, base64, urlsafe_base64)",
                )

            logger.debug(
                "transaction_bytes_decoded",
                byte_length=len(tx_bytes),
                first_bytes=tx_bytes[:16].hex() if len(tx_bytes) >= 16 else tx_bytes.hex(),
            )

            # Try to parse as VersionedTransaction first, then fall back to legacy Transaction
            tx = None
            signed_tx_bytes = None

            # Detect if transaction is versioned or legacy
            # Transaction format: [num_signatures: 1 byte][signatures: 64 * n][version or message]
            # Version byte (for versioned tx) is at position 1 + 64 * num_signatures
            # If that byte is >= 0x80, it's a versioned transaction
            is_versioned = False
            if len(tx_bytes) > 0:
                num_signatures = tx_bytes[0]
                version_byte_pos = 1 + (64 * num_signatures)
                if len(tx_bytes) > version_byte_pos:
                    is_versioned = tx_bytes[version_byte_pos] >= 0x80
                    logger.debug(
                        "tx_version_check",
                        num_signatures=num_signatures,
                        version_byte_pos=version_byte_pos,
                        version_byte=tx_bytes[version_byte_pos] if len(tx_bytes) > version_byte_pos else None,
                        is_versioned=is_versioned,
                    )

            if is_versioned:
                try:
                    tx = VersionedTransaction.from_bytes(tx_bytes)
                    logger.debug("parsed_as_versioned_transaction")

                    # For VersionedTransaction, we need to:
                    # 1. Get the message bytes to sign
                    # 2. Sign those bytes
                    # 3. Create a new VersionedTransaction with the signature
                    from solders.message import to_bytes_versioned

                    message_bytes = to_bytes_versioned(tx.message)
                    signature = keypair.sign_message(message_bytes)

                    # Create the signed transaction using populate
                    signed_tx = VersionedTransaction.populate(tx.message, [signature])
                    signed_tx_bytes = bytes(signed_tx)

                    logger.debug(
                        "versioned_tx_signed",
                        signature=str(signature)[:20],
                    )
                except Exception as e:
                    logger.warning(
                        "versioned_tx_parse_failed",
                        error=str(e),
                        trying_legacy=True,
                    )

            # Try legacy Transaction if versioned failed or wasn't detected
            if signed_tx_bytes is None:
                try:
                    tx = Transaction.from_bytes(tx_bytes)
                    logger.debug("parsed_as_legacy_transaction")

                    # For legacy transactions, sign the message and create signed transaction
                    from solders.message import Message as LegacyMessage

                    message_bytes = bytes(tx.message)
                    signature = keypair.sign_message(message_bytes)

                    # Create signed transaction
                    signed_tx = Transaction.populate(tx.message, [signature])
                    signed_tx_bytes = bytes(signed_tx)

                    logger.debug("legacy_tx_signed")
                except Exception as e:
                    logger.warning(
                        "legacy_tx_parse_failed",
                        error=str(e),
                    )

            # If both failed, log error
            if signed_tx_bytes is None:
                logger.error(
                    "all_tx_parse_methods_failed",
                    error="Could not parse as VersionedTransaction or Transaction",
                    tx_bytes_hex=tx_bytes[:64].hex() if len(tx_bytes) >= 64 else tx_bytes.hex(),
                )
                return BagsAPIResponse(
                    success=False,
                    error="Failed to parse and sign transaction",
                )

            # Submit via Helius RPC
            from bags_sniper.services.solana_rpc import SolanaRPCClient

            rpc = SolanaRPCClient(self.settings)
            await rpc.connect()

            try:
                response = await rpc.send_transaction(signed_tx_bytes)

                if response.success:
                    logger.info(
                        "transaction_submitted",
                        signature=response.result,
                    )
                    return BagsAPIResponse(
                        success=True,
                        data={"signature": response.result},
                    )
                else:
                    logger.error(
                        "transaction_submit_failed",
                        error=response.error,
                    )
                    return BagsAPIResponse(
                        success=False,
                        error=response.error or "Transaction submission failed",
                    )
            finally:
                await rpc.close()

        except Exception as e:
            logger.error("sign_submit_error", error=str(e), exc_info=True)
            return BagsAPIResponse(success=False, error=str(e))

    async def buy_token(
        self,
        mint_address: str,
        amount_sol: Decimal,
        slippage_bps: int = 500,  # 5% default
        use_jupiter_fallback: bool = True,
    ) -> BagsAPIResponse:
        """
        Execute a buy order for a token using the two-step quote -> swap flow.
        Falls back to Jupiter API if Bags.fm returns errors.

        Args:
            mint_address: Token mint address
            amount_sol: Amount in SOL to spend
            slippage_bps: Slippage tolerance in basis points
            use_jupiter_fallback: Whether to use Jupiter as fallback
        """
        logger.info(
            "executing_buy",
            mint=mint_address,
            amount_sol=str(amount_sol),
            slippage_bps=slippage_bps,
        )

        # Don't try to buy SOL with SOL
        if mint_address == self.SOL_MINT:
            return BagsAPIResponse(
                success=False,
                error="Cannot buy SOL with SOL - invalid token",
            )

        try:
            # Get wallet public key
            user_pubkey = self._get_wallet_public_key()

            # Convert SOL to lamports
            amount_lamports = int(amount_sol * Decimal("1e9"))

            # Step 1: Get quote (SOL -> Token)
            quote_response = await self.get_trade_quote(
                input_mint=self.SOL_MINT,
                output_mint=mint_address,
                amount=amount_lamports,
                slippage_bps=slippage_bps,
            )

            if not quote_response.success or not quote_response.data:
                # Try Jupiter fallback for 500 errors
                if use_jupiter_fallback and (quote_response.status_code >= 500 or "Max retries" in (quote_response.error or "")):
                    logger.warning(
                        "bags_api_failed_trying_jupiter",
                        mint=mint_address[:8],
                        error=quote_response.error,
                    )
                    return await self._buy_via_jupiter(mint_address, amount_sol, slippage_bps)

                return BagsAPIResponse(
                    success=False,
                    error=f"Failed to get quote: {quote_response.error}",
                )

            # The API might wrap the response - check for nested 'response' field
            quote_data = quote_response.data
            if "response" in quote_data:
                quote_data = quote_data["response"]

            logger.info(
                "quote_received",
                mint=mint_address[:8],
                request_id=quote_data.get("requestId"),
                in_amount=quote_data.get("inAmount"),
                out_amount=quote_data.get("outAmount"),
            )

            # Step 2: Create swap transaction (pass the unwrapped quote data)
            swap_response = await self.create_swap_transaction(
                quote_response=quote_data,
                user_public_key=user_pubkey,
            )

            if not swap_response.success or not swap_response.data:
                return BagsAPIResponse(
                    success=False,
                    error=f"Failed to create swap: {swap_response.error}",
                )

            # Get swap transaction from response
            swap_data = swap_response.data
            logger.info(
                "swap_response_full",
                mint=mint_address[:8],
                swap_data_type=type(swap_data).__name__,
                swap_data_keys=list(swap_data.keys()) if isinstance(swap_data, dict) else None,
                swap_data_preview=str(swap_data)[:500] if swap_data else None,
            )

            if "response" in swap_data:
                swap_data = swap_data["response"]
                logger.info(
                    "swap_response_unwrapped",
                    mint=mint_address[:8],
                    unwrapped_type=type(swap_data).__name__,
                    unwrapped_keys=list(swap_data.keys()) if isinstance(swap_data, dict) else None,
                )

            swap_tx = swap_data.get("swapTransaction")

            if not swap_tx:
                logger.error(
                    "no_swap_tx",
                    swap_data_keys=list(swap_data.keys()) if isinstance(swap_data, dict) else type(swap_data),
                    swap_data_full=str(swap_data)[:1000],
                )
                return BagsAPIResponse(
                    success=False,
                    error="No swap transaction in response",
                )

            logger.info(
                "swap_tx_received",
                mint=mint_address[:8],
                tx_length=len(swap_tx) if swap_tx else 0,
                tx_start=swap_tx[:100] if swap_tx else None,
            )

            # Step 3: Sign and submit transaction with retry on slippage errors
            submit_response = await self.sign_and_submit_transaction(swap_tx)

            if submit_response.success:
                out_amount = quote_data.get("outAmount", "1")
                return BagsAPIResponse(
                    success=True,
                    data={
                        "signature": submit_response.data.get("signature"),
                        "tokens_received": out_amount,
                        "price": amount_sol / Decimal(out_amount) if out_amount else Decimal(0),
                    },
                )

            # Check for retryable errors
            error_msg = submit_response.error or ""

            # Error 0x1 = slippage exceeded, retry with higher slippage
            if "0x1" in error_msg or "slippage" in error_msg.lower():
                if slippage_bps < 1500:  # Don't retry if already very high
                    higher_slippage = min(slippage_bps + 300, 1500)  # Increase by 3%, cap at 15%
                    logger.warning(
                        "retrying_with_higher_slippage",
                        mint=mint_address[:8],
                        original_slippage=slippage_bps,
                        new_slippage=higher_slippage,
                    )
                    return await self.buy_token(
                        mint_address=mint_address,
                        amount_sol=amount_sol,
                        slippage_bps=higher_slippage,
                        use_jupiter_fallback=use_jupiter_fallback,
                    )

            # Insufficient funds error - try Jupiter which may have better routing
            if "debit an account" in error_msg.lower() or "insufficient" in error_msg.lower():
                if use_jupiter_fallback:
                    logger.warning(
                        "insufficient_funds_trying_jupiter",
                        mint=mint_address[:8],
                        error=error_msg[:100],
                    )
                    return await self._buy_via_jupiter(mint_address, amount_sol, slippage_bps)

            return BagsAPIResponse(
                success=False,
                error=submit_response.error,
            )

        except Exception as e:
            logger.error("buy_token_error", error=str(e), mint=mint_address[:8])
            return BagsAPIResponse(success=False, error=str(e))

    async def sell_token(
        self,
        mint_address: str,
        amount_tokens: Optional[int] = None,
        percent: Optional[int] = None,
        slippage_bps: int = 1000,  # 10% default for sells
    ) -> BagsAPIResponse:
        """
        Execute a sell order for a token using the two-step quote -> swap flow.

        Args:
            mint_address: Token mint address
            amount_tokens: Exact amount of tokens to sell (mutually exclusive with percent)
            percent: Percentage of holdings to sell (mutually exclusive with amount_tokens)
            slippage_bps: Slippage tolerance in basis points
        """
        if amount_tokens is None and percent is None:
            return BagsAPIResponse(
                success=False, error="Must specify amount_tokens or percent"
            )

        logger.info(
            "executing_sell",
            mint=mint_address,
            amount_tokens=amount_tokens,
            percent=percent,
            slippage_bps=slippage_bps,
        )

        try:
            # Get wallet public key
            user_pubkey = self._get_wallet_public_key()

            # If percent specified, get holdings first to calculate amount
            if percent is not None:
                holdings_response = await self.get_holdings()
                if not holdings_response.success:
                    return BagsAPIResponse(
                        success=False,
                        error=f"Failed to get holdings: {holdings_response.error}",
                    )

                # Find token in holdings
                holdings = holdings_response.data or []
                token_holding = None
                for holding in holdings:
                    if holding.get("mint") == mint_address:
                        token_holding = holding
                        break

                if not token_holding:
                    return BagsAPIResponse(
                        success=False,
                        error=f"Token {mint_address} not found in holdings",
                    )

                total_tokens = int(token_holding.get("amount", 0))
                amount_tokens = int(total_tokens * percent / 100)

            if amount_tokens is None or amount_tokens <= 0:
                return BagsAPIResponse(
                    success=False,
                    error="Invalid token amount to sell",
                )

            # Step 1: Get quote (Token -> SOL)
            quote_response = await self.get_trade_quote(
                input_mint=mint_address,
                output_mint=self.SOL_MINT,
                amount=amount_tokens,
                slippage_bps=slippage_bps,
            )

            if not quote_response.success or not quote_response.data:
                return BagsAPIResponse(
                    success=False,
                    error=f"Failed to get quote: {quote_response.error}",
                )

            # Step 2: Create swap transaction
            swap_response = await self.create_swap_transaction(
                quote_response=quote_response.data,
                user_public_key=user_pubkey,
            )

            if not swap_response.success or not swap_response.data:
                return BagsAPIResponse(
                    success=False,
                    error=f"Failed to create swap: {swap_response.error}",
                )

            swap_tx = swap_response.data.get("response", {}).get("swapTransaction")
            if not swap_tx:
                swap_tx = swap_response.data.get("swapTransaction")

            if not swap_tx:
                return BagsAPIResponse(
                    success=False,
                    error="No swap transaction in response",
                )

            # Step 3: Sign and submit transaction
            submit_response = await self.sign_and_submit_transaction(swap_tx)

            if submit_response.success:
                sol_received = Decimal(quote_response.data.get("outAmount", 0)) / Decimal("1e9")
                return BagsAPIResponse(
                    success=True,
                    data={
                        "signature": submit_response.data.get("signature"),
                        "sol_received": sol_received,
                        "tokens_sold": amount_tokens,
                    },
                )
            else:
                return BagsAPIResponse(
                    success=False,
                    error=submit_response.error,
                )

        except Exception as e:
            logger.error("sell_token_error", error=str(e), mint=mint_address[:8])
            return BagsAPIResponse(success=False, error=str(e))

    async def sell_all(
        self,
        mint_address: str,
        slippage_bps: int = 1500,  # Higher slippage for emergency exits
    ) -> BagsAPIResponse:
        """Sell entire position in a token."""
        return await self.sell_token(
            mint_address=mint_address,
            percent=100,
            slippage_bps=slippage_bps,
        )

    # =========================================================================
    # Data Endpoints
    # =========================================================================

    async def get_token_info(self, mint_address: str) -> BagsAPIResponse:
        """Get token information including price and market cap."""
        return await self._request(
            "GET",
            f"/token/{mint_address}",
            priority=OrderPriority.NORMAL,
        )

    async def get_token_price(self, mint_address: str) -> Optional[Decimal]:
        """Get current token price in SOL."""
        response = await self.get_token_info(mint_address)
        if response.success and response.data:
            price = response.data.get("price_sol") or response.data.get("price")
            if price:
                return Decimal(str(price))
        return None

    async def get_holdings(self) -> BagsAPIResponse:
        """Get current token holdings for the authenticated wallet."""
        return await self._request(
            "GET",
            "/wallet/holdings",
            priority=OrderPriority.NORMAL,
        )

    async def get_wallet_balance(self) -> Optional[Decimal]:
        """
        Get SOL balance for the authenticated wallet.

        Note: Bags.fm API doesn't have a wallet balance endpoint, so we use
        Solana RPC directly to fetch the balance.
        """
        try:
            from bags_sniper.services.solana_rpc import SolanaRPCClient

            # Get wallet public key
            wallet_pubkey = self._get_wallet_public_key()

            # Use RPC to get balance
            rpc = SolanaRPCClient(self.settings)
            await rpc.connect()

            try:
                balance_lamports = await rpc.get_balance(wallet_pubkey)
                if balance_lamports is not None:
                    # Convert lamports to SOL (1 SOL = 1e9 lamports)
                    balance_sol = Decimal(balance_lamports) / Decimal("1000000000")
                    logger.debug(
                        "wallet_balance_fetched",
                        wallet=wallet_pubkey[:8],
                        balance_sol=str(balance_sol),
                    )
                    return balance_sol
                else:
                    logger.warning(
                        "wallet_balance_fetch_failed",
                        wallet=wallet_pubkey[:8],
                    )
                    return None
            finally:
                await rpc.close()

        except Exception as e:
            logger.error("wallet_balance_error", error=str(e))
            return None

    async def get_new_tokens(
        self,
        limit: int = 50,
        min_mcap_sol: Optional[float] = None,
    ) -> BagsAPIResponse:
        """
        Get recently launched tokens.

        Args:
            limit: Maximum number of tokens to return
            min_mcap_sol: Minimum market cap filter
        """
        params = {"limit": limit}
        if min_mcap_sol:
            params["min_mcap"] = min_mcap_sol

        return await self._request(
            "GET",
            "/tokens/new",
            priority=OrderPriority.HIGH,  # Token discovery is time-sensitive
            params=params,
        )

    async def get_token_trades(
        self,
        mint_address: str,
        limit: int = 100,
    ) -> BagsAPIResponse:
        """Get recent trades for a token."""
        return await self._request(
            "GET",
            f"/token/{mint_address}/trades",
            priority=OrderPriority.LOW,
            params={"limit": limit},
        )

    async def _buy_via_jupiter(
        self,
        mint_address: str,
        amount_sol: Decimal,
        slippage_bps: int,
    ) -> BagsAPIResponse:
        """
        Execute buy via Jupiter API as fallback.

        Args:
            mint_address: Token mint address
            amount_sol: Amount in SOL to spend
            slippage_bps: Slippage tolerance in basis points
        """
        try:
            from bags_sniper.services.jupiter_api import JupiterAPIClient
            from bags_sniper.services.solana_rpc import SolanaRPCClient

            logger.info(
                "executing_buy_via_jupiter_fallback",
                mint=mint_address[:8],
                amount_sol=str(amount_sol),
            )

            jupiter = JupiterAPIClient(self.settings)
            rpc = SolanaRPCClient(self.settings)

            await jupiter.connect()
            await rpc.connect()

            try:
                response = await jupiter.buy_token(
                    mint_address=mint_address,
                    amount_sol=amount_sol,
                    slippage_bps=slippage_bps,
                    rpc_client=rpc,
                )

                # Convert JupiterResponse to BagsAPIResponse
                if response.success:
                    logger.info(
                        "jupiter_buy_succeeded",
                        mint=mint_address[:8],
                        signature=response.data.get("signature") if response.data else None,
                    )
                    return BagsAPIResponse(
                        success=True,
                        data=response.data,
                    )
                else:
                    logger.warning(
                        "jupiter_buy_failed",
                        mint=mint_address[:8],
                        error=response.error,
                    )
                    return BagsAPIResponse(
                        success=False,
                        error=f"Jupiter fallback failed: {response.error}",
                    )
            finally:
                await jupiter.close()
                await rpc.close()

        except Exception as e:
            logger.error("jupiter_fallback_error", error=str(e))
            return BagsAPIResponse(
                success=False,
                error=f"Jupiter fallback error: {str(e)}",
            )

    async def _sell_via_jupiter(
        self,
        mint_address: str,
        amount_tokens: int,
        slippage_bps: int,
    ) -> BagsAPIResponse:
        """
        Execute sell via Jupiter API as fallback.

        Args:
            mint_address: Token mint address
            amount_tokens: Amount of tokens to sell
            slippage_bps: Slippage tolerance in basis points
        """
        try:
            from bags_sniper.services.jupiter_api import JupiterAPIClient
            from bags_sniper.services.solana_rpc import SolanaRPCClient

            logger.info(
                "executing_sell_via_jupiter_fallback",
                mint=mint_address[:8],
                amount_tokens=amount_tokens,
            )

            jupiter = JupiterAPIClient(self.settings)
            rpc = SolanaRPCClient(self.settings)

            await jupiter.connect()
            await rpc.connect()

            try:
                response = await jupiter.sell_token(
                    mint_address=mint_address,
                    amount_tokens=amount_tokens,
                    slippage_bps=slippage_bps,
                    rpc_client=rpc,
                )

                if response.success:
                    logger.info(
                        "jupiter_sell_succeeded",
                        mint=mint_address[:8],
                        signature=response.data.get("signature") if response.data else None,
                    )
                    return BagsAPIResponse(
                        success=True,
                        data=response.data,
                    )
                else:
                    return BagsAPIResponse(
                        success=False,
                        error=f"Jupiter fallback failed: {response.error}",
                    )
            finally:
                await jupiter.close()
                await rpc.close()

        except Exception as e:
            logger.error("jupiter_sell_fallback_error", error=str(e))
            return BagsAPIResponse(
                success=False,
                error=f"Jupiter fallback error: {str(e)}",
            )

    async def health_check(self) -> bool:
        """Check if API is reachable by checking rate limit status."""
        # Note: Bags.fm API doesn't have a dedicated /health endpoint
        # We check connectivity by verifying we can reach the API
        # and checking our rate limit headers
        if self._client is None:
            return False

        # If we have recent successful rate limit data, API is healthy
        if self._rate_limit_remaining > 0:
            return True

        # Try a lightweight request to verify connectivity
        try:
            response = await self._client.get("/", timeout=5.0)
            return response.status_code < 500
        except Exception:
            return False
