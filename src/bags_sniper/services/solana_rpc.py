"""
Solana RPC client with multi-endpoint failover.
Handles all direct Solana blockchain interactions.

Rate Limits (Helius):
- Credit-based system, varies by plan
- 429 errors when exceeded
- Automatic retry with exponential backoff
"""

import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import structlog
from solders.pubkey import Pubkey
from solders.signature import Signature

from bags_sniper.core.config import Settings
from bags_sniper.services.rate_limiter import (
    RateLimiter,
    RateLimitService,
    get_rate_limiter,
)

logger = structlog.get_logger()


@dataclass
class RPCEndpoint:
    """Represents an RPC endpoint with health tracking."""

    url: str
    is_primary: bool = False
    consecutive_failures: int = 0
    last_success: float = field(default_factory=time.monotonic)
    last_failure: float = 0.0
    avg_latency_ms: float = 0.0
    total_requests: int = 0

    @property
    def is_healthy(self) -> bool:
        """Endpoint is healthy if recent failures < 3."""
        return self.consecutive_failures < 3

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining in cooldown after failures."""
        if self.consecutive_failures == 0:
            return 0
        # Exponential backoff: 2^failures seconds, max 60s
        cooldown = min(2**self.consecutive_failures, 60)
        elapsed = time.monotonic() - self.last_failure
        return max(0, cooldown - elapsed)

    def record_success(self, latency_ms: float):
        """Record successful request."""
        self.consecutive_failures = 0
        self.last_success = time.monotonic()
        self.total_requests += 1
        # Exponential moving average for latency
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms

    def record_failure(self):
        """Record failed request."""
        self.consecutive_failures += 1
        self.last_failure = time.monotonic()


@dataclass
class RPCResponse:
    """Standardized RPC response."""

    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    endpoint_used: str = ""


class SolanaRPCClient:
    """
    Multi-endpoint Solana RPC client with automatic failover.
    Uses Helius as primary with public RPC as backup.
    Includes rate limiting for Helius API.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None
        self._endpoints: list[RPCEndpoint] = []
        self._rate_limiter: Optional[RateLimiter] = None
        self._setup_endpoints()

    def _setup_endpoints(self):
        """Initialize RPC endpoints with Helius as primary."""
        # Primary: Helius
        helius_url = (
            f"{self.settings.helius_rpc_url}?"
            f"api-key={self.settings.helius_api_key.get_secret_value()}"
        )
        self._endpoints.append(RPCEndpoint(url=helius_url, is_primary=True))

        # Backups
        for url in self.settings.backup_rpc_urls:
            self._endpoints.append(RPCEndpoint(url=url, is_primary=False))

        logger.info(
            "rpc_endpoints_configured",
            primary=self.settings.helius_rpc_url,
            backup_count=len(self.settings.backup_rpc_urls),
        )

    async def __aenter__(self) -> "SolanaRPCClient":
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def connect(self):
        """Initialize HTTP client and rate limiter."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=5.0),
            headers={"Content-Type": "application/json"},
        )
        self._rate_limiter = await get_rate_limiter()

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_available_endpoints(self) -> list[RPCEndpoint]:
        """Get endpoints sorted by preference (healthy primary first)."""
        available = [
            ep for ep in self._endpoints if ep.is_healthy or ep.cooldown_remaining == 0
        ]

        # Sort: primary first, then by latency
        available.sort(key=lambda ep: (not ep.is_primary, ep.avg_latency_ms))
        return available

    async def _rpc_call(
        self,
        method: str,
        params: Any,  # Can be list (standard RPC) or dict (DAS API)
        retries: int = 3,
    ) -> RPCResponse:
        """
        Make RPC call with automatic failover and rate limiting.

        Handles:
        - Multi-endpoint failover
        - Rate limiting (429 responses)
        - Exponential backoff on failures
        - Both standard RPC (array params) and DAS API (object params)
        """
        if not self._client:
            await self.connect()

        # Apply rate limiting for Helius
        if self._rate_limiter:
            wait_time = await self._rate_limiter.acquire(RateLimitService.HELIUS)
            if wait_time > 0:
                logger.debug(
                    "helius_rate_limit_wait",
                    wait_seconds=wait_time,
                    method=method,
                )
                await asyncio.sleep(wait_time)

        available = self._get_available_endpoints()
        if not available:
            # All endpoints failed, reset and try anyway
            for ep in self._endpoints:
                ep.consecutive_failures = 0
            available = self._endpoints

        last_error = None
        for attempt in range(retries):
            for endpoint in available:
                if endpoint.cooldown_remaining > 0:
                    continue

                start_time = time.monotonic()
                try:
                    response = await self._client.post(
                        endpoint.url,
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": method,
                            "params": params,
                        },
                    )
                    latency_ms = (time.monotonic() - start_time) * 1000

                    # Handle rate limit (429)
                    if response.status_code == 429:
                        endpoint.record_failure()
                        if self._rate_limiter and endpoint.is_primary:
                            # Extract retry-after from headers
                            retry_after = None
                            headers_dict = dict(response.headers)
                            if "Retry-After" in headers_dict:
                                retry_after = float(headers_dict["Retry-After"])

                            backoff = await self._rate_limiter.report_rate_limit(
                                RateLimitService.HELIUS,
                                retry_after=retry_after,
                                headers=headers_dict,
                            )
                            logger.warning(
                                "helius_rate_limited",
                                method=method,
                                backoff_seconds=backoff,
                                attempt=attempt,
                            )
                            # Wait before next attempt
                            await asyncio.sleep(backoff)
                        last_error = "Rate limited (429)"
                        continue

                    if response.status_code != 200:
                        endpoint.record_failure()
                        last_error = f"HTTP {response.status_code}"
                        continue

                    data = response.json()
                    if "error" in data:
                        endpoint.record_failure()
                        last_error = data["error"].get("message", str(data["error"]))
                        continue

                    endpoint.record_success(latency_ms)

                    # Report success to rate limiter
                    if self._rate_limiter and endpoint.is_primary:
                        await self._rate_limiter.report_success(RateLimitService.HELIUS)

                    return RPCResponse(
                        success=True,
                        result=data.get("result"),
                        latency_ms=latency_ms,
                        endpoint_used=endpoint.url.split("?")[0],  # Hide API key
                    )

                except httpx.TimeoutException:
                    endpoint.record_failure()
                    last_error = "Timeout"
                    logger.warning(
                        "rpc_timeout",
                        endpoint=endpoint.url.split("?")[0],
                        method=method,
                    )
                except httpx.RequestError as e:
                    endpoint.record_failure()
                    last_error = str(e)
                    logger.error(
                        "rpc_request_error",
                        endpoint=endpoint.url.split("?")[0],
                        error=str(e),
                    )

            # Brief delay between retry rounds
            await asyncio.sleep(0.5 * (attempt + 1))

        return RPCResponse(
            success=False,
            error=last_error or "All endpoints failed",
        )

    # =========================================================================
    # Account Methods
    # =========================================================================

    async def get_account_info(
        self,
        pubkey: str | Pubkey,
        encoding: str = "base64",
    ) -> RPCResponse:
        """Get account information."""
        address = str(pubkey)
        return await self._rpc_call(
            "getAccountInfo",
            [address, {"encoding": encoding}],
        )

    async def get_balance(self, pubkey: str | Pubkey) -> Optional[int]:
        """Get SOL balance in lamports."""
        address = str(pubkey)
        response = await self._rpc_call("getBalance", [address])
        if response.success and response.result:
            return response.result.get("value", 0)
        return None

    async def get_token_accounts_by_owner(
        self,
        owner: str | Pubkey,
        mint: Optional[str | Pubkey] = None,
        program_id: Optional[str] = None,
    ) -> RPCResponse:
        """Get token accounts owned by a wallet."""
        address = str(owner)
        filters = {}
        if mint:
            filters["mint"] = str(mint)
        if program_id:
            filters["programId"] = program_id

        return await self._rpc_call(
            "getTokenAccountsByOwner",
            [address, filters, {"encoding": "jsonParsed"}],
        )

    async def get_token_balance(
        self,
        owner: str | Pubkey,
        mint: str | Pubkey,
    ) -> Optional[int]:
        """Get token balance for a specific mint."""
        response = await self.get_token_accounts_by_owner(owner, mint=mint)
        if response.success and response.result:
            accounts = response.result.get("value", [])
            if accounts:
                return int(
                    accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"][
                        "amount"
                    ]
                )
        return None

    # =========================================================================
    # Transaction Methods
    # =========================================================================

    async def get_transaction(
        self,
        signature: str | Signature,
        max_supported_version: int = 0,
    ) -> RPCResponse:
        """Get transaction details."""
        sig = str(signature)
        return await self._rpc_call(
            "getTransaction",
            [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": max_supported_version}],
        )

    async def get_signatures_for_address(
        self,
        address: str | Pubkey,
        limit: int = 100,
        before: Optional[str] = None,
    ) -> RPCResponse:
        """Get recent transaction signatures for an address."""
        addr = str(address)
        params: dict[str, Any] = {"limit": limit}
        if before:
            params["before"] = before

        return await self._rpc_call(
            "getSignaturesForAddress",
            [addr, params],
        )

    async def send_transaction(
        self,
        transaction: bytes,
        skip_preflight: bool = False,
    ) -> RPCResponse:
        """Send a signed transaction."""
        tx_base64 = base64.b64encode(transaction).decode()
        return await self._rpc_call(
            "sendTransaction",
            [
                tx_base64,
                {
                    "encoding": "base64",
                    "skipPreflight": skip_preflight,
                    "preflightCommitment": "confirmed",
                },
            ],
        )

    async def confirm_transaction(
        self,
        signature: str,
        timeout_seconds: float = 30.0,
    ) -> bool:
        """Wait for transaction confirmation."""
        start = time.monotonic()
        while time.monotonic() - start < timeout_seconds:
            response = await self._rpc_call(
                "getSignatureStatuses",
                [[signature]],
            )
            if response.success and response.result:
                statuses = response.result.get("value", [])
                if statuses and statuses[0]:
                    status = statuses[0]
                    if status.get("confirmationStatus") in ["confirmed", "finalized"]:
                        return status.get("err") is None
            await asyncio.sleep(0.5)
        return False

    # =========================================================================
    # Block Methods
    # =========================================================================

    async def get_slot(self) -> Optional[int]:
        """Get current slot."""
        response = await self._rpc_call("getSlot", [])
        if response.success:
            return response.result
        return None

    async def get_recent_blockhash(self) -> Optional[str]:
        """Get recent blockhash for transaction building."""
        response = await self._rpc_call(
            "getLatestBlockhash",
            [{"commitment": "confirmed"}],
        )
        if response.success and response.result:
            return response.result.get("value", {}).get("blockhash")
        return None

    # =========================================================================
    # Monitoring Methods
    # =========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check health of all endpoints."""
        results = {}
        for endpoint in self._endpoints:
            start = time.monotonic()
            try:
                response = await self._client.post(
                    endpoint.url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "getHealth", "params": []},
                    timeout=5.0,
                )
                latency = (time.monotonic() - start) * 1000
                results[endpoint.url.split("?")[0]] = {
                    "healthy": response.status_code == 200,
                    "latency_ms": round(latency, 2),
                    "is_primary": endpoint.is_primary,
                }
            except Exception as e:
                results[endpoint.url.split("?")[0]] = {
                    "healthy": False,
                    "error": str(e),
                    "is_primary": endpoint.is_primary,
                }
        return results

    async def get_program_accounts(
        self,
        program_id: str,
        filters: Optional[list[dict]] = None,
        encoding: str = "base64",
    ) -> RPCResponse:
        """
        Get all accounts owned by a program.

        Useful for finding tokens created by a specific program.
        """
        config: dict[str, Any] = {"encoding": encoding}
        if filters:
            config["filters"] = filters

        return await self._rpc_call("getProgramAccounts", [program_id, config])

    async def get_asset(self, mint_address: str) -> RPCResponse:
        """
        Get asset metadata using Helius DAS API.

        This is more reliable than manual Metaplex PDA derivation.
        Returns token name, symbol, and other metadata.

        Docs: https://www.helius.dev/docs/das-api
        """
        return await self._rpc_call(
            "getAsset",
            {"id": mint_address},  # DAS API uses object params, not array
        )

    def get_endpoint_stats(self) -> list[dict[str, Any]]:
        """Get statistics for all endpoints."""
        return [
            {
                "url": ep.url.split("?")[0],
                "is_primary": ep.is_primary,
                "is_healthy": ep.is_healthy,
                "consecutive_failures": ep.consecutive_failures,
                "avg_latency_ms": round(ep.avg_latency_ms, 2),
                "total_requests": ep.total_requests,
            }
            for ep in self._endpoints
        ]
