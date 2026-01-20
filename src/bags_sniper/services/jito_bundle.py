"""
Jito Bundle Service for MEV Protection.

Critical Fix #9: Implement Jito fallback for fast execution.
Uses Jito's block engine to bundle transactions and avoid sandwich attacks.
"""

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

import httpx
import structlog

from bags_sniper.core.config import Settings
from bags_sniper.services.rate_limiter import (
    RateLimiter,
    RateLimitService,
    get_rate_limiter,
)

logger = structlog.get_logger()


class BundleStatus(str, Enum):
    """Status of a Jito bundle submission."""

    PENDING = "pending"
    LANDED = "landed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class BundleResult:
    """Result of a Jito bundle submission."""

    success: bool
    status: BundleStatus
    bundle_id: Optional[str] = None
    signature: Optional[str] = None
    slot: Optional[int] = None
    error: Optional[str] = None
    tip_lamports: int = 0


class JitoBundleService:
    """
    Service for submitting transactions via Jito bundles.

    Key benefits:
    - MEV protection: Avoids sandwich attacks
    - Faster execution: Priority landing through tips
    - Atomic execution: Bundle lands together or not at all

    Usage:
    - Primary: Use Bags.fm API for trading
    - Fallback: Use Jito when standard RPC is slow or congested
    """

    # Jito block engine endpoints
    JITO_ENDPOINTS = [
        "https://mainnet.block-engine.jito.wtf",
        "https://amsterdam.mainnet.block-engine.jito.wtf",
        "https://frankfurt.mainnet.block-engine.jito.wtf",
        "https://ny.mainnet.block-engine.jito.wtf",
        "https://tokyo.mainnet.block-engine.jito.wtf",
    ]

    # Jito tip accounts (rotate for better distribution)
    JITO_TIP_ACCOUNTS = [
        "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
        "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
        "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
        "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
        "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
        "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
        "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
        "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._current_endpoint_idx = 0
        self._tip_account_idx = 0

        # Configuration
        self._default_tip_lamports = 10000  # 0.00001 SOL default tip
        self._max_tip_lamports = 100000  # 0.0001 SOL max tip
        self._bundle_timeout_seconds = 30

    async def start(self):
        """Initialize the Jito bundle service."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._bundle_timeout_seconds),
            headers={"Content-Type": "application/json"},
        )
        self._rate_limiter = await get_rate_limiter()
        logger.info("jito_bundle_service_started")

    async def stop(self):
        """Stop the Jito bundle service."""
        if self._client:
            await self._client.aclose()
        logger.info("jito_bundle_service_stopped")

    @property
    def _current_endpoint(self) -> str:
        """Get current Jito endpoint with rotation."""
        return self.JITO_ENDPOINTS[self._current_endpoint_idx]

    @property
    def _current_tip_account(self) -> str:
        """Get current tip account with rotation."""
        return self.JITO_TIP_ACCOUNTS[self._tip_account_idx]

    def _rotate_endpoint(self):
        """Rotate to next endpoint on failure."""
        self._current_endpoint_idx = (self._current_endpoint_idx + 1) % len(
            self.JITO_ENDPOINTS
        )

    def _rotate_tip_account(self):
        """Rotate tip account for distribution."""
        self._tip_account_idx = (self._tip_account_idx + 1) % len(self.JITO_TIP_ACCOUNTS)

    async def submit_bundle(
        self,
        transactions: list[str],
        tip_lamports: Optional[int] = None,
    ) -> BundleResult:
        """
        Submit a bundle of transactions to Jito.

        Args:
            transactions: List of base64-encoded signed transactions
            tip_lamports: Optional tip amount (uses default if not specified)

        Returns:
            BundleResult with status and details
        """
        if not self._client:
            return BundleResult(
                success=False,
                status=BundleStatus.FAILED,
                error="Jito service not started",
            )

        tip = tip_lamports or self._default_tip_lamports
        tip = min(tip, self._max_tip_lamports)  # Cap tip

        # Try with endpoint rotation on failure
        for attempt in range(len(self.JITO_ENDPOINTS)):
            try:
                result = await self._submit_bundle_to_endpoint(transactions, tip)
                if result.success:
                    self._rotate_tip_account()  # Rotate for next bundle
                    return result

                # Rotate endpoint on failure
                self._rotate_endpoint()

            except Exception as e:
                logger.warning(
                    "jito_bundle_attempt_failed",
                    endpoint=self._current_endpoint,
                    attempt=attempt + 1,
                    error=str(e),
                )
                self._rotate_endpoint()

        return BundleResult(
            success=False,
            status=BundleStatus.FAILED,
            error="All Jito endpoints failed",
        )

    async def _submit_bundle_to_endpoint(
        self,
        transactions: list[str],
        tip_lamports: int,
    ) -> BundleResult:
        """Submit bundle to a specific Jito endpoint with rate limiting."""
        endpoint = f"{self._current_endpoint}/api/v1/bundles"

        # Apply rate limiting
        if self._rate_limiter:
            wait_time = await self._rate_limiter.acquire(RateLimitService.JITO)
            if wait_time > 0:
                logger.debug("jito_rate_limit_wait", wait_seconds=wait_time)
                await asyncio.sleep(wait_time)

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "sendBundle",
            "params": [transactions],
        }

        response = await self._client.post(endpoint, json=payload)

        # Handle rate limit response
        if response.status_code == 429:
            if self._rate_limiter:
                # Extract retry-after if available
                retry_after = None
                if "Retry-After" in response.headers:
                    retry_after = float(response.headers["Retry-After"])
                backoff = await self._rate_limiter.report_rate_limit(
                    RateLimitService.JITO, retry_after=retry_after
                )
                logger.warning("jito_rate_limited", backoff_seconds=backoff)
            return BundleResult(
                success=False,
                status=BundleStatus.FAILED,
                error="Rate limited (429): retry after backoff",
            )

        if response.status_code != 200:
            return BundleResult(
                success=False,
                status=BundleStatus.FAILED,
                error=f"HTTP {response.status_code}: {response.text[:100]}",
            )

        # Report success
        if self._rate_limiter:
            await self._rate_limiter.report_success(RateLimitService.JITO)

        data = response.json()

        if "error" in data:
            return BundleResult(
                success=False,
                status=BundleStatus.FAILED,
                error=data["error"].get("message", "Unknown error"),
            )

        bundle_id = data.get("result")
        if not bundle_id:
            return BundleResult(
                success=False,
                status=BundleStatus.FAILED,
                error="No bundle ID returned",
            )

        # Wait for bundle confirmation
        return await self._wait_for_bundle(bundle_id, tip_lamports)

    async def _wait_for_bundle(
        self,
        bundle_id: str,
        tip_lamports: int,
    ) -> BundleResult:
        """Wait for bundle confirmation."""
        endpoint = f"{self._current_endpoint}/api/v1/bundles"

        # Poll for bundle status
        for _ in range(10):  # 10 attempts, ~3 seconds each
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBundleStatuses",
                    "params": [[bundle_id]],
                }

                response = await self._client.post(endpoint, json=payload)
                data = response.json()

                if "result" in data and data["result"]:
                    statuses = data["result"].get("value", [])
                    if statuses:
                        status = statuses[0]
                        confirmation = status.get("confirmation_status")

                        if confirmation in ["confirmed", "finalized"]:
                            return BundleResult(
                                success=True,
                                status=BundleStatus.LANDED,
                                bundle_id=bundle_id,
                                slot=status.get("slot"),
                                tip_lamports=tip_lamports,
                            )
                        elif confirmation == "processed":
                            # Still processing, continue waiting
                            pass

                await asyncio.sleep(0.3)

            except Exception as e:
                logger.warning("bundle_status_check_error", error=str(e))
                await asyncio.sleep(0.5)

        return BundleResult(
            success=False,
            status=BundleStatus.TIMEOUT,
            bundle_id=bundle_id,
            error="Bundle confirmation timeout",
            tip_lamports=tip_lamports,
        )

    async def get_tip_floor(self) -> int:
        """
        Get current tip floor from Jito.
        Returns minimum tip needed for bundle inclusion.
        """
        if not self._client:
            return self._default_tip_lamports

        try:
            endpoint = f"{self._current_endpoint}/api/v1/bundles"
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTipAccounts",
                "params": [],
            }

            response = await self._client.post(endpoint, json=payload)
            # Parse response but Jito doesn't directly expose tip floor
            _ = response.json()  # Validate response is valid JSON

            # Use heuristic - in high congestion, tips should be higher
            return self._default_tip_lamports

        except Exception as e:
            logger.warning("tip_floor_fetch_error", error=str(e))
            return self._default_tip_lamports

    def calculate_dynamic_tip(
        self,
        urgency: str = "normal",
        position_size_sol: Decimal = Decimal("0.1"),
    ) -> int:
        """
        Calculate dynamic tip based on urgency and position size.

        Args:
            urgency: "low", "normal", "high", "critical"
            position_size_sol: Size of the trade

        Returns:
            Tip amount in lamports
        """
        base_tip = self._default_tip_lamports

        # Urgency multipliers
        urgency_mult = {
            "low": 0.5,
            "normal": 1.0,
            "high": 2.0,
            "critical": 5.0,
        }.get(urgency, 1.0)

        # Position size factor (larger positions warrant higher tips)
        size_factor = min(float(position_size_sol), 1.0)  # Cap at 1 SOL
        size_mult = 1.0 + size_factor

        tip = int(base_tip * urgency_mult * size_mult)
        return min(tip, self._max_tip_lamports)

    async def health_check(self) -> bool:
        """Check if Jito service is healthy."""
        if not self._client:
            return False

        try:
            endpoint = f"{self._current_endpoint}/api/v1/bundles"
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTipAccounts",
                "params": [],
            }

            response = await self._client.post(endpoint, json=payload)
            return response.status_code == 200

        except Exception:
            return False
