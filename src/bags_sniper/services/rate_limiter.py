"""
Rate Limiter Service for API Rate Limit Management.

Handles rate limiting for all external APIs:
- Bags.fm API: 1,000 requests/hour (~0.28 RPS)
- Helius RPC: Variable by plan, 429 on limit
- Telegram Bot API: 30 msg/sec global, 1 msg/sec per chat
- Jito Block Engine: 5 RPS default
- Deepseek API: No hard limits, dynamic throttling

Sources:
- Bags.fm: https://docs.bags.fm/principles/rate-limits
- Helius: https://www.helius.dev/docs/billing/plans-and-rate-limits
- Telegram: https://core.telegram.org/bots/faq
- Jito: https://jito-labs.gitbook.io/mev/searcher-resources/json-rpc-api-reference/rate-limits
- Deepseek: https://api-docs.deepseek.com/quick_start/rate_limit
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar("T")


class RateLimitService(str, Enum):
    """Supported services with rate limiting."""

    BAGS_FM = "bags_fm"
    HELIUS = "helius"
    TELEGRAM = "telegram"
    JITO = "jito"
    DEEPSEEK = "deepseek"


@dataclass
class RateLimitConfig:
    """Configuration for a rate-limited service."""

    service: RateLimitService
    requests_per_second: float  # Max RPS (0 = unlimited)
    requests_per_hour: int  # Max per hour (0 = unlimited)
    burst_limit: int  # Max burst requests
    min_interval_seconds: float  # Minimum time between requests
    max_retries: int = 5
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    jitter_factor: float = 0.1  # Random jitter to prevent thundering herd


# Pre-configured rate limits based on documentation
RATE_LIMIT_CONFIGS = {
    RateLimitService.BAGS_FM: RateLimitConfig(
        service=RateLimitService.BAGS_FM,
        requests_per_second=0.28,  # ~1000/hour
        requests_per_hour=1000,
        burst_limit=5,
        min_interval_seconds=3.6,  # 1000/hour = 1 per 3.6s
        max_retries=3,
        base_backoff_seconds=5.0,
    ),
    RateLimitService.HELIUS: RateLimitConfig(
        service=RateLimitService.HELIUS,
        requests_per_second=10.0,  # Conservative default
        requests_per_hour=0,  # Credit-based, not hourly
        burst_limit=20,
        min_interval_seconds=0.1,
        max_retries=5,
        base_backoff_seconds=1.0,
    ),
    RateLimitService.TELEGRAM: RateLimitConfig(
        service=RateLimitService.TELEGRAM,
        requests_per_second=30.0,  # Global limit
        requests_per_hour=0,
        burst_limit=30,
        min_interval_seconds=0.034,  # ~30/sec
        max_retries=5,
        base_backoff_seconds=1.0,
    ),
    RateLimitService.JITO: RateLimitConfig(
        service=RateLimitService.JITO,
        requests_per_second=5.0,  # Default limit
        requests_per_hour=0,
        burst_limit=5,
        min_interval_seconds=0.2,  # 5/sec
        max_retries=3,
        base_backoff_seconds=0.5,
    ),
    RateLimitService.DEEPSEEK: RateLimitConfig(
        service=RateLimitService.DEEPSEEK,
        requests_per_second=0,  # No hard limit
        requests_per_hour=0,
        burst_limit=10,
        min_interval_seconds=0.5,  # Be conservative
        max_retries=3,
        base_backoff_seconds=2.0,
        max_backoff_seconds=30.0,  # Queuing can take time
    ),
}


@dataclass
class RateLimitState:
    """Track rate limit state for a service."""

    tokens: float = 0.0  # Token bucket tokens
    last_request_time: float = 0.0
    hourly_requests: int = 0
    hourly_reset_time: float = 0.0
    consecutive_429s: int = 0
    backoff_until: float = 0.0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded and retries exhausted."""

    def __init__(
        self,
        service: RateLimitService,
        retry_after: Optional[float] = None,
        message: str = "Rate limit exceeded",
    ):
        self.service = service
        self.retry_after = retry_after
        super().__init__(f"{service.value}: {message}")


@dataclass
class RateLimitResponse:
    """Response from a rate-limited request."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    retries_used: int = 0
    wait_time_total: float = 0.0


class RateLimiter:
    """
    Universal rate limiter for all external APIs.

    Features:
    - Token bucket algorithm for smooth rate limiting
    - Hourly quota tracking (for Bags.fm)
    - Exponential backoff with jitter on 429 errors
    - Automatic retry with configurable limits
    - Per-chat rate limiting for Telegram
    - Header-based reset time extraction
    """

    def __init__(self):
        self._states: dict[RateLimitService, RateLimitState] = {}
        self._telegram_chat_states: dict[str, RateLimitState] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize rate limiter states."""
        for service in RateLimitService:
            self._states[service] = RateLimitState()
            # Initialize token bucket
            config = RATE_LIMIT_CONFIGS[service]
            self._states[service].tokens = float(config.burst_limit)
            self._states[service].hourly_reset_time = time.time() + 3600

        self._initialized = True
        logger.info("rate_limiter_initialized", services=len(self._states))

    def _get_config(self, service: RateLimitService) -> RateLimitConfig:
        """Get rate limit config for a service."""
        return RATE_LIMIT_CONFIGS[service]

    def _get_state(
        self, service: RateLimitService, chat_id: Optional[str] = None
    ) -> RateLimitState:
        """Get state for a service, with optional per-chat tracking for Telegram."""
        if service == RateLimitService.TELEGRAM and chat_id:
            if chat_id not in self._telegram_chat_states:
                self._telegram_chat_states[chat_id] = RateLimitState()
                self._telegram_chat_states[chat_id].tokens = 1.0  # 1 msg/sec per chat
            return self._telegram_chat_states[chat_id]
        return self._states[service]

    async def acquire(
        self,
        service: RateLimitService,
        chat_id: Optional[str] = None,
    ) -> float:
        """
        Acquire permission to make a request.

        Returns the time to wait before proceeding (0 if immediate).
        Raises RateLimitError if in backoff period.
        """
        if not self._initialized:
            await self.initialize()

        config = self._get_config(service)
        state = self._get_state(service, chat_id)

        async with state.lock:
            now = time.time()

            # Check if in backoff period
            if now < state.backoff_until:
                wait_time = state.backoff_until - now
                logger.debug(
                    "rate_limit_backoff",
                    service=service.value,
                    wait_seconds=wait_time,
                )
                return wait_time

            # Check hourly quota (for Bags.fm)
            if config.requests_per_hour > 0:
                if now >= state.hourly_reset_time:
                    state.hourly_requests = 0
                    state.hourly_reset_time = now + 3600
                    logger.debug("hourly_quota_reset", service=service.value)

                if state.hourly_requests >= config.requests_per_hour:
                    wait_time = state.hourly_reset_time - now
                    logger.warning(
                        "hourly_quota_exceeded",
                        service=service.value,
                        reset_in_seconds=wait_time,
                    )
                    return wait_time

            # Token bucket refill
            elapsed = now - state.last_request_time
            if config.requests_per_second > 0:
                tokens_to_add = elapsed * config.requests_per_second
                state.tokens = min(config.burst_limit, state.tokens + tokens_to_add)

            # Check if we have tokens
            if state.tokens < 1.0:
                # Calculate wait time for 1 token
                if config.requests_per_second > 0:
                    wait_time = (1.0 - state.tokens) / config.requests_per_second
                else:
                    wait_time = config.min_interval_seconds
                return wait_time

            # Consume token
            state.tokens -= 1.0
            state.last_request_time = now
            state.hourly_requests += 1

            return 0.0

    async def report_success(
        self,
        service: RateLimitService,
        chat_id: Optional[str] = None,
    ):
        """Report a successful request (reset consecutive 429 counter)."""
        state = self._get_state(service, chat_id)
        async with state.lock:
            state.consecutive_429s = 0

    async def report_rate_limit(
        self,
        service: RateLimitService,
        retry_after: Optional[float] = None,
        headers: Optional[dict[str, str]] = None,
        chat_id: Optional[str] = None,
    ) -> float:
        """
        Report a 429 rate limit response.

        Args:
            service: The service that returned 429
            retry_after: Explicit retry-after value (from response)
            headers: Response headers (for X-RateLimit-Reset, etc.)
            chat_id: Optional chat ID for Telegram per-chat limiting

        Returns:
            Calculated backoff time in seconds
        """
        config = self._get_config(service)
        state = self._get_state(service, chat_id)

        async with state.lock:
            state.consecutive_429s += 1
            now = time.time()

            # Extract retry-after from various sources
            backoff_time = self._calculate_backoff(
                config, state.consecutive_429s, retry_after, headers
            )

            state.backoff_until = now + backoff_time

            logger.warning(
                "rate_limit_hit",
                service=service.value,
                consecutive_429s=state.consecutive_429s,
                backoff_seconds=backoff_time,
                retry_after_source="header" if retry_after or headers else "calculated",
            )

            return backoff_time

    def _calculate_backoff(
        self,
        config: RateLimitConfig,
        consecutive_429s: int,
        retry_after: Optional[float],
        headers: Optional[dict[str, str]],
    ) -> float:
        """Calculate backoff time with exponential increase and jitter."""
        # Priority 1: Explicit retry-after
        if retry_after is not None:
            return retry_after

        # Priority 2: Headers
        if headers:
            # Telegram adaptive_retry (in milliseconds)
            if "adaptive_retry" in headers:
                return float(headers["adaptive_retry"]) / 1000.0

            # X-RateLimit-Reset (Unix timestamp)
            if "X-RateLimit-Reset" in headers:
                reset_time = float(headers["X-RateLimit-Reset"])
                return max(0, reset_time - time.time())

            # Retry-After header
            if "Retry-After" in headers:
                return float(headers["Retry-After"])

        # Priority 3: Exponential backoff
        import random

        base = config.base_backoff_seconds
        backoff = base * (2 ** (consecutive_429s - 1))
        backoff = min(backoff, config.max_backoff_seconds)

        # Add jitter
        jitter = backoff * config.jitter_factor * random.random()
        return backoff + jitter

    async def execute_with_retry(
        self,
        service: RateLimitService,
        func: Callable[[], T],
        chat_id: Optional[str] = None,
        extract_retry_after: Optional[Callable[[Any], Optional[float]]] = None,
        extract_headers: Optional[Callable[[Any], Optional[dict]]] = None,
        is_rate_limit_error: Optional[Callable[[Exception], bool]] = None,
    ) -> RateLimitResponse:
        """
        Execute a function with automatic rate limiting and retry.

        Args:
            service: The service being called
            func: Async function to execute
            chat_id: Optional chat ID for Telegram
            extract_retry_after: Function to extract retry-after from response
            extract_headers: Function to extract headers from response
            is_rate_limit_error: Function to check if exception is rate limit

        Returns:
            RateLimitResponse with result or error
        """
        if not self._initialized:
            await self.initialize()

        config = self._get_config(service)
        total_wait_time = 0.0
        last_error = None

        for attempt in range(config.max_retries + 1):
            # Acquire permission (wait if needed)
            wait_time = await self.acquire(service, chat_id)
            if wait_time > 0:
                logger.debug(
                    "rate_limit_waiting",
                    service=service.value,
                    wait_seconds=wait_time,
                    attempt=attempt,
                )
                await asyncio.sleep(wait_time)
                total_wait_time += wait_time

            try:
                # Execute the function
                result = await func()

                # Report success
                await self.report_success(service, chat_id)

                return RateLimitResponse(
                    success=True,
                    data=result,
                    retries_used=attempt,
                    wait_time_total=total_wait_time,
                )

            except Exception as e:
                last_error = e

                # Check if this is a rate limit error
                is_429 = False
                retry_after = None
                headers = None

                if is_rate_limit_error and is_rate_limit_error(e):
                    is_429 = True
                elif hasattr(e, "status_code") and e.status_code == 429:
                    is_429 = True
                elif "429" in str(e) or "rate limit" in str(e).lower():
                    is_429 = True

                if is_429:
                    # Extract retry info if possible
                    if extract_retry_after:
                        retry_after = extract_retry_after(e)
                    if extract_headers:
                        headers = extract_headers(e)

                    backoff = await self.report_rate_limit(
                        service, retry_after, headers, chat_id
                    )

                    if attempt < config.max_retries:
                        logger.info(
                            "rate_limit_retry",
                            service=service.value,
                            attempt=attempt + 1,
                            max_retries=config.max_retries,
                            backoff_seconds=backoff,
                        )
                        await asyncio.sleep(backoff)
                        total_wait_time += backoff
                        continue
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(
                        "request_failed",
                        service=service.value,
                        error=str(e),
                        attempt=attempt,
                    )
                    break

        # All retries exhausted
        return RateLimitResponse(
            success=False,
            error=str(last_error) if last_error else "Max retries exceeded",
            retries_used=config.max_retries,
            wait_time_total=total_wait_time,
        )

    def get_status(self) -> dict[str, Any]:
        """Get current rate limiter status for all services."""
        now = time.time()
        status = {}

        for service, state in self._states.items():
            config = self._get_config(service)
            status[service.value] = {
                "tokens_available": round(state.tokens, 2),
                "burst_limit": config.burst_limit,
                "hourly_requests": state.hourly_requests,
                "hourly_limit": config.requests_per_hour,
                "hourly_reset_in": (
                    round(state.hourly_reset_time - now, 0)
                    if config.requests_per_hour > 0
                    else None
                ),
                "in_backoff": now < state.backoff_until,
                "backoff_remaining": (
                    round(state.backoff_until - now, 1)
                    if now < state.backoff_until
                    else 0
                ),
                "consecutive_429s": state.consecutive_429s,
            }

        return status


# Global singleton instance
_rate_limiter: Optional[RateLimiter] = None


async def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        await _rate_limiter.initialize()
    return _rate_limiter
