"""
Tests for the rate limiter service.
"""

import asyncio
import pytest
from bags_sniper.services.rate_limiter import (
    RateLimiter,
    RateLimitService,
    RateLimitConfig,
    RATE_LIMIT_CONFIGS,
)


class TestRateLimiterBasics:
    """Test basic rate limiter functionality."""

    @pytest.fixture
    async def rate_limiter(self):
        """Create a rate limiter for testing."""
        limiter = RateLimiter()
        await limiter.initialize()
        return limiter

    async def test_initialization(self, rate_limiter):
        """Test rate limiter initializes all services."""
        status = rate_limiter.get_status()
        assert len(status) == len(RateLimitService)
        for service in RateLimitService:
            assert service.value in status

    async def test_acquire_returns_zero_when_tokens_available(self, rate_limiter):
        """Test acquire returns 0 when tokens are available."""
        wait_time = await rate_limiter.acquire(RateLimitService.HELIUS)
        assert wait_time == 0

    async def test_report_success_resets_consecutive_429s(self, rate_limiter):
        """Test that reporting success resets the 429 counter."""
        # Simulate rate limit
        await rate_limiter.report_rate_limit(RateLimitService.HELIUS)
        status = rate_limiter.get_status()
        assert status["helius"]["consecutive_429s"] == 1

        # Report success
        await rate_limiter.report_success(RateLimitService.HELIUS)
        status = rate_limiter.get_status()
        assert status["helius"]["consecutive_429s"] == 0

    async def test_report_rate_limit_increments_counter(self, rate_limiter):
        """Test that reporting 429 increments the counter."""
        await rate_limiter.report_rate_limit(RateLimitService.JITO)
        status = rate_limiter.get_status()
        assert status["jito"]["consecutive_429s"] == 1

        await rate_limiter.report_rate_limit(RateLimitService.JITO)
        status = rate_limiter.get_status()
        assert status["jito"]["consecutive_429s"] == 2

    async def test_backoff_increases_with_consecutive_429s(self, rate_limiter):
        """Test that backoff time increases exponentially."""
        backoff1 = await rate_limiter.report_rate_limit(RateLimitService.DEEPSEEK)
        backoff2 = await rate_limiter.report_rate_limit(RateLimitService.DEEPSEEK)

        # Backoff should roughly double (with some jitter)
        assert backoff2 > backoff1


class TestRateLimitConfigs:
    """Test rate limit configurations."""

    def test_all_services_have_configs(self):
        """Test all services have configurations."""
        for service in RateLimitService:
            assert service in RATE_LIMIT_CONFIGS

    def test_bags_fm_config_matches_docs(self):
        """Test Bags.fm config matches documentation (1000 req/hour)."""
        config = RATE_LIMIT_CONFIGS[RateLimitService.BAGS_FM]
        assert config.requests_per_hour == 1000
        # ~0.28 RPS for 1000/hour
        assert 0.2 < config.requests_per_second < 0.35

    def test_telegram_config_matches_docs(self):
        """Test Telegram config matches documentation (30 msg/sec)."""
        config = RATE_LIMIT_CONFIGS[RateLimitService.TELEGRAM]
        assert config.requests_per_second == 30.0

    def test_jito_config_matches_docs(self):
        """Test Jito config matches documentation (5 RPS)."""
        config = RATE_LIMIT_CONFIGS[RateLimitService.JITO]
        assert config.requests_per_second == 5.0


class TestTelegramPerChatRateLimiting:
    """Test Telegram per-chat rate limiting."""

    @pytest.fixture
    async def rate_limiter(self):
        """Create a rate limiter for testing."""
        limiter = RateLimiter()
        await limiter.initialize()
        return limiter

    async def test_different_chats_have_separate_limits(self, rate_limiter):
        """Test that different chat IDs have separate rate limits."""
        chat_id_1 = "123456"
        chat_id_2 = "789012"

        # Acquire for chat 1
        await rate_limiter.acquire(RateLimitService.TELEGRAM, chat_id=chat_id_1)

        # Report rate limit for chat 1
        await rate_limiter.report_rate_limit(
            RateLimitService.TELEGRAM, chat_id=chat_id_1
        )

        # Chat 2 should not be affected
        state_2 = rate_limiter._get_state(RateLimitService.TELEGRAM, chat_id=chat_id_2)
        assert state_2.consecutive_429s == 0
