"""
Deepseek AI Service for Narrative Analysis.

Detects cultural moments and evaluates token narrative strength.
Runs ASYNCHRONOUSLY - never blocks the trading hot path.

Key insight: Tokens riding cultural moments (memes, events, trends)
have higher graduation rates than random tokens.

Rate Limits (from https://api-docs.deepseek.com/quick_start/rate_limit):
- No hard rate limits, dynamic throttling
- Queues requests when overloaded
- Returns 429 with backoff guidance
"""

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
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


class NarrativeCategory(str, Enum):
    """Categories of token narratives."""

    CULTURAL_MOMENT = "cultural_moment"  # Tied to current event/meme
    CELEBRITY = "celebrity"              # Celebrity/influencer related
    POLITICAL = "political"              # Political figure/event
    CRYPTO_NATIVE = "crypto_native"      # Crypto memes (pepe, doge, etc.)
    SEASONAL = "seasonal"                # Holiday/seasonal
    RANDOM = "random"                    # No clear narrative
    UNKNOWN = "unknown"                  # Could not determine


@dataclass
class NarrativeAnalysis:
    """Result of AI narrative analysis."""

    token_name: str
    token_symbol: str
    narrative_score: float  # 0-100
    category: NarrativeCategory
    cultural_relevance: float  # 0-100 how tied to current events
    virality_potential: float  # 0-100 likelihood to go viral
    reasoning: str
    keywords_detected: list[str]
    confidence: float  # 0-1 AI confidence in analysis
    analyzed_at: datetime
    analysis_time_ms: float


# Prompt templates for narrative analysis
NARRATIVE_ANALYSIS_PROMPT = """You are an expert memecoin analyst specializing in viral token detection on Solana.

TOKEN INFO:
- Name: {name}
- Symbol: {symbol}
- Description: {description}

CURRENT CONTEXT (Date: {current_date}):
{trending_context}

YOUR TASK:
Analyze if this token can MAKE MONEY based on:
1. Is it tied to something trending RIGHT NOW? (memes, news, events, celebrities)
2. Does the name/symbol have viral potential?
3. Would crypto Twitter buy this?
4. Is there urgency or FOMO potential?

SCORING GUIDE:
- 80-100: STRONG BUY - Clear viral narrative, perfect timing
- 60-79: GOOD - Decent narrative, might work
- 40-59: WEAK - Generic or poor timing
- 0-39: AVOID - No narrative or negative sentiment

Respond ONLY with JSON:
{{
    "narrative_score": <0-100>,
    "category": "<cultural_moment|celebrity|political|crypto_native|seasonal|random>",
    "cultural_relevance": <0-100>,
    "virality_potential": <0-100>,
    "reasoning": "<why would people buy this? be specific>",
    "keywords": ["<keyword1>", "<keyword2>"],
    "confidence": <0.0-1.0>
}}"""

# Default trending context when web search unavailable
DEFAULT_TRENDING_CONTEXT = """Current crypto trends:
- Memecoins on Solana are hot
- AI-related tokens gaining traction
- Political tokens around elections
- Celebrity/influencer tokens perform well
- Animal-themed coins (cats, dogs, frogs) remain popular"""


class DeepseekAIService:
    """
    Async service for AI-powered narrative analysis.

    Design principles:
    - NEVER blocks trading decisions (async queue)
    - Results cached and attached to tokens after the fact
    - Used to ENHANCE decisions, not gate them
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.deepseek_api_key.get_secret_value()
        self.model = settings.deepseek_model
        self.timeout = settings.deepseek_timeout_seconds

        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._cache: dict[str, NarrativeAnalysis] = {}
        self._cache_ttl = timedelta(hours=1)

        # Analysis queue for background processing
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the AI service and background worker."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        self._rate_limiter = await get_rate_limiter()
        self._worker_task = asyncio.create_task(self._process_queue())

        # Initialize trending context cache
        self._trending_context = DEFAULT_TRENDING_CONTEXT
        self._trending_last_update = datetime.utcnow() - timedelta(hours=2)  # Force first update

        # Start background trending update
        self._trending_task = asyncio.create_task(self._update_trending_loop())

        logger.info("deepseek_service_started")

    async def _update_trending_loop(self):
        """Background task to update trending context periodically."""
        while True:
            try:
                await self._fetch_trending_context()
                await asyncio.sleep(300)  # Update every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("trending_update_error", error=str(e))
                await asyncio.sleep(60)

    async def _fetch_trending_context(self):
        """Fetch current trending topics for context."""
        # Only update if cache is stale (older than 5 minutes)
        if datetime.utcnow() - self._trending_last_update < timedelta(minutes=5):
            return

        try:
            # Try to fetch trending from crypto Twitter / news
            # Using a simple web search approach
            trends = []

            # Check common crypto news for trending topics
            # This is a simplified version - in production you'd use Twitter API
            search_terms = ["solana meme coin", "crypto trending", "viral token"]

            # For now, use a curated list of common trending topics
            # In production, this would fetch from Twitter API or news feeds
            current_trends = await self._get_current_crypto_trends()

            if current_trends:
                self._trending_context = current_trends
                self._trending_last_update = datetime.utcnow()
                logger.debug("trending_context_updated")

        except Exception as e:
            logger.debug("trending_fetch_error", error=str(e))

    async def _get_current_crypto_trends(self) -> str:
        """Get current crypto trends - simplified version."""
        # In production, this would fetch from:
        # 1. Twitter API for trending hashtags
        # 2. CoinGecko trending
        # 3. Crypto news feeds

        # For now, return a dynamic context based on current date
        now = datetime.utcnow()

        context_parts = [
            "Current crypto/memecoin trends:",
            "- Solana memecoins are the hottest category",
            "- AI-themed tokens continue to trend",
            "- Animal memes (cats, dogs, frogs) always work",
        ]

        # Add seasonal context
        month = now.month
        if month == 1:
            context_parts.append("- New Year optimism driving speculation")
        elif month == 2:
            context_parts.append("- Valentine's themed tokens might trend")
        elif month == 10:
            context_parts.append("- Spooky/Halloween themed tokens trending")
        elif month == 11 or month == 12:
            context_parts.append("- Holiday/Christmas themed tokens popular")

        # Add day of week context (weekends often see memecoin pumps)
        if now.weekday() >= 5:  # Saturday/Sunday
            context_parts.append("- Weekend trading - memecoins often pump")

        context_parts.extend([
            "- Political tokens around any political news",
            "- Elon Musk related tokens always get attention",
            "- Viral TikTok/Twitter meme references do well",
            "- Tokens with '$' or '-illion' in name signal memecoin",
        ])

        return "\n".join(context_parts)

    async def stop(self):
        """Stop the AI service."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if hasattr(self, '_trending_task') and self._trending_task:
            self._trending_task.cancel()
            try:
                await self._trending_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.aclose()

        logger.info("deepseek_service_stopped")

    async def analyze_token(
        self,
        name: str,
        symbol: str,
        description: str = "",
        wait: bool = False,
        timeout_seconds: float = 5.0,  # Increased timeout for DeepSeek API
    ) -> Optional[NarrativeAnalysis]:
        """
        Analyze a token's narrative strength.

        Args:
            name: Token name
            symbol: Token symbol
            description: Token description or URI content
            wait: If True, wait for result. If False, queue and return None.
            timeout_seconds: Max time to wait for AI response when wait=True

        Returns:
            NarrativeAnalysis if wait=True or cached, None otherwise
        """
        # Clean inputs - strip null bytes and whitespace (Metaplex metadata often has padding)
        name = (name or "").replace('\x00', '').strip()
        symbol = (symbol or "").replace('\x00', '').strip()
        description = (description or "").replace('\x00', '').strip()

        # Use defaults if still empty
        if not name:
            name = "Unknown"
        if not symbol:
            symbol = "UNKNOWN"

        cache_key = f"{symbol}:{name}"

        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if datetime.utcnow() - cached.analyzed_at < self._cache_ttl:
                return cached

        if wait:
            # Synchronous analysis with timeout - use heuristics as fast fallback
            try:
                result = await asyncio.wait_for(
                    self._analyze(name, symbol, description),
                    timeout=timeout_seconds,
                )
                if result:
                    self._cache[cache_key] = result
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "ai_analysis_timeout_using_heuristic",
                    symbol=symbol,
                    name=name,
                    timeout_seconds=timeout_seconds,
                    reason="DeepSeek API too slow, using fast heuristic fallback",
                )
                result = self._heuristic_analysis(name, symbol, description)
                self._cache[cache_key] = result
                return result
        else:
            # Queue for background processing
            await self._queue.put((name, symbol, description, cache_key))
            return None

    async def get_cached_analysis(
        self, symbol: str, name: str
    ) -> Optional[NarrativeAnalysis]:
        """Get cached analysis if available."""
        cache_key = f"{symbol}:{name}"
        cached = self._cache.get(cache_key)

        if cached and datetime.utcnow() - cached.analyzed_at < self._cache_ttl:
            return cached
        return None

    async def _process_queue(self):
        """Background worker to process analysis queue."""
        while True:
            try:
                name, symbol, description, cache_key = await self._queue.get()

                # Skip if recently analyzed
                if cache_key in self._cache:
                    cached = self._cache[cache_key]
                    if datetime.utcnow() - cached.analyzed_at < self._cache_ttl:
                        continue

                # Analyze
                result = await self._analyze(name, symbol, description)
                if result:
                    self._cache[cache_key] = result

                # Rate limiting - don't spam the API
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("narrative_queue_error", error=str(e))
                await asyncio.sleep(1)

    async def _analyze(
        self,
        name: str,
        symbol: str,
        description: str,
    ) -> Optional[NarrativeAnalysis]:
        """Perform actual AI analysis with rate limiting."""

        # Log analysis attempt
        logger.debug(
            "ai_analysis_attempt",
            symbol=symbol or "UNKNOWN",
            name=name or "Unknown",
        )

        # Check for valid API key
        has_client = self._client is not None
        key_len = len(self.api_key) if self.api_key else 0
        key_prefix = self.api_key[:8] if self.api_key and len(self.api_key) > 8 else "N/A"
        api_key_valid = (
            has_client and
            self.api_key and
            key_len > 10 and
            not self.api_key.startswith("your_") and
            not self.api_key.startswith("sk-xxx") and
            not self.api_key.startswith("placeholder")
        )

        if not api_key_valid:
            logger.warning(
                "deepseek_api_key_invalid",
                has_client=has_client,
                key_length=key_len,
                key_prefix=key_prefix,
                reason="Using heuristic fallback - configure DEEPSEEK_API_KEY in .env",
            )
            return self._heuristic_analysis(name, symbol, description)

        start_time = asyncio.get_event_loop().time()

        logger.info(
            "deepseek_api_key_valid",
            symbol=symbol,
            key_prefix=key_prefix,
        )

        # Apply rate limiting
        if self._rate_limiter:
            wait_time = await self._rate_limiter.acquire(RateLimitService.DEEPSEEK)
            if wait_time > 0:
                logger.info(
                    "deepseek_rate_limit_wait",
                    wait_seconds=wait_time,
                    symbol=symbol,
                )
                await asyncio.sleep(wait_time)

        try:
            # Get trending context for the AI
            trending_ctx = getattr(self, '_trending_context', DEFAULT_TRENDING_CONTEXT)

            prompt = NARRATIVE_ANALYSIS_PROMPT.format(
                name=name or "Unknown",
                symbol=symbol or "???",
                description=description[:500] if description else "Not provided",
                current_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                trending_context=trending_ctx,
            )

            logger.info(
                "calling_deepseek_api",
                symbol=symbol,
                name=name,
                model=self.model,
            )

            response = await self._client.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
            )

            # Handle rate limit (429)
            if response.status_code == 429:
                if self._rate_limiter:
                    # Extract retry-after from headers
                    retry_after = None
                    headers_dict = dict(response.headers)
                    if "Retry-After" in headers_dict:
                        retry_after = float(headers_dict["Retry-After"])

                    backoff = await self._rate_limiter.report_rate_limit(
                        RateLimitService.DEEPSEEK,
                        retry_after=retry_after,
                        headers=headers_dict,
                    )
                    logger.warning(
                        "deepseek_rate_limited",
                        backoff_seconds=backoff,
                        symbol=symbol,
                    )
                    # Wait and retry once
                    await asyncio.sleep(backoff)
                    response = await self._client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.3,
                            "max_tokens": 500,
                        },
                    )

            if response.status_code != 200:
                logger.warning(
                    "deepseek_api_error",
                    status=response.status_code,
                    token=symbol,
                )
                return self._heuristic_analysis(name, symbol, description)

            # Report success
            if self._rate_limiter:
                await self._rate_limiter.report_success(RateLimitService.DEEPSEEK)

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse JSON response
            analysis = self._parse_response(content, name, symbol)
            analysis.analysis_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

            logger.info(
                "deepseek_ai_analysis_success",
                symbol=symbol,
                name=name,
                score=analysis.narrative_score,
                category=analysis.category.value,
                reasoning=analysis.reasoning[:50] if analysis.reasoning else "",
                time_ms=analysis.analysis_time_ms,
            )

            return analysis

        except Exception as e:
            logger.error("deepseek_analysis_error", error=str(e), symbol=symbol)
            return self._heuristic_analysis(name, symbol, description)

    def _parse_response(
        self,
        content: str,
        name: str,
        symbol: str,
    ) -> NarrativeAnalysis:
        """Parse AI response into NarrativeAnalysis."""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            category_str = data.get("category", "unknown").lower()
            try:
                category = NarrativeCategory(category_str)
            except ValueError:
                category = NarrativeCategory.UNKNOWN

            return NarrativeAnalysis(
                token_name=name,
                token_symbol=symbol,
                narrative_score=float(data.get("narrative_score", 50)),
                category=category,
                cultural_relevance=float(data.get("cultural_relevance", 50)),
                virality_potential=float(data.get("virality_potential", 50)),
                reasoning=data.get("reasoning", ""),
                keywords_detected=data.get("keywords", []),
                confidence=float(data.get("confidence", 0.5)),
                analyzed_at=datetime.utcnow(),
                analysis_time_ms=0,
            )

        except Exception as e:
            logger.warning("parse_error", error=str(e), content=content[:100])
            return self._heuristic_analysis(name, symbol, "")

    def _heuristic_analysis(
        self,
        name: str,
        symbol: str,
        description: str,
    ) -> NarrativeAnalysis:
        """
        Fast heuristic analysis when AI is unavailable.
        Uses keyword matching and pattern recognition.

        This is a SMART fallback - uses extensive keyword lists
        to identify potentially viral tokens.
        """
        name_lower = name.lower() if name else ""
        symbol_lower = symbol.lower() if symbol else ""
        desc_lower = description.lower() if description else ""
        combined = f"{name_lower} {symbol_lower} {desc_lower}"

        # Base score starts at 45 - allows trades with even minimal signals
        # This is important because memecoin sniping is a volume game
        # Better to enter more positions with small sizes than miss opportunities
        score = 45
        category = NarrativeCategory.RANDOM
        keywords = []
        confidence = 0.48

        # TIER 0: SUPER HIGH VALUE - proven meme formats (these always get attention)
        super_memes = ["pepe", "doge", "shib", "moon", "wojak"]
        for meme in super_memes:
            if meme in combined:
                score += 30  # Strong boost
                category = NarrativeCategory.CRYPTO_NATIVE
                keywords.append(f"SUPER:{meme}")
                confidence = max(confidence, 0.65)

        # HIGH VALUE: Crypto-native memes (very high viral potential)
        tier1_memes = ["chad", "wagmi", "gm", "frog", "cat", "dog", "ape",
                       "monkey", "nft", "ai", "kek", "based", "pump", "lambo",
                       "diamond", "hands", "hodl", "wen", "ser", "anon", "whale"]
        for meme in tier1_memes:
            if meme in combined:
                score += 20
                category = NarrativeCategory.CRYPTO_NATIVE
                keywords.append(meme)
                confidence = max(confidence, 0.55)

        # HIGH VALUE: Prefixes that indicate memecoin (Baby, Mini, etc.)
        meme_prefixes = ["baby", "mini", "micro", "mega", "super", "ultra", "giga", "king", "queen"]
        for prefix in meme_prefixes:
            if combined.startswith(prefix) or f" {prefix}" in combined:
                score += 15
                keywords.append(f"prefix:{prefix}")
                confidence = max(confidence, 0.5)

        # HIGH VALUE: Celebrity/influencer markers (always get attention)
        celebrities = ["elon", "trump", "biden", "musk", "kanye", "drake",
                       "snoop", "cz", "vitalik", "sam", "taylor", "swift",
                       "bezos", "zuck", "mark", "obama", "ye", "kimk", "kylie",
                       "andrew", "tate", "logan", "jake", "paul", "pewdie"]
        for celeb in celebrities:
            if celeb in combined:
                score += 28
                category = NarrativeCategory.CELEBRITY
                keywords.append(celeb)
                confidence = max(confidence, 0.6)

        # MEDIUM VALUE: Political markers (volatile but can moon)
        political = ["maga", "trump", "biden", "democrat", "republican",
                     "vote", "election", "america", "usa", "freedom", "liberty",
                     "patriot", "brandon", "dark"]
        for pol in political:
            if pol in combined:
                score += 15
                category = NarrativeCategory.POLITICAL
                keywords.append(pol)
                confidence = max(confidence, 0.5)

        # MEDIUM VALUE: AI/Tech themes (currently hot)
        ai_themes = ["ai", "gpt", "chat", "agent", "bot", "neural", "deep",
                     "learn", "robot", "sentient", "conscious", "openai",
                     "claude", "gemini", "llm", "copilot", "grok", "llama",
                     "mistral", "anthropic", "meta", "agi", "singularity"]
        for ai in ai_themes:
            if ai in combined:
                score += 22  # Slightly higher for AI themes
                category = NarrativeCategory.CULTURAL_MOMENT
                keywords.append(f"ai:{ai}")
                confidence = max(confidence, 0.58)

        # MEDIUM VALUE: Current events / cultural moments
        cultural = ["viral", "trend", "tiktok", "based", "sigma", "alpha",
                    "gigachad", "rare", "epic", "legendary", "banger", "fire",
                    "lit", "goat", "skibidi", "ohio", "rizz"]
        for cult in cultural:
            if cult in combined:
                score += 12
                category = NarrativeCategory.CULTURAL_MOMENT
                keywords.append(cult)

        # Seasonal markers (check current date)
        now = datetime.utcnow()
        seasonal_active = []
        if now.month == 1:
            seasonal_active = ["newyear", "2026", "resolution", "january"]
        elif now.month == 2:
            seasonal_active = ["valentine", "love", "heart", "cupid"]
        elif now.month == 10:
            seasonal_active = ["halloween", "spooky", "ghost", "pumpkin", "scary", "october"]
        elif now.month in [11, 12]:
            seasonal_active = ["christmas", "xmas", "santa", "holiday", "winter"]

        for season in seasonal_active:
            if season in combined:
                score += 15
                category = NarrativeCategory.SEASONAL
                keywords.append(f"seasonal:{season}")
                confidence = max(confidence, 0.5)

        # BONUS: Symbol patterns that suggest memecoin
        if "$" in symbol_lower or "$" in name_lower:
            score += 5
            keywords.append("has_dollar_sign")
        if "illion" in combined or "billion" in combined or "trillion" in combined:
            score += 8
            keywords.append("illion_suffix")

        # BONUS: Animal combos (BabyDoge, CatPepe, etc.)
        animals = ["doge", "dog", "inu", "shib", "cat", "frog", "pepe", "ape", "monkey", "bear", "bull"]
        animal_count = sum(1 for a in animals if a in combined)
        if animal_count >= 1:
            score += 10
            keywords.append("animal_theme")
        if animal_count >= 2:
            score += 10  # Extra bonus for combos like BabyDoge
            keywords.append("animal_combo")

        # NEGATIVE: Red flags (significantly reduce score)
        red_flags = ["test", "fake", "scam", "rug", "honeypot", "honey",
                     "airdrop", "free", "giveaway", "presale"]
        for neg in red_flags:
            if neg in combined:
                score -= 30
                keywords.append(f"[RED_FLAG:{neg}]")
                confidence = max(0.2, confidence - 0.15)

        # NEGATIVE: Generic/low effort names (minimal penalty - don't want to miss trades)
        generic = ["token", "coin"]  # Only flag very generic ones
        generic_count = sum(1 for g in generic if g in combined)
        if generic_count >= 2:
            score -= 5
            keywords.append("[generic_name]")

        # NEGATIVE: Just random letters/numbers (very minimal penalty)
        # Many legitimate memecoins have short names
        if name and len(name) <= 2 and name.isalnum():  # Only if 2 chars or less
            if not any(m in name_lower for m in super_memes + tier1_memes + celebrities):
                score -= 5
                keywords.append("[very_short]")

        # Cap score
        score = max(0, min(100, score))

        # Set reasoning based on score
        if score >= 70:
            reasoning = f"Strong narrative: {', '.join(keywords[:3])}"
        elif score >= 55:
            reasoning = f"Good potential: {', '.join(keywords[:3]) if keywords else 'decent signals'}"
        elif score >= 40:
            reasoning = f"Moderate: {', '.join(keywords[:2]) if keywords else 'some signals'}"
        else:
            reasoning = f"Weak: {', '.join(keywords[:2]) if keywords else 'no clear theme'}"

        # Log heuristic analysis result for debugging
        logger.info(
            "heuristic_analysis_result",
            symbol=symbol or "UNKNOWN",
            name=name or "Unknown",
            score=score,
            keywords=keywords[:5],
            category=category.value,
            confidence=confidence,
        )

        return NarrativeAnalysis(
            token_name=name,
            token_symbol=symbol,
            narrative_score=score,
            category=category,
            cultural_relevance=score * 0.8,
            virality_potential=score * 0.7,
            reasoning=reasoning,
            keywords_detected=keywords[:10],  # Limit keywords
            confidence=confidence,
            analyzed_at=datetime.utcnow(),
            analysis_time_ms=1.0,
        )

    def get_queue_size(self) -> int:
        """Get number of pending analyses."""
        return self._queue.qsize()

    def clear_cache(self):
        """Clear the analysis cache."""
        self._cache.clear()
