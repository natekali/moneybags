"""
Configuration management using Pydantic Settings.
Loads from environment variables with validation.
"""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Bags.fm API
    bags_api_key: SecretStr = Field(..., description="Bags.fm API key")
    bags_api_base_url: str = Field(
        default="https://public-api-v2.bags.fm/api/v1",
        description="Bags.fm API base URL",
    )
    bags_rate_limit_per_hour: int = Field(
        default=1000, description="Bags.fm API rate limit per hour"
    )

    # Helius RPC
    helius_api_key: SecretStr = Field(..., description="Helius API key for Solana RPC")
    helius_rpc_url: str = Field(
        default="https://mainnet.helius-rpc.com",
        description="Helius RPC base URL",
    )

    # Backup RPC endpoints
    backup_rpc_urls: list[str] = Field(
        default_factory=lambda: [
            "https://api.mainnet-beta.solana.com",
        ],
        description="Backup RPC endpoints for failover",
    )

    # Deepseek AI
    deepseek_api_key: SecretStr = Field(..., description="Deepseek API key")
    deepseek_model: str = Field(
        default="deepseek-chat", description="Deepseek model to use"
    )
    deepseek_timeout_seconds: float = Field(
        default=10.0, description="Deepseek API timeout"
    )

    # Telegram
    telegram_bot_token: SecretStr = Field(..., description="Telegram bot token")
    telegram_chat_id: str = Field(..., description="Telegram chat ID for alerts")

    # Wallet
    wallet_private_key: SecretStr = Field(
        ..., description="Solana wallet private key (base58)"
    )

    # Database
    database_url: SecretStr = Field(
        ..., description="PostgreSQL connection string"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )

    # Trading Parameters
    max_position_size_sol: float = Field(
        default=0.3, ge=0.01, le=5.0, description="Maximum position size in SOL"
    )
    base_risk_percent: float = Field(
        default=2.0, ge=0.5, le=10.0, description="Base risk percentage per trade"
    )
    min_deployer_score: float = Field(
        default=60.0, ge=0, le=100, description="Minimum deployer score to trade"
    )
    min_graduation_rate: float = Field(
        default=0.042,  # 3x baseline of 1.4%
        ge=0,
        le=1.0,
        description="Minimum deployer graduation rate",
    )

    # Circuit Breaker Thresholds
    circuit_breaker_l1_percent: float = Field(
        default=3.0, description="Level 1 circuit breaker (reduce size 50%)"
    )
    circuit_breaker_l2_percent: float = Field(
        default=5.0, description="Level 2 circuit breaker (reduce size 75%)"
    )
    circuit_breaker_l3_percent: float = Field(
        default=8.0, description="Level 3 circuit breaker (new entries paused)"
    )
    circuit_breaker_l4_percent: float = Field(
        default=10.0, description="Level 4 circuit breaker (full shutdown)"
    )

    # Take-Profit Tiers
    tp_tier1_multiple: float = Field(default=2.5, description="First TP at 2.5x")
    tp_tier1_percent: float = Field(default=25.0, description="Sell 25% at tier 1")
    tp_tier2_multiple: float = Field(default=4.0, description="Second TP at 4x")
    tp_tier2_percent: float = Field(default=35.0, description="Sell 35% at tier 2")
    tp_tier3_multiple: float = Field(default=7.0, description="Third TP at 7x")
    tp_tier3_percent: float = Field(default=20.0, description="Sell 20% at tier 3")
    trailing_stop_percent: float = Field(
        default=15.0, description="Trailing stop for remainder"
    )

    # Stop Loss
    stop_loss_percent: float = Field(
        default=30.0, description="Hard stop loss percentage"
    )

    # Latency Budget
    max_latency_ms: int = Field(
        default=1500, description="Maximum acceptable latency in milliseconds"
    )

    # Jito MEV Protection
    jito_tip_lamports: int = Field(
        default=10000, description="Jito bundle tip in lamports"
    )
    use_jito_bundles: bool = Field(
        default=True, description="Enable Jito bundle protection"
    )

    # Operational
    dry_run: bool = Field(
        default=True, description="Run in simulation mode (no real trades)"
    )
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
