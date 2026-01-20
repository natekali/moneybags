"""Pytest fixtures for Bags Sniper tests."""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import AsyncGenerator
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bags_sniper.core.config import Settings
from bags_sniper.models.database import Base, Deployer, TokenLaunch


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    return Settings(
        bags_api_key="test_bags_key",
        bags_api_base_url="https://test.bags.fm/api/v1",
        helius_api_key="test_helius_key",
        deepseek_api_key="test_deepseek_key",
        telegram_bot_token="test_telegram_token",
        telegram_chat_id="123456789",
        wallet_private_key="test_wallet_key",
        database_url="sqlite+aiosqlite:///:memory:",
        dry_run=True,
        max_position_size_sol=0.3,
        min_deployer_score=60.0,
        min_graduation_rate=0.042,
    )


@pytest.fixture
async def db_session(mock_settings) -> AsyncGenerator[AsyncSession, None]:
    """Create in-memory database session for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def sample_deployer(db_session: AsyncSession) -> Deployer:
    """Create sample deployer for testing."""
    deployer = Deployer(
        wallet_address="DeployerWallet123456789012345678901234567890AB",
        total_launches=10,
        graduated_launches=3,
        graduation_rate=0.3,
        avg_peak_mcap_sol=Decimal("500"),
        estimated_profit_sol=Decimal("100"),
        last_launch_at=datetime.utcnow(),
        deployer_score=75.0,
        score_updated_at=datetime.utcnow(),  # Prevent refresh
    )
    db_session.add(deployer)
    await db_session.commit()
    return deployer


@pytest.fixture
async def elite_deployer(db_session: AsyncSession) -> Deployer:
    """Create elite deployer (top performer) for testing."""
    deployer = Deployer(
        wallet_address="EliteDeployer12345678901234567890123456789012",
        total_launches=20,
        graduated_launches=8,
        graduation_rate=0.4,  # 40% graduation rate
        avg_peak_mcap_sol=Decimal("2000"),
        estimated_profit_sol=Decimal("5000"),
        last_launch_at=datetime.utcnow(),
        deployer_score=95.0,
        score_updated_at=datetime.utcnow(),  # Prevent refresh
    )
    db_session.add(deployer)
    await db_session.commit()
    return deployer


@pytest.fixture
async def poor_deployer(db_session: AsyncSession) -> Deployer:
    """Create poor performer deployer for testing."""
    deployer = Deployer(
        wallet_address="PoorDeployer123456789012345678901234567890123",
        total_launches=15,
        graduated_launches=0,
        graduation_rate=0.0,
        avg_peak_mcap_sol=Decimal("10"),
        estimated_profit_sol=Decimal("-50"),
        last_launch_at=datetime.utcnow(),
        deployer_score=5.0,
    )
    db_session.add(deployer)
    await db_session.commit()
    return deployer


@pytest.fixture
async def sample_token_launch(
    db_session: AsyncSession, sample_deployer: Deployer
) -> TokenLaunch:
    """Create sample token launch for testing."""
    token = TokenLaunch(
        mint_address="TokenMint123456789012345678901234567890123456",
        deployer_id=sample_deployer.id,
        name="Test Token",
        symbol="TEST",
        launched_at=datetime.utcnow(),
        graduated=False,
    )
    db_session.add(token)
    await db_session.commit()
    return token
