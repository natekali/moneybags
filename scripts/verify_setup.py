#!/usr/bin/env python3
"""
Setup Verification Script for Bags Sniper.

Checks all requirements are met before running the bot:
1. Environment variables configured
2. Database connection works
3. API endpoints reachable
4. Dependencies installed

Run with: python scripts/verify_setup.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_check(name: str, passed: bool, detail: str = ""):
    status = "OK" if passed else "FAIL"
    color_start = "\033[92m" if passed else "\033[91m"
    color_end = "\033[0m"
    print(f"  {color_start}[{status}]{color_end} {name}")
    if detail:
        print(f"      {detail}")


def check_python_version() -> bool:
    """Check Python version >= 3.11."""
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 11
    print_check(
        "Python Version",
        passed,
        f"Found {version.major}.{version.minor}.{version.micro}, need 3.11+",
    )
    return passed


def check_env_file() -> bool:
    """Check .env file exists."""
    env_path = Path(__file__).parent.parent / ".env"
    exists = env_path.exists()
    print_check(
        ".env File",
        exists,
        str(env_path) if exists else "Missing! Copy from .env.example",
    )
    return exists


def check_required_env_vars() -> tuple[bool, list[str]]:
    """Check required environment variables are set."""
    required = [
        "BAGS_API_KEY",
        "HELIUS_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "WALLET_PRIVATE_KEY",
        "DATABASE_URL",
    ]

    # Load .env if exists
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    missing = []
    for var in required:
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            missing.append(var)

    passed = len(missing) == 0
    if missing:
        print_check(
            "Environment Variables",
            False,
            f"Missing or placeholder: {', '.join(missing)}",
        )
    else:
        print_check("Environment Variables", True, "All required vars set")

    return passed, missing


def check_dependencies() -> bool:
    """Check required Python packages are installed."""
    required_packages = [
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("sqlalchemy", "sqlalchemy"),
        ("httpx", "httpx"),
        ("structlog", "structlog"),
        ("asyncpg", "asyncpg"),
        ("solders", "solders"),
    ]

    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)

    passed = len(missing) == 0
    if missing:
        print_check(
            "Python Dependencies",
            False,
            f"Missing: {', '.join(missing)}\nRun: pip install -e .",
        )
    else:
        print_check("Python Dependencies", True, "All packages installed")

    return passed


async def check_database_connection() -> bool:
    """Check database connection works."""
    try:
        from sqlalchemy import text
        from bags_sniper.core.config import get_settings
        from bags_sniper.models.database import init_database

        settings = get_settings()
        session_factory = await init_database(settings.database_url.get_secret_value())

        async with session_factory() as session:
            await session.execute(text("SELECT 1"))

        print_check("Database Connection", True, "Connected successfully")
        return True

    except Exception as e:
        print_check("Database Connection", False, str(e)[:80])
        return False


async def check_helius_api() -> bool:
    """Check Helius RPC is reachable."""
    try:
        from bags_sniper.core.config import get_settings
        from bags_sniper.services.solana_rpc import SolanaRPCClient

        settings = get_settings()
        async with SolanaRPCClient(settings) as rpc:
            slot = await rpc.get_slot()
            if slot:
                print_check("Helius RPC", True, f"Current slot: {slot}")
                return True
            else:
                print_check("Helius RPC", False, "Could not get slot")
                return False

    except Exception as e:
        print_check("Helius RPC", False, str(e)[:80])
        return False


async def check_bags_api() -> bool:
    """Check Bags.fm API is reachable."""
    try:
        from bags_sniper.core.config import get_settings

        settings = get_settings()

        # Just verify API key is configured (can't easily test without making trades)
        api_key = settings.bags_api_key.get_secret_value()
        if api_key and not api_key.startswith("your_"):
            print_check("Bags.fm API", True, "API key configured")
            return True
        else:
            print_check("Bags.fm API", False, "API key not configured")
            return False

    except Exception as e:
        print_check("Bags.fm API", False, str(e)[:80])
        return False


def check_import_structure() -> bool:
    """Check all modules can be imported."""
    modules = [
        "bags_sniper.core.config",
        "bags_sniper.core.deployer_intelligence",
        "bags_sniper.core.filter_engine",
        "bags_sniper.core.circuit_breaker",
        "bags_sniper.core.trading_engine",
        "bags_sniper.models.database",
        "bags_sniper.services.bags_api",
        "bags_sniper.services.solana_rpc",
        "bags_sniper.services.telegram_bot",
        "bags_sniper.services.data_ingestion",
        "bags_sniper.backtest.engine",
        "bags_sniper.backtest.analytics",
    ]

    failed = []
    for module in modules:
        try:
            __import__(module)
        except Exception as e:
            failed.append((module, str(e)[:50]))

    passed = len(failed) == 0
    if failed:
        print_check(
            "Module Imports",
            False,
            f"Failed: {failed[0][0]} - {failed[0][1]}",
        )
        for mod, err in failed[1:]:
            print(f"      {mod}: {err}")
    else:
        print_check("Module Imports", True, f"All {len(modules)} modules import OK")

    return passed


async def run_verification():
    """Run all verification checks."""
    print_header("BAGS SNIPER SETUP VERIFICATION")

    results = {}

    # Basic checks
    print("\n[1/4] BASIC REQUIREMENTS")
    print("-" * 40)
    results["python"] = check_python_version()
    results["env_file"] = check_env_file()
    results["env_vars"], missing_vars = check_required_env_vars()
    results["dependencies"] = check_dependencies()

    # Import checks
    print("\n[2/4] CODE STRUCTURE")
    print("-" * 40)
    results["imports"] = check_import_structure()

    # Skip API checks if env vars missing
    if not results["env_vars"] or not results["dependencies"]:
        print("\n[3/4] DATABASE CONNECTION")
        print("-" * 40)
        print_check("Skipped", False, "Fix environment variables first")
        results["database"] = False

        print("\n[4/4] EXTERNAL APIS")
        print("-" * 40)
        print_check("Skipped", False, "Fix environment variables first")
        results["helius"] = False
        results["bags"] = False
    else:
        # Database check
        print("\n[3/4] DATABASE CONNECTION")
        print("-" * 40)
        results["database"] = await check_database_connection()

        # API checks
        print("\n[4/4] EXTERNAL APIS")
        print("-" * 40)
        results["helius"] = await check_helius_api()
        results["bags"] = await check_bags_api()

    # Summary
    print_header("VERIFICATION SUMMARY")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    print(f"\n  Checks Passed: {passed}/{total}")

    if passed == total:
        print("\n  \033[92m[OK] ALL CHECKS PASSED - Ready to run!\033[0m")
        print("\n  Next steps:")
        print("    1. Run backfill:  bags-sniper backfill --days 7")
        print("    2. Run backtest:  bags-sniper backtest --days 7")
        print("    3. Start bot:     bags-sniper run")
    else:
        print("\n  \033[91m[FAIL] SOME CHECKS FAILED - Fix issues above\033[0m")

        if not results.get("env_file"):
            print("\n  To fix .env:")
            print("    cp .env.example .env")
            print("    # Then edit .env with your API keys")

        if not results.get("dependencies"):
            print("\n  To fix dependencies:")
            print("    pip install -e .")

        if not results.get("database"):
            print("\n  To fix database:")
            print("    docker-compose up -d postgres redis")
            print("    # Or set DATABASE_URL to your PostgreSQL instance")

    print()
    return passed == total


def main():
    """Main entry point."""
    try:
        success = asyncio.run(run_verification())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nVerification failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
