"""
Command-line interface for Bags Sniper operations.
Provides utilities for backfill, debugging, and manual operations.
"""

import argparse
import asyncio
import logging
from datetime import datetime
from decimal import Decimal

import structlog

from bags_sniper.core.config import get_settings


def configure_logging(verbose: bool = False):
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


async def cmd_backfill(args):
    """Run historical data backfill."""
    from bags_sniper.models.database import init_database
    from bags_sniper.services.data_ingestion import DataIngestionService

    settings = get_settings()
    print(f"\n{'='*60}")
    print("DEPLOYER DATA BACKFILL")
    print(f"{'='*60}")
    print(f"Days back: {args.days}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"{'='*60}\n")

    session_factory = await init_database(settings.database_url.get_secret_value())
    service = DataIngestionService(settings, session_factory)

    await service.run_backfill(
        days_back=args.days,
        max_tokens=args.max_tokens,
    )

    print("\nBackfill complete!")


async def cmd_stats(args):
    """Show deployer statistics."""
    from sqlalchemy import select, func

    from bags_sniper.models.database import Deployer, TokenLaunch, Trade, init_database

    settings = get_settings()
    session_factory = await init_database(settings.database_url.get_secret_value())

    async with session_factory() as session:
        # Deployer stats
        deployer_result = await session.execute(
            select(
                func.count(Deployer.id).label("total"),
                func.avg(Deployer.graduation_rate).label("avg_grad"),
                func.max(Deployer.deployer_score).label("max_score"),
            )
        )
        deployer_stats = deployer_result.one()

        # Token stats
        token_result = await session.execute(
            select(
                func.count(TokenLaunch.id).label("total"),
                func.sum(func.cast(TokenLaunch.graduated, int)).label("graduated"),
            )
        )
        token_stats = token_result.one()

        # Trade stats
        trade_result = await session.execute(
            select(
                func.count(Trade.id).label("total"),
                func.sum(Trade.realized_pnl_sol).label("total_pnl"),
            )
        )
        trade_stats = trade_result.one()

        # High score deployers
        top_deployers = await session.execute(
            select(Deployer)
            .where(Deployer.total_launches >= 3)
            .order_by(Deployer.deployer_score.desc())
            .limit(10)
        )
        top_deployers = top_deployers.scalars().all()

    print(f"\n{'='*60}")
    print("DATABASE STATISTICS")
    print(f"{'='*60}")

    print("\nDEPLOYERS:")
    print(f"  Total: {deployer_stats.total or 0}")
    print(f"  Avg Graduation Rate: {(deployer_stats.avg_grad or 0) * 100:.2f}%")
    print(f"  Max Score: {deployer_stats.max_score or 0:.1f}")

    print("\nTOKENS:")
    print(f"  Total Tracked: {token_stats.total or 0}")
    print(f"  Graduated: {token_stats.graduated or 0}")
    if token_stats.total:
        grad_rate = (token_stats.graduated or 0) / token_stats.total * 100
        print(f"  Overall Grad Rate: {grad_rate:.2f}%")

    print("\nTRADES:")
    print(f"  Total: {trade_stats.total or 0}")
    print(f"  Total P&L: {trade_stats.total_pnl or 0:.4f} SOL")

    if top_deployers:
        print("\nTOP 10 DEPLOYERS (min 3 launches):")
        print("-" * 60)
        print(f"{'Wallet':<12} {'Score':>8} {'Launches':>10} {'Grad Rate':>12}")
        print("-" * 60)
        for d in top_deployers:
            print(
                f"{d.wallet_address[:12]:<12} "
                f"{d.deployer_score:>8.1f} "
                f"{d.total_launches:>10} "
                f"{d.graduation_rate * 100:>11.2f}%"
            )

    print(f"\n{'='*60}\n")


async def cmd_check_deployer(args):
    """Check a specific deployer's profile."""
    from bags_sniper.core.deployer_intelligence import DeployerIntelligence
    from bags_sniper.models.database import init_database
    from bags_sniper.services.solana_rpc import SolanaRPCClient

    settings = get_settings()
    session_factory = await init_database(settings.database_url.get_secret_value())

    async with SolanaRPCClient(settings) as rpc:
        deployer_intel = DeployerIntelligence(settings, rpc)

        async with session_factory() as session:
            profile = await deployer_intel.analyze_deployer(
                session, args.wallet, force_refresh=args.refresh
            )

    print(f"\n{'='*60}")
    print(f"DEPLOYER PROFILE: {args.wallet[:20]}...")
    print(f"{'='*60}")
    print(f"\nOverall Score: {profile.deployer_score:.1f}/100")
    print(f"Recommendation: {profile.recommendation}")
    print("\nMETRICS:")
    print(f"  Total Launches: {profile.total_launches}")
    print(f"  Graduated: {profile.graduated_launches}")
    print(f"  Graduation Rate: {profile.graduation_rate * 100:.2f}%")
    print(f"  Avg Peak MCAP: {profile.avg_peak_mcap_sol:.2f} SOL")
    print(f"  Est. Profit: {profile.estimated_profit_sol:.2f} SOL")
    print(f"  Days Since Last: {profile.days_since_last_launch or 'N/A'}")
    print(f"  Launches (7d): {profile.launches_last_7d}")
    print(f"  New Deployer: {'Yes' if profile.is_new_deployer else 'No'}")

    print("\nSCORE COMPONENTS:")
    for component, value in profile.score_components.items():
        print(f"  {component.replace('_', ' ').title()}: {value:.1f}")

    print(f"\n{'='*60}\n")


async def cmd_top_deployers(args):
    """List top deployers for watchlist."""
    from sqlalchemy import select

    from bags_sniper.models.database import Deployer, init_database

    settings = get_settings()
    session_factory = await init_database(settings.database_url.get_secret_value())

    async with session_factory() as session:
        result = await session.execute(
            select(Deployer)
            .where(Deployer.total_launches >= args.min_launches)
            .where(Deployer.deployer_score >= args.min_score)
            .order_by(Deployer.deployer_score.desc())
            .limit(args.limit)
        )
        deployers = result.scalars().all()

    print(f"\n{'='*80}")
    print(f"TOP DEPLOYERS (Score >= {args.min_score}, Launches >= {args.min_launches})")
    print(f"{'='*80}")

    if not deployers:
        print("\nNo deployers match criteria. Run backfill first.")
    else:
        print(
            f"\n{'#':>3} {'Wallet':<44} {'Score':>7} {'Launches':>8} {'Grad%':>7} {'Last':>10}"
        )
        print("-" * 80)
        for i, d in enumerate(deployers, 1):
            last_launch = "N/A"
            if d.last_launch_at:
                days_ago = (datetime.utcnow() - d.last_launch_at.replace(tzinfo=None)).days
                last_launch = f"{days_ago}d ago"

            print(
                f"{i:>3} "
                f"{d.wallet_address:<44} "
                f"{d.deployer_score:>7.1f} "
                f"{d.total_launches:>8} "
                f"{d.graduation_rate * 100:>6.1f}% "
                f"{last_launch:>10}"
            )

    print(f"\n{'='*80}\n")


async def cmd_test_filters(args):
    """Test filter engine on a token."""
    from bags_sniper.core.config import get_settings
    from bags_sniper.core.deployer_intelligence import DeployerIntelligence
    from bags_sniper.core.filter_engine import FilterEngine, TokenCandidate
    from bags_sniper.models.database import init_database
    from bags_sniper.services.bags_api import BagsAPIClient
    from bags_sniper.services.solana_rpc import SolanaRPCClient

    settings = get_settings()
    session_factory = await init_database(settings.database_url.get_secret_value())

    async with BagsAPIClient(settings) as bags:
        async with SolanaRPCClient(settings) as rpc:
            deployer_intel = DeployerIntelligence(settings, rpc)
            filter_engine = FilterEngine(settings, deployer_intel, bags)

            candidate = TokenCandidate(
                mint_address=args.mint,
                deployer_wallet=args.deployer,
            )

            async with session_factory() as session:
                result = await filter_engine.evaluate(session, candidate)

    print(f"\n{'='*60}")
    print(f"FILTER TEST: {args.mint[:20]}...")
    print(f"{'='*60}")

    print(f"\nPASSED: {'YES' if result.passed_all_filters else 'NO'}")
    if result.rejection_reason:
        print(f"Rejection: {result.rejection_reason}")
    print(f"Total Filter Time: {result.total_filter_time_ms:.1f}ms")

    print("\nFILTER RESULTS:")
    for f in result.filter_results:
        status = "PASS" if f.result.value == "pass" else "FAIL"
        print(f"  [{status}] {f.filter_name}: {f.reason} ({f.latency_ms:.1f}ms)")

    if result.deployer_profile:
        print("\nDEPLOYER PROFILE:")
        print(f"  Score: {result.deployer_profile.deployer_score:.1f}")
        print(f"  Grad Rate: {result.deployer_profile.graduation_rate * 100:.2f}%")
        print(f"  Recommendation: {result.deployer_profile.recommendation}")

    print(f"\n{'='*60}\n")


async def cmd_backtest(args):
    """Run strategy backtest."""
    from datetime import timedelta

    from bags_sniper.backtest.analytics import PerformanceAnalytics
    from bags_sniper.backtest.engine import BacktestConfig, BacktestEngine
    from bags_sniper.models.database import init_database

    settings = get_settings()

    # Parse dates
    end_date = datetime.utcnow()
    if args.end_date:
        end_date = datetime.fromisoformat(args.end_date)

    if args.start_date:
        start_date = datetime.fromisoformat(args.start_date)
    else:
        start_date = end_date - timedelta(days=args.days)

    print(f"\n{'='*70}")
    print("STRATEGY BACKTEST")
    print(f"{'='*70}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Balance: {args.balance} SOL")
    print(f"Position Size: {args.position_size} SOL")
    print(f"Min Deployer Score: {args.min_score}")
    print(f"Min Graduation Rate: {args.min_grad_rate * 100:.1f}%")
    print(f"{'='*70}\n")

    session_factory = await init_database(settings.database_url.get_secret_value())

    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_balance_sol=Decimal(str(args.balance)),
        position_size_sol=Decimal(str(args.position_size)),
        min_deployer_score=args.min_score,
        min_graduation_rate=args.min_grad_rate,
    )

    engine = BacktestEngine(config, session_factory)
    state = await engine.run()

    analytics = PerformanceAnalytics(state, config)
    report = analytics.generate_report()
    print(report)

    # Export trades if requested
    if args.export:
        analytics.export_trades_csv(args.export)
        print(f"Trades exported to: {args.export}")


async def cmd_backtest_sweep(args):
    """Run parameter sweep to find optimal settings."""
    from datetime import timedelta

    from bags_sniper.backtest.simulator import TradeSimulator

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    print(f"\n{'='*70}")
    print("PARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Testing score thresholds: {args.scores}")
    print(f"Testing grad rate thresholds: {args.grad_rates}")
    print(f"{'='*70}\n")

    simulator = TradeSimulator()
    await simulator.initialize()

    results = await simulator.parameter_sweep(
        start_date=start_date,
        end_date=end_date,
        score_thresholds=[float(s) for s in args.scores.split(",")],
        grad_rate_thresholds=[float(g) for g in args.grad_rates.split(",")],
    )

    # Display results
    print(f"\n{'Score':>8} {'GradRate':>10} {'Trades':>8} {'WinRate':>10} {'Return':>10} {'Sharpe':>8}")
    print("-" * 60)

    for r in sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True):
        print(
            f"{r['min_score']:>8.0f} "
            f"{r['min_grad_rate']*100:>9.1f}% "
            f"{r['total_trades']:>8} "
            f"{r['win_rate']:>9.1f}% "
            f"{r['total_return_pct']:>9.1f}% "
            f"{r['sharpe_ratio']:>8.2f}"
        )

    # Highlight best
    best = max(results, key=lambda x: x["sharpe_ratio"])
    print("\nBEST PARAMETERS (by Sharpe):")
    print(f"  Min Score: {best['min_score']}")
    print(f"  Min Grad Rate: {best['min_grad_rate']*100:.1f}%")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")


async def cmd_health(args):
    """Check system health."""
    from bags_sniper.services.bags_api import BagsAPIClient
    from bags_sniper.services.solana_rpc import SolanaRPCClient

    settings = get_settings()

    print(f"\n{'='*60}")
    print("SYSTEM HEALTH CHECK")
    print(f"{'='*60}\n")

    # Check Bags API
    print("Bags.fm API: ", end="", flush=True)
    try:
        async with BagsAPIClient(settings) as bags:
            if await bags.health_check():
                print("OK")
            else:
                print("UNHEALTHY")
    except Exception as e:
        print(f"ERROR - {e}")

    # Check RPC
    print("Solana RPC: ", end="", flush=True)
    try:
        async with SolanaRPCClient(settings) as rpc:
            health = await rpc.health_check()
            healthy = sum(1 for v in health.values() if v.get("healthy"))
            print(f"OK ({healthy}/{len(health)} endpoints)")
            if args.verbose:
                for endpoint, status in health.items():
                    status_str = "OK" if status.get("healthy") else "FAIL"
                    latency = status.get("latency_ms", "N/A")
                    print(f"    {endpoint}: {status_str} ({latency}ms)")
    except Exception as e:
        print(f"ERROR - {e}")

    # Check Database
    print("Database: ", end="", flush=True)
    try:
        from bags_sniper.models.database import init_database

        await init_database(settings.database_url.get_secret_value())
        print("OK")
    except Exception as e:
        print(f"ERROR - {e}")

    print(f"\n{'='*60}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bags Sniper CLI - Deployer Intelligence Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Backfill command
    backfill_parser = subparsers.add_parser(
        "backfill", help="Run historical data backfill"
    )
    backfill_parser.add_argument(
        "--days", type=int, default=7, help="Days of history to fetch (default: 7)"
    )
    backfill_parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Max tokens to process (default: 500)",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # Check deployer command
    check_parser = subparsers.add_parser("check", help="Check a specific deployer")
    check_parser.add_argument("wallet", help="Deployer wallet address")
    check_parser.add_argument(
        "--refresh", action="store_true", help="Force refresh stats"
    )

    # Top deployers command
    top_parser = subparsers.add_parser("top", help="List top deployers")
    top_parser.add_argument(
        "--limit", type=int, default=20, help="Number of deployers (default: 20)"
    )
    top_parser.add_argument(
        "--min-score", type=float, default=50, help="Minimum score (default: 50)"
    )
    top_parser.add_argument(
        "--min-launches", type=int, default=3, help="Minimum launches (default: 3)"
    )

    # Test filters command
    test_parser = subparsers.add_parser("test-filters", help="Test filters on a token")
    test_parser.add_argument("mint", help="Token mint address")
    test_parser.add_argument("deployer", help="Deployer wallet address")

    # Health command (no additional arguments needed)
    subparsers.add_parser("health", help="Check system health")

    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run strategy backtest")
    backtest_parser.add_argument(
        "--days", type=int, default=7, help="Days to backtest (default: 7)"
    )
    backtest_parser.add_argument(
        "--start-date", type=str, help="Start date (ISO format, e.g., 2024-01-01)"
    )
    backtest_parser.add_argument(
        "--end-date", type=str, help="End date (ISO format)"
    )
    backtest_parser.add_argument(
        "--balance", type=float, default=10.0, help="Initial balance in SOL (default: 10)"
    )
    backtest_parser.add_argument(
        "--position-size", type=float, default=0.3, help="Position size in SOL (default: 0.3)"
    )
    backtest_parser.add_argument(
        "--min-score", type=float, default=60.0, help="Min deployer score (default: 60)"
    )
    backtest_parser.add_argument(
        "--min-grad-rate", type=float, default=0.042, help="Min graduation rate (default: 0.042)"
    )
    backtest_parser.add_argument(
        "--export", type=str, help="Export trades to CSV file"
    )

    # Parameter sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run parameter sweep optimization")
    sweep_parser.add_argument(
        "--days", type=int, default=14, help="Days to analyze (default: 14)"
    )
    sweep_parser.add_argument(
        "--scores", type=str, default="40,50,60,70,80",
        help="Comma-separated score thresholds to test"
    )
    sweep_parser.add_argument(
        "--grad-rates", type=str, default="0.02,0.03,0.042,0.05",
        help="Comma-separated graduation rates to test"
    )

    # Run command (starts the bot)
    run_parser = subparsers.add_parser("run", help="Start the trading bot")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in simulation mode (default: True)",
    )

    args = parser.parse_args()
    configure_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return

    # Route to appropriate command
    commands = {
        "backfill": cmd_backfill,
        "stats": cmd_stats,
        "check": cmd_check_deployer,
        "top": cmd_top_deployers,
        "test-filters": cmd_test_filters,
        "health": cmd_health,
        "backtest": cmd_backtest,
        "sweep": cmd_backtest_sweep,
    }

    if args.command == "run":
        from bags_sniper.main import main as run_bot

        run_bot()
    elif args.command in commands:
        asyncio.run(commands[args.command](args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
