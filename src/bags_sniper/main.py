"""
Main entry point for the Bags Sniper bot.
"""

import asyncio
import logging
import os
import signal
import subprocess
import sys
from typing import Optional

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

from bags_sniper.core.config import get_settings


def cleanup_existing_processes():
    """
    Kill any existing bags_sniper processes before starting a new instance.
    This ensures only one bot instance runs at a time.
    """
    current_pid = os.getpid()

    if sys.platform == "win32":
        # Windows: Use WMIC to find and kill Python processes running bags_sniper
        try:
            # Find processes
            result = subprocess.run(
                ['wmic', 'process', 'where',
                 "commandline like '%bags_sniper%' and name='python.exe'",
                 'get', 'processid'],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Parse PIDs from output
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line.isdigit():
                    pid = int(line)
                    if pid != current_pid:
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                                         capture_output=True, timeout=5)
                            print(f"Killed existing bot process: PID {pid}")
                        except Exception:
                            pass
        except Exception as e:
            # Silently continue if cleanup fails
            pass
    else:
        # Unix: Use pkill or pgrep
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'bags_sniper'],
                capture_output=True,
                text=True
            )
            for line in result.stdout.strip().split('\n'):
                if line.strip().isdigit():
                    pid = int(line.strip())
                    if pid != current_pid:
                        try:
                            os.kill(pid, signal.SIGTERM)
                            print(f"Killed existing bot process: PID {pid}")
                        except ProcessLookupError:
                            pass
        except Exception:
            pass


from bags_sniper.core.trading_engine import TradingEngine
from bags_sniper.models.database import init_database
from bags_sniper.services.bags_api import BagsAPIClient
from bags_sniper.services.solana_rpc import SolanaRPCClient
from bags_sniper.services.telegram_bot import TelegramBot


def configure_logging(log_level: str):
    """Configure structured logging."""
    # Convert string log level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_log_level,
            TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            JSONRenderer() if log_level.upper() != "DEBUG" else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


class BagsSniper:
    """Main application class."""

    def __init__(self):
        self.settings = get_settings()
        self.engine: Optional[TradingEngine] = None
        self.bags_client: Optional[BagsAPIClient] = None
        self.rpc_client: Optional[SolanaRPCClient] = None
        self.telegram: Optional[TelegramBot] = None
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start the bot."""
        logger = structlog.get_logger()
        logger.info(
            "bags_sniper_starting",
            version="0.1.0",
            dry_run=self.settings.dry_run,
        )

        # Initialize database
        logger.info("initializing_database")
        session_factory = await init_database(
            self.settings.database_url.get_secret_value()
        )

        # Initialize clients
        self.bags_client = BagsAPIClient(self.settings)
        await self.bags_client.connect()

        self.rpc_client = SolanaRPCClient(self.settings)
        await self.rpc_client.connect()

        # Initialize Telegram
        self.telegram = TelegramBot(self.settings)
        await self.telegram.start()

        # Initialize trading engine
        self.engine = TradingEngine(
            settings=self.settings,
            session_factory=session_factory,
            bags_client=self.bags_client,
            rpc_client=self.rpc_client,
            telegram=self.telegram,
        )

        # Set up signal handlers (Unix only - Windows uses KeyboardInterrupt)
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_event_loop().add_signal_handler(
                    sig, lambda: asyncio.create_task(self.shutdown())
                )

        # Start the engine
        try:
            await self.engine.start()
        except asyncio.CancelledError:
            pass

        # Wait for shutdown
        await self._shutdown_event.wait()

    async def shutdown(self):
        """Graceful shutdown."""
        logger = structlog.get_logger()
        logger.info("shutdown_initiated")

        if self.engine:
            await self.engine.stop()

        if self.telegram:
            await self.telegram.stop()

        if self.bags_client:
            await self.bags_client.close()

        if self.rpc_client:
            await self.rpc_client.close()

        self._shutdown_event.set()
        logger.info("shutdown_complete")


def main():
    """Main entry point."""
    import traceback

    # Kill any existing bot instances before starting
    cleanup_existing_processes()

    settings = get_settings()
    configure_logging(settings.log_level)

    logger = structlog.get_logger()

    if settings.dry_run:
        logger.warning("RUNNING IN DRY RUN MODE - No real trades will be executed")

    app = BagsSniper()

    try:
        asyncio.run(app.start())
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
    except Exception as e:
        logger.error("fatal_error", error=str(e), traceback=traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
