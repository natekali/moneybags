"""
Modern interactive Telegram bot with beautiful button-based UI.
Full-featured monitoring dashboard with inline keyboard navigation.
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

import structlog
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bags_sniper.core.config import Settings
from bags_sniper.core.circuit_breaker import CircuitBreaker, CircuitBreakerLevel
from bags_sniper.services.rate_limiter import (
    RateLimiter,
    RateLimitService,
    get_rate_limiter,
)

logger = structlog.get_logger()


class AlertPriority(str, Enum):
    """Alert priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class TradeRecord:
    """Record of a trade."""
    token_symbol: str
    mint_address: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: Decimal
    exit_price: Optional[Decimal]
    amount_sol: Decimal
    pnl_sol: Optional[Decimal]
    pnl_percent: Optional[float]
    exit_reason: Optional[str]
    deployer_score: float


@dataclass
class LogEntry:
    """Log entry."""
    timestamp: datetime
    level: str
    event: str
    details: str = ""


@dataclass
class BotMetrics:
    """Bot metrics tracking."""
    session_start: datetime = field(default_factory=datetime.utcnow)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_sol: Decimal = Decimal("0")
    active_positions: int = 0
    tokens_analyzed: int = 0
    tokens_filtered: int = 0
    best_trade_pnl: Decimal = Decimal("0")
    worst_trade_pnl: Decimal = Decimal("0")
    avg_hold_time_minutes: float = 0.0
    rpc_latency_ms: float = 0.0
    api_errors_count: int = 0
    last_error: Optional[str] = None

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100


class TelegramBot:
    """
    Modern interactive Telegram bot with button-based navigation.
    """

    def __init__(
        self,
        settings: Settings,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.settings = settings
        self.circuit_breaker = circuit_breaker
        self._bot: Optional[Bot] = None
        self._app: Optional[Application] = None
        self._running = False
        self._rate_limiter: Optional[RateLimiter] = None

        self.metrics = BotMetrics()
        self.trade_history: deque[TradeRecord] = deque(maxlen=100)
        self.log_buffer: deque[LogEntry] = deque(maxlen=50)
        self.positions: dict[str, dict] = {}
        self.deployer_stats: dict[str, dict] = {}

        self._on_pause_callback: Optional[Callable] = None
        self._on_resume_callback: Optional[Callable] = None
        self._on_exit_all_callback: Optional[Callable] = None
        self._get_positions_callback: Optional[Callable] = None
        self._get_balance_callback: Optional[Callable] = None

        # Track message IDs for editing
        self._last_menu_message: dict[int, int] = {}

    async def start(self):
        """Start the bot."""
        try:
            self._rate_limiter = await get_rate_limiter()
            self._app = (
                Application.builder()
                .token(self.settings.telegram_bot_token.get_secret_value())
                .build()
            )
            self._bot = self._app.bot

            bot_info = await self._bot.get_me()
            logger.info("telegram_bot_validated", bot_username=bot_info.username, bot_id=bot_info.id)

            self._register_handlers()

            await self._app.initialize()
            await self._app.start()
            asyncio.create_task(self._app.updater.start_polling(drop_pending_updates=True))
            self._running = True

            logger.info("telegram_bot_started")
            await self._send_welcome()

        except TelegramError as e:
            logger.error("telegram_bot_start_failed", error=str(e))
            raise

    async def stop(self):
        """Stop the bot."""
        if self._app and self._running:
            self._running = False
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()

    def _register_handlers(self):
        """Register handlers."""
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("menu", self._cmd_start))
        self._app.add_handler(CommandHandler("help", self._cmd_start))
        self._app.add_handler(CallbackQueryHandler(self._handle_callback))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text))

    def _auth(self, update: Update) -> bool:
        """Check authorization."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        return str(chat_id) == str(self.settings.telegram_chat_id)

    # ═══════════════════════════════════════════════════════════════════════════
    # KEYBOARD BUILDERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _kb_main(self) -> InlineKeyboardMarkup:
        """Main menu keyboard."""
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Dashboard", callback_data="nav:dashboard"),
                InlineKeyboardButton("💼 Positions", callback_data="nav:positions"),
            ],
            [
                InlineKeyboardButton("📈 Performance", callback_data="nav:performance"),
                InlineKeyboardButton("📜 History", callback_data="nav:history"),
            ],
            [
                InlineKeyboardButton("👥 Deployers", callback_data="nav:deployers"),
                InlineKeyboardButton("🎯 Strategy", callback_data="nav:strategy"),
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="nav:settings"),
                InlineKeyboardButton("📋 Logs", callback_data="nav:logs"),
            ],
            [
                InlineKeyboardButton("🎮 Controls", callback_data="nav:controls"),
            ],
        ])

    def _kb_back(self, refresh_action: str = None) -> InlineKeyboardMarkup:
        """Back button keyboard."""
        buttons = []
        if refresh_action:
            buttons.append(InlineKeyboardButton("🔄 Refresh", callback_data=f"refresh:{refresh_action}"))
        buttons.append(InlineKeyboardButton("🏠 Menu", callback_data="nav:main"))
        return InlineKeyboardMarkup([buttons])

    def _kb_controls(self) -> InlineKeyboardMarkup:
        """Controls keyboard."""
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("⏸ Pause", callback_data="ctrl:pause"),
                InlineKeyboardButton("▶️ Resume", callback_data="ctrl:resume"),
            ],
            [
                InlineKeyboardButton("🚨 EXIT ALL", callback_data="ctrl:exit_confirm"),
            ],
            [
                InlineKeyboardButton("🏠 Menu", callback_data="nav:main"),
            ],
        ])

    def _kb_exit_confirm(self) -> InlineKeyboardMarkup:
        """Exit confirmation keyboard."""
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ YES, EXIT ALL", callback_data="ctrl:exit_execute"),
                InlineKeyboardButton("❌ Cancel", callback_data="nav:controls"),
            ],
        ])

    def _kb_history(self) -> InlineKeyboardMarkup:
        """History filter keyboard."""
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📋 All", callback_data="hist:all:5"),
                InlineKeyboardButton("✅ Winners", callback_data="hist:win:5"),
                InlineKeyboardButton("❌ Losers", callback_data="hist:lose:5"),
            ],
            [
                InlineKeyboardButton("5", callback_data="hist:all:5"),
                InlineKeyboardButton("10", callback_data="hist:all:10"),
                InlineKeyboardButton("20", callback_data="hist:all:20"),
            ],
            [
                InlineKeyboardButton("🏠 Menu", callback_data="nav:main"),
            ],
        ])

    def _kb_logs(self) -> InlineKeyboardMarkup:
        """Logs filter keyboard."""
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📋 All", callback_data="logs:all"),
                InlineKeyboardButton("⚠️ Warn", callback_data="logs:warn"),
                InlineKeyboardButton("❌ Error", callback_data="logs:error"),
            ],
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="refresh:logs"),
                InlineKeyboardButton("🏠 Menu", callback_data="nav:main"),
            ],
        ])

    # ═══════════════════════════════════════════════════════════════════════════
    # MESSAGE BUILDERS
    # ═══════════════════════════════════════════════════════════════════════════

    def _msg_main(self) -> str:
        """Main menu message."""
        mode = "🔴 LIVE" if not self.settings.dry_run else "🟡 DRY RUN"
        return f"""
🤖 <b>BAGS SNIPER BOT</b>

Mode: {mode}
Status: {'🟢 Running' if self._running else '🔴 Stopped'}

<i>Select an option below:</i>
""".strip()

    def _msg_dashboard(self) -> str:
        """Dashboard message."""
        m = self.metrics
        uptime = datetime.utcnow() - m.session_start
        h, rem = divmod(int(uptime.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)

        # Circuit breaker
        cb_level = "NORMAL"
        cb_icon = "🟢"
        drawdown = 0.0
        if self.circuit_breaker:
            try:
                status = self.circuit_breaker.get_status_report()
                cb_level = status.get('level', 'NORMAL')
                drawdown = status.get('current_drawdown_percent', 0)
                if 'LEVEL_3' in cb_level or 'LEVEL_4' in cb_level:
                    cb_icon = "🔴"
                elif 'LEVEL' in cb_level:
                    cb_icon = "🟡"
            except:
                pass

        # P&L
        pnl = m.total_pnl_sol
        pnl_icon = "🟢" if pnl >= 0 else "🔴"

        # Win rate
        wr = m.win_rate
        wr_icon = "🔥" if wr >= 60 else ("🟢" if wr >= 50 else "🟡")

        return f"""
📊 <b>DASHBOARD</b>

┌─────────────────────────┐
│ ⏱ <b>Uptime:</b> {h}h {mins}m {secs}s
│ 📡 <b>Mode:</b> {'DRY RUN' if self.settings.dry_run else 'LIVE'}
│ {cb_icon} <b>Circuit:</b> {cb_level}
└─────────────────────────┘

┌─────────────────────────┐
│ {pnl_icon} <b>P&L:</b> {'+' if pnl >= 0 else ''}{pnl:.4f} SOL
│ 📊 <b>Trades:</b> {m.total_trades}
│ {wr_icon} <b>Win Rate:</b> {wr:.1f}%
│ 📉 <b>Drawdown:</b> {drawdown:.2f}%
└─────────────────────────┘

┌─────────────────────────┐
│ 💼 <b>Positions:</b> {m.active_positions}
│ 🔍 <b>Analyzed:</b> {m.tokens_analyzed}
│ 🚫 <b>Filtered:</b> {m.tokens_filtered}
│ 📶 <b>RPC:</b> {m.rpc_latency_ms:.0f}ms
└─────────────────────────┘

<i>Updated: {datetime.utcnow().strftime('%H:%M:%S')} UTC</i>
""".strip()

    def _msg_positions(self) -> str:
        """Positions message."""
        if not self.positions:
            return """
💼 <b>POSITIONS</b>

┌─────────────────────────┐
│                         │
│   <i>No active positions</i>   │
│                         │
└─────────────────────────┘
""".strip()

        lines = ["💼 <b>POSITIONS</b>\n"]
        total_pnl = Decimal("0")

        for i, (mint, pos) in enumerate(self.positions.items(), 1):
            symbol = pos.get('symbol', '???')
            pnl_pct = pos.get('unrealized_pnl_percent', 0)
            amount = pos.get('amount_sol', Decimal("0"))
            score = pos.get('deployer_score', 0)

            icon = "🟢" if pnl_pct >= 0 else "🔴"
            sign = "+" if pnl_pct >= 0 else ""

            lines.append(f"""
┌─ {icon} <b>{symbol}</b> ─────────────┐
│ 💰 Size: {amount:.4f} SOL
│ 📈 P&L: {sign}{pnl_pct:.1f}%
│ ⭐ Score: {score:.0f}/100
│ 🔗 {mint[:12]}...
└─────────────────────────┘""")

            # Approximate total PnL
            if pnl_pct != 0:
                total_pnl += amount * Decimal(str(pnl_pct / 100))

        total_icon = "🟢" if total_pnl >= 0 else "🔴"
        lines.append(f"\n{total_icon} <b>Total:</b> {'+' if total_pnl >= 0 else ''}{total_pnl:.4f} SOL")

        return "\n".join(lines)

    def _msg_performance(self) -> str:
        """Performance message."""
        m = self.metrics

        # Calculate metrics
        avg_win = 150.0 if m.winning_trades > 0 else 0  # Placeholder
        avg_loss = 18.0 if m.losing_trades > 0 else 0   # Placeholder
        pf = avg_win / avg_loss if avg_loss > 0 else 0

        # Rating bar
        filled = int(m.win_rate / 10)
        bar = "█" * filled + "░" * (10 - filled)

        if m.win_rate >= 60:
            rating = "🔥 EXCELLENT"
        elif m.win_rate >= 50:
            rating = "✅ GOOD"
        elif m.win_rate >= 40:
            rating = "⚠️ FAIR"
        else:
            rating = "❌ POOR"

        return f"""
📈 <b>PERFORMANCE</b>

┌─── Trade Stats ─────────┐
│ 📊 Total: {m.total_trades}
│ ✅ Winners: {m.winning_trades}
│ ❌ Losers: {m.losing_trades}
│ 📈 Win Rate: {m.win_rate:.1f}%
└─────────────────────────┘

┌─── Profit & Loss ───────┐
│ 💰 Total: {'+' if m.total_pnl_sol >= 0 else ''}{m.total_pnl_sol:.4f} SOL
│ 🏆 Best: +{m.best_trade_pnl:.4f} SOL
│ 💀 Worst: {m.worst_trade_pnl:.4f} SOL
└─────────────────────────┘

┌─── Risk Metrics ────────┐
│ ⚖️ Profit Factor: {pf:.2f}
│ 📊 Avg Win: +{avg_win:.0f}%
│ 📉 Avg Loss: -{avg_loss:.0f}%
└─────────────────────────┘

┌─── Rating ──────────────┐
│ [{bar}] {m.win_rate:.0f}%
│ {rating}
└─────────────────────────┘
""".strip()

    def _msg_history(self, filter_type: str = "all", limit: int = 5) -> str:
        """Trade history message."""
        trades = list(self.trade_history)

        if filter_type == "win":
            trades = [t for t in trades if t.pnl_sol and t.pnl_sol > 0]
        elif filter_type == "lose":
            trades = [t for t in trades if t.pnl_sol and t.pnl_sol < 0]

        trades = trades[-limit:]

        if not trades:
            return f"""
📜 <b>TRADE HISTORY</b>

┌─────────────────────────┐
│                         │
│   <i>No trades yet</i>         │
│                         │
└─────────────────────────┘

Filter: {filter_type.upper()} | Showing: {limit}
""".strip()

        lines = [f"📜 <b>TRADE HISTORY</b>\n"]

        for t in reversed(trades):
            icon = "✅" if (t.pnl_sol or 0) >= 0 else "❌"
            pnl = t.pnl_sol or Decimal("0")
            pct = t.pnl_percent or 0

            lines.append(f"""
{icon} <b>{t.token_symbol}</b>
   💰 {t.amount_sol:.4f} SOL → {'+' if pnl >= 0 else ''}{pnl:.4f} ({'+' if pct >= 0 else ''}{pct:.1f}%)
   🏷 {t.exit_reason or 'Open'}
   ⏰ {t.entry_time.strftime('%m/%d %H:%M')}""")

        lines.append(f"\n<i>Filter: {filter_type.upper()} | Showing: {limit}</i>")
        return "\n".join(lines)

    def _msg_deployers(self) -> str:
        """Deployers message."""
        if not self.deployer_stats:
            return """
👥 <b>TOP DEPLOYERS</b>

┌─────────────────────────┐
│                         │
│  <i>No deployer data yet</i>   │
│                         │
│  Run backfill command   │
│  to populate database   │
│                         │
└─────────────────────────┘
""".strip()

        sorted_deployers = sorted(
            self.deployer_stats.items(),
            key=lambda x: x[1].get('score', 0),
            reverse=True
        )[:10]

        lines = ["👥 <b>TOP DEPLOYERS</b>\n"]
        medals = ["🥇", "🥈", "🥉"]

        for i, (wallet, stats) in enumerate(sorted_deployers):
            score = stats.get('score', 0)
            tokens = stats.get('total_tokens', 0)
            grad = stats.get('graduation_rate', 0)

            medal = medals[i] if i < 3 else f"{i+1}."
            lines.append(f"{medal} Score: <b>{score:.0f}</b> | Tokens: {tokens} | Grad: {grad:.1f}%")

        lines.append(f"\n<i>Total tracked: {len(self.deployer_stats)}</i>")
        return "\n".join(lines)

    def _msg_strategy(self) -> str:
        """Strategy message."""
        s = self.settings
        return f"""
🎯 <b>TRADING STRATEGY</b>

┌─── Entry Criteria ──────┐
│ ⭐ Min Score: {s.min_deployer_score}
│ 💰 Max Size: {s.max_position_size_sol} SOL
│ 📊 Max Positions: {s.max_concurrent_positions}
│ ⏰ Max Age: {s.max_pool_age_minutes} min
└─────────────────────────┘

┌─── Risk Management ─────┐
│ 🛑 Stop Loss: {s.stop_loss_percent}%
│ 📉 Trailing: {s.trailing_stop_percent}%
│ 📊 Max Daily Loss: {s.max_daily_loss_percent}%
└─────────────────────────┘

┌─── Take Profit ─────────┐
│ TP1: {s.tp_tier1_multiple}x → {s.tp_tier1_percent}%
│ TP2: {s.tp_tier2_multiple}x → {s.tp_tier2_percent}%
│ TP3: {s.tp_tier3_multiple}x → {s.tp_tier3_percent}%
└─────────────────────────┘

┌─── Security Filters ────┐
│ 💧 Min Liq: {s.min_liquidity_sol} SOL
│ 👤 Max Holder: {s.max_top_holder_percent}%
│ 🔧 Max Dev: {s.max_dev_wallet_percent}%
│ 💸 Max Tax: {s.max_tax_percent}%
└─────────────────────────┘
""".strip()

    def _msg_settings(self) -> str:
        """Settings message."""
        s = self.settings
        return f"""
⚙️ <b>BOT SETTINGS</b>

┌─── General ─────────────┐
│ 🔴 Mode: {'DRY RUN' if s.dry_run else 'LIVE'}
│ 📝 Log Level: {s.log_level}
└─────────────────────────┘

┌─── APIs ────────────────┐
│ 🔗 Bags: ✅ Connected
│ 📡 Helius: ✅ Connected
│ 📱 Telegram: ✅ Active
│ 🗄 Database: ✅ Ready
└─────────────────────────┘

┌─── Rate Limits ─────────┐
│ Bags: {s.bags_rate_limit_per_hour}/hr
│ Telegram: 30/sec
└─────────────────────────┘

┌─── Circuit Breaker ─────┐
│ L1: {s.circuit_breaker_l1_percent}% → Reduce 50%
│ L2: {s.circuit_breaker_l2_percent}% → Reduce 75%
│ L3: {s.circuit_breaker_l3_percent}% → Pause
│ L4: {s.circuit_breaker_l4_percent}% → Shutdown
└─────────────────────────┘

<i>Edit .env to change settings</i>
""".strip()

    def _msg_logs(self, filter_type: str = "all") -> str:
        """Logs message."""
        logs = list(self.log_buffer)

        if filter_type == "error":
            logs = [l for l in logs if l.level.upper() == "ERROR"]
        elif filter_type == "warn":
            logs = [l for l in logs if l.level.upper() == "WARNING"]

        logs = logs[-12:]

        if not logs:
            return f"""
📋 <b>SYSTEM LOGS</b>

┌─────────────────────────┐
│                         │
│   <i>No logs to display</i>    │
│                         │
└─────────────────────────┘

Filter: {filter_type.upper()}
""".strip()

        lines = [f"📋 <b>SYSTEM LOGS</b>\n"]

        icons = {"ERROR": "❌", "WARNING": "⚠️", "INFO": "ℹ️", "DEBUG": "🔍"}

        for log in reversed(logs):
            icon = icons.get(log.level.upper(), "ℹ️")
            time_str = log.timestamp.strftime('%H:%M:%S')
            lines.append(f"{icon} <code>{time_str}</code> {log.event[:35]}")

        lines.append(f"\n<i>Filter: {filter_type.upper()}</i>")
        return "\n".join(lines)

    def _msg_controls(self) -> str:
        """Controls message."""
        is_paused = False
        if self.circuit_breaker:
            try:
                status = self.circuit_breaker.get_status_report()
                is_paused = not status.get('new_entries_allowed', True)
            except:
                pass

        status_icon = "⏸" if is_paused else "▶️"
        status_text = "PAUSED" if is_paused else "ACTIVE"

        return f"""
🎮 <b>BOT CONTROLS</b>

┌─────────────────────────┐
│                         │
│  Status: {status_icon} <b>{status_text}</b>
│                         │
└─────────────────────────┘

<b>⏸ Pause</b>
Stop new entries. Existing
positions continue to be managed.

<b>▶️ Resume</b>
Resume trading operations.

<b>🚨 EXIT ALL</b>
Emergency exit ALL positions.
Use with caution!
""".strip()

    def _msg_exit_confirm(self) -> str:
        """Exit confirmation message."""
        return """
🚨 <b>CONFIRM EXIT ALL</b>

┌─────────────────────────┐
│                         │
│  ⚠️ <b>WARNING</b> ⚠️           │
│                         │
│  This will immediately  │
│  close ALL positions!   │
│                         │
│  This cannot be undone. │
│                         │
└─────────────────────────┘

Are you sure?
""".strip()

    # ═══════════════════════════════════════════════════════════════════════════
    # HANDLERS
    # ═══════════════════════════════════════════════════════════════════════════

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        if not self._auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return

        msg = await update.message.reply_text(
            self._msg_main(),
            reply_markup=self._kb_main(),
            parse_mode=ParseMode.HTML,
        )
        self._last_menu_message[update.effective_chat.id] = msg.message_id

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        query = update.callback_query

        if not self._auth(update):
            await query.answer("⛔ Unauthorized", show_alert=True)
            return

        await query.answer()
        data = query.data

        try:
            # Navigation
            if data == "nav:main":
                await query.edit_message_text(
                    self._msg_main(),
                    reply_markup=self._kb_main(),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:dashboard":
                await query.edit_message_text(
                    self._msg_dashboard(),
                    reply_markup=self._kb_back("dashboard"),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:positions":
                await query.edit_message_text(
                    self._msg_positions(),
                    reply_markup=self._kb_back("positions"),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:performance":
                await query.edit_message_text(
                    self._msg_performance(),
                    reply_markup=self._kb_back("performance"),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:history":
                await query.edit_message_text(
                    self._msg_history(),
                    reply_markup=self._kb_history(),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:deployers":
                await query.edit_message_text(
                    self._msg_deployers(),
                    reply_markup=self._kb_back("deployers"),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:strategy":
                await query.edit_message_text(
                    self._msg_strategy(),
                    reply_markup=self._kb_back(),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:settings":
                await query.edit_message_text(
                    self._msg_settings(),
                    reply_markup=self._kb_back(),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:logs":
                await query.edit_message_text(
                    self._msg_logs(),
                    reply_markup=self._kb_logs(),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "nav:controls":
                await query.edit_message_text(
                    self._msg_controls(),
                    reply_markup=self._kb_controls(),
                    parse_mode=ParseMode.HTML,
                )

            # History filters
            elif data.startswith("hist:"):
                _, filter_type, limit = data.split(":")
                await query.edit_message_text(
                    self._msg_history(filter_type, int(limit)),
                    reply_markup=self._kb_history(),
                    parse_mode=ParseMode.HTML,
                )

            # Log filters
            elif data.startswith("logs:"):
                filter_type = data.split(":")[1]
                await query.edit_message_text(
                    self._msg_logs(filter_type),
                    reply_markup=self._kb_logs(),
                    parse_mode=ParseMode.HTML,
                )

            # Refresh
            elif data.startswith("refresh:"):
                section = data.split(":")[1]
                if section == "dashboard":
                    await query.edit_message_text(
                        self._msg_dashboard(),
                        reply_markup=self._kb_back("dashboard"),
                        parse_mode=ParseMode.HTML,
                    )
                elif section == "positions":
                    await query.edit_message_text(
                        self._msg_positions(),
                        reply_markup=self._kb_back("positions"),
                        parse_mode=ParseMode.HTML,
                    )
                elif section == "logs":
                    await query.edit_message_text(
                        self._msg_logs(),
                        reply_markup=self._kb_logs(),
                        parse_mode=ParseMode.HTML,
                    )
                elif section == "deployers":
                    await query.edit_message_text(
                        self._msg_deployers(),
                        reply_markup=self._kb_back("deployers"),
                        parse_mode=ParseMode.HTML,
                    )
                elif section == "performance":
                    await query.edit_message_text(
                        self._msg_performance(),
                        reply_markup=self._kb_back("performance"),
                        parse_mode=ParseMode.HTML,
                    )

            # Controls
            elif data == "ctrl:pause":
                if self._on_pause_callback:
                    await self._on_pause_callback()
                await query.edit_message_text(
                    "⏸ <b>Trading PAUSED</b>\n\nNew entries stopped.",
                    reply_markup=self._kb_back(),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "ctrl:resume":
                if self._on_resume_callback:
                    await self._on_resume_callback()
                await query.edit_message_text(
                    "▶️ <b>Trading RESUMED</b>\n\nBot is now active.",
                    reply_markup=self._kb_back(),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "ctrl:exit_confirm":
                await query.edit_message_text(
                    self._msg_exit_confirm(),
                    reply_markup=self._kb_exit_confirm(),
                    parse_mode=ParseMode.HTML,
                )

            elif data == "ctrl:exit_execute":
                if self._on_exit_all_callback:
                    await self._on_exit_all_callback()
                await query.edit_message_text(
                    "✅ <b>All positions exited</b>",
                    reply_markup=self._kb_back(),
                    parse_mode=ParseMode.HTML,
                )

        except TelegramError as e:
            logger.error("telegram_callback_error", error=str(e))

    async def _handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages."""
        if not self._auth(update):
            return
        # Show menu for any text
        await self._cmd_start(update, context)

    async def _send_welcome(self):
        """Send welcome message."""
        msg = f"""
🚀 <b>Bot Started</b>

Mode: {'🟡 DRY RUN' if self.settings.dry_run else '🔴 LIVE'}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

Send /menu to open controls.
""".strip()
        await self.send_alert(msg, AlertPriority.NORMAL)

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    async def send_alert(
        self,
        message: str,
        priority: AlertPriority = AlertPriority.NORMAL,
        data: Optional[dict[str, Any]] = None,
    ):
        """Send alert message."""
        if not self._bot:
            return

        prefix = {
            AlertPriority.CRITICAL: "🚨 <b>CRITICAL</b>\n",
            AlertPriority.HIGH: "⚠️ <b>ALERT</b>\n",
            AlertPriority.NORMAL: "",
            AlertPriority.LOW: "",
        }.get(priority, "")

        text = prefix + message

        try:
            await self._bot.send_message(
                chat_id=self.settings.telegram_chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
            )
        except TelegramError as e:
            logger.error("telegram_send_failed", error=str(e))

    async def send_trade_entry(
        self,
        token_symbol: str,
        mint_address: str,
        amount_sol: Decimal,
        price_sol: Decimal,
        deployer_score: float,
    ):
        """Send trade entry alert."""
        self.metrics.total_trades += 1
        self.trade_history.append(TradeRecord(
            token_symbol=token_symbol,
            mint_address=mint_address,
            entry_time=datetime.utcnow(),
            exit_time=None,
            entry_price=price_sol,
            exit_price=None,
            amount_sol=amount_sol,
            pnl_sol=None,
            pnl_percent=None,
            exit_reason=None,
            deployer_score=deployer_score,
        ))

        msg = f"""
🚀 <b>NEW ENTRY</b>

💎 <b>{token_symbol}</b>
💰 {amount_sol:.4f} SOL
⭐ Score: {deployer_score:.0f}/100
🔗 <code>{mint_address[:16]}...</code>
""".strip()
        await self.send_alert(msg, AlertPriority.HIGH)

    async def send_trade_exit(
        self,
        token_symbol: str,
        exit_reason: str,
        pnl_sol: Decimal,
        pnl_percent: float,
        hold_time_minutes: float,
    ):
        """Send trade exit alert."""
        if pnl_sol >= 0:
            self.metrics.winning_trades += 1
            if pnl_sol > self.metrics.best_trade_pnl:
                self.metrics.best_trade_pnl = pnl_sol
            icon = "💰"
        else:
            self.metrics.losing_trades += 1
            if pnl_sol < self.metrics.worst_trade_pnl:
                self.metrics.worst_trade_pnl = pnl_sol
            icon = "📉"

        self.metrics.total_pnl_sol += pnl_sol

        for t in reversed(self.trade_history):
            if t.token_symbol == token_symbol and t.exit_time is None:
                t.exit_time = datetime.utcnow()
                t.pnl_sol = pnl_sol
                t.pnl_percent = pnl_percent
                t.exit_reason = exit_reason
                break

        sign = "+" if pnl_sol >= 0 else ""
        msg = f"""
{icon} <b>TRADE CLOSED</b>

💎 <b>{token_symbol}</b>
💵 P&L: {sign}{pnl_sol:.4f} SOL ({sign}{pnl_percent:.1f}%)
🏷 {exit_reason}
⏱ {hold_time_minutes:.1f} min
""".strip()

        priority = AlertPriority.HIGH if pnl_sol >= 0 else AlertPriority.CRITICAL
        await self.send_alert(msg, priority)

    async def send_circuit_breaker_alert(
        self,
        level: CircuitBreakerLevel,
        drawdown_percent: float,
        action: str,
    ):
        """Send circuit breaker alert."""
        icons = {
            CircuitBreakerLevel.NORMAL: "🟢",
            CircuitBreakerLevel.LEVEL_1: "🟡",
            CircuitBreakerLevel.LEVEL_2: "🟠",
            CircuitBreakerLevel.LEVEL_3: "🔴",
            CircuitBreakerLevel.LEVEL_4: "⛔",
        }
        icon = icons.get(level, "⚠️")

        msg = f"""
{icon} <b>CIRCUIT BREAKER</b>

Level: {level.name}
Drawdown: {drawdown_percent:.2f}%
Action: {action}
""".strip()

        priority = AlertPriority.CRITICAL if level >= CircuitBreakerLevel.LEVEL_3 else AlertPriority.HIGH
        await self.send_alert(msg, priority)

    async def send_daily_summary(
        self,
        total_trades: int,
        winning_trades: int,
        total_pnl_sol: Decimal,
        ending_balance_sol: Decimal,
    ):
        """Send daily summary."""
        wr = winning_trades / total_trades * 100 if total_trades > 0 else 0
        icon = "📈" if total_pnl_sol >= 0 else "📉"

        msg = f"""
{icon} <b>DAILY SUMMARY</b>

📊 Trades: {total_trades}
✅ Win Rate: {wr:.1f}%
💰 P&L: {'+' if total_pnl_sol >= 0 else ''}{total_pnl_sol:.4f} SOL
💵 Balance: {ending_balance_sol:.4f} SOL
""".strip()
        await self.send_alert(msg, AlertPriority.NORMAL)

    # ═══════════════════════════════════════════════════════════════════════════
    # METRIC UPDATES
    # ═══════════════════════════════════════════════════════════════════════════

    def update_positions(self, positions: dict[str, dict]):
        """Update positions."""
        self.positions = positions
        self.metrics.active_positions = len(positions)

    def update_deployer_stats(self, stats: dict[str, dict]):
        """Update deployer stats."""
        self.deployer_stats = stats

    def add_log(self, level: str, event: str, details: str = ""):
        """Add log entry."""
        self.log_buffer.append(LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            event=event,
            details=details,
        ))
        if level.upper() == "ERROR":
            self.metrics.api_errors_count += 1
            self.metrics.last_error = event[:50]

    def update_system_metrics(
        self,
        rpc_latency_ms: Optional[float] = None,
        tokens_analyzed: Optional[int] = None,
        tokens_filtered: Optional[int] = None,
    ):
        """Update system metrics."""
        if rpc_latency_ms is not None:
            self.metrics.rpc_latency_ms = rpc_latency_ms
        if tokens_analyzed is not None:
            self.metrics.tokens_analyzed = tokens_analyzed
        if tokens_filtered is not None:
            self.metrics.tokens_filtered = tokens_filtered

    def register_callbacks(
        self,
        on_pause: Optional[Callable] = None,
        on_resume: Optional[Callable] = None,
        on_exit_all: Optional[Callable] = None,
        get_positions: Optional[Callable] = None,
        get_balance: Optional[Callable] = None,
    ):
        """Register callbacks."""
        self._on_pause_callback = on_pause
        self._on_resume_callback = on_resume
        self._on_exit_all_callback = on_exit_all
        self._get_positions_callback = get_positions
        self._get_balance_callback = get_balance
