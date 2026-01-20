# Bags Sniper Bot - Quick Start Guide

A deployer intelligence trading bot for Solana memecoins on Bags.fm.

**Core Edge**: Top deployers have 7,700x better performance than average.

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Python | 3.11 or higher |
| Docker | Docker Desktop with Docker Compose |
| Bags.fm API Key | From your Bags.fm account |
| Helius API Key | Free at https://helius.dev |
| Telegram Bot | Create via @BotFather |
| Solana Wallet | Dedicated wallet (NOT your main wallet!) |

---

## Installation

### Step 1: Install Python Package

```powershell
cd moneybags
pip install -e .
```

### Step 2: Start Database

```powershell
docker-compose up -d postgres redis
```

> **Note**: Docker PostgreSQL runs on port **5433** to avoid conflicts with local PostgreSQL installations.

### Step 3: Configure Environment

```powershell
copy .env.example .env
```

Edit `.env` with your values:

```env
# WALLET - Export private key from Bags.fm (base58 format, ~88 chars)
WALLET_PRIVATE_KEY=your_base58_private_key_here

# API KEYS
BAGS_API_KEY=your_bags_fm_api_key
HELIUS_API_KEY=your_helius_api_key

# TELEGRAM - Create bot via @BotFather, get chat ID from @userinfobot
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=your_numeric_chat_id

# DATABASE - Uses port 5433 (not 5432)
DATABASE_URL=postgresql+asyncpg://bags_sniper:bags_sniper_dev@localhost:5433/bags_sniper

# SAFETY - Keep true until ready for live trading
DRY_RUN=true
```

### Step 4: Verify Setup

```powershell
python scripts/verify_setup.py
```

All 8 checks should pass.

---

## CLI Commands

All commands use `python -m bags_sniper.cli`:

```powershell
# Show help
python -m bags_sniper.cli --help

# Verify system health
python -m bags_sniper.cli health

# Populate deployer data (required before first run)
python -m bags_sniper.cli backfill --days 7

# View top deployers
python -m bags_sniper.cli top --limit 20

# Run backtest
python -m bags_sniper.cli backtest --days 7

# Start bot (uses DRY_RUN setting from .env)
python -m bags_sniper.cli run
```

---

## Testing (Dry Run)

### 1. Populate Historical Data

```powershell
python -m bags_sniper.cli backfill --days 7 --max-tokens 500
```

### 2. Validate Strategy

```powershell
python -m bags_sniper.cli backtest --days 7
```

### 3. Start Bot in Dry Run Mode

With `DRY_RUN=true` in `.env`:

```powershell
python -m bags_sniper.cli run
```

The bot will:
- Monitor new token launches
- Score deployers and filter tokens
- Log what it WOULD trade (no real trades)
- Send Telegram alerts

### 4. Test Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Verify bot is active |
| `/status` | View status & circuit breaker level |
| `/balance` | Check wallet balance |
| `/help` | List all commands |

---

## Going Live

### Pre-Launch Checklist

- [ ] Ran dry run for 24+ hours
- [ ] Telegram alerts working
- [ ] Backtest results acceptable
- [ ] Wallet funded with SOL
- [ ] Understand risk parameters

### Start Live Trading

1. Edit `.env`:
```env
DRY_RUN=false
MAX_POSITION_SIZE_SOL=0.1
MAX_CONCURRENT_POSITIONS=2
```

2. Start bot:
```powershell
python -m bags_sniper.cli run
```

### Telegram Control Commands

| Command | Description |
|---------|-------------|
| `/status` | Check bot status & P&L |
| `/balance` | View wallet balance |
| `/pause` | Pause new entries |
| `/resume` | Resume trading |
| `/exitall` | Emergency exit all positions |

---

## Monitoring

### Start Grafana & Prometheus

```powershell
docker-compose --profile monitoring up -d
```

| Service | URL | Login |
|---------|-----|-------|
| Grafana | http://localhost:3000 | admin / bags_sniper |
| Prometheus | http://localhost:9091 | - |

---

## Safety Features

### Circuit Breaker

Automatically reduces risk during drawdowns:

| Level | Drawdown | Action |
|-------|----------|--------|
| Normal | <3% | Full position sizes |
| L1 | 3-5% | 50% positions |
| L2 | 5-8% | 25% positions |
| L3 | 8-10% | New entries paused |
| L4 | >10% | Full shutdown |

### Take-Profit Tiers

- 25% at 2.5x
- 35% at 4x
- 20% at 7x
- Remainder: trailing stop

### Rate Limiting

All APIs automatically rate-limited:
- Bags.fm: 1,000 req/hour
- Helius: Credit-based
- Telegram: 30 msg/sec
- Jito: 5 req/sec

---

## Troubleshooting

### "Database connection failed"

```powershell
# Restart containers
docker-compose down
docker-compose up -d postgres redis

# Wait 10 seconds, then verify
python scripts/verify_setup.py
```

### "Port 5432 already in use"

You have a local PostgreSQL. The bot uses port **5433** instead. Ensure your `.env` has:
```
DATABASE_URL=postgresql+asyncpg://bags_sniper:bags_sniper_dev@localhost:5433/bags_sniper
```

### "bags-sniper command not found"

Use the Python module instead:
```powershell
python -m bags_sniper.cli run
```

### "Telegram bot not responding"

1. Verify token from @BotFather
2. Get numeric chat ID from @userinfobot
3. Send `/start` to your bot first
4. Restart the bot

### "No deployers found"

```powershell
python -m bags_sniper.cli backfill --days 14 --max-tokens 2000
```

---

## Configuration Reference

### Required Settings

| Variable | Description |
|----------|-------------|
| `WALLET_PRIVATE_KEY` | Base58 private key from wallet |
| `BAGS_API_KEY` | Bags.fm API key |
| `HELIUS_API_KEY` | Helius RPC API key |
| `TELEGRAM_BOT_TOKEN` | From @BotFather |
| `TELEGRAM_CHAT_ID` | Your numeric chat ID |
| `DATABASE_URL` | PostgreSQL connection (port 5433) |

### Trading Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | `true` | No real trades when true |
| `MAX_POSITION_SIZE_SOL` | `0.3` | Max SOL per trade |
| `MIN_DEPLOYER_SCORE` | `60.0` | Minimum score to trade |
| `STOP_LOSS_PERCENT` | `30.0` | Hard stop loss |

---

## Quick Reference

```powershell
# SETUP
pip install -e .
docker-compose up -d postgres redis
copy .env.example .env
# Edit .env with your keys

# VERIFY
python scripts/verify_setup.py

# TEST (Dry Run)
python -m bags_sniper.cli backfill --days 7
python -m bags_sniper.cli backtest --days 7
python -m bags_sniper.cli run

# GO LIVE
# Set DRY_RUN=false in .env
python -m bags_sniper.cli run

# TELEGRAM
/status  - Check status
/pause   - Pause trading
/resume  - Resume trading
/exitall - Emergency exit

# MONITORING
docker-compose --profile monitoring up -d
# Grafana: http://localhost:3000
```

---

## Warning

**This is experimental software.**

- Trading memecoins is extremely risky
- Never trade money you can't afford to lose
- Past performance does NOT guarantee future results
- You are solely responsible for any losses
