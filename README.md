# Bags Sniper Bot

**Deployer Intelligence Trading Bot for Solana Memecoins**

An automated trading bot that leverages deployer wallet analysis and AI narrative scoring to identify high-probability trading opportunities in the Solana memecoin market via [Bags.fm](https://bags.fm).

## The Edge

Historical analysis reveals that **top deployers outperform by 7,700x**:
- Top 10 deployers: 365,000 SOL profit
- Average deployers: 47.3 SOL profit

This bot identifies and tracks these elite deployers to catch their token launches early.

## Features

| Feature | Description |
|---------|-------------|
| **Deployer Intelligence** | Tracks deployer wallets, graduation rates, and historical performance |
| **AI Narrative Analysis** | DeepSeek-powered analysis of token cultural relevance and virality potential |
| **Multi-Stage Filtering** | Fast-fail architecture for efficient token evaluation |
| **Circuit Breaker** | 4-level risk management system with automatic position reduction |
| **Tiered Take-Profits** | Structured exit strategy (1.5x/2x/3x/5x tiers) |
| **Professional Trader Logic** | Fee-aware, momentum-based entry/exit decisions |
| **MEV Protection** | Jito bundle integration to prevent sandwich attacks |
| **Adaptive Position Sizing** | Position sizes scale with wallet balance |
| **Telegram Control** | Real-time alerts and bot control via Telegram |
| **Monitoring** | Prometheus/Grafana dashboards for performance tracking |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      BAGS SNIPER BOT                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Token      │  │   Deployer   │  │    AI Narrative      │  │
│  │  Discovery   │──│ Intelligence │──│     Analysis         │  │
│  │  (Helius)    │  │  (Scoring)   │  │    (DeepSeek)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                │                     │                │
│         ▼                ▼                     ▼                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    FILTER ENGINE                         │   │
│  │   Deployer → Basic → Security → Market Cap → Quality    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               TRADING ENGINE                             │   │
│  │  Position Sizer → Entry → Exit Management → Recording   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                          │            │
│         ▼                                          ▼            │
│  ┌──────────────┐                         ┌──────────────┐     │
│  │  Bags.fm API │                         │   Telegram   │     │
│  │  (Trading)   │                         │   (Alerts)   │     │
│  └──────────────┘                         └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- [Bags.fm](https://bags.fm) account + API key
- [Helius](https://helius.dev) API key (free tier available)
- Telegram bot (create via [@BotFather](https://t.me/BotFather))
- Dedicated Solana wallet (NOT your main wallet!)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bags-sniper.git
cd bags-sniper

# Install Python package
pip install -e .

# Start database services
docker-compose up -d postgres redis

# Configure environment
cp .env.example .env
# Edit .env with your API keys and wallet
```

### Configuration

Edit `.env` with your credentials:

```env
# Required
WALLET_PRIVATE_KEY=your_base58_private_key
BAGS_API_KEY=your_bags_api_key
HELIUS_API_KEY=your_helius_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional AI (improves token scoring)
DEEPSEEK_API_KEY=your_deepseek_api_key

# Safety - keep true until ready
DRY_RUN=true
```

### Verify Setup

```bash
python scripts/verify_setup.py
```

### Run the Bot

```bash
# Dry run mode (no real trades)
python -m bags_sniper.cli run

# After testing, set DRY_RUN=false in .env for live trading
```

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## CLI Commands

```bash
# Health check
python -m bags_sniper.cli health

# Populate historical deployer data
python -m bags_sniper.cli backfill --days 7

# View top performing deployers
python -m bags_sniper.cli top --limit 20

# Run backtest simulation
python -m bags_sniper.cli backtest --days 7

# Start the trading bot
python -m bags_sniper.cli run
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize bot |
| `/status` | View bot status and P&L |
| `/balance` | Check wallet balance |
| `/positions` | View open positions |
| `/pause` | Pause new entries |
| `/resume` | Resume trading |
| `/exitall` | Emergency exit all positions |

## Risk Management

### Circuit Breaker

Automatically reduces exposure during drawdowns:

| Level | Drawdown | Action |
|-------|----------|--------|
| Normal | < 3% | Full position sizes |
| L1 | 3-5% | 50% position sizes |
| L2 | 5-8% | 25% position sizes |
| L3 | 8-10% | New entries paused |
| L4 | > 10% | Full shutdown, exit all |

### Take-Profit Strategy

| Level | Trigger | Action |
|-------|---------|--------|
| TP1 | 1.5x | Sell 30% |
| TP2 | 2.0x | Sell 30% |
| TP3 | 3.0x | Sell 20% |
| TP4 | 5.0x | Sell remaining |
| Stop Loss | -30% | Exit position |

### Hard Exit Triggers

- Volume drops 60% from peak
- Top holder sells >8% of holdings
- Circuit breaker Level 4

## Monitoring

Start Grafana and Prometheus:

```bash
docker-compose --profile monitoring up -d
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin / bags_sniper |
| Prometheus | http://localhost:9091 | - |

## Project Structure

```
bags-sniper/
├── src/bags_sniper/
│   ├── core/              # Core trading logic
│   │   ├── trading_engine.py
│   │   ├── filter_engine.py
│   │   ├── deployer_intelligence.py
│   │   ├── quality_gate.py
│   │   ├── circuit_breaker.py
│   │   └── config.py
│   ├── services/          # External service integrations
│   │   ├── bags_api.py
│   │   ├── solana_rpc.py
│   │   ├── deepseek_ai.py
│   │   ├── telegram_bot.py
│   │   └── jupiter_api.py
│   ├── models/            # Database models
│   └── backtest/          # Backtesting engine
├── tests/                 # Unit and integration tests
├── monitoring/            # Prometheus/Grafana configs
├── docker-compose.yml
├── pyproject.toml
└── QUICKSTART.md
```

## Configuration Reference

See [.env.example](.env.example) for all configuration options.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | `true` | Simulate trades without execution |
| `MAX_POSITION_SIZE_SOL` | `0.3` | Maximum SOL per trade |
| `MIN_DEPLOYER_SCORE` | `60.0` | Minimum deployer score to trade |
| `STOP_LOSS_PERCENT` | `30.0` | Hard stop loss percentage |
| `USE_JITO_BUNDLES` | `true` | Enable MEV protection |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check src/ tests/

# Type checking
mypy src/
```

## Disclaimer

**This software is for educational purposes only.**

- Trading memecoins is **extremely risky**
- You can lose **100% of your investment**
- Past performance does **NOT guarantee future results**
- **Never trade money you cannot afford to lose**
- You are **solely responsible** for any trading decisions

The authors and contributors are not liable for any financial losses incurred through the use of this software.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [Bags.fm](https://bags.fm) - Trading platform
- [Helius](https://helius.dev) - Solana RPC provider
- [DeepSeek](https://deepseek.com) - AI analysis
- [Jito Labs](https://jito.wtf) - MEV protection
