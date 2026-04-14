# AI Trading Agent — Delta Exchange

An AI-powered perpetual futures trading agent that connects to **Delta Exchange (India)** and uses a pluggable LLM backend (Claude, GPT-4o, DeepSeek, OpenRouter, or any local model) to analyse markets and execute trades automatically.

## What It Does

1. Fetches real-time OHLCV candle data from Delta Exchange and computes technical indicators locally (EMA, RSI, MACD, ATR, Bollinger Bands, ADX, OBV, VWAP)
2. Sends full market context, account state, and risk limits to the configured LLM
3. LLM returns buy / sell / hold decisions with allocation, take-profit, and stop-loss levels
4. Risk manager validates every decision before execution
5. Executes market or limit orders with bracket TP/SL via the Delta Exchange REST API

## Supported LLM Providers

Set `LLM_PROVIDER` in `.env` to switch between providers — no code changes required.

| `LLM_PROVIDER` | Models | Tool-calling | Thinking |
|---|---|---|---|
| `anthropic` *(default)* | claude-opus-4-5, claude-sonnet-4, claude-haiku-4-5 | ✅ Native | ✅ Extended |
| `openai` | gpt-4o, gpt-4o-mini, o1, o3-mini | ✅ | ❌ |
| `deepseek` | deepseek-chat, deepseek-reasoner | ✅ | ❌ |
| `openrouter` | 1000+ models via single key | ✅ | ❌ |
| `openai_compat` | Ollama, LM Studio, Groq, vLLM, etc. | ✅ | ❌ |

## Tradeable Markets

Any perpetual futures available on Delta Exchange (India), including:

- **Crypto**: BTC, ETH, SOL, AVAX, LINK, and 100+ more
- **Start with testnet**: `DELTA_BASE_URL=https://cdn-ind.testnet.deltaex.org` (default)
- **Switch to live**: `DELTA_BASE_URL=https://api.india.delta.exchange`

## Safety Guards

All enforced in code, not just LLM prompts. Configurable via `.env`:

| Guard | Default | Description |
|---|---|---|
| Max Position Size | 20% | Single position capped at 20% of portfolio |
| Force Close | -20% | Auto-close any position at 20% loss |
| Max Leverage | 10× | Hard leverage cap (auto-applied at startup) |
| Total Exposure | 80% | All open positions combined capped at 80% |
| Daily Circuit Breaker | -25% | Halts new trades at 25% daily drawdown |
| Mandatory Stop-Loss | 5% | Auto-sets SL if the LLM omits one |
| Max Positions | 10 | Concurrent open position limit |
| Balance Reserve | 10% | Never trade below 10% of the initial balance |

## Setup

### Prerequisites

- Python 3.12+
- A [Delta Exchange India](https://india.delta.exchange) account (or [testnet](https://demo.delta.exchange))
- API key + secret from Delta Exchange → Account → API Keys
- At least one LLM API key (Anthropic, OpenAI, DeepSeek, or OpenRouter)

### Install

```bash
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — fill in DELTA_API_KEY, DELTA_API_SECRET, and your LLM key
```

#### Minimum required variables

```env
# Delta Exchange
DELTA_API_KEY=your_key
DELTA_API_SECRET=your_secret
DELTA_BASE_URL=https://cdn-ind.testnet.deltaex.org   # testnet — safe for testing

# LLM (pick one)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-4-5-20251101
ANTHROPIC_API_KEY=sk-ant-...

# Assets & interval
ASSETS="BTC ETH SOL"
INTERVAL=5m
```

#### Switching LLM providers

```env
# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-...

# DeepSeek
LLM_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
DEEPSEEK_API_KEY=sk-...

# OpenRouter (access any model with one key)
LLM_PROVIDER=openrouter
LLM_MODEL=meta-llama/llama-3.3-70b-instruct
OPENROUTER_API_KEY=sk-or-...

# Local Ollama
LLM_PROVIDER=openai_compat
LLM_MODEL=llama3.2
OPENAI_COMPAT_BASE_URL=http://localhost:11434/v1
OPENAI_COMPAT_API_KEY=none
```

### Run

```bash
python src/main.py
```

Or override assets and interval via CLI:

```bash
python src/main.py --assets "BTC ETH SOL" --interval 1h
```

## Project Structure

```
src/
  main.py                  # Entry point, trading loop, local API server
  config_loader.py         # Env-based config with typed defaults
  risk_manager.py          # Safety guards (position limits, loss protection)
  agent/
    decision_maker.py      # LLM prompt construction, tool-call loop, output parsing
    llm_provider.py        # Provider abstraction (Anthropic / OpenAI-compat factory)
  indicators/
    local_indicators.py    # EMA, RSI, MACD, ATR, BBands, ADX, OBV, VWAP (computed locally)
    taapi_client.py        # Legacy — kept for reference, unused
  trading/
    delta_api.py           # Delta Exchange REST client (orders, candles, positions)
    delta_auth.py          # HMAC-SHA256 request signing for Delta Exchange
  utils/
    formatting.py          # Number formatting helpers
    prompt_utils.py        # JSON serialisation helpers
```

## How It Works

Each loop iteration:

1. **Account snapshot** — fetches balance, open positions, unrealised PnL from Delta Exchange
2. **Force-close** — any position with ≥ 20% loss is market-closed immediately
3. **Market data** — fetches OHLCV candles for every configured asset; computes all indicators locally
4. **LLM decision** — sends the full context (account state + indicators + risk limits) to the configured LLM
5. **Risk validation** — the risk manager caps allocation, enforces SL, checks exposure limits
6. **Order execution** — places market or limit orders; attaches bracket TP/SL via Delta's bracket-order API
7. **Sleep** — waits for the configured `INTERVAL` before the next iteration

## API Endpoints

When running, a lightweight local HTTP server is available:

- `GET /diary` — Recent trade diary entries as JSON
- `GET /logs` — LLM request logs

Default: `http://localhost:3000` (configurable via `API_HOST` / `API_PORT`).

## Advanced Options

| Variable | Default | Description |
|---|---|---|
| `DELTA_LEVERAGE` | `5` | Leverage auto-applied to every traded product at startup |
| `ENABLE_TOOL_CALLING` | `true` | Let the LLM call `fetch_indicator` dynamically (Anthropic & OpenAI) |
| `THINKING_ENABLED` | `false` | Claude extended thinking (Anthropic only) |
| `THINKING_BUDGET_TOKENS` | `10000` | Token budget for extended thinking |
| `MAX_TOKENS` | `4096` | Max LLM output tokens |
| `SANITIZE_MODEL` | *(unset)* | Cheap Anthropic model used to normalise LLM output before parsing |
| `API_HOST` | `0.0.0.0` | Local API server host |
| `API_PORT` | `3000` | Local API server port |

## License

Use at your own risk. No guarantee of returns. This software has not been audited. Always test on testnet before trading real funds.
