"""Seed the database with a default admin user and all known config settings."""

import os
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.models import User, Setting
from src.api.auth import hash_password

logger = logging.getLogger(__name__)

# ─── Default admin credentials ────────────────────────────────────────────────

DEFAULT_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
DEFAULT_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
DEFAULT_EMAIL = os.getenv("ADMIN_EMAIL", "admin@tradingagent.local")

# ─── All known settings with metadata ────────────────────────────────────────

SETTING_DEFINITIONS = [
    # Exchange
    {"key": "DELTA_TESTNET_API_KEY",    "category": "exchange", "label": "Delta Testnet API Key",    "description": "API key from Delta Exchange testnet account", "is_secret": True},
    {"key": "DELTA_TESTNET_API_SECRET", "category": "exchange", "label": "Delta Testnet API Secret", "description": "API secret from Delta Exchange testnet account", "is_secret": True},
    {"key": "DELTA_PROD_API_KEY",       "category": "exchange", "label": "Delta Prod API Key",       "description": "API key from Delta Exchange production account", "is_secret": True},
    {"key": "DELTA_PROD_API_SECRET",    "category": "exchange", "label": "Delta Prod API Secret",    "description": "API secret from Delta Exchange production account", "is_secret": True},
    {"key": "DELTA_BASE_URL",           "category": "exchange", "label": "Delta Base URL",           "description": "Testnet: https://cdn-ind.testnet.deltaex.org | Mainnet: https://api.india.delta.exchange", "is_secret": False},
    {"key": "DELTA_LEVERAGE",           "category": "exchange", "label": "Leverage",                 "description": "Leverage multiplier applied to all positions (integer)", "is_secret": False},

    # LLM
    {"key": "LLM_PROVIDER",          "category": "llm", "label": "LLM Provider",           "description": "anthropic | openai | deepseek | openrouter | openai_compat", "is_secret": False},
    {"key": "LLM_MODEL",             "category": "llm", "label": "LLM Model",              "description": "Model identifier for the selected provider", "is_secret": False},
    {"key": "ANTHROPIC_API_KEY",     "category": "llm", "label": "Anthropic API Key",      "description": "API key for Anthropic Claude", "is_secret": True},
    {"key": "OPENAI_API_KEY",        "category": "llm", "label": "OpenAI API Key",         "description": "API key for OpenAI GPT models", "is_secret": True},
    {"key": "DEEPSEEK_API_KEY",      "category": "llm", "label": "DeepSeek API Key",       "description": "API key for DeepSeek models", "is_secret": True},
    {"key": "OPENROUTER_API_KEY",    "category": "llm", "label": "OpenRouter API Key",     "description": "API key for OpenRouter (1000+ models)", "is_secret": True},
    {"key": "OPENAI_COMPAT_BASE_URL","category": "llm", "label": "OpenAI Compat Base URL", "description": "Base URL for custom OpenAI-compatible endpoint (Ollama, LM Studio...)", "is_secret": False},
    {"key": "OPENAI_COMPAT_API_KEY", "category": "llm", "label": "OpenAI Compat API Key",  "description": "API key for the custom OpenAI-compatible endpoint", "is_secret": True},
    {"key": "SANITIZE_MODEL",        "category": "llm", "label": "Sanitize Model",         "description": "Cheap model for output normalisation (usually a Claude Haiku)", "is_secret": False},
    {"key": "MAX_TOKENS",            "category": "llm", "label": "Max Output Tokens",      "description": "Maximum tokens in LLM response (default: 4096)", "is_secret": False},
    {"key": "ENABLE_TOOL_CALLING",   "category": "llm", "label": "Enable Tool Calling",    "description": "Allow LLM to call fetch_indicator tool (true/false)", "is_secret": False},
    {"key": "THINKING_ENABLED",      "category": "llm", "label": "Extended Thinking",      "description": "Enable Claude extended thinking (Anthropic only)", "is_secret": False},
    {"key": "THINKING_BUDGET_TOKENS","category": "llm", "label": "Thinking Token Budget",  "description": "Token budget for extended thinking (default: 10000)", "is_secret": False},

    # Trading
    {"key": "ASSETS",   "category": "trading", "label": "Trading Assets",   "description": "Space-separated list of perpetual futures symbols (e.g. BTC ETH SOL)", "is_secret": False},
    {"key": "INTERVAL", "category": "trading", "label": "Decision Interval", "description": "How often the agent makes decisions: 1m 5m 15m 1h 4h", "is_secret": False},

    # Risk
    {"key": "MAX_POSITION_PCT",             "category": "risk", "label": "Max Position %",          "description": "Max single position as % of portfolio (default: 20)", "is_secret": False},
    {"key": "MAX_LOSS_PER_POSITION_PCT",    "category": "risk", "label": "Max Loss per Position %", "description": "Force-close threshold % loss per position (default: 20)", "is_secret": False},
    {"key": "MAX_LEVERAGE",                 "category": "risk", "label": "Max Leverage",            "description": "Hard leverage cap enforced by risk manager (default: 10)", "is_secret": False},
    {"key": "MAX_TOTAL_EXPOSURE_PCT",       "category": "risk", "label": "Max Total Exposure %",    "description": "Max total notional as % of portfolio (default: 80)", "is_secret": False},
    {"key": "DAILY_LOSS_CIRCUIT_BREAKER_PCT","category": "risk","label": "Daily Loss Circuit Breaker %","description": "Stop new trades at this daily drawdown % (default: 25)", "is_secret": False},
    {"key": "MANDATORY_SL_PCT",            "category": "risk", "label": "Mandatory SL %",          "description": "Auto SL distance % if LLM doesn't set one (default: 5)", "is_secret": False},
    {"key": "MAX_CONCURRENT_POSITIONS",    "category": "risk", "label": "Max Concurrent Positions","description": "Max simultaneous positions (default: 10)", "is_secret": False},
    {"key": "MIN_BALANCE_RESERVE_PCT",     "category": "risk", "label": "Min Balance Reserve %",   "description": "Don't trade below this % of initial balance (default: 10)", "is_secret": False},
    {"key": "MIN_RISK_REWARD",             "category": "risk", "label": "Min Risk/Reward Ratio",   "description": "Minimum TP/SL ratio required to allow a trade, e.g. 1.5 means TP must be 1.5× the SL distance (default: 1.5)", "is_secret": False},

    # Advanced
    {"key": "API_HOST",  "category": "advanced", "label": "API Host", "description": "Host for the aiohttp data API (default: 0.0.0.0)", "is_secret": False},
    {"key": "API_PORT",  "category": "advanced", "label": "API Port", "description": "Port for the aiohttp data API (default: 3000)", "is_secret": False},
    {"key": "SECRET_KEY","category": "advanced", "label": "JWT Secret Key", "description": "Secret key for signing JWT tokens — change in production!", "is_secret": True},
]


async def seed_users(db: AsyncSession) -> None:
    """Create the default admin user if no users exist."""
    result = await db.execute(select(User))
    existing = result.scalars().first()
    if existing:
        logger.info("Users already exist — skipping user seed.")
        return

    admin = User(
        username=DEFAULT_USERNAME,
        email=DEFAULT_EMAIL,
        hashed_password=hash_password(DEFAULT_PASSWORD),
        is_active=True,
    )
    db.add(admin)
    await db.commit()
    logger.info("Default admin user created: username=%s", DEFAULT_USERNAME)


async def seed_settings(db: AsyncSession) -> None:
    """Populate settings table from .env values, skipping existing keys."""
    for defn in SETTING_DEFINITIONS:
        key = defn["key"]
        result = await db.execute(select(Setting).where(Setting.key == key))
        existing = result.scalar_one_or_none()
        if existing:
            continue  # Don't overwrite existing settings

        env_value = os.getenv(key, "")
        setting = Setting(
            key=key,
            value=env_value,
            category=defn["category"],
            label=defn.get("label", key),
            description=defn.get("description", ""),
            is_secret=defn.get("is_secret", False),
        )
        db.add(setting)

    await db.commit()
    logger.info("Settings seeded from .env values.")


async def run_seed(db: AsyncSession) -> None:
    """Run all seeders."""
    await seed_users(db)
    await seed_settings(db)
