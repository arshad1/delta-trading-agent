"""Decision-making agent that orchestrates LLM prompts and indicator lookups.

Supports multiple LLM backends via ``src.agent.llm_provider``:
    anthropic (default) | openai | deepseek | openrouter | openai_compat

The backend is selected by setting ``LLM_PROVIDER`` in ``.env``.
"""

import asyncio
import json
import logging
from datetime import datetime

from src.config_loader import CONFIG
from src.indicators.local_indicators import compute_all, last_n, latest
from src.agent.llm_provider import (
    LLMProvider,
    LLMResponse,
    ToolDefinition,
    AnthropicProvider,
    OpenAICompatProvider,
    create_provider,
)

logger = logging.getLogger(__name__)


class TradingAgent:
    """High-level trading agent that delegates reasoning to a pluggable LLM."""

    def __init__(self, hyperliquid=None):
        self.hyperliquid = hyperliquid
        self.max_tokens = int(CONFIG.get("max_tokens") or 4096)
        self.thinking_enabled: bool = bool(CONFIG.get("thinking_enabled"))
        self.thinking_budget: int = int(CONFIG.get("thinking_budget_tokens") or 10000)
        self.enable_tools: bool = bool(CONFIG.get("enable_tool_calling", False))

        # Primary provider (for trade decisions)
        self.provider: LLMProvider = create_provider(CONFIG)

        # Sanitise provider — always Anthropic if key is available, otherwise same provider
        # Falls back to the primary provider if no Anthropic key is configured.
        sanitize_model = CONFIG.get("sanitize_model") or "claude-haiku-4-5-20251001"
        anthropic_key = CONFIG.get("anthropic_api_key")
        if anthropic_key and not isinstance(self.provider, AnthropicProvider):
            # Use cheap Claude for sanitisation even when primary is non-Anthropic
            from src.agent.llm_provider import AnthropicProvider as _AP
            self._sanitize_provider: LLMProvider = _AP(api_key=anthropic_key, model=sanitize_model)
        elif isinstance(self.provider, AnthropicProvider):
            from src.agent.llm_provider import AnthropicProvider as _AP, _DEFAULT_MODELS
            self._sanitize_provider = _AP(api_key=anthropic_key, model=sanitize_model)
        else:
            # No Anthropic key — use primary provider for sanitisation too
            self._sanitize_provider = self.provider

        logger.info(
            "TradingAgent ready. Provider=%s  thinking=%s  tools=%s",
            type(self.provider).__name__,
            self.thinking_enabled,
            self.enable_tools,
        )

    # ------------------------------------------------------------------
    # Public API & Multi-Agent Routing
    # ------------------------------------------------------------------

    async def decide_trade(self, assets, context):
        """Quant decides for multiple assets, then Risk Reviewer filters buys/sells."""
        quant_out = await self._decide(context, assets=assets)
        if not isinstance(quant_out, dict):
            return quant_out
            
        reviewed_decisions = []
        for dec in quant_out.get("trade_decisions", []):
            if dec.get("action") in ("buy", "sell"):
                asset = dec.get("asset", "unknown")
                logger.info("Submitting %s %s proposal to Risk Reviewer...", asset, dec["action"])
                review = await self._review_decision(asset, context, dec)
                if not review.get("approved"):
                    dec["action"] = "hold"
                    original_reason = dec.get("rationale", "")
                    dec["rationale"] = f"RISK REJECTED: {review.get('reasoning')} | Quant: {original_reason}"
                    logger.info("Risk Reviewer REJECTED trade for %s.", asset)
                else:
                    logger.info("Risk Reviewer APPROVED trade for %s.", asset)
            reviewed_decisions.append(dec)
            
        quant_out["trade_decisions"] = reviewed_decisions
        return quant_out

    async def _review_decision(self, asset: str, context: str, decision: dict) -> dict:
        """Second LLM pass to filter false-positives while allowing active trading."""
        system_prompt = (
            "You are a pragmatic Risk Officer evaluating an AI Quant's proposed trade.\n"
            "Your job is to independently verify that the trade meets the trading system's core parameters without hallucinating flaws. You MUST allow valid intraday scalps and momentum trades.\n"
            "Evaluate the volume events, VWAP positioning, support/resistance location, and the current volatility regime based ONLY on the provided data.\n"
            "Crucially: If price is > VWAP, do not call it 'VWAP resistance'. It is support. Do not invent reasons to reject if the provided data aligns with the Quant's rationale.\n"
            "Reject the trade ONLY if there are genuine, explicit structural flaws in the provided data (e.g. explicitly 'very LOW' volume on a breakout, or trading directly into VWAP resistance when price < VWAP).\n"
            "Do NOT reject short-term momentum scalps just because higher timeframes (4h) aren't perfectly aligned. MODERATE volume is sufficient for scalp entries. Level-touch trades near support/resistance within 2% are valid even on quiet volume.\n\n"
            "Output ONLY a strict JSON object with two properties:\n"
            "  • \"approved\": boolean (true/false) indicating if the trade passes extreme scrutiny.\n"
            "  • \"reasoning\": string explaining exactly why you accepted or rejected it."
        )
        messages = [{"role": "user", "content": f"Market Data Context:\n{context}\n\nProposed Trade by Quant for {asset}:\n{json.dumps(decision)}"}]
        
        for _ in range(3):
            try:
                resp = self.provider.chat(
                    system=system_prompt,
                    messages=messages,
                    max_tokens=1000,
                )
                
                raw_text = resp.text.strip()
                
                # Robust JSON extraction
                import re
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL | re.IGNORECASE)
                if match:
                    cleaned = match.group(1).strip()
                else:
                    start = raw_text.find('{')
                    end = raw_text.rfind('}')
                    cleaned = raw_text[start:end+1] if start != -1 and end != -1 and end > start else raw_text

                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "approved" in parsed:
                    return parsed
            except Exception as e:
                logger.error("Reviewer LLM API error: %s", e)
                
        return {"approved": False, "reasoning": "Risk Reviewer failed to parse or respond properly. Auto-rejecting trade."}

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self, assets) -> str:
        return (
            "You are a rigorous QUANTITATIVE TRADER and interdisciplinary MATHEMATICIAN-ENGINEER "
            "optimizing risk-adjusted returns for perpetual futures under real execution, margin, "
            "and funding constraints.\n"
            "You will receive market + account context for SEVERAL assets, including:\n"
            f"- assets = {json.dumps(list(assets))}\n"
            "- explicitly computed technical events (e.g. MACD crosses) under 'intraday_events', 'bridge_events', and 'macro_events'\n"
            "- per-asset intraday (5m) and higher-timeframe (4h) raw metrics\n"
            "- Active Trades with Exit Plans\n"
            "- Recent Trading History\n"
            "- Risk management limits (hard-enforced by the system, not just guidelines)\n\n"
            "PAY EXTREME ATTENTION to the `intraday_events`, `bridge_events`, and `macro_events` arrays. Never hallucinate indicators; rely unconditionally on the explicit events provided in those arrays (e.g., 'Price is below VWAP', 'Volume is exceptionally HIGH').\n\n"
            "Always use the 'current time' provided in the user message to evaluate any time-based "
            "conditions, such as cooldown expirations or timed exit plans.\n\n"
            "Your goal: make decisive, first-principles decisions per asset that minimize churn "
            "while capturing edge.\n\n"
            "Aggressively pursue setups where calculated risk is outweighed by expected edge; "
            "size positions so downside is controlled while upside remains meaningful.\n\n"
            "Core policy (low-churn, position-aware)\n"
            "1) Respect prior plans: If an active trade has an exit_plan with explicit invalidation "
            "(e.g., \"close if 4h close above EMA50\"), DO NOT close or flip early unless that "
            "invalidation (or a stronger one) has occurred.\n"
            "2) Three valid setup families exist:\n"
            "   a) Trend-following breakout/continuation: require BOTH the 4h (macro_events) and 1h (bridge_events) trends aligned, OR at least one timeframe trending with the other ranging. NEVER buy/sell a breakout if the recent volume events explicitly flag 'very LOW' volume.\n"
            "   b) 4h or 1h support/resistance reaction trade: you MAY buy near support or sell near resistance even if 5m volume is quiet, as long as price is clearly near that level, the stop is beyond the level, and the take-profit points back toward the opposite side or next liquidity area.\n"
            "   c) Intraday Momentum/Scalp: When higher timeframes are ranging or unclear, you may take 5m/15m momentum trades if 5m volume is HIGH or MODERATE, and price crosses EMA20 or VWAP with momentum. RSI in the 50-70 (bullish) or 30-50 (bearish) momentum zone is sufficient confirmation. Use tight SL and TP.\n"
            "3) Prefer level-touch entries over chasing extension. When price is near support, prefer buy or buy-limit; when near resistance, prefer sell or sell-limit. Look for intraday opportunities actively.\n"
            "4) Agility: Be willing to take new trades or flip direction if intraday momentum clearly shifts. "
            "If the 5m/15m trend strongly reverses (e.g. crossing VWAP with high volume, MACD crossing), "
            "do not hesitate to close the previous trade and open a new one in the new direction.\n"
            "5) Cooldown: After opening, adding, reducing, or flipping, impose a self-cooldown of "
            "at least 2 bars of the decision timeframe (e.g., 2×5m = 10m) before another direction "
            "change, unless a hard invalidation occurs. Encode this in exit_plan (e.g., "
            "\"cooldown_bars:2 until 2025-10-19T15:50Z\"). You must honor your own cooldowns on "
            "future cycles.\n"
            "6) Funding is a tilt, not a trigger: Do NOT open/close/flip solely due to funding "
            "unless expected funding over your intended holding horizon meaningfully exceeds expected "
            "edge (e.g., > ~0.25×ATR).\n"
            "7) Overbought/oversold ≠ reversal by itself: Treat RSI extremes as risk-of-pullback. "
            "You need structure + momentum confirmation to bet against trend.\n"
            "8) Prefer adjustments over exits: If the thesis weakens but is not invalidated, first "
            "consider: tighten stop, trail TP, or reduce size. Flip only on hard invalidation + "
            "fresh confluence.\n\n"
            "4h level execution rules\n"
            "- Use the provided `levels_4h` object directly.\n"
            "- Define 'near a level' as roughly within 1.5% by default, or within 0.8x of 4h ATR14 when that is smaller.\n"
            "- If buying at 4h support, default to a limit order near support unless momentum is already rejecting strongly upward.\n"
            "- If selling at 4h resistance, default to a limit order near resistance unless momentum is already rejecting strongly downward.\n"
            "- For level-touch trades, TP should not be tiny; target a meaningful move away from the level. SL must sit beyond the tested level, not inside noise.\n\n"
            "Decision discipline (per asset)\n"
            "- Choose one: buy / sell / hold.\n"
            "- Proactively harvest profits when price action presents a clear, high-quality "
            "opportunity that aligns with your thesis.\n"
            "- You control allocation_usd. ALWAYS base it on the actual account size: use the "
            "`max_allocation_usd` and `suggested_allocation_usd` fields from risk_limits. "
            "Never invent large allocations — if those fields are present, your allocation_usd "
            "MUST be <= max_allocation_usd. Default to suggested_allocation_usd for normal trades.\n"
            "- Order type: set order_type to \"market\" for immediate execution, or \"limit\" for "
            "resting orders.\n"
            "  • For limit orders, you MUST set limit_price.\n"
            "  • For market orders, limit_price should be null.\n"
            "  • Default is \"market\" if omitted.\n"
            "- TP/SL sanity:\n"
            "  • BUY: tp_price > current_price, sl_price < current_price\n"
            "  • SELL: tp_price < current_price, sl_price > current_price\n"
            "  If sensible TP/SL cannot be set, use null and explain the logic. A mandatory SL "
            "will be auto-applied if you don't set one.\n"
            "- exit_plan must include at least ONE explicit invalidation trigger.\n\n"
            "Leverage policy (perpetual futures)\n"
            "- You can use leverage, but the system enforces a hard cap. Stay within the limits.\n"
            "- In high volatility (elevated ATR) or during funding spikes, reduce or avoid leverage.\n"
            "- Treat allocation_usd as notional exposure.\n\n"
            "Tool usage\n"
            "- Use the fetch_indicator tool whenever an additional datapoint could sharpen your "
            "thesis; parameters: indicator, asset, interval, optional period.\n"
            "- Incorporate tool findings into your reasoning, but NEVER paste raw tool responses "
            "into the final JSON — summarize the insight instead.\n\n"
            "Output contract\n"
            "- Output ONLY a strict JSON object (no markdown, no code fences) with exactly two "
            "properties:\n"
            "  • \"reasoning\": long-form string capturing detailed, step-by-step analysis.\n"
            "  • \"trade_decisions\": array ordered to match the provided assets list.\n"
            "- Each item inside trade_decisions must contain the keys: asset, action, "
            "allocation_usd, order_type, limit_price, tp_price, sl_price, exit_plan, rationale.\n"
            "  • order_type: \"market\" (default) or \"limit\"\n"
            "  • limit_price: required if order_type is \"limit\", null otherwise\n"
            "- Do not emit Markdown or any extra properties.\n"
        )

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="fetch_indicator",
                description=(
                    "Fetch technical indicators computed locally from exchange candle data. "
                    "Works for ALL perpetual markets (crypto, commodities, indices). "
                    "Available indicators: ema, sma, rsi, macd, bbands, atr, adx, obv, "
                    "vwap, stoch_rsi, all. Returns latest values and recent series."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "indicator": {
                            "type": "string",
                            "enum": ["ema", "sma", "rsi", "macd", "bbands", "atr",
                                     "adx", "obv", "vwap", "stoch_rsi", "all"],
                        },
                        "asset": {
                            "type": "string",
                            "description": "Asset symbol, e.g. BTC, ETH, SOL",
                        },
                        "interval": {
                            "type": "string",
                            "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                        },
                        "period": {
                            "type": "integer",
                            "description": "Indicator period (default varies by indicator)",
                        },
                    },
                    "required": ["indicator", "asset", "interval"],
                },
            )
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a tool call and return the result as a JSON string."""
        logger.info("LLM is calling tool: '%s' with arguments: %s", tool_name, tool_input)
        if tool_name != "fetch_indicator":
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            asset = tool_input["asset"]
            interval = tool_input["interval"]
            indicator = tool_input["indicator"]

            candles = await self.hyperliquid.get_candles(asset, interval, 100)

            all_inds = compute_all(candles)

            if indicator == "all":
                result = {
                    k: {"latest": latest(v) if isinstance(v, list) else v,
                        "series": last_n(v, 10) if isinstance(v, list) else v}
                    for k, v in all_inds.items()
                }
            elif indicator == "macd":
                result = {
                    "macd": {"latest": latest(all_inds.get("macd", [])), "series": last_n(all_inds.get("macd", []), 10)},
                    "signal": {"latest": latest(all_inds.get("macd_signal", [])), "series": last_n(all_inds.get("macd_signal", []), 10)},
                    "histogram": {"latest": latest(all_inds.get("macd_histogram", [])), "series": last_n(all_inds.get("macd_histogram", []), 10)},
                }
            elif indicator == "bbands":
                result = {
                    "upper": {"latest": latest(all_inds.get("bbands_upper", [])), "series": last_n(all_inds.get("bbands_upper", []), 10)},
                    "middle": {"latest": latest(all_inds.get("bbands_middle", [])), "series": last_n(all_inds.get("bbands_middle", []), 10)},
                    "lower": {"latest": latest(all_inds.get("bbands_lower", [])), "series": last_n(all_inds.get("bbands_lower", []), 10)},
                }
            elif indicator in ("ema", "sma"):
                period = tool_input.get("period", 20)
                from src.indicators.local_indicators import ema as _ema, sma as _sma
                closes = [c["c"] for c in candles]
                series = _ema(closes, period) if indicator == "ema" else _sma(closes, period)
                result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
            elif indicator == "rsi":
                period = tool_input.get("period", 14)
                from src.indicators.local_indicators import rsi as _rsi
                series = _rsi(candles, period)
                result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
            elif indicator == "atr":
                period = tool_input.get("period", 14)
                from src.indicators.local_indicators import atr as _atr
                series = _atr(candles, period)
                result = {"latest": latest(series), "series": last_n(series, 10), "period": period}
            else:
                key_map = {"adx": "adx", "obv": "obv", "vwap": "vwap", "stoch_rsi": "stoch_rsi"}
                mapped = key_map.get(indicator, indicator)
                series = all_inds.get(mapped, [])
                result = {
                    "latest": latest(series) if isinstance(series, list) else series,
                    "series": last_n(series, 10) if isinstance(series, list) else series,
                }
            return json.dumps(result, default=str)
        except Exception as ex:
            logger.error("Tool call error: %s", ex)
            return json.dumps({"error": str(ex)})

    # ------------------------------------------------------------------
    # Output sanitisation
    # ------------------------------------------------------------------

    def _sanitize_output(self, raw_content: str, assets_list) -> dict:
        """Use a cheap LLM call to normalise malformed output."""
        try:
            sanitize_system = (
                "You are a strict JSON normalizer. Return ONLY a JSON object with two keys: "
                "\"reasoning\" (string) and \"trade_decisions\" (array). "
                "Each trade_decisions item must have: asset, action (buy/sell/hold), "
                "allocation_usd (number), order_type (\"market\" or \"limit\"), "
                "limit_price (number or null), tp_price (number or null), sl_price (number or null), "
                "exit_plan (string), rationale (string). "
                f"Valid assets: {json.dumps(list(assets_list))}. "
                "If input is wrapped in markdown or has prose, extract just the JSON. "
                "Do not add fields."
            )
            resp = self._sanitize_provider.chat(
                system=sanitize_system,
                messages=[{"role": "user", "content": raw_content}],
                max_tokens=2048,
            )
            raw_text = resp.text.strip()
            
            import re
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                cleaned = match.group(1).strip()
            else:
                start = raw_text.find('{')
                end = raw_text.rfind('}')
                cleaned = raw_text[start:end+1] if start != -1 and end != -1 and end > start else raw_text

            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "trade_decisions" in parsed:
                return parsed
        except Exception as se:
            logger.error("Sanitize failed: %s", se)
        return {"reasoning": "", "trade_decisions": []}

    # ------------------------------------------------------------------
    # Core decision loop
    # ------------------------------------------------------------------

    async def _decide(self, context: str, assets) -> dict:
        """Dispatch decision request to the configured provider and enforce output contract."""
        system_prompt = self._build_system_prompt(assets)
        tools = self._build_tools() if self.enable_tools else None

        messages: list[dict] = [{"role": "user", "content": context}]

        # Anthropic has native thinking + tool-use with multi-turn content blocks.
        # OpenAI-compat providers need different message construction for tool results.
        is_anthropic = isinstance(self.provider, AnthropicProvider)

        # ---- Anthropic path (native tool-use + thinking) ----
        if is_anthropic:
            return await self._decide_anthropic(system_prompt, messages, tools, assets)

        # ---- OpenAI-compat path ----
        return await self._decide_openai_compat(system_prompt, messages, tools, assets)

    # ------------------------------------------------------------------
    # Anthropic decision loop
    # ------------------------------------------------------------------

    async def _decide_anthropic(self, system_prompt: str, messages: list[dict],
                           tools, assets) -> dict:
        """Tool-use loop for the Anthropic provider."""
        assert isinstance(self.provider, AnthropicProvider)

        for iteration in range(6):
            try:
                raw = self.provider.chat_raw(
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=self.max_tokens,
                    thinking_enabled=self.thinking_enabled,
                    thinking_budget=self.thinking_budget,
                )
            except Exception as e:
                logger.error("Anthropic API error: %s", e)
                break

            logger.info("Anthropic: stop_reason=%s usage=%s", raw.stop_reason, raw.usage)

            tool_use_blocks = [b for b in raw.content if b.type == "tool_use"]
            text_blocks = [b for b in raw.content if b.type == "text"]

            if tool_use_blocks and raw.stop_reason == "tool_use":
                # Append full assistant message (including thinking blocks)
                messages.append(
                    self.provider.build_assistant_message(
                        LLMResponse(text="", tool_calls=[]), raw
                    )
                )
                tool_results = []
                for block in tool_use_blocks:
                    result_str = await self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            raw_text = "".join(b.text for b in text_blocks)
            return self._parse_text_response(raw_text, assets)

        return self._empty_response(assets, "tool loop cap")

    # ------------------------------------------------------------------
    # OpenAI-compat decision loop
    # ------------------------------------------------------------------

    async def _decide_openai_compat(self, system_prompt: str, messages: list[dict],
                               tools, assets) -> dict:
        """Tool-use loop for OpenAI / DeepSeek / OpenRouter / compat providers."""
        assert isinstance(self.provider, OpenAICompatProvider)

        for iteration in range(15):
            try:
                resp = self.provider.chat(
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                logger.error("LLM API error (%s): %s", type(self.provider).__name__, e)
                break

            if resp.tool_calls and resp.stop_reason == "tool_use":
                # Execute tools and build continuation messages
                results = []
                for tc in resp.tool_calls:
                    result_str = await self._execute_tool(tc["name"], tc["arguments"])
                    results.append((tc["id"], result_str))

                continuation = self.provider.build_tool_result_messages(resp, results)
                # continuation = [assistant_msg, tool_result_msg, ...]
                messages.extend(continuation)
                continue

            return self._parse_text_response(resp.text, assets)

        return self._empty_response(assets, "tool loop cap")

    # ------------------------------------------------------------------
    # Shared response parsing
    # ------------------------------------------------------------------

    def _parse_text_response(self, raw_text: str, assets) -> dict:
        """Strip markdown fences and parse JSON from LLM response."""
        if not raw_text.strip():
            logger.error("Empty LLM response")
            return self._empty_response(assets, "empty response")

        raw_text = raw_text.strip()
        import re
        match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL | re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()
        else:
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            cleaned = raw_text[start:end+1] if start != -1 and end != -1 and end > start else raw_text

        try:
            parsed = json.loads(cleaned)
            if not isinstance(parsed, dict):
                logger.error("Expected dict, got %s; sanitizing", type(parsed))
                return self._sanitize_output(raw_text, assets)

            reasoning = parsed.get("reasoning", "") or ""
            decisions = parsed.get("trade_decisions")

            if isinstance(decisions, list):
                normalized = []
                for item in decisions:
                    if isinstance(item, dict):
                        item.setdefault("allocation_usd", 0.0)
                        item.setdefault("order_type", "market")
                        item.setdefault("limit_price", None)
                        item.setdefault("tp_price", None)
                        item.setdefault("sl_price", None)
                        item.setdefault("exit_plan", "")
                        item.setdefault("rationale", "")
                        normalized.append(item)
                return {"reasoning": reasoning, "trade_decisions": normalized}

            logger.error("trade_decisions missing; sanitizing")
            sanitized = self._sanitize_output(raw_text, assets)
            if sanitized.get("trade_decisions"):
                return sanitized
            return {"reasoning": reasoning, "trade_decisions": []}

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error("JSON parse error: %s — content: %s", e, raw_text[:200])
            sanitized = self._sanitize_output(raw_text, assets)
            if sanitized.get("trade_decisions"):
                return sanitized
            return {
                "reasoning": "Parse error",
                "trade_decisions": [
                    {
                        "asset": a,
                        "action": "hold",
                        "allocation_usd": 0.0,
                        "order_type": "market",
                        "limit_price": None,
                        "tp_price": None,
                        "sl_price": None,
                        "exit_plan": "",
                        "rationale": "Parse error",
                    }
                    for a in assets
                ],
            }

    @staticmethod
    def _empty_response(assets, reason: str) -> dict:
        return {
            "reasoning": reason,
            "trade_decisions": [
                {
                    "asset": a,
                    "action": "hold",
                    "allocation_usd": 0.0,
                    "order_type": "market",
                    "limit_price": None,
                    "tp_price": None,
                    "sl_price": None,
                    "exit_plan": "",
                    "rationale": reason,
                }
                for a in assets
            ],
        }
