"""Pluggable LLM provider layer for the trading agent.

Supports four backends, selected via the ``LLM_PROVIDER`` env var:

╔══════════════════════════╦═════════════════════════════════════════════╗
║ LLM_PROVIDER             ║ API                                         ║
╠══════════════════════════╬═════════════════════════════════════════════╣
║ anthropic  (default)     ║ Anthropic native SDK  (Claude models)       ║
║ openai                   ║ OpenAI API  (gpt-4o, o1, o3 …)             ║
║ deepseek               ║ DeepSeek API  (deepseek-chat, -reasoner …)  ║
║ openrouter             ║ OpenRouter  (1000+ models via single key)    ║
╚══════════════════════════╩═════════════════════════════════════════════╝

All OpenAI-compatible backends (openai / deepseek / openrouter) share the
same ``OpenAICompatProvider`` class and simply differ in base_url + key.
Only the Anthropic provider supports Claude-specific features like
``thinking`` blocks and native tool-use.  The OpenAI-compat provider maps
the agent's tool-use protocol to the OpenAI function-calling format.

Usage (set in .env):
    LLM_PROVIDER=anthropic          LLM_MODEL=claude-opus-4-5
    LLM_PROVIDER=openai             LLM_MODEL=gpt-4o
    LLM_PROVIDER=deepseek           LLM_MODEL=deepseek-chat
    LLM_PROVIDER=openrouter         LLM_MODEL=meta-llama/llama-3.3-70b-instruct
    LLM_PROVIDER=openai_compat      LLM_MODEL=your-model
                                    OPENAI_COMPAT_BASE_URL=http://localhost:11434/v1
                                    OPENAI_COMPAT_API_KEY=none
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes shared across all providers
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """A tool the LLM can call (provider-agnostic)."""
    name: str
    description: str
    parameters: dict  # JSON Schema


@dataclass
class LLMResponse:
    """Normalised response returned by every provider."""
    text: str                                   # Final text output (JSON string from the agent)
    tool_calls: list[dict] = field(default_factory=list)   # [{name, arguments, id}]
    stop_reason: str = "end_turn"              # "end_turn" | "tool_use" | "length"
    input_tokens: int = 0
    output_tokens: int = 0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract interface all concrete providers must implement."""

    @abstractmethod
    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition] | None = None,
        *,
        max_tokens: int = 4096,
        thinking_enabled: bool = False,
        thinking_budget: int = 10000,
    ) -> LLMResponse:
        """Send a chat request and return a normalised response."""

    def log_request(self, model: str, messages: list[dict], log_path: str = "llm_requests.log") -> None:
        """Write a brief request summary to the log file."""
        try:
            last = messages[-1] if messages else {}
            content_preview = str(last.get("content", ""))[:500]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== {datetime.now()} ===\n")
                f.write(f"Model: {model}\n")
                f.write(f"Messages count: {len(messages)}\n")
                f.write(f"Last role: {last.get('role')}\n")
                f.write(f"Last content (truncated): {content_preview}\n")
        except Exception:
            pass

    def log_response(self, resp: LLMResponse, log_path: str = "llm_requests.log") -> None:
        """Write response metadata to the log file."""
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Stop reason: {resp.stop_reason}\n")
                f.write(f"Usage: input={resp.input_tokens}, output={resp.output_tokens}\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Anthropic (Claude) provider
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    """Uses the official Anthropic SDK — supports thinking + native tool-use."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic as _anthropic  # lazy import
        self._client = _anthropic.Anthropic(api_key=api_key)
        self._model = model

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition] | None = None,
        *,
        max_tokens: int = 4096,
        thinking_enabled: bool = False,
        thinking_budget: int = 10000,
    ) -> LLMResponse:
        import anthropic as _anthropic

        self.log_request(self._model, messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }

        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        if thinking_enabled:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            kwargs["max_tokens"] = max(max_tokens, 16000)

        resp = self._client.messages.create(**kwargs)
        logger.info(
            "Anthropic response: stop_reason=%s, usage=%s",
            resp.stop_reason,
            resp.usage,
        )

        # Extract text and tool-use blocks
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,  # already a dict
                })

        result = LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=resp.stop_reason or "end_turn",
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
        self.log_response(result)
        return result

    def build_assistant_message(self, resp: LLMResponse, raw_response: Any) -> dict:
        """Build a proper Anthropic-format assistant message including thinking."""
        assistant_content: list[dict] = []
        for block in raw_response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "thinking":
                assistant_content.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                })
        return {"role": "assistant", "content": assistant_content}

    def chat_raw(self, system: str, messages: list[dict], tools: list[ToolDefinition] | None = None,
                 *, max_tokens: int = 4096, thinking_enabled: bool = False, thinking_budget: int = 10000):
        """Return the raw Anthropic response object (for tool-use loop)."""
        import anthropic as _anthropic

        self.log_request(self._model, messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if thinking_enabled:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            kwargs["max_tokens"] = max(max_tokens, 16000)
        return self._client.messages.create(**kwargs)


# ---------------------------------------------------------------------------
# OpenAI-compatible provider (OpenAI / DeepSeek / OpenRouter / local LLMs)
# ---------------------------------------------------------------------------

_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "openai_compat": "gpt-4o",
}


class OpenAICompatProvider(LLMProvider):
    """OpenAI-compatible chat completions — works for OpenAI, DeepSeek, OpenRouter,
    Ollama, LM Studio, vLLM, Groq, Together AI, and any OpenAI-compat endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str, provider_name: str = "openai_compat") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAI/DeepSeek/OpenRouter/Compat support. "
                "Install it with: pip install openai"
            ) from exc
        from openai import OpenAI as _OpenAI
        self._client = _OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._provider = provider_name

    def chat(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition] | None = None,
        *,
        max_tokens: int = 4096,
        thinking_enabled: bool = False,   # not supported; silently ignored
        thinking_budget: int = 10000,
    ) -> LLMResponse:

        self.log_request(self._model, messages)

        # Build openai-format message list
        oai_messages = [{"role": "system", "content": system}] + messages

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": oai_messages,
            "max_tokens": max_tokens,
        }

        # Map ToolDefinition → OpenAI function-calling format
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        logger.info(
            "%s response: finish_reason=%s, usage=%s",
            self._provider,
            resp.choices[0].finish_reason,
            resp.usage,
        )

        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        tool_calls: list[dict] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        stop_reason = "end_turn"
        if choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif choice.finish_reason == "length":
            stop_reason = "length"

        result = LLMResponse(
            text=text,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        self.log_response(result)
        return result

    def build_assistant_message(self, resp: LLMResponse) -> dict:
        """Build the assistant message including tool_calls for the conversation."""
        content: Any = resp.text or ""
        if resp.tool_calls:
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in resp.tool_calls
                ],
            }
        return {"role": "assistant", "content": content}

    def build_tool_result_messages(self, resp: LLMResponse, results: list[tuple[str, str]]) -> list[dict]:
        """Build tool result messages in OpenAI format.

        Parameters
        ----------
        resp:     The LLMResponse that contained the tool calls.
        results:  List of (tool_call_id, result_json_string) tuples.
        """
        msgs = [self.build_assistant_message(resp)]
        for tc_id, result_str in results:
            msgs.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result_str,
            })
        return msgs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_provider(config: dict) -> LLMProvider:
    """Instantiate and return the configured LLM provider.

    Reads from *config* (which maps directly to CONFIG from config_loader):

        LLM_PROVIDER  : anthropic | openai | deepseek | openrouter | openai_compat
        LLM_MODEL     : overrides the default model for the chosen provider
        ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY
        OPENAI_COMPAT_BASE_URL, OPENAI_COMPAT_API_KEY  (for custom endpoints)
    """
    provider_name: str = (config.get("llm_provider") or "anthropic").lower().strip()

    # Model: explicit override → provider default
    model: str = (
        config.get("llm_model")
        or _DEFAULT_MODELS.get(provider_name, "gpt-4o")
    )

    if provider_name == "anthropic":
        api_key = config.get("anthropic_api_key") or ""
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the anthropic LLM provider.")
        logger.info("LLM provider: Anthropic Claude  model=%s", model)
        return AnthropicProvider(api_key=api_key, model=model)

    if provider_name in ("openai", "deepseek", "openrouter"):
        key_map = {
            "openai": "openai_api_key",
            "deepseek": "deepseek_api_key",
            "openrouter": "openrouter_api_key",
        }
        api_key = config.get(key_map[provider_name]) or ""
        if not api_key:
            raise RuntimeError(
                f"API key for provider '{provider_name}' is missing. "
                f"Set {key_map[provider_name].upper()} in your .env file."
            )
        base_url = _PROVIDER_URLS[provider_name]
        logger.info("LLM provider: %s  model=%s  base_url=%s", provider_name, model, base_url)
        return OpenAICompatProvider(api_key=api_key, model=model, base_url=base_url, provider_name=provider_name)

    if provider_name == "openai_compat":
        api_key = config.get("openai_compat_api_key") or "none"
        base_url = config.get("openai_compat_base_url") or "http://localhost:11434/v1"
        logger.info("LLM provider: openai_compat  model=%s  base_url=%s", model, base_url)
        return OpenAICompatProvider(api_key=api_key, model=model, base_url=base_url, provider_name="openai_compat")

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider_name}'. "
        "Choose one of: anthropic, openai, deepseek, openrouter, openai_compat"
    )
