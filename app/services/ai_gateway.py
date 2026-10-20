"""Generic OpenAI-compatible model discovery and chat transport.

Model IDs and pricing are read from the authenticated live catalog; no model or
price roster is embedded in the application.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from .. import config
from .market import get_http_session

logger = logging.getLogger("armando.ai_gateway")

_CACHE_TTL_SECONDS = 300
_CACHE_LOCK = asyncio.Lock()
_cached_models: list["AIModel"] = []
_cached_at = 0.0


class AIGatewayError(RuntimeError):
    """Safe, user-facing-neutral error; never contains credentials or prompts."""


class AIGatewayNotConfigured(AIGatewayError):
    """The service has no API key configured."""


@dataclass(frozen=True)
class AIModel:
    id: str
    name: str
    pricing: Literal["free", "paid", "unknown"]

    @property
    def price_label(self) -> str:
        return {
            "free": "🆓 رایگان",
            "paid": "💳 پریمیوم",
            "unknown": "❔ قیمت نامشخص",
        }[self.pricing]


def model_ref(model_id: str) -> str:
    """Short stable reference suitable for Telegram's 64-byte callback limit."""
    return hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]


def _base_url() -> str:
    """Normalize an OpenAI-compatible API origin for raw HTTP requests.

    SDK examples often accept the bare host and add ``/v1`` internally. This
    service sends HTTP requests directly, so add ``/v1`` when the configured
    URL is only an origin. Explicit API prefixes (for example ``/v1`` or
    ``/openai/v1``) are preserved as supplied.
    """
    base = (config.settings.ai_chat_base_url or "").strip().rstrip("/")
    if not base:
        return ""
    from urllib.parse import urlsplit

    parsed = urlsplit(base)
    if (parsed.scheme in {"http", "https"} and parsed.netloc
            and parsed.path in {"", "/"} and not parsed.query and not parsed.fragment):
        return f"{base}/v1"
    return base


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "free", "on"}
    return False


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        candidate = Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, ValueError, TypeError):
        # Accept common display forms such as "$0.12 / 1M tokens" while still
        # extracting only a leading numeric rate, never a marketing claim.
        match = re.match(r"\s*\$?\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", str(value))
        if not match:
            return None
        try:
            candidate = Decimal(match.group(1))
        except InvalidOperation:
            return None
    if not candidate.is_finite() or candidate < 0:
        return None
    return candidate


_INPUT_PRICE_KEYS = (
    "input", "prompt", "input_price", "prompt_price", "input_cost",
    "prompt_cost", "input_rate", "input_price_usd", "prompt_price_usd",
    "input_per_token", "prompt_per_token", "input_per_million",
    "input_per_1m", "input_per_million_tokens", "input_price_per_million",
    "input_price_per_1m", "input_price_per_million_tokens", "input_price_per_1m_tokens",
    "prompt_price_per_million", "prompt_price_per_1m", "prompt_price_per_million_tokens",
    "input_cost_per_million", "input_cost_per_1m", "input_cost_per_1m_tokens",
    "input_token_price", "prompt_token_price", "price_input", "inputPrice", "promptPrice",
    "inputPricePerMTok", "promptPricePerMTok",
)
_OUTPUT_PRICE_KEYS = (
    "output", "completion", "output_price", "completion_price", "output_cost",
    "completion_cost", "output_rate", "output_price_usd", "completion_price_usd",
    "output_per_token", "completion_per_token", "output_per_million",
    "output_per_1m", "output_per_million_tokens", "output_price_per_million",
    "output_price_per_1m", "output_price_per_million_tokens", "output_price_per_1m_tokens",
    "completion_price_per_million", "completion_price_per_1m", "completion_price_per_million_tokens",
    "output_cost_per_million", "output_cost_per_1m", "output_cost_per_1m_tokens",
    "output_token_price", "completion_token_price", "price_output", "outputPrice", "completionPrice",
    "outputPricePerMTok", "completionPricePerMTok",
)


def _pricing_sources(raw: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [raw]
    for key in ("pricing", "price", "prices", "cost", "costs", "pricing_info"):
        value = raw.get(key)
        if isinstance(value, dict):
            sources.append(value)
        elif isinstance(value, list):
            sources.extend(item for item in value if isinstance(item, dict))
    return sources


def _find_price(sources: list[dict[str, Any]], keys: tuple[str, ...]) -> Decimal | None:
    wanted = {key.casefold() for key in keys}
    for source in sources:
        for key, value in source.items():
            if str(key).casefold() not in wanted:
                continue
            result = _decimal(value)
            if result is not None:
                return result
    return None


def _explicit_free(raw: dict[str, Any]) -> bool:
    """Read an explicit boolean from the live API, never infer from model slug/name."""
    for source in _pricing_sources(raw):
        for key in ("is_free", "free", "free_tier", "isFree", "freeTier"):
            if key in source and _is_true(source.get(key)):
                return True
    return False


def _observed_prices(sources: list[dict[str, Any]]) -> list[Decimal]:
    """Return numeric rates explicitly exposed by API pricing objects."""
    exact_price_keys = {
        "input", "output", "prompt", "completion", "request", "image", "audio",
        "video", "reasoning", "cache", "cache_read", "cache_write", "input_cache_read",
        "input_cache_write", "prompt_cache_read", "prompt_cache_write",
    }
    values: list[Decimal] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)
        else:
            number = _decimal(value)
            if number is not None:
                values.append(number)

    for source in sources:  # only fields that look like prices are read from model metadata
        for key, value in source.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
            if (normalized in exact_price_keys or "price" in normalized
                    or "cost" in normalized or "rate" in normalized):
                collect(value)
    return values


def _model_from_api(raw: Any) -> AIModel | None:
    if not isinstance(raw, dict):
        return None
    model_id = raw.get("id") or raw.get("model") or raw.get("model_id")
    if not isinstance(model_id, str):
        return None
    model_id = model_id.strip()
    if not model_id or len(model_id) > 200:
        return None
    name = raw.get("name") or raw.get("display_name") or raw.get("displayName") or model_id
    name = str(name).strip()[:200] or model_id

    sources = _pricing_sources(raw)
    input_price = _find_price(sources, _INPUT_PRICE_KEYS)
    output_price = _find_price(sources, _OUTPUT_PRICE_KEYS)
    observed_prices = _observed_prices(sources)
    if (any(value > 0 for value in observed_prices)
            or (input_price is not None and input_price > 0)
            or (output_price is not None and output_price > 0)):
        price_class: Literal["free", "paid", "unknown"] = "paid"
    elif input_price == 0 and output_price == 0:
        price_class = "free"
    elif _explicit_free(raw):
        price_class = "free"
    else:
        price_class = "unknown"
    return AIModel(id=model_id, name=name, pricing=price_class)


def _model_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("data", "models", "items", "results"):
        items = payload.get(key)
        if isinstance(items, list):
            return items
        if isinstance(items, dict):
            nested = items.get("data") or items.get("models")
            if isinstance(nested, list):
                return nested
    return []


def _sort_models(models: list[AIModel]) -> list[AIModel]:
    order = {"free": 0, "paid": 1, "unknown": 2}
    return sorted(models, key=lambda model: (order[model.pricing], model.name.casefold(), model.id.casefold()))


async def list_models(*, force_refresh: bool = False) -> list[AIModel]:
    """Fetch the models allowed by this API key and cache them for five minutes."""
    global _cached_models, _cached_at
    api_key = (config.settings.ai_chat_api_key or "").strip()
    base_url = _base_url()
    if not api_key or not base_url:
        raise AIGatewayNotConfigured("AI_CHAT_API_KEY and AI_CHAT_BASE_URL are required")
    now = time.monotonic()
    if not force_refresh and _cached_models and now - _cached_at < _CACHE_TTL_SECONDS:
        return list(_cached_models)

    async with _CACHE_LOCK:
        now = time.monotonic()
        if not force_refresh and _cached_models and now - _cached_at < _CACHE_TTL_SECONDS:
            return list(_cached_models)
        try:
            import aiohttp

            session = await get_http_session()
            timeout = aiohttp.ClientTimeout(total=20)
            async with session.get(
                f"{_base_url()}/models",
                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                timeout=timeout,
            ) as response:
                if response.status >= 400:
                    logger.warning("AI provider model catalog returned HTTP %s", response.status)
                    raise AIGatewayError(f"model catalog HTTP {response.status}")
                payload = await response.json(content_type=None)
        except AIGatewayError:
            raise
        except Exception as exc:  # noqa: BLE001 - network/API errors must stay local
            logger.warning("AI provider model catalog unavailable (%s)", exc.__class__.__name__)
            raise AIGatewayError("model catalog unavailable") from None

        models: list[AIModel] = []
        seen: set[str] = set()
        for raw in _model_items(payload):
            model = _model_from_api(raw)
            if model is None or model.id in seen:
                continue
            seen.add(model.id)
            models.append(model)
        if not models:
            logger.warning("AI provider model catalog returned no valid models")
            raise AIGatewayError("empty model catalog")
        _cached_models = _sort_models(models)
        _cached_at = time.monotonic()
        return list(_cached_models)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks).strip()
    return ""


async def chat_completion(model: str, messages: list[dict[str, str]], *,
                          max_tokens: int = 1200) -> str:
    """Submit an OpenAI-compatible chat completion and return its text content."""
    api_key = (config.settings.ai_chat_api_key or "").strip()
    if not api_key or not _base_url():
        raise AIGatewayNotConfigured("AI_CHAT_API_KEY and AI_CHAT_BASE_URL are required")
    if not model or len(model) > 200:
        raise AIGatewayError("invalid model id")
    if not messages:
        raise AIGatewayError("no messages supplied")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max(1, min(int(max_tokens), 1600)),
        "stream": False,
    }
    try:
        import aiohttp

        session = await get_http_session()
        timeout = aiohttp.ClientTimeout(total=45)
        async with session.post(
            f"{_base_url()}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        ) as response:
            if response.status >= 400:
                logger.warning("AI provider completion returned HTTP %s", response.status)
                raise AIGatewayError(f"completion HTTP {response.status}")
            data = await response.json(content_type=None)
    except AIGatewayError:
        raise
    except Exception as exc:  # noqa: BLE001 - do not leak request or credential data
        logger.warning("AI provider completion unavailable (%s)", exc.__class__.__name__)
        raise AIGatewayError("completion unavailable") from None

    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise AIGatewayError("empty completion")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    answer = _content_text(message.get("content")) if isinstance(message, dict) else ""
    if not answer:
        raise AIGatewayError("completion contained no text")
    return answer


def reset_model_cache() -> None:
    """Clear cached catalog (used by tests and configuration reloads)."""
    global _cached_models, _cached_at
    _cached_models = []
    _cached_at = 0.0
