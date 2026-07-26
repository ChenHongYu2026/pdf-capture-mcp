"""Multi-provider VLM client supporting any OpenAI-compatible API.

Supports: OpenAI, Anthropic (via compatible endpoint), MiniMax, Ollama, vLLM, etc.
Configuration is persisted to ~/.cache/pdf-capture-mcp/vlm_config.json.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_cache_dir, get_logger

logger = get_logger("llm_client")

# ── Config persistence ──────────────────────────────────────────────────────

_CONFIG_PATH = get_cache_dir() / "vlm_config.json"

# Rate limiting
_RATE_LIMIT_INTERVAL = 0.5
_MAX_RETRIES = 2
_BASE_DELAY = 2.0

_last_call_ts: float = 0.0

# 1x1 transparent PNG used to probe a model's vision capability
_TEST_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _load_config() -> dict[str, Any]:
    """Load persisted VLM configuration."""
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(config: dict[str, Any]) -> None:
    """Persist VLM configuration with restricted file permissions."""
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    # Restrict permissions: owner read/write only (protect API key)
    try:
        _CONFIG_PATH.chmod(0o600)
    except OSError:
        pass  # Windows doesn't support chmod


def get_vlm_config() -> dict[str, Any]:
    """Get current VLM configuration (API key is NEVER included in output).

    Returns:
        Dict with enabled, provider, model, api_base (no api_key).
    """
    config = _load_config()
    # SECURITY: never expose api_key to callers
    safe = {k: v for k, v in config.items() if k != "api_key"}
    safe["has_api_key"] = bool(config.get("api_key") or os.getenv("PDF_CAPTURE_VLM_API_KEY", ""))
    return safe


def _resolve_api_key() -> str:
    """Resolve API key from config file or environment variable."""
    config = _load_config()
    return config.get("api_key", "") or os.getenv("PDF_CAPTURE_VLM_API_KEY", "").strip()


def is_vlm_enabled() -> bool:
    """Check if VLM is configured and enabled with a valid API key."""
    config = _load_config()
    return config.get("enabled", False) and bool(_resolve_api_key())


def get_vlm_policy() -> str:
    """Return the configured VLM usage policy ('full' or 'tables_only').

    Configs created before the policy field existed default to 'full' —
    a user who configured a VLM expressed the intent to use it.
    """
    policy = str(_load_config().get("policy", "full"))
    return policy if policy in ("full", "tables_only") else "full"


# ── Setup & validation ──────────────────────────────────────────────────────


def setup_vlm(
    model: str,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    provider: str = "openai",
    policy: str = "full",
) -> dict[str, Any]:
    """Configure and validate VLM access.

    Performs a test call to verify:
    1. API key is valid
    2. Model exists and is accessible
    3. Model supports vision (image input)

    Args:
        model: Model name (e.g. 'gpt-4o', 'claude-sonnet-4-20250514', 'minimax-m3').
        api_key: API key for the provider.
        api_base: API base URL (OpenAI-compatible endpoint).
        provider: Provider name for display purposes.
        policy: What the pipeline may use the VLM for by default when
            callers pass 'auto': 'full' (table repair + figure descriptions)
            or 'tables_only' (table repair only). Explicit True/False
            arguments always override the policy.

    Returns:
        Dict with ok, message, and validation details.
    """
    if not model.strip():
        return {"ok": False, "error": "Model name is required."}
    if not api_key.strip():
        return {"ok": False, "error": "API key is required."}
    if policy not in ("full", "tables_only"):
        return {"ok": False, "error": f"Invalid policy: {policy!r}. Use: full, tables_only"}

    # Step 1: Test API connectivity and model access
    try:
        import httpx
    except ImportError:
        # Fallback: use urllib
        return _setup_vlm_urllib(model, api_key, api_base, provider, policy)

    try:
        client = httpx.Client(timeout=30.0)

        # Test with a minimal vision request (1x1 transparent PNG)
        test_image_b64 = _TEST_IMAGE_B64

        response = client.post(
            f"{api_base.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{test_image_b64}"},
                            },
                            {"type": "text", "text": "Reply with just: OK"},
                        ],
                    }
                ],
            },
        )

        if response.status_code == 401:
            return {
                "ok": False,
                "error": "API key authentication failed. Please verify your API key is valid.",
                "status_code": 401,
            }
        if response.status_code == 404:
            return {
                "ok": False,
                "error": f"Model '{model}' not found. Please check the model name.",
                "status_code": 404,
            }
        if response.status_code != 200:
            body = response.text[:300]
            # Check for vision-not-supported errors
            body_l = body.lower()
            if "image" in body_l and ("not support" in body_l or "invalid" in body_l):
                return {
                    "ok": False,
                    "error": (
                        f"Model '{model}' does not support vision/image input. "
                        "Please use a VLM-capable model (e.g. gpt-4o, claude-sonnet-4-20250514, "
                        "qwen-vl-max, gemini-pro-vision)."
                    ),
                    "status_code": response.status_code,
                }
            return {
                "ok": False,
                "error": f"API returned status {response.status_code}: {body}",
                "status_code": response.status_code,
            }

        # Success — model supports vision
        result = response.json()
        reply = ""
        choices = result.get("choices", [])
        if choices:
            reply = choices[0].get("message", {}).get("content", "")

        # Save config
        config = {
            "enabled": True,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "api_base": api_base,
            "policy": policy,
            "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_config(config)

        policy_msg = (
            "Full VLM capability is now ACTIVE BY DEFAULT: broken tables are "
            "repaired from page renders and figure descriptions are injected "
            "for RAG. Pass policy='tables_only' to limit this, or override "
            "per call via enable_table_enrich / enrich_figures."
            if policy == "full"
            else "Policy 'tables_only': broken tables are repaired by default; "
            "figure descriptions stay off unless enrich_figures=True."
        )
        return {
            "ok": True,
            "message": (
                f"VLM configured successfully: {provider}/{model}. "
                f"Vision capability confirmed. {policy_msg} "
                "Note: VLM calls consume tokens — each table/figure uses "
                "approximately 500-2000 tokens depending on image complexity."
            ),
            "model": model,
            "provider": provider,
            "policy": policy,
            "test_reply": reply[:50],
        }

    except httpx.TimeoutException:
        return {"ok": False, "error": "Connection timed out. Check your network and API base URL."}
    except Exception as exc:
        return {"ok": False, "error": f"Validation failed: {type(exc).__name__}: {exc}"}


def _setup_vlm_urllib(
    model: str, api_key: str, api_base: str, provider: str, policy: str = "full"
) -> dict[str, Any]:
    """Fallback VLM setup using urllib (no httpx dependency)."""
    import urllib.request

    test_image_b64 = _TEST_IMAGE_B64
    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{test_image_b64}"},
                        },
                        {"type": "text", "text": "Reply with just: OK"},
                    ],
                }
            ],
        }
    ).encode()

    req = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            json.loads(resp.read())
            config = {
                "enabled": True,
                "provider": provider,
                "model": model,
                "api_key": api_key,
                "api_base": api_base,
                "policy": policy,
                "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            _save_config(config)
            return {"ok": True, "message": f"VLM configured: {provider}/{model}"}
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"ok": False, "error": "API key authentication failed."}
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def disable_vlm() -> dict[str, Any]:
    """Disable VLM features."""
    config = _load_config()
    config["enabled"] = False
    _save_config(config)
    return {"ok": True, "message": "VLM features disabled."}


# ── VLM call ────────────────────────────────────────────────────────────────


def call_vlm(
    prompt: str,
    image_base64: str,
    *,
    max_tokens: int = 4096,
    timeout: float = 90.0,
) -> str:
    """Call the configured VLM with a text prompt and base64-encoded image.

    Uses the OpenAI-compatible chat completions API with vision support.

    Args:
        prompt: Text instruction for the model.
        image_base64: Base64-encoded PNG image data.
        max_tokens: Maximum response tokens.
        timeout: Request timeout in seconds.

    Returns:
        Model response text, or empty string on failure.
    """
    global _last_call_ts

    config = _load_config()
    api_key = _resolve_api_key()
    if not config.get("enabled") or not api_key:
        return ""

    api_base = config.get("api_base", "https://api.openai.com/v1").rstrip("/")
    model = config["model"]

    # Rate limiting
    now = time.monotonic()
    elapsed = now - _last_call_ts
    if elapsed < _RATE_LIMIT_INTERVAL:
        time.sleep(_RATE_LIMIT_INTERVAL - elapsed)

    import urllib.request

    payload = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
    ).encode()

    for attempt in range(_MAX_RETRIES):
        try:
            _last_call_ts = time.monotonic()
            req = urllib.request.Request(
                f"{api_base}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
                choices = result.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    return str(content).strip()
                return ""

        except Exception as exc:
            delay = _BASE_DELAY * (2**attempt)
            logger.warning("VLM call failed (attempt %d): %s", attempt + 1, exc)
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)

    return ""


def encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64 encoding."""
    import base64

    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")
