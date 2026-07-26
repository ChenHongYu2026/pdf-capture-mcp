"""Embedding client: OpenAI-compatible /embeddings API for RAG indexing.

Mirrors llm_client.py's configuration pattern: persisted JSON config with
0600 permissions, env-var key override, validation call before saving.
Region-safe by design — any OpenAI-compatible endpoint works (OpenAI,
MiniMax embo-01, SiliconFlow/BGE, Ollama's /v1 shim for fully-local).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger

logger = get_logger("embedding_client")

_CONFIG_PATH = Path.home() / ".pdf_capture_mcp" / "embedding_config.json"


def _load_config() -> dict[str, Any]:
    if _CONFIG_PATH.exists():
        try:
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(config: dict[str, Any]) -> None:
    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        _CONFIG_PATH.chmod(0o600)  # protect API key
    except OSError:
        pass


def _resolve_api_key() -> str:
    config = _load_config()
    return config.get("api_key", "") or os.getenv("PDF_CAPTURE_EMBEDDING_API_KEY", "").strip()


def is_embedding_enabled() -> bool:
    config = _load_config()
    return config.get("enabled", False) and bool(_resolve_api_key())


def get_embedding_info() -> dict[str, Any]:
    """Public view of the embedding configuration (never exposes the key)."""
    config = _load_config()
    return {
        "enabled": is_embedding_enabled(),
        "provider": config.get("provider", ""),
        "model": config.get("model", ""),
        "api_base": config.get("api_base", ""),
        "dimensions": config.get("dimensions", 0),
        "validated_at": config.get("validated_at", ""),
    }


def setup_embedding(
    model: str,
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    provider: str = "openai",
) -> dict[str, Any]:
    """Configure and validate an OpenAI-compatible embedding endpoint.

    Performs one real embedding call to verify credentials AND record the
    vector dimensionality — the vector store needs it at collection
    creation, and a silent dimension change would corrupt the index.
    """
    if not model.strip():
        return {"ok": False, "error": "Model name is required."}
    if not api_key.strip():
        return {"ok": False, "error": "API key is required."}

    try:
        vectors = _post_embeddings(
            api_base, api_key, model, ["dimension probe"], timeout=30, provider=provider
        )
    except Exception as exc:  # noqa: BLE001 — report validation failure verbatim
        return {"ok": False, "error": f"Validation call failed: {exc}"}
    if not vectors or not vectors[0]:
        return {"ok": False, "error": "Endpoint returned no embedding vector."}
    dimensions = len(vectors[0])

    _save_config(
        {
            "enabled": True,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "api_base": api_base,
            "dimensions": dimensions,
            "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    return {
        "ok": True,
        "message": (
            f"Embedding configured: {provider}/{model} ({dimensions} dimensions). "
            "build_vector_index and search_corpus are now available."
        ),
        "model": model,
        "provider": provider,
        "dimensions": dimensions,
    }


def disable_embedding() -> dict[str, Any]:
    config = _load_config()
    config["enabled"] = False
    _save_config(config)
    return {"ok": True, "message": "Embedding disabled."}


def _post_embeddings(
    api_base: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: float,
    *,
    provider: str = "openai",
    purpose: str = "db",
) -> list[list[float]]:
    import httpx

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{api_base.rstrip('/')}/embeddings"

    if provider == "minimax":
        # MiniMax speaks its own dialect: 'texts' + 'type' (db|query —
        # asymmetric embeddings, which we exploit: documents are embedded
        # as 'db', search queries as 'query'), response key 'vectors'.
        response = httpx.post(
            url,
            headers=headers,
            json={"model": model, "texts": texts, "type": purpose},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        body = response.json()
        base = body.get("base_resp", {})
        if base.get("status_code", 0) != 0:
            raise RuntimeError(f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}")
        vectors = body.get("vectors") or []
        if len(vectors) != len(texts):
            raise RuntimeError(f"Expected {len(texts)} vectors, got {len(vectors)}")
        return vectors

    response = httpx.post(
        url,
        headers=headers,
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
    data = response.json().get("data", [])
    # Providers may return out of order; sort by index to stay aligned.
    data.sort(key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def embed_texts(texts: list[str], batch_size: int = 64, purpose: str = "db") -> list[list[float]]:
    """Embed texts in batches with one retry per batch. Raises on failure —
    a partially embedded index is worse than a clear error.

    purpose: 'db' for documents being indexed, 'query' for search queries
    (asymmetric-embedding providers like MiniMax use it; others ignore it).
    """
    config = _load_config()
    if not is_embedding_enabled():
        raise RuntimeError("Embedding not configured. Call setup_embedding first.")
    api_key = _resolve_api_key()
    provider = config.get("provider", "openai")
    out: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            out += _post_embeddings(
                config["api_base"],
                api_key,
                config["model"],
                batch,
                60,
                provider=provider,
                purpose=purpose,
            )
        except Exception:  # noqa: BLE001 — one retry for transient faults
            logger.warning("Embedding batch %d failed, retrying once", start // batch_size)
            time.sleep(2)
            out += _post_embeddings(
                config["api_base"],
                api_key,
                config["model"],
                batch,
                60,
                provider=provider,
                purpose=purpose,
            )
    return out


def get_dimensions() -> int:
    return int(_load_config().get("dimensions", 0))
