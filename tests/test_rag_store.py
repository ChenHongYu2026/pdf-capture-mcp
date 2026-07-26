"""Tests for the RAG layer (embedding client config + Qdrant store).

No network: embed_texts is monkeypatched with a deterministic topic-aware
fake embedder, and Qdrant runs in :memory: mode shared across build/search
within each test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client")

from pdf_capture_mcp import embedding_client, rag_store  # noqa: E402
from pdf_capture_mcp.packaging import file_sha256  # noqa: E402

DOC = "ragdoc0123456789"
DIM = 8

_TOPICS = ["cat", "finance", "table", "neural"]


def _fake_embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Deterministic topic-similarity embedding: same topic -> close vectors."""
    out = []
    for t in texts:
        low = t.lower()
        v = [0.05] * DIM
        for i, topic in enumerate(_TOPICS):
            if topic in low:
                v[i] = 1.0
        # small content-dependent jitter for uniqueness
        h = int(hashlib.sha1(low.encode()).hexdigest()[:4], 16)
        v[DIM - 1] = (h % 100) / 1000
        out.append(v)
    return out


def _mk_package(tmp_path: Path, chunks: list[dict], md_text: str = "# Doc\n\nBody.\n") -> Path:
    pkg = tmp_path / "pkg"
    (pkg / "data").mkdir(parents=True)
    main_md = pkg / "pkg.md"
    main_md.write_text(md_text, encoding="utf-8")
    (pkg / "data" / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
    )
    meta = {
        "schema_version": "1",
        "doc_id": DOC,
        "title": "RAG Test Doc",
        "slug": "pkg",
        "content_hash": file_sha256(main_md),
    }
    (pkg / "data" / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return pkg


def _chunk(i: int, text: str, ctype: str = "text", page: int | None = 1) -> dict:
    return {
        "chunk_id": hashlib.sha1(f"{DOC}|{text}".encode()).hexdigest()[:16],
        "doc_id": DOC,
        "seq": i,
        "heading_path": ["S"],
        "page": page,
        "chunk_type": ctype,
        "content": text,
        "embed_text": text,
        "token_est": 5,
    }


@pytest.fixture()
def rag_env(monkeypatch, tmp_path):
    """Shared in-memory Qdrant + fake embedding config."""
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    monkeypatch.setattr(rag_store, "_get_client", lambda: client)
    monkeypatch.setattr(embedding_client, "is_embedding_enabled", lambda: True)
    monkeypatch.setattr(embedding_client, "get_dimensions", lambda: DIM)
    monkeypatch.setattr(embedding_client, "embed_texts", _fake_embed)
    yield client


BASE_CHUNKS = [
    _chunk(0, "The cat sat on the mat and purred."),
    _chunk(1, "Quarterly finance revenue grew by 12 percent.", ctype="table", page=3),
    _chunk(2, "Neural networks learn hierarchical representations.", page=5),
]


def test_build_then_incremental_noop(rag_env, tmp_path):
    pkg = _mk_package(tmp_path, BASE_CHUNKS)
    r1 = rag_store.build_vector_index(pkg)
    assert r1["ok"] and r1["embedded"] == 3 and r1["unchanged"] == 0
    # Second run: content-addressed ids make everything free.
    r2 = rag_store.build_vector_index(pkg)
    assert r2["ok"] and r2["embedded"] == 0 and r2["unchanged"] == 3


def test_incremental_diff_on_change(rag_env, tmp_path):
    pkg = _mk_package(tmp_path, BASE_CHUNKS)
    rag_store.build_vector_index(pkg)
    changed = [BASE_CHUNKS[0], BASE_CHUNKS[1], _chunk(2, "Neural nets now UPDATED text.", page=5)]
    (pkg / "data" / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in changed) + "\n", encoding="utf-8"
    )
    r = rag_store.build_vector_index(pkg)
    assert r["embedded"] == 1 and r["unchanged"] == 2 and r["deleted"] == 1


def test_stale_package_refused(rag_env, tmp_path):
    """N4 contract: hand-edited markdown must block indexing."""
    pkg = _mk_package(tmp_path, BASE_CHUNKS)
    (pkg / "pkg.md").write_text("# Doc\n\nEDITED BY HAND.\n", encoding="utf-8")
    r = rag_store.build_vector_index(pkg)
    assert not r["ok"]
    assert "STALE" in r["error"]


def test_search_semantic_and_filters(rag_env, tmp_path):
    pkg = _mk_package(tmp_path, BASE_CHUNKS)
    rag_store.build_vector_index(pkg)

    r = rag_store.search_corpus("cat purring on a mat", top_k=2)
    assert r["ok"] and r["hits"][0]["content"].startswith("The cat")

    # chunk_type filter: only the finance table qualifies
    r2 = rag_store.search_corpus("finance revenue", chunk_type="table")
    assert r2["hits"] and all(h["chunk_type"] == "table" for h in r2["hits"])
    assert r2["hits"][0]["page"] == 3

    # page range filter excludes page-3 table
    r3 = rag_store.search_corpus("finance revenue", page_from=4)
    assert all((h["page"] or 0) >= 4 for h in r3["hits"])


def test_search_without_config(monkeypatch):
    monkeypatch.setattr(embedding_client, "is_embedding_enabled", lambda: False)
    r = rag_store.search_corpus("anything")
    assert not r["ok"] and "setup_embedding" in r["error"]


def test_build_rejects_non_package(rag_env, tmp_path):
    r = rag_store.build_vector_index(tmp_path)
    assert not r["ok"]


def test_embedding_config_roundtrip(monkeypatch, tmp_path):
    """setup_embedding validates via a real call — mock the HTTP layer."""
    monkeypatch.setattr(embedding_client, "_CONFIG_PATH", tmp_path / "emb.json")
    monkeypatch.setattr(embedding_client, "_post_embeddings", lambda *a, **k: [[0.1] * 1536])
    r = embedding_client.setup_embedding("embo-01", "sk-test", "https://api.minimaxi.com/v1")
    assert r["ok"] and r["dimensions"] == 1536
    info = embedding_client.get_embedding_info()
    assert info["enabled"] and info["model"] == "embo-01"
    assert "api_key" not in info  # never exposed
    assert embedding_client.disable_embedding()["ok"]
    assert not embedding_client.is_embedding_enabled()
