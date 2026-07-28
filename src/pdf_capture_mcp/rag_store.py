"""RAG vector store: Qdrant (embedded local mode) over knowledge packages.

Architecture decision (user-approved): Qdrant local mode starts embedded —
pure Python, zero services, file-persisted under <output_root>/vector_store.
When the corpus outgrows a laptop, point PDF_CAPTURE_QDRANT_URL at a
Docker/cluster deployment: SAME API, zero code changes. The collection
schema is enterprise-grade from day one:

- Point id: UUID derived from the content-addressed chunk_id — re-indexing
  an unchanged chunk overwrites itself (idempotent upsert).
- Payload: doc_id, title, heading_path, page, chunk_type, seq, content —
  indexed for filtering (doc_type filters, "tables only", page ranges).
- Incremental sync per document: diff existing point ids vs current
  chunks.jsonl — only new/changed chunks are embedded (this is where the
  content-addressed chunk_id design pays off), vanished chunks are deleted.

Staleness contract (audit N4): build_vector_index verifies the package's
content_hash against the main markdown before indexing. A hand-edited md
means chunks.jsonl is stale — we refuse to index silently drifted data.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger
from pdf_capture_mcp.packaging import file_sha256

logger = get_logger("rag_store")

COLLECTION = "pdf_capture_chunks"


def _default_store_path() -> Path:
    root = os.environ.get("PDF_CAPTURE_OUTPUT_ROOT", str(Path.home() / "Documents" / "pdf-capture"))
    return Path(root) / "vector_store"


_client: Any = None
_client_lock = threading.Lock()


def _get_client() -> Any:
    """Embedded local client by default; PDF_CAPTURE_QDRANT_URL switches to a
    server deployment with the same API (the enterprise upgrade path).

    Singleton (v0.9.5): qdrant local mode allows ONE instance per path per
    process — constructing a client per call made batch_convert(index=True)
    collide with concurrent search_corpus ("already accessed by another
    instance").
    """
    global _client
    with _client_lock:
        if _client is not None:
            return _client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - guarded by callers
            raise RuntimeError(
                "qdrant-client is not installed. Install the RAG extra: "
                "pip install 'pdf-capture-mcp[rag]'"
            ) from exc

        url = os.environ.get("PDF_CAPTURE_QDRANT_URL", "").strip()
        if url:
            _client = QdrantClient(url=url)
        else:
            path = _default_store_path()
            path.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(path))
        return _client


def _point_id(chunk_id: str) -> str:
    """Stable UUID from the content-addressed chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def _ensure_collection(client: Any, dimensions: int) -> None:
    from qdrant_client import models

    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE),
    )
    # Payload indexes: the filter dimensions we promise (doc / type / page)
    for field, ftype in (
        ("doc_id", models.PayloadSchemaType.KEYWORD),
        ("chunk_type", models.PayloadSchemaType.KEYWORD),
        ("page", models.PayloadSchemaType.INTEGER),
    ):
        client.create_payload_index(COLLECTION, field_name=field, field_schema=ftype)


def build_vector_index(package_dir: Path | str) -> dict[str, Any]:
    """Index one knowledge package incrementally. Embeds embed_text; stores
    content in the payload for display."""
    from qdrant_client import models

    from pdf_capture_mcp.embedding_client import (
        embed_texts,
        get_dimensions,
        is_embedding_enabled,
    )

    pkg = Path(package_dir)
    meta_path = pkg / "data" / "metadata.json"
    if not meta_path.exists():
        return {"ok": False, "error": f"Not a knowledge package (no data/metadata.json): {pkg}"}
    if not is_embedding_enabled():
        return {"ok": False, "error": "Embedding not configured — call setup_embedding first."}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    main_md = pkg / f"{meta['slug']}.md"

    # N4 staleness contract: refuse to index silently drifted data.
    if main_md.exists() and file_sha256(main_md) != meta.get("content_hash"):
        return {
            "ok": False,
            "error": (
                "STALE PACKAGE: the main markdown has been edited since chunks.jsonl "
                "was generated (content_hash mismatch). Re-run pdf_to_markdown to "
                "regenerate chunks, or revert the edit. Indexing stale chunks would "
                "make search results drift from the actual document."
            ),
            "doc_id": meta["doc_id"],
        }

    chunks = [
        json.loads(line)
        for line in (pkg / "data" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not chunks:
        return {"ok": False, "error": "chunks.jsonl is empty."}

    client = _get_client()
    _ensure_collection(client, get_dimensions())
    doc_id = meta["doc_id"]

    # Incremental diff: content-addressed ids make unchanged chunks free.
    existing: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            with_payload=["chunk_id"],
            limit=512,
            offset=offset,
        )
        existing.update(p.payload["chunk_id"] for p in points)
        if offset is None:
            break

    current = {c["chunk_id"] for c in chunks}
    to_add = [c for c in chunks if c["chunk_id"] not in existing]
    to_delete = existing - current

    if to_add:
        vectors = embed_texts([c["embed_text"] for c in to_add])
        client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=_point_id(c["chunk_id"]),
                    vector=v,
                    payload={
                        "chunk_id": c["chunk_id"],
                        "doc_id": c["doc_id"],
                        "title": meta.get("title", ""),
                        "heading_path": c["heading_path"],
                        "page": c.get("page"),
                        "chunk_type": c["chunk_type"],
                        "seq": c["seq"],
                        "content": c["content"],
                    },
                )
                for c, v in zip(to_add, vectors)
            ],
        )
    if to_delete:
        client.delete(
            COLLECTION,
            points_selector=models.PointIdsList(points=[_point_id(c) for c in to_delete]),
        )

    logger.info(
        "Indexed %s: +%d embedded, %d unchanged (free), -%d removed",
        meta.get("title", doc_id),
        len(to_add),
        len(current & existing),
        len(to_delete),
    )
    return {
        "ok": True,
        "doc_id": doc_id,
        "title": meta.get("title", ""),
        "embedded": len(to_add),
        "unchanged": len(current & existing),
        "deleted": len(to_delete),
        "total_chunks": len(chunks),
        "store": os.environ.get("PDF_CAPTURE_QDRANT_URL") or str(_default_store_path()),
    }


def search_corpus(
    query: str,
    top_k: int = 5,
    doc_id: str = "",
    chunk_type: str = "",
    page_from: int = 0,
    page_to: int = 0,
) -> dict[str, Any]:
    """Semantic search over indexed knowledge packages with metadata filters."""
    from qdrant_client import models

    from pdf_capture_mcp.embedding_client import embed_texts, is_embedding_enabled

    if not query.strip():
        return {"ok": False, "error": "Query is empty."}
    if not is_embedding_enabled():
        return {"ok": False, "error": "Embedding not configured — call setup_embedding first."}

    client = _get_client()
    if not client.collection_exists(COLLECTION):
        return {"ok": False, "error": "No index yet — run build_vector_index on a package first."}

    conditions: list[Any] = []
    if doc_id:
        conditions.append(
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
        )
    if chunk_type:
        conditions.append(
            models.FieldCondition(key="chunk_type", match=models.MatchValue(value=chunk_type))
        )
    if page_from or page_to:
        conditions.append(
            models.FieldCondition(
                key="page",
                range=models.Range(gte=page_from or None, lte=page_to or None),
            )
        )

    vector = embed_texts([query], purpose="query")[0]
    hits = client.query_points(
        COLLECTION,
        query=vector,
        limit=max(1, min(top_k, 50)),
        query_filter=models.Filter(must=conditions) if conditions else None,
        with_payload=True,
    ).points

    return {
        "ok": True,
        "query": query,
        "hits": [
            {
                "score": round(h.score, 4),
                "title": h.payload.get("title", ""),
                "heading_path": h.payload.get("heading_path", []),
                "page": h.payload.get("page"),
                "chunk_type": h.payload.get("chunk_type", ""),
                "content": h.payload.get("content", "")[:1500],
                "doc_id": h.payload.get("doc_id", ""),
                "chunk_id": h.payload.get("chunk_id", ""),
            }
            for h in hits
        ],
    }
