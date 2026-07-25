#!/usr/bin/env python3
"""End-to-end test: exercise all 6 MCP tools against a real PDF."""

from __future__ import annotations

import asyncio
import json
import sys

TEST_PDF = (
    "/Volumes/Luis_MacData/AgentSystem/output/pdf_pipeline/"
    "from_agi_to_asi-with-annotations_20260614_173026/extraction/"
    "From AGI to ASI-with-annotations/auto/From AGI to ASI-with-annotations_origin.pdf"
)


async def call(mcp, tool: str, args: dict) -> dict:
    """Call an MCP tool and parse the JSON payload."""
    result = await mcp.call_tool(tool, args)
    # FastMCP returns CallToolResult or tuple; extract text content
    text = ""
    content = getattr(result, "content", result)
    if isinstance(content, (list, tuple)):
        for block in content:
            if hasattr(block, "text"):
                text = block.text
                break
            if isinstance(block, str):
                text = block
                break
    elif hasattr(content, "text"):
        text = content.text
    elif isinstance(content, str):
        text = content
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": str(text)[:200]}


async def main() -> int:
    from pdf_capture_mcp.server import mcp

    passed, failed = [], []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            passed.append(name)
            print(f"  ✅ {name} {detail}")
        else:
            failed.append(name)
            print(f"  ❌ {name} {detail}")

    # ── 1. setup_vlm status ─────────────────────────────────────────
    print("\n[1/6] setup_vlm (action=status)")
    r = await call(mcp, "setup_vlm", {"action": "status"})
    check("setup_vlm.status ok", r.get("ok") is True)
    check("setup_vlm.no_api_key_leak", "api_key" not in json.dumps(r))
    print(f"      configured={r.get('configured')}, enabled={r.get('enabled')}")

    # ── 2. setup_vlm enable without key (error path) ────────────────
    print("\n[2/6] setup_vlm (action=enable, missing params)")
    r = await call(mcp, "setup_vlm", {"action": "enable", "model": "qwen-vl-max"})
    check("setup_vlm.enable_no_key rejected", r.get("ok") is False)
    check("setup_vlm.error_mentions_env", "PDF_CAPTURE_VLM_API_KEY" in r.get("error", ""))

    # ── 3. check_environment ────────────────────────────────────────
    print("\n[3/6] check_environment")
    r = await call(mcp, "check_environment", {})
    check("check_environment responds", "ready" in r)
    print(f"      ready={r.get('ready')}")
    print(f"      engines={json.dumps(r.get('engines', {}), ensure_ascii=False)}")
    deps_ok = all(d.get("available") for d in r.get("dependencies", {}).values())
    check("core dependencies available", deps_ok)
    env_ready = r.get("ready", False)

    # ── 4. pdf_info ─────────────────────────────────────────────────
    print("\n[4/6] pdf_info (real PDF)")
    r = await call(mcp, "pdf_info", {"pdf_path": TEST_PDF})
    check("pdf_info ok", r.get("ok") is True)
    check("pdf_info page_count > 0", r.get("page_count", 0) > 0, f"pages={r.get('page_count')}")
    check("pdf_info has_text_layer detected", r.get("has_text_layer") is not None)
    print(f"      size={r.get('size_mb')}MB, scanned={r.get('is_scanned')}")

    # ── 5. classify_document ────────────────────────────────────────
    print("\n[5/6] classify_document (real PDF)")
    r = await call(mcp, "classify_document", {"pdf_path": TEST_PDF})
    check("classify ok", r.get("ok") is True)
    check("classify has doc_type", bool(r.get("doc_type")))
    print(
        f"      doc_type={r.get('doc_type')}, confidence={r.get('confidence')}, "
        f"lang={r.get('language')}, formulas={r.get('has_formulas')}"
    )

    # ── 6. extract_tables ───────────────────────────────────────────
    print("\n[6/6] extract_tables (real PDF, pdfplumber)")
    r = await call(mcp, "extract_tables", {"pdf_path": TEST_PDF, "strategy": "pdfplumber"})
    check("extract_tables ok", r.get("ok") is True)
    stats = r.get("stats", {})
    print(f"      tables found: {stats.get('pdfplumber_tables', 0)}")

    # ── 7. pdf_to_markdown (only if an engine is ready) ─────────────
    if env_ready:
        print("\n[extra] pdf_to_markdown (full pipeline)")
        r = await call(mcp, "pdf_to_markdown", {"pdf_path": TEST_PDF})
        check("pdf_to_markdown ok", r.get("ok") is True)
        md_len = len(r.get("markdown_text", ""))
        print(f"      markdown chars: {md_len}, qc={r.get('qc_verdict')}")
    else:
        print("\n[extra] pdf_to_markdown SKIPPED (no engine installed — expected on dev box)")
        # Verify graceful error instead
        r = await call(mcp, "pdf_to_markdown", {"pdf_path": TEST_PDF})
        check(
            "pdf_to_markdown graceful error",
            r.get("ok") is False and bool(r.get("error")),
            f"error={str(r.get('error'))[:80]}",
        )

    # ── Error path: invalid file ────────────────────────────────────
    print("\n[error paths]")
    r = await call(mcp, "pdf_info", {"pdf_path": "/nonexistent/ghost.pdf"})
    check("pdf_info nonexistent handled", r.get("ok") is False)
    r = await call(mcp, "extract_tables", {"pdf_path": "/etc/hosts"})
    check("extract_tables non-pdf rejected", r.get("ok") is False)

    print(f"\n{'=' * 50}")
    print(f"E2E RESULT: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("Failed checks:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
