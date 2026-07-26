"""VLM arbitration: vision-based repair for defects beyond geometric reach.

The geometric repairers in ``quality/repair.py`` work from the PDF text
layer, which cannot express table SEMANTICS (merged cells, multi-row group
headers). This module escalates to a vision language model that reads a
hi-res render of the region — the only channel that actually "sees" the
table the way a human does.

Discipline (same repair-or-report contract as repair.py):

- Table transcription output is HTML ``<table>`` — Markdown tables cannot
  express rowspan/colspan/multi-level headers, HTML can, and every major
  Markdown renderer accepts inline HTML.
- NUMERIC GATE: the VLM must neither invent numbers absent from the
  region's text layer (hallucination) nor drop numbers present in the
  broken markdown block (data loss). Gate failure -> report, never patch.
- Figure descriptions are injected as quoted text under the image link so
  text-only RAG can retrieve figure-embedded information (MD-202 gap).

VLM calls consume tokens; both features are opt-in via pdf_to_markdown
parameters (``enable_table_enrich``, ``enrich_figures``).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger
from pdf_capture_mcp.quality.md_audit import _TOKEN, AuditIssue
from pdf_capture_mcp.quality.repair import (
    STATUS_REPAIRED,
    STATUS_REPORTED,
    RepairAction,
    _table_block_at,
)

logger = get_logger("quality.vlm_repair")

# Rules whose table defects justify a VLM escalation.
TABLE_RULES = ("MD-104", "MD-105", "MD-107")

_NUM = re.compile(r"\d+(?:\.\d+)?")
_TAG = re.compile(r"<[^>]+>")

_TABLE_PROMPT = (
    "Transcribe this table EXACTLY into a single HTML <table>. Requirements:\n"
    "- Preserve every value verbatim, including decimal points, units and "
    "signs (e.g. 6.0 x 10^-4 as 6.0×10<sup>−4</sup>).\n"
    "- Use <thead> for header rows; use rowspan/colspan to express merged "
    "cells and multi-row group headers exactly as shown.\n"
    "- Do NOT add, omit, or normalize any value. Output ONLY the <table> "
    "element, no commentary."
)

_FIGURE_PROMPT = (
    "Describe this figure from a document in 2-4 sentences for a search "
    "index. Name the visible components, labels, axes and relationships "
    "verbatim where possible. Output plain text only, no markdown."
)


def _strip_reply(reply: str) -> str:
    """Remove thinking blocks and code fences from a VLM reply."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    reply = re.sub(r"```(?:html)?", "", reply)
    return reply.strip()


def _render_region(pdf_path: Path | str, page_no: int, rect: Any, zoom: float = 3.0) -> str:
    """Render a page region to base64 PNG at high resolution."""
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        pix = doc[page_no].get_pixmap(clip=rect, matrix=fitz.Matrix(zoom, zoom))
        return base64.b64encode(pix.tobytes("png")).decode()


def _locate_table_region(
    pdf_path: Path | str, block_lines: list[str]
) -> tuple[int, Any, str] | None:
    """Find the source page region of a markdown table via token matching.

    Returns (page_no, padded_rect, region_text) or None. The rect is the
    union of word boxes whose tokens appear in the table block, restricted
    to the page with the best overlap.
    """
    import fitz

    block_tokens = {t.lower() for line in block_lines for t in _TOKEN.findall(line)}
    block_nums = set(_NUM.findall(" ".join(block_lines)))
    probes = block_tokens | block_nums
    if not probes:
        return None

    best: tuple[int, list[Any]] | None = None
    with fitz.open(str(pdf_path)) as doc:
        for pno, page in enumerate(doc):
            hits = []
            for w in page.get_text("words"):
                word = w[4].lower()
                word_nums = set(_NUM.findall(word))
                if (
                    word.strip(".,()%") in probes
                    or any(t in probes for t in _TOKEN.findall(word))
                    or (word_nums & block_nums)  # numeric cells: '0.5M', '6.0x10-4'
                ):
                    hits.append(fitz.Rect(w[:4]))
            if best is None or len(hits) > len(best[1]):
                best = (pno, hits)

        if best is None or len(best[1]) < 6:  # too few anchors to trust
            return None
        pno, hits = best
        rect = hits[0]
        for r in hits[1:]:
            rect.include_rect(r)
        page = doc[pno]
        padded = fitz.Rect(
            max(rect.x0 - 10, 0),
            max(rect.y0 - 8, 0),
            min(rect.x1 + 10, page.rect.x1),
            min(rect.y1 + 8, page.rect.y1),
        )
        region_text = page.get_text(clip=padded)
    return pno, padded, region_text


def _numeric_gate(html: str, region_text: str, md_block_text: str) -> tuple[bool, str]:
    """Anti-hallucination / anti-loss gate for VLM table output.

    - invented: numbers in the HTML that the region's text layer never had
      (hallucination) -> hard fail.
    - lost: numbers from the broken markdown block that the HTML neither
      contains directly nor as part of a recombined value (torn fragments
      '6' + '0' legitimately become '6.0') -> hard fail.
    Region-edge numbers missing from the HTML (captions etc.) are tolerated.
    """
    html_nums = set(_NUM.findall(_TAG.sub(" ", html)))
    region_nums = set(_NUM.findall(region_text))
    md_nums = set(_NUM.findall(md_block_text))

    invented = html_nums - region_nums
    if invented:
        return False, f"invented numbers: {sorted(invented)[:8]}"

    dotless = [h.replace(".", "") for h in html_nums]
    lost = {
        n
        for n in md_nums - html_nums
        if not any(n in d for d in dotless)  # fragment recombination check
    }
    if lost:
        return False, f"numbers lost from source table: {sorted(lost)[:8]}"
    return True, f"{len(html_nums)} numbers conserved"


def vlm_repair_table(
    md_lines: list[str],
    issue: AuditIssue,
    pdf_path: Path | str,
) -> RepairAction:
    """Escalate one broken table to the VLM; replace with HTML on gate pass."""
    from pdf_capture_mcp.llm_client import call_vlm

    bounds = _table_block_at(md_lines, issue.lines[0])
    if bounds is None:
        return RepairAction(issue.rule, STATUS_REPORTED, "Table block not found.", issue.lines)
    start, end = bounds
    block = md_lines[start : end + 1]

    located = _locate_table_region(pdf_path, block)
    if located is None:
        return RepairAction(
            issue.rule,
            STATUS_REPORTED,
            "VLM escalation skipped: source region could not be located.",
            issue.lines,
        )
    page_no, rect, region_text = located

    try:
        img_b64 = _render_region(pdf_path, page_no, rect)
        reply = _strip_reply(call_vlm(_TABLE_PROMPT, img_b64, max_tokens=4096, timeout=120))
    except Exception as exc:  # noqa: BLE001 — VLM path is strictly best-effort
        return RepairAction(issue.rule, STATUS_REPORTED, f"VLM call failed: {exc}", issue.lines)

    m = re.search(r"<table.*?</table>", reply, flags=re.S)
    if not m:
        return RepairAction(
            issue.rule,
            STATUS_REPORTED,
            "VLM reply contained no <table> element.",
            issue.lines,
        )
    html = m.group(0)

    ok, detail = _numeric_gate(html, region_text, "\n".join(block))
    if not ok:
        return RepairAction(
            issue.rule,
            STATUS_REPORTED,
            f"VLM transcription failed the numeric gate: {detail}. Original kept.",
            issue.lines,
            evidence=html[:400],
        )

    md_lines[start : end + 1] = [html]
    return RepairAction(
        issue.rule,
        STATUS_REPAIRED,
        f"Table replaced with VLM-transcribed HTML (page {page_no + 1}); "
        f"numeric gate passed ({detail}).",
        issue.lines,
        evidence=html[:400],
    )


_IMG_LINE = re.compile(r"^!\[[^\]]*\]\((images/[^)\s]+)\)\s*$")


def vlm_describe_figures(
    md_lines: list[str],
    base_dir: Path | str,
    max_figures: int = 20,
) -> list[RepairAction]:
    """Inject VLM descriptions under image links (closes the RAG gap).

    Figure-embedded text never reaches the markdown body (MD-202), so
    text-only RAG cannot retrieve it. A short indexed description quoted
    under each image makes that information searchable. Skips images that
    already have a description block.
    """
    from pdf_capture_mcp.llm_client import call_vlm

    actions: list[RepairAction] = []
    base = Path(base_dir)
    done = 0
    i = 0
    while i < len(md_lines) and done < max_figures:
        m = _IMG_LINE.match(md_lines[i].strip())
        if not m:
            i += 1
            continue
        if i + 1 < len(md_lines) and md_lines[i + 1].startswith("> **Figure"):
            i += 1
            continue  # already described
        img_path = base / m.group(1)
        if not img_path.exists():
            i += 1
            continue
        try:
            img_b64 = base64.b64encode(img_path.read_bytes()).decode()
            desc = _strip_reply(call_vlm(_FIGURE_PROMPT, img_b64, max_tokens=512, timeout=90))
        except Exception as exc:  # noqa: BLE001 — best-effort per figure
            logger.warning("Figure description failed for %s: %s", img_path.name, exc)
            i += 1
            continue
        if desc and len(desc) > 20:
            desc_line = "> **Figure (VLM description):** " + " ".join(desc.split())
            md_lines.insert(i + 1, desc_line)
            actions.append(
                RepairAction(
                    "MD-202",
                    STATUS_REPAIRED,
                    f"Injected VLM description under {m.group(1)} "
                    "(figure content now retrievable by text-only RAG).",
                    lines=[i + 1],
                    evidence=desc[:200],
                )
            )
            done += 1
            i += 1
        i += 1
    return actions


def run_vlm_arbitration(
    markdown_text: str,
    pdf_path: Path | str,
    issues: list[AuditIssue],
    *,
    base_dir: Path | str | None = None,
    repair_tables: bool = True,
    describe_figures: bool = False,
) -> dict[str, Any]:
    """Run VLM-based table repair and/or figure description injection.

    Returns {"text", "modified", "actions"} — same contract as
    repair.repair_markdown so the pipeline treats both uniformly.
    """
    from pdf_capture_mcp.llm_client import is_vlm_enabled

    if not is_vlm_enabled():
        return {"text": markdown_text, "modified": False, "actions": []}

    md_lines = markdown_text.splitlines()
    actions: list[RepairAction] = []

    if repair_tables:
        for issue in issues:
            if issue.rule in TABLE_RULES:
                actions.append(vlm_repair_table(md_lines, issue, pdf_path))

    if describe_figures and base_dir is not None:
        actions += vlm_describe_figures(md_lines, base_dir)

    text = "\n".join(md_lines) + ("\n" if markdown_text.endswith("\n") else "")
    repaired = sum(1 for a in actions if a.status == STATUS_REPAIRED)
    logger.info(
        "VLM arbitration: %d repaired/injected, %d reported",
        repaired,
        len(actions) - repaired,
    )
    return {"text": text, "modified": text != markdown_text, "actions": actions}
