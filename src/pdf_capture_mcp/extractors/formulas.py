"""Formula re-recognition: fix broken LaTeX from extraction engines.

Detects broken formula placeholders (??, empty $...$, garbled LaTeX) and
attempts repair via VLM (Anthropic) when available.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger

logger = get_logger("extractors.formulas")

# Patterns for broken formulas
_BROKEN_PATTERNS = [
    re.compile(r"\?\?+"),  # ?? placeholders
    re.compile(r"\$\s+\$"),  # empty inline math
    re.compile(r"\$\$\s+\$\$"),  # empty display math
    re.compile(r"\$[A-Z]\s_\s\{[^}]*\}\$"),  # garbled: $P _ {tex}$
]

# Pattern to detect spaced-out LaTeX (common MinerU artifact)
_SPACED_LATEX = re.compile(r"\$([^$]*\s_\s\{[^$]*)\$")

# VLM prompt for formula recognition
_FORMULA_PROMPT = (
    "This image contains a mathematical formula or equation. "
    "Convert it to LaTeX notation. Return ONLY the LaTeX code, "
    "wrapped in $ for inline or $$ for display math. "
    "If you cannot identify a formula, return empty string."
)


def _find_broken_formulas(text: str) -> list[dict[str, Any]]:
    """Find all broken formula instances in markdown text."""
    broken = []
    for pattern in _BROKEN_PATTERNS:
        for match in pattern.finditer(text):
            broken.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "pattern": pattern.pattern,
                }
            )
    return broken


def _fix_spaced_latex(text: str) -> tuple[str, int]:
    """Fix spaced-out LaTeX like $P _ {t e x}$ → $P_{tex}$."""
    count = 0

    def _fix_match(m: re.Match) -> str:
        nonlocal count
        inner = m.group(1)
        # Remove spaces around _ and ^
        fixed = re.sub(r"\s*_\s*", "_", inner)
        fixed = re.sub(r"\s*\^\s*", "^", fixed)
        # Remove spaces inside {}
        fixed = re.sub(r"\{\s*", "{", fixed)
        fixed = re.sub(r"\s*\}", "}", fixed)
        # Remove single-char spaces in subscripts
        fixed = re.sub(r"(?<=[_^{])\s+", "", fixed)
        if fixed != inner:
            count += 1
        return f"${fixed}$"

    result = _SPACED_LATEX.sub(_fix_match, text)
    return result, count


def rerecognize_formulas(
    pdf_path: str | Path,
    markdown_text: str,
    *,
    enable_vlm: bool = False,
) -> dict[str, Any]:
    """Fix broken formulas in extracted markdown.

    Phase 1: Rule-based fixes (spaced LaTeX, empty math removal).
    Phase 2: VLM-based re-recognition (optional, requires API key).

    Args:
        pdf_path: Source PDF (for page rendering in VLM mode).
        markdown_text: Extracted markdown with potential formula errors.
        enable_vlm: Enable VLM fallback for unfixable formulas.

    Returns:
        Dict with ok, text, fixed_count, broken_count, remaining_broken.
    """
    text = markdown_text

    # Phase 1: Rule-based fixes
    text, space_fixed = _fix_spaced_latex(text)

    # Remove empty math blocks
    empty_removed = 0
    for pattern in [re.compile(r"\$\s+\$"), re.compile(r"\$\$\s+\$\$")]:
        matches = pattern.findall(text)
        if matches:
            text = pattern.sub("", text)
            empty_removed += len(matches)

    # Phase 2: Count remaining broken
    remaining = _find_broken_formulas(text)

    # Phase 3: VLM fallback (optional)
    vlm_fixed = 0
    if enable_vlm and remaining:
        from pdf_capture_mcp.llm_client import is_vlm_enabled

        if is_vlm_enabled():
            logger.info("VLM formula fix: %d broken formulas to attempt", len(remaining))
            # VLM-based fix would render PDF pages and call the API
            # For now, log that it's available but not fully implemented
            # in the standalone package (requires page-level bbox from content_json)
            pass

    total_fixed = space_fixed + empty_removed
    if total_fixed > 0:
        logger.info(
            "Formula fix: %d space-fixed, %d empty-removed, %d remaining",
            space_fixed,
            empty_removed,
            len(remaining),
        )

    return {
        "ok": True,
        "text": text,
        "fixed_count": space_fixed,
        "empty_removed_count": empty_removed,
        "vlm_fixed_count": vlm_fixed,
        "broken_count": len(remaining) + total_fixed,
        "remaining_broken": len(remaining),
    }
