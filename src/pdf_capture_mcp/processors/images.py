"""Image processing: decorative image filtering and caption generation."""

from __future__ import annotations

import re
from typing import Any

from pdf_capture_mcp.config import get_logger

logger = get_logger("processors.images")

# Image reference pattern in markdown
_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# Caption patterns (Exhibit N, Figure N, 图N)
_CAPTION_SOURCE_RE = re.compile(
    r"^(?:Exhibit|Sidebar|Figure|Fig\.?|CHAPTER|图|表)\s*[\d.\-]+",
    re.IGNORECASE,
)


def filter_decorative_images(
    markdown_text: str,
    content_json_path: str = "",
) -> dict[str, Any]:
    """Remove decorative/styling images from markdown.

    Uses heuristics to identify non-content images:
    - Images with no alt text and small file references
    - Images classified as decorative by content analysis

    Args:
        markdown_text: Markdown with image references.
        content_json_path: Path to content_list.json (optional, for classification).

    Returns:
        Dict with ok, text, kept, removed.
    """
    kept = 0
    removed = 0

    def _is_decorative(alt: str, src: str) -> bool:
        """Heuristic: images with empty alt and generic names are decorative."""
        if alt.strip():
            return False  # Has description → keep
        # Common decorative patterns
        src_lower = src.lower()
        decorative_hints = ("logo", "icon", "bullet", "separator", "decoration", "bg_")
        return any(hint in src_lower for hint in decorative_hints)

    def _replace(m: re.Match) -> str:
        nonlocal kept, removed
        alt = m.group(1)
        src = m.group(2)
        if _is_decorative(alt, src):
            removed += 1
            return ""
        kept += 1
        return str(m.group(0))

    cleaned = _IMAGE_PATTERN.sub(_replace, markdown_text)
    # Collapse blank lines from removals
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    if removed > 0:
        logger.info("Image filter: %d kept, %d decorative removed", kept, removed)

    return {"ok": True, "text": cleaned, "kept": kept, "removed": removed}


def generate_captions(markdown_text: str) -> dict[str, Any]:
    """Generate captions for uncaptioned images from surrounding context.

    Looks for nearby heading text or 'Exhibit N' / 'Figure N' patterns
    to use as image captions.

    Args:
        markdown_text: Markdown with image references.

    Returns:
        Dict with ok, text, captions_added.
    """
    lines = markdown_text.split("\n")
    captions_added = 0
    result_lines: list[str] = []

    for i, line in enumerate(lines):
        result_lines.append(line)

        # If this line is an image with empty alt
        m = _IMAGE_PATTERN.match(line.strip())
        if m and not m.group(1).strip():
            # Look backwards for a caption source
            caption = ""
            for j in range(max(0, i - 3), i):
                prev = lines[j].strip()
                if _CAPTION_SOURCE_RE.match(prev):
                    caption = prev[:120]
                    break
                # Or a heading
                if prev.startswith("#"):
                    caption = prev.lstrip("#").strip()[:120]
                    break

            if caption:
                result_lines.append(f"\n*{caption}*\n")
                captions_added += 1

    text = "\n".join(result_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if captions_added > 0:
        logger.info("Caption generation: %d captions added", captions_added)

    return {"ok": True, "text": text, "captions_added": captions_added}
