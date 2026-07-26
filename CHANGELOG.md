# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2026-07-26

Field report: converted documents looked "formula-free" in math-enabled
Markdown viewers, which errored with "You can't use macro parameter
character # in math mode". Root cause: the engine escapes brackets in
citation-anchor links (`[\[MCCD13,](#page-71-0)`); MathJax/KaTeX renderers
read `\[` as a display-math opener and choke on the anchor's `#`,
wrecking formula rendering document-wide (108 collision spans in one
audited paper). The formulas themselves were intact all along.

### Added

- `MD-108` (warn, auto-fix): citation-link brackets are de-escaped in the
  sanitizer — `[\[` → `[[` and `\]](` → `]](`. Link-scoped shapes only, so
  genuine display math (`\[x^2\]`) is never touched; CommonMark rendering
  is unchanged (brackets still display literally).
- `MD-108` residual detector: any remaining escaped bracket co-located with
  an anchor link on the same line is reported for manual review.

## [0.4.1] - 2026-07-26

Calibration patch driven by a real two-column paper conversion where the QC
system produced two false alarms (a wrongful HALT and an inflated content-loss
ratio). On that document the verdict now moves from HALT to PASS while real
defects remain reported.

### Fixed

- `qc_gate` formula integrity: single-letter formulas like `$k$`, `$M$`, `$K$`
  are legitimate math symbols and no longer counted as broken (previously any
  `len < 2` inner content failed, HALTing documents with few formulas).
- `MD-201` coverage is now hyphenation-normalized: PDF text-layer words ending
  in `-` are merged with the next word, so correct dehyphenation in the
  markdown no longer shows up as fake deficits (~3.4pp on the test paper).
- `MD-201` no longer bills figure-embedded text as body content loss: words
  inside merged vector-drawing/image regions are accounted separately.

### Added

- `MD-202` (info): figure-text omission report — tokens embedded in vector
  figures that never reach the markdown body (expected engine behavior;
  suggests VLM enrichment when figure text matters).
- `MD-107` (warn): flattened multi-row group headers — detects wide tables
  whose header cells repeat identical trailing metric names with extra glued
  group words (e.g. 'Knowledge Substring EM' / 'Temporal Substring EM').
- `repair.py` MD-201 injection now uses the same body-token accounting as
  detection (hyphenation-normalized, figure text excluded).

## [0.4.0] - 2026-07-26

Closes the loop opened in 0.3.0: defects that could only be DETECTED are now
REPAIRED when a verification gate passes ("repair-or-report" guarantee — no
defect is ever silently guessed at, and none silently passes).

### Added

- New module `src/pdf_capture_mcp/quality/repair.py` — cross-channel repair
  using the pymupdf text layer (independent of engine layout analysis):
  - `MD-104` torn scientific notation: true cell values recovered from
    text-layer word runs. GATE: candidate minus decimal points must equal
    the joined fragments — digits are never changed, only the lost '.'
    restored.
  - `MD-105` fused table headers: the whole table is rebuilt from text-layer
    word geometry (columns via x-interval clustering). GATE: token multiset
    conservation — content is rearranged, never invented.
  - `MD-201` content loss: missing word runs re-injected next to anchors
    that exist in the markdown. GATES: per-token counts may never exceed the
    PDF text layer (over-injection rolls back everything); bulk deficits
    (>5%) are refused and reported.
- Pipeline phase `repair` between audit and QC gate; verified repairs are
  re-audited so cleared issues disappear from `qc_report.audit_issues`;
  all attempts (repaired and reported) listed in `qc_report.repairs` with
  evidence strings.
- `pdf_to_markdown` gains `auto_repair: bool = True`.
- Regression corpus `tests/test_repair.py`: synthetic golden PDFs reproduce
  each defect class; gates are proven to refuse mismatched sources.

## [0.3.0] - 2026-07-26

Driven by a manual quality audit of a real 75-page paper conversion: every
actual defect (control chars, torn numeric columns, fused table headers,
author-block content loss) passed the existing statistical QC unnoticed.

### Added

- New module `src/pdf_capture_mcp/quality/md_audit.py` — content-aware audit:
  - `MD-101` garbled chars (U+FFFD / private-use area) — detect
  - `MD-102` C0 control chars at in-cell word wraps — detect + auto-fix
    (chars removed, word fragments rejoined)
  - `MD-103` all-empty table header rows — detect
  - `MD-104` numeric column tearing (scientific notation split across cells,
    decimal point lost) — detect with per-row line numbers
  - `MD-105` table header fused with first data row — detect
  - `MD-106` empty `<span></span>` placeholder cells — detect + auto-fix
  - `MD-201` content-loss coverage: token-multiset comparison against the
    PDF text layer (pymupdf), reporting missing-token examples
- `pdf_to_markdown` response gains `qc_report` (verdict, dimension scores,
  `audit_issues`, `audit_fixes`, counts); sanitized markdown is written back
  to `full_text.md`.
- Any `critical` audit finding escalates a `PASS` verdict to `WARN`.
- README (EN/中文): new "Quality Audit Rules" section with the full rule
  catalog, auto-fix policy, and remediation guidance.

### Fixed

- Wired the existing multi-dimensional `qc_gate` (text completeness, heading
  structure, formula integrity, table coverage) into the conversion pipeline
  — it was previously implemented but never invoked; the pipeline only ran a
  character-count check.

## [0.2.0] - 2026-07-26

Driven by real-world usage on slow/restricted networks and large documents
(a 75-page paper that took ~30 minutes to convert — far beyond any MCP
client timeout).

### Added

- **Background job mode** (`src/pdf_capture_mcp/jobs.py`): long-running work
  now runs in daemon threads with state persisted to `<cache>/jobs/*.json`,
  surviving server restarts.
- `pdf_to_markdown` gains `mode` (`auto` / `sync` / `async`) and `page_range`
  parameters. In `auto` (default), PDFs above 15 pages return a `job_id`
  immediately instead of blocking until the MCP client times out.
- New tool `get_job_status(job_id)`: poll job stage/elapsed; returns
  `markdown_path` plus a 2000-char preview when done. Empty `job_id` lists
  the 10 most recent jobs.
- New tool `download_models(engine)`: pre-download all marker/surya models
  (~2GB) outside surya's 300s inference-server startup window — the main
  failure mode on slow networks. Runs as a background job.
- New module `src/pdf_capture_mcp/models.py`: model cache inspection
  (`check_model_cache`) and mirror-aware download helpers.
- `check_environment` now reports per-model cache status (`models_ready`)
  and current network configuration (`HF_ENDPOINT`, `HF_HUB_OFFLINE`, proxy).
- Proxy sanitation at server startup: when an HTTP proxy is configured,
  `localhost,127.0.0.1` is appended to `NO_PROXY` automatically so surya's
  local inference health checks are not routed through the proxy.
- HF mirror compatibility: when `HF_ENDPOINT` points to a mirror,
  `HF_HUB_DISABLE_XET=1` is applied automatically (mirrors such as
  hf-mirror.com do not support the Xet transfer protocol).
- README (EN/中文): new "Large PDFs & Timeouts" and "Slow / Restricted
  Networks" sections, expanded environment variable and troubleshooting
  tables.

### Changed

- `pdf_to_markdown` internals refactored into a reusable `_run_pipeline`
  shared by the sync path and background jobs (behavior in `mode="sync"`
  is 100% backward compatible).

## [0.1.0] - 2026-07-19

- Initial release: pymupdf/marker/mineru engines, VLM enhancement,
  quality gate, 7 MCP tools, PyPI publishing.
