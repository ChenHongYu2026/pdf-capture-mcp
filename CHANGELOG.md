# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] - 2026-07-29

### Added

- **Vector indexing is now standard equipment.** Same doctrine as the
  v0.6.0 VLM features: configuring embedding (setup_embedding) IS the
  opt-in. `pdf_to_markdown` and `batch_convert` gained a tri-state
  `index` parameter defaulting to 'auto' — every packaged conversion is
  indexed into the local Qdrant store automatically, incrementally
  (content-addressed chunks are never re-embedded). Explicit 'off'
  wins; index failures NEVER fail a finished conversion (reported in
  the response `index` field and the `features.vector_index` block).
  batch_convert's old `index: bool = False` flips to 'auto' and the
  per-file child owns the indexing (no double work).

## [0.11.3] - 2026-07-29

### Fixed

- `list_recent` ordered jobs by file mtime, but `_mutate` persists
  outside the registry lock — an older job finishing late could touch
  its file after a newer job was created, flipping the order (stable
  repro on fast Linux CI runners). Now ordered by the job's own
  created_at.

## [0.11.2] - 2026-07-29

### Fixed

Final-audit findings from the 902-page field run:

- `_merge_segment` renamed macOS AppleDouble junk (`._page_x.jpeg` ->
  `s0_.__page_x.jpeg`), laundering the `._` marker away — the 902-page
  package carried 714 ghost files (one per real figure) that no longer
  looked like metadata. Junk is now skipped at merge time.
- `_ensure_healthy_stdin` leaked the scratch /dev/null descriptor on
  every repair (review finding); it is now closed after dup2.

## [0.11.1] - 2026-07-29

### Fixed

Three process-lifecycle guards, each paying for a lesson the 902-page
scanned-book field run taught the hard way, plus a version-desync fix
(`__init__.__version__` was left at 0.10.0 by the 0.11.0 release, so
tool responses reported the wrong generator version):

- **Orphan watchdog**: segment and pipeline children now hard-exit when
  reparented. Killing the coordinator never cascaded (daemon=False is
  required for marker's own workers), and an orphaned segment child once
  kept OCRing for 2.6 h, racing the successor run's seg_* dirs.
- **stdin immunity**: the parent re-anchors a dead fd 0 to /dev/null
  before every spawn. nohup without `< /dev/null` left stdin on the
  launching pty; after pty reclaim every spawned child died in
  init_sys_streams ("Bad file descriptor") before any Python ran.
- **surya client-timeout sync**: segment children align
  SURYA_INFERENCE_TIMEOUT_SECONDS with their segment budget before surya
  is imported. The 600 s default mass-expired all queued requests of an
  80-page OCR window (13 consecutive APITimeoutErrors, zero output)
  while the segment budget was still fine. Explicit user settings win.

## [0.11.0] - 2026-07-28

A post-import audit of a real Obsidian knowledge package surfaced the
retrieval noise the chunker used to emit: 8-token image-only chunks that
could never anchor to a page, footnote fragments posing as sections
(`['Introduction', '1']`), and `<sup>` footnote lines leaking into table
embed context. This release closes those gaps and derives human-readable
document titles instead of download-artifact filenames.

### Added

- **N13 micro-chunk merge.** Text chunks under `MIN_CHUNK_TOKENS` (25) —
  including image-only links that slip past the N6 fold at section
  boundaries — merge into an adjacent text chunk under the same heading,
  BEFORE page anchoring so the merged content anchors properly. The merge
  never crosses a heading_path boundary, never touches table/figure/code
  chunks, and never exceeds the hard cap; unmergeable noise survives
  tagged `extra["micro"]`.
- **N14 footnote demotion.** Pure-numeric H4+ pseudo-headings (`#### 3`,
  marker's rendering of footnote definitions) no longer become fake
  heading_path entries; the marker digit joins the note body as an
  `"N. "` text prefix.
- **`_derive_title`.** The document title now comes from the PDF metadata
  Title field when it looks like real prose (sane length, has words, not
  an authoring-tool source filename), falling back to the filename stem.
  `2026_..._050526-4_69f9b41ca0945` becomes
  "Microsoft 2026 Work Trend Index Annual Report" in frontmatter, README
  and slug.

### Fixed

- Table embed context no longer picks `<sup>` footnote lines as captions
  (S2 refinement — `(--) Additional country data...` was masquerading as
  a table caption in embed_text).
- The package README file map omits the `tables/` row when zero CSVs were
  extracted — the directory does not exist in that case, and the row sent
  agents to a dead path.

### Changed

- Re-converting an existing document can change some chunk_ids (merged
  chunks re-derive their content-addressed ids, S6/N1 contract intact).
  `build_vector_index` reconciles incrementally: stale points are
  deleted, unchanged chunks are not re-embedded.

## [0.10.0] - 2026-07-28

### Added

- **Scanned-document support hardening.** A 900-page image-only scan used
  to be a worst case: OCR-slow segments hit the fixed 20-min timeout, then
  "degraded" to the pymupdf text-layer engine — which returns EMPTY text
  for image-only pages while reporting ok=True. Whole 80-page windows
  could vanish silently. Now:
  - `classifier.detect_text_layer()` — single source of truth for scan
    detection (evenly sampled pages; `is_scanned` +
    `text_layer_coverage` on ClassifyResult and `pdf_info`).
  - Scanned documents automatically run marker with `force_ocr=True` and
    a 3x per-segment budget.
  - Segment children emit a `ready` sentinel after model warmup, so the
    1-3 minute spawned model load no longer eats the OCR budget
    (`SEGMENT_READY_TIMEOUT_S`, two-phase wait).
  - Honest loss over fake rescue: a failed window with no text layer is
    never sent to pymupdf. It gets one half-window OCR retry; whatever
    still fails is recorded in `metadata.missing_segments` with an
    explicit `> [WARNING] ... content NOT captured.` placeholder in the
    markdown, surfaces in qc_report, package metadata, and a top-level
    response `warning`, and downgrades PASS to WARN.
  - Near-empty guard: any fallback product under 50 chars/page counts as
    a missing window, not a degradation — empty text can no longer
    impersonate preserved content.
  - Segment checkpoints: finished segments persist rewritten markdown +
    a kwargs-hash `.seg_meta.json`; re-running the same out_dir after an
    interruption reuses them instead of re-OCRing (a 6-15 h scan job now
    loses at most the current segment on restart). Cleaned up only after
    a successful merge.
  - `PDF_CAPTURE_SEGMENT_TIMEOUT_S` env var (default 1200).
  - Honest ETA (1.5 min/page for scans) in the async hint;
    `batch_convert` stretches its per-file breaker to
    `max(user value, pages * 1.5 min)` for scanned files (a 900-page scan
    is no longer guaranteed dead at the default 40 min).
  - qc_report `notes` states that MD-201/MD-202 and VLM table repair are
    not applicable without a text layer (blindness made explicit).

## [0.9.6] - 2026-07-28

### Fixed

- Jobs GC throttle used 0.0 as its "never ran" sentinel — but
  time.monotonic() has an UNDEFINED origin (boot time on Linux), so on a
  freshly booted machine/CI container the first sweep was silently
  skipped for up to an hour. Sentinel is now None. Found by the
  v0.9.5 spec-completion CI run (the new GC tests caught it on Linux
  while passing on a long-uptime macOS).

## [0.9.5] - 2026-07-28

Audit release: a full-codebase three-perspective review (completeness /
correctness / impact) at v0.9.4 surfaced 13 findings plus 2 discovered
during planning. All fixed here; default behavior is preserved except
where noted.

### Security

- `restart_inference_services` no longer runs a system-wide `pkill -f`:
  kills are restricted to the CURRENT USER and patterns are anchored to
  our own venv/cache paths, with a user-scoped broad fallback only on
  zero match — unrelated processes (e.g. your own llama-server) are never
  touched.

### Fixed

- `metadata.json` now records `degraded_segments`, fulfilling the 0.9.4
  contract (previously only qc_report had it); also fixed a latent bug
  where `skip_qc=True` dropped the degradation info entirely.
- MD-106 fix counts placeholders BEFORE replacing (was always reporting
  the line count, undercounting multi-per-line cells).
- Job state mutations are lock-protected with snapshots persisted outside
  the lock; `get_job` returns consistent copies (no more torn reads).
- Qdrant embedded client is a process-wide singleton — fixes
  "already accessed by another instance" when batch indexing runs
  concurrently with search_corpus.
- Thread-safe VLM rate limiting (reserve-slot: concurrent callers space
  out without serializing HTTP requests).

### Changed

- `export_to_obsidian` / `export_package_to_vault`: a DIVERGED vault copy
  (hand-edited or unreadable metadata) is refused with `conflict=True`
  instead of silently overwritten; pass `overwrite=True` to replace.
  batch_convert records `vault_conflict` per file.
- qc_gate table coverage now uses the shared table-block parser instead
  of the row-count/3 heuristic — scores are more accurate; historical
  WARN judgments may calibrate (WARN dimension only, never HALT).
- `enable_tatr` is documented as a deprecated no-op (use
  extract_tables(strategy='tatr')); a warning is logged when passed.
- Removed the never-wired `processors/layout.py` and the "layout
  cleaning" pipeline claim from docs (its role is covered by the
  chunker's running-header filter).
- Embedding config unified into the shared cache dir (honors
  PDF_CAPTURE_CACHE_DIR); existing config at ~/.pdf_capture_mcp/ is
  lazily COPIED (never deleted) on first read.
- Persisted job files are garbage-collected (30 days / 500 newest,
  throttled hourly); README tool count corrected to 14.

## [0.9.4] - 2026-07-28

Fourth pilot forensic finding, structural fix. The 245-page regression
showed segment 1 (80 pages) passing cleanly in 8.6 min while segment 2
wedged the OCR service with 10 timeouts: windowing solved TOTAL overload,
but a single ultra-dense page can still kill its segment — and marker
retries internally forever, so the failure is invisible from inside.

### Added

- **Per-segment circuit breaker**: each segment runs in its own subprocess
  with a hard timeout (20 min default). Timeout/failure -> inference
  services restarted -> one retry in a fresh subprocess.
- **Honest degradation**: a segment failing twice falls back to the
  pymupdf text-layer engine (content preserved in seconds, layout fidelity
  reduced) instead of killing the document. Degraded segments are recorded
  in metadata['degraded_segments'], surfaced in qc_report, and force the
  verdict to at least WARN — degradation is never silent. Batch runs no
  longer have a "whole document FAILED" category for oversized files.

### Fixed

- Batch per-file isolation used a daemonic subprocess; daemonic processes
  may not have children, which broke segmented extraction (and marker's
  own workers) when nested inside batch_convert. Now daemon=False.

## [0.9.3] - 2026-07-27

Scale-resilience release. A 10-document stratified pilot (498 pages) produced
three forensic findings: a 245-page PDF overloaded the resident inference
service AND its backlog poisoned every subsequent document; a formula-dense
58-page paper ran 115 minutes; memory stayed flat (the feared leak never
happened). Each finding gets a structural fix.

### Added

- **Segmented extraction for oversized documents**: PDFs above 100 pages
  are extracted in 80-page windows and merged — per-segment image
  namespacing (s<k>_ prefix + reference rewrite), ordered md concatenation,
  one retry per segment after a service restart. Keeps every inference
  batch inside the service's safe zone.
- **`restart_inference_services()`** (marker engine): kills the resident
  llama-server / surya helpers so the next call starts clean; marker
  re-spawns them on demand. Called automatically on segment failure and
  batch-file timeout — the contagion path is now self-healing.
- **Per-file circuit breaker in batch_convert**
  (`per_file_timeout_minutes=40`): each file runs in a fresh subprocess —
  timeouts are enforceable, memory returns to the OS between documents, a
  timed-out file is recorded and the batch moves on. The night budget is
  now bounded.

### Fixed

- marker engine accepted `page_range` via kwargs but never wired it into
  the marker config — page-limited extraction silently converted the whole
  document. Fixed (segmented extraction depends on it).

## [0.9.2] - 2026-07-27

Forensics follow-up to 0.9.1: re-running the magazine's Exhibit-4 table
revealed a third layer of the problem — marker had read the year headers
from pixels but placed them in a figure ALT-TEXT next to the table, so the
L1 baseline missed them and the sparse table block made the L2 ratio
meaningless.

### Fixed

- Numeric-gate L1 baseline now includes the md NEIGHBORHOOD (±6 lines
  around the table block): marker's independently-seen numbers count no
  matter where its layout analysis placed them.
- L2 requires a minimum sample (>=4 known numbers) before the coverage
  ratio may hard-veto; sparse evidence escalates to L3 verification
  instead of failing.

## [0.9.1] - 2026-07-27

Autonomous-gate release, driven by an InDesign magazine field audit (79
pages, per-page nav bars, corrupted ligature text layer). Goal: 100-magazine
batches with ZERO per-table human review.

### Fixed

- **Three-tier autonomous numeric gate** (vlm_repair): a corrupted text
  layer no longer vetoes correct VLM transcriptions.
  - L1 dual-vision baseline: marker's OCR output is an independent visual
    channel — numbers seen by both vision systems are never "invented".
  - L2 text-layer health: a layer covering <50% of marker's numbers is
    unfit to veto; disputed numbers escalate instead of hard-failing.
  - L3 VLM self-verification: one discrimination round ("are these numbers
    visible in the image?") — far lower hallucination rate than generation.
    Unconfirmed -> report (keep marker's table); pipeline never blocks on
    a human.
- **MD-201 running-furniture exemption**: lines repeating across >=30% of
  pages (chapter nav bars, branded footers) are exempt from the coverage
  deficit — the magazine audit showed the engine's correct noise cleanup
  being billed as 7.71% "critical content loss".

## [0.9.0] - 2026-07-27

Scale-out release: directory-level batch processing plus the two precision
items observed during v0.8.x field validation.

### Added

- **`batch_convert` tool**: convert every PDF under a directory into
  knowledge packages as one background job (poll get_job_status; per-file
  progress and results, one failure never aborts the batch). Optional
  per-file vault export and vector indexing. Content-addressed dedup:
  a PDF whose doc_id already has a package is skipped regardless of
  filename (skip_existing=True).

### Fixed

- heading_path / heading_tree metadata no longer carry marker's
  `<span id="page-N">` anchors (main markdown keeps them — internal links
  still work; citations are now clean).
- VLM table-region location uses the densest y-band of token hits: stray
  matches elsewhere on the page (citations sharing a number) no longer
  inflate the crop — fewer numeric-gate rejections for VLM table repair
  (the v0.5.x precision item, closed).

## [0.8.1] - 2026-07-27

Field-validation patch: the v0.8.0 end-to-end test against a real provider
surfaced a dialect gap, fixed the same day.

### Fixed

- MiniMax's /embeddings endpoint speaks its own dialect ('texts' + 'type'
  request, 'vectors' response) instead of the OpenAI schema; provider=
  'minimax' now adapts automatically. Bonus: MiniMax's asymmetric
  embeddings are exploited properly — documents embed as type='db',
  search queries as type='query' (better retrieval quality).
- Field-validated end to end: 277 chunks of a real 75-page paper indexed
  via embo-01 (1536 dims); semantic queries hit the pages previously
  verified by the stranger-agent test.

## [0.8.0] - 2026-07-27

RAG layer: the knowledge packages become a searchable corpus. Qdrant runs
EMBEDDED by default (zero services, file-persisted) and upgrades to a
Docker/cluster deployment by setting PDF_CAPTURE_QDRANT_URL — same API,
zero code changes. This is the "enterprise-grade from day one, enterprise-
cost from day never" architecture the vector-store evaluation settled on.

### Added

- `embedding_client.py`: OpenAI-compatible /embeddings client (OpenAI,
  MiniMax embo-01, SiliconFlow/BGE, local Ollama shim). Validation call
  records vector dimensionality; key stored 0600 or via
  PDF_CAPTURE_EMBEDDING_API_KEY.
- `rag_store.py` + three tools:
  - `setup_embedding` — configure/status/disable.
  - `build_vector_index` — incremental per-document sync: content-
    addressed chunk ids make unchanged chunks FREE (only new/changed
    chunks are embedded, vanished chunks deleted). Enforces the N4
    staleness contract: refuses to index a package whose main markdown
    was edited after chunking (content_hash mismatch).
  - `search_corpus` — dense semantic search + payload filters (doc_id,
    chunk_type, page range); returns heading_path + page for citation.
    The MCP tool IS the RAG API — no extra HTTP service to run.
- Collection schema (enterprise-grade day one): cosine vectors sized from
  the validated embedding dimension; payload indexes on doc_id /
  chunk_type / page; point ids are UUIDs derived from chunk_id
  (idempotent upsert).
- Optional dependency group `[rag]` (qdrant-client); base install stays
  lean.

## [0.7.1] - 2026-07-27

Polish release: the three defects found by the v0.7.0 "stranger agent"
acceptance test (a zero-context agent scored the package 9/10; these were
the deductions).

### Fixed

- marker engine could report page_count=0 (metadata key varies across
  versions) -> pymupdf fallback count; READMEs no longer say "0 pages".
- `---` thematic breaks were mistaken for repeated running headers by the
  chunker's N12 filter; header candidates now require actual content.
- Summary extraction collapses marker's ligature-echo artifact
  ("task-speciﬁc ﬁne-tuning task-specific fine-tuning") via NFKC folding +
  adjacent n-gram dedup; distant legitimate repetition untouched.

## [0.7.0] - 2026-07-27

Knowledge packages: from "conversion output" to a self-describing knowledge
asset — readable by any LLM agent from one README, vault-ready for Obsidian,
and chunked for RAG. The design went through TWO audit rounds (13 severe +
11 medium findings) before a line of code was written; every fix below
references its audit finding.

### Added

- **`chunking/chunker.py`** — hierarchical chunker:
  - Heading-path chunks with same-named-sibling disambiguation (N11);
    tables/code as dedicated chunks; oversized tables split by row groups
    with the header repeated per part + part_n/total metadata (S1).
  - `content` vs `embed_text` field separation (N2): context (caption,
    heading path) enriches embeddings (S2) without affecting chunk ids.
  - Content-addressed `chunk_id` = sha1(doc_id | heading_path | dup_index
    | content) — position-independent (S6), collision-free for repeated
    blocks (N1).
  - Bare figure links fold into surrounding text; only VLM-described
    figures become chunks (N6).
  - Zero-dependency token estimation — CJK-aware, HTML tags stripped for
    tables (S7/M1/N8); no tiktoken CDN dependency.
  - Monotonic page anchoring against the PDF text layer (S4); scanned
    PDFs report page=null honestly (M4); running headers dropped and
    reported (N12).
- **`packaging.py`** — self-describing knowledge package:
  - Layout: `<slug>/<slug>.md` (Obsidian [[slug]] direct hit) + README.md
    (agent entry map with file table + chunks schema) + images/ + tables/
    (page-stamped CSVs, N9) + data/ (chunks.jsonl, metadata.json,
    qc_report.json — now archived on disk).
  - Naming standard: doc_id = sha256(pdf)[:16] content identity; NFC slug,
    fullwidth-char cleanup, ≤60 chars, hash suffix only on collision
    (M3/N10); timestamps never in file names (idempotent re-runs).
  - Frontmatter injected LAST, after all QC phases (N3); metadata carries
    content_hash so a future vector indexer can detect hand-edited
    markdown before indexing stale chunks (N4).
  - Honest summary chain: abstract -> first substantial paragraph ->
    explicit "unavailable" (N7), source recorded.
- **MD-110** (`quality/cross_page_tables.py`): cross-page table merge
  behind a geometric three-evidence gate — previous table touches page
  bottom AND next table touches page top AND no caption between (S3).
  Same-column but separate tables are reported, never merged. Continuation
  detection tolerates heading-only gaps (M2).
- **`export_to_obsidian` tool**: copies the whole package into a vault
  (optional category), idempotent by content_hash. Never flattens, never
  rewrites links — the folder is the namespace (N5 superseded audit fix
  S5's asset relocation).
- `pdf_to_markdown` gains `package: bool = True`; default output root is
  now `$PDF_CAPTURE_OUTPUT_ROOT` or `~/Documents/pdf-capture` instead of a
  temp dir when packaging.
- `AuditIssue` gains a structured `evidence` field (consumed by MD-110).

## [0.6.0] - 2026-07-26

Feature-activation redesign: deep capabilities no longer sleep silently.
Users who configured a VLM often didn't know they still had to pass extra
flags per call — the quality tier they paid to unlock never ran.

### Changed

- `pdf_to_markdown` VLM flags are now tri-state strings defaulting to
  `'auto'`: `enable_table_enrich` and `enrich_figures` activate
  automatically when a VLM is configured and its stored policy allows.
  Configuring a VLM (setup_vlm) is itself the opt-in to spend tokens on
  quality. Explicit `'on'`/`'off'` (or booleans) always override.
- Zero-cost tiers (sanitize, audit, geometric repair) remain always-on.

### Added

- `setup_vlm` gains `policy: 'full' | 'tables_only'` (default `'full'`),
  persisted with the VLM config; `'full'` also enables figure descriptions
  under `'auto'`. Legacy configs default to `'full'`.
- Response `features` section: per-capability enabled/disabled state with
  the reason and how to unlock — no more guessing what actually ran.

## [0.5.0] - 2026-07-26

VLM arbitration: the escalation tier for defects beyond geometric reach.
Geometric repair (0.4.0) reads the text layer but cannot express table
SEMANTICS — merged cells, multi-row group headers. A vision model reading a
hi-res page render can. Field-validated with MiniMax-M3 on a real paper's
hyperparameter table (8 columns, scientific-notation values, sub-scripted
headers) before implementation.

### Added

- New module `src/pdf_capture_mcp/quality/vlm_repair.py`:
  - `vlm_repair_table`: locates the source region of a broken table
    (MD-104/105/107) via token matching, renders a 3x zoom crop, asks the
    VLM for an exact HTML `<table>` transcription, and replaces the broken
    markdown block only when the NUMERIC GATE passes.
  - Numeric gate: the VLM may neither invent numbers absent from the
    region's text layer (hallucination) nor drop numbers present in the
    broken block (data loss); torn fragments recombining into decimal
    values ('6'+'0' -> '6.0') are recognized as legitimate.
  - `vlm_describe_figures`: injects a short VLM description quoted under
    each extracted figure image, making figure-embedded content (the
    MD-202 gap) retrievable by text-only RAG. Idempotent.
- `pdf_to_markdown`: `enable_table_enrich` is now fully implemented
  (previously accepted but unused) and gains `enrich_figures: bool = False`.
  Both are opt-in — VLM calls consume API tokens.
- Pipeline phase `vlm_arbitration` after geometric repair; actions appear
  in `qc_report.repairs`, repaired text is re-audited.
- HTML `<table>` output for complex tables — Markdown tables cannot express
  rowspan/colspan/multi-level headers; inline HTML renders everywhere.

## [0.4.3] - 2026-07-26

Field report: every image in converted documents rendered as broken. Root
cause: the marker engine saves images into an `images/` subdirectory but
leaves bare filenames in the markdown links, so relative resolution fails.

### Fixed

- marker engine now rewrites image references to `images/<name>` after
  saving files, so links resolve relative to `full_text.md`.

### Added

- `MD-109` (warn, auto-fix): image-link integrity check — references that
  do not resolve from the markdown's directory are rewritten when the file
  exists under `images/<basename>`; unresolvable references are reported.
  `run_markdown_audit` gains an optional `base_dir` argument; the pipeline
  passes the extraction directory automatically.

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
