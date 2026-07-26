# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
