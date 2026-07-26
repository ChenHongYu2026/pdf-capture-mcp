<a id="english"></a>

# pdf-capture-mcp

**English** | [中文](#chinese)

Multi-phase PDF capture pipeline as an [MCP](https://modelcontextprotocol.io) server.
Convert PDF documents into high-quality structured Markdown — with formula recognition,
table extraction, layout cleaning, and a built-in quality gate.

[![CI](https://github.com/ChenHongYu2026/pdf-capture-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/ChenHongYu2026/pdf-capture-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Triple extraction engines**
  - [pymupdf4llm](https://github.com/pymupdf/pymupdf4llm) (built-in) — zero setup, fast, always available
  - [marker](https://github.com/datalab-to/marker) (recommended) — highest quality for complex layouts
  - [MinerU](https://github.com/opendatalab/MinerU) (optional) — best for multi-column/InDesign PDFs, auto-managed in an isolated venv
- **9 MCP tools** — `pdf_to_markdown`, `get_job_status`, `download_models`, `extract_tables`, `classify_document`, `pdf_info`, `setup_vlm`, `check_environment`, `install_engine`
- **Async job mode** — large PDFs convert in a background job (no MCP client timeouts); model downloads can be pre-fetched without time limits
- **Optional VLM enhancement** — plug in any vision-capable model (Qwen-VL, GLM-4V, MiniMax, Moonshot, OpenAI, local Ollama…) for better table/formula extraction. **No extra dependencies needed** — works out of the box with the base install.
- **Quality gate** — multi-dimensional QC (text completeness, heading structure, formula integrity, table coverage) **plus content-aware audit rules** that catch defects statistical checks miss (control chars, torn numeric columns, fused table headers, content loss)
- **Progressive setup** — works immediately with zero config; enhance with marker/VLM on demand
- **Privacy-first** — API keys are stored locally with `chmod 600` and never echoed back in responses

## Quick Start

### 1. Install

```bash
# Base package (pymupdf engine + VLM support, ~80MB) — works immediately
pip install pdf-capture-mcp

# With marker engine (recommended for complex PDFs, includes PyTorch, ~2.5GB)
pip install "pdf-capture-mcp[marker]"

# Everything (marker + TATR table detection + DePlot charts, ~3GB)
pip install "pdf-capture-mcp[all]"
```

### 2. Add to your MCP client

For Qoder / Claude Desktop / Cursor, add to your `mcp.json`:

```json
{
  "mcpServers": {
    "pdf-capture": {
      "command": "uvx",
      "args": ["pdf-capture-mcp"]
    }
  }
}
```

With marker engine (recommended):

```json
{
  "mcpServers": {
    "pdf-capture": {
      "command": "uvx",
      "args": ["pdf-capture-mcp[marker]"]
    }
  }
}
```

Or if installed via pip:

```json
{
  "mcpServers": {
    "pdf-capture": {
      "command": "pdf-capture-mcp"
    }
  }
}
```

### 3. Use it

Ask your AI agent things like:

> "Convert ~/Downloads/paper.pdf to Markdown"
> "Extract all tables from this report"
> "Is this PDF a scanned document?"

The server works immediately with the built-in pymupdf engine. For higher quality on
complex layouts, the agent can install marker on demand (or you can pre-install it).

## Tools

| Tool | Description |
|------|-------------|
| `pdf_to_markdown` | Full pipeline: extract → clean → QC → structured Markdown (async for large PDFs) |
| `get_job_status` | Poll background jobs (large conversions / model downloads) |
| `download_models` | Pre-download marker models (recommended on slow networks) |
| `extract_tables` | Table extraction (pdfplumber rules + optional TATR deep learning) |
| `classify_document` | Document type detection (academic paper, consulting report, …) |
| `pdf_info` | Fast metadata: page count, text layer, scanned detection |
| `setup_vlm` | Configure optional VLM enhancement (any vision-capable provider) |
| `check_environment` | Verify engines, dependencies, model cache, and network config |
| `install_engine` | Install marker/ml engines on behalf of the user |

## Large PDFs & Timeouts

Converting a big document (e.g. a 75-page paper) can take longer than most MCP
client timeouts. `pdf_to_markdown` handles this automatically:

- **`mode="auto"`** (default): PDFs ≤ 15 pages return inline; larger PDFs start a
  **background job** and immediately return a `job_id` with an ETA — no client timeout.
- **`mode="async"`**: always return a `job_id`. **`mode="sync"`**: always inline
  (previous behavior; may time out on large files).
- Poll with `get_job_status(job_id)` — it reports the current stage
  (`classify → extracting → table_extraction → qc → done`) and, when finished,
  the `markdown_path` plus a content preview.
- The result is always written to `<out_dir>/extraction/full_text.md`, so even if
  a client disconnects, nothing is lost.
- Need a fast preview? Pass `page_range="0-9"` to convert only the first pages.

## Slow / Restricted Networks (e.g. mainland China)

The marker engine downloads ~2GB of models on first use — **inside a 300s startup
window**. On slow networks this fails with an opaque timeout. Avoid it:

1. **Pre-download models first** (no time limit, runs as a background job):
   ask your agent to run `download_models`, then poll `get_job_status`.
2. **huggingface.co unreachable?** Use a mirror — the Xet-incompatibility
   workaround (`HF_HUB_DISABLE_XET=1`) is applied automatically:
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. **Using an HTTP proxy?** Localhost must bypass it, or internal inference
   health checks fail. The server enforces `NO_PROXY=localhost,127.0.0.1`
   automatically at startup — but check your client config if you override env.
   Note: some setups exclude `huggingface.co` from the proxy via `NO_PROXY`;
   remove that entry if you want HF downloads to go through the proxy.
4. **All models cached?** Go fully offline for reliable startups:
   ```bash
   export HF_HUB_OFFLINE=1
   ```

`check_environment` reports per-model cache status (`models_ready`) and the
current network configuration, so your agent can diagnose this in one call.

## Quality Audit Rules

Every `pdf_to_markdown` run finishes with a two-layer quality check whose
results are returned in `qc_report` (verdict, dimension scores, issues, fixes):

1. **Statistical gate** — text completeness (chars/page), heading structure,
   formula integrity, table coverage. Catches gross failures.
2. **Content-aware audit** — rules born from a real 75-page paper audit where
   every actual defect passed the statistical gate unnoticed:

| Rule | Detects | Severity | Auto-fix |
|------|---------|----------|----------|
| `MD-101` | Garbled chars (U+FFFD, private-use area) | critical | — |
| `MD-102` | C0 control chars at in-cell word wraps (e.g. `En\x02lightenment`) | critical | ✅ removed, words rejoined |
| `MD-103` | Table with an all-empty header row (misread multi-column layout) | warn | — |
| `MD-104` | Numeric column tearing — scientific notation split across cells (`6 \| 0 \| × 10 \| − 4`, decimal point lost) | critical | — |
| `MD-105` | Table header fused with the first data row (header cells contain standalone numbers) | critical | — |
| `MD-106` | Empty `<span></span>` placeholder cells | info | ✅ removed |
| `MD-201` | Content loss — token-multiset comparison against the PDF text layer (pymupdf, independent of the engine's layout analysis); missing-token examples included | info/warn/critical by ratio | — |

**Auto-fix policy**: only deterministic, information-preserving fixes are
applied automatically (the sanitized markdown is written back to
`full_text.md`). Structural defects (`MD-103/104/105`) are located precisely
but never rewritten — automated guessing could corrupt values further.
Recommended remediation, in order:

1. Cross-check the affected region with `extract_tables` (pdfplumber — an
   independent extraction channel that bypasses layout analysis).
2. Re-run with VLM table enrichment (`setup_vlm`, then
   `enable_table_enrich=True`).
3. Any `critical` finding escalates a `PASS` verdict to `WARN`, so agents
   know to inspect `qc_report.audit_issues` before trusting the output.

## VLM Enhancement (Optional)

VLM re-extracts complex tables and broken formulas from page images.
Works with **any provider whose model supports image input**:

| Provider | Example model | API base |
|----------|--------------|----------|
| Alibaba Qwen | `qwen-vl-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| Zhipu AI | `glm-4v` | `https://open.bigmodel.cn/api/paas/v4` |
| MiniMax | `minimax-m3` | `https://api.minimaxi.com/v1` |
| Moonshot | `moonshot-v1-vision` | `https://api.moonshot.cn/v1` |
| OpenAI | `gpt-4o` | `https://api.openai.com/v1` |
| Ollama (local) | `llama3.2-vision` | `http://localhost:11434/v1` |

Set your key via environment variable (recommended — never typed into chat):

```bash
export PDF_CAPTURE_VLM_API_KEY=your_key_here
```

Then tell your agent: *"Enable VLM with qwen-vl-max"* — it validates vision capability
before saving. **Note: using VLM consumes your API tokens.**

## MinerU Engine (Optional)

For the highest extraction quality on complex layouts (multi-column, InDesign PDFs):

```bash
pdf-capture-mcp setup-mineru   # requires Python 3.11 on PATH
```

Models (~2GB) auto-download from ModelScope on first extraction.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_CAPTURE_ENGINE` | `auto` | Default engine: `marker` / `mineru` / `pymupdf` / `auto` |
| `PDF_CAPTURE_VLM_API_KEY` | — | VLM API key (preferred over passing in chat) |
| `PDF_CAPTURE_CACHE_DIR` | `~/.cache/pdf-capture-mcp` | Model & config cache (also stores job state) |
| `PDF_CAPTURE_MINERU_VENV` | `<cache>/venv-mineru` | MinerU venv location |
| `PDF_CAPTURE_LOG_LEVEL` | `INFO` | Logging level |
| `MINERU_MODEL_SOURCE` | `modelscope` | MinerU model source: `modelscope` / `huggingface` / `local` |
| `HF_ENDPOINT` | huggingface.co | HuggingFace mirror for model downloads (e.g. `https://hf-mirror.com`) |
| `HF_HUB_DISABLE_XET` | — | Set `1` when using a mirror (auto-set by `download_models`) |
| `HF_HUB_OFFLINE` | — | Set `1` after all models are cached for fully offline runs |
| `NO_PROXY` | — | Must include `localhost,127.0.0.1` when a proxy is set (auto-enforced) |

## Development

```bash
git clone https://github.com/ChenHongYu2026/pdf-capture-mcp.git
cd pdf-capture-mcp
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/pdf_capture_mcp/
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Operation not supported` during install | External/exFAT drive: `export UV_LINK_MODE=copy` then retry |
| `uvx: command not found` | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| MCP server not appearing in tools | Restart your MCP client; check `mcp.json` syntax |
| MCP call times out on a large PDF | Expected with `mode="sync"` — use the default `mode="auto"` and poll `get_job_status`; the result is still written to `out_dir` |
| First conversion fails with `fast_layout/ocr_error server failed to become healthy` | Model download exceeded the 300s startup window — run `download_models` first |
| Downloads stall on huggingface.co | Set `HF_ENDPOINT=https://hf-mirror.com` (see Slow Networks section) |
| All health checks fail behind a proxy | Ensure `NO_PROXY` includes `localhost,127.0.0.1` (auto-enforced at startup) |
| marker engine slow on first run | Downloads ~2GB models on first use; run `download_models` ahead of time |
| Python version too low | Requires 3.11+: `uv python install 3.11` |

## License

MIT — see [LICENSE](LICENSE).

Third-party notices: MinerU (AGPL-3.0, invoked as a separate subprocess, not linked),
marker (Apache-2.0), pdfplumber (MIT), Table Transformer (MIT), pymupdf4llm (Apache-2.0).

---

<a id="chinese"></a>

# pdf-capture-mcp（中文）

[English](#english) | **中文**

多阶段 PDF 捕获管线，以 [MCP](https://modelcontextprotocol.io) 服务器形式提供。
将 PDF 文档转换为高质量结构化 Markdown —— 支持公式识别、表格提取、版面清洁和内置质量门控。

## 功能特性

- **三提取引擎**
  - [pymupdf4llm](https://github.com/pymupdf/pymupdf4llm)（内置）—— 零配置、快速、始终可用
  - [marker](https://github.com/datalab-to/marker)（推荐）—— 复杂版面提取质量最高
  - [MinerU](https://github.com/opendatalab/MinerU)（可选）—— 多栏/InDesign 排版最佳，自动管理独立虚拟环境
- **9 个 MCP 工具** —— `pdf_to_markdown`、`get_job_status`、`download_models`、`extract_tables`、`classify_document`、`pdf_info`、`setup_vlm`、`check_environment`、`install_engine`
- **异步任务模式** —— 大型 PDF 在后台任务中转换（不再触发 MCP 客户端超时）；模型可提前预下载，不受时间窗口限制
- **可选 VLM 增强** —— 接入任何具备视觉能力的模型（通义千问 Qwen-VL、智谱 GLM-4V、MiniMax、月之暗面 Moonshot、OpenAI、本地 Ollama 等），提升表格/公式提取质量。**无需额外依赖**，基础安装即可使用。
- **质量门控** —— 多维度 QC 评估（文本完整度、标题结构、公式完好率、表格覆盖率），**另含内容感知审计规则**，捕获统计指标无法发现的缺陷（控制字符、数值列撕裂、表头融合、内容丢失）
- **渐进式配置** —— 零配置即可工作；按需增强 marker/VLM
- **隐私优先** —— API Key 以 `chmod 600` 权限本地存储，绝不在响应中回显

## 快速开始

### 1. 安装

```bash
# 基础包（pymupdf 引擎 + VLM 支持，约 80MB）—— 安装即可用
pip install pdf-capture-mcp

# 含 marker 引擎（推荐复杂 PDF，包含 PyTorch，约 2.5GB）
pip install "pdf-capture-mcp[marker]"

# 完整安装（marker + TATR 表格检测 + DePlot 图表提取，约 3GB）
pip install "pdf-capture-mcp[all]"
```

### 2. 添加到 MCP 客户端

Qoder / Claude Desktop / Cursor 用户，在 `mcp.json` 中添加：

```json
{
  "mcpServers": {
    "pdf-capture": {
      "command": "uvx",
      "args": ["pdf-capture-mcp"]
    }
  }
}
```

含 marker 引擎（推荐）：

```json
{
  "mcpServers": {
    "pdf-capture": {
      "command": "uvx",
      "args": ["pdf-capture-mcp[marker]"]
    }
  }
}
```

如果已通过 pip 安装：

```json
{
  "mcpServers": {
    "pdf-capture": {
      "command": "pdf-capture-mcp"
    }
  }
}
```

### 3. 开始使用

直接对你的 AI 助手说：

> “把 ~/Downloads/论文.pdf 转成 Markdown”
> “提取这份报告里的所有表格”
> “这个 PDF 是扫描件吗？”

服务器使用内置 pymupdf 引擎即可立即工作。对于复杂版面，助手可按需安装 marker 引擎。

## 工具列表

| 工具 | 说明 |
|------|------|
| `pdf_to_markdown` | 完整管线：提取 → 清洁 → QC → 结构化 Markdown（大文件自动异步） |
| `get_job_status` | 轮询后台任务（大文件转换 / 模型下载） |
| `download_models` | 预下载 marker 模型（慢速网络强烈推荐） |
| `extract_tables` | 表格提取（pdfplumber 规则 + 可选 TATR 深度学习） |
| `classify_document` | 文档类型检测（学术论文、咨询报告等） |
| `pdf_info` | 快速元数据：页数、文本层、扫描件检测 |
| `setup_vlm` | 配置可选的 VLM 增强（支持任何具备视觉能力的供应商） |
| `check_environment` | 校验引擎、依赖、模型缓存与网络配置 |
| `install_engine` | 代用户安装 marker/ml 引擎 |

## 大文件与超时

转换大型文档（如 75 页论文）的耗时往往超过 MCP 客户端超时限制。
`pdf_to_markdown` 会自动处理：

- **`mode="auto"`**（默认）：≤ 15 页直接返回结果；更大的 PDF 自动转为**后台任务**，立即返回 `job_id` 和预估耗时 —— 不再触发客户端超时。
- **`mode="async"`**：总是返回 `job_id`。**`mode="sync"`**：保持旧版同步行为（大文件可能超时）。
- 用 `get_job_status(job_id)` 轮询进度，可看到当前阶段（`classify → extracting → table_extraction → qc → done`）；完成后返回 `markdown_path` 和内容预览。
- 结果始终写入 `<out_dir>/extraction/full_text.md`，即使客户端断开也不丢失。
- 需要快速预览？传 `page_range="0-9"` 只转换前几页。

## 慢速 / 受限网络（如中国大陆）

marker 引擎首次使用时会在 **300 秒启动窗口内**下载约 2GB 模型 ——
慢速网络下必然超时失败。规避方法：

1. **先预下载模型**（无时间限制，后台任务运行）：让助手调用 `download_models`，再用 `get_job_status` 轮询。
2. **连不上 huggingface.co？** 使用镜像站（Xet 协议兼容问题会自动处理，即自动设置 `HF_HUB_DISABLE_XET=1`）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
3. **使用 HTTP 代理？** localhost 必须绕过代理，否则内部推理服务的健康检查会被代理劫持而失败。服务启动时会自动确保 `NO_PROXY` 包含 `localhost,127.0.0.1`。另注意：若你的环境把 `huggingface.co` 加入了 `NO_PROXY`（即 HF 不走代理），想让 HF 下载走代理时需移除该条目。
4. **模型全部缓存完成后**，建议开启完全离线模式，启动更稳定：
   ```bash
   export HF_HUB_OFFLINE=1
   ```

`check_environment` 会逐一报告模型缓存状态（`models_ready`）和当前网络配置，助手一次调用即可完成诊断。

## 质量审计规则

每次 `pdf_to_markdown` 运行结束时都会执行双层质量检查，结果在 `qc_report`
中返回（结论、维度分数、问题清单、已修复项）：

1. **统计门控** —— 文本完整度（字符/页）、标题结构、公式完好率、表格覆盖率，捕获粗粒度失败。
2. **内容感知审计** —— 源自一次真实的 75 页论文审计：当时所有实际缺陷都骗过了统计门控：

| 规则 | 检测内容 | 严重度 | 自动修复 |
|------|---------|--------|----------|
| `MD-101` | 乱码字符（U+FFFD、私有区字符） | critical | — |
| `MD-102` | 单元格内换行处的 C0 控制字符（如 `En\x02lightenment`） | critical | ✅ 移除并拼回断词 |
| `MD-103` | 全空表头行（多栏版式被误识为表格） | warn | — |
| `MD-104` | 数值列撕裂 —— 科学计数法被拆进多个单元格（`6 \| 0 \| × 10 \| − 4`，小数点丢失） | critical | — |
| `MD-105` | 表头与首行数据融合（表头单元格含独立数字） | critical | — |
| `MD-106` | 空 `<span></span>` 占位单元格 | info | ✅ 移除 |
| `MD-201` | 内容丢失 —— 与 PDF 文本层（pymupdf，独立于引擎版面分析的通道）做 token 多重集比对，附缺失 token 样例 | 按比例 info/warn/critical | — |

**自动修复策略**：仅自动应用确定性、信息无损的修复（修复后的 markdown 会回写
`full_text.md`）。结构性缺陷（`MD-103/104/105`）只做精确定位、绝不自动改写 ——
自动猜测可能进一步破坏数值。推荐的补救顺序：

1. 用 `extract_tables`（pdfplumber —— 绕过版面分析的独立提取通道）交叉校验受影响区域；
2. 开启 VLM 表格增强重新转换（`setup_vlm` + `enable_table_enrich=True`）；
3. 任何 `critical` 发现都会把 `PASS` 升级为 `WARN`，Agent 应先检查
   `qc_report.audit_issues` 再信任输出。

## VLM 增强（可选）

VLM 会从页面图像中重新提取复杂表格和损坏的公式。
支持**任何模型具备图片输入能力的供应商**：

| 供应商 | 示例模型 | API 地址 |
|--------|----------|----------|
| 阿里通义千问 | `qwen-vl-max` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 智谱 AI | `glm-4v` | `https://open.bigmodel.cn/api/paas/v4` |
| MiniMax | `minimax-m3` | `https://api.minimaxi.com/v1` |
| 月之暗面 | `moonshot-v1-vision` | `https://api.moonshot.cn/v1` |
| OpenAI | `gpt-4o` | `https://api.openai.com/v1` |
| Ollama（本地） | `llama3.2-vision` | `http://localhost:11434/v1` |

推荐通过环境变量设置 Key（避免在对话中输入）：

```bash
export PDF_CAPTURE_VLM_API_KEY=你的密钥
```

然后告诉助手："启用 VLM，用 qwen-vl-max" —— 系统会先验证模型的视觉能力再保存配置。
**注意：使用 VLM 功能会消耗你的 Token。**

## MinerU 引擎（可选）

针对复杂版面（多栏、InDesign 排版 PDF）获得最高提取质量：

```bash
pdf-capture-mcp setup-mineru   # 需要 PATH 中有 Python 3.11
```

首次提取时会从 ModelScope 自动下载模型（约 2GB）。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PDF_CAPTURE_ENGINE` | `auto` | 默认引擎：`marker` / `mineru` / `pymupdf` / `auto` |
| `PDF_CAPTURE_VLM_API_KEY` | — | VLM API Key（推荐方式，避免对话中传递） |
| `PDF_CAPTURE_CACHE_DIR` | `~/.cache/pdf-capture-mcp` | 模型与配置缓存目录（同时存储任务状态） |
| `PDF_CAPTURE_MINERU_VENV` | `<cache>/venv-mineru` | MinerU 虚拟环境位置 |
| `PDF_CAPTURE_LOG_LEVEL` | `INFO` | 日志级别 |
| `MINERU_MODEL_SOURCE` | `modelscope` | MinerU 模型源：`modelscope` / `huggingface` / `local` |
| `HF_ENDPOINT` | huggingface.co | HuggingFace 镜像站（如 `https://hf-mirror.com`） |
| `HF_HUB_DISABLE_XET` | — | 使用镜像站时设为 `1`（`download_models` 会自动设置） |
| `HF_HUB_OFFLINE` | — | 模型全部缓存后设为 `1`，完全离线运行 |
| `NO_PROXY` | — | 设置代理时必须包含 `localhost,127.0.0.1`（启动时自动保障） |

## 本地开发

```bash
git clone https://github.com/ChenHongYu2026/pdf-capture-mcp.git
cd pdf-capture-mcp
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/pdf_capture_mcp/
```

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| 安装时报 `Operation not supported` | 外置/exFAT 磁盘：`export UV_LINK_MODE=copy` 后重试 |
| `uvx: command not found` | 安装 uv：`curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| MCP 服务器未出现在工具列表 | 重启 MCP 客户端；检查 `mcp.json` 格式 |
| 大 PDF 转换时 MCP 调用超时 | `mode="sync"` 下属预期行为 —— 使用默认 `mode="auto"` 并轮询 `get_job_status`；结果仍会写入 `out_dir` |
| 首次转换报 `fast_layout/ocr_error server failed to become healthy` | 模型下载超出 300 秒启动窗口 —— 先运行 `download_models` |
| huggingface.co 下载卡住 | 设置 `HF_ENDPOINT=https://hf-mirror.com`（见“慢速网络”一节） |
| 挂代理后所有健康检查失败 | 确保 `NO_PROXY` 包含 `localhost,127.0.0.1`（启动时自动保障） |
| marker 引擎首次运行慢 | 首次使用需下载约 2GB 模型，建议提前运行 `download_models` |
| Python 版本过低 | 需要 3.11+：`uv python install 3.11` |

## 许可证

MIT —— 详见 [LICENSE](LICENSE)。

第三方声明：MinerU（AGPL-3.0，通过独立子进程调用，未链接）、
marker（Apache-2.0）、pdfplumber（MIT）、Table Transformer（MIT）、pymupdf4llm（Apache-2.0）。
