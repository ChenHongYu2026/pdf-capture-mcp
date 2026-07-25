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

- **Dual extraction engines**
  - [marker](https://github.com/datalab-to/marker) (default) — fast, accurate, zero special setup
  - [MinerU](https://github.com/opendatalab/MinerU) (optional) — highest quality for complex layouts, auto-managed in an isolated venv
- **6 MCP tools** — `pdf_to_markdown`, `extract_tables`, `classify_document`, `pdf_info`, `setup_vlm`, `check_environment`
- **Optional VLM enhancement** — plug in any vision-capable model (Qwen-VL, GLM-4V, MiniMax, Moonshot, OpenAI, local Ollama…) for better table/formula extraction
- **Quality gate** — multi-dimensional QC (text completeness, heading structure, formula integrity, table coverage)
- **Guided onboarding** — the server walks your AI agent through VLM setup and environment checks on first use
- **Privacy-first** — API keys are stored locally with `chmod 600` and never echoed back in responses

## Quick Start

### 1. Install

```bash
# Base package (rule-based extraction, ~50MB)
pip install pdf-capture-mcp

# With marker engine (recommended, includes PyTorch)
pip install "pdf-capture-mcp[marker]"

# Everything (marker + TATR table detection + DePlot charts)
pip install "pdf-capture-mcp[all]"
```

### 2. Add to your MCP client

For Qoder / Claude Desktop / Cursor, add to your `mcp.json`:

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

On first use, the agent will guide you through optional VLM setup and verify your environment.

## Tools

| Tool | Description |
|------|-------------|
| `pdf_to_markdown` | Full pipeline: extract → clean → QC → structured Markdown |
| `extract_tables` | Table extraction (pdfplumber rules + optional TATR deep learning) |
| `classify_document` | Document type detection (academic paper, consulting report, …) |
| `pdf_info` | Fast metadata: page count, text layer, scanned detection |
| `setup_vlm` | Configure optional VLM enhancement (any vision-capable provider) |
| `check_environment` | Verify engines and dependencies are ready |

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
| `PDF_CAPTURE_ENGINE` | `auto` | Default engine: `marker` / `mineru` / `auto` |
| `PDF_CAPTURE_VLM_API_KEY` | — | VLM API key (preferred over passing in chat) |
| `PDF_CAPTURE_CACHE_DIR` | `~/.cache/pdf-capture-mcp` | Model & config cache |
| `PDF_CAPTURE_MINERU_VENV` | `<cache>/venv-mineru` | MinerU venv location |
| `PDF_CAPTURE_LOG_LEVEL` | `INFO` | Logging level |
| `MINERU_MODEL_SOURCE` | `modelscope` | MinerU model source: `modelscope` / `huggingface` / `local` |

## Development

```bash
git clone https://github.com/ChenHongYu2026/pdf-capture-mcp.git
cd pdf-capture-mcp
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/pdf_capture_mcp/
```

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

- **双提取引擎**
  - [marker](https://github.com/datalab-to/marker)（默认）—— 快速、准确、零特殊配置
  - [MinerU](https://github.com/opendatalab/MinerU)（可选）—— 复杂版面提取质量最高，自动管理独立虚拟环境
- **6 个 MCP 工具** —— `pdf_to_markdown`、`extract_tables`、`classify_document`、`pdf_info`、`setup_vlm`、`check_environment`
- **可选 VLM 增强** —— 接入任何具备视觉能力的模型（通义千问 Qwen-VL、智谱 GLM-4V、MiniMax、月之暗面 Moonshot、OpenAI、本地 Ollama 等），提升表格/公式提取质量
- **质量门控** —— 多维度 QC 评估（文本完整度、标题结构、公式完好率、表格覆盖率）
- **引导式初始化** —— 首次使用时，服务器会引导 AI 助手完成 VLM 配置和环境检查
- **隐私优先** —— API Key 以 `chmod 600` 权限本地存储，绝不在响应中回显

## 快速开始

### 1. 安装

```bash
# 基础包（规则提取，约 50MB）
pip install pdf-capture-mcp

# 含 marker 引擎（推荐，包含 PyTorch）
pip install "pdf-capture-mcp[marker]"

# 完整安装（marker + TATR 表格检测 + DePlot 图表提取）
pip install "pdf-capture-mcp[all]"
```

### 2. 添加到 MCP 客户端

Qoder / Claude Desktop / Cursor 用户，在 `mcp.json` 中添加：

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

> "把 ~/Downloads/论文.pdf 转成 Markdown"
> "提取这份报告里的所有表格"
> "这个 PDF 是扫描件吗？"

首次使用时，助手会引导你完成可选的 VLM 配置和环境校验。

## 工具列表

| 工具 | 说明 |
|------|------|
| `pdf_to_markdown` | 完整管线：提取 → 清洁 → QC → 结构化 Markdown |
| `extract_tables` | 表格提取（pdfplumber 规则 + 可选 TATR 深度学习） |
| `classify_document` | 文档类型检测（学术论文、咨询报告等） |
| `pdf_info` | 快速元数据：页数、文本层、扫描件检测 |
| `setup_vlm` | 配置可选的 VLM 增强（支持任何具备视觉能力的供应商） |
| `check_environment` | 校验引擎与依赖是否就绪 |

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
| `PDF_CAPTURE_ENGINE` | `auto` | 默认引擎：`marker` / `mineru` / `auto` |
| `PDF_CAPTURE_VLM_API_KEY` | — | VLM API Key（推荐方式，避免对话中传递） |
| `PDF_CAPTURE_CACHE_DIR` | `~/.cache/pdf-capture-mcp` | 模型与配置缓存目录 |
| `PDF_CAPTURE_MINERU_VENV` | `<cache>/venv-mineru` | MinerU 虚拟环境位置 |
| `PDF_CAPTURE_LOG_LEVEL` | `INFO` | 日志级别 |
| `MINERU_MODEL_SOURCE` | `modelscope` | MinerU 模型源：`modelscope` / `huggingface` / `local` |

## 本地开发

```bash
git clone https://github.com/ChenHongYu2026/pdf-capture-mcp.git
cd pdf-capture-mcp
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/pdf_capture_mcp/
```

## 许可证

MIT —— 详见 [LICENSE](LICENSE)。

第三方声明：MinerU（AGPL-3.0，通过独立子进程调用，未链接）、
marker（Apache-2.0）、pdfplumber（MIT）、Table Transformer（MIT）、pymupdf4llm（Apache-2.0）。
