"""Entry point: python -m pdf_capture_mcp"""

from __future__ import annotations

import sys


def main() -> None:
    """Start the PDF Capture MCP server."""
    from pdf_capture_mcp.config import setup_logging
    from pdf_capture_mcp.server import mcp

    setup_logging()

    # Handle setup-mineru subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "setup-mineru":
        from pdf_capture_mcp.engines.mineru_engine import ensure_mineru_env

        try:
            python_exe = ensure_mineru_env()
            print(f"MinerU environment ready: {python_exe}")
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    # Determine transport
    if "--sse" in sys.argv:
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
