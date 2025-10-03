#!/usr/bin/env python
"""
Minimal CLI to run pharmaceutical capability checks via EnhancedNeMoClient.

Prints a JSON report including endpoint usage and overall status. Exits with
code 0 on success/partial, 1 on failure or unexpected errors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pharmaceutical capability checks")
    parser.add_argument("--api-key", dest="api_key", default=None, help="Override NVIDIA API key (optional)")
    parser.add_argument(
        "--fallback",
        dest="fallback",
        default=None,
        help="Enable/disable fallback (true/false). Defaults to config behavior.",
    )
    parser.add_argument(
        "--pretty",
        dest="pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args(argv)

    # Lazy import to avoid startup cost when running other tools
    try:
        from src.clients.nemo_client_enhanced import EnhancedNeMoClient
        from src.enhanced_config import EnhancedRAGConfig
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"Failed to import components: {e}\n")
        return 1

    cfg = EnhancedRAGConfig.from_env()

    enable_fallback = _bool(args.fallback) if args.fallback is not None else True
    api_key = args.api_key or os.getenv("NVIDIA_API_KEY")

    try:
        client = EnhancedNeMoClient(
            config=cfg,
            enable_fallback=enable_fallback,
            pharmaceutical_optimized=True,
            api_key=api_key,
        )
        report: dict[str, Any] = client.test_pharmaceutical_capabilities()
        if args.pretty:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(json.dumps(report, separators=(",", ":")))
        status = str(report.get("overall_status", "failed")).lower()
        return 0 if status in {"success", "partial"} else 1
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"Benchmark run failed: {e}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
