"""
Colab Persistence Utilities

Lightweight helpers to make Google Colab workflows smoother:
- Auto-mount Google Drive only when available and not already mounted
- Save and restore minimal artifacts (.env, cache directory, run summaries)
- Wipe cache on demand for clean benchmarking

All functions are safe to import in non-Colab environments (no-ops).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def is_colab() -> bool:
    try:
        import google.colab  # type: ignore
        return True
    except Exception:
        return False


def ensure_drive_mounted(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive if running in Colab and not yet mounted.

    Returns True when Drive is mounted (or not needed), False on failure.
    """
    if not is_colab():
        return True
    try:
        mp = Path(mount_point)
        if mp.exists() and any(mp.iterdir()):
            return True
        from google.colab import drive  # type: ignore

        drive.mount(mount_point, force_remount=False)
        return True
    except Exception:
        return False


def save_env(dest_dir: str) -> bool:
    """Copy .env to dest_dir for persistence (Colab or local)."""
    try:
        src = Path(".env")
        if not src.exists():
            return False
        target_root = Path(dest_dir)
        target_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target_root / ".env")
        return True
    except Exception:
        return False


def restore_env(src_dir: str) -> bool:
    """Restore .env from src_dir into current working directory."""
    try:
        src = Path(src_dir) / ".env"
        if not src.exists():
            return False
        shutil.copy2(src, Path(".env"))
        return True
    except Exception:
        return False


def persist_cache(src_cache_dir: str = "query_cache", dest_dir: str = "./persist") -> bool:
    """Copy cache directory to dest_dir/query_cache for persistence."""
    try:
        src = Path(src_cache_dir)
        if not src.exists():
            return False
        target = Path(dest_dir) / src.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(src, target)
        return True
    except Exception:
        return False


def wipe_cache(cache_dir: str = "query_cache") -> bool:
    """Delete cache directory safely (useful for clean benchmarks)."""
    try:
        path = Path(cache_dir)
        if path.exists():
            shutil.rmtree(path)
        return True
    except Exception:
        return False


def save_run_summary(summary: Dict[str, Any], dest_dir: str = "./persist", filename: Optional[str] = None) -> Optional[Path]:
    """Persist a JSON summary of the last run.

    Returns the path to the saved file or None on failure.
    """
    try:
        import json
        from datetime import datetime

        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        if not filename:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = f"run_summary_{ts}.json"
        out_path = Path(dest_dir) / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return out_path
    except Exception:
        return None

