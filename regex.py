"""Compatibility shim that prefers the third-party ``regex`` package when available."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_third_party_regex() -> object | None:
    """Attempt to import the real ``regex`` package, temporarily hiding this shim."""

    current_dir = Path(__file__).resolve().parent

    original_module = sys.modules.pop("regex", None)
    original_path = list(sys.path)

    try:
        sys.path = [p for p in sys.path if Path(p).resolve() != current_dir]
        return importlib.import_module("regex")
    except Exception:
        return None
    finally:
        sys.path = original_path
        if original_module is not None:
            sys.modules["regex"] = original_module


_regex_module = _load_third_party_regex()
if _regex_module is None:
    import re as _regex_module


for _name in dir(_regex_module):
    if _name.startswith("__") and _name not in {"__all__", "__doc__", "__name__", "__package__", "__path__"}:
        continue
    globals()[_name] = getattr(_regex_module, _name)

if hasattr(_regex_module, "__all__"):
    __all__ = list(_regex_module.__all__)  # type: ignore[attr-defined]
else:
    __all__ = [name for name in globals() if not name.startswith("_")]
