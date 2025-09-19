# Minimal shim to satisfy modules expecting the third-party `regex` package.
# Falls back to Python's built-in `re` implementation.
import re as _re

# Re-export public attributes from the standard library `re` module.
for _name in dir(_re):
    globals()[_name] = getattr(_re, _name)

__all__ = [name for name in globals() if not name.startswith('_')]
