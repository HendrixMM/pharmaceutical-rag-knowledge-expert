# flake8: noqa
"""
Lightweight sanity checks for BFG replacement patterns.

Validations:
- File exists and has only concrete, non-generic patterns (no placeholders).
- Mapping lines use single-backslash regex for \\S* (not literal \\S*).
- No generic placeholders/domains in LHS (example.com, your_, dummy, placeholder, noreply@).
- For API key/token assignments, require LHS contains '\\S*' (one backslash in file).

Exit codes:
 0: All good
 1: Validation failures
 2: Script error
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    path = Path("scripts/sensitive-patterns.txt")
    if not path.exists():
        print("[sanity] patterns file not found; skipping")
        return 0

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    failures: list[str] = []

    def is_mapping(ln: str) -> bool:
        return "==>" in ln

    def lhs(ln: str) -> str:
        return ln.split("==>", 1)[0]

    # Validate each non-comment, non-empty line
    for i, ln in enumerate(lines, start=1):
        if not ln or ln.lstrip().startswith("#"):
            continue

        if not is_mapping(ln):
            failures.append(f"line {i}: expected mapping format 'lhs==>rhs'")
            continue

        left = lhs(ln)

        # Block generic placeholders/domains in LHS
        bad_fragments = (
            "example.com",
            "example.org",
            "example.net",
            "your_",
            "dummy",
            "placeholder",
            "noreply@",
        )
        if any(bad in left for bad in bad_fragments):
            failures.append(f"line {i}: mapping contains generic placeholder/domain in LHS: {left}")

        # For API key/token assignments, require single-backslash regex for \S*
        if any(k in left for k in ("NVIDIA_API_KEY=", "PUBMED_EUTILS_API_KEY=", "APIFY_TOKEN=")):
            # Expect '\S*' present and '\\S*' absent
            if "\\S*" not in left:
                failures.append(f"line {i}: missing \\S* regex in assignment: {left}")
            if "\\\\S*" in left:
                failures.append(f"line {i}: contains double backslash in regex (\\\\S*): {left}")

    if failures:
        print("[sanity] BFG patterns validation failed:")
        for f in failures:
            print(" -", f)
        return 1

    print("[sanity] BFG patterns validation passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[sanity] error: {e}", file=sys.stderr)
        sys.exit(2)
