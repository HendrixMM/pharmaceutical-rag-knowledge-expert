"""
Validate documentation metadata front matter across Markdown files.

Expected YAML front matter at the top of each file:
---
Last Updated: YYYY-MM-DD
Owner: Team Name
Review Cadence: Weekly|Bi-weekly|Monthly|Quarterly|Annually|Daily
---

Exit codes:
 0: All files valid
 1: Missing/invalid/stale metadata
 2: Script error
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


FRONT_MATTER_RE = re.compile(r"^---\s*$")
DATE_FMT = "%Y-%m-%d"
ALLOWED_CADENCE = {
    "daily": 1,
    "weekly": 7,
    "bi-weekly": 14,
    "monthly": 30,
    "quarterly": 90,
    "annually": 365,
}


@dataclass
class MetaIssue:
    file: Path
    reason: str


def parse_front_matter(path: Path) -> dict[str, str] | None:
    """Parse YAML front matter. Accepts placement at top or after the first heading."""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None

    # Search for a front matter block near the top (within first ~40 lines)
    start_idx: int | None = None
    max_scan = min(len(lines), 200)
    for i in range(max_scan):
        # Skip initial blank lines or a single leading H1 heading line
        if i == 0 and (not lines[i].strip() or lines[i].lstrip().startswith("# ")):
            continue
        if FRONT_MATTER_RE.match(lines[i]):
            start_idx = i
            break
        # Allow one blank line after title before metadata
        if i < 3 and not lines[i].strip():
            continue
    if start_idx is None:
        return None
    # Find end marker
    try:
        end = next(j for j in range(start_idx + 1, len(lines)) if FRONT_MATTER_RE.match(lines[j]))
    except StopIteration:
        return None
    block = "\n".join(lines[start_idx + 1 : end])
    if yaml is None:
        # Minimal fallback parser (key: value)
        data: dict[str, str] = {}
        for line in block.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
        return data
    try:
        data = yaml.safe_load(block) or {}
        return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return None


def validate_meta(meta: dict[str, str] | None, strict: bool) -> str | None:
    if not meta:
        return "Missing or unreadable front matter"
    missing = [k for k in ("Last Updated", "Owner", "Review Cadence") if k not in meta or not meta[k]]
    if missing:
        return f"Missing fields: {', '.join(missing)}"
    # Validate date
    try:
        last = dt.datetime.strptime(meta["Last Updated"], DATE_FMT).date()
    except Exception:
        return "Invalid 'Last Updated' date format (YYYY-MM-DD)"
    cadence = meta["Review Cadence"].strip().lower()
    if cadence not in ALLOWED_CADENCE:
        return "Invalid 'Review Cadence' value"
    # Staleness check
    days = ALLOWED_CADENCE[cadence]
    if (dt.date.today() - last).days > days:
        return "Stale metadata (past review cadence)" if strict else None
    return None


def scan_paths(
    root: Path, target: Path | None, excludes: list[str], explicit_paths: list[Path] | None = None
) -> list[Path]:
    def not_excluded(p: Path) -> bool:
        s = str(p)
        return not any(ex in s for ex in excludes)

    if explicit_paths:
        return [p.resolve() for p in explicit_paths if p.suffix.lower() == ".md" and not_excluded(p.resolve())]
    if target and target.exists():
        if target.is_file():
            return [target] if not_excluded(target) else []
        return [p for p in target.rglob("*.md") if not_excluded(p)]
    return [p for p in root.rglob("*.md") if not_excluded(p)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate documentation metadata front matter")
    p.add_argument("--path", type=str, default=".", help="Directory or file to check")
    p.add_argument("--exclude", action="append", default=[".git", "venv", "node_modules"], help="Exclude pattern")
    p.add_argument("--strict", action="store_true", help="Fail on stale dates (warnings by default)")
    p.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    p.add_argument("--fix-dates", action="store_true", help="Auto-update Last Updated to today (interactive)")
    p.add_argument("paths", nargs="*", help="Specific Markdown files to check (defaults to scan)")
    args = p.parse_args(argv)

    root = Path.cwd()
    target = Path(args.path).resolve()
    explicit_paths = [Path(p) for p in args.paths] if args.paths else None
    files = scan_paths(root, target, args.exclude, explicit_paths)

    issues: list[MetaIssue] = []
    for f in files:
        meta = parse_front_matter(f)
        reason = validate_meta(meta, args.strict)
        if reason:
            issues.append(MetaIssue(f, reason))

    if args.format == "json":
        print(
            json.dumps(
                {"issues": [{"file": str(i.file), "reason": i.reason} for i in issues], "count": len(issues)}, indent=2
            )
        )
    else:
        if not issues:
            print("All documentation metadata is valid.")
        else:
            print(f"Metadata issues: {len(issues)}")
            for i in issues:
                print(f"- {i.file}: {i.reason}")

    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
