"""
Generate or update a Table of Contents for a Markdown file.

Looks for markers <!-- TOC --> ... <!-- /TOC --> and replaces content.
If no markers are present, inserts TOC after the first heading by default.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(\s*#*\s*)?$")


def in_code_block(line: str, state: dict[str, bool]) -> bool:
    if line.strip().startswith("```"):
        state["code"] = not state.get("code", False)
    return state.get("code", False)


def github_anchor(text: str) -> str:
    anchor = text.strip().lower()
    anchor = re.sub(r"[\s]+", "-", anchor)
    anchor = re.sub(r"[^a-z0-9\-]", "", anchor)
    return anchor


def collect_headings(lines: list[str], min_depth: int, max_depth: int) -> list[tuple[int, int, str]]:
    """Return list of (line_no, depth, text)."""
    out: list[tuple[int, int, str]] = []
    state: dict[str, bool] = {"code": False}
    for idx, line in enumerate(lines, start=1):
        if in_code_block(line, state):
            continue
        m = HEADING_RE.match(line)
        if not m:
            continue
        depth = len(m.group(1))
        if depth < min_depth or depth > max_depth:
            continue
        text = m.group(2).strip()
        out.append((idx, depth, text))
    return out


def build_toc(items: list[tuple[int, int, str]], no_links: bool, min_depth: int) -> list[str]:
    lines: list[str] = []
    seen: dict[str, int] = {}
    for _ln, depth, text in items:
        indent = "  " * (depth - min_depth)
        if no_links:
            lines.append(f"{indent}- {text}")
        else:
            base = github_anchor(text)
            count = seen.get(base, 0)
            seen[base] = count + 1
            anchor = base if count == 0 else f"{base}-{count}"
            lines.append(f"{indent}- [{text}](#{anchor})")
    return lines


def find_marker_indices(lines: list[str], marker_start: str, marker_end: str) -> tuple[int | None, int | None]:
    start_idx: int | None = None
    end_idx: int | None = None
    for i, line in enumerate(lines):
        if start_idx is None and marker_start in line:
            start_idx = i
        if marker_end in line:
            end_idx = i
            break
    return start_idx, end_idx


def insert_or_replace_toc(lines: list[str], toc_lines: list[str], marker_start: str, marker_end: str) -> list[str]:
    start_idx, end_idx = find_marker_indices(lines, marker_start, marker_end)
    if start_idx is not None and end_idx is not None and start_idx < end_idx:
        return lines[: start_idx + 1] + toc_lines + lines[end_idx:]

    # Insert after first heading
    for i, line in enumerate(lines):
        if HEADING_RE.match(line):
            return lines[: i + 1] + ["", marker_start] + toc_lines + [marker_end, ""] + lines[i + 1 :]
    # Else, prepend
    return [marker_start] + toc_lines + [marker_end, ""] + lines


def process_file(
    path: Path,
    min_depth: int,
    max_depth: int,
    marker_start: str,
    marker_end: str,
    no_links: bool,
    in_place: bool,
    check_only: bool,
    quiet: bool,
) -> int:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return 1
    lines = text.splitlines()
    has_markers = all(idx is not None for idx in find_marker_indices(lines, marker_start, marker_end))
    headings = collect_headings(lines, min_depth, max_depth)
    toc_lines = build_toc(headings, no_links, min_depth)
    if not toc_lines:
        print(f"No headings found in {path}")
        return 0
    new_lines = insert_or_replace_toc(lines, toc_lines, marker_start, marker_end)
    output = "\n".join(new_lines) + ("\n" if text.endswith("\n") else "")
    if check_only:
        if not has_markers:
            if not quiet:
                print(f"[skip] {path} (no TOC markers)")
            return 0
        changed = output != text
        if changed:
            if not quiet:
                print(f"[drift] {path} TOC is out of date")
            return 1
        if not quiet:
            print(f"[ok] {path} TOC up to date")
        return 0
    if in_place:
        try:
            path.write_text(output, encoding="utf-8")
        except Exception as e:
            print(f"Error writing {path}: {e}", file=sys.stderr)
            return 1
        return 0
    else:
        print(output)
        return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate or update a Markdown TOC.")
    p.add_argument("file", metavar="FILE", help="Markdown file to process")
    p.add_argument("--max-depth", type=int, default=3, help="Max heading level to include")
    p.add_argument("--min-depth", type=int, default=2, help="Min heading level to include (skip h1)")
    p.add_argument("--marker-start", default="<!-- TOC -->", help="Custom start marker")
    p.add_argument("--marker-end", default="<!-- /TOC -->", help="Custom end marker")
    p.add_argument("--no-links", action="store_true", help="Generate TOC without links")
    p.add_argument("--dry-run", action="store_true", help="Show output without writing")
    p.add_argument("--in-place", action="store_true", help="Modify file in place")
    p.add_argument("--check", action="store_true", help="Check if TOC is up to date (non-writing)")
    p.add_argument("--quiet", action="store_true", help="Reduce output verbosity in check mode")
    args = p.parse_args(argv)

    return process_file(
        Path(args.file),
        args.min_depth,
        args.max_depth,
        args.marker_start,
        args.marker_end,
        args.no_links,
        args.in_place and not args.dry_run,
        args.check,
        args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
