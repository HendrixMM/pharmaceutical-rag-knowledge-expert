"""
Validate links in Markdown files.

Features:
- Scans .md files recursively (excluding .git, venv, node_modules).
- Extracts inline and reference-style links.
- Validates internal file links and anchors.
- Optionally validates external http(s) links with timeout and cache.
- Reports broken links with file and line number.

Exit codes:
 0: All links valid
 1: Broken links found
 2: Script error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import requests  # type: ignore
except Exception:
    requests = None  # External checks will be disabled if requests missing


MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
MD_REF_DEF_RE = re.compile(r"^\s*\[([^\]]+)\]:\s*(\S+)\s*$")
MD_REF_USE_RE = re.compile(r"\[([^\]]+)\]\[([^\]]+)\]")
MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(\s*#*\s*)?$")


@dataclass
class LinkIssue:
    file: Path
    line_no: int
    link: str
    reason: str


def is_excluded(path: Path, excludes: list[str]) -> bool:
    parts = set(path.parts)
    if any(p in parts for p in {".git", "venv", "node_modules"}):
        return True
    for pat in excludes:
        if pat and pat in str(path):
            return True
    return False


def find_markdown_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.md") if not is_excluded(p, [])]


def collect_reference_defs(lines: list[str]) -> dict[str, str]:
    defs: dict[str, str] = {}
    for line in lines:
        m = MD_REF_DEF_RE.match(line)
        if m:
            defs[m.group(1).lower()] = m.group(2)
    return defs


def extract_links(lines: list[str]) -> list[tuple[int, str]]:
    links: list[tuple[int, str]] = []
    in_code = False
    for i, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for m in MD_LINK_RE.finditer(line):
            links.append((i, m.group(2)))
        for m in MD_REF_USE_RE.finditer(line):
            # Will be resolved using reference defs
            links.append((i, f"REF:{m.group(2)}"))
    return links


def github_anchor(text: str) -> str:
    anchor = text.strip().lower()
    anchor = re.sub(r"[\s]+", "-", anchor)
    anchor = re.sub(r"[^a-z0-9\-]", "", anchor)
    return anchor


def collect_anchors(md_path: Path) -> set[str]:
    anchors: set[str] = set()
    try:
        content = md_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return anchors
    in_code = False
    for line in content:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        m = MD_HEADING_RE.match(line)
        if m:
            anchors.add(github_anchor(m.group(2)))
    return anchors


def is_http(link: str) -> bool:
    return link.startswith("http://") or link.startswith("https://")


def is_mailto(link: str) -> bool:
    return link.startswith("mailto:")


def check_external(link: str, timeout: float, cache: dict[str, tuple[int, float]]) -> str | None:
    if requests is None:
        return None  # External checks disabled
    now = time.time()
    if link in cache:
        status, ts = cache[link]
        if now - ts < 600:
            return None if 200 <= status < 400 else f"HTTP {status} (cached)"
    try:
        resp = requests.head(link, allow_redirects=True, timeout=timeout)
        status = resp.status_code
        cache[link] = (status, now)
        if 200 <= status < 400:
            return None
        # Some servers block HEAD, try GET minimal
        resp = requests.get(link, allow_redirects=True, timeout=timeout)
        status = resp.status_code
        cache[link] = (status, now)
        return None if 200 <= status < 400 else f"HTTP {status}"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def validate(
    repo_root: Path,
    skip_external: bool,
    timeout: float,
    excludes: list[str],
    verbose: bool,
    explicit_paths: list[Path] | None = None,
) -> tuple[int, list[LinkIssue]]:
    if explicit_paths:
        md_files = [
            p.resolve() for p in explicit_paths if p.suffix.lower() == ".md" and not is_excluded(p.resolve(), excludes)
        ]
    else:
        md_files = [p for p in repo_root.rglob("*.md") if not is_excluded(p, excludes)]
    issues: list[LinkIssue] = []
    external_cache: dict[str, tuple[int, float]] = {}

    for md in md_files:
        try:
            lines = md.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as e:
            issues.append(LinkIssue(md, 1, "(file)", f"Read error: {e}"))
            continue
        ref_defs = collect_reference_defs(lines)
        links = extract_links(lines)
        for ln, link in links:
            if link.startswith("REF:"):
                ref = link[4:].lower()
                if ref not in ref_defs:
                    issues.append(LinkIssue(md, ln, link, "Missing reference definition"))
                    continue
                link = ref_defs[ref]
            # Ignore anchors-only links
            if link.startswith("#"):
                continue
            if is_mailto(link):
                continue
            if is_http(link):
                if skip_external:
                    if verbose:
                        print(f"[skip] {md}:{ln} -> {link}")
                    continue
                reason = check_external(link, timeout, external_cache)
                if reason:
                    issues.append(LinkIssue(md, ln, link, reason))
                elif verbose:
                    print(f"[ok] {md}:{ln} -> {link}")
                continue

            # Internal link: may include anchor
            target, _, frag = link.partition("#")
            target_path = (md.parent / target).resolve()
            if not target:
                continue
            if not target_path.exists():
                issues.append(LinkIssue(md, ln, link, "File not found"))
                continue
            if frag:
                anchors = collect_anchors(target_path)
                if github_anchor(frag) not in anchors:
                    issues.append(LinkIssue(md, ln, link, "Anchor not found"))
            elif verbose:
                print(f"[ok] {md}:{ln} -> {link}")

    return (0 if not issues else 1), issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate links in Markdown documentation.")
    parser.add_argument("--skip-external", action="store_true", help="Skip http/https link validation")
    parser.add_argument("--timeout", type=float, default=5.0, help="Timeout for external checks (seconds)")
    parser.add_argument("--exclude", action="append", default=[], help="Exclude path pattern")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument("paths", nargs="*", help="Specific Markdown files to check (defaults to repository scan)")
    args = parser.parse_args(argv)

    try:
        explicit_paths = [Path(p) for p in args.paths] if args.paths else None
        code, issues = validate(
            Path.cwd(), args.skip_external, args.timeout, args.exclude, args.verbose, explicit_paths
        )
    except Exception as e:
        print(f"Script error: {e}", file=sys.stderr)
        return 2

    if args.format == "json":
        out = [
            {
                "file": str(i.file),
                "line": i.line_no,
                "link": i.link,
                "reason": i.reason,
            }
            for i in issues
        ]
        print(json.dumps({"broken_links": out, "count": len(issues)}, indent=2))
    else:
        if not issues:
            print("No broken links found.")
        else:
            print(f"Broken links: {len(issues)}")
            for i in issues:
                print(f"- {i.file}:{i.line_no} -> {i.link} ({i.reason})")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
