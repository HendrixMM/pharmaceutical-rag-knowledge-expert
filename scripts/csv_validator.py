"""
CSV Versioning & Validation Utility

Features (Phase 7):
- Create manifest JSON alongside CSVs (version, checksum, preview rows, date)
- Validate CSV headers, encoding, and sample row shapes
- Print manifest info on startup

Usage:
  python -m scripts.csv_validator --file Data/my_overlay.csv
  python -m scripts.csv_validator --dir Data/overlays
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _preview_rows(path: Path, limit: int = 3) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append({k: (v[:120] if isinstance(v, str) else v) for k, v in row.items()})
            if i + 1 >= limit:
                break
    return rows


def create_manifest(csv_path: Path) -> Path:
    manifest = {
        "version": 1,
        "created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filename": csv_path.name,
        "checksum": _checksum(csv_path),
        "preview": _preview_rows(csv_path),
    }
    out = csv_path.with_suffix(csv_path.suffix + ".manifest.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out


def validate_csv(csv_path: Path) -> List[str]:
    errs: List[str] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            if not headers or any(h is None or str(h).strip() == "" for h in headers):
                errs.append("Missing or empty headers")
            # Sample a couple rows to check shape
            row = next(reader, None)
            if row is not None and len(row) != len(headers):
                errs.append("Row length does not match headers length (sample)")
    except UnicodeDecodeError:
        errs.append("File not UTF-8 encoded")
    except Exception as exc:
        errs.append(f"Error reading CSV: {exc}")
    return errs


def _process_file(path: Path, create: bool) -> None:
    print(f"Validating: {path}")
    errs = validate_csv(path)
    if errs:
        for e in errs:
            print(f" - {e}")
    else:
        print(" - OK")
    if create:
        man = create_manifest(path)
        print(f"Manifest: {man.name}")


def main() -> int:
    p = argparse.ArgumentParser(description="CSV Versioning & Validation")
    p.add_argument("--file", help="CSV file to validate")
    p.add_argument("--dir", help="Directory of CSVs to validate")
    p.add_argument("--create-manifest", action="store_true", help="Write manifest JSON next to CSV(s)")
    args = p.parse_args()

    if not args.file and not args.dir:
        print("Provide --file or --dir")
        return 2

    if args.file:
        csv_path = Path(args.file)
        if not csv_path.exists():
            print("Missing file:", csv_path)
            return 2
        _process_file(csv_path, create=args.create_manifest)
    else:
        d = Path(args.dir)
        if not d.exists():
            print("Missing directory:", d)
            return 2
        for path in d.glob("*.csv"):
            _process_file(path, create=args.create_manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

