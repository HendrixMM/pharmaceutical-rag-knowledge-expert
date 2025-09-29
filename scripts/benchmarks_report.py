"""
Benchmarks Report Helper

Reads `persist/benchmarks.csv` (or a given file) and prints simple
p50/p95 latencies per phase and basic aggregates.

Usage:
  python -m scripts.benchmarks_report --file persist/benchmarks.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round((pct / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmarks percentile report")
    p.add_argument("--file", default="persist/benchmarks.csv", help="Benchmarks CSV path")
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print("Missing:", path)
        return 2

    fields = [
        ("health_latency_ms", "Health"),
        ("pubmed_latency_ms", "PubMed"),
        ("embed_latency_ms", "Embed"),
        ("rerank_latency_ms", "Rerank"),
    ]
    data = {k: [] for k, _ in fields}

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for k, _ in fields:
                try:
                    data[k].append(float(row.get(k, 0) or 0))
                except Exception:
                    pass

    print("=== Percentiles (ms) ===")
    for k, label in fields:
        vals = data[k]
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        print(f"{label:7s}  p50={p50:8.1f}  p95={p95:8.1f}  n={len(vals)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

