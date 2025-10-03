#!/usr/bin/env python3
"""
Benchmark Orchestrator

High-ROI wrapper to run preflight (latency + concurrency recommendations)
and then execute the consolidated benchmark run with gating and summary export.

Usage example:
  python scripts/orchestrate_benchmarks.py \
    --mode both \
    --preset cloud_first_adaptive \
    --output results/benchmark_runs/orchestrated \
    --summary-output results/benchmark_runs/orchestrated/summary.json \
    --auto-concurrency --skip-classifier-validation \
    --preflight-sample-count 2 --preflight-min-concurrency 2 --fail-on-preflight \
    --min-cloud-score 0.40 --max-cloud-latency-ms 6000 --fail-on-regressions
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list[str], env: dict | None = None) -> int:
    proc = subprocess.run(cmd, env=env or os.environ.copy())
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Orchestrate preflight and benchmark run")
    ap.add_argument("--mode", choices=["cloud", "self_hosted", "both"], default="both")
    ap.add_argument("--preset")
    ap.add_argument("--output", default="results/benchmark_runs/orchestrated")
    ap.add_argument("--summary-output")
    ap.add_argument("--auto-concurrency", action="store_true")
    ap.add_argument("--skip-classifier-validation", action="store_true")
    ap.add_argument("--max-queries", type=int)
    ap.add_argument("--min-cloud-score", type=float)
    ap.add_argument("--max-cloud-latency-ms", type=float)
    ap.add_argument("--fail-on-regressions", action="store_true")
    ap.add_argument("--preflight-sample-count", type=int, default=1)
    ap.add_argument("--preflight-min-concurrency", type=int)
    ap.add_argument("--fail-on-preflight", action="store_true")
    ap.add_argument("--launch-stagger-ms", type=int)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    preflight_dir = out_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    preflight_json = preflight_dir / f"preflight_{ts}.json"

    # Optional stagger via environment variable
    env = os.environ.copy()
    if args.launch_stagger_ms is not None:
        env["BENCHMARK_LAUNCH_STAGGER_MS"] = str(int(args.launch_stagger_ms))

    # Step 1: Preflight (preflight-only)
    preflight_cmd = [
        sys.executable,
        "scripts/run_pharmaceutical_benchmarks.py",
        "--preflight",
        "--preflight-only",
        "--preflight-output",
        str(preflight_json),
        "--preflight-sample-count",
        str(int(args.preflight_sample_count)),
        "--mode",
        args.mode,
    ]
    if args.preset:
        preflight_cmd += ["--preset", args.preset]
    if args.preflight_min_concurrency is not None:
        preflight_cmd += ["--preflight-min-concurrency", str(int(args.preflight_min_concurrency))]
    if args.fail_on_preflight:
        preflight_cmd += ["--fail-on-preflight"]

    print("[orchestrate] Running preflight...\n  ", " ".join(preflight_cmd))
    rc = run_cmd(preflight_cmd, env)
    if rc != 0:
        print(f"[orchestrate] Preflight failed with exit code {rc}")
        return rc

    # Step 2: Benchmark run using preflight map
    run_cmdline = [
        sys.executable,
        "scripts/run_pharmaceutical_benchmarks.py",
        "--mode",
        args.mode,
        "--save-results",
        "--output",
        str(out_dir),
        "--preflight-map-input",
        str(preflight_json),
    ]
    if args.summary_output:
        run_cmdline += ["--summary-output", args.summary_output]
    if args.preset:
        run_cmdline += ["--preset", args.preset]
    if args.auto_concurrency:
        run_cmdline += ["--auto-concurrency"]
    if args.skip_classifier_validation:
        run_cmdline += ["--skip-classifier-validation"]
    if args.max_queries is not None:
        run_cmdline += ["--max-queries", str(int(args.max_queries))]
    if args.min_cloud_score is not None:
        run_cmdline += ["--min-cloud-score", str(float(args.min_cloud_score))]
    if args.max_cloud_latency_ms is not None:
        run_cmdline += ["--max-cloud-latency-ms", str(float(args.max_cloud_latency_ms))]
    if args.fail_on_regressions:
        run_cmdline += ["--fail-on-regressions"]

    print("[orchestrate] Running benchmarks...\n  ", " ".join(run_cmdline))
    rc2 = run_cmd(run_cmdline, env)
    print(f"[orchestrate] Run finished with exit code {rc2}")
    return rc2


if __name__ == "__main__":
    sys.exit(main())
