"""
PubMed NIM-Native Pipeline CLI

Implements a cheapest, NIM-native PubMed pipeline with:
- Robust configuration validation and strict health gating (skip via --skip-health)
- Compact Markdown config summary (endpoint → model → status), masked API key
- PubMed E-utilities fetch (titles/abstracts), overlay summaries, and persistence
- Embedding + Reranking via NVIDIA NeMo Retriever NIMs with batch/top-k controls
- Credit-aware per-phase reporting (NVIDIA_BUILD_FREE_TIER toggle) and warnings
- Colab Drive persistence helpers for .env, cache, and run summaries
- CSV manifest/versioning utilities and simple percentile benchmarking CSV

Usage examples:
  python scripts/pubmed_nim_pipeline.py --query "metformin pharmacokinetics" \
      --batch-size 3 --top-k 5 --health-latency-ms 2500

  python scripts/pubmed_nim_pipeline.py --query "cyp3a4 inhibitors" --skip-health

Environment (recommended):
  NVIDIA_API_KEY=... (provisioned for Embedding + Reranking)
  ENABLE_NEMO_EXTRACTION=true
  NEMO_EXTRACTION_STRATEGY=nemo
  NEMO_EXTRACTION_STRICT=true
  APP_ENV=production
  PHARMA_DOMAIN_OVERLAY=true
  NVIDIA_BUILD_FREE_TIER=true (optional credit logging)
  PUBMED_EUTILS_API_KEY=... and PUBMED_EMAIL=... (optional, higher rate limits)
  EMBEDDING_MODEL and RERANK_MODEL (optional overrides)
  NEMO_*_ENDPOINT (optional self-hosted overrides)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Optional dotenv
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

# Local imports
from src.enhanced_config import EnhancedRAGConfig
from src.nemo_retriever_client import (
    NVIDIABuildCreditsMonitor,
    create_nemo_client,
)
from src.pubmed_eutils_client import PubMedEutilsClient
from src.pharmaceutical_processor import PharmaceuticalProcessor
from scripts.config_validator import validate_complete_configuration
from scripts.colab_persistence import (
    ensure_drive_mounted,
    is_colab,
    save_env,
    persist_cache,
    save_run_summary,
)


# -------------------------------
# Utils
# -------------------------------


def _load_env() -> None:
    env_path = ROOT / ".env"
    if load_dotenv is not None and env_path.exists():
        try:
            load_dotenv(dotenv_path=str(env_path))
            return
        except Exception:
            pass
    # Fallback: manual parse
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass


def _mask_key(key: Optional[str]) -> str:
    if not key:
        return "(unset)"
    k = key.strip()
    if len(k) <= 8:
        return f"{k[0:2]}***{k[-2:]}"
    return f"{k[0:4]}***{k[-4:]}"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round((pct / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _write_benchmark_row(
    out_csv: Path,
    row: Dict[str, Any],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _checksum_texts(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _compact_markdown_table(rows: List[Tuple[str, str, str, str]]) -> str:
    # rows of (Service, Endpoint, Model, Status)
    lines = ["| Service | Endpoint | Model | Status |", "|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


# -------------------------------
# Phases
# -------------------------------


@dataclass
class PhaseResult:
    ok: bool
    latency_ms: float = 0.0
    credits: int = 0
    details: Dict[str, Any] = None  # type: ignore


async def phase_config_and_health(
    skip_health: bool,
    health_latency_ms: float,
    credits_monitor: Optional[NVIDIABuildCreditsMonitor],
) -> Tuple[PhaseResult, Dict[str, Any]]:
    # Validate config
    cfg_result = validate_complete_configuration()
    if not cfg_result.valid:
        return PhaseResult(False, details={"errors": cfg_result.errors, "warnings": cfg_result.warnings}), {}

    # Summarize endpoints/models
    rag_cfg = EnhancedRAGConfig.from_env()
    endpoints = rag_cfg.get_effective_endpoints()
    models = rag_cfg.get_effective_models()
    api_key = os.getenv("NVIDIA_API_KEY")

    # Perform health check unless skipped
    health_rows: List[Tuple[str, str, str, str]] = []
    health_detail: Dict[str, Any] = {}
    t0 = time.time()
    if skip_health:
        for svc in ("embedding", "reranking"):
            health_rows.append((svc, endpoints.get(svc, ""), models.get(svc, ""), "skipped"))
        md = _compact_markdown_table(health_rows)
        print("=== CONFIG SUMMARY ===")
        print(f"API Key: {_mask_key(api_key)}")
        print(md)
        return PhaseResult(True, latency_ms=(time.time() - t0) * 1000, details={"skipped": True}), {
            "endpoints": endpoints,
            "models": models,
            "api_key_masked": _mask_key(api_key),
        }

    client = await create_nemo_client(credits_monitor=credits_monitor)
    health = await client.health_check(force=True)

    ok = True
    for svc in ("embedding", "reranking"):
        st = health.get(svc) or {}
        stat = st.get("status", "unknown")
        lat = float(st.get("response_time_ms", 0.0))
        if stat != "healthy":
            ok = False
        if health_latency_ms and lat > health_latency_ms:
            stat = f"unhealthy (latency {lat:.1f}ms > {health_latency_ms:.1f}ms)"
            ok = False
        health_rows.append((svc, endpoints.get(svc, ""), models.get(svc, ""), stat))
        health_detail[svc] = {"status": stat, "latency_ms": lat, "endpoint": endpoints.get(svc), "model": models.get(svc)}

    md = _compact_markdown_table(health_rows)
    print("=== CONFIG SUMMARY ===")
    print(f"API Key: {_mask_key(api_key)}")
    print(md)

    if credits_monitor:
        credits_monitor.log_api_call("health", tokens_used=1)

    return PhaseResult(ok, latency_ms=(time.time() - t0) * 1000, credits=1 if credits_monitor else 0, details=health_detail), {
        "endpoints": endpoints,
        "models": models,
        "api_key_masked": _mask_key(api_key),
        "health": health_detail,
    }


def phase_pubmed_fetch(query: str, max_items: int = 5) -> PhaseResult:
    t0 = time.time()
    client = PubMedEutilsClient()
    try:
        items = client.search_and_fetch(query, max_items=max_items)
    except Exception:
        # Retry once without credentials to handle malformed env keys or 4xx
        try:
            client2 = PubMedEutilsClient(email="", api_key="")
            # ensure params won't include these
            client2.email = None
            client2.api_key = None
            items = client2.search_and_fetch(query, max_items=max_items)
        except Exception as exc2:
            return PhaseResult(False, details={"error": str(exc2)})

    # Fast-fail if malformed
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    for it in items:
        title = (it.get("title") or "").strip()
        abstract = (it.get("abstract") or "").strip()
        if not title and not abstract:
            continue
        texts.append((title + ". " + abstract).strip())
        meta.append(it)

    if not texts:
        return PhaseResult(False, details={"error": "No valid PubMed texts returned"})

    return PhaseResult(True, latency_ms=(time.time() - t0) * 1000, details={"texts": texts, "metadata": meta})


async def phase_embed_and_rerank(
    query: str,
    texts: List[str],
    batch_size: int,
    top_k: int,
    credits_monitor: Optional[NVIDIABuildCreditsMonitor],
) -> Tuple[PhaseResult, PhaseResult]:
    client = await create_nemo_client(credits_monitor=credits_monitor)
    # Embedding
    t0 = time.time()
    try:
        batch = [t[:32768] for t in texts[: max(1, batch_size)]]
        embed_res = await client.embed_texts(batch, model="nv-embedqa-e5-v5", use_langchain=True)
        embed_ok = embed_res.success
        embed_latency = embed_res.response_time_ms or (time.time() - t0) * 1000
        if credits_monitor:
            # Deduct exactly one request per API call
            credits_monitor.log_api_call("embedding", tokens_used=1)
    except Exception as exc:
        return PhaseResult(False, details={"error": f"Embedding error: {exc}"}), PhaseResult(False)

    # Reranking
    t1 = time.time()
    try:
        rerank_res = await client.rerank_passages(
            query=query,
            passages=[t[:512] for t in texts],
            top_k=top_k,
        )
        rerank_ok = rerank_res.success
        rerank_latency = rerank_res.response_time_ms or (time.time() - t1) * 1000
        if credits_monitor:
            # Deduct exactly one request per API call
            credits_monitor.log_api_call("reranking", tokens_used=1)
    except Exception as exc:
        return PhaseResult(True, latency_ms=embed_latency, details={"embeddings": True}), PhaseResult(False, details={"error": f"Rerank error: {exc}"})

    return (
        PhaseResult(embed_ok, latency_ms=embed_latency, details={"count": len(texts[:batch_size])}),
        PhaseResult(rerank_ok, latency_ms=rerank_latency, details={"top_k": top_k}),
    )


def phase_overlay_summary(texts: List[str]) -> PhaseResult:
    overlay_on = os.getenv("PHARMA_DOMAIN_OVERLAY", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not overlay_on:
        return PhaseResult(True, details={"overlay": False})
    t0 = time.time()
    proc = PharmaceuticalProcessor()
    drug_canon = set(); reg_tags = set(); agencies = set(); evidence = set(); species = set(); pk = set(); risks = []
    for text in texts[:3]:
        meta = proc.enhance_document_metadata({"page_content": text, "metadata": {}})["metadata"]
        drug_canon.update([str(n) for n in (meta.get("drug_canonical_names") or [])])
        reg_tags.update([str(t) for t in (meta.get("regulatory_tags") or [])])
        agencies.update([str(a) for a in (meta.get("regulatory_agencies") or [])])
        if "cyp_risk_label" in meta:
            risks.append(str(meta["cyp_risk_label"]))
        if "evidence_level" in meta:
            evidence.add(str(meta["evidence_level"]))
        if meta.get("species"):
            species.add(str(meta["species"]))
        pk.update([str(p) for p in (meta.get("pk_signals_present") or [])])

    print("=== PHARMACEUTICAL OVERLAY ===")
    print(f"Drug canonical: {sorted(drug_canon) if drug_canon else []}")
    print(f"Regulatory tags: {sorted(reg_tags) if reg_tags else []}")
    print(f"Agencies: {sorted(agencies) if agencies else []}")
    if risks:
        from collections import Counter
        print(f"CYP risk: {Counter(risks).most_common(1)[0][0]}")
    print(f"Evidence: {sorted(evidence) if evidence else []}")
    print(f"Species: {sorted(species) if species else []}")
    print(f"PK: {sorted(pk) if pk else []}")
    return PhaseResult(True, latency_ms=(time.time() - t0) * 1000, details={"overlay": True})


def _persist_outputs(
    base_dir: Path,
    query: str,
    texts: List[str],
    metadata: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    # Save summary JSON
    save_run_summary(summary, dest_dir=str(base_dir), filename=f"pubmed_run_{ts}.json")
    # Save minimal CSV of results
    csv_path = base_dir / f"pubmed_results_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid", "doi", "title", "journal", "publication_date"])
        w.writeheader()
        for m in metadata:
            w.writerow({
                "pmid": m.get("pmid"),
                "doi": m.get("doi"),
                "title": m.get("title"),
                "journal": m.get("journal"),
                "publication_date": m.get("publication_date"),
            })

    # Write a manifest with checksum + preview
    manifest = {
        "version": 1,
        "created": _now_iso(),
        "query": query,
        "result_csv": csv_path.name,
        "checksum": _checksum_texts(texts),
        "preview": [{"title": (m.get("title") or "")[:120], "pmid": m.get("pmid")} for m in metadata[:3]],
    }
    with (base_dir / f"pubmed_results_{ts}.manifest.json").open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, ensure_ascii=False)


async def main() -> int:
    _load_env()

    # Enforce strict NIM-only behavior via environment defaults
    os.environ.setdefault("ENABLE_NEMO_EXTRACTION", "true")
    os.environ.setdefault("NEMO_EXTRACTION_STRATEGY", "nemo")
    os.environ.setdefault("NEMO_EXTRACTION_STRICT", "true")
    os.environ.setdefault("APP_ENV", "production")

    parser = argparse.ArgumentParser(description="PubMed NIM-native pipeline")
    parser.add_argument("--query", required=True, help="PubMed query string")
    parser.add_argument("--batch-size", type=int, default=3, help="Number of texts to embed")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K passages to rerank")
    parser.add_argument("--max-results", type=int, default=5, help="Max PubMed items to fetch")
    parser.add_argument("--skip-health", action="store_true", help="Bypass health gate")
    parser.add_argument("--health-latency-ms", type=float, default=float(os.getenv("HEALTH_LATENCY_MS", "5000")), help="Max allowed p95 latency")
    parser.add_argument("--credits-warn-pct", type=int, default=int(os.getenv("CREDITS_WARN_PCT", "10")), help="Warn when credits remaining < pct")
    parser.add_argument("--output-dir", default=str(ROOT / "persist"), help="Directory for summaries/benchmarks")
    parser.add_argument("--pubmed-only", action="store_true", help="Fetch PubMed + overlay + persist; skip NIM embed/rerank")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # Colab persistence helpers
    if is_colab():
        ensure_drive_mounted()
    save_env(str(out_dir))

    # Optional credits monitoring
    free_tier = os.getenv("NVIDIA_BUILD_FREE_TIER", "").strip().lower() in {"1","true","yes","on"}
    credits_monitor = NVIDIABuildCreditsMonitor(os.getenv("NVIDIA_API_KEY")) if free_tier else None

    # Phase 0-1: Config + Health
    cfg_health, cfg_meta = await phase_config_and_health(
        skip_health=args.skip_health,
        health_latency_ms=float(args.health_latency_ms),
        credits_monitor=credits_monitor,
    )
    if not cfg_health.ok:
        print("❌ Health/config validation failed.")
        errs = (cfg_health.details or {}).get("errors") or []
        if errs:
            for e in errs:
                print(f" - {e}")
        return 2

    # Phase 2: PubMed fetch
    pubmed = phase_pubmed_fetch(args.query, max_items=max(1, min(10, args.max_results)))
    if not pubmed.ok:
        print("❌ PubMed fetch failed:", (pubmed.details or {}).get("error", "unknown"))
        return 3
    texts: List[str] = (pubmed.details or {}).get("texts", [])
    metadata: List[Dict[str, Any]] = (pubmed.details or {}).get("metadata", [])
    if not texts:
        print("❌ No valid texts from PubMed")
        return 3
    print(f"Fetched {len(texts)} PubMed items (capped).")

    # Overlay
    overlay = phase_overlay_summary(texts)

    # PubMed-only testing mode (skip NIM calls)
    if args.pubmed_only:
        embed_res = PhaseResult(True, latency_ms=0.0, details={"count": 0})
        rerank_res = PhaseResult(True, latency_ms=0.0, details={"top_k": 0})
    else:
        # Phase 3: Embed + Rerank
        embed_res, rerank_res = await phase_embed_and_rerank(
            query=args.query,
            texts=texts,
            batch_size=max(1, args.batch_size),
            top_k=max(1, args.top_k),
            credits_monitor=credits_monitor,
        )
        if not embed_res.ok or not rerank_res.ok:
            print("❌ Embed/rerank failed.")
            return 4

    # Phase 4: Credit monitoring & summary
    total_credits = 0
    per_phase = {}
    if credits_monitor:
        # Our monitor is approximate; split by phases we logged
        total_credits = credits_monitor.credits_used
        per_phase = {
            "health": 1 if (cfg_health.details or {}).get("skipped") is not True else 0,
            "embedding": (embed_res.details or {}).get("count", 0),
            "reranking": (rerank_res.details or {}).get("top_k", 0),
        }
        remaining = credits_monitor.credits_remaining
        pct_remaining = (remaining / (remaining + total_credits)) * 100 if (remaining + total_credits) > 0 else 100
        print("=== CREDITS SUMMARY ===")
        print(f"Health: {per_phase['health']}  Embedding: {per_phase['embedding']}  Reranking: {per_phase['reranking']}")
        print(f"Total used (approx): {total_credits}  Remaining (approx): {remaining} ({pct_remaining:.1f}%)")
        if pct_remaining < max(1, int(args.credits_warn_pct)):
            print(f"⚠️  Low free-tier credits (<{args.credits_warn_pct}%). Check NVIDIA Build console.")

    # Benchmarks: record single-sample latencies
    bench_row = {
        "timestamp": _now_iso(),
        "query": args.query,
        "batch_size": int(args.batch_size),
        "top_k": int(args.top_k),
        "embedding_endpoint": (cfg_meta.get("endpoints") or {}).get("embedding", ""),
        "reranking_endpoint": (cfg_meta.get("endpoints") or {}).get("reranking", ""),
        "embedding_model": (cfg_meta.get("models") or {}).get("embedding", ""),
        "reranking_model": (cfg_meta.get("models") or {}).get("reranking", ""),
        "health_latency_ms": float(cfg_health.latency_ms),
        "pubmed_latency_ms": float(pubmed.latency_ms),
        "embed_latency_ms": float(embed_res.latency_ms),
        "rerank_latency_ms": float(rerank_res.latency_ms),
        "credits_used": int(total_credits),
    }
    _write_benchmark_row(out_dir / "benchmarks.csv", bench_row)

    # Persist run artifacts
    summary = {
        "timestamp": bench_row["timestamp"],
        "query": args.query,
        "config": cfg_meta,
        "counts": {"texts": len(texts), "metadata": len(metadata)},
        "latencies_ms": {
            "health": float(cfg_health.latency_ms),
            "pubmed": float(pubmed.latency_ms),
            "embed": float(embed_res.latency_ms),
            "rerank": float(rerank_res.latency_ms),
        },
        "credits": {"per_phase": per_phase, "total": total_credits},
    }
    _persist_outputs(out_dir, args.query, texts, metadata, summary)
    persist_cache("query_cache", str(out_dir))

    # Final summary
    print("=== PIPELINE SUMMARY ===")
    print(f"Query: {args.query}")
    print(f"Texts: {len(texts)}  Batch: {args.batch_size}  Top-K: {args.top_k}")
    print(f"Latency(ms): health={cfg_health.latency_ms:.1f} pubmed={pubmed.latency_ms:.1f} embed={embed_res.latency_ms:.1f} rerank={rerank_res.latency_ms:.1f}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
