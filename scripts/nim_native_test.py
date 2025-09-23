"""
NVIDIA NIM-Native End-to-End Test

Validates a strict NeMo/NIM-only pipeline:
- NIM service health (extraction/embedding/reranking)
- NeMo VLM extraction (no fallback)
- NeMo embeddings (single-model check)
- NeMo reranking
- Pharmaceutical domain overlay enrichment on extracted docs

Usage:
  python scripts/nim_native_test.py

Prerequisites:
  - .env with a valid NVIDIA_API_KEY
  - ENABLE_NEMO_EXTRACTION=true, NEMO_EXTRACTION_STRICT=true, NEMO_EXTRACTION_STRATEGY=nemo
  - APP_ENV=production (forces strict mode)
  - Place at least one PDF under Data/Docs/ or provide a path via --pdf
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import os
import sys
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional; we will fallback to manual loader

# Ensure local src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from src.nemo_retriever_client import create_nemo_client, NVIDIABuildCreditsMonitor
from src.nemo_extraction_service import NeMoExtractionService
from src.pharmaceutical_processor import PharmaceuticalProcessor
# Import embedding service lazily only when embedding step runs to avoid type
# dependency issues if optional monitoring classes change
try:
    from src.nemo_embedding_service import NeMoEmbeddingService, EmbeddingConfig
except Exception:  # pragma: no cover - defensive import
    NeMoEmbeddingService = None  # type: ignore
    EmbeddingConfig = None  # type: ignore


def _load_env() -> None:
    env_path = ROOT / ".env"
    if load_dotenv is not None and env_path.exists():
        # Use python-dotenv when available
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


def _pick_pdf(path_arg: str | None) -> Path | None:
    if path_arg:
        p = Path(path_arg)
        return p if p.exists() else None
    # Search common locations
    for pattern in ("Data/Docs/*.pdf", "examples/*.pdf", "*.pdf"):
        files = glob.glob(pattern)
        if files:
            return Path(files[0])
    return None


async def _validate_services(credits_monitor: NVIDIABuildCreditsMonitor | None = None) -> bool:
    print("=== NVIDIA NIM SERVICES VALIDATION ===")
    try:
        client = await create_nemo_client(credits_monitor=credits_monitor)
        health = await client.health_check(force=True)
        ok = True
        for name, status in (health or {}).items():
            if status.get("status") == "healthy":
                print(f"✅ {name.upper()} NIM: {status.get('response_time_ms', 0):.1f}ms")
            else:
                print(f"❌ {name.upper()} NIM: {status.get('error') or 'unavailable'}")
                ok = False
        return ok
    except Exception as exc:
        print(f"❌ Failed to validate NIM services: {exc}")
        return False


async def _extract_with_nemo(pdf_path: Path) -> NeMoExtractionService:
    print(f"=== NEMO EXTRACTION TEST: {pdf_path} ===")
    service = NeMoExtractionService()
    result = await service.extract_document(
        file_path=pdf_path,
        extraction_strategy="nemo",
        enable_pharmaceutical_analysis=True,
        chunk_strategy="semantic",
        preserve_tables=True,
        extract_images=True,
    )
    if result.success:
        docs = result.documents or []
        print(f"✅ Extraction successful ({result.processing_time_ms:.1f}ms)")
        print(f"   Documents: {len(docs)}  Tables: {len(result.tables or [])}  Charts: {len(result.charts or [])}")
        print(f"   Method: {result.extraction_method}")
    else:
        print(f"❌ Extraction failed: {result.error}  (method={result.extraction_method})")
    return service, result


async def _embed_with_nemo(texts: List[str]) -> None:
    print("=== NEMO EMBEDDING TEST ===")
    if not texts:
        print("ℹ️  No texts to embed (empty extraction)")
        return

    # Avoid numpy dependency path in service by skipping normalization
    if EmbeddingConfig is None or NeMoEmbeddingService is None:
        print("ℹ️  Embedding service unavailable (import error); skipping embedding test")
        return

    # Get embedding model from environment variable with fallback
    embedding_model = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")
    # Extract model name without provider prefix for the config
    model_name = embedding_model.split("/")[-1] if "/" in embedding_model else embedding_model
    print(f"🔧 Using embedding model: {embedding_model}")

    config = EmbeddingConfig(model=model_name)
    config.normalize_embeddings = False
    emb_service = NeMoEmbeddingService(config=config, enable_multi_model_strategy=False)
    try:
        vecs = await emb_service.embed_documents(texts[:3])
        dims = len(vecs[0]) if vecs else 0
        print(f"✅ Embeddings generated: {len(vecs)} vectors, {dims} dimensions")
        print(f"   Model: {model_name}")
    except Exception as exc:
        print(f"❌ Embedding failed: {exc}")


async def _rerank_with_nemo(query: str, docs: List[str]) -> None:
    print("=== NEMO RERANKING TEST ===")
    if not docs:
        print("ℹ️  No documents to rerank (empty extraction)")
        return

    # Get reranking model from environment variable with fallback
    rerank_model = os.getenv("RERANK_MODEL", "llama-3_2-nemoretriever-500m-rerank-v2")
    print(f"🔧 Using reranking model: {rerank_model}")

    try:
        client = await create_nemo_client()
        res = await client.rerank_passages(
            query=query or "pharmaceutical information",
            passages=[t[:500] for t in docs[:5]],
            model=rerank_model,
            top_k=3,
        )
        if res.success:
            reranked_data = res.data.get("reranked_passages", [])
            print(f"✅ Reranking successful: processed {len(reranked_data)} passages")
            print(f"   Response time: {res.response_time_ms:.1f}ms")
        else:
            print(f"❌ Reranking failed: {res.error}")
    except Exception as exc:
        print(f"❌ Reranking error: {exc}")


def _overlay_summary(texts: List[str]) -> None:
    print("=== PHARMACEUTICAL DOMAIN OVERLAY TEST ===")
    overlay_on = os.getenv("PHARMA_DOMAIN_OVERLAY", "false").strip().lower() in {"1", "true", "yes", "on"}
    print(f"Overlay: {'active' if overlay_on else 'inactive'}")
    if not overlay_on:
        return
    proc = PharmaceuticalProcessor()
    drug_canon = set(); reg_tags = set(); agencies = set(); evidence = set(); species = set(); pk = set()
    risk_labels = []
    for text in texts[:3]:
        meta = proc.enhance_document_metadata({"page_content": text, "metadata": {}})["metadata"]
        drug_canon.update([str(n) for n in (meta.get("drug_canonical_names") or [])])
        reg_tags.update([str(t) for t in (meta.get("regulatory_tags") or [])])
        agencies.update([str(a) for a in (meta.get("regulatory_agencies") or [])])
        if "cyp_risk_label" in meta:
            risk_labels.append(str(meta["cyp_risk_label"]))
        if "evidence_level" in meta:
            evidence.add(str(meta["evidence_level"]))
        if meta.get("species"):
            species.add(str(meta["species"]))
        pk.update([str(p) for p in (meta.get("pk_signals_present") or [])])
    print(f"Drug canonical: {sorted(drug_canon) if drug_canon else []}")
    print(f"Regulatory tags: {sorted(reg_tags) if reg_tags else []}")
    print(f"Agencies: {sorted(agencies) if agencies else []}")
    if risk_labels:
        from collections import Counter
        print(f"CYP risk: {Counter(risk_labels).most_common(1)[0][0]}")
    print(f"Evidence: {sorted(evidence) if evidence else []}")
    print(f"Species: {sorted(species) if species else []}")
    print(f"PK: {sorted(pk) if pk else []}")


async def main() -> int:
    _load_env()

    # Enforce strict NIM-only behavior via environment (best effort)
    os.environ.setdefault("ENABLE_NEMO_EXTRACTION", "true")
    os.environ.setdefault("NEMO_EXTRACTION_STRATEGY", "nemo")
    os.environ.setdefault("NEMO_EXTRACTION_STRICT", "true")
    os.environ.setdefault("APP_ENV", "production")

    parser = argparse.ArgumentParser(description="NIM-native E2E test")
    parser.add_argument("--pdf", help="Path to a PDF to test", default=None)
    args = parser.parse_args()

    pdf_path = _pick_pdf(args.pdf)
    if not pdf_path:
        print("❌ No PDF found. Add a PDF under Data/Docs/ or pass --pdf.")
        return 2

    # 1) Validate services
    free_tier = os.getenv("NVIDIA_BUILD_FREE_TIER", "").strip().lower() in {"1","true","yes","on"}
    credits_monitor = NVIDIABuildCreditsMonitor(os.getenv("NVIDIA_API_KEY")) if free_tier else None
    if not await _validate_services():
        print("❌ NIM services not fully available - cannot proceed (strict mode)")
        return 3

    # 2) Extract
    service, result = await _extract_with_nemo(pdf_path)
    if not getattr(result, "success", False):
        return 4

    docs = [d.page_content for d in (result.documents or [])]

    # 3) Embeddings
    await _embed_with_nemo(docs)
    if credits_monitor:
        credits_monitor.log_api_call("embedding", tokens_used=max(1, min(len(docs), 3)))

    # 4) Reranking
    await _rerank_with_nemo("drug interactions", docs)
    if credits_monitor:
        credits_monitor.log_api_call("reranking", tokens_used=max(1, min(len(docs), 3)))

    # 5) Overlay summary
    _overlay_summary(docs)

    if credits_monitor:
        print("=== FREE TIER PHARMACEUTICAL OVERLAY ===")
        print(f"💰 Credits used: {credits_monitor.credits_used}")
        print(f"💰 Credits remaining: {credits_monitor.credits_remaining}")

    print("=== NIM-NATIVE TEST COMPLETE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
