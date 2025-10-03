#!/usr/bin/env python3
"""
AI Engineering Smoke Test

Runs a minimal end-to-end check when NVIDIA_API_KEY is present and PDFs exist.
Otherwise runs offline import + filesystem checks. Produces logs/smoke.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass
class Step:
    name: str
    status: str  # pass|fail|na|skip
    detail: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


def _safe_import(name: str) -> Step:
    try:
        __import__(name)
        return Step(name=f"import:{name}", status="pass")
    except Exception as e:  # noqa: BLE001
        return Step(name=f"import:{name}", status="fail", detail=str(e))


def _list_pdfs(docs: Path) -> list[str]:
    if not docs.exists():
        return []
    return [p.name for p in list(docs.glob("*.pdf")) + list(docs.glob("*.PDF"))]


def run_smoke(docs_dir: str, out_path: str, strict: bool = False) -> int:
    steps: list[Step] = []
    docs = Path(docs_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Ensure repository root is importable (so `src.*` can be imported)
    try:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    except Exception:
        pass

    # Offline checks first
    for mod in ("langchain", "langchain_core", "requests"):
        steps.append(_safe_import(mod))
    for mod in ("src.rag_agent", "src.nvidia_embeddings", "src.vector_database", "src.document_loader"):
        try:
            __import__(mod)
            steps.append(Step(name=f"import:{mod}", status="pass"))
        except Exception as e:  # noqa: BLE001
            steps.append(Step(name=f"import:{mod}", status="fail", detail=str(e)))

    pdfs = _list_pdfs(docs)
    steps.append(Step(name="docs", status="pass" if docs.exists() else "na", detail=f"{len(pdfs)} PDFs in {docs}"))

    # Online mode only if NVIDIA_API_KEY and at least one PDF
    nv_key = os.getenv("NVIDIA_API_KEY")
    if nv_key and pdfs:
        try:
            from src.rag_agent import RAGAgent  # type: ignore

            agent = RAGAgent(docs_folder=str(docs), api_key=nv_key)
            steps.append(Step(name="agent-init", status="pass", detail=f"model={agent.embeddings.model_name}"))

            ok = agent.setup_knowledge_base(force_rebuild=False)
            steps.append(Step(name="kb-setup", status="pass" if ok else "fail"))

            resp = agent.ask_question("What is the scope of this knowledge base?", k=2)
            steps.append(
                Step(
                    name="query",
                    status="pass" if bool(resp and resp.answer) else "fail",
                    detail=f"sources={len(resp.source_documents)} time={resp.processing_time:.2f}s",
                )
            )
        except Exception as e:  # noqa: BLE001
            steps.append(Step(name="e2e", status="fail", detail=str(e)))
    else:
        reason = "no-key" if not nv_key else "no-pdfs"
        steps.append(Step(name="e2e", status="na", detail=reason))

    payload = {
        "timestamp": time.time(),
        "docs": str(docs),
        "steps": [asdict(s) for s in steps],
    }
    out.write_text(json.dumps(payload, indent=2))

    if strict:
        # Fail if any hard failures
        if any(s.status == "fail" for s in steps):
            return 1
    return 0


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="AI engineering smoke test")
    ap.add_argument("--docs", default="Data", help="Documents directory with PDFs")
    ap.add_argument("--output", default="logs/smoke.json", help="Output JSON path")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero on any failure")
    args = ap.parse_args(argv)

    return run_smoke(args.docs, args.output, args.strict)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
