#!/usr/bin/env python3
"""
AI Engineering Preflight
Runs environment, secrets, network, vector DB, and asset checks.
Writes JSON report to logs/preflight.json (configurable).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "na" | "skip"
    detail: str | None = None
    suggestion: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    timestamp: float
    system: dict[str, Any]
    results: list[CheckResult]

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "system": self.system,
                "results": [asdict(r) for r in self.results],
            },
            indent=2,
            sort_keys=False,
        )


def safe_import(mod_name: str) -> tuple[bool, str | None]:
    try:
        __import__(mod_name)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def check_python_env() -> CheckResult:
    py_ver = platform.python_version()
    ok = tuple(map(int, py_ver.split(".")[:2])) >= (3, 10)
    return CheckResult(
        name="python-version",
        status="pass" if ok else "fail",
        detail=f"Detected Python {py_ver} (require >= 3.10)",
        suggestion=None if ok else "Use pyenv or system Python to upgrade to >=3.10",
        data={"python": py_ver},
    )


PKGS = [
    # Core ecosystem (checked if installed; not all are required for every setup)
    "openai",
    "anthropic",
    "cohere",
    "qdrant_client",
    "pinecone",
    "weaviate",
    "chromadb",
    "llama_index",
    "langchain",
    "langgraph",
    "sentence_transformers",
    "tiktoken",
]


def check_packages() -> CheckResult:
    missing: dict[str, str] = {}
    for m in PKGS:
        ok, err = safe_import(m)
        if not ok and err:
            missing[m] = err
    status = "pass" if not missing else "fail"
    suggestion = None
    if missing:
        suggestion = "pip install " + " ".join(sorted(missing.keys()))
    return CheckResult(
        name="python-packages",
        status=status,
        detail=("All expected packages present" if status == "pass" else "Missing or import errors detected"),
        suggestion=suggestion,
        data={"missing": missing},
    )


def check_gpu() -> CheckResult:
    ok, err = safe_import("torch")
    if not ok:
        return CheckResult(
            name="gpu-cuda",
            status="skip",
            detail="torch not installed; skipping GPU check",
            suggestion="pip install torch --index-url https://download.pytorch.org/whl/cu121 (or CPU build)",
        )
    try:
        import torch  # type: ignore

        data = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
        }
        status = "pass" if data["cuda_available"] else "fail"
        return CheckResult(
            name="gpu-cuda",
            status=status,
            detail=f"CUDA available={data['cuda_available']}, version={data['cuda_version']}",
            suggestion=None if status == "pass" else "Install NVIDIA drivers/CUDA and torch with CUDA support",
            data=data,
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="gpu-cuda",
            status="fail",
            detail=f"Error checking CUDA: {e}",
            suggestion="Validate CUDA installation and torch build",
        )


ENV_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "COHERE_API_KEY",
    "NVIDIA_API_KEY",
    "PINECONE_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "WEAVIATE_URL",
    "WEAVIATE_API_KEY",
    "MILVUS_URI",
    "LANGSMITH_API_KEY",
    "WANDB_API_KEY",
]


def check_env_keys() -> CheckResult:
    presence = {k: bool(os.getenv(k)) for k in ENV_KEYS}
    any_present = any(presence.values())
    status = "pass" if any_present else "fail"
    missing = [k for k, v in presence.items() if not v]
    sugg = None
    if not any_present:
        sugg = "Create .env and add at least one provider + vector DB key"
    return CheckResult(
        name="env-keys",
        status=status,
        detail=("Some keys present" if any_present else "No relevant keys detected"),
        suggestion=sugg,
        data={"present": presence, "missing": missing},
    )


def http_check(url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> int:
    # use stdlib to avoid adding deps
    import urllib.request

    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            return int(resp.getcode())
    except Exception:
        return 0


def check_network() -> list[CheckResult]:
    out: list[CheckResult] = []
    if os.getenv("OPENAI_API_KEY"):
        code = http_check(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
        )
        out.append(
            CheckResult(
                name="network-openai",
                status="pass" if code in (200, 401) else "fail",
                detail=f"HTTP {code} from OpenAI models endpoint",
                suggestion="Verify OPENAI_API_KEY and network egress" if code == 0 else None,
            )
        )
    else:
        out.append(CheckResult(name="network-openai", status="na", detail="No OPENAI_API_KEY"))

    if os.getenv("ANTHROPIC_API_KEY"):
        code = http_check(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": os.getenv("ANTHROPIC_API_KEY", "")},
        )
        out.append(
            CheckResult(
                name="network-anthropic",
                status="pass" if code in (200, 401) else "fail",
                detail=f"HTTP {code} from Anthropic models endpoint",
            )
        )
    else:
        out.append(CheckResult(name="network-anthropic", status="na", detail="No ANTHROPIC_API_KEY"))

    # Vector DB public endpoints (best-effort)
    if os.getenv("PINECONE_API_KEY"):
        code = http_check("https://api.pinecone.io/")
        out.append(
            CheckResult(
                name="network-pinecone",
                status="pass" if code in (200, 401, 403) else "fail",
                detail=f"HTTP {code} from Pinecone",
            )
        )
    else:
        out.append(CheckResult(name="network-pinecone", status="na", detail="No PINECONE_API_KEY"))

    qurl = os.getenv("QDRANT_URL")
    if qurl:
        code = http_check(f"{qurl.rstrip('/')}/healthz", headers={"api-key": os.getenv("QDRANT_API_KEY", "")})
        out.append(
            CheckResult(
                name="network-qdrant",
                status="pass" if code in (200, 401, 403) else "fail",
                detail=f"HTTP {code} from Qdrant /healthz",
            )
        )
    else:
        out.append(CheckResult(name="network-qdrant", status="na", detail="No QDRANT_URL"))

    wurl = os.getenv("WEAVIATE_URL")
    if wurl:
        code = http_check(wurl)
        out.append(
            CheckResult(
                name="network-weaviate",
                status="pass" if code in (200, 401, 403) else "fail",
                detail=f"HTTP {code} from Weaviate URL",
            )
        )
    else:
        out.append(CheckResult(name="network-weaviate", status="na", detail="No WEAVIATE_URL"))

    return out


def check_vector_clients() -> list[CheckResult]:
    results: list[CheckResult] = []

    # Qdrant collections
    if os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY"):
        ok, err = safe_import("qdrant_client")
        if ok:
            try:
                from qdrant_client import QdrantClient  # type: ignore

                c = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
                cols = [col.name for col in c.get_collections().collections]
                results.append(
                    CheckResult(
                        name="qdrant-collections",
                        status="pass",
                        detail=f"Collections: {cols}",
                        data={"collections": cols},
                    )
                )
            except Exception as e:  # noqa: BLE001
                results.append(
                    CheckResult(
                        name="qdrant-collections",
                        status="fail",
                        detail=f"Qdrant error: {e}",
                        suggestion="Verify QDRANT_URL/QDRANT_API_KEY and server availability",
                    )
                )
        else:
            results.append(
                CheckResult(
                    name="qdrant-collections",
                    status="skip",
                    detail=f"qdrant_client not installed ({err})",
                    suggestion="pip install qdrant-client",
                )
            )
    else:
        results.append(CheckResult(name="qdrant-collections", status="na", detail="No QDRANT_URL/API_KEY"))

    # Pinecone indexes
    if os.getenv("PINECONE_API_KEY"):
        ok, err = safe_import("pinecone")
        if ok:
            try:
                from pinecone import Pinecone  # type: ignore

                pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                idxs = [i["name"] for i in pc.list_indexes()]
                results.append(
                    CheckResult(
                        name="pinecone-indexes",
                        status="pass",
                        detail=f"Indexes: {idxs}",
                        data={"indexes": idxs},
                    )
                )
            except Exception as e:  # noqa: BLE001
                results.append(
                    CheckResult(
                        name="pinecone-indexes",
                        status="fail",
                        detail=f"Pinecone error: {e}",
                        suggestion="Verify PINECONE_API_KEY and account access",
                    )
                )
        else:
            results.append(
                CheckResult(
                    name="pinecone-indexes",
                    status="skip",
                    detail=f"pinecone not installed ({err})",
                    suggestion="pip install pinecone-client",
                )
            )
    else:
        results.append(CheckResult(name="pinecone-indexes", status="na", detail="No PINECONE_API_KEY"))

    return results


ASSET_DIRS = [
    "Data",
    "vector_db",
    "cache",
    "pubmed_cache",
]


def check_assets() -> CheckResult:
    present = {d: Path(d).exists() for d in ASSET_DIRS}
    status = "pass" if any(present.values()) else "fail"
    detail = "+".join([f"{d}={'yes' if p else 'no'}" for d, p in present.items()])
    sugg = None if status == "pass" else "Create required data/cache directories as needed"
    return CheckResult(
        name="asset-directories",
        status=status,
        detail=detail,
        suggestion=sugg,
        data={"present": present},
    )


def check_guardrails() -> CheckResult:
    exists = Path("guardrails").exists()
    return CheckResult(
        name="guardrails-config",
        status="pass" if exists else "na",
        detail=("guardrails/ present" if exists else "no guardrails/ directory"),
    )


def _auto_fix_dirs() -> CheckResult:
    created: list[str] = []
    skipped: list[str] = []
    errors: dict[str, str] = {}
    for d in ASSET_DIRS:
        p = Path(d)
        try:
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
                created.append(d)
            else:
                skipped.append(d)
        except Exception as e:  # noqa: BLE001
            errors[d] = str(e)
    status = "pass" if created and not errors else ("fail" if errors else "skip")
    detail = f"created={created or []}; skipped={skipped or []}"
    return CheckResult(
        name="auto-fix-dirs",
        status=status,
        detail=detail,
        suggestion=None if status == "pass" else "Manually create directories with proper permissions",
        data={"created": created, "skipped": skipped, "errors": errors},
    )


def _write_env_template(path: Path, missing_keys: list[str]) -> CheckResult:
    try:
        lines = [
            "# Generated by ai_preflight.py — fill in values and rename to .env if desired",
            "# Never commit real secrets to source control",
            "",
        ]
        for k in missing_keys:
            lines.append(f"{k}=# TODO: set value")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return CheckResult(
            name="env-template",
            status="pass",
            detail=f"Wrote template with {len(missing_keys)} keys to {path}",
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="env-template",
            status="fail",
            detail=f"Failed to write {path}: {e}",
            suggestion="Check filesystem permissions and path",
        )


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="AI preflight readiness checks")
    ap.add_argument("--output", default="logs/preflight.json", help="Output JSON path")
    ap.add_argument("--pretty", action="store_true", help="Print human summary")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if any fail")
    ap.add_argument("--auto-fix-dirs", action="store_true", help="Create missing asset directories")
    ap.add_argument(
        "--write-env-template", default=None, help="Write missing keys to given env file (e.g., .env.local)"
    )
    ap.add_argument(
        "--require-keys", action="store_true", help="Treat missing provider keys as failure in non-strict mode"
    )
    args = ap.parse_args(argv)

    results: list[CheckResult] = []
    results.append(check_python_env())
    results.append(check_packages())
    results.append(check_gpu())
    env_res = check_env_keys()
    results.append(env_res)
    results.extend(check_network())
    results.extend(check_vector_clients())
    results.append(check_assets())
    results.append(check_guardrails())

    # Optional auto-fix actions
    if args.auto_fix_dirs:
        results.append(_auto_fix_dirs())

    if args.write_env_template:
        if env_res.status == "pass":
            missing_keys: list[str] = [k for k, v in (env_res.data.get("present") or {}).items() if not v]
        else:
            missing_keys = env_res.data.get("missing") or []
        results.append(_write_env_template(Path(args.write_env_template), missing_keys))

    report = PreflightReport(
        timestamp=time.time(),
        system={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        results=results,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.to_json())

    if args.pretty:
        # Compact human summary
        print("AI Preflight Summary")
        for r in results:
            print(f"- {r.name}: {r.status} — {r.detail or ''}")
            if r.suggestion and r.status in {"fail"}:
                print(f"  fix: {r.suggestion}")
        print(f"\nSaved report: {out_path}")

    # Determine failure conditions
    if args.strict:
        has_fail = any(r.status == "fail" for r in results)
        return 1 if has_fail else 0
    else:
        # In non-strict mode, optionally still fail if require-keys flag is set and env keys are missing
        if getattr(args, "require_keys", False) and env_res.status == "fail":
            return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
