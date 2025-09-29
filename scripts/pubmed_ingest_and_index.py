#!/usr/bin/env python3
"""
Fetch a PubMed record (by PMID or URL) via PubMed E-utilities and extract
structured metadata + abstract. Optionally embed and index into local vector DB.

Usage:
  # PubMed API only (default; prints JSON)
  python scripts/pubmed_ingest_and_index.py --pmid 40452389
  python scripts/pubmed_ingest_and_index.py --url https://pubmed.ncbi.nlm.nih.gov/40452389/

  # Include simple related-article PMIDs via ELink (neighbor)
  python scripts/pubmed_ingest_and_index.py --pmid 40452389 --show-related

  # Opt-in: embed + index, then run a quick similarity query (requires NVIDIA_API_KEY)
  python scripts/pubmed_ingest_and_index.py --pmid 40452389 --embed --db ./vector_db
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import json
import requests
from typing import Dict, Any


def extract_pmid_from_url(url: str) -> str:
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?", url)
    if not m:
        raise SystemExit("Could not extract PMID from URL")
    return m.group(1)


def fetch_pubmed_record(pmid: str) -> Dict[str, Any]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    s = requests.get(f"{base}/esummary.fcgi", params={"db": "pubmed", "id": pmid, "retmode": "json"}, timeout=30)
    s.raise_for_status()
    summ = s.json()["result"][pmid]
    x = requests.get(
        f"{base}/efetch.fcgi",
        params={"db": "pubmed", "id": pmid, "rettype": "abstract", "retmode": "xml"},
        timeout=30,
    )
    x.raise_for_status()
    xml = x.text
    abs_text = " ".join(re.findall(r"<AbstractText(?:[^>]*)>(.*?)</AbstractText>", xml, flags=re.S))
    abs_text = re.sub(r"<[^>]+>", " ", abs_text).strip()
    doi = None
    for aid in summ.get("articleids", []):
        if aid.get("idtype") == "doi":
            doi = aid.get("value")
            break
    return {
        "pmid": pmid,
        "title": summ.get("title"),
        "journal": summ.get("fulljournalname"),
        "pubdate": summ.get("pubdate"),
        "authors": [a.get("name") for a in summ.get("authors", []) if a.get("name")],
        "doi": doi,
        "abstract": abs_text,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pmid", help="PubMed PMID")
    p.add_argument("--url", help="PubMed URL")
    p.add_argument("--db", default="./vector_db", help="Vector DB directory (used only with --embed)")
    p.add_argument("--embed", action="store_true", help="Embed + index into local vector DB (requires NVIDIA_API_KEY)")
    p.add_argument("--show-related", action="store_true", help="Show related PMIDs via ELink (PubMed API only)")
    args = p.parse_args()

    if not (args.pmid or args.url):
        print("Provide --pmid or --url", file=sys.stderr)
        return 2

    pmid = args.pmid or extract_pmid_from_url(args.url)
    rec = fetch_pubmed_record(pmid)
    if not rec.get("abstract"):
        print("No abstract found; indexing title only.")

    if not args.embed:
        # PubMed API only: print structured record
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        if args.show_related:
            # ELink to fetch related PMIDs (neighbors)
            el = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
                params={"dbfrom": "pubmed", "id": pmid, "retmode": "json", "cmd": "neighbor"},
                timeout=30,
            )
            try:
                el.raise_for_status()
                data = el.json()
                linksets = data.get("linksets", [])
                related = []
                for ls in linksets:
                    for lk in ls.get("linksetdbs", []) or []:
                        if lk.get("linkname", "").endswith("pubmed_pubmed"):
                            related.extend([d.get("id") for d in lk.get("links", []) if d.get("id")])
                print(json.dumps({"related_pmids": related[:20]}, indent=2))
            except Exception as e:
                print(json.dumps({"related_error": str(e)}))
        return 0

    # Embed + index branch (requires NVIDIA_API_KEY)
    try:
        from langchain_core.documents import Document
    except Exception:
        from langchain.schema import Document  # fallback for older LC versions

    page_content = rec.get("abstract") or rec.get("title") or ""
    metadata = {
        "pmid": rec.get("pmid"),
        "title": rec.get("title"),
        "journal": rec.get("journal"),
        "pubdate": rec.get("pubdate"),
        "authors": rec.get("authors"),
        "doi": rec.get("doi"),
        "url": rec.get("url"),
        "source": "pubmed",
    }
    doc = Document(page_content=page_content, metadata=metadata)

    try:
        from src.nvidia_embeddings import NVIDIAEmbeddings
        from src.vector_database import VectorDatabase
    except ModuleNotFoundError:
        sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
        from src.nvidia_embeddings import NVIDIAEmbeddings  # type: ignore
        from src.vector_database import VectorDatabase  # type: ignore

    emb = NVIDIAEmbeddings(embedding_model_name=os.getenv('EMBEDDING_MODEL_NAME', 'nvidia/nv-embed-v1'))
    vdb = VectorDatabase(embeddings=emb, db_path=args.db)
    ok = vdb.add_documents([doc])
    print("Indexed:", ok)

    # Quick similarity test after indexing
    query = f"insomnia risk with centanafadine vs methylphenidate"
    results = vdb.similarity_search(query, k=3)
    out = [{
        "title": r.metadata.get("title"),
        "pmid": r.metadata.get("pmid"),
        "score": None,
        "snippet": (r.page_content or "")[:240],
    } for r in results]
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
