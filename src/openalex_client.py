"""
OpenAlex Client (fallback source for literature metadata)

Lightweight wrapper around OpenAlex works API with mapping to the
project's unified article schema.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return default
    return str(v).strip()


def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    if not isinstance(inverted_index, dict) or not inverted_index:
        return ""
    try:
        max_pos = 0
        for positions in inverted_index.values():
            if positions:
                max_pos = max(max_pos, max(positions))
        tokens: List[Optional[str]] = [None] * (max_pos + 1)
        for token, positions in inverted_index.items():
            for pos in positions:
                if 0 <= pos < len(tokens):
                    tokens[pos] = token
        return " ".join([t for t in tokens if isinstance(t, str) and t])
    except Exception as exc:
        logger.debug("Failed to reconstruct abstract: %s", exc)
        return ""


def _extract_pmid(ids: Optional[Dict[str, Any]]) -> str:
    if not isinstance(ids, dict):
        return ""
    pmid_url = ids.get("pmid") or ""
    if not isinstance(pmid_url, str):
        return ""
    import re
    m = re.search(r"(\d{6,9})", pmid_url)
    return m.group(1) if m else ""


def _normalize_authors(authorships: Optional[List[Dict[str, Any]]]) -> str:
    if not isinstance(authorships, list):
        return ""
    names: List[str] = []
    for a in authorships:
        try:
            author = a.get("author") or {}
            name = author.get("display_name") or author.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        except Exception:
            continue
    return ", ".join(names)


def _normalize_journal(work: Dict[str, Any]) -> str:
    host = work.get("host_venue") or {}
    src = work.get("primary_location", {}).get("source") or {}
    for candidate in (host.get("display_name"), src.get("display_name")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _full_text_url(work: Dict[str, Any]) -> str:
    oa = work.get("open_access") or {}
    if isinstance(oa, dict):
        u = oa.get("oa_url")
        if isinstance(u, str) and u.strip():
            return u.strip()
    primary = work.get("primary_location") or {}
    if isinstance(primary, dict):
        u = primary.get("url")
        if isinstance(u, str) and u.strip():
            return u.strip()
    return ""


class OpenAlexClient:
    def __init__(self,
                 base_url: Optional[str] = None,
                 email: Optional[str] = None) -> None:
        self.base_url = (base_url or _env("OPENALEX_BASE_URL", "https://api.openalex.org")).rstrip("/")
        self.email = email or _env("OPENALEX_EMAIL")

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        if self.email and "mailto" not in params:
            params["mailto"] = self.email
        url = self.base_url + path
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def search_works(self, query: str, max_items: int = 30) -> List[Dict[str, Any]]:
        if not query or not str(query).strip():
            return []
        data = self._get("/works", {"search": query.strip(), "per_page": max(1, min(int(max_items), 200))})
        return list(data.get("results") or [])

    def normalize_work(self, work: Dict[str, Any]) -> Dict[str, Any]:
        ids = work.get("ids") or {}
        title = work.get("display_name") or ""
        doi = work.get("doi") or ids.get("doi") or ""
        if isinstance(doi, str) and doi.startswith("https://doi.org/"):
            doi = doi.split("https://doi.org/")[-1]
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        pub_date = work.get("publication_date") or work.get("from_publication_date") or ""
        authors = _normalize_authors(work.get("authorships"))
        pmid = _extract_pmid(ids)
        journal = _normalize_journal(work)
        full_url = _full_text_url(work)

        return {
            "pmid": pmid or "",
            "doi": doi or "",
            "title": title or "",
            "abstract": abstract or "",
            "authors": authors or "",
            "publication_date": pub_date or "",
            "journal": journal or "",
            "full_text_url": full_url or "",
            "provider": "openalex",
            "provider_family": "openalex",
            "ingestion": "openalex",
            "openalex_id": work.get("id"),
        }

