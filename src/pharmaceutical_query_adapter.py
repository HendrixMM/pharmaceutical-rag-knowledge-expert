"""Helpers for wiring the enhanced pharmaceutical query engine into applications.

Environment variables
---------------------
- APIFY_TOKEN: required by `PubMedScraper` to authenticate with EasyAPI/Apify.
- QUERY_ENGINE_CACHE_DIR: optional override for the on-disk JSON cache directory.
- EASYAPI_ACTOR_ID: optional Apify actor id for the PubMed integration.
"""

from __future__ import annotations

from typing import Any, Optional

from .pubmed_scraper import PubMedScraper
from .query_engine import EnhancedQueryEngine
from .ranking_filter import StudyRankingFilter


def build_pharmaceutical_query_engine(
    *,
    scraper: Optional[PubMedScraper] = None,
    ranking_filter: Optional[StudyRankingFilter] = None,
    **engine_kwargs: Any,
) -> EnhancedQueryEngine:
    """Return a configured `EnhancedQueryEngine` ready for pharmaceutical queries.

    Parameters
    ----------
    scraper:
        Optional pre-configured `PubMedScraper`. When omitted a new instance is
        built using environment variables (see module docstring).
    ranking_filter:
        Optional custom `StudyRankingFilter`. Defaults to a new instance with
        standard weights.
    engine_kwargs:
        Additional keyword arguments forwarded to the `EnhancedQueryEngine`
        constructor (e.g. `cache_dir`, `cache_ttl_hours`).
    """

    scraper = scraper or PubMedScraper()
    ranking_filter = ranking_filter or StudyRankingFilter()
    return EnhancedQueryEngine(scraper, ranking_filter=ranking_filter, **engine_kwargs)


__all__ = ["build_pharmaceutical_query_engine"]
