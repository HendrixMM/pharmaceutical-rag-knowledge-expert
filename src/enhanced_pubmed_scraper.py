"""Backward-compatible entry point for PubMed scraping with rate limiting defaults."""
from __future__ import annotations

import os
from typing import Optional

from .pubmed_scraper import PubMedScraper
from .rate_limiting import NCBIRateLimiter


class EnhancedPubMedScraper(PubMedScraper):
    """Compatibility wrapper that enables rate limiting by default."""

    def __init__(
        self,
        *args,
        rate_limiter: Optional[NCBIRateLimiter] = None,
        enable_rate_limiting: Optional[bool] = None,
        **kwargs,
    ) -> None:
        if enable_rate_limiting is None:
            feature_flag_enabled = (
                os.getenv("ENABLE_ENHANCED_PUBMED_SCRAPER", "false").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            enable_rate_limiting = True if feature_flag_enabled else None
        super().__init__(
            *args,
            rate_limiter=rate_limiter,
            enable_rate_limiting=enable_rate_limiting,
            **kwargs,
        )
