"""
PubMed Scraper using Apify
Handles scraping PubMed search results with caching and deduplication.

Note: The scraper now preserves native PubMed ordering by default. Pass
``rank=True`` or set ``ENABLE_STUDY_RANKING=true`` to enable study ranking.

Usage:
  python -m src.pubmed_scraper "query"
  python src/pubmed_scraper.py "query"
Set APIFY_TOKEN environment variable before running.
"""

import os
import sys
import json
import time
import hashlib
import logging
import re
import argparse
import urllib.parse
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union, Set, Literal
from pathlib import Path

try:
    from apify_client import ApifyClient
except ImportError:
    ApifyClient = None

try:
    from apify_client import ApifyApiError  # type: ignore[attr-defined]
except ImportError:
    try:
        from apify_client._errors import ApifyApiError  # type: ignore[attr-defined]
    except ImportError:
        ApifyApiError = None

from langchain.schema import Document


def _env_true(name: str, default: bool = True) -> bool:
    """Interpret boolean-like environment flags consistently."""
    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    return default

try:
    from .rate_limiting import (
        DailyQuotaExceeded,
        NCBIRateLimiter,
        RateLimitStatus,
        get_process_limiter,
    )
except ImportError:  # pragma: no cover - allows use without optional module during packaging
    NCBIRateLimiter = None  # type: ignore
    RateLimitStatus = None  # type: ignore
    DailyQuotaExceeded = None  # type: ignore
    get_process_limiter = None  # type: ignore

# Set up logging
logger = logging.getLogger(__name__)

# Cache schema version for coordinated invalidations
CACHE_SCHEMA_VERSION = '3'

# Terms appended when pharmaceutical enhancement triggers
PHARMACEUTICAL_ENHANCEMENT_TERMS = [
    'drug interaction',
    'pharmacokinetics',
    'pharmacodynamics',
    'CYP2C9',
    'CYP3A4',
    'CYP2D6',
    'drug metabolism',
    'adverse effects'
]

# Substrings that indicate the query is already focused on pharmacology topics
PHARMACEUTICAL_SIGNAL_KEYWORDS = {
    'drug interaction',
    'drug interactions',
    'drug-drug',
    'cyp',
    'metabol',
    'pharmacokinetic',
    'pharmacokinetics',
    'pharmacodynamic',
    'pharmacodynamics',
    'pharmacology',
    'dose',
    'dosing',
    'toxicity',
    'adverse effect',
    'adverse effects',
}

# Lightweight lexicon of common drug names to reduce false positives without external dependencies
KNOWN_DRUG_LEXICON: Set[str] = {
    'acetaminophen',
    'amoxicillin',
    'atorvastatin',
    'clopidogrel',
    'ibuprofen',
    'levothyroxine',
    'lisinopril',
    'metformin',
    'omeprazole',
    'simvastatin',
    'warfarin',
}


class PubMedAccessError(RuntimeError):
    """Raised when Apify indicates the PubMed actor requires elevated access."""


SUBSCRIPTION_ERROR_MESSAGE = (
    "EasyAPI actor may require an active subscription or access. Verify APIFY_TOKEN and EasyAPI plan."
)


class PubMedScraper:
    """PubMed scraper using Apify with caching and deduplication"""

    class PubMedQuotaExceeded(RuntimeError):
        """Raised when the shared PubMed daily quota is exhausted."""

    class SchemaValidationError(RuntimeError):
        """Raised when the actor rejects a request due to schema validation issues."""

    def __init__(
        self,
        apify_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
        rate_limiter: Optional["NCBIRateLimiter"] = None,
        enable_rate_limiting: Optional[bool] = None,
    ):
        """
        Initialize PubMed scraper

        Args:
            apify_token: Apify API token
            cache_dir: Directory for caching results
            cache_ttl_seconds: Cache TTL in seconds (default: read from env or 24 hours)
            rate_limiter: Optional NCBI rate limiter instance. When omitted and
                rate limiting is enabled via environment flags, a limiter is
                created automatically using `NCBIRateLimiter.from_env()`.
            enable_rate_limiting: Optional override for enabling rate limiting
                regardless of environment configuration.

        For multi-instance deployments, pass a shared `NCBIRateLimiter` (for
        example the process-wide limiter returned by
        `src.rate_limiting.get_process_limiter()`) so every scraper in the
        process adheres to a unified budget.
        """
        self.apify_token = apify_token or os.getenv("APIFY_TOKEN")
        if not self.apify_token:
            raise ValueError("Apify token is required. Set APIFY_TOKEN environment variable.")

        if not ApifyClient:
            raise ImportError("apify_client is required. Install with: pip install apify-client>=1.6.0,<2.0.0")

        self.client = ApifyClient(self.apify_token)
        self.actor_id = os.getenv("EASYAPI_ACTOR_ID", "easyapi/pubmed-search-scraper")
        self.cache_dir = Path(cache_dir or os.getenv("PUBMED_CACHE_DIR", "./pubmed_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration from environment
        self.default_max_items = int(os.getenv("DEFAULT_MAX_ITEMS", "30"))
        self.hard_cap_max_items = int(os.getenv("HARD_CAP_MAX_ITEMS", "100"))
        self.enable_study_ranking = os.getenv("ENABLE_STUDY_RANKING", "false").lower() == "true"
        if self.enable_study_ranking:
            logger.info(
                "Study ranking enabled; set ENABLE_STUDY_RANKING=false or PRESERVE_PUBMED_ORDER=true to retain raw PubMed ordering."
            )
        # Support both new and old environment variable names for backward compatibility
        enable_dedup = os.getenv("ENABLE_DEDUPLICATION")
        legacy_dedup_env = os.getenv("ENABLE_PMID_DEDUPLICATION")
        if enable_dedup is None:
            # Fallback to old variable name for backward compatibility
            enable_dedup = legacy_dedup_env or "true"
        if legacy_dedup_env is not None:
            logger.warning(
                "ENABLE_PMID_DEDUPLICATION is deprecated; use ENABLE_DEDUPLICATION instead."
            )
        self.enable_deduplication = enable_dedup.lower() == "true"
        self.use_full_abstracts = os.getenv("USE_FULL_ABSTRACTS", "true").lower() == "true"
        self.extract_tags = os.getenv("EXTRACT_TAGS", "true").lower() == "true"
        self.enable_pharma_query_enhancement = os.getenv("ENABLE_PHARMA_QUERY_ENHANCEMENT", "true").lower() == "true"
        self.pharma_terms: List[str] = list(PHARMACEUTICAL_ENHANCEMENT_TERMS)
        extra_terms_env = os.getenv("PHARMA_EXTRA_TERMS", "")
        if extra_terms_env:
            extra_terms = [term.strip() for term in extra_terms_env.split(',') if term.strip()]
            if extra_terms:
                existing_lower = {term.lower(): term for term in self.pharma_terms}
                for term in extra_terms:
                    term_lower = term.lower()
                    if term_lower not in existing_lower:
                        self.pharma_terms.append(term)
                        existing_lower[term_lower] = term
        mesh_qualifiers_env = os.getenv(
            "PHARMA_MESH_QUALIFIERS",
            "drug interactions[MeSH Terms],Cytochrome P-450 Enzyme System[MeSH]",
        )
        self.pharma_mesh_qualifiers: List[str] = []
        if mesh_qualifiers_env:
            seen_mesh = set()
            for qualifier in [entry.strip() for entry in mesh_qualifiers_env.split(',') if entry.strip()]:
                qualifier_lower = qualifier.lower()
                if qualifier_lower not in seen_mesh:
                    self.pharma_mesh_qualifiers.append(qualifier)
                    seen_mesh.add(qualifier_lower)
        self.pharma_max_terms = int(os.getenv("PHARMA_MAX_TERMS", "8"))
        self.max_query_length = int(os.getenv("MAX_QUERY_LENGTH", "1800"))
        self.enable_schema_fallback = os.getenv("ENABLE_EASYAPI_SCHEMA_FALLBACK", "false").lower() == "true"
        self.scraper_provider = os.getenv("SCRAPER_PROVIDER", "apify-easyapi")
        self.smart_schema_fallback = _env_true("EASYAPI_SMART_SCHEMA_FALLBACK", True)
        fallback_schemas_env = os.getenv("EASYAPI_FALLBACK_SCHEMAS", "")
        raw_fallback_schemas = [schema.strip() for schema in fallback_schemas_env.split(',') if schema.strip()]
        supported_schemas = {'searchUrls'}
        invalid_schemas = [schema for schema in raw_fallback_schemas if schema not in supported_schemas]
        if invalid_schemas:
            logger.warning(
                "Ignoring unsupported fallback schemas %s (see https://apify.com/easyapi/pubmed-search-scraper#input)",
                invalid_schemas,
            )
        self.easyapi_fallback_schemas = [schema for schema in raw_fallback_schemas if schema in supported_schemas]
        logger.info(
            "EasyAPI fallback schemas: %s",
            ', '.join(self.easyapi_fallback_schemas) if self.easyapi_fallback_schemas else 'none'
        )
        if not self.enable_schema_fallback:
            logger.info("EasyAPI schema fallback disabled; only searchUrls will be used.")
        monthly_budget_limit = os.getenv("MONTHLY_BUDGET_LIMIT")
        if monthly_budget_limit:
            logger.info(f"Monthly budget limit configured: ${monthly_budget_limit}")

        # Set cache TTL from parameter, environment, or default (24 hours)
        if cache_ttl_seconds is not None:
            self.cache_ttl = cache_ttl_seconds
        else:
            env_ttl_hours = os.getenv("PUBMED_CACHE_TTL_HOURS")
            if env_ttl_hours:
                try:
                    self.cache_ttl = int(float(env_ttl_hours) * 3600)  # Convert hours to seconds
                    logger.info(f"Using cache TTL from environment: {env_ttl_hours} hours ({self.cache_ttl} seconds)")
                except ValueError:
                    logger.warning(f"Invalid PUBMED_CACHE_TTL_HOURS '{env_ttl_hours}', using default: 24 hours")
                    self.cache_ttl = 24 * 60 * 60
            else:
                self.cache_ttl = 24 * 60 * 60  # Default: 24 hours

        cache_empty_env = os.getenv("CACHE_EMPTY_RESULTS", "false")
        self.cache_empty_results = cache_empty_env.strip().lower() in {"1", "true", "yes", "on"}

        empty_cache_ttl_env = os.getenv("EMPTY_CACHE_TTL_SECONDS")
        self.empty_cache_ttl = self.cache_ttl
        if self.cache_empty_results:
            default_empty_ttl = min(self.cache_ttl, 300)
            parsed_empty_ttl = default_empty_ttl
            if empty_cache_ttl_env:
                try:
                    parsed_empty_ttl = int(float(empty_cache_ttl_env))
                except ValueError:
                    logger.warning(
                        f"Invalid EMPTY_CACHE_TTL_SECONDS '{empty_cache_ttl_env}', falling back to {default_empty_ttl} seconds"
                    )
                    parsed_empty_ttl = default_empty_ttl

            if parsed_empty_ttl <= 0:
                logger.warning(
                    f"EMPTY_CACHE_TTL_SECONDS '{parsed_empty_ttl}' must be positive; falling back to {default_empty_ttl} seconds"
                )
                parsed_empty_ttl = default_empty_ttl

            self.empty_cache_ttl = min(parsed_empty_ttl, self.cache_ttl)


        if enable_rate_limiting is None:
            rate_limit_env = os.getenv("ENABLE_RATE_LIMITING", "false")
            rate_limiting_requested = rate_limit_env.strip().lower() in {"1", "true", "yes", "on"}
        else:
            rate_limiting_requested = enable_rate_limiting

        self.rate_limiter = rate_limiter
        if rate_limiter is not None and enable_rate_limiting is None:
            self._rate_limiting_requested = True
        else:
            self._rate_limiting_requested = bool(rate_limiting_requested)
        use_process_wide_env = os.getenv("USE_PROCESS_WIDE_LIMITER", "false")
        self._use_process_wide_limiter = use_process_wide_env.strip().lower() in {"1", "true", "yes", "on"}
        self._rate_limiting_enabled = bool(self._rate_limiting_requested)

        def _parse_optional_bool(name: str) -> Optional[bool]:
            value = os.getenv(name)
            if value is None:
                return None
            return value.strip().lower() in {"1", "true", "yes", "on"}

        self._rate_limit_raise_on_daily_limit = _parse_optional_bool("RATE_LIMIT_RAISE_ON_DAILY_LIMIT")

        max_wait_env = os.getenv("RATE_LIMIT_MAX_DAILY_WAIT_SECONDS")
        if max_wait_env is None or max_wait_env == "":
            self._rate_limit_max_wait_seconds = None
        else:
            try:
                parsed_wait = int(max_wait_env)
                if parsed_wait < 0:
                    raise ValueError
                self._rate_limit_max_wait_seconds = parsed_wait
            except ValueError:
                logger.warning(
                    "RATE_LIMIT_MAX_DAILY_WAIT_SECONDS must be a non-negative integer; ignoring provided value '%s'",
                    max_wait_env,
                )
                self._rate_limit_max_wait_seconds = None

        if self._rate_limiting_enabled and self.rate_limiter is None:
            if NCBIRateLimiter is None:
                raise ImportError(
                    "NCBIRateLimiter is required for rate limiting but could not be imported."
                )
            if self._use_process_wide_limiter and get_process_limiter is not None:
                self.rate_limiter = get_process_limiter()
            else:
                self.rate_limiter = NCBIRateLimiter.from_env()

        if self.rate_limiter and not self._rate_limiting_enabled:
            logger.info("Rate limiter provided but rate limiting disabled via configuration; limiter will be inactive.")

        if self.rate_limiter and self._rate_limiting_enabled:
            logger.info(
                "NCBI rate limiting enabled (max %s req/sec, daily limit=%s)",
                getattr(self.rate_limiter, "max_requests_per_second", "?"),
                getattr(self.rate_limiter, "daily_request_limit", "?"),
            )

        logger.info(f"Initialized PubMed scraper with cache dir: {self.cache_dir}")
        logger.info(
            "PubMed scraper configured (cache_dir=%s, ttl_seconds=%s, actor=%s, default_max_items=%s)",
            self.cache_dir,
            self.cache_ttl,
            self.actor_id,
            self.default_max_items,
        )

        # Verify ApifyApiError import compatibility for error handling
        if ApifyApiError is None:
            logger.warning("ApifyApiError could not be imported; HTTP status code error handling may be limited")
        else:
            # Lightweight self-check: ensure we can create a mock error with status_code attribute
            try:
                mock_error = type('MockApifyApiError', (ApifyApiError,), {'status_code': 429})()
                if not hasattr(mock_error, 'status_code'):
                    logger.warning("ApifyApiError compatibility issue: status_code attribute not accessible")
            except Exception as e:
                logger.warning(f"ApifyApiError compatibility check failed: {str(e)}")
            else:
                logger.debug("ApifyApiError import compatibility verified")

    def _rate_limit_active(self) -> bool:
        return bool(self.rate_limiter and self._rate_limiting_enabled)

    def _call_actor(self, run_input: Dict, schema: Optional[str] = None) -> Dict:
        if self._rate_limit_active():
            acquire_kwargs = {}
            if self._rate_limit_raise_on_daily_limit is not None:
                acquire_kwargs["raise_on_daily_limit"] = self._rate_limit_raise_on_daily_limit
            if self._rate_limit_max_wait_seconds is not None:
                acquire_kwargs["max_wait_seconds"] = self._rate_limit_max_wait_seconds
            try:
                self.rate_limiter.acquire(**acquire_kwargs)
            except DailyQuotaExceeded as exc:  # type: ignore[arg-type]
                reset_seconds = int(self.rate_limiter.seconds_until_daily_reset()) if self.rate_limiter else 0
                raise PubMedScraper.PubMedQuotaExceeded(
                    "Daily PubMed quota reached. Try again after the reset window (~"
                    f"{reset_seconds}s remaining) or adjust RATE_LIMIT_RAISE_ON_DAILY_LIMIT / "
                    "RATE_LIMIT_MAX_DAILY_WAIT_SECONDS to change behaviour."
                ) from exc
        try:
            # Call the EasyAPI actor and return the run object
            actor_client = self.client.actor(self.actor_id)
            run = actor_client.call(run_input)
            if not run or not run.get("defaultDatasetId"):
                run_repr = list(run.keys()) if isinstance(run, dict) else run
                logger.warning(
                    "Actor response missing defaultDatasetId (schema=%s, run keys=%s)",
                    schema or 'unknown',
                    run_repr,
                )
                try_optional_retry = _env_true("EASYAPI_RETRY_ON_MISSING_DATASET", True)
                if try_optional_retry:
                    optional_flags = ['includeTags', 'includeAbstract']
                    retry_input = {k: v for k, v in run_input.items() if k not in optional_flags}
                    if retry_input != run_input:
                        logger.debug("Retrying actor call without optional flags after missing datasetId")
                        retry_run = actor_client.call(retry_input)
                        if retry_run and retry_run.get("defaultDatasetId"):
                            return retry_run
                raise RuntimeError(
                    f"Actor run missing defaultDatasetId (actor={self.actor_id}, schema={schema or 'unknown'}, run_keys={run_repr})."
                )
            return run
        except Exception as e:
            # Check for validation errors due to unknown actor parameters
            if (ApifyApiError and isinstance(e, ApifyApiError) and
                hasattr(e, 'status_code') and 400 <= e.status_code < 500):
                # Check if error message suggests parameter validation issues
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['parameter', 'unknown', 'invalid', 'validation']):
                    optional_flags = ['includeTags', 'includeAbstract']
                    offending_flags = [flag for flag in optional_flags if flag in run_input]

                    if offending_flags:
                        logger.warning(
                            "Actor parameter validation error, retrying without optional flags (%s). Error: %s",
                            offending_flags,
                            str(e),
                        )
                        # Create new input without optional flags
                        retry_input = {k: v for k, v in run_input.items() if k not in optional_flags}
                        retry_run = actor_client.call(retry_input)
                        if retry_run and retry_run.get("defaultDatasetId"):
                            return retry_run
                        logger.warning("Retry without optional flags still missing defaultDatasetId or failed")

                    if self.smart_schema_fallback:
                        raise PubMedScraper.SchemaValidationError(
                            f"Actor validation error for schema '{schema or 'unknown'}': {str(e)}"
                        ) from e

            # Re-raise the original exception if not a validation error or retry failed
            raise

    def get_rate_limit_status(self) -> Optional["RateLimitStatus"]:
        if not self._rate_limit_active() or RateLimitStatus is None:
            return None
        return self.rate_limiter.get_status()

    def get_rate_limit_report(self) -> Optional[Dict[str, object]]:
        if not self._rate_limit_active():
            return None
        return self.rate_limiter.get_compliance_report()

    def is_optimal_timing(self) -> Optional[bool]:
        if not self._rate_limit_active():
            return None
        return self.rate_limiter.is_optimal_timing()

    def seconds_until_next_request(self) -> Optional[float]:
        if not self._rate_limit_active():
            return None
        return self.rate_limiter.seconds_until_next_request()

    def remaining_daily_requests(self) -> Optional[int]:
        if not self._rate_limit_active():
            return None
        return self.rate_limiter.remaining_daily_requests()

    def _build_actor_input(
        self,
        enhanced_query: str,
        max_items: int,
        schema: Literal['searchUrls'],
        include_tags: Optional[bool] = None,
        include_abstract: Optional[bool] = None,
        force_include_tags: bool = False,
        force_include_abstract: bool = False,
    ) -> dict:
        """Build actor input dictionary for EasyAPI (searchUrls only)."""
        encoded_query = urllib.parse.quote_plus(enhanced_query)
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={encoded_query}"
        run_input: Dict[str, object] = {
            "searchUrls": [pubmed_url],
            "maxItems": max_items,
        }
        include_tags_gate = force_include_tags or _env_true("EASYAPI_INCLUDE_TAGS", True)
        include_abstract_gate = force_include_abstract or _env_true("EASYAPI_INCLUDE_ABSTRACT", True)

        if include_tags is not None and include_tags_gate:
            run_input["includeTags"] = include_tags
        if include_abstract is not None and include_abstract_gate:
            run_input["includeAbstract"] = include_abstract
        return run_input

    def _get_cache_key(
        self,
        enhanced_query: str,
        max_items: int,
        apply_ranking: bool,
        pharma_enhance_enabled: bool,
        include_tags_effective: bool,
        include_abstract_effective: bool,
    ) -> str:
        """Generate cache key for enhanced query"""
        content = (
            f"{enhanced_query}:{max_items}:actor={self.actor_id}:"
            f"tags={int(include_tags_effective)}:"
            f"abstract={int(include_abstract_effective)}:"
            f"enhance={int(pharma_enhance_enabled)}:rank={int(apply_ranking)}:"
            f"dedup={int(self.enable_deduplication)}:"
            f"pharmaMax={self.pharma_max_terms}:"
            f"v={CACHE_SCHEMA_VERSION}"
        )
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_results(self, cache_key: str, apply_ranking: bool) -> Optional[List[Dict]]:
        """Get cached results if they exist and are not expired"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            # Check if cache is expired
            cache_time = cache_file.stat().st_mtime
            if time.time() - cache_time > self.cache_ttl:
                logger.info(f"Cache expired for key: {cache_key}")
                cache_file.unlink()  # Remove expired cache
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            # Check if ranking recomputation is needed
            if apply_ranking and cached_data:
                # Check if any results lack ranking_score
                needs_ranking = any('ranking_score' not in result for result in cached_data)
                if needs_ranking:
                    logger.info("Recomputing ranking for cached results missing ranking scores")
                    for result in cached_data:
                        if 'ranking_score' not in result:
                            self._apply_study_ranking(result)
                    # Sort by ranking score after recomputation
                    cached_data.sort(key=lambda r: r.get('ranking_score', 0), reverse=True)
                    # Persist updated scores back to cache
                    self._cache_results(cache_key, cached_data)

            logger.info(f"Using cached results for key: {cache_key}")
            return cached_data

        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            return None

    def _cache_results(self, cache_key: str, results: List[Dict]) -> None:
        """Cache results to disk"""
        if not results and not self.cache_empty_results:
            return
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            if not results and self.cache_empty_results and self.empty_cache_ttl < self.cache_ttl:
                # Age empty-cache files so they expire after a shorter TTL window
                expire_offset = self.cache_ttl - self.empty_cache_ttl
                aged_timestamp = time.time() - expire_offset
                os.utime(cache_file, (aged_timestamp, aged_timestamp))

            logger.info(f"Cached {len(results)} results with key: {cache_key}")

        except Exception as e:
            logger.warning(f"Failed to cache results: {str(e)}")

    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI by removing prefixes and converting to lowercase"""
        if not doi:
            return ""

        # Convert to lowercase and strip whitespace
        normalized = doi.lower().strip()

        # Strip common prefixes (case-insensitive, including trailing slash variants)
        prefixes_to_strip = [
            'doi:',
            'https://doi.org/',
            'https://doi.org',
            'http://doi.org/',
            'http://doi.org',
            'doi.org/',
            'doi.org',
            'www.doi.org/',
            'www.doi.org'
        ]

        for prefix in prefixes_to_strip:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                # Remove leading slash if it remains after prefix removal
                if normalized.startswith('/'):
                    normalized = normalized[1:]
                break

        # Strip trailing punctuation
        normalized = normalized.rstrip('.,;)')

        return normalized.strip()

    def _deduplicate_results(self, items: List[Dict]) -> List[Dict]:
        if not self.enable_deduplication or not items:
            return items or []
        by_doi, by_pmid, by_title = {}, {}, {}
        out = []
        for it in items:
            doi = self._normalize_doi(str(it.get('doi','')))
            pmid = str(it.get('pmid','')).strip()
            title_key = (it.get('title') or '').strip().lower()
            if doi:
                if doi in by_doi: continue
                by_doi[doi] = True
            elif pmid:
                if pmid in by_pmid: continue
                by_pmid[pmid] = True
            elif title_key:
                if title_key in by_title: continue
                by_title[title_key] = True
            out.append(it)
        return out

    def _enhance_pharmaceutical_query(self, query: str, enhancement_enabled: bool) -> str:
        """Enhance pharmaceutical queries with relevant synonyms and keywords."""
        if not enhancement_enabled:
            return query

        allow_advanced = _env_true("ENABLE_PHARMA_ENHANCE_ADVANCED", False)

        # Detect advanced PubMed query features that require explicit opt-in.
        has_field_qualifier = bool(re.search(r'\[[^\]]+\]', query))
        has_slash_qualifier = bool(re.search(r'\[[^\]]*/[^\]]*\]', query))
        has_mesh_qualifier = bool(re.search(r'\[\s*MeSH[^\]]*\]', query, re.IGNORECASE))

        # Track nested parentheses depth to flag complex boolean queries.
        depth = 0
        max_depth = 0
        for char in query:
            if char == '(':
                depth += 1
                max_depth = max(max_depth, depth)
            elif char == ')':
                depth -= 1
        has_excessive_nesting = max_depth > 2

        has_quoted_phrase = '"' in query

        advanced_features_present = (
            has_field_qualifier
            or has_slash_qualifier
            or has_mesh_qualifier
            or has_excessive_nesting
            or has_quoted_phrase
        )

        if advanced_features_present and not allow_advanced:
            # Respect opt-out for complex PubMed queries (field qualifiers, deep nesting, quoted phrases).
            logger.debug("Skipping pharma enhancement: advanced query features detected and ENABLE_PHARMA_ENHANCE_ADVANCED is false")
            return query

        query_lower = query.lower()

        # Apply synonym mapping for CYP and DDI terms
        synonym_map = {
            'cyp450': 'cytochrome p450',
            'cyp-450': 'cytochrome p450',
            'cytochrome p-450': 'cytochrome p450',
        }

        # Apply synonym normalization to query
        enhanced_query = query
        for synonym, canonical in synonym_map.items():
            if synonym in query_lower and canonical not in query_lower:
                # Case-preserving replacement
                enhanced_query = re.sub(re.escape(synonym), canonical, enhanced_query, flags=re.IGNORECASE)
                query_lower = enhanced_query.lower()  # Update for subsequent checks

        # Require clear pharmacology signals before extending the query.
        interaction_signals = tuple(signal for signal in PHARMACEUTICAL_SIGNAL_KEYWORDS if 'interaction' in signal)
        non_interaction_signals = tuple(
            signal for signal in PHARMACEUTICAL_SIGNAL_KEYWORDS if 'interaction' not in signal
        )

        interaction_signal_hit = any(signal in query_lower for signal in interaction_signals)
        other_signal_hit = any(signal in query_lower for signal in non_interaction_signals)

        tokens = re.split(r'[^a-z0-9]+', query_lower)
        token_set = {token for token in tokens if token}
        lexicon_hit = any(token in KNOWN_DRUG_LEXICON for token in token_set)

        interaction_anchor_hit = (
            'interaction' in query_lower and
            any(anchor in query_lower for anchor in ('drug', 'pharmaco', 'cyp'))
        )

        has_signal = other_signal_hit or lexicon_hit or interaction_signal_hit or interaction_anchor_hit

        if not has_signal:
            logger.debug("Skipping pharma enhancement: no pharmacology signal detected in query '%s'", query)
            return enhanced_query

        enhanced_terms: List[str] = []
        existing_terms_lower = {term.lower() for term in self.pharma_terms if term.lower() in query_lower}

        # Add 'drug-drug interaction' if not already present
        if ('drug-drug interaction' not in query_lower) and ('drug interaction' not in query_lower):
            enhanced_terms.append('drug-drug interaction')
            existing_terms_lower.add('drug-drug interaction')

        for term in self.pharma_terms:
            term_lower = term.lower()
            if term_lower not in existing_terms_lower and term_lower not in query_lower:
                enhanced_terms.append(term)
                existing_terms_lower.add(term_lower)

        mesh_qualifiers = self.pharma_mesh_qualifiers
        for qualifier in mesh_qualifiers:
            qualifier_lower = qualifier.lower()
            if qualifier_lower not in existing_terms_lower and qualifier_lower not in query_lower:
                enhanced_terms.append(qualifier)
                existing_terms_lower.add(qualifier_lower)

        if enhanced_terms:
            # Prioritize key pharmacology signals before enforcing a maximum list size.
            priority_terms = []
            for index, term in enumerate(enhanced_terms):
                term_lower = term.lower()
                if term_lower == 'drug-drug interaction':
                    priority = 0
                elif 'cyp' in term_lower:
                    priority = 1
                elif ('pharmacokinet' in term_lower) or ('pharmacodynam' in term_lower):
                    priority = 2
                else:
                    priority = 3
                priority_terms.append((priority, index, term))

            priority_terms.sort(key=lambda entry: (entry[0], entry[1]))

            max_terms = self.pharma_max_terms
            if max_terms <= 0:
                max_terms = len(priority_terms)

            limited_terms = [term for _, _, term in priority_terms[:max_terms]]

            if len(priority_terms) > max_terms:
                truncated_terms = [term for _, _, term in priority_terms[max_terms:]]
                logger.info(
                    "Truncated enhanced query terms to %d items (omitted: %s)",
                    max_terms,
                    ', '.join(truncated_terms),
                )

            enhanced_terms = limited_terms

            final_enhanced_query = f"({enhanced_query}) OR ({' OR '.join(enhanced_terms)})"

            # Guard against overly long URLs by checking total URL length
            full_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={urllib.parse.quote_plus(final_enhanced_query)}"
            if len(full_url) > self.max_query_length:
                logger.warning(
                    "Enhanced query URL length (%d) exceeds limit (%d), using original query",
                    len(full_url),
                    self.max_query_length,
                )
                logger.debug(
                    "Enhanced query skipped due to URL length: length=%d threshold=%d query='%s'",
                    len(full_url),
                    self.max_query_length,
                    final_enhanced_query,
                )
                return enhanced_query

            return final_enhanced_query

        return enhanced_query

    def _classify_study_type(self, tags: Optional[List[str]], mesh_terms: Optional[List[str]] = None) -> Tuple[str, float]:
        """Classify study type based on tags and return type with confidence score

        Args:
            tags: Optional list of study tags (can be None or empty list)
            mesh_terms: Optional list of MeSH terms to use as fallback when tags is empty

        Returns:
            Tuple of (study_type, confidence_score)
        """
        # MeSH term to study type mapping for fallback
        mesh_study_mapping = {
            'randomized controlled trials as topic': ('RCT', 0.85),
            'randomized controlled trial': ('RCT', 0.85),
            'systematic reviews as topic': ('Systematic Review', 0.8),
            'systematic review': ('Systematic Review', 0.8),
            'meta-analysis as topic': ('Meta-Analysis', 0.8),
            'meta-analysis': ('Meta-Analysis', 0.8),
            'clinical trials as topic': ('Clinical Trial', 0.75),
            'clinical trial': ('Clinical Trial', 0.75),
            'cohort studies': ('Cohort Study', 0.7),
            'case-control studies': ('Case-Control Study', 0.65),
            'observational study': ('Observational Study', 0.6),
            'cross-sectional studies': ('Cross-Sectional Study', 0.6),
            'review literature as topic': ('Review', 0.55),
            'pilot projects': ('Pilot Study', 0.45),
            'case reports': ('Case Report', 0.45),
        }

        # Handle missing or empty tags gracefully
        if not tags:
            tags = []
        tags_lower = [tag.lower() for tag in tags]
        tags_str = ' '.join(tags_lower)

        # Check for specific study types in order of priority
        if any('randomized controlled trial' in tag for tag in tags_lower) or 'rct' in tags_str:
            return ('RCT', 0.9)
        elif any('systematic review' in tag for tag in tags_lower):
            return ('Systematic Review', 0.85)
        elif any('meta-analysis' in tag for tag in tags_lower):
            return ('Meta-Analysis', 0.85)
        elif any('clinical trial' in tag for tag in tags_lower):
            return ('Clinical Trial', 0.8)
        elif any('cohort study' in tag for tag in tags_lower):
            return ('Cohort Study', 0.75)
        elif any('case-control study' in tag for tag in tags_lower):
            return ('Case-Control Study', 0.7)
        elif any('observational study' in tag for tag in tags_lower):
            return ('Observational Study', 0.6)
        elif any('cross-sectional' in tag for tag in tags_lower):
            return ('Cross-Sectional Study', 0.6)
        review_exclusions = {'peer review', 'peer-review', 'reviewer', 'reviewers', 'reviewing'}

        def _is_review_tag(tag: str) -> bool:
            stripped = tag.strip()
            if not stripped:
                return False
            if any(exclusion in stripped for exclusion in review_exclusions):
                return False
            if stripped == 'review' or stripped.endswith(' review') or stripped.endswith('-review'):
                return True
            return False

        if any(_is_review_tag(tag) for tag in tags_lower):
            return ('Review', 0.6)
        elif any('pilot study' in tag for tag in tags_lower):
            return ('Pilot Study', 0.5)
        elif any('case report' in tag for tag in tags_lower):
            return ('Case Report', 0.5)

        # Fallback to MeSH terms if tags didn't yield a classification and mesh_terms are available
        if mesh_terms:
            mesh_lower = [term.lower() for term in mesh_terms]
            for mesh_term in mesh_lower:
                if mesh_term in mesh_study_mapping:
                    return mesh_study_mapping[mesh_term]

            # Check for partial matches in MeSH terms
            for mesh_term in mesh_lower:
                if 'randomized' in mesh_term and 'trial' in mesh_term:
                    return ('RCT', 0.75)
                elif 'systematic' in mesh_term and 'review' in mesh_term:
                    return ('Systematic Review', 0.7)
                elif 'meta-analysis' in mesh_term or 'meta analysis' in mesh_term:
                    return ('Meta-Analysis', 0.7)
                elif 'clinical' in mesh_term and 'trial' in mesh_term:
                    return ('Clinical Trial', 0.65)
                elif 'cohort' in mesh_term:
                    return ('Cohort Study', 0.6)
                elif 'case-control' in mesh_term or 'case control' in mesh_term:
                    return ('Case-Control Study', 0.55)
                elif 'observational' in mesh_term and 'study' in mesh_term:
                    return ('Observational Study', 0.55)
                elif 'cross-sectional' in mesh_term or 'cross sectional' in mesh_term:
                    return ('Cross-Sectional Study', 0.55)
                elif 'pilot' in mesh_term and ('study' in mesh_term or 'project' in mesh_term):
                    return ('Pilot Study', 0.4)
                elif 'review' in mesh_term:
                    return ('Review', 0.5)
                elif 'case report' in mesh_term:
                    return ('Case Report', 0.4)

        return ('Other', 0.4)

    def _apply_study_ranking(self, paper: Dict) -> Dict:
        """Apply study ranking based on multiple factors"""
        base_score = 0.5

        # Study type score
        tags = paper.get('tags', [])
        mesh_terms = paper.get('mesh_terms', [])
        study_type, type_score = self._classify_study_type(tags, mesh_terms)
        paper['study_type'] = study_type
        paper['study_type_confidence'] = type_score
        paper['study_types'] = [study_type]
        base_score = type_score

        # Recency bonus (from year)
        year = paper.get('year')
        if year:
            try:
                year_int = int(year)
                current_year = datetime.now().year
                years_old = max(0, current_year - year_int)
                recency_bonus = max(0, 0.2 - (years_old * 0.01))  # Decrease by 1% per year
                base_score += recency_bonus
            except ValueError:
                pass

        # Abstract length bonus (longer abstracts often indicate more comprehensive studies)
        abstract = paper.get('abstract', '')
        if abstract and len(abstract) > 500:
            base_score += 0.1

        # Pharmaceutical keyword bonus
        content_text = f"{paper.get('title', '')} {abstract}".lower()
        pharma_matches = 0

        # Specific terms - use exact substring matching
        specific_keywords = [
            'drug interaction',
            'pharmacokinetics',
            'metabolism',
            'adverse event',
            'adverse drug reaction',
            'cyp2c9',
            'cyp3a4',
            'cyp2d6',
        ]
        for keyword in specific_keywords:
            if keyword in content_text:
                pharma_matches += 1

        # Generic terms - use word boundary regex to avoid false positives
        cyp_pattern = re.compile(r'\bcyp(?:\d+[a-z0-9]*)?\b', re.IGNORECASE)
        if cyp_pattern.search(content_text):
            pharma_matches += 1

        adverse_pattern = re.compile(
            r'\badverse\s+(event|events|drug\s+reaction|drug\s+reactions)\b',
            re.IGNORECASE,
        )
        if adverse_pattern.search(content_text):
            pharma_matches += 1
        base_score += pharma_matches * 0.05

        paper['ranking_score'] = min(1.0, base_score)  # Cap at 1.0
        return paper

    def _extract_doi_from_citation(self, citation_str: str) -> Optional[str]:
        """Extract DOI from citation string"""
        if not citation_str:
            return None

        # Look for DOI pattern in citation
        doi_pattern = r'(?:doi[:\s]*)?10\.\d{4,9}/[^\s,;)]+'
        match = re.search(doi_pattern, citation_str, re.IGNORECASE)
        if match:
            doi = match.group().replace('doi:', '').replace('DOI:', '').strip()
            return self._normalize_doi(doi)

        return None

    def _extract_year_from_citation(self, citation_str: str) -> Optional[int]:
        """Extract publication year from citation string"""
        if not citation_str:
            return None

        # Look for 4-digit year in citation
        year_pattern = r'\b(?:19|20)\d{2}\b'
        matches = re.findall(year_pattern, citation_str)
        if matches:
            # Return the first valid year found
            upper_bound = datetime.now().year + 1
            for match in matches:
                year = int(match)
                if 1900 <= year <= upper_bound:  # Reasonable year range
                    return year

        return None

    def _deduplicate_by_doi_pmid(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicates based on DOI and PMID"""
        seen = set()
        deduplicated = []

        missing_identifier_indexes: List[int] = []

        for idx, result in enumerate(results):
            # Create identifier based on DOI or PMID
            doi = result.get('doi', '').strip()
            pmid = result.get('pmid', '').strip()

            identifier = None
            if doi:
                normalized_doi = self._normalize_doi(doi)
                if normalized_doi:
                    identifier = f"doi:{normalized_doi}"
            elif pmid:
                identifier = f"pmid:{pmid}"
            else:
                # Use normalized title MD5 hash as fallback
                title = result.get('title', '').strip()
                if title:
                    # Normalize title: remove punctuation, extra spaces, convert to lowercase
                    normalized_title = re.sub(r'[^\w\s]', '', title.lower())
                    normalized_title = re.sub(r'\s+', ' ', normalized_title).strip()

                    # Create MD5 hash of normalized title
                    title_hash = hashlib.md5(normalized_title.encode('utf-8')).hexdigest()

                    # Optionally add year to reduce false positives
                    year = result.get('year', '')
                    journal = result.get('journal', '')
                    journal_normalized = ''
                    if journal:
                        journal_normalized = re.sub(r'\s+', ' ', str(journal).lower()).strip()

                    if year:
                        identifier = f"title:{title_hash}:year:{year}"
                    else:
                        identifier = f"title:{title_hash}"

                    if journal_normalized:
                        identifier = f"{identifier}:journal:{journal_normalized}"

            if identifier and identifier not in seen:
                seen.add(identifier)
                deduplicated.append(result)
            elif not identifier:
                # If no identifier, keep it anyway
                missing_identifier_indexes.append(idx)
                deduplicated.append(result)

        logger.info(f"Deduplicated {len(results)} -> {len(deduplicated)} results")
        if len(missing_identifier_indexes) >= 2:
            logger.warning(
                "Detected %d results without DOI, PMID, or title (indexes: %s); manual review recommended for potential duplicates.",
                len(missing_identifier_indexes),
                missing_identifier_indexes[:5],
            )
        return deduplicated

    def _process_run_results(self, run_id_or_run: Union[str, Dict], apply_ranking: bool) -> List[Dict]:
        """Normalize, optionally rank, sort, and deduplicate results from a run"""
        if isinstance(run_id_or_run, dict):
            dataset_id = run_id_or_run.get("defaultDatasetId")
        else:
            dataset_id = run_id_or_run

        if not dataset_id:
            raise ValueError("Run object missing defaultDatasetId for result processing")

        dataset_client = self.client.dataset(dataset_id)
        processed_results: List[Dict] = []

        for item in dataset_client.iterate_items():
            normalized_item = self._normalize_item(item)
            if apply_ranking:
                normalized_item = self._apply_study_ranking(normalized_item)
            processed_results.append(normalized_item)

        if apply_ranking:
            # Sort once prior to deduplication so the highest-quality duplicate wins.
            processed_results.sort(key=lambda r: r.get('ranking_score', 0), reverse=True)

        if self.enable_deduplication:
            processed_results = self._deduplicate_by_doi_pmid(processed_results)

        if apply_ranking:
            # Maintain ranking order for downstream consumers after duplicates are removed.
            processed_results.sort(key=lambda r: r.get('ranking_score', 0), reverse=True)

        return processed_results

    def _try_alternative_inputs(
        self,
        enhanced_query: str,
        max_items: int,
        apply_ranking: bool,
        force_include_tags: bool = False,
        include_abstracts: Optional[bool] = None,
        exclude: set = None,
        already_tried: set = None,
    ) -> List[Dict]:
        """Placeholder for alternative schemas (EasyAPI actor only supports searchUrls)."""
        logger.info("Alternative input schemas are disabled for EasyAPI actor; skipping fallback.")
        return []

    def search_pubmed(
        self,
        query: str,
        max_items: Optional[int] = None,
        rank: Optional[bool] = None,
        pharma_enhance: Optional[bool] = None,
    ) -> List[Dict]:
        """Search PubMed through the EasyAPI actor with caching, ranking, and enrichment.

        By default the scraper preserves native PubMed ordering. Enable study
        ranking by passing ``rank=True`` or setting ``ENABLE_STUDY_RANKING=true``.
        ``PRESERVE_PUBMED_ORDER=true`` always disables ranking regardless of the
        other settings.

        When ``EXTRACT_TAGS=false`` and ranking is requested, tags are still
        fetched to keep scoring signals available. With preserve-order runs the
        scraper skips tag collection to reduce latency and cost, and abstracts
        are omitted unless ranking is enabled.

        Args:
            query: PubMed search term.
            max_items: Maximum items to request (clamped by ``HARD_CAP_MAX_ITEMS``).
            rank: Optional override for study ranking. ``None`` respects
                ``ENABLE_STUDY_RANKING`` / ``PRESERVE_PUBMED_ORDER`` behaviour.
            pharma_enhance: Optional override for pharmaceutical query expansion.

        Returns:
            List of normalized PubMed article dictionaries. Ranked results include
            ``ranking_score`` metadata.

        Raises:
            PubMedAccessError: Raised when the actor indicates a subscription or
                permission issue (HTTP 401/402/403).
        """
        # Check for PRESERVE_PUBMED_ORDER override and tag extraction constraints
        preserve_order = os.getenv('PRESERVE_PUBMED_ORDER', 'false').lower() == 'true'

        if rank is True:
            apply_ranking = True
        elif rank is False:
            apply_ranking = False
        else:
            apply_ranking = self.enable_study_ranking

        if preserve_order and apply_ranking:
            logger.info(
                "PRESERVE_PUBMED_ORDER=true; ignoring rank override and returning results in crawl order."
            )
            apply_ranking = False

        should_include_tags = bool(apply_ranking) or bool(self.extract_tags and not preserve_order)
        should_include_abstracts = bool(self.use_full_abstracts) and (apply_ranking or not preserve_order)
        if not apply_ranking and not preserve_order and self.use_full_abstracts:
            logger.info(
                "Including full abstracts (USE_FULL_ABSTRACTS=true, PRESERVE_PUBMED_ORDER=false). Set USE_FULL_ABSTRACTS=false or PRESERVE_PUBMED_ORDER=true to skip abstracts."
            )
        effective_enhance = (
            self.enable_pharma_query_enhancement if pharma_enhance is None else pharma_enhance
        )

        # Check for ranking/tag extraction mismatch
        if apply_ranking and not self.extract_tags:
            logger.info(
                "EXTRACT_TAGS=false but ranking is enabled; includeTags will be requested to support ranking signals."
            )

        # Apply defaults and caps to enforce HARD_CAP_MAX_ITEMS
        requested_max = self.default_max_items if max_items is None else max_items
        requested_label = requested_max if max_items is not None else f"default({requested_max})"

        if requested_max < 1:
            logger.warning(
                "Requested max_items %s is below 1; adjusting to minimum of 1.",
                requested_label,
            )
        effective_max = max(1, requested_max)

        if effective_max > self.hard_cap_max_items:
            logger.warning(
                "Clamping max_items from %s to HARD_CAP_MAX_ITEMS=%s to limit actor cost.",
                requested_label,
                self.hard_cap_max_items,
            )
            effective_max = self.hard_cap_max_items

        logger.info(
            "Using effective_max=%s (requested=%s, hard_cap=%s)",
            effective_max,
            requested_label,
            self.hard_cap_max_items,
        )

        # Enhance pharmaceutical queries first
        enhanced_query = self._enhance_pharmaceutical_query(query, effective_enhance)

        include_tags = should_include_tags
        include_abstract = should_include_abstracts

        cache_key = self._get_cache_key(
            enhanced_query,
            effective_max,
            apply_ranking,
            effective_enhance,
            include_tags_effective=include_tags,
            include_abstract_effective=include_abstract,
        )

        # Try to get cached results first
        cached_results = self._get_cached_results(cache_key, apply_ranking)
        if cached_results is not None:
            return cached_results
        logger.info(f"Searching PubMed for: {enhanced_query} (effective_max: {effective_max})")

        # Prepare input for EasyAPI actor - use searchUrls as primary, searchQuery as fallback
        # Note: includeTags and includeAbstract are optional EasyAPI flags that are guarded by schema fallback.
        # If the actor schema changes, these flags can be gated behind additional env vars in _build_actor_input().
        # Refer to EasyAPI PubMed Search Scraper documentation for current input schema.
        run_input = self._build_actor_input(
            enhanced_query,
            effective_max,
            'searchUrls',
            include_tags=include_tags,
            include_abstract=include_abstract,
            force_include_tags=apply_ranking,
            force_include_abstract=apply_ranking,
        )

        logger.debug(f"EasyAPI actor input (searchUrls primary): {run_input}")

        def _perform_schema_fallback() -> List[Dict]:
            logger.info("Alternative schema fallback disabled for EasyAPI actor")
            return []

        max_retries = 3
        base_delay = 1

        schema_fallback_tried = False
        for attempt in range(max_retries):
            try:
                # Run the actor
                logger.debug(f"Calling actor {self.actor_id} with input: {run_input}")
                run = self._call_actor(run_input, schema='searchUrls')

                results = self._process_run_results(run, apply_ranking)
                results = self._deduplicate_results(results)

                # Check for zero results and attempt schema fallback if enabled
                if len(results) == 0 and self.enable_schema_fallback and not schema_fallback_tried:
                    schema_fallback_tried = True
                    results = _perform_schema_fallback()

                # Cache results when we have data, optionally allow empty caches via env flag
                if results:
                    self._cache_results(cache_key, results)
                elif self.cache_empty_results:
                    self._cache_results(cache_key, results)

                logger.info(f"Successfully retrieved {len(results)} PubMed articles")
                return results

            except PubMedScraper.SchemaValidationError as validation_exc:
                logger.warning("Actor rejected searchUrls schema: %s", validation_exc)
                if self.enable_schema_fallback and not schema_fallback_tried:
                    schema_fallback_tried = True
                    results = _perform_schema_fallback()
                    results = self._deduplicate_results(results)
                    if results:
                        self._cache_results(cache_key, results)
                    elif self.cache_empty_results:
                        self._cache_results(cache_key, results)
                    return results
                raise
            except Exception as e:
                if ApifyApiError and isinstance(e, ApifyApiError):
                    status_code = getattr(e, 'status_code', None)
                    if status_code == 429:
                        delay = float(base_delay * (2 ** attempt))
                        retry_after_header = None
                        response = getattr(e, 'response', None)
                        if response is not None:
                            headers = getattr(response, 'headers', None)
                            if headers:
                                retry_after_header = headers.get('Retry-After') or headers.get('retry-after')

                        if retry_after_header is not None:
                            try:
                                parsed_delay = float(retry_after_header)
                                delay = max(parsed_delay, 0.0)
                                logger.debug(
                                    "Using Retry-After header value %.2fs for Apify rate limit backoff.",
                                    delay,
                                )
                            except (TypeError, ValueError):
                                logger.debug(
                                    "Retry-After header '%s' not numeric; falling back to exponential delay %.2fs.",
                                    retry_after_header,
                                    delay,
                                )

                        logger.warning(f"Apify rate limit hit (429). Retrying in {delay}s...")
                        time.sleep(delay)
                        continue

                # Check for subscription/permission errors first
                if ApifyApiError and isinstance(e, ApifyApiError):
                    if hasattr(e, 'status_code') and e.status_code in [401, 402, 403]:
                        logger.error(SUBSCRIPTION_ERROR_MESSAGE)
                        raise PubMedAccessError(SUBSCRIPTION_ERROR_MESSAGE) from e

                # Fallback heuristic when ApifyApiError is None or not available
                error_msg = str(e)
                if ApifyApiError is None or not isinstance(e, ApifyApiError):
                    error_msg_lower = error_msg.lower()
                    if any(term in error_msg_lower for term in ['unauthorized', 'payment', 'subscription', 'forbidden']):
                        logger.error(SUBSCRIPTION_ERROR_MESSAGE)
                        raise PubMedAccessError(SUBSCRIPTION_ERROR_MESSAGE) from e

                # Log additional context for input-related errors
                status_code = getattr(e, 'status_code', None) if ApifyApiError and isinstance(e, ApifyApiError) else None

                # Check for schema mismatch: explicit 400 status code OR text-based detection
                is_schema_mismatch = (
                    (status_code == 400) or
                    ("input" in error_msg.lower() or "parameter" in error_msg.lower())
                )

                if is_schema_mismatch:
                    logger.error(f"Possible input schema mismatch. Actor: {self.actor_id}, Input: {run_input}, Status: {status_code}, Error: {error_msg}")

                if is_schema_mismatch and self.enable_schema_fallback and not schema_fallback_tried:
                    schema_fallback_tried = True
                    results = _perform_schema_fallback()
                    results = self._deduplicate_results(results)
                    if results:
                        self._cache_results(cache_key, results)
                        logger.info(
                            "Successfully retrieved %s PubMed articles with schema fallback",
                            len(results),
                        )
                        return results
                    elif self.cache_empty_results:
                        self._cache_results(cache_key, results)
                        return results
                    logger.warning("All alternative schemas failed, continuing with normal retry")

                if attempt < max_retries - 1:
                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to search PubMed after {max_retries} attempts: {str(e)}")
                    raise

        return []

    def _normalize_authors(self, authors) -> str:
        """
        Normalize authors field to consistent string format

        Args:
            authors: Authors field (could be string, list, dict, or other)

        Returns:
            Normalized authors string
        """
        if isinstance(authors, list):
            # Handle list of strings or objects with fullName/name
            normalized_authors = []
            for author in authors:
                author_name = self._get_author_name(author)
                if author_name:
                    normalized_authors.append(author_name)

            return ", ".join(normalized_authors)
        elif isinstance(authors, dict):
            # Handle dict inputs by checking various keys
            author_value = (authors.get('full') or
                           authors.get('short') or
                           authors.get('list') or
                           authors.get('items'))
            if author_value:
                if isinstance(author_value, list):
                    # Recursively handle list within dict
                    return self._normalize_authors(author_value)
                else:
                    return str(author_value).strip()
            return ""
        elif isinstance(authors, str):
            return authors.strip()
        else:
            return ""

    def _get_author_name(self, author) -> str:
        """
        Helper function to safely extract author name from dict or string

        Args:
            author: Author entry (dict with full/short/fullName/name keys or string)

        Returns:
            Normalized author name string

        Test cases:
        - Dict input: {'fullName': 'John Doe'} -> 'John Doe'
        - Dict input: {'full': 'John Doe'} -> 'John Doe'
        - String input: 'Jane Smith' -> 'Jane Smith'
        - Mixed list: [{'fullName': 'John Doe'}, 'Jane Smith'] -> ['John Doe', 'Jane Smith']
        """
        if isinstance(author, dict):
            return str(
                author.get('fullName')
                or author.get('name')
                or author.get('full')
                or author.get('short')
                or author
            ).strip()
        return str(author).strip()

    def _normalize_item(self, item: dict) -> dict:
        """
        Normalize actor items with fallbacks for common key variants

        Args:
            item: Raw item dictionary from Apify actor

        Returns:
            Normalized item dictionary
        """
        normalized = {}

        # DOI with fallbacks - always normalize before setting
        doi = item.get('doi') or item.get('DOI') or ''
        if doi:
            raw_doi = str(doi)
            normalized_doi = self._normalize_doi(raw_doi)
            normalized['doi'] = normalized_doi
            normalized['raw_doi'] = raw_doi  # Retain for traceability
        elif 'citation' in item and item['citation']:
            # Try to extract DOI from citation
            citation = item['citation']
            if isinstance(citation, dict):
                # Prefer short, then full
                c_text = citation.get('short') or citation.get('full') or ''
            else:
                c_text = str(citation)
            extracted_doi = self._extract_doi_from_citation(c_text)
            if extracted_doi:
                normalized_doi = self._normalize_doi(extracted_doi)
                normalized['doi'] = normalized_doi
                normalized['raw_doi'] = extracted_doi  # Retain for traceability

        # PMID with fallbacks
        pmid = item.get('pmid') or item.get('PMID') or item.get('pubMedId') or item.get('pmID') or ''
        if pmid:
            normalized['pmid'] = str(pmid).strip()

        # PMCID with fallbacks
        pmcid = item.get('pmcid') or item.get('PMCID') or ''
        # Check for PMCID within an identifiers object
        if not pmcid and 'identifiers' in item and isinstance(item['identifiers'], dict):
            pmcid = item['identifiers'].get('pmcid') or item['identifiers'].get('PMCID') or ''
        if pmcid:
            normalized['pmcid'] = str(pmcid).strip()

        # Year extraction
        year = item.get('year')
        if year:
            normalized['year'] = str(year).strip()
        elif 'citation' in item and item['citation']:
            # Try to extract year from citation
            citation = item['citation']
            if isinstance(citation, dict):
                # Prefer short, then full
                c_text = citation.get('short') or citation.get('full') or ''
            else:
                c_text = str(citation)
            extracted_year = self._extract_year_from_citation(c_text)
            if extracted_year:
                normalized['year'] = str(extracted_year)

        # Publication date with fallbacks
        pub_date = (item.get('publishedDate') or
                   item.get('publicationDate') or
                   item.get('pubDate') or '')
        if pub_date:
            normalized['publication_date'] = str(pub_date).strip()

        # Authors with EasyAPI structure handling
        authors = item.get('authors', '')
        if isinstance(authors, dict):
            # Check for list or items within the dict
            authors_list_data = authors.get('list') or authors.get('items')
            if authors_list_data and isinstance(authors_list_data, list):
                # Populate both authors (normalized string) and authors_list (array of names)
                normalized['authors'] = self._normalize_authors(authors_list_data)
                # Optimize to avoid double-calling _get_author_name
                author_names = [self._get_author_name(a) for a in authors_list_data]
                normalized['authors_list'] = [name for name in author_names if name]
            elif 'full' in authors:
                normalized['authors'] = str(authors['full']).strip()
            else:
                # Try fallback to 'short' if 'full' is not present
                fallback_authors = authors.get('short')
                if fallback_authors:
                    normalized['authors'] = str(fallback_authors).strip()
                else:
                    normalized['authors'] = self._normalize_authors(authors)
        else:
            normalized['authors'] = self._normalize_authors(authors)
            # Add authors_list when original input is a list
            if isinstance(authors, list):
                # Optimize to avoid double-calling _get_author_name
                author_names = [self._get_author_name(a) for a in authors]
                normalized['authors_list'] = [name for name in author_names if name]

        # Abstract with EasyAPI structure handling
        abstract = item.get('abstract', '')
        if isinstance(abstract, dict):
            if self.use_full_abstracts and 'full' in abstract:
                normalized['abstract'] = str(abstract['full']).strip()
            else:
                # Fall back to short or summary
                fallback_abstract = abstract.get('short') or abstract.get('summary') or ''
                if fallback_abstract:
                    normalized['abstract'] = str(fallback_abstract).strip()
        elif abstract:
            normalized['abstract'] = str(abstract).strip()

        # URL mapping with articleUrl
        url = item.get('articleUrl') or item.get('url') or ''
        if url:
            normalized['url'] = str(url).strip()

        # Article ID mapping
        article_id = item.get('articleId')
        if article_id:
            normalized['article_id'] = str(article_id).strip()

        # Journal extraction with improved robustness
        journal = item.get('journal', '')
        if not journal and 'citation' in item and item['citation']:
            citation = item['citation']
            if isinstance(citation, dict):
                # Prefer short, then full
                c_text = citation.get('short') or citation.get('full') or ''
            else:
                c_text = str(citation)
            citation_str = c_text

            # Try multiple regex patterns for robust journal parsing
            journal_patterns = [
                # Pattern 1: Journal name before year or volume (original primary pattern)
                r'^([^.]+?)\s*\.\s*(\d{4}|Vol)',
                # Pattern 2: Journal name before period and year
                r'^(.+?)\.\s*(19|20)\d{2}\b',
                # Pattern 3: Journal name ending with semicolon before year/volume
                r'^(.+?);\s*(\d{4}|Vol|\d+\()',
                # Pattern 4: Journal name before comma and year
                r'^(.+?),\s*(19|20)\d{2}\b',
                # Pattern 5: Journal name before space and year (loose fallback)
                r'^(.+?)\s+(19|20)\d{2}\b'
            ]

            for pattern in journal_patterns:
                journal_match = re.search(pattern, citation_str)
                if journal_match:
                    potential_journal = journal_match.group(1).strip()
                    # Remove trailing punctuation and common separators
                    potential_journal = re.sub(r'[.,;:]+$', '', potential_journal)
                    if potential_journal and len(potential_journal) > 3:  # Basic sanity check
                        journal = potential_journal
                        break

        if journal:
            normalized['journal'] = str(journal).strip()

        # Tags handling - prefer 'tags', fall back to 'articleTypes'/'publicationTypes'
        tags = (item.get('tags') or
                item.get('articleTypes') or
                item.get('publicationTypes') or [])

        if not tags and self.extract_tags:
            # Known alternates observed in EasyAPI responses.
            for alt_key in ('articleTags', 'tagCategories', 'article_categories'):
                alt_value = item.get(alt_key)
                if alt_value:
                    tags = alt_value
                    logger.debug("Recovered tags from alternate key '%s'", alt_key)
                    break

        def _coerce_tag_value(tag_entry) -> str:
            if isinstance(tag_entry, dict):
                for key in ('label', 'name', 'type', 'value'):
                    value = tag_entry.get(key)
                    if value:
                        return str(value).strip()
                return str(tag_entry).strip()
            return str(tag_entry).strip()

        normalized_tags: List[str] = []
        seen_tags: Set[str] = set()

        if tags:
            iterable = tags if isinstance(tags, list) else [tags]
            for entry in iterable:
                coerced = _coerce_tag_value(entry)
                coerced_lower = coerced.lower()
                if coerced and coerced_lower not in seen_tags:
                    normalized_tags.append(coerced)
                    seen_tags.add(coerced_lower)
            normalized['tags'] = normalized_tags
        else:
            normalized['tags'] = []
            if self.extract_tags:
                logger.debug(
                    "Tag extraction enabled but no tags found; available keys: %s",
                    list(item.keys()),
                )

        # MeSH terms with fallbacks - normalize to deduped list of strings
        mesh_terms = (item.get('meshTerms') or
                     item.get('meshHeadings') or
                     item.get('mesh_terms') or
                     item.get('MeSH') or [])
        normalized_mesh_terms = self._normalize_mesh_terms_to_strings(mesh_terms)
        normalized['mesh_terms'] = normalized_mesh_terms if normalized_mesh_terms else []

        # Title with fallbacks for alternate keys
        title = item.get('title') or item.get('articleTitle') or ''
        if title:
            normalized['title'] = str(title).strip()

        # Citation handling for downstream display
        citation = item.get('citation')
        if citation and isinstance(citation, dict):
            if 'short' in citation:
                normalized['citation_short'] = str(citation['short']).strip()
            if 'full' in citation:
                normalized['citation_full'] = str(citation['full']).strip()
        elif citation:
            # If citation is a string, treat it as short citation
            normalized['citation_short'] = str(citation).strip()

        # Add source/provider information for downstream filters
        normalized['source'] = 'pubmed'
        normalized['ingestion'] = 'apify'
        normalized['provider'] = self.scraper_provider
        normalized['provider_family'] = 'apify'
        if self.actor_id:
            normalized['actor_id'] = self.actor_id

        return normalized

    def _normalize_mesh_terms_to_strings(self, mesh_terms) -> List[str]:
        """
        Normalize mesh_terms to List[str] by coercing dicts/objects to string labels
        De-duplicates case-insensitively while preserving original casing and stable ordering

        Args:
            mesh_terms: MeSH terms field (could be strings, dicts, or mixed)

        Returns:
            Normalized mesh_terms as List[str] with duplicates removed case-insensitively
        """
        if not mesh_terms:
            return []

        normalized: List[str] = []
        seen_lower: Set[str] = set()

        def _append_terms(value: str) -> None:
            for part in value.split(','):
                part = part.strip()
                if part:
                    part_lower = part.lower()
                    if part_lower not in seen_lower:
                        seen_lower.add(part_lower)
                        normalized.append(part)

        if isinstance(mesh_terms, str):
            _append_terms(mesh_terms)
            return normalized

        if not isinstance(mesh_terms, list):
            mesh_terms = [mesh_terms]

        for term in mesh_terms:
            if isinstance(term, dict):
                label = term.get('label') or term.get('name') or term.get('term') or str(term)
                if label and str(label).strip():
                    _append_terms(str(label))
            elif isinstance(term, str):
                _append_terms(term)
            else:
                term_str = str(term).strip()
                if term_str:
                    _append_terms(term_str)

        return normalized

    def to_documents(self, results: List[Dict]) -> List[Document]:
        """
        Convert PubMed results to LangChain Documents

        Args:
            results: List of PubMed article dictionaries

        Returns:
            List of Document objects
        """
        documents = []

        for result in results:
            # Create document content
            content_parts = []

            title = result.get('title', '').strip()
            if title:
                content_parts.append(f"Title: {title}")

            authors = self._normalize_authors(result.get('authors', ''))
            if authors:
                content_parts.append(f"Authors: {authors}")

            abstract = result.get('abstract', '').strip()
            if abstract:
                content_parts.append(f"Abstract: {abstract}")

            journal = result.get('journal', '').strip()
            if journal:
                content_parts.append(f"Journal: {journal}")

            # Add Identifiers block if DOI, PMID, or PMCID present
            identifiers = []
            doi = result.get('doi', '').strip()
            if doi:
                identifiers.append(f"DOI: {doi}")

            pmid = result.get('pmid', '').strip()
            if pmid:
                identifiers.append(f"PMID: {pmid}")

            pmcid = result.get('pmcid', '').strip()
            if pmcid:
                identifiers.append(f"PMCID: {pmcid}")

            if identifiers:
                content_parts.append(f"Identifiers: {', '.join(identifiers)}")

            # Combine content
            content = "\n\n".join(content_parts)

            if not content.strip():
                continue

            # Prepare metadata
            metadata = {
                'source': 'pubmed',
                'ingestion': 'apify',
                # provider: concrete scraper identifier (e.g., apify-easyapi)
                'provider': self.scraper_provider,
                # provider_family: logical family for backward compatibility with filters expecting 'apify'
                'provider_family': 'apify',
                # scraper_provider: legacy alias retained for downstream compatibility
                'scraper_provider': self.scraper_provider,
                'title': title,
                'authors': authors,
                'journal': journal,
                'publication_date': result.get('publication_date', ''),
                'url': result.get('url', '')
            }

            actor_id = result.get('actor_id')
            if actor_id:
                metadata['actor_id'] = actor_id

            # Add year if present
            year = result.get('year', '').strip()
            if year:
                metadata['year'] = year

            # Add optional fields
            doi = result.get('doi', '').strip()
            if doi:
                metadata['doi'] = doi

            pmid = result.get('pmid', '').strip()
            if pmid:
                metadata['pmid'] = pmid

            pmcid = result.get('pmcid', '').strip()
            if pmcid:
                metadata['pmcid'] = pmcid

            # Always include tags and mesh_terms as lists
            tags = result.get('tags', [])
            metadata['tags'] = tags if isinstance(tags, list) else []

            mesh_terms = result.get('mesh_terms', [])
            if mesh_terms:
                if isinstance(mesh_terms, list) and all(isinstance(term, str) for term in mesh_terms):
                    metadata['mesh_terms'] = mesh_terms
                else:
                    normalized_mesh = self._normalize_mesh_terms_to_strings(mesh_terms)
                    metadata['mesh_terms'] = normalized_mesh if normalized_mesh else []
            else:
                metadata['mesh_terms'] = []

            # Add study type, ranking score, and confidence if present

            if 'study_type' in result and result['study_type']:
                metadata['study_type'] = result['study_type']

            if 'study_types' in result and isinstance(result['study_types'], list) and result['study_types']:
                metadata['study_types'] = result['study_types']
            elif metadata.get('study_type'):
                metadata['study_types'] = [metadata['study_type']]

            if 'study_type_confidence' in result and result['study_type_confidence'] is not None:
                metadata['study_type_confidence'] = result['study_type_confidence']

            if 'ranking_score' in result and result['ranking_score'] is not None:
                metadata['ranking_score'] = result['ranking_score']

            # Add citation fields if present
            if 'citation_short' in result and result['citation_short']:
                metadata['citation_short'] = result['citation_short']

            if 'citation_full' in result and result['citation_full']:
                metadata['citation_full'] = result['citation_full']

            # Add authors_list to metadata if available
            if result.get('authors_list') and isinstance(result['authors_list'], list) and result['authors_list']:
                metadata['authors_list'] = result['authors_list']

            # Create document
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        logger.info(f"Converted {len(results)} results to {len(documents)} documents")
        return documents

    def get_cache_info(self) -> Dict:
        """Get information about the cache"""
        cache_files = list(self.cache_dir.glob("*.json"))
        total_files = len(cache_files)

        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

    def write_sidecar_for_pdf(self, pdf_path: Path, article: Dict) -> Path:
        """
        Write sidecar JSON file for a PDF with normalized PubMed metadata

        Args:
            pdf_path: Path to the PDF file
            article: Article dictionary from PubMed search

        Returns:
            Path to the created sidecar file
        """
        sidecar_path = pdf_path.with_suffix('.pubmed.json')

        mesh_terms_raw = article.get('mesh_terms', [])
        mesh_terms: List[str] = []
        if isinstance(mesh_terms_raw, str):
            mesh_terms = [term.strip() for term in re.split(r'[;,]', mesh_terms_raw) if term.strip()]
        elif isinstance(mesh_terms_raw, list):
            for term in mesh_terms_raw:
                term_str = str(term).strip()
                if term_str:
                    mesh_terms.append(term_str)

        authors_value = self._normalize_authors(article.get('authors', ''))
        normalized_data = {
            'doi': str(article.get('doi', '')).strip() or None,
            'pmid': str(article.get('pmid', '')).strip() or None,
            'title': str(article.get('title', '')).strip() or None,
            'authors': authors_value or None,
            'abstract': str(article.get('abstract', '')).strip() or None,
            'publication_date': str(article.get('publication_date', '')).strip() or None,
            'journal': str(article.get('journal', '')).strip() or None,
            'mesh_terms': mesh_terms,
        }

        # Remove None values
        normalized_data = {k: v for k, v in normalized_data.items() if v is not None}

        try:
            with open(sidecar_path, 'w', encoding='utf-8') as f:
                json.dump(normalized_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Wrote sidecar file: {sidecar_path.name}")
            return sidecar_path

        except Exception as e:
            logger.error(f"Failed to write sidecar for {pdf_path.name}: {str(e)}")
            raise

    def _extract_pubmed_metadata_from_text(self, text: str) -> Dict:
        """
        Extract PubMed metadata from PDF text content
        (Reused from document_loader for matching)

        Args:
            text: Text content from first 1-3 pages

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}

        # Extract DOI using regex (tightened to avoid trailing bracketed text)
        doi_pattern = r'(?:doi[:\s]*)?10\.\d{4,9}/[^\s)>\]]+'
        doi_match = re.search(doi_pattern, text, re.IGNORECASE)
        if doi_match:
            raw_doi = doi_match.group().replace('doi:', '').replace('DOI:', '').strip()
            # Apply same normalization as used elsewhere
            normalized_doi = self._normalize_doi(raw_doi)
            if normalized_doi:
                metadata['doi'] = normalized_doi

        # Extract PMID using regex
        pmid_pattern = r'PMID[:\s]+(\d+)'
        pmid_match = re.search(pmid_pattern, text, re.IGNORECASE)
        if pmid_match:
            metadata['pmid'] = pmid_match.group(1)

        return metadata

    def export_sidecars(self, docs_dir: str, results: List[Dict]) -> int:
        """
        Export sidecar JSON files for PDFs in docs directory by matching DOI/PMID

        Args:
            docs_dir: Directory containing PDF files
            results: List of PubMed article results

        Returns:
            Number of sidecar files created
        """
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            logger.error(f"Docs directory does not exist: {docs_dir}")
            return 0

        # Get all PDF files
        pdf_files = list(docs_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {docs_dir}")
            return 0

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        # Create lookup dictionaries for articles by DOI and PMID
        articles_by_doi = {}
        articles_by_pmid = {}

        for article in results:
            doi = article.get('doi', '').strip()
            pmid = article.get('pmid', '').strip()

            if doi:
                normalized_doi = self._normalize_doi(doi)
                if normalized_doi:
                    articles_by_doi[normalized_doi] = article
            if pmid:
                articles_by_pmid[pmid] = article

        created_count = 0

        for pdf_path in pdf_files:
            try:
                # Check if sidecar already exists
                sidecar_path = pdf_path.with_suffix('.pubmed.json')
                if sidecar_path.exists():
                    logger.debug(f"Sidecar already exists for {pdf_path.name}")
                    continue

                logger.info(f"Processing {pdf_path.name}")

                # Read first few pages to extract metadata
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(pdf_path))
                    pdf_documents = loader.load()

                    # Get text from first 3 pages
                    text_to_analyze = ""
                    for i, page in enumerate(pdf_documents[:3]):
                        text_to_analyze += page.page_content + "\n"

                    # Extract DOI/PMID from PDF
                    pdf_metadata = self._extract_pubmed_metadata_from_text(text_to_analyze)

                except Exception as e:
                    logger.warning(f"Could not extract text from {pdf_path.name}: {str(e)}")
                    continue

                # Try to match by DOI first
                matched_article = None
                pdf_doi = pdf_metadata.get('doi', '').strip()
                if pdf_doi:
                    normalized_pdf_doi = self._normalize_doi(pdf_doi)
                    if normalized_pdf_doi and normalized_pdf_doi in articles_by_doi:
                        matched_article = articles_by_doi[normalized_pdf_doi]
                        logger.info(f"Matched {pdf_path.name} by DOI: {normalized_pdf_doi}")

                # Try to match by PMID if DOI match failed
                if not matched_article:
                    pdf_pmid = pdf_metadata.get('pmid', '').strip()
                    if pdf_pmid and pdf_pmid in articles_by_pmid:
                        matched_article = articles_by_pmid[pdf_pmid]
                        logger.info(f"Matched {pdf_path.name} by PMID: {pdf_pmid}")

                # Create sidecar if we found a match
                if matched_article:
                    self.write_sidecar_for_pdf(pdf_path, matched_article)
                    created_count += 1
                else:
                    logger.debug(f"No PubMed match found for {pdf_path.name}")

            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                continue

        logger.info(f"Created {created_count} sidecar files out of {len(pdf_files)} PDFs")
        return created_count


def main():
    """CLI interface for PubMed scraper"""
    parser = argparse.ArgumentParser(
        description="PubMed scraper using Apify",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python -m src.pubmed_scraper "covid-19 treatment"
  python -m src.pubmed_scraper "covid-19 treatment" --max-items 50
  python -m src.pubmed_scraper "covid-19 treatment" --no-rank
  python -m src.pubmed_scraper "covid-19 treatment" --export-sidecars ./Data/Docs

Note: Results are ranked by study quality/relevance by default. Use --no-rank to preserve original PubMed ordering.
        """
    )

    parser.add_argument(
        "query",
        type=str,
        help="PubMed search query"
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to retrieve (default: from DEFAULT_MAX_ITEMS env var, capped by HARD_CAP_MAX_ITEMS)"
    )

    parser.add_argument(
        "--no-rank",
        action="store_true",
        help="Disable study ranking and preserve original PubMed result order. "
             "By default, results are ranked by study quality/relevance/recency. "
             "Can also be disabled globally with ENABLE_STUDY_RANKING=false."
    )

    parser.add_argument(
        "--export-sidecars",
        type=str,
        metavar="DOCS_DIR",
        help="Export sidecar JSON files for PDFs in specified directory"
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        return e.code

    try:
        scraper = PubMedScraper()

        print(f"Searching PubMed for: '{args.query}' (max_items: {args.max_items})")
        results = scraper.search_pubmed(args.query, args.max_items, rank=not args.no_rank)

        print(f"\nFound {len(results)} articles")

        # Convert to documents
        documents = scraper.to_documents(results)
        print(f"Converted to {len(documents)} documents")

        # Show cache info
        cache_info = scraper.get_cache_info()
        print(f"\nCache info:")
        print(f"  Directory: {cache_info['cache_dir']}")
        print(f"  Files: {cache_info['total_files']}")
        print(f"  Size: {cache_info['total_size_mb']} MB")

        # Export sidecars if requested
        if args.export_sidecars:
            print(f"\nExporting sidecars to: {args.export_sidecars}")
            created_count = scraper.export_sidecars(args.export_sidecars, results)
            print(f"Created {created_count} sidecar files")

        # Show first few results
        print(f"\nFirst 3 results:")
        for i, result in enumerate(results[:3]):
            print(f"\n{i+1}. {result.get('title', 'No title')}")
            print(f"   Authors: {result.get('authors', 'No authors')}")
            print(f"   Journal: {result.get('journal', 'No journal')}")
            print(f"   DOI: {result.get('doi', 'No DOI')}")
            print(f"   PMID: {result.get('pmid', 'No PMID')}")

    except PubMedAccessError as e:
        print(f"Access Error: {str(e)}")
        print("Please check your APIFY_TOKEN and verify your EasyAPI subscription status.")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
