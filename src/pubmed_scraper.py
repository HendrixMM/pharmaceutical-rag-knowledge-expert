"""
PubMed Scraper using Apify
Handles scraping PubMed search results with caching and deduplication.

Note: The scraper now preserves native PubMed ordering by default. Pass
``rank=True`` or set ``ENABLE_STUDY_RANKING=true`` to enable study ranking.

Usage:
  python -m src.pubmed_scraper "query"
  python src/pubmed_scraper.py "query"

  # With options:
  python -m src.pubmed_scraper "cancer immunotherapy" --max-items 50 --rank --cache-ttl-hours 12
  python -m src.pubmed_scraper "cancer immunotherapy" --no-docs --write-sidecars

Set APIFY_TOKEN environment variable before running.

Flags:
  --max-items N          Maximum number of items to retrieve (default: configured value)
  --rank/--no-rank       Enable/disable study ranking (default: environment setting)
  --cache-ttl-hours H    Cache TTL in hours (default: configured value)
  --no-docs              Output raw PubMed data without converting to LangChain documents
  --write-sidecars       Write .pubmed.json sidecars for PDFs in DOCS_FOLDER
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
from typing import List, Dict, Optional, Tuple, Union, Set, Literal, Any
from pathlib import Path

try:
    from apify_client import ApifyClient
except ImportError as e:
    ApifyClient = None
    logging.getLogger(__name__).warning(
        "apify-client>=1.7.0,<2.0.0 recommended for EasyAPI integration: %s", e
    )

try:
    from apify_client import ApifyApiError  # type: ignore[attr-defined]
except ImportError:
    try:
        from apify_client._errors import ApifyApiError  # type: ignore[attr-defined]
    except ImportError:
        ApifyApiError = None

try:
    import requests
except ImportError:  # pragma: no cover - optional fallback path when requests unavailable
    requests = None

from langchain_core.documents import Document

# Import unified DOI/PMID patterns and utilities
from .paper_schema import normalize_doi, normalize_pmid


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


def _get_env_int(name: str, default: int) -> int:
    """Return positive integer configuration from environment with fallback."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    try:
        parsed = int(value.strip())
        if parsed <= 0:
            logger.warning("Environment variable %s must be positive. Using default %s.", name, default)
            return default
        return parsed
    except ValueError:
        logger.warning("Environment variable %s=%s is not an integer. Using default %s.", name, value, default)
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

_DEFAULT_APIFY_SUBSCRIPTION_PATTERNS = (
    "rent a paid actor",
    "paid actor",
    "subscription",
    "payment required",
    "forbidden",
    "plan limit",
    "plan limits",
    "plan quota",
    "limit reached",
    "upgrade required",
    "upgrade your plan",
    "upgrade to continue",
)

_subscription_patterns_override = os.getenv("APIFY_SUBSCRIPTION_PATTERNS")
if _subscription_patterns_override:
    APIFY_SUBSCRIPTION_PATTERNS = tuple(
        pattern.strip().lower()
        for pattern in _subscription_patterns_override.split(',')
        if pattern.strip()
    )
else:
    APIFY_SUBSCRIPTION_PATTERNS = tuple(pattern.lower() for pattern in _DEFAULT_APIFY_SUBSCRIPTION_PATTERNS)


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
            raise ValueError(
                "APIFY_TOKEN is required. Obtain an API token from https://console.apify.com/ and set it"
                " via the APIFY_TOKEN environment variable before running the PubMed scraper."
            )

        if not ApifyClient:
            raise ImportError("apify_client is required. Install with: pip install apify-client>=1.7.0,<2.0.0")

        self.client = ApifyClient(self.apify_token)
        self.actor_id = os.getenv("EASYAPI_ACTOR_ID", "easyapi/pubmed-search-scraper")
        self.cache_dir = Path(cache_dir or os.getenv("PUBMED_CACHE_DIR", "./pubmed_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load configuration from environment
        self.default_max_items = int(os.getenv("DEFAULT_MAX_ITEMS", "30"))
        self.hard_cap_max_items = int(os.getenv("HARD_CAP_MAX_ITEMS", "100"))
        self.enable_study_ranking = _env_true("ENABLE_STUDY_RANKING", False)
        if self.enable_study_ranking:
            logger.info(
                "Study ranking enabled; set ENABLE_STUDY_RANKING=false or PRESERVE_PUBMED_ORDER=true to retain raw PubMed ordering."
            )
        # Support both new and old environment variable names for backward compatibility
        def _parse_truthy(value: str) -> bool:
            return value.strip().lower() in {"1", "true", "yes", "on"}

        enable_dedup_env = os.getenv("ENABLE_DEDUPLICATION")
        legacy_dedup_env = os.getenv("ENABLE_PMID_DEDUPLICATION")

        if legacy_dedup_env is not None:
            logger.warning(
                "ENABLE_PMID_DEDUPLICATION is deprecated; use ENABLE_DEDUPLICATION instead."
            )

        if enable_dedup_env is None:
            if legacy_dedup_env is not None:
                enable_deduplication = _parse_truthy(legacy_dedup_env)
                logger.info(
                    "Applying legacy ENABLE_PMID_DEDUPLICATION=%s; set ENABLE_DEDUPLICATION explicitly to override.",
                    legacy_dedup_env,
                )
            else:
                enable_deduplication = True
                logger.info(
                    "ENABLE_DEDUPLICATION not set; defaulting to true (duplicate PMIDs will be removed). "
                    "Set ENABLE_DEDUPLICATION=false to retain raw PubMed ordering and counts.",
                )
        else:
            enable_deduplication = _parse_truthy(enable_dedup_env)

        self.enable_deduplication = enable_deduplication
        # Default false to reduce Apify usage cost; override with env var when needed
        self.use_full_abstracts = _env_true("USE_FULL_ABSTRACTS", False)
        self.extract_tags = _env_true("EXTRACT_TAGS", True)
        self.include_tags_with_preserve_order = _env_true("INCLUDE_TAGS_WITH_PRESERVE_ORDER", False)
        self.enable_pharma_query_enhancement = _env_true("ENABLE_PHARMA_QUERY_ENHANCEMENT", True)
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
        self.enable_schema_fallback = _env_true("ENABLE_EASYAPI_SCHEMA_FALLBACK", False)
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
            run = actor_client.call(run_input=run_input)
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
                        retry_run = actor_client.call(run_input=retry_input)
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
                        retry_run = actor_client.call(run_input=retry_input)
                        if retry_run and retry_run.get("defaultDatasetId"):
                            return retry_run
                        logger.warning("Retry without optional flags still missing defaultDatasetId or failed")

                    if self.smart_schema_fallback:
                        raise PubMedScraper.SchemaValidationError(
                            f"Actor validation error for schema '{schema or 'unknown'}': {str(e)}"
                        ) from e

            # Re-raise the original exception if not a validation error or retry failed
            raise

    def _entrez_fetch_abstract(self, pmid: str) -> Optional[str]:
        """Fetch abstract text from NCBI when requested."""

        if requests is None:
            return None

        try:
            # Apply rate limiting if enabled
            if self._rate_limit_active():
                logger.debug("Applying NCBI rate limiting to Entrez abstract fetch")
                self.rate_limiter.acquire()  # type: ignore

            fetch_params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'text',
                'rettype': 'abstract',
            }
            response = requests.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                params=fetch_params,
                timeout=30,
            )
            response.raise_for_status()
            abstract_text = response.text.strip()
            return abstract_text or None
        except Exception as exc:  # pragma: no cover - network dependent fallback
            logger.debug("Entrez abstract fetch failed for PMID %s: %s", pmid, exc)
            return None

    def _fallback_entrez_search(
        self,
        query: str,
        max_items: int,
        *,
        apply_ranking: bool,
        include_abstract: bool,
    ) -> List[Dict[str, Any]]:
        """Fallback to NCBI E-utilities when the EasyAPI actor is unavailable."""

        if requests is None:
            logger.error("requests library unavailable; cannot execute Entrez fallback")
            return []

        esearch_params = {
            'db': 'pubmed',
            'term': query,
            'retmode': 'json',
            'retmax': max(1, max_items),
        }

        try:
            # Apply rate limiting if enabled
            if self._rate_limit_active():
                logger.debug("Applying NCBI rate limiting to Entrez esearch")
                self.rate_limiter.acquire()  # type: ignore

            esearch_resp = requests.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                params=esearch_params,
                timeout=30,
            )
            esearch_resp.raise_for_status()
            esearch_data = esearch_resp.json()
        except Exception as exc:  # pragma: no cover - network dependent fallback
            logger.error("Entrez esearch fallback failed: %s", exc)
            return []

        id_list = esearch_data.get('esearchresult', {}).get('idlist', [])
        if not id_list:
            logger.info("Entrez fallback returned no PubMed IDs for query '%s'", query)
            return []

        # Note: When rate limiting is enabled, all Entrez requests will be rate limited
        # to ensure NCBI compliance. This includes esearch, esummary, and efetch calls.
        if self._rate_limit_active():
            logger.info("NCBI rate limiting is active - Entrez fallback requests will comply with rate limits")

        summary_params = {
            'db': 'pubmed',
            'id': ','.join(id_list),
            'retmode': 'json',
        }

        try:
            # Apply rate limiting if enabled
            if self._rate_limit_active():
                logger.debug("Applying NCBI rate limiting to Entrez esummary")
                self.rate_limiter.acquire()  # type: ignore

            summary_resp = requests.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi',
                params=summary_params,
                timeout=30,
            )
            summary_resp.raise_for_status()
            summary_data = summary_resp.json()
        except Exception as exc:  # pragma: no cover - network dependent fallback
            logger.error("Entrez esummary fallback failed: %s", exc)
            return []

        result_payload = summary_data.get('result', {})
        normalized_results: List[Dict[str, Any]] = []

        for uid in id_list:
            doc = result_payload.get(uid)
            if not doc:
                continue

            article_ids = doc.get('articleids') or []
            doi_value = None
            for article_id in article_ids:
                if article_id.get('idtype') == 'doi' and article_id.get('value'):
                    doi_value = article_id['value']
                    break

            authors = doc.get('authors') or []
            author_entries = [
                {'name': author.get('name')}
                for author in authors
                if isinstance(author, dict) and author.get('name')
            ]

            raw_item: Dict[str, Any] = {
                'title': doc.get('title') or doc.get('sorttitle') or '',
                'journal': doc.get('fulljournalname') or doc.get('source'),
                'source': doc.get('fulljournalname') or doc.get('source'),
                'publicationDate': doc.get('pubdate') or doc.get('epubdate') or doc.get('sortpubdate'),
                'pmid': uid,
                'articleids': article_ids,
                'authors': author_entries,
                'url': doc.get('availablefromurl') or f'https://pubmed.ncbi.nlm.nih.gov/{uid}/',
                'meshHeadings': doc.get('meshheadinglist') or [],
                'tags': doc.get('pubtype') or [],
            }

            if doi_value:
                raw_item['doi'] = doi_value

            if include_abstract:
                abstract_text = self._entrez_fetch_abstract(uid)
                if abstract_text:
                    raw_item['abstract'] = abstract_text

            normalized = self._normalize_item(raw_item)

            # Override provider fields to identify Entrez fallback results
            normalized['provider'] = 'entrez'
            normalized['provider_detail'] = 'ncbi-eutils'
            normalized['provider_variant'] = 'esearch/esummary'
            normalized['source_pipeline'] = 'entrez_fallback'
            normalized['ingestion'] = 'entrez'

            if not normalized.get('year'):
                pubdate = raw_item.get('publicationDate') or ''
                extracted_year = self._extract_year_from_citation(str(pubdate))
                if extracted_year:
                    normalized['year'] = str(extracted_year)

            if apply_ranking:
                normalized = self._apply_study_ranking(normalized)

            normalized_results.append(normalized)

        if self.enable_deduplication and normalized_results:
            normalized_results = self._deduplicate_by_doi_pmid(normalized_results)

        return normalized_results

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
        include_tags_effective: bool,
        include_abstract_effective: bool,
    ) -> dict:
        """Build actor input dictionary for EasyAPI."""
        encoded_query = urllib.parse.quote_plus(enhanced_query)
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={encoded_query}"
        run_input: Dict[str, object] = {
            "searchUrls": [pubmed_url],
            "maxItems": max_items,
        }
        if include_tags_effective:
            run_input["includeTags"] = True
        if include_abstract_effective:
            run_input["includeAbstract"] = True
        return run_input

    def _get_cache_key(
        self,
        enhanced_query: str,
        max_items: int,
        apply_ranking: bool,
        pharma_enhance_enabled: bool,
        include_tags_effective: bool,
        include_abstract_effective: bool,
        preserve_order: bool,
    ) -> str:
        """Generate cache key for enhanced query"""
        content = (
            f"{enhanced_query}:{max_items}:actor={self.actor_id}:"
            f"tags={int(include_tags_effective)}:"
            f"abstract={int(include_abstract_effective)}:"
            f"preserve={int(preserve_order)}:"
            f"enhance={int(pharma_enhance_enabled)}:rank={int(apply_ranking)}:"
            f"dedup={int(self.enable_deduplication)}:"
            f"pharmaMax={self.pharma_max_terms}:"
            f"v={CACHE_SCHEMA_VERSION}"
        )
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_results(self, cache_key: str, apply_ranking: bool) -> Optional[List[Dict]]:
        """Get cached results if they exist and are not expired"""
        advanced_cache_file = (self.cache_dir / "advanced" / f"{cache_key}.json")
        if advanced_cache_file.exists():
            # Advanced caching takes precedence when present so that newer backends own the key lifecycle.
            logger.debug(
                "Ignoring advanced cache entry for key %s in legacy cache lookup", cache_key
            )
            return None

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

        except json.JSONDecodeError as e:
            # Handle JSON corruption specifically
            logger.warning(f"Cache file {cache_file.name} is corrupted (invalid JSON): {str(e)}")
            # Move corrupt file to .bad extension
            bad_file = cache_file.with_suffix('.json.bad')
            try:
                cache_file.rename(bad_file)
                logger.info(f"Moved corrupt cache file to {bad_file.name}")
            except Exception as rename_error:
                logger.warning(f"Failed to rename corrupt cache file: {str(rename_error)}")
                try:
                    cache_file.unlink()  # Fallback to deletion
                    logger.info(f"Deleted corrupt cache file {cache_file.name}")
                except Exception as delete_error:
                    logger.error(f"Failed to delete corrupt cache file {cache_file.name}: {str(delete_error)}")
            return None

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
        """Normalize DOI using shared utility function"""
        return normalize_doi(doi)

  
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
        # Only add MeSH qualifiers if explicitly enabled
        if _env_true("ENABLE_PHARMA_MESH_QUALIFIERS", False):
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
            'randomised controlled trials as topic': ('RCT', 0.85),
            'randomized controlled trial': ('RCT', 0.85),
            'randomised controlled trial': ('RCT', 0.85),
            'systematic reviews as topic': ('Systematic Review', 0.8),
            'systematic review': ('Systematic Review', 0.8),
            'meta-analysis as topic': ('Meta-Analysis', 0.8),
            'meta analysis as topic': ('Meta-Analysis', 0.8),
            'meta-analysis': ('Meta-Analysis', 0.8),
            'meta analysis': ('Meta-Analysis', 0.8),
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

        structured_tag_mapping = {
            'clinical trial, phase i': ('Phase I Clinical Trial', 0.82),
            'clinical trial, phase ii': ('Phase II Clinical Trial', 0.84),
            'clinical trial, phase iii': ('Phase III Clinical Trial', 0.86),
            'clinical trial, phase iv': ('Phase IV Clinical Trial', 0.86),
            'clinical trial, phase i/ii': ('Phase I/II Clinical Trial', 0.83),
            'clinical trial, phase ii/iii': ('Phase II/III Clinical Trial', 0.85),
            'clinical trial, phase iii/iv': ('Phase III/IV Clinical Trial', 0.85),
            'guideline': ('Guideline', 0.6),
            'practice guideline': ('Practice Guideline', 0.65),
        }
        for tag_value in tags_lower:
            normalized_tag = tag_value.strip()
            if normalized_tag in structured_tag_mapping:
                return structured_tag_mapping[normalized_tag]

        # Check for specific study types in order of priority
        if any('randomized controlled trial' in tag or 'randomised controlled trial' in tag for tag in tags_lower) or 'rct' in tags_str:
            return ('RCT', 0.9)
        elif any('systematic review' in tag or 'systematic review' in tag for tag in tags_lower):
            return ('Systematic Review', 0.85)
        elif any('meta-analysis' in tag or 'meta analysis' in tag for tag in tags_lower):
            return ('Meta-Analysis', 0.85)
        elif any('clinical trial' in tag or 'clinical trial' in tag for tag in tags_lower):
            return ('Clinical Trial', 0.8)
        elif any(re.search(r'\bcohort stud(?:y|ies)\b', tag) for tag in tags_lower):
            return ('Cohort Study', 0.75)
        elif any(re.search(r'\bcase-?control stud(?:y|ies)\b', tag) for tag in tags_lower):
            return ('Case-Control Study', 0.7)
        elif any('observational study' in tag for tag in tags_lower):
            return ('Observational Study', 0.6)
        elif any(re.search(r'cross[-\s]?sectional(?: stud(?:y|ies))?', tag) for tag in tags_lower):
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
                if ('randomized' in mesh_term or 'randomised' in mesh_term) and 'trial' in mesh_term:
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

    def _extract_all_study_types(self, tags: List[str], mesh_terms: List[str]) -> List[str]:
        """Extract all matched study types from tags and MeSH terms.

        Returns a list of all study types found, ordered by confidence.
        """
        study_types = set()
        tags_lower = [tag.lower() for tag in tags] if tags else []
        tags_str = ' '.join(tags_lower)

        # Define study type patterns
        study_type_patterns = {
            'RCT': ['randomized controlled trial', 'randomised controlled trial', 'rct'],
            'Systematic Review': ['systematic review'],
            'Meta-Analysis': ['meta-analysis', 'meta analysis'],
            'Clinical Trial': ['clinical trial'],
            'Phase I Clinical Trial': ['clinical trial, phase i', 'phase i clinical trial'],
            'Phase II Clinical Trial': ['clinical trial, phase ii', 'phase ii clinical trial'],
            'Phase III Clinical Trial': ['clinical trial, phase iii', 'phase iii clinical trial'],
            'Phase IV Clinical Trial': ['clinical trial, phase iv', 'phase iv clinical trial'],
            'Cohort Study': ['cohort study', 'cohort studies'],
            'Case-Control Study': ['case-control study', 'case-control studies'],
            'Observational Study': ['observational study', 'observational studies'],
            'Cross-Sectional Study': ['cross-sectional study', 'cross-sectional studies'],
            'Review': ['review'],
            'Case Report': ['case report'],
            'Pilot Study': ['pilot study'],
        }

        # Check each pattern against tags
        for study_type, patterns in study_type_patterns.items():
            for pattern in patterns:
                if any(pattern in tag for tag in tags_lower) or pattern in tags_str:
                    study_types.add(study_type)
                    break

        # Check MeSH terms
        mesh_study_mapping = {
            'randomized controlled trials as topic': 'RCT',
            'randomized controlled trial': 'RCT',
            'systematic reviews as topic': 'Systematic Review',
            'systematic review': 'Systematic Review',
            'meta-analysis as topic': 'Meta-Analysis',
            'meta-analysis': 'Meta-Analysis',
            'clinical trials as topic': 'Clinical Trial',
            'clinical trial': 'Clinical Trial',
            'cohort studies': 'Cohort Study',
            'case-control studies': 'Case-Control Study',
            'observational study': 'Observational Study',
            'cross-sectional studies': 'Cross-Sectional Study',
            'review literature as topic': 'Review',
            'pilot projects': 'Pilot Study',
            'case reports': 'Case Report',
        }

        for mesh_term in mesh_terms:
            mesh_lower = mesh_term.lower()
            if mesh_lower in mesh_study_mapping:
                study_types.add(mesh_study_mapping[mesh_lower])

        # Return sorted by typical confidence order
        confidence_order = [
            'Systematic Review', 'Meta-Analysis', 'RCT',
            'Phase III Clinical Trial', 'Phase II Clinical Trial', 'Phase I Clinical Trial',
            'Phase IV Clinical Trial', 'Clinical Trial', 'Cohort Study',
            'Case-Control Study', 'Observational Study', 'Cross-Sectional Study',
            'Review', 'Pilot Study', 'Case Report'
        ]

        sorted_types = []
        for stype in confidence_order:
            if stype in study_types:
                sorted_types.append(stype)

        # Add any remaining types not in the confidence order
        for stype in study_types:
            if stype not in sorted_types:
                sorted_types.append(stype)

        return sorted_types

    def _apply_study_ranking(self, paper: Dict) -> Dict:
        """Apply study ranking based on multiple factors"""
        base_score = 0.5

        # Study type score
        tags = paper.get('tags', [])
        mesh_terms = paper.get('mesh_terms', [])

        # Get primary study type for scoring
        study_type, type_score = self._classify_study_type(tags, mesh_terms)
        paper['study_type'] = study_type
        paper['study_type_confidence'] = type_score

        # Collect ALL matched study types from tags
        all_study_types = self._extract_all_study_types(tags, mesh_terms)
        paper['study_types'] = all_study_types if all_study_types else [study_type]
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
            current_year = datetime.now().year
            upper_bound = current_year + 1
            valid_years = []
            for match in matches:
                year = int(match)
                if 1900 <= year <= upper_bound:  # Reasonable year range
                    valid_years.append(year)
            if valid_years:
                return max(valid_years)

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
            normalized_item = self._normalize_item(item, prefer_full_abstract=apply_ranking)
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
        preserve_order = _env_true('PRESERVE_PUBMED_ORDER', False)

        if rank is True:
            apply_ranking = True
        elif rank is False:
            apply_ranking = False
        else:
            apply_ranking = self.enable_study_ranking

        if preserve_order and apply_ranking and rank is None:
            logger.info(
                "PRESERVE_PUBMED_ORDER=true; using default behavior and returning results in crawl order."
            )
            apply_ranking = False

        should_include_tags = bool(apply_ranking) or bool(self.extract_tags and (not preserve_order or self.include_tags_with_preserve_order))
        should_include_abstracts = bool(apply_ranking) or (bool(self.use_full_abstracts) and not preserve_order)
        if apply_ranking and not self.use_full_abstracts:
            logger.info(
                "Ranking requested; forcing abstract retrieval despite USE_FULL_ABSTRACTS=false to preserve scoring signals."
            )
        elif not apply_ranking and not preserve_order and self.use_full_abstracts:
            logger.info(
                "Including full abstracts (USE_FULL_ABSTRACTS=true, PRESERVE_PUBMED_ORDER=false). Set USE_FULL_ABSTRACTS=false or PRESERVE_PUBMED_ORDER=true to skip abstracts."
            )

        # Log when tags are included due to INCLUDE_TAGS_WITH_PRESERVE_ORDER
        if not apply_ranking and preserve_order and self.extract_tags and self.include_tags_with_preserve_order:
            logger.info(
                "INCLUDE_TAGS_WITH_PRESERVE_ORDER=true; including tags despite PRESERVE_PUBMED_ORDER to support metadata goals."
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

        include_tags = bool(should_include_tags)
        include_abstract = bool(should_include_abstracts)
        force_include_tags = bool(apply_ranking)
        force_include_abstract = bool(apply_ranking)

        include_tags_gate = _env_true("EASYAPI_INCLUDE_TAGS", True)
        include_abstract_gate = _env_true("EASYAPI_INCLUDE_ABSTRACT", True)
        include_tags_effective = force_include_tags or (include_tags and include_tags_gate)
        include_abstract_effective = force_include_abstract or (include_abstract and include_abstract_gate)

        cache_key = self._get_cache_key(
            enhanced_query,
            effective_max,
            apply_ranking,
            effective_enhance,
            include_tags_effective=include_tags_effective,
            include_abstract_effective=include_abstract_effective,
            preserve_order=preserve_order,
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
            include_tags_effective=include_tags_effective,
            include_abstract_effective=include_abstract_effective,
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
                    results = self._deduplicate_by_doi_pmid(results)
                    if results:
                        self._cache_results(cache_key, results)
                    elif self.cache_empty_results:
                        self._cache_results(cache_key, results)
                    return results
                raise
            except Exception as e:
                def _attempt_entrez_fallback() -> Optional[List[Dict[str, Any]]]:
                    fallback_results = self._fallback_entrez_search(
                        enhanced_query,
                        effective_max,
                        apply_ranking=apply_ranking,
                        include_abstract=include_abstract,
                    )
                    if fallback_results:
                        self._cache_results(cache_key, fallback_results)
                        logger.info(
                            "Retrieved %s PubMed articles via Entrez fallback after actor failure",
                            len(fallback_results),
                        )
                        return fallback_results
                    return None

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

                        logger.warning(
                            "Apify rate limit hit (429). Retrying in %ss. Reduce --max-items or try again later.",
                            delay,
                        )
                        time.sleep(delay)
                        continue

                # Check for subscription/permission errors first
                if ApifyApiError and isinstance(e, ApifyApiError):
                    if hasattr(e, 'status_code') and e.status_code in [401, 402, 403]:
                        logger.error(
                            "%s (status %s). Verify APIFY_TOKEN credentials or upgrade EasyAPI subscription.",
                            SUBSCRIPTION_ERROR_MESSAGE,
                            getattr(e, 'status_code', 'unknown'),
                        )
                        fallback_results = _attempt_entrez_fallback()
                        if fallback_results is not None:
                            return fallback_results
                        raise PubMedAccessError(SUBSCRIPTION_ERROR_MESSAGE) from e

                # Fallback heuristic when ApifyApiError is None or not available
                error_msg = str(e)
                error_msg_lower = error_msg.lower()
                if ApifyApiError is None or not isinstance(e, ApifyApiError):
                    if any(term in error_msg_lower for term in ['unauthorized', 'payment', 'subscription', 'forbidden']):
                        logger.error(
                            "%s (detected in error message). Verify token or subscription access.",
                            SUBSCRIPTION_ERROR_MESSAGE,
                        )
                        fallback_results = _attempt_entrez_fallback()
                        if fallback_results is not None:
                            return fallback_results
                        raise PubMedAccessError(SUBSCRIPTION_ERROR_MESSAGE) from e

                if any(pattern in error_msg_lower for pattern in APIFY_SUBSCRIPTION_PATTERNS):
                    logger.warning(
                        "Detected Apify subscription restriction (%s). Attempting Entrez fallback.",
                        error_msg,
                    )
                    fallback_results = _attempt_entrez_fallback()
                    if fallback_results is not None:
                        return fallback_results

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
                    results = self._deduplicate_by_doi_pmid(results)
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
                    fallback_results = _attempt_entrez_fallback()
                    if fallback_results is not None:
                        return fallback_results
                    raise

        return []

    def _normalize_authors(self, authors) -> str:
        """
        Normalize authors field to consistent string format.
        Always returns a comma-separated string for consistency with metadata standards.

        Args:
            authors: Authors field (could be string, list, dict, or other)

        Returns:
            Normalized authors string (comma-separated)
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

    def _normalize_item(self, item: dict, *, prefer_full_abstract: bool = False) -> dict:
        """
        Normalize actor items with fallbacks for common key variants

        Args:
            item: Raw item dictionary from Apify actor
            prefer_full_abstract: When True, prefer full abstract if available for ranking

        Returns:
            Normalized item dictionary
        """
        normalized = {}

        # Title with fallbacks - extract from multiple sources
        title = item.get('title') or item.get('articleTitle') or item.get('titleText') or ''
        if not title and 'citation' in item and item['citation']:
            # Try to extract title from citation
            citation = item['citation']
            if isinstance(citation, dict):
                title = citation.get('title') or citation.get('titleText') or ''

        # Basic title sanitization
        if title:
            title_str = str(title).strip()
            # Collapse multiple spaces and remove leading/trailing whitespace
            title_str = ' '.join(title_str.split())
            # Reject invalid values
            if title_str and title_str.lower() not in {'pubmed', '', 'unknown', 'not available'}:
                normalized['title'] = title_str

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
            # When ranking is enabled, prefer full abstract even if use_full_abstracts is False
            if (prefer_full_abstract or self.use_full_abstracts) and 'full' in abstract:
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
        def _coerce_journal_value(candidate: object) -> str:
            if isinstance(candidate, list) and candidate:
                return _coerce_journal_value(candidate[0])
            if isinstance(candidate, dict):
                for key in ('name', 'title', 'full', 'label', 'value', 'text', 'short'):
                    value = candidate.get(key)
                    if value:
                        return str(value).strip()
                return str(candidate).strip()
            return str(candidate).strip()

        structured_journal_candidates = (
            item.get('journal'),
            item.get('journalTitle'),
            item.get('journal_title'),
            item.get('source'),
            item.get('sourceTitle'),
            item.get('publication'),
            item.get('publicationTitle'),
        )

        journal = ''
        for candidate in structured_journal_candidates:
            if not candidate:
                continue
            candidate_text = _coerce_journal_value(candidate)
            if candidate_text:
                normalized_candidate = candidate_text.strip()
                if normalized_candidate and normalized_candidate.lower() not in {'pubmed', 'apify'}:
                    journal = normalized_candidate
                    break

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

        # Title with fallbacks for alternate keys - only set if not already set
        if 'title' not in normalized:
            title = item.get('title') or item.get('articleTitle') or ''
            if title:
                title_str = str(title).strip()
                # Collapse multiple spaces and remove leading/trailing whitespace
                title_str = ' '.join(title_str.split())
                # Reject invalid values
                if title_str and title_str.lower() not in {'pubmed', '', 'unknown', 'not available'}:
                    normalized['title'] = title_str

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
        # Set provider to 'apify' for backward compatibility with downstream filters
        normalized['provider'] = 'apify'
        normalized['provider_family'] = 'apify'
        # Use provider_detail/variant to distinguish specific scraper implementation
        normalized['provider_detail'] = self.scraper_provider
        normalized['provider_variant'] = self.actor_id
        if self.actor_id:
            normalized['actor_id'] = self.actor_id

        return normalized

    def _normalize_mesh_terms_to_strings(self, mesh_terms) -> List[str]:
        """
        Normalize mesh_terms to List[str] by coercing dicts/objects to string labels.
        De-duplicates case-insensitively while preserving original casing and stable ordering.
        Always returns a list of strings for consistency with metadata standards.

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
        """Convert scraped PubMed items into LangChain ``Document`` objects."""

        documents: List[Document] = []

        for item in results:
            if not isinstance(item, dict):
                logger.debug("Skipping non-dict result when building documents: %r", item)
                continue

            abstract = item.get("abstract")
            title = item.get("title")

            page_content = ""
            if isinstance(abstract, str) and abstract.strip():
                page_content = abstract.strip()
            elif isinstance(title, str) and title.strip():
                page_content = title.strip()

            if not page_content:
                logger.debug("Skipping PubMed item with no abstract/title content: %s", item.get("pmid"))
                continue

            metadata: Dict[str, Any] = {"provider": item.get("provider") or self.scraper_provider}

            if isinstance(title, str) and title.strip():
                metadata["title"] = title.strip()

            doi_value = item.get("doi") or item.get("raw_doi")
            if doi_value:
                try:
                    normalized_doi = normalize_doi(str(doi_value))
                except Exception:  # pragma: no cover - defensive against unexpected DOI values
                    normalized_doi = None
                if normalized_doi:
                    metadata["doi"] = normalized_doi

            pmid_value = item.get("pmid")
            if pmid_value:
                try:
                    normalized_pmid = normalize_pmid(str(pmid_value))
                except Exception:  # pragma: no cover - defensive against unexpected PMID values
                    normalized_pmid = None
                if normalized_pmid:
                    metadata["pmid"] = normalized_pmid

            journal = item.get("journal")
            if isinstance(journal, str) and journal.strip():
                metadata["journal"] = journal.strip()

            publication_date = item.get("publication_date") or item.get("published")
            if isinstance(publication_date, str) and publication_date.strip():
                metadata["publication_date"] = publication_date.strip()

            authors_raw = item.get("authors")
            normalized_authors = self._normalize_authors(authors_raw)
            if normalized_authors:
                metadata["authors"] = normalized_authors
            elif isinstance(authors_raw, list):
                simplified_authors = [self._get_author_name(author) for author in authors_raw]
                simplified_authors = [author for author in simplified_authors if author]
                if simplified_authors:
                    metadata["authors"] = simplified_authors

            mesh_terms_raw = item.get("mesh_terms")
            normalized_mesh_terms = self._normalize_mesh_terms_to_strings(mesh_terms_raw)
            if normalized_mesh_terms:
                metadata["mesh_terms"] = normalized_mesh_terms

            url = item.get("url")
            if isinstance(url, str) and url.strip():
                metadata["url"] = url.strip()

            ranking_score = item.get("ranking_score")
            if ranking_score is not None:
                metadata["ranking_score"] = ranking_score

            # Add study metadata when present
            study_type = item.get("study_type")
            if study_type:
                metadata["study_type"] = study_type

            study_type_confidence = item.get("study_type_confidence")
            if study_type_confidence is not None:
                metadata["study_type_confidence"] = study_type_confidence

            study_types = item.get("study_types")
            if study_types:
                # Ensure study_types is a list
                if isinstance(study_types, list):
                    metadata["study_types"] = study_types
                elif isinstance(study_types, str):
                    metadata["study_types"] = [study_types]

            tags = item.get("tags")
            if tags:
                # Ensure tags is a list
                if isinstance(tags, list):
                    metadata["tags"] = tags
                elif isinstance(tags, str):
                    metadata["tags"] = [tags]

            documents.append(Document(page_content=page_content, metadata=metadata))

        logger.info("Converted %s PubMed items into %s LangChain documents", len(results), len(documents))
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

    def write_sidecar_for_pdf(self, pdf_path: Path, article: Dict[str, Any]) -> Path:
        """
        Write sidecar JSON file for a PDF with normalized PubMed metadata.

        Args:
            pdf_path: Path to the PDF file
            article: Article dictionary from PubMed search

        Returns:
            Path to the created sidecar file.
        """
        sidecar_path = pdf_path.with_suffix('.pubmed.json')

        sidecar_data: Dict[str, Any] = {}

        doi = str(article.get('doi', '')).strip()
        if doi:
            normalized_doi = normalize_doi(doi)
            if normalized_doi:
                sidecar_data['doi'] = normalized_doi

        pmid = str(article.get('pmid', '')).strip()
        if pmid:
            normalized_pmid = normalize_pmid(pmid)
            if normalized_pmid:
                sidecar_data['pmid'] = normalized_pmid

        title = str(article.get('title', '')).strip()
        if title:
            sidecar_data['title'] = title

        normalized_authors = self._normalize_authors(article.get('authors', ''))
        if normalized_authors:
            sidecar_data['authors'] = normalized_authors

        abstract = str(article.get('abstract', '')).strip()
        if abstract:
            sidecar_data['abstract'] = abstract

        publication_date = str(article.get('publication_date', '')).strip()
        if publication_date:
            sidecar_data['publication_date'] = publication_date

        journal = str(article.get('journal', '')).strip()
        if journal:
            sidecar_data['journal'] = journal

        mesh_terms_raw = article.get('mesh_terms', [])
        mesh_terms: List[str] = []
        if isinstance(mesh_terms_raw, str):
            mesh_terms = [term.strip() for term in re.split(r'[;,]', mesh_terms_raw) if term.strip()]
        elif isinstance(mesh_terms_raw, (list, tuple, set)):
            for term in mesh_terms_raw:
                term_str = str(term).strip()
                if term_str:
                    mesh_terms.append(term_str)
        elif mesh_terms_raw is not None:
            term_str = str(mesh_terms_raw).strip()
            if term_str:
                mesh_terms = [term_str]

        if mesh_terms:
            sidecar_data['mesh_terms'] = mesh_terms

        try:
            with open(sidecar_path, 'w', encoding='utf-8') as f:
                json.dump(sidecar_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Wrote sidecar file: {sidecar_path.name}")
            return sidecar_path

        except Exception as exc:
            logger.error(f"Failed to write sidecar for {pdf_path.name}: {exc}")
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

                    # Get text from first PUBMED_SCAN_PAGES pages
                    text_to_analyze = ""
                    scan_pages = _get_env_int("PUBMED_SCAN_PAGES", 3)
                    for i, page in enumerate(pdf_documents[:scan_pages]):
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
                    sidecar_path = self.write_sidecar_for_pdf(pdf_path, matched_article)
                    if sidecar_path:
                        created_count += 1
                else:
                    logger.debug(f"No PubMed match found for {pdf_path.name}")

            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {str(e)}")
                continue

        logger.info(f"Created {created_count} sidecar files out of {len(pdf_files)} PDFs")
        return created_count


def test_sidecar_standardization():
    """
    Lightweight test to verify JSON sidecar field standardization between scraper and loader.
    Tests write_sidecar_for_pdf() with a mocked article and verifies field compatibility.
    """
    import tempfile
    import os
    from pathlib import Path

    # Mock article dictionary with all expected fields
    mock_article = {
        'doi': 'DOI: 10.1016/j.example.2023.01.001',  # With prefix to test normalization
        'pmid': '12345678',
        'title': 'Test Article Title',
        'authors': ['Smith, John', 'Doe, Jane'],  # List format to test normalization
        'abstract': 'This is a test abstract for validation purposes.',
        'publication_date': '2023-01-15',
        'journal': 'Test Journal of Medicine',
        'mesh_terms': ['Drug Therapy', 'Clinical Trial', 'Humans']  # List format
    }

    # Create a temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        test_pdf_path = Path(temp_dir) / "test_article.pdf"
        sidecar_path = test_pdf_path.with_suffix('.pubmed.json')

        # Create a dummy PDF file (just for path testing)
        test_pdf_path.touch()

        try:
            # Create scraper instance and write sidecar
            scraper = PubMedScraper()
            generated_sidecar_path = scraper.write_sidecar_for_pdf(test_pdf_path, mock_article)

            # Verify sidecar file was created
            assert sidecar_path.exists(), "Sidecar file was not created"
            assert generated_sidecar_path == sidecar_path, "Returned sidecar path mismatch"

            # Load and verify sidecar content
            with open(sidecar_path, 'r', encoding='utf-8') as f:
                sidecar_data = json.load(f)

            # Verify all expected fields are present with correct types
            expected_fields = ['doi', 'pmid', 'title', 'authors', 'abstract', 'publication_date', 'journal', 'mesh_terms']
            for field in expected_fields:
                assert field in sidecar_data, f"Field '{field}' missing from sidecar"

            # Verify field types and normalization
            assert isinstance(sidecar_data['doi'], str), "DOI should be string"
            assert sidecar_data['doi'] == '10.1016/j.example.2023.01.001', "DOI not properly normalized"

            assert isinstance(sidecar_data['pmid'], str), "PMID should be string"
            assert sidecar_data['pmid'] == '12345678', "PMID not properly normalized"

            assert isinstance(sidecar_data['authors'], str), "Authors should be normalized to string"
            assert 'Smith, John' in sidecar_data['authors'], "Authors not properly normalized"

            assert isinstance(sidecar_data['mesh_terms'], list), "MeSH terms should be list"
            assert len(sidecar_data['mesh_terms']) == 3, "MeSH terms list should have 3 items"

            # Now test if PDFDocumentLoader can read the sidecar correctly
            # Import here to avoid circular imports
            try:
                from .document_loader import PDFDocumentLoader
                loader = PDFDocumentLoader(temp_dir)
                metadata = loader._extract_pubmed_metadata(test_pdf_path)

                # Verify loader can read all fields correctly
                for field in expected_fields:
                    assert field in metadata, f"Loader failed to read field '{field}' from sidecar"

                # Verify field types after loader processing
                assert isinstance(metadata['authors'], str), "Loader should return authors as string"
                assert isinstance(metadata['mesh_terms'], list), "Loader should return mesh_terms as list"

                print("✅ Sidecar standardization test passed!")
                print(f"   - Created sidecar with {len(sidecar_data)} fields")
                print(f"   - Loader successfully read {len(metadata)} fields")
                return True

            except ImportError:
                print("⚠️  Could not import PDFDocumentLoader for loader test")
                print("✅ Sidecar creation test passed!")
                return True

        except Exception as e:
            print(f"❌ Sidecar standardization test failed: {str(e)}")
            return False


def main() -> int:
    """CLI entry point for running the PubMed scraper via ``python -m src.pubmed_scraper``."""

    parser = argparse.ArgumentParser(description="Run the PubMed scraper using the EasyAPI actor.")
    parser.add_argument("query", help="PubMed search query string")
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to retrieve (defaults to configured value)",
    )

    rank_group = parser.add_mutually_exclusive_group()
    rank_group.set_defaults(rank=None)
    rank_group.add_argument(
        "--rank",
        dest="rank",
        action="store_true",
        help="Force EasyAPI study ranking regardless of environment defaults.",
    )
    rank_group.add_argument(
        "--no-rank",
        dest="rank",
        action="store_false",
        help="Disable study ranking and preserve raw PubMed ordering.",
    )

    parser.add_argument(
        "--cache-ttl-hours",
        type=float,
        default=None,
        help="Cache TTL in hours (defaults to configured value)",
    )

    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Output raw PubMed data without converting to LangChain documents",
    )

    parser.add_argument(
        "--write-sidecars",
        action="store_true",
        help="Write .pubmed.json sidecars for PDFs in DOCS_FOLDER",
    )

    args = parser.parse_args()

    try:
        # Convert hours to seconds for cache_ttl
        cache_ttl_seconds = int(args.cache_ttl_hours * 3600) if args.cache_ttl_hours else None
        scraper = PubMedScraper(cache_ttl_seconds=cache_ttl_seconds)
        results = scraper.search_pubmed(
            args.query,
            max_items=args.max_items,
            rank=args.rank,
        )
    except Exception as exc:
        print(f"Error running PubMed scraper: {exc}", file=sys.stderr)
        return 1

    total_results = len(results)
    print(f"Retrieved {total_results} PubMed results for query: '{args.query}'")

    if total_results == 0:
        return 0

    # Store raw results for potential sidecar writing
    raw_results = results

    # Convert to documents if --no-docs is not specified
    if not args.no_docs:
        results = scraper.to_documents(results)
        print(f"Converted to {len(results)} LangChain documents")

    # Write sidecars if requested
    if args.write_sidecars:
        docs_dir = os.getenv("DOCS_FOLDER", "Data/Docs")
        created = scraper.export_sidecars(docs_dir, raw_results)
        print(f"Created {created} sidecar files in {docs_dir}")

    print("Top results:")
    for index, item in enumerate(results[:3], start=1):
        if args.no_docs:
            # Raw PubMed data
            title = (item.get("title") or "(no title)").strip() if isinstance(item.get("title"), str) else "(no title)"
            doi = item.get("doi") or "n/a"
            pmid = item.get("pmid") or "n/a"
            print(f" {index}. {title}")
            print(f"    DOI: {doi} | PMID: {pmid}")
        else:
            # LangChain document
            title = item.metadata.get("title", "(no title)")
            doi = item.metadata.get("doi", "n/a")
            pmid = item.metadata.get("pmid", "n/a")
            print(f" {index}. {title}")
            print(f"    DOI: {doi} | PMID: {pmid}")
            if hasattr(item, 'page_content') and item.page_content:
                # Show first 100 characters of content
                content_preview = item.page_content[:100].replace('\n', ' ')
                print(f"    Content: {content_preview}...")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
