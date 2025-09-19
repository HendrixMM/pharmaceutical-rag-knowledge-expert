"""Pharmaceutical query orchestration with caching, ranking, and filtering."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal, Iterable, Set

from .pharmaceutical_processor import PharmaceuticalProcessor
from .pubmed_scraper import PubMedScraper
from .ranking_filter import StudyRankingFilter

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_TTL_HOURS = 24
_MAX_ITEMS_CAP = 100

_VALID_SORT_FIELDS = {"ranking_score", "recency", "date", "relevance"}
_SORT_FIELD_ALIASES = {
    "relevance": "ranking_score",
}

_SPECIES_KEYWORDS = {
    "human": {"human", "humans"},  # Removed generic "patient", "patients"
    "mouse": {"mouse", "mice"},
    "rat": {"rat", "rats"},
    "dog": {"dog", "dogs", "canine", "canines"},
    "monkey": {"monkey", "monkeys", "nonhuman primate", "macaque"},
}

_CLINICAL_STUDY_TAGS = {
    "clinical trial", "randomized controlled trial", "controlled clinical trial",
    "phase i", "phase ii", "phase iii", "phase iv", "rct", "clinical study",
    "human study", "patient study", "clinical investigation"
}

_NEGATION_TERMS = {
    "in vitro", "cell culture", "cultured cells", "tissue culture",
    "cell line", "isolated cells", "artificial"
}


def _tokenize_species_string(text: str) -> Set[str]:
    """Normalize species strings by tokenizing on non-alphanumeric characters."""
    import re
    if not text:
        return set()
    return set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))


@dataclass
class CacheMetadata:
    """Metadata describing cached payload state."""

    created_at: datetime
    expires_at: datetime
    cache_key: str
    cache_hit: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
        }


class EnhancedQueryEngine:
    """Coordinates EasyAPI PubMed searches with caching, enrichment, and ranking.

    Accepted ``sort_by`` values are ``"ranking_score"`` (default), ``"recency"``,
    or ``"date"`` which both favour recent publications.
    """

    def __init__(
        self,
        scraper: PubMedScraper,
        *,
        cache_dir: Optional[str] = None,
        cache_ttl_hours: Optional[int] = None,
        ranking_filter: Optional[StudyRankingFilter] = None,
        pharma_processor: Optional[PharmaceuticalProcessor] = None,
        enable_pharma_enrichment: bool = True,
        enable_query_enhancement: bool = True,
        enhancement_mode: Literal["and", "or", "none"] = "and",
        infer_species_on_filter: bool = True,
        strict_species_inference: bool = True,
        re_rank: bool = True,
    ) -> None:
        self.scraper = scraper
        ttl_hours = max(1, int(cache_ttl_hours or _DEFAULT_CACHE_TTL_HOURS))
        self.cache_ttl = timedelta(hours=ttl_hours)
        cache_root = os.getenv("QUERY_ENGINE_CACHE_DIR", cache_dir or "./query_cache")
        self.cache_dir = Path(cache_root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ranking_filter = ranking_filter or StudyRankingFilter()
        self.enable_pharma_enrichment = enable_pharma_enrichment
        normalized_mode = enhancement_mode.lower()
        if normalized_mode not in {"and", "or", "none"}:
            raise ValueError("enhancement_mode must be one of 'and', 'or', or 'none'")
        self.enhancement_mode = normalized_mode
        self.enable_query_enhancement = enable_query_enhancement and normalized_mode != "none"
        self.pharma_processor = pharma_processor or (
            PharmaceuticalProcessor() if self.enable_pharma_enrichment else None
        )
        self.infer_species_on_filter = infer_species_on_filter
        self.strict_species_inference = strict_species_inference
        self.re_rank = re_rank

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pharmaceutical_query(
        self,
        query: str,
        max_items: int = 30,
        sort_by: str = "relevance",
        filters: Optional[Dict[str, Any]] = None,
        include_unknown_species: bool = True,
    ) -> Dict[str, Any]:
        """Execute pharmaceutical PubMed workflow with caching and ranking."""
        if query is None or not str(query).strip():
            raise ValueError("Query string must be provided.")

        normalized_query = query.strip()
        max_items = self._validate_max_items(max_items)
        filters = filters or {}
        filters.setdefault("include_unknown_species", include_unknown_species)
        sort_by_normalized = (sort_by or "ranking_score").lower()
        sort_by_mapped = _SORT_FIELD_ALIASES.get(sort_by_normalized, sort_by_normalized)

        if sort_by_mapped not in _VALID_SORT_FIELDS:
            logger.warning("Unsupported sort_by '%s'; defaulting to 'ranking_score'", sort_by)
            sort_by_mapped = "ranking_score"
        sort_by = sort_by_mapped
        processing_start = datetime.now(timezone.utc)

        enhanced_query, enhancement_applied = self._enhance_pharmaceutical_query(normalized_query)
        filtered_cache_key = self._generate_cache_key(
            query=normalized_query,
            enhanced_query=enhanced_query,
            max_items=max_items,
            sort_by=sort_by,
            filters=filters,
        )

        cached_payload, cache_metadata = self._get_cached_result(filtered_cache_key)
        if cached_payload is not None:
            logger.debug("Cache hit for query engine (key=%s)", filtered_cache_key)
            processing_end = datetime.now(timezone.utc)
            cached_results = self._maybe_enrich_results(cached_payload.get("results", []))
            cached_meta = cached_payload.get("metadata", {})
            cached_filters = cached_meta.get("filters") or self._summarize_filters(filters)
            dedup_info = cached_meta.get("deduplication")
            cached_sort = cached_meta.get("sort_by") or sort_by
            return self._build_response(
                query=normalized_query,
                enhanced_query=enhanced_query,
                results=cached_results,
                cache_metadata=cache_metadata,
                filters_applied=cached_filters,
                enhancement_applied=enhancement_applied,
                start_time=processing_start,
                end_time=processing_end,
                sort_by=cached_sort,
                dedup_info=dedup_info,
            )

        raw_cache_key = self._generate_raw_cache_key(
            query=normalized_query,
            enhanced_query=enhanced_query,
            max_items=max_items,
        )
        raw_cached_payload, raw_cache_metadata = self._get_cached_result(raw_cache_key)
        if raw_cached_payload is not None:
            logger.debug(
                "Raw cache hit for query engine (raw_key=%s -> filtered_key=%s)",
                raw_cache_key,
                filtered_cache_key,
            )
            ranked_results = self._maybe_enrich_results(raw_cached_payload.get("results", []))
            dedup_info = (raw_cached_payload.get("metadata") or {}).get("deduplication")
            resorted_results = self._sort_ranked_results(ranked_results, sort_by)
            filtered_results, applied_filters = self._apply_filters(resorted_results, filters)
            cache_metadata = CacheMetadata(
                created_at=raw_cache_metadata.created_at,
                expires_at=raw_cache_metadata.expires_at,
                cache_key=filtered_cache_key,
                cache_hit=True,
            )

            filtered_payload = {
                "results": filtered_results,
                "metadata": {
                    "cached_at": cache_metadata.created_at.isoformat(),
                    "filters": applied_filters,
                    "deduplication": dedup_info,
                    "sort_by": sort_by,
                    "enhanced_query": enhanced_query,
                    "raw_cache_key": raw_cache_key,
                },
            }
            self._cache_result(filtered_cache_key, filtered_payload)

            processing_end = datetime.now(timezone.utc)
            return self._build_response(
                query=normalized_query,
                enhanced_query=enhanced_query,
                results=filtered_results,
                cache_metadata=cache_metadata,
                filters_applied=applied_filters,
                enhancement_applied=enhancement_applied,
                start_time=processing_start,
                end_time=processing_end,
                sort_by=sort_by,
                dedup_info=dedup_info,
            )

        try:
            raw_results = self.scraper.search_pubmed(enhanced_query, max_items=max_items)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception("PubMed search failed for %s", enhanced_query)
            processing_end = datetime.now(timezone.utc)
            error_metadata = CacheMetadata(
                created_at=processing_end,
                expires_at=processing_end + self.cache_ttl,
                cache_key=filtered_cache_key,
                cache_hit=False,
            )
            return self._build_response(
                query=normalized_query,
                enhanced_query=enhanced_query,
                results=[],
                cache_metadata=error_metadata,
                filters_applied={},
                enhancement_applied=enhancement_applied,
                start_time=processing_start,
                end_time=processing_end,
                sort_by=sort_by,
                dedup_info=None,
                error=str(exc),
            )

        # Always run deduplication before ranking
        deduped_results, dedup_info = self._deduplicate_results(raw_results)
        enriched_results = self._maybe_enrich_results(deduped_results)

        # Check if ranking scores were already provided by scraper
        has_ranking_scores = raw_results and any("ranking_score" in item for item in raw_results)

        # Apply ranking if enabled or if no ranking scores present
        if self.re_rank or not has_ranking_scores:
            ranked_results = self._rank_results(enriched_results, query=normalized_query, sort_by=sort_by)
        else:
            # Skip ranking since scraper already provided ranking scores
            ranked_results = self._sort_ranked_results(enriched_results, sort_by)
        filtered_results, applied_filters = self._apply_filters(ranked_results, filters)
        created_at = datetime.now(timezone.utc)
        cache_metadata = CacheMetadata(
            created_at=created_at,
            expires_at=created_at + self.cache_ttl,
            cache_key=filtered_cache_key,
            cache_hit=False,
        )

        cached_at_iso = cache_metadata.created_at.isoformat()
        raw_payload = {
            "results": ranked_results,
            "metadata": {
                "cached_at": cached_at_iso,
                "deduplication": dedup_info,
                "sort_by": sort_by,
                "enhanced_query": enhanced_query,
            },
        }
        self._cache_result(raw_cache_key, raw_payload)

        cache_payload = {
            "results": filtered_results,
            "metadata": {
                "cached_at": cached_at_iso,
                "filters": applied_filters,
                "deduplication": dedup_info,
                "sort_by": sort_by,
                "enhanced_query": enhanced_query,
                "raw_cache_key": raw_cache_key,
            },
        }
        self._cache_result(filtered_cache_key, cache_payload)

        processing_end = datetime.now(timezone.utc)
        return self._build_response(
            query=normalized_query,
            enhanced_query=enhanced_query,
            results=filtered_results,
            cache_metadata=cache_metadata,
            filters_applied=applied_filters,
            enhancement_applied=enhancement_applied,
            start_time=processing_start,
            end_time=processing_end,
            sort_by=sort_by,
            dedup_info=dedup_info,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_max_items(self, value: int) -> int:
        if value <= 0:
            raise ValueError("max_items must be greater than zero")
        if value > _MAX_ITEMS_CAP:
            logger.warning("max_items capped at %d", _MAX_ITEMS_CAP)
        return min(value, _MAX_ITEMS_CAP)

    def _generate_cache_key(
        self,
        *,
        query: str,
        enhanced_query: str,
        max_items: int,
        sort_by: str,
        filters: Dict[str, Any],
    ) -> str:
        payload = {
            "query": query,
            "enhanced_query": enhanced_query,
            "max_items": max_items,
            "sort_by": sort_by,
            "filters": self._normalize_for_cache(filters),
        }
        payload_str = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload_str.encode("utf-8")).hexdigest()

    def _generate_raw_cache_key(
        self,
        *,
        query: str,
        enhanced_query: str,
        max_items: int,
    ) -> str:
        return self._generate_cache_key(
            query=query,
            enhanced_query=enhanced_query,
            max_items=max_items,
            sort_by="__raw__",
            filters={},
        )

    def _get_cached_result(self, cache_key: str) -> Tuple[Optional[Dict[str, Any]], CacheMetadata]:
        cache_file = self.cache_dir / f"{cache_key}.json"
        metadata = CacheMetadata(
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            cache_key=cache_key,
            cache_hit=False,
        )

        if not cache_file.exists():
            return None, metadata

        try:
            with open(cache_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to read cache file %s: %s", cache_file, exc)
            return None, metadata

        meta = payload.get("metadata", {})
        created_at_str = meta.get("cached_at")
        if created_at_str:
            try:
                parsed_created_at = datetime.fromisoformat(created_at_str)
                # Normalize naive datetime to UTC timezone-aware
                if parsed_created_at.tzinfo is None:
                    parsed_created_at = parsed_created_at.replace(tzinfo=timezone.utc)
                metadata.created_at = parsed_created_at
            except ValueError:
                metadata.created_at = datetime.now(timezone.utc)
        else:
            # For legacy cache files without cached_at, use file modification time
            try:
                file_mtime = cache_file.stat().st_mtime
                metadata.created_at = datetime.fromtimestamp(file_mtime, tz=timezone.utc)
            except (OSError, ValueError):
                metadata.created_at = datetime.now(timezone.utc)
        metadata.expires_at = metadata.created_at + self.cache_ttl

        try:
            now_utc = datetime.now(timezone.utc)
            is_expired = now_utc >= metadata.expires_at
        except TypeError:
            # Defensive fallback for datetime comparison errors
            is_expired = True

        if is_expired:
            logger.info("Cache expired for key %s", cache_key)
            cache_file.unlink(missing_ok=True)
            return None, metadata

        metadata.cache_hit = True
        return payload, metadata

    def _cache_result(self, cache_key: str, payload: Dict[str, Any]) -> None:
        cache_file = self.cache_dir / f"{cache_key}.json"
        payload = dict(payload)
        metadata = payload.setdefault("metadata", {})
        metadata.setdefault("cached_at", datetime.now(timezone.utc).isoformat())
        metadata["ttl_hours"] = self.cache_ttl.total_seconds() / 3600
        try:
            with open(cache_file, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to write cache file %s: %s", cache_file, exc)

    def _enhance_pharmaceutical_query(self, query: str) -> Tuple[str, bool]:
        normalized = query.strip()
        if not self.enable_query_enhancement:
            return normalized, False

        lower = normalized.lower()
        enhancement_terms: List[str] = []
        if not any(token in lower for token in ["drug interaction", "pharmacokinetics", "pharmacodynamics"]):
            enhancement_terms.extend(["drug interaction", "pharmacokinetics"])
        if "cyp" not in lower:
            enhancement_terms.append("cytochrome P450")
        if "metabolism" not in lower:
            enhancement_terms.append("drug metabolism")

        if not enhancement_terms:
            return normalized, False

        if self.enhancement_mode == "or":
            clause = " OR ".join(enhancement_terms)
            enhanced_query = f"({normalized}) OR ({clause})"
        else:  # default "and"
            enhanced_query = f"{normalized} AND (" + " OR ".join(enhancement_terms) + ")"
        return enhanced_query, True

    def _maybe_enrich_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
        if not self.enable_pharma_enrichment or self.pharma_processor is None:
            return list(results)
        return [self._enrich_single_paper(paper) for paper in results]

    def _enrich_single_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(paper)
        text_fragments = [
            enriched.get("title"),
            enriched.get("abstract"),
            enriched.get("summary"),
            enriched.get("content"),
        ]
        combined_text = "\n".join(fragment for fragment in text_fragments if fragment)

        mesh_terms = enriched.get("mesh_terms") or enriched.get("mesh") or []
        normalized_mesh = self.pharma_processor.normalize_mesh_terms(mesh_terms)
        if mesh_terms:
            enriched["mesh_terms"] = normalized_mesh
        if not enriched.get("therapeutic_areas") and normalized_mesh:
            enriched["therapeutic_areas"] = self.pharma_processor.identify_therapeutic_areas(normalized_mesh)

        if combined_text:
            # Skip drug extraction if annotations already exist
            if not enriched.get("drug_annotations"):
                if not enriched.get("drug_names"):
                    annotations = self.pharma_processor.extract_drug_names(combined_text)
                    if annotations:
                        enriched["drug_names"] = [entry["name"] for entry in annotations]
                        enriched["drug_annotations"] = annotations
                elif enriched.get("drug_names"):
                    # Only extract annotations if drug names exist but annotations don't
                    annotations = self.pharma_processor.extract_drug_names(combined_text)
                    if annotations:
                        enriched["drug_annotations"] = annotations

            if not enriched.get("cyp_enzymes"):
                enriched["cyp_enzymes"] = self.pharma_processor.extract_cyp_enzymes(combined_text)

            if not enriched.get("pharmacokinetics"):
                enriched["pharmacokinetics"] = self.pharma_processor.extract_pharmacokinetic_parameters(combined_text)

            if not enriched.get("dosage_information"):
                enriched["dosage_information"] = self.pharma_processor.extract_dosage_information(combined_text)

        if self.infer_species_on_filter and not enriched.get("species"):
            inferred_species = self._infer_species_from_text(enriched, normalized_mesh)
            if inferred_species:
                enriched["species"] = inferred_species

        return enriched

    def _infer_species_from_text(
        self,
        paper: Dict[str, Any],
        mesh_terms: Optional[Iterable[str]] = None,
    ) -> List[str]:
        combined_sources = [paper.get("title"), paper.get("abstract"), paper.get("summary")]
        if mesh_terms is None:
            mesh_terms = paper.get("mesh_terms") or []
        combined_sources.extend(mesh_terms)
        combined_text = " ".join(str(value) for value in combined_sources if value)

        # Tokenize text using non-alphanumeric delimiters for stricter matching
        tokens = _tokenize_species_string(combined_text)

        # Check for negation terms that indicate non-human contexts
        has_negation = any(term in combined_text.lower() for term in _NEGATION_TERMS)

        matches: List[str] = []
        for species, keywords in _SPECIES_KEYWORDS.items():
            if species == "human" and self.strict_species_inference:
                # For humans, require either MeSH "Humans" or clinical study context
                mesh_tokens = _tokenize_species_string(" ".join(str(term) for term in mesh_terms))
                study_types = paper.get("study_types", []) or paper.get("tags", [])
                study_context = " ".join(str(tag).lower() for tag in study_types)

                # Check MeSH terms for "Humans"
                if "humans" in mesh_tokens:
                    if not has_negation:
                        matches.append(species)
                # Check for clinical study tags and human-related keywords together
                elif any(clinical_tag in study_context for clinical_tag in _CLINICAL_STUDY_TAGS):
                    if any(keyword in tokens for keyword in keywords) and not has_negation:
                        matches.append(species)
                # Allow "patient"/"patients" only with clinical context
                elif any(clinical_tag in study_context for clinical_tag in _CLINICAL_STUDY_TAGS):
                    patient_tokens = {"patient", "patients"}
                    if patient_tokens.intersection(tokens) and not has_negation:
                        matches.append(species)
            else:
                # For non-human species, use standard matching
                if any(keyword in tokens for keyword in keywords):
                    if not (has_negation and species != "human"):  # Allow in vitro for non-human
                        if species not in matches:
                            matches.append(species)

        return matches

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        seen_ids = set()
        deduped: List[Dict[str, Any]] = []
        duplicates = 0
        key_order = ["doi", "pmid", "pmcid", "title"]
        for item in results:
            identifier = None
            normalized_item = {str(key).lower(): value for key, value in item.items()}
            for key in key_order:
                value = normalized_item.get(key)
                if value:
                    identifier = f"{key}:{str(value).strip()}".lower()
                    break
            if not identifier:
                title_value = normalized_item.get("title")
                if title_value:
                    normalized_title = self._normalize_title_identifier(str(title_value))
                    if normalized_title:
                        identifier = f"title:{normalized_title}"
            if not identifier:
                identifier = json.dumps(item, sort_keys=True)
            if identifier in seen_ids:
                duplicates += 1
                continue
            seen_ids.add(identifier)
            deduped.append(item)
        dedup_info = {
            "input_size": len(results),
            "output_size": len(deduped),
            "duplicates_removed": duplicates,
        }
        return deduped, dedup_info

    def _rank_results(
        self,
        results: List[Dict[str, Any]],
        *,
        query: str,
        sort_by: str,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        try:
            ranked = self.ranking_filter.rank_studies(results, query=query)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Ranking failed, returning unranked results: %s", exc)
            ranked = list(results)
        return self._sort_ranked_results(ranked, sort_by)

    def _sort_ranked_results(self, results: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        if not results:
            return []
        sorted_results = list(results)
        if sort_by == "ranking_score":
            sorted_results.sort(key=lambda item: item.get("ranking_score", 0.0), reverse=True)
        elif sort_by in {"recency", "date"}:
            sorted_results.sort(key=self._recency_sort_key, reverse=True)
        else:  # default to ranking score ordering
            sorted_results.sort(key=lambda item: item.get("ranking_score", 0.0), reverse=True)
        return sorted_results

    def _recency_sort_key(self, item: Dict[str, Any]) -> float:
        year = item.get("publication_year") or item.get("year")
        if year is None:
            date_str = item.get("publication_date")
            if date_str:
                try:
                    year = int(str(date_str).split("-")[0])
                except ValueError:
                    year = None
        if year is None:
            return 0.0
        try:
            return float(year)
        except (TypeError, ValueError):
            return 0.0

    def _apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not results:
            return [], {}

        applied: Dict[str, Any] = {}
        year_range = filters.get("year_range")
        if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
            applied["year_range"] = list(year_range)
        else:
            year_range = None

        study_types = filters.get("study_types")
        if study_types:
            applied["study_types"] = list(study_types)

        min_sample_size = filters.get("min_sample_size")
        if not isinstance(min_sample_size, int) or min_sample_size <= 0:
            min_sample_size = None
        else:
            applied["min_sample_size"] = min_sample_size

        min_ranking_score = filters.get("min_ranking_score")
        if isinstance(min_ranking_score, (int, float)):
            min_ranking_score = float(min_ranking_score)
            applied["min_ranking_score"] = min_ranking_score
        else:
            min_ranking_score = None

        ranked_filtered = self.ranking_filter.filter_by_criteria(
            results,
            year_range=year_range,
            study_types=study_types,
            min_sample_size=min_sample_size,
            min_ranking_score=min_ranking_score,
        )

        therapeutic_areas = filters.get("therapeutic_areas")
        if therapeutic_areas:
            areas_lower = {str(area).lower() for area in therapeutic_areas}
            ranked_filtered = [
                item
                for item in ranked_filtered
                if any(str(area).lower() in areas_lower for area in item.get("therapeutic_areas", []) or [])
            ]
            applied["therapeutic_areas"] = list(therapeutic_areas)

        drug_names = filters.get("drug_names")
        if drug_names:
            raw_filters = (
                list(drug_names)
                if isinstance(drug_names, (list, tuple, set))
                else [drug_names]
            )
            canonical_filters: Set[str] = set()
            for raw in raw_filters:
                if raw is None:
                    continue
                value_str = str(raw).strip()
                if not value_str:
                    continue
                canonical_filters.add(value_str.lower())
                if self.pharma_processor:
                    for entry in self.pharma_processor.extract_drug_names(value_str):
                        canonical_filters.add(entry["name"].lower())

            # Skip filtering if canonical_filters is empty to avoid dropping all results
            if canonical_filters:
                # Memoization for per-document extracted drug sets
                doc_drug_cache: Dict[str, Set[str]] = {}

                def _extract_doc_drugs(item: Dict[str, Any]) -> Set[str]:
                    # Create a cache key based on document identifier
                    doc_id = item.get("id") or item.get("pmid") or item.get("title") or ""
                    cache_key = str(doc_id).strip() if doc_id else id(item)

                    # Return cached result if available
                    if cache_key in doc_drug_cache:
                        return doc_drug_cache[cache_key]

                    names: Set[str] = set()
                    # Skip per-document text extraction when drug_annotations are already present
                    drug_annotations = item.get("drug_annotations", []) or []
                    if drug_annotations:
                        for entry in drug_annotations:
                            if isinstance(entry, dict):
                                name = entry.get("name")
                                if name:
                                    names.add(str(name).lower())
                    else:
                        for entry in item.get("drug_names", []) or []:
                            if isinstance(entry, dict):
                                name = entry.get("name")
                                if name:
                                    names.add(str(name).lower())
                            elif entry:
                                names.add(str(entry).lower())
                        # Only run extract_drug_names if enrichment is disabled
                        if self.pharma_processor and not self.enable_pharma_enrichment:
                            doc_text = " ".join(
                                filter(
                                    None,
                                    [
                                        item.get("title"),
                                        item.get("abstract"),
                                        item.get("summary"),
                                    ],
                                )
                            )
                            if doc_text:
                                for annotation in self.pharma_processor.extract_drug_names(doc_text):
                                    names.add(annotation["name"].lower())

                    # Cache the result
                    doc_drug_cache[cache_key] = names
                    return names

                ranked_filtered = [
                    item
                    for item in ranked_filtered
                    if _extract_doc_drugs(item).intersection(canonical_filters)
                ]
                applied["drug_names"] = list(raw_filters)

        species_preference = filters.get("species_preference")
        if species_preference:
            if isinstance(species_preference, (list, tuple, set)):
                preferred_values = [str(value).lower() for value in species_preference if value]
                applied["species_preference"] = list(species_preference)
            else:
                preferred_values = [str(species_preference).lower()]
                applied["species_preference"] = species_preference

            def _normalise_species(value: Any) -> List[str]:
                if value is None:
                    return []
                if isinstance(value, str):
                    return [value.lower()]
                if isinstance(value, (list, tuple, set)):
                    return [str(item).lower() for item in value if item]
                return [str(value).lower()]

            # Check include_unknown_species flag
            include_unknown_species = filters.get("include_unknown_species", True)

            filtered_items: List[Dict[str, Any]] = []
            for item in ranked_filtered:
                species_values = _normalise_species(item.get("species"))
                if not species_values and self.infer_species_on_filter:
                    inferred = self._infer_species_from_text(item)
                    if inferred:
                        item = dict(item)
                        item.setdefault("species", inferred)
                        species_values = [value.lower() for value in inferred]

                # Handle unknown species based on flag
                if not species_values:
                    if include_unknown_species:
                        # Include documents with unknown species when flag is True
                        filtered_items.append(item)
                    # Exclude when flag is False (skip this document)
                    continue

                # Use tokenized matching to reduce false positives
                species_tokens = set()
                for value in species_values:
                    species_tokens.update(_tokenize_species_string(value))
                preferred_tokens = set()
                for pref in preferred_values:
                    preferred_tokens.update(_tokenize_species_string(pref))
                if preferred_tokens.intersection(species_tokens):
                    filtered_items.append(item)
            ranked_filtered = filtered_items

        if not ranked_filtered:
            applied = applied or {}
        return ranked_filtered, applied

    @staticmethod
    def _normalize_title_identifier(title: str) -> str:
        lowered = title.lower()
        cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in lowered)
        return " ".join(cleaned.split())

    @staticmethod
    def _cache_sort_key(item: Any) -> str:
        try:
            return json.dumps(item, sort_keys=True, ensure_ascii=False)
        except TypeError:
            return repr(item)

    @staticmethod
    def _normalize_for_cache(value: Any) -> Any:
        if isinstance(value, dict):
            normalized = {
                str(key): EnhancedQueryEngine._normalize_for_cache(val)
                for key, val in value.items()
            }
            return dict(sorted(normalized.items()))
        if isinstance(value, list):
            return [EnhancedQueryEngine._normalize_for_cache(item) for item in value]
        if isinstance(value, tuple):
            return [EnhancedQueryEngine._normalize_for_cache(item) for item in value]
        if isinstance(value, set):
            normalized_items = [EnhancedQueryEngine._normalize_for_cache(item) for item in value]
            return sorted(normalized_items, key=EnhancedQueryEngine._cache_sort_key)
        return value

    def _summarize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        summary = {}
        for key, value in (filters or {}).items():
            if value is None:
                continue
            summary[key] = value
        return summary

    def _build_response(
        self,
        *,
        query: str,
        enhanced_query: str,
        results: List[Dict[str, Any]],
        cache_metadata: CacheMetadata,
        filters_applied: Optional[Dict[str, Any]],
        enhancement_applied: bool,
        start_time: datetime,
        end_time: datetime,
        sort_by: str,
        dedup_info: Optional[Dict[str, Any]],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        response: Dict[str, Any] = {
            "query": query,
            "enhanced_query": enhanced_query,
            "results": results,
            "results_count": len(results),
            "cache": cache_metadata.to_dict(),
            "cache_hit": cache_metadata.cache_hit,
            "filters_applied": filters_applied or {},
            "processing": {
                "started_at": start_time.isoformat(),
                "completed_at": end_time.isoformat(),
                "duration_ms": duration_ms,
            },
            "sort_by": sort_by,
            "enhancement_applied": enhancement_applied,
        }
        if dedup_info is not None:
            response["deduplication"] = dedup_info
        if error is not None:
            response["error"] = error
        return response


__all__ = ["EnhancedQueryEngine"]
