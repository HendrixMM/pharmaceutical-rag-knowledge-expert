"""Study ranking and filtering utilities for pharmaceutical literature."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Set

DEFAULT_WEIGHTS = {
    "quality": 0.4,
    "recency": 0.2,
    "sample_size": 0.15,
    "species": 0.1,
    "pharma_relevance": 0.15,
}

_HIGH_QUALITY_TAGS = {
    "systematic review": 0.95,
    "meta-analysis": 0.95,
    "randomized controlled trial": 0.9,
    "clinical trial": 0.8,
    "phase iv clinical trial": 0.88,
    "phase iii clinical trial": 0.88,
    "phase ii clinical trial": 0.82,
    "phase i clinical trial": 0.78,
    "observational study": 0.7,
    "case-control studies": 0.7,
    "cohort studies": 0.7,
}

_LOWER_QUALITY_TAGS = {
    "case reports": 0.4,
    "animal study": 0.4,
    "in vitro": 0.35,
    "letter": 0.3,
    "editorial": 0.25,
}

_SPECIES_PRIORITIES = {
    "human": 1.0,
    "humans": 1.0,
    "rat": 0.6,
    "rats": 0.6,
    "mouse": 0.55,
    "mice": 0.55,
    "dog": 0.5,
    "dogs": 0.5,
    "in vitro": 0.35,
}

_PHARMA_RELEVANCE_KEYWORDS = {
    "pharmacokinetics": 0.3,
    "pharmacodynamics": 0.25,
    "drug interaction": 0.35,
    "cytochrome": 0.2,
    "cyp": 0.2,
    "clearance": 0.15,
    "half-life": 0.2,
    "auc": 0.2,
    "exposure": 0.1,
}

_KNOWN_CYP_ENZYMES = {
    "cyp1a2",
    "cyp2b6",
    "cyp2c8",
    "cyp2c9",
    "cyp2c19",
    "cyp2d6",
    "cyp2e1",
    "cyp3a4",
    "cyp3a5",
    "cyp3a7",
}

_SAMPLE_SIZE_PATTERNS = [
    # Prefer explicit declarations such as "n=120"
    (
        "n_equals",
        re.compile(r"\bn\s*=\s*(?P<count>\d{1,3}(?:,\d{3})*|\d{1,5})\b", re.IGNORECASE),
    ),
    # Accept "patients = 85" or "participants: 60"
    (
        "term_assignment",
        re.compile(
            r"\b(?:patients|subjects|participants)\s*(?:[:=]\s*)?(?P<count>\d{1,3}(?:,\d{3})*|\d{1,5})\b",
            re.IGNORECASE,
        ),
    ),
    # Handle trailing descriptors such as "120 patients"
    (
        "term_trailing",
        re.compile(
            r"\b(?P<count>\d{1,3}(?:,\d{3})*|\d{1,5})\s+(?:patients|subjects|participants)\b",
            re.IGNORECASE,
        ),
    ),
]


@dataclass
class RankingBreakdown:
    quality: float
    recency: float
    sample_size: float
    species: float
    pharma_relevance: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "quality": self.quality,
            "recency": self.recency,
            "sample_size": self.sample_size,
            "species": self.species,
            "pharma_relevance": self.pharma_relevance,
        }


class StudyRankingFilter:
    """Ranks and filters PubMed study metadata for pharmaceutical relevance."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        *,
        recency_decay_years: Optional[float] = None,
        diversity_threshold: float = 0.92,
        max_pairs: Optional[int] = 5000,
    ) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self._normalise_weights()
        decay_value = float(recency_decay_years or 10.0)
        if decay_value <= 0:
            raise ValueError("recency_decay_years must be greater than zero")
        self.recency_decay_years = decay_value
        self.diversity_threshold = float(diversity_threshold)
        if not (0.0 < self.diversity_threshold < 1.0):
            raise ValueError("diversity_threshold must be between 0 and 1")
        if max_pairs is not None and max_pairs <= 0:
            max_pairs = None
        self.max_pairs = max_pairs

    def rank_studies(self, papers: List[Dict[str, Any]], query: str = "") -> List[Dict[str, Any]]:
        """Attach ranking metadata to each paper and sort in-place."""
        if not papers:
            return papers

        ranked: List[Dict[str, Any]] = []
        diversified = self.apply_diversity_filter(papers)
        for paper in diversified:
            tags = paper.get("study_types") or paper.get("tags") or []
            quality = self._calculate_study_quality_score(tags)
            recency = self._calculate_recency_score(paper.get("publication_year") or paper.get("year"))
            sample_size = self._calculate_sample_size_score(paper)
            species_source = " ".join(
                str(paper.get(key, "")) for key in ("species", "mesh_terms", "title", "abstract")
            )
            species = self._calculate_species_preference_score(species_source)
            pharma_relevance = self._calculate_pharmaceutical_relevance_score(paper, query)

            breakdown = RankingBreakdown(
                quality=quality,
                recency=recency,
                sample_size=sample_size,
                species=species,
                pharma_relevance=pharma_relevance,
            )
            score = self._weighted_score(breakdown)

            paper = dict(paper)  # shallow copy to avoid mutating input
            paper["ranking_score"] = round(score, 4)
            paper["ranking_breakdown"] = breakdown.to_dict()
            ranked.append(paper)

        ranked.sort(key=lambda item: item.get("ranking_score", 0.0), reverse=True)
        return ranked

    def apply_diversity_filter(
        self,
        papers: Iterable[Dict[str, Any]],
        *,
        threshold: Optional[float] = None,
        max_pairs: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Drop near duplicates based on title/abstract similarity with optimizations for large sets."""
        threshold = self.diversity_threshold if threshold is None else float(threshold)
        papers_list = list(papers)
        input_size = len(papers_list)

        # Adaptive max_pairs based on input size
        if max_pairs is None:
            if input_size <= 100:
                max_pairs = None  # No limit for small sets
            elif input_size <= 500:
                max_pairs = 5000
            elif input_size <= 1000:
                max_pairs = 10000
            else:
                max_pairs = 15000  # Cap for very large sets
        else:
            max_pairs = None if max_pairs <= 0 else int(max_pairs)

        filtered: List[Dict[str, Any]] = []
        seen_texts: List[str] = []
        seen_signatures: List[Set[str]] = []
        seen_hashes: List[int] = []  # Fast prefilter using hash
        comparisons = 0

        for paper in papers_list:
            full_text = " ".join(
                str(paper.get(key, "")) for key in ("title", "abstract", "summary")
            ).strip()
            normalised = re.sub(r"\s+", " ", full_text.lower())
            if not normalised:
                filtered.append(paper)
                continue

            # Fast prefilter using simple hash
            text_hash = hash(normalised)
            if text_hash in seen_hashes:
                # Potential exact duplicate, skip detailed comparison
                continue

            is_duplicate = False
            signature = set(normalised.split()[:50])

            if max_pairs is not None and comparisons >= max_pairs:
                seen_texts.append(normalised)
                seen_signatures.append(signature)
                seen_hashes.append(text_hash)
                filtered.append(paper)
                continue

            # Fast signature-based prefilter
            for i, (seen, seen_sig) in enumerate(zip(seen_texts, seen_signatures)):
                if max_pairs is not None and comparisons >= max_pairs:
                    break

                # Quick signature overlap check (fast prefilter)
                if seen_sig and signature:
                    overlap_ratio = len(signature.intersection(seen_sig)) / len(signature.union(seen_sig))
                    if overlap_ratio < 0.3:  # Skip if very little overlap
                        continue

                if not self._lengths_similar(normalised, seen):
                    continue

                comparisons += 1
                similarity = SequenceMatcher(None, normalised, seen).ratio()
                if similarity >= threshold:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            seen_texts.append(normalised)
            seen_signatures.append(signature)
            seen_hashes.append(text_hash)
            filtered.append(paper)

        return filtered

    def filter_by_criteria(
        self,
        papers: Iterable[Dict[str, Any]],
        *,
        year_range: Optional[List[int]] = None,
        study_types: Optional[Iterable[str]] = None,
        min_sample_size: Optional[int] = None,
        min_ranking_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filter ranked papers according to user-provided criteria."""
        def _within_year_range(paper: Dict[str, Any]) -> bool:
            if not year_range or len(year_range) != 2:
                return True
            year = paper.get("publication_year") or paper.get("year")
            if not isinstance(year, int):
                return True
            start, end = year_range
            if start is not None and year < start:
                return False
            if end is not None and year > end:
                return False
            return True

        def _matches_study_type(paper: Dict[str, Any]) -> bool:
            if not study_types:
                return True
            types_available = [
                str(value).lower()
                for value in (paper.get("study_types") or paper.get("tags") or [])
            ]
            allowed = {str(st).lower() for st in study_types}
            return bool(allowed.intersection(types_available))

        def _meets_sample_size(paper: Dict[str, Any]) -> bool:
            if min_sample_size is None or min_sample_size <= 0:
                return True
            sample_size = paper.get("sample_size")
            if isinstance(sample_size, (int, float)):
                return sample_size >= min_sample_size
            abstract = paper.get("abstract") or ""
            estimated = self._estimate_sample_size_from_text(abstract)
            return estimated is None or estimated >= min_sample_size

        def _meets_ranking(paper: Dict[str, Any]) -> bool:
            if min_ranking_score is None:
                return True
            return paper.get("ranking_score", 0.0) >= float(min_ranking_score)

        filtered: List[Dict[str, Any]] = []
        for paper in papers:
            if not (_within_year_range(paper) and _matches_study_type(paper) and _meets_sample_size(paper) and _meets_ranking(paper)):
                continue
            filtered.append(paper)
        return filtered

    def get_ranking_explanation(self, paper: Dict[str, Any], *, verbose: bool = False) -> str:
        """Return a human-readable explanation for the ranking score."""
        breakdown = paper.get("ranking_breakdown") or {}
        score = paper.get("ranking_score", 0.0)
        parts = [f"overall score {score:.2f}"]
        for key in ("quality", "recency", "sample_size", "species", "pharma_relevance"):
            value = breakdown.get(key)
            if value is None:
                continue
            parts.append(f"{key.replace('_', ' ')} {float(value):.2f}")

        if verbose:
            # Extract basic details from existing paper metadata rather than
            # referencing non-existent ranking_details
            drug_names = paper.get("drug_names", [])
            if drug_names:
                drug_list = [str(drug) for drug in drug_names[:3]]
                if drug_list:
                    parts.append("drugs " + ", ".join(drug_list))

            mesh_terms = paper.get("mesh_terms", [])
            if mesh_terms:
                mesh_list = [str(term) for term in mesh_terms[:3]]
                if mesh_list:
                    parts.append("mesh " + ", ".join(mesh_list))

            cyp_enzymes = paper.get("cyp_enzymes", [])
            if cyp_enzymes:
                cyp_list = [str(enzyme) for enzyme in cyp_enzymes[:3]]
                if cyp_list:
                    parts.append("cyp " + ", ".join(cyp_list))

            # Add year and study type context when available
            year = paper.get("publication_year") or paper.get("year")
            if year:
                parts.append(f"year {year}")

            # Always include study type if available
            study_type = paper.get("study_type")
            if study_type:
                parts.append(f"type {study_type}")
            else:
                # If no primary study_type, use first one or two from study_types
                study_types = paper.get("study_types", [])
                if study_types and isinstance(study_types, list):
                    if len(study_types) == 1:
                        parts.append(f"type {study_types[0]}")
                    elif len(study_types) >= 2:
                        parts.append(f"type {study_types[0]}, {study_types[1]}")
                    else:
                        parts.append(f"type {study_types[0]}")

        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _normalise_weights(self) -> None:
        total = sum(abs(value) for value in self.weights.values()) or 1.0
        for key, value in list(self.weights.items()):
            self.weights[key] = max(0.0, float(value)) / total

    def _weighted_score(self, breakdown: RankingBreakdown) -> float:
        return sum(
            self.weights[name] * getattr(breakdown, name)
            for name in self.weights
            if hasattr(breakdown, name)
        )

    def _calculate_study_quality_score(self, tags: Any) -> float:
        if not tags:
            return 0.5
        if isinstance(tags, str):
            tags_iterable = [tags]
        else:
            tags_iterable = list(tags)
        scores: List[float] = []
        for tag in tags_iterable:
            lower = str(tag).lower()
            if lower in _HIGH_QUALITY_TAGS:
                scores.append(_HIGH_QUALITY_TAGS[lower])
            elif lower in _LOWER_QUALITY_TAGS:
                scores.append(_LOWER_QUALITY_TAGS[lower])
            else:
                scores.append(0.6)
        return sum(scores) / len(scores)

    def _calculate_recency_score(self, year: Any) -> float:
        current_year = datetime.utcnow().year
        try:
            year_int = int(year)
        except (TypeError, ValueError):
            return 0.5
        age = max(0, current_year - year_int)
        # Exponential decay favouring recent publications, configurable via recency_decay_years
        return math.exp(-age / self.recency_decay_years)

    def _calculate_sample_size_score(self, paper: Dict[str, Any]) -> float:
        sample_size = paper.get("sample_size")
        if isinstance(sample_size, (int, float)):
            sample_size = int(sample_size)
        elif isinstance(sample_size, str):
            sample_size = self._parse_sample_size_candidate(sample_size.replace(",", ""))
        else:
            sample_size = None

        if sample_size is None:
            abstract = paper.get("abstract") or paper.get("summary") or ""
            sample_size = self._estimate_sample_size_from_text(abstract)

        if sample_size is None:
            return 0.55
        if sample_size >= 1000:
            return 1.0
        if sample_size >= 300:
            return 0.9
        if sample_size >= 100:
            return 0.8
        if sample_size >= 50:
            return 0.65
        if sample_size >= 20:
            return 0.5
        return 0.35

    def _estimate_sample_size_from_text(self, text: str) -> Optional[int]:
        if not text:
            return None

        counts_by_priority: Dict[str, List[int]] = {name: [] for name, _ in _SAMPLE_SIZE_PATTERNS}
        for name, pattern in _SAMPLE_SIZE_PATTERNS:
            for match in pattern.finditer(text):
                raw_value = match.group("count")
                parsed = self._parse_sample_size_candidate(raw_value)
                if parsed is not None:
                    counts_by_priority[name].append(parsed)

        for priority_name in ("n_equals", "term_assignment", "term_trailing"):
            values = counts_by_priority.get(priority_name, [])
            if values:
                return max(values)
        return None

    def _parse_sample_size_candidate(self, raw_value: Optional[str]) -> Optional[int]:
        if not raw_value:
            return None
        normalized = raw_value.replace(",", "")
        try:
            value = int(normalized)
        except ValueError:
            return None
        if len(normalized) == 4 and 1900 <= value <= 2100:
            return None
        return value

    def _calculate_species_preference_score(self, text: str) -> float:
        if not text:
            return 0.5

        # Tokenize text for stricter matching
        import re
        tokens = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))

        # Check for negation terms
        negation_terms = {"in vitro", "cell culture", "cultured cells", "tissue culture", "cell line"}
        has_negation = any(term in text.lower() for term in negation_terms)

        # Check for exact token matches to reduce false positives
        for species, weight in _SPECIES_PRIORITIES.items():
            if species in tokens:
                # Down-rank human scores if negation terms present
                if species in ("human", "humans") and has_negation:
                    return weight * 0.3  # Significantly reduce score
                return weight
        return 0.45

    def _calculate_pharmaceutical_relevance_score(self, paper: Dict[str, Any], query: str = "") -> float:
        text_fields = [
            paper.get("title"),
            paper.get("abstract"),
            paper.get("summary"),
            " ".join(str(term) for term in (paper.get("mesh_terms") or [])),
            " ".join(str(term) for term in (paper.get("keywords") or [])),
        ]
        combined_text = " ".join(fragment for fragment in text_fields if fragment)
        lower_text = combined_text.lower()
        if not lower_text:
            lower_text = ""

        score = 0.0
        for keyword, weight in _PHARMA_RELEVANCE_KEYWORDS.items():
            if keyword in lower_text:
                score += weight
        if query:
            lower_query = query.lower()
            for keyword, weight in _PHARMA_RELEVANCE_KEYWORDS.items():
                if keyword in lower_query:
                    score += weight * 0.5

        query_terms = {
            token
            for token in re.split(r"[^a-z0-9]+", query.lower())
            if len(token) >= 4
        }

        drug_terms: Set[str] = set()
        for entry in paper.get("drug_names", []) or []:
            if isinstance(entry, dict):
                name = entry.get("name")
            else:
                name = entry
            if name:
                drug_terms.add(str(name).lower())

        if drug_terms:
            score += min(0.2, 0.05 * len(drug_terms))

        if query_terms:
            overlap = {term for term in query_terms if term in lower_text or term in drug_terms}
            if overlap:
                score += min(0.2, 0.05 * len(overlap))

        mesh_terms = {str(term).lower() for term in (paper.get("mesh_terms") or []) if term}
        if query_terms and mesh_terms:
            mesh_overlap = query_terms.intersection(mesh_terms)
            if mesh_overlap:
                score += min(0.15, 0.05 * len(mesh_overlap))

        cyp_entries = {
            str(entry).lower().replace(" ", "")
            for entry in (paper.get("cyp_enzymes") or [])
            if entry
        }
        if cyp_entries:
            overlap = cyp_entries & _KNOWN_CYP_ENZYMES
            if query_terms:
                overlap |= {cyp for cyp in cyp_entries if any(term in cyp for term in query_terms)}
            if overlap:
                score += min(0.2, 0.1 * len(overlap))

        return min(score, 1.0)

    @staticmethod
    def _lengths_similar(lhs: str, rhs: str) -> bool:
        max_len = max(len(lhs), len(rhs), 1)
        diff = abs(len(lhs) - len(rhs))
        return diff / max_len <= 0.6


__all__ = ["StudyRankingFilter"]
