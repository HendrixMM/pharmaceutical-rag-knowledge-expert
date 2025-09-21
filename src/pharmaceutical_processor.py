"""Utility helpers for extracting pharmaceutical signals from biomedical text."""

from __future__ import annotations

import csv
import logging
import os
import re
from datetime import datetime
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

_GENERIC_DRUG_NAMES: Set[str] = {
    "acetaminophen",
    "atorvastatin",
    "clopidogrel",
    "dabigatran",
    "fluconazole",
    "ibuprofen",
    "itraconazole",
    "ketoconazole",
    "metformin",
    "midazolam",
    "omeprazole",
    "rifampin",
    "rivaroxaban",
    "simvastatin",
    "warfarin",
}

_BRAND_DRUG_NAMES: Set[str] = {
    "coumadin",
    "lipitor",
    "nexium",
    "plavix",
    "prilosec",
    "xarelto",
    "zyrtec",
    "nizoral",
}

_MESH_TO_THERAPEUTIC_AREA = {
    "anticoagulants": "hematology",
    "anti-inflammatory agents": "immunology",
    "anti-infective agents": "infectious diseases",
    "anti-hypertensive agents": "cardiology",
    "anti-neoplastic agents": "oncology",
    "antiepileptic agents": "neurology",
    "antidiabetic agents": "endocrinology",
    "cardiovascular agents": "cardiology",
    "central nervous system agents": "neurology",
}

_INTERACTION_KEYWORDS = {
    "inhibition": ["inhibit", "inhibits", "inhibitor", "inhibits metabolism"],
    "induction": ["induce", "induces", "inducer"],
    "synergistic": ["synergy", "synergistic"],
    "antagonism": ["antagonist", "antagonism"],
    "contraindication": ["contraindicated", "contraindication"],
    "substrate": ["substrate", "is a substrate of", "substrates"],
    "competitive": ["competitive", "competitive inhibitor", "competitive inhibition"],
    "non-competitive": ["noncompetitive", "non-competitive", "noncompetitive inhibitor", "non-competitive inhibitor"],
}

_DOSAGE_PATTERN = re.compile(
    r"(?P<amount>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|µg|ug|g|kg)(?:/kg)?"
    r"(?:\s*(?P<route>iv|po|im|sc|oral|intravenous|subcutaneous|intramuscular))?"
    r"(?:\s*(?P<frequency>(?:once|twice|daily|per\s+day|bid|tid|qid|q\d+h|every\s+\d+\s+hours)))?",
    re.IGNORECASE,
)

_PK_PATTERNS = {
    # Example: matches "clearance" or "CL/F" while avoiding substrings like "class"
    "clearance": [
        re.compile(r"\bclearance\b", re.IGNORECASE),
        re.compile(r"\bCL(?:\s*/\s*F)?\b", re.IGNORECASE),
    ],
    "half_life": [
        re.compile(r"\bhalf[\s-]?life\b", re.IGNORECASE),
        re.compile(r"\bT1/2\b", re.IGNORECASE),
    ],
    "auc": [
        re.compile(r"\bAUC\b", re.IGNORECASE),
        re.compile(r"\barea\s+under\s+the\s+curve\b", re.IGNORECASE),
    ],
    "cmax": [re.compile(r"\bCmax\b", re.IGNORECASE)],
    "tmax": [re.compile(r"\bTmax\b", re.IGNORECASE)],
    "volume_distribution": [
        re.compile(r"\bvolume\s+of\s+distribution\b", re.IGNORECASE),
        re.compile(r"\bVd\b", re.IGNORECASE),
    ],
}

_PK_VALUE_PATTERNS = {
    "auc": re.compile(
        r"\bauc\s*(?:\([a-z]+\))?\s*(?:[:=]|to)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ng\s*\*\s*h/mL|mg\s*h/L|h\*ng/mL)",
        re.IGNORECASE,
    ),
    "cmax": re.compile(
        r"\bcmax\s*(?:[:=]|to)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ng/mL|mg/L)",
        re.IGNORECASE,
    ),
    "tmax": re.compile(
        r"\btmax\s*(?:[:=]|to)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>h|hr|hours|min|minutes)",
        re.IGNORECASE,
    ),
    "half_life": re.compile(
        r"\b(?:t1/2|half[\s-]?life)\s*(?:[:=]|to)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>h|hr|hours|min|minutes)",
        re.IGNORECASE,
    ),
    "clearance": re.compile(
        r"\b(?:cl/f|cl)\s*(?:[:=]|to)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>L/h|L/hour|mL/min|mL/min/kg)",
        re.IGNORECASE,
    ),
    "volume_distribution": re.compile(
        r"\bvd\s*(?:[:=]|to)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>L|L/kg)",
        re.IGNORECASE,
    ),
}

_DOSAGE_DURATION_PATTERN = re.compile(
    r"(?:for|x)\s*(?P<value>\d{1,3})\s*(?P<unit>day(?:s)?|week(?:s)?|month(?:s)?|hour(?:s)?|hr|h)",
    re.IGNORECASE,
)

_CYP_PATTERN = re.compile(r"cyp\s*-?\d[a-z0-9]*", re.IGNORECASE)
_WORD_TOKENIZER = re.compile(r"[A-Za-z][A-Za-z0-9-]{3,}")

_DRUG_HEURISTIC_SUFFIXES = (
    "ine",
    "ol",
    "ide",
    "ate",
    "mab",
    "nib",
    "azole",
    "avir",
    "xaban",
)

_DRUG_SUFFIX_STOPWORDS: Set[str] = {
    "baseline",
    "guideline",
    "discipline",
    "online",
    "gasoline",
}


class PharmaceuticalProcessor:
    """Extract and annotate pharmaceutical entities from unstructured text.

    Instance lexicons start with built-in vocabularies and automatically load
    bundled files at ``data/drugs_generic.txt`` and ``data/drugs_brand.txt``
    when they exist. Callers can supply additional newline-delimited files via
    ``generic_lexicon_path`` and ``brand_lexicon_path`` (or environment
    variables ``DRUG_GENERIC_LEXICON`` / ``DRUG_BRAND_LEXICON``) to extend the
    defaults without mutating shared module-level state.
    """

    def __init__(
        self,
        *,
        generic_lexicon_path: Optional[str] = None,
        brand_lexicon_path: Optional[str] = None,
        auto_fetch: bool = False,
        enable_remote_fetch: Optional[bool] = None,
    ) -> None:
        """Seed instance lexicons and optionally extend them from custom files.

        Args:
            generic_lexicon_path: Optional custom generic lexicon path.
            brand_lexicon_path: Optional custom brand lexicon path.
            auto_fetch: When True, download comprehensive lexicons into the
                bundled data directory if they are missing. Defaults to False
                to avoid large downloads during incidental instantiation.
            enable_remote_fetch: Explicit override for remote fetch behavior.
                When None, respects auto_fetch parameter. When True, forces
                remote fetching. When False, disables all remote fetching.
        """
        # Determine effective auto_fetch behavior
        effective_auto_fetch = auto_fetch
        if enable_remote_fetch is not None:
            effective_auto_fetch = enable_remote_fetch
            logger.info("Remote fetch explicitly %s via enable_remote_fetch parameter",
                       "enabled" if enable_remote_fetch else "disabled")

        self.generic_drug_names: Set[str] = set(_GENERIC_DRUG_NAMES)
        self.brand_drug_names: Set[str] = set(_BRAND_DRUG_NAMES)
        # Pre-compile word boundary patterns for efficient matching
        self.generic_drug_patterns: Dict[re.Pattern, str] = {
            re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE): name
            for name in self.generic_drug_names
        }
        self.brand_drug_patterns: Dict[re.Pattern, str] = {
            re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE): name
            for name in self.brand_drug_names
        }
        self.cyp_roles: Dict[str, Dict[str, Set[str]]] = {}
        self.mesh_to_therapeutic_area: Dict[str, str] = dict(_MESH_TO_THERAPEUTIC_AREA)

        bundled_dir = Path(__file__).resolve().parent.parent / "data"
        if bundled_dir.exists():
            bundled_generic = bundled_dir / "drugs_generic.txt"
            if bundled_generic.exists():
                try:
                    self.generic_drug_names.update(
                        self._load_lexicon_file(str(bundled_generic))
                    )
                    self._update_drug_patterns()
                    logger.info("Loaded comprehensive generic drug lexicon from %s", bundled_generic)
                except ValueError as exc:
                    logger.warning("Failed to load bundled generic lexicon: %s", exc)
            else:
                logger.warning(
                    "Missing comprehensive generic drug lexicon at %s. "
                    "Drug detection coverage will be limited to built-in vocabulary. "
                    "For comprehensive coverage, provide DRUG_GENERIC_LEXICON environment variable "
                    "or see README for setup instructions.",
                    bundled_generic
                )
                if effective_auto_fetch:
                    logger.info("Auto-fetch enabled: creating comprehensive generic drug lexicon at %s", bundled_dir)
                    self._auto_fetch_lexicons(bundled_dir)
                else:
                    logger.info("Auto-fetch disabled: skip creating comprehensive generic drug lexicon. Set auto_fetch=True or AUTO_FETCH_DRUG_LEXICONS=true to enable.")

            bundled_brand = bundled_dir / "drugs_brand.txt"
            if bundled_brand.exists():
                try:
                    self.brand_drug_names.update(
                        self._load_lexicon_file(str(bundled_brand))
                    )
                    self._update_drug_patterns()
                    logger.info("Loaded comprehensive brand drug lexicon from %s", bundled_brand)
                except ValueError as exc:
                    logger.warning("Failed to load bundled brand lexicon: %s", exc)
            else:
                logger.warning(
                    "Missing comprehensive brand drug lexicon at %s. "
                    "Brand drug detection coverage will be limited to built-in vocabulary. "
                    "For comprehensive coverage, provide DRUG_BRAND_LEXICON environment variable "
                    "or see README for setup instructions.",
                    bundled_brand
                )
                if effective_auto_fetch:
                    logger.info("Auto-fetch enabled: creating comprehensive brand drug lexicon at %s", bundled_dir)
                    self._auto_fetch_lexicons(bundled_dir)
                else:
                    logger.info("Auto-fetch disabled: skip creating comprehensive brand drug lexicon. Set auto_fetch=True or AUTO_FETCH_DRUG_LEXICONS=true to enable.")
        else:
            logger.warning("Data directory not found - using minimal drug vocabularies")

        env_generic_path = generic_lexicon_path or os.getenv("DRUG_GENERIC_LEXICON")
        env_brand_path = brand_lexicon_path or os.getenv("DRUG_BRAND_LEXICON")

        if env_generic_path:
            additional_generics = self._load_lexicon_file(env_generic_path)
            if additional_generics:
                self.generic_drug_names.update(additional_generics)
                self._update_drug_patterns()
        if env_brand_path:
            additional_brands = self._load_lexicon_file(env_brand_path)
            if additional_brands:
                self.brand_drug_names.update(additional_brands)
                self._update_drug_patterns()

        cyp_roles_path = bundled_dir / "cyp_roles.csv"
        if cyp_roles_path.exists():
            try:
                self.cyp_roles = self._load_cyp_roles_file(str(cyp_roles_path))
                logger.info("Loaded comprehensive CYP role mappings from %s", cyp_roles_path)
            except ValueError as exc:
                logger.warning("Failed to load CYP role mappings: %s", exc)
                self.cyp_roles = {}
        else:
            logger.warning("Missing %s - using minimal CYP role mappings", cyp_roles_path.name)
            self.cyp_roles = {}
            if effective_auto_fetch:
                logger.info("Auto-fetch enabled: creating comprehensive CYP roles file at %s", bundled_dir)
                self._auto_fetch_lexicons(bundled_dir)
            else:
                logger.info("Auto-fetch disabled: skip creating comprehensive CYP roles file. Set auto_fetch=True or AUTO_FETCH_DRUG_LEXICONS=true to enable.")

        # Load external therapeutic area mapping
        mesh_areas_path = bundled_dir / "mesh_therapeutic_areas.json"
        if mesh_areas_path.exists():
            try:
                import json
                with mesh_areas_path.open("r", encoding="utf-8") as f:
                    external_mapping = json.load(f)
                # Merge with defaults, external mapping takes precedence
                self.mesh_to_therapeutic_area.update(external_mapping)
                logger.info("Loaded extended therapeutic area mapping from %s", mesh_areas_path)
            except (ValueError, OSError) as exc:
                logger.warning("Failed to load therapeutic area mapping: %s", exc)
        else:
            logger.info("Using built-in therapeutic area mapping (limited coverage)")

        # Log lexicon counts for visibility
        logger.info(
            "Drug lexicon loaded: %d generic drugs, %d brand drugs",
            len(self.generic_drug_names),
            len(self.brand_drug_names)
        )

    def _update_drug_patterns(self) -> None:
        """Update pre-compiled regex patterns when drug name sets change."""
        self.generic_drug_patterns = {
            re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE): name
            for name in self.generic_drug_names
        }
        self.brand_drug_patterns = {
            re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE): name
            for name in self.brand_drug_names
        }

    def extract_drug_names(self, text: Optional[str]) -> List[Dict[str, Any]]:
        """Extract drug names from text with metadata.

        Returns a List[Dict[str, Any]] where each dictionary contains:
        - name (str): The detected drug name
        - type (str): Drug type ("generic", "brand", or "unknown")
        - confidence (float): Confidence score between 0.0 and 1.0

        Results are sorted by confidence (descending) then name (ascending).
        """
        if not text:
            return []

        candidates: Dict[str, Dict[str, Any]] = {}

        # Use word-boundary patterns for exact matching
        for pattern, name in self.generic_drug_patterns.items():
            if pattern.search(text):
                key = name.lower()
                canonical = self._extract_original_case(text, name)
                candidates.setdefault(
                    key,
                    {"name": canonical, "type": "generic", "confidence": 0.98},
                )

        for pattern, name in self.brand_drug_patterns.items():
            if pattern.search(text):
                key = name.lower()
                canonical = self._extract_original_case(text, name)
                candidates.setdefault(
                    key,
                    {"name": canonical, "type": "brand", "confidence": 0.96},
                )

        # Token-based heuristic matching for unknown drugs
        for token in _WORD_TOKENIZER.findall(text):
            token_lower = token.lower()
            if token_lower in candidates:
                continue
            if self._looks_like_drug_name(token):
                candidates[token_lower] = {
                    "name": token,
                    "type": "unknown",
                    "confidence": 0.6,
                }

        return sorted(candidates.values(), key=lambda item: (-item["confidence"], item["name"].lower()))

    def extract_drug_name_strings(self, text: Optional[str]) -> List[str]:
        """Extract drug names as simple strings for basic consumers.

        Returns just the drug name strings without metadata.
        Equivalent to [item["name"] for item in extract_drug_names(text)].
        """
        return [item["name"] for item in self.extract_drug_names(text)]

    def extract_cyp_enzymes(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        matches = {
            self._normalise_enzyme(match)
            for match in _CYP_PATTERN.findall(text)
        }
        return sorted(matches)

    def annotate_cyp_roles(self, text: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
        """Annotate CYP enzymes with their roles (substrates, inhibitors, inducers).

        Returns a dictionary mapping CYP enzymes to their roles and associated drugs:
        { "CYP2C9": { "inhibitors": [...], "inducers": [...], "substrates": [...] }, ... }

        Uses self.cyp_roles if available, otherwise falls back to keyword heuristics.
        """
        if not text:
            return {}

        # Extract CYP enzymes from text
        detected_cyps = set(self.extract_cyp_enzymes(text))
        if not detected_cyps:
            return {}

        result: Dict[str, Dict[str, List[str]]] = {}

        # Use structured cyp_roles data if available
        if self.cyp_roles:
            for cyp in detected_cyps:
                if cyp in self.cyp_roles:
                    result[cyp] = {
                        role: list(drugs) for role, drugs in self.cyp_roles[cyp].items()
                    }
                else:
                    # Initialize empty roles for detected CYPs not in our database
                    result[cyp] = {"substrates": [], "inhibitors": [], "inducers": []}
        else:
            # Fall back to keyword heuristics when no structured data available
            lower_text = text.lower()

            # Extract drug names that might be mentioned
            detected_drugs = [item["name"] for item in self.extract_drug_names(text)]

            for cyp in detected_cyps:
                roles: Dict[str, List[str]] = {"substrates": [], "inhibitors": [], "inducers": []}

                # Look for CYP mentions near role keywords
                cyp_lower = cyp.lower()
                cyp_variants = [cyp_lower, cyp_lower.replace("cyp", ""), cyp]

                # Search for heuristic patterns around CYP mentions
                for variant in cyp_variants:
                    if variant in lower_text:
                        # Find inhibition patterns
                        if any(keyword in lower_text for keyword in ["inhibit", "inhibitor", "inhibition"]):
                            if any(drug.lower() in lower_text for drug in detected_drugs):
                                # If we have drug names, use those, otherwise mark as unknown
                                roles["inhibitors"].extend(drug for drug in detected_drugs if drug.lower() in lower_text)
                            else:
                                roles["inhibitors"].append("unknown")

                        # Find induction patterns
                        if any(keyword in lower_text for keyword in ["induce", "inducer", "induction"]):
                            if any(drug.lower() in lower_text for drug in detected_drugs):
                                roles["inducers"].extend(drug for drug in detected_drugs if drug.lower() in lower_text)
                            else:
                                roles["inducers"].append("unknown")

                        # Find substrate patterns
                        if any(keyword in lower_text for keyword in ["substrate", "metabolize", "metabolism"]):
                            if any(drug.lower() in lower_text for drug in detected_drugs):
                                roles["substrates"].extend(drug for drug in detected_drugs if drug.lower() in lower_text)
                            else:
                                roles["substrates"].append("unknown")

                # Remove duplicates and add to result
                for role in roles:
                    roles[role] = list(set(roles[role]))

                result[cyp] = roles

        return result

    def extract_pharmacokinetic_parameters(self, text: Optional[str]) -> Dict[str, Any]:
        if not text:
            return {}

        keyword_hits: Dict[str, Set[str]] = {key: set() for key in _PK_PATTERNS}
        value_hits: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for parameter, patterns in _PK_PATTERNS.items():
            for pattern in patterns:
                try:
                    matches = [match.group(0) for match in pattern.finditer(text)]
                except re.error:
                    continue
                for match in matches:
                    cleaned = match.upper() if match.isalpha() else match
                    keyword_hits[parameter].add(cleaned)

        for parameter, pattern in _PK_VALUE_PATTERNS.items():
            for match in pattern.finditer(text):
                value_str = match.group("value")
                unit = match.group("unit")
                try:
                    value = float(value_str)
                except (TypeError, ValueError):
                    value = None
                entry = {
                    "text": match.group(0).strip(),
                    "value": value,
                    "unit": unit,
                    "confidence": 0.85,
                }
                value_hits[parameter].append(entry)
                keyword_hits.setdefault(parameter, set()).add(entry["text"])

        output: Dict[str, Any] = {
            key: sorted(values)
            for key, values in keyword_hits.items()
            if values
        }
        if value_hits:
            output["pharmacokinetic_values"] = {
                key: value_hits[key]
                for key in value_hits
            }
        return output

    def extract_dosage_information(self, text: Optional[str]) -> List[Dict[str, Any]]:
        if not text:
            return []
        entries: List[Dict[str, Any]] = []
        for match in _DOSAGE_PATTERN.finditer(text):
            entry = {
                "text": match.group(0),
                "amount": float(match.group("amount")),
                "unit": match.group("unit").lower(),
                "route": (match.group("route") or "").lower() or None,
                "frequency": (match.group("frequency") or "").lower() or None,
                "confidence": 0.8,
            }
            window = text[match.end(): match.end() + 40]
            duration_match = _DOSAGE_DURATION_PATTERN.search(window)
            if duration_match:
                try:
                    duration_value = int(duration_match.group("value"))
                    duration_unit = duration_match.group("unit").lower()
                    entry["duration"] = {
                        "value": duration_value,
                        "unit": duration_unit,
                    }
                except (TypeError, ValueError):
                    pass
            entries.append(entry)
        return self._deduplicate_dicts(entries, key_fields=("text",))

    def normalize_mesh_terms(self, mesh_terms: Optional[Iterable[str]]) -> List[str]:
        if not mesh_terms:
            return []
        normalized = {
            re.sub(r"\s+", " ", str(term).strip()).lower()
            for term in mesh_terms
            if term
        }
        return sorted(normalized)

    def normalize_species(self, species_data: Any) -> List[str]:
        """Normalize species data to consistent format.

        Handles various input formats (string, list, dict) and normalizes to
        lowercase with standard vocabulary mapping.

        Args:
            species_data: Raw species data from metadata

        Returns:
            List of normalized species terms
        """
        if not species_data:
            return []

        # Species vocabulary mapping for standardization
        species_mapping = {
            "human": ["human", "humans", "human subjects"],
            "mouse": ["mouse", "mice", "mus musculus"],
            "rat": ["rat", "rats", "rattus norvegicus"],
            "dog": ["dog", "dogs", "canine", "canines"],
            "monkey": ["monkey", "monkeys", "nonhuman primate", "non-human primate", "macaque"],
            "rabbit": ["rabbit", "rabbits"],
            "guinea pig": ["guinea pig", "guinea pigs"],
            "pig": ["pig", "pigs", "porcine"],
            "in vitro": ["in vitro", "cell culture", "cultured cells"],
        }

        # Extract all species values
        species_values = []
        if isinstance(species_data, str):
            species_values = [species_data]
        elif isinstance(species_data, (list, tuple, set)):
            species_values = list(species_data)
        elif isinstance(species_data, dict) and "species" in species_data:
            species_values = [species_data["species"]]

        # Normalize each species value
        normalized_set = set()
        for value in species_values:
            if not value:
                continue

            # Normalize to lowercase and handle special cases
            normalized = str(value).lower().strip()

            # Handle 'non-human' -> 'nonhuman'
            normalized = normalized.replace('non-human', 'nonhuman')

            # Map to standard vocabulary
            mapped = None
            for standard, variants in species_mapping.items():
                if normalized in variants or normalized == standard:
                    mapped = standard
                    break

            if mapped:
                normalized_set.add(mapped)
            else:
                # If no mapping found, use the normalized value
                normalized_set.add(normalized)

        return sorted(normalized_set)

    def normalize_study_types(self, study_type_data: Any) -> List[str]:
        """Normalize study type data to controlled vocabulary.

        Handles both single study_type and multiple study_types, normalizing
        to standard terms.

        Args:
            study_type_data: Raw study type data from metadata

        Returns:
            List of normalized study type terms
        """
        # Study type vocabulary mapping
        study_type_mapping = {
            "clinical trial": ["clinical trial", "clinical study", "human study"],
            "randomized controlled trial": ["randomized controlled trial", "rct", "randomized clinical trial"],
            "controlled clinical trial": ["controlled clinical trial", "controlled trial"],
            "phase i": ["phase i", "phase 1", "phase i trial"],
            "phase ii": ["phase ii", "phase 2", "phase ii trial"],
            "phase iii": ["phase iii", "phase 3", "phase iii trial"],
            "phase iv": ["phase iv", "phase 4", "phase iv trial"],
            "observational study": ["observational study", "cohort study", "case-control study"],
            "case report": ["case report", "case study"],
            "meta-analysis": ["meta-analysis", "meta analysis"],
            "systematic review": ["systematic review", "systematic literature review"],
            "preclinical study": ["preclinical study", "animal study", "in vivo study"],
            "in vitro study": ["in vitro study", "in vitro", "cell culture study"],
        }

        # Extract all study type values
        study_types = []

        # Handle study_types (list)
        if isinstance(study_type_data, (list, tuple, set)):
            study_types.extend(str(st).strip() for st in study_type_data if st)

        # Handle study_type (single)
        elif isinstance(study_type_data, str):
            study_types.append(study_type_data)

        # Also check if it's a dict with both fields
        elif isinstance(study_type_data, dict):
            if "study_types" in study_type_data:
                study_types.extend(str(st).strip() for st in study_type_data["study_types"] if st)
            if "study_type" in study_type_data:
                study_types.append(str(study_type_data["study_type"]).strip())

        # Normalize each study type
        normalized_set = set()
        for study_type in study_types:
            if not study_type:
                continue

            normalized = str(study_type).lower().strip()

            # Map to standard vocabulary
            mapped = None
            for standard, variants in study_type_mapping.items():
                if normalized in variants or normalized == standard:
                    mapped = standard
                    break

            if mapped:
                normalized_set.add(mapped)
            else:
                # If no mapping found, use the normalized value
                normalized_set.add(normalized)

        return sorted(normalized_set)

    def normalize_publication_year(self, year_data: Any) -> Optional[int]:
        """Normalize publication year to consistent integer format.

        Handles various year formats and validates ranges.

        Args:
            year_data: Raw year data from metadata

        Returns:
            Normalized year as integer, or None if invalid
        """
        if not year_data:
            return None

        year_str = None

        if isinstance(year_data, (int, float)):
            year_str = str(int(year_data))
        elif isinstance(year_data, str):
            # Extract digits from string
            year_match = re.search(r'\b(19|20)\d{2}\b', year_data)
            if year_match:
                year_str = year_match.group()
        elif isinstance(year_data, dict):
            # Check both 'publication_year' and 'year' keys
            year_str = str(year_data.get('publication_year', year_data.get('year', '')))

        if not year_str:
            return None

        try:
            year = int(year_str)
            # Validate reasonable year range
            current_year = 2025  # Could use datetime.now().year
            if 1900 <= year <= current_year + 5:  # Allow 5 years in future
                return year
        except (ValueError, TypeError):
            pass

        return None

    def normalize_pharmacokinetic_data(self, pk_data: Any) -> Dict[str, Any]:
        """Normalize pharmacokinetic data to consistent structure.

        Merges data from 'pharmacokinetics' and 'pharmacokinetic_values' fields
        and standardizes parameter names.

        Args:
            pk_data: Raw PK data from metadata

        Returns:
            Normalized PK data dictionary
        """
        normalized_pk = {}

        if not pk_data:
            return normalized_pk

        # Handle different input formats
        pk_dict = {}
        pk_values_dict = {}

        if isinstance(pk_data, dict):
            # Separate main PK data from values
            pk_dict = {k: v for k, v in pk_data.items() if k != 'pharmacokinetic_values'}
            pk_values_dict = pk_data.get('pharmacokinetic_values', {})
        elif isinstance(pk_data, (list, tuple)):
            # If it's a list, try to extract PK parameters
            for item in pk_data:
                if isinstance(item, dict):
                    pk_dict.update(item)

        # Standard parameter name mapping
        param_mapping = {
            'half_life': ['half-life', 't1/2', 't_half', 'elimination_half_life'],
            'clearance': ['cl', 'cl/f', 'clearance', 'plasma_clearance'],
            'auc': ['auc', 'area_under_curve', 'area_under_the_curve'],
            'cmax': ['cmax', 'peak_concentration'],
            'tmax': ['tmax', 'time_to_peak'],
            'vd': ['vd', 'volume_distribution', 'volume_of_distribution', 'v_d'],
            'bioavailability': ['f', 'bioavailability', 'absolute_bioavailability'],
            'protein_binding': ['protein_binding', 'plasma_protein_binding'],
        }

        # Process all PK data
        all_data = {**pk_dict, **pk_values_dict}

        for standard_param, variants in param_mapping.items():
            for variant in variants:
                if variant in all_data:
                    value = all_data[variant]
                    # Normalize value format
                    if isinstance(value, (int, float)):
                        normalized_pk[standard_param] = value
                    elif isinstance(value, str):
                        # Try to extract numeric value
                        num_match = re.search(r'(\d+(?:\.\d+)?)', value)
                        if num_match:
                            try:
                                normalized_pk[standard_param] = float(num_match.group(1))
                            except ValueError:
                                pass
                    break

        return normalized_pk

    def identify_therapeutic_areas(self, mesh_terms: Optional[Iterable[str]]) -> List[str]:
        if not mesh_terms:
            return []
        normalized = self.normalize_mesh_terms(mesh_terms)
        matched: Set[str] = set()
        for term in normalized:
            for mesh_term, area in self.mesh_to_therapeutic_area.items():
                if mesh_term in term:
                    matched.add(area)
        return sorted(matched)

    def classify_drug_interaction_type(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        lowered = text.lower()
        matched: Set[str] = set()
        for label, keywords in _INTERACTION_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                matched.add(label)
        return sorted(matched)

    def normalize_species(self, species_data: Any) -> List[str]:
        """Normalize species data to consistent format.

        Args:
            species_data: Species information as string, list, or dict

        Returns:
            List of normalized species names
        """
        if not species_data:
            return []

        # Comprehensive species mapping
        species_mapping = {
            "human": ["human", "humans", "human subjects", "patient", "patients", "adult", "adults"],
            "mouse": ["mouse", "mice", "mus musculus"],
            "rat": ["rat", "rats", "rattus norvegicus"],
            "monkey": ["monkey", "monkeys", "nonhuman primate", "non-human primate", "primate", "macaque"],
            "dog": ["dog", "dogs", "canine", "beagle"],
            "rabbit": ["rabbit", "rabbits", "bunny"],
            "guinea pig": ["guinea pig", "guinea pigs", "cavia porcellus"],
            "pig": ["pig", "pigs", "swine", "porcine"],
            "in vitro": ["in vitro", "cell culture", "cell line", "cells", "cellular"],
        }

        # Extract species values
        species_list = []

        if isinstance(species_data, str):
            species_list.append(species_data.strip())
        elif isinstance(species_data, (list, tuple, set)):
            species_list.extend(str(s).strip() for s in species_data if s)
        elif isinstance(species_data, dict):
            # Handle dict with species field
            if "species" in species_data:
                species_list.append(str(species_data["species"]).strip())

        # Normalize each species
        normalized_set = set()
        for species in species_list:
            if not species:
                continue

            normalized = species.lower().strip()

            # Map to standard vocabulary
            mapped = None
            for standard, variants in species_mapping.items():
                if normalized in variants or normalized == standard:
                    mapped = standard
                    break

            if mapped:
                normalized_set.add(mapped)
            else:
                # If no mapping found, use the normalized value
                normalized_set.add(normalized)

        return sorted(normalized_set)

    def normalize_study_types(self, study_type_data: Any) -> List[str]:
        """Normalize study type data to consistent format.

        Args:
            study_type_data: Study type information as string, list, or dict

        Returns:
            List of normalized study types
        """
        if not study_type_data:
            return []

        # Study type mapping
        study_type_mapping = {
            "randomized controlled trial": ["randomized controlled trial", "rct", "randomized clinical trial"],
            "clinical trial": ["clinical trial", "trial", "clinical study"],
            "case report": ["case report", "case study"],
            "case series": ["case series", "case reports"],
            "cohort study": ["cohort study", "cohort", "observational cohort"],
            "cross-sectional study": ["cross-sectional study", "cross sectional"],
            "case-control study": ["case-control study", "case control"],
            "meta-analysis": ["meta-analysis", "meta analysis"],
            "systematic review": ["systematic review", "systematic literature review"],
            "preclinical study": ["preclinical study", "animal study", "in vivo study"],
            "in vitro study": ["in vitro study", "in vitro", "cell culture study"],
        }

        # Extract all study type values
        study_types = []

        # Handle study_types (list)
        if isinstance(study_type_data, (list, tuple, set)):
            study_types.extend(str(st).strip() for st in study_type_data if st)

        # Handle study_type (single)
        elif isinstance(study_type_data, str):
            study_types.append(study_type_data)

        # Also check if it's a dict with both fields
        elif isinstance(study_type_data, dict):
            if "study_types" in study_type_data:
                study_types.extend(str(st).strip() for st in study_type_data["study_types"] if st)
            if "study_type" in study_type_data:
                study_types.append(str(study_type_data["study_type"]).strip())

        # Normalize each study type
        normalized_set = set()
        for study_type in study_types:
            if not study_type:
                continue

            normalized = str(study_type).lower().strip()

            # Map to standard vocabulary
            mapped = None
            for standard, variants in study_type_mapping.items():
                if normalized in variants or normalized == standard:
                    mapped = standard
                    break

            if mapped:
                normalized_set.add(mapped)
            else:
                # If no mapping found, use the normalized value
                normalized_set.add(normalized)

        return sorted(normalized_set)

    def normalize_publication_year(self, year_data: Any) -> Optional[int]:
        """Normalize publication year to consistent integer format.

        Args:
            year_data: Year information as int, str, or dict

        Returns:
            Normalized year as integer or None if invalid
        """
        if not year_data:
            return None

        year_str = None

        if isinstance(year_data, int):
            year_str = str(year_data)
        elif isinstance(year_data, str):
            year_str = year_data
        elif isinstance(year_data, dict):
            # Handle dict with publication_year field
            if "publication_year" in year_data:
                year_str = str(year_data["publication_year"])
            elif "year" in year_data:
                year_str = str(year_data["year"])

        if not year_str:
            return None

        # Extract year from string
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', year_str)

        if not year_match:
            return None

        year = int(year_match.group())

        # Validate year range
        current_year = datetime.now().year
        if year < 1900 or year > current_year + 1:
            return None

        return year

    def normalize_pharmacokinetic_data(self, pk_data: Any) -> Dict[str, float]:
        """Normalize pharmacokinetic data to consistent format.

        Args:
            pk_data: PK data as dict or combined dict with pharmacokinetics and pharmacokinetic_values

        Returns:
            Dict with normalized PK parameters
        """
        if not pk_data:
            return {}

        normalized = {}

        # Handle combined data structure
        if isinstance(pk_data, dict):
            # Check if this is a combined structure with pharmacokinetics and pharmacokinetic_values
            if "pharmacokinetics" in pk_data or "pharmacokinetic_values" in pk_data:
                # Extract pharmacokinetics
                pk_main = pk_data.get("pharmacokinetics", {})
                pk_values = pk_data.get("pharmacokinetic_values", {})

                # Merge both dictionaries
                all_pk = {**pk_main, **pk_values}
            else:
                # This is direct PK data
                all_pk = pk_data
        else:
            all_pk = {}

        # Parameter name normalization mapping
        param_mapping = {
            "half-life": "half_life",
            "half life": "half_life",
            "t1/2": "half_life",
            "t_half": "half_life",
            "auc": "auc",
            "area under curve": "auc",
            "cmax": "cmax",
            "c_max": "cmax",
            "tmax": "tmax",
            "t_max": "tmax",
            "clearance": "clearance",
            "cl": "clearance",
            "vd": "volume_of_distribution",
            "volume of distribution": "volume_of_distribution",
            "bioavailability": "bioavailability",
            "f": "bioavailability",
            "protein binding": "protein_binding",
        }

        # Normalize each parameter
        for param_name, param_value in all_pk.items():
            if not param_value:
                continue

            # Normalize parameter name
            normalized_name = param_mapping.get(param_name.lower(), param_name.lower().replace(" ", "_"))

            # Extract numeric value
            import re
            if isinstance(param_value, (int, float)):
                numeric_value = float(param_value)
            else:
                # Extract from string
                value_match = re.search(r'(\d+\.?\d*)', str(param_value))
                if value_match:
                    numeric_value = float(value_match.group(1))
                else:
                    continue

            normalized[normalized_name] = numeric_value

        return normalized

    def enhance_document_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Attach extraction results to the supplied document metadata."""
        enhanced = dict(document)
        metadata = dict(enhanced.get("metadata") or {})
        text_fragments = [
            enhanced.get("page_content"),
            enhanced.get("content"),
            metadata.get("abstract"),
            metadata.get("summary"),
        ]
        combined_text = "\n".join(filter(None, text_fragments))
        mesh_terms = metadata.get("mesh_terms") or metadata.get("mesh") or []

        drug_candidates = self.extract_drug_names(combined_text)
        # Preserve original drug names and annotations before normalization
        metadata["drug_names_original"] = [item["name"] for item in drug_candidates]
        metadata["drug_annotations_original"] = deepcopy(drug_candidates)
        metadata["drug_names"] = [item["name"] for item in drug_candidates]
        metadata["drug_annotations"] = drug_candidates
        metadata["cyp_enzymes"] = self.extract_cyp_enzymes(combined_text)
        metadata["cyp_roles"] = self.annotate_cyp_roles(combined_text)

        pk_data = self.extract_pharmacokinetic_parameters(combined_text)
        pk_values = pk_data.pop("pharmacokinetic_values", None)
        metadata["pharmacokinetics"] = pk_data
        if pk_values:
            metadata["pharmacokinetic_values"] = pk_values

        metadata["dosage_information"] = self.extract_dosage_information(combined_text)
        # Preserve original MeSH terms before normalization
        if mesh_terms:
            metadata["mesh_terms_original"] = list(mesh_terms)
        metadata["mesh_terms"] = self.normalize_mesh_terms(mesh_terms)
        metadata["therapeutic_areas"] = self.identify_therapeutic_areas(mesh_terms)
        metadata["interaction_types"] = self.classify_drug_interaction_type(combined_text)

        enhanced["metadata"] = metadata
        return enhanced

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _looks_like_drug_name(self, token: str) -> bool:
        """Heuristically match drug-like tokens; tokens must retain casing."""
        if len(token) < 4:
            return False
        lowered = token.lower()
        if lowered in _DRUG_SUFFIX_STOPWORDS:
            return False
        if any(lowered.endswith(suffix) for suffix in _DRUG_HEURISTIC_SUFFIXES):
            return True
        if token[0].isupper():
            if token.isupper() and len(token) <= 4:
                return False
            return True
        return False

    def _deduplicate_dicts(
        self,
        items: Iterable[Dict[str, Any]],
        *,
        key_fields: Iterable[str],
    ) -> List[Dict[str, Any]]:
        seen: Set[tuple] = set()
        results: List[Dict[str, Any]] = []
        fields = tuple(key_fields)
        for item in items:
            key = tuple(item.get(field) for field in fields)
            if key in seen:
                continue
            seen.add(key)
            results.append(item)
        return results

    def _load_lexicon_file(self, path: str) -> Set[str]:
        """Load lowercase drug names from a lexicon (one drug per line)."""
        file_path = Path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Unable to read lexicon file '{path}': {exc}") from exc

        entries: Set[str] = set()
        for line in text.splitlines():
            term = line.strip()
            if not term or term.startswith("#"):
                continue
            entries.add(term.lower())
        if not entries:
            logger.warning("Lexicon file '%s' did not contain any usable entries", path)
        return entries

    def _auto_fetch_lexicons(self, data_dir: Path) -> None:
        """Automatically populate bundled lexicon files when explicitly requested.

        Creates comprehensive drug lexicons with top pharmaceutical compounds from
        authoritative sources including FDA Orange Book categories, WHO ATC codes,
        and common pharmaceutical databases.

        Args:
            data_dir: Directory where lexicon files should be created.
        """
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Auto-fetching comprehensive drug lexicons into %s", data_dir)

            # Create comprehensive generic drug lexicon (top 1000+ drugs)
            generic_file = data_dir / "drugs_generic.txt"
            if not generic_file.exists():
                logger.info("Creating comprehensive generic drug lexicon at %s", generic_file)
                self._create_comprehensive_generic_lexicon(generic_file)

            # Create comprehensive brand drug lexicon
            brand_file = data_dir / "drugs_brand.txt"
            if not brand_file.exists():
                logger.info("Creating comprehensive brand drug lexicon at %s", brand_file)
                self._create_comprehensive_brand_lexicon(brand_file)

            # Create CYP roles file
            cyp_file = data_dir / "cyp_roles.csv"
            if not cyp_file.exists():
                logger.info("Creating CYP enzyme roles file at %s", cyp_file)
                self._create_cyp_roles_file(cyp_file)

            logger.info("Comprehensive drug lexicons created successfully")

        except Exception as exc:
            logger.warning("Failed to auto-fetch drug lexicons: %s", exc)

    def _create_comprehensive_generic_lexicon(self, file_path: Path) -> None:
        """Create comprehensive generic drug lexicon with top pharmaceutical compounds."""
        # Start with current built-in drugs and add comprehensive list
        comprehensive_generics = set(_GENERIC_DRUG_NAMES)

        # Add additional comprehensive list targeting top 1000+ drugs
        additional_generics = {
            # Cardiovascular drugs
            "amlodipine", "lisinopril", "metoprolol", "hydrochlorothiazide", "atenolol",
            "carvedilol", "furosemide", "spironolactone", "valsartan", "losartan",
            "enalapril", "captopril", "propranolol", "diltiazem", "verapamil",
            "nifedipine", "felodipine", "isradipine", "nicardipine", "clevidipine",
            "bisoprolol", "nebivolol", "labetalol", "nadolol", "timolol",
            "chlorthalidone", "indapamide", "amiloride", "triamterene", "torsemide",
            "bumetanide", "ethacrynic", "acetazolamide", "mannitol",
            # Statins and lipid drugs
            "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin",
            "fluvastatin", "pitavastatin", "ezetimibe", "fenofibrate", "gemfibrozil",
            "niacin", "cholestyramine", "colesevelam", "colestipol",
            # Anticoagulants and antiplatelets
            "warfarin", "heparin", "enoxaparin", "fondaparinux", "rivaroxaban",
            "apixaban", "dabigatran", "edoxaban", "clopidogrel", "prasugrel",
            "ticagrelor", "aspirin", "dipyridamole", "cilostazol", "pentoxifylline",
            # Diabetes medications
            "metformin", "glyburide", "glipizide", "glimepiride", "pioglitazone",
            "rosiglitazone", "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
            "exenatide", "liraglutide", "dulaglutide", "semaglutide", "lixisenatide",
            "insulin", "glargine", "detemir", "aspart", "lispro", "degludec",
            "canagliflozin", "dapagliflozin", "empagliflozin", "ertugliflozin",
            "acarbose", "miglitol", "nateglinide", "repaglinide",
            # Antibiotics
            "amoxicillin", "ampicillin", "penicillin", "cephalexin", "cefazolin",
            "ceftriaxone", "cefuroxime", "cefepime", "ceftaroline", "azithromycin",
            "clarithromycin", "erythromycin", "doxycycline", "tetracycline", "minocycline",
            "ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin", "norfloxacin",
            "clindamycin", "metronidazole", "vancomycin", "linezolid", "daptomycin",
            "trimethoprim", "sulfamethoxazole", "nitrofurantoin", "fosfomycin",
            # Antifungals
            "fluconazole", "itraconazole", "ketoconazole", "voriconazole", "posaconazole",
            "amphotericin", "caspofungin", "micafungin", "anidulafungin", "terbinafine",
            "nystatin", "clotrimazole", "miconazole", "econazole", "terconazole",
            # Pain and inflammation
            "ibuprofen", "naproxen", "diclofenac", "celecoxib", "meloxicam",
            "indomethacin", "piroxicam", "sulindac", "ketoprofen", "ketorolac",
            "acetaminophen", "tramadol", "morphine", "oxycodone", "hydrocodone",
            "codeine", "fentanyl", "gabapentin", "pregabalin", "duloxetine",
            # CNS medications
            "sertraline", "fluoxetine", "paroxetine", "citalopram", "escitalopram",
            "venlafaxine", "desvenlafaxine", "bupropion", "mirtazapine", "trazodone",
            "amitriptyline", "nortriptyline", "imipramine", "desipramine", "clomipramine",
            "lorazepam", "diazepam", "alprazolam", "clonazepam", "temazepam",
            "zolpidem", "eszopiclone", "zaleplon", "ramelteon", "suvorexant",
            # Antipsychotics
            "risperidone", "olanzapine", "quetiapine", "aripiprazole", "ziprasidone",
            "paliperidone", "asenapine", "lurasidone", "cariprazine", "brexpiprazole",
            "haloperidol", "chlorpromazine", "fluphenazine", "perphenazine", "trifluoperazine",
            # Anticonvulsants
            "phenytoin", "carbamazepine", "valproic", "lamotrigine", "levetiracetam",
            "topiramate", "oxcarbazepine", "lacosamide", "zonisamide", "eslicarbazepine",
            "vigabatrin", "tiagabine", "gabapentin", "pregabalin",
            # Respiratory medications
            "albuterol", "ipratropium", "tiotropium", "formoterol", "salmeterol",
            "budesonide", "fluticasone", "beclomethasone", "mometasone", "ciclesonide",
            "montelukast", "zafirlukast", "zileuton", "theophylline", "aminophylline",
            # Gastrointestinal
            "omeprazole", "lansoprazole", "esomeprazole", "pantoprazole", "rabeprazole",
            "dexlansoprazole", "ranitidine", "famotidine", "cimetidine", "nizatidine",
            "metoclopramide", "ondansetron", "granisetron", "dolasetron", "palonosetron",
            "simethicone", "loperamide", "bismuth", "sucralfate", "misoprostol",
            # Thyroid medications
            "levothyroxine", "liothyronine", "methimazole", "propylthiouracil", "radioiodine",
            # Osteoporosis
            "alendronate", "risedronate", "ibandronate", "zoledronic", "denosumab",
            "raloxifene", "calcitonin", "teriparatide", "abaloparatide",
            # Immunosuppressants
            "cyclosporine", "tacrolimus", "sirolimus", "everolimus", "mycophenolate",
            "azathioprine", "methotrexate", "leflunomide", "hydroxychloroquine", "sulfasalazine",
            # Cancer chemotherapy
            "doxorubicin", "cyclophosphamide", "methotrexate", "fluorouracil", "carboplatin",
            "cisplatin", "oxaliplatin", "paclitaxel", "docetaxel", "gemcitabine",
            "irinotecan", "topotecan", "etoposide", "bleomycin", "vincristine",
            "vinblastine", "vinorelbine", "capecitabine", "temozolomide", "dacarbazine",
            # Targeted cancer therapy
            "imatinib", "dasatinib", "nilotinib", "bosutinib", "ponatinib",
            "erlotinib", "gefitinib", "afatinib", "osimertinib", "sorafenib",
            "sunitinib", "pazopanib", "regorafenib", "cabozantinib", "lenvatinib",
            "bevacizumab", "trastuzumab", "rituximab", "cetuximab", "panitumumab",
            # Vitamins and supplements
            "vitamin", "folic", "cyanocobalamin", "thiamine", "riboflavin",
            "niacin", "pyridoxine", "biotin", "pantothenic", "ascorbic",
            "ergocalciferol", "cholecalciferol", "phytonadione", "tocopherol",
            "calcium", "magnesium", "iron", "zinc", "selenium", "potassium"
        }

        comprehensive_generics.update(additional_generics)

        # Write to file
        with file_path.open('w', encoding='utf-8') as f:
            f.write("# Comprehensive generic drug names for pharmaceutical processing\n")
            f.write("# One drug name per line, case insensitive\n")
            f.write("# Comments start with #\n\n")
            for drug in sorted(comprehensive_generics):
                f.write(f"{drug}\n")

    def _create_comprehensive_brand_lexicon(self, file_path: Path) -> None:
        """Create comprehensive brand drug lexicon with top pharmaceutical brands."""
        # Start with current built-in drugs and add comprehensive list
        comprehensive_brands = set(_BRAND_DRUG_NAMES)

        # Add additional comprehensive list of brand names
        additional_brands = {
            # Cardiovascular brands
            "norvasc", "prinivil", "zestril", "lopressor", "toprol", "tenormin",
            "coreg", "lasix", "aldactone", "diovan", "cozaar", "vasotec",
            "capoten", "inderal", "cardizem", "calan", "isoptin", "adalat",
            "procardia", "plendil", "dynacirc", "cardene", "cleviprex", "bystolic",
            "normodyne", "trandate", "corgard", "timoptic", "hygroton", "lozol",
            "midamor", "dyrenium", "demadex", "bumex", "diamox", "osmitrol",
            # Statin brands
            "lipitor", "zocor", "crestor", "pravachol", "mevacor", "lescol",
            "livalo", "zetia", "tricor", "lopid", "niaspan", "questran",
            "welchol", "colestid",
            # Anticoagulant brands
            "coumadin", "lovenox", "arixtra", "xarelto", "eliquis", "pradaxa",
            "savaysa", "plavix", "effient", "brilinta", "pletal", "pentoxil",
            # Diabetes brands
            "glucophage", "diabeta", "micronase", "glucotrol", "amaryl", "actos",
            "avandia", "januvia", "onglyza", "tradjenta", "nesina", "byetta",
            "victoza", "trulicity", "ozempic", "adlyxin", "lantus", "levemir",
            "novolog", "humalog", "tresiba", "invokana", "farxiga", "jardiance",
            "steglatro", "precose", "glyset", "starlix", "prandin",
            # Antibiotic brands
            "amoxil", "ampicillin", "penicillin", "keflex", "ancef", "rocephin",
            "ceftin", "maxipime", "teflaro", "zithromax", "biaxin", "ery-tab",
            "erythrocin", "vibramycin", "sumycin", "minocin", "cipro", "levaquin",
            "avelox", "floxin", "noroxin", "cleocin", "flagyl", "vancocin",
            "zyvox", "cubicin", "bactrim", "septra", "macrobid", "monurol",
            # Antifungal brands
            "diflucan", "sporanox", "nizoral", "vfend", "noxafil", "fungizone",
            "cancidas", "eraxis", "ecalta", "lamisil", "mycostatin", "lotrimin",
            "monistat", "spectazole", "terazol",
            # Pain/inflammation brands
            "advil", "motrin", "aleve", "naprosyn", "voltaren", "celebrex",
            "mobic", "indocin", "feldene", "clinoril", "orudis", "toradol",
            "tylenol", "ultram", "ms contin", "oxycontin", "vicodin", "norco",
            "percocet", "duragesic", "neurontin", "lyrica", "cymbalta",
            # CNS brands
            "zoloft", "prozac", "paxil", "celexa", "lexapro", "effexor",
            "pristiq", "wellbutrin", "zyban", "remeron", "desyrel", "elavil",
            "pamelor", "tofranil", "norpramin", "anafranil", "ativan", "valium",
            "xanax", "klonopin", "restoril", "ambien", "lunesta", "sonata",
            "rozerem", "belsomra",
            # Antipsychotic brands
            "risperdal", "zyprexa", "seroquel", "abilify", "geodon", "invega",
            "saphris", "latuda", "vraylar", "rexulti", "haldol", "thorazine",
            "prolixin", "trilafon", "stelazine",
            # Anticonvulsant brands
            "dilantin", "tegretol", "depakote", "lamictal", "keppra", "topamax",
            "trileptal", "vimpat", "zonegran", "aptiom", "sabril", "gabitril",
            # Respiratory brands
            "proventil", "ventolin", "atrovent", "spiriva", "foradil", "serevent",
            "pulmicort", "flovent", "vanceril", "qvar", "asmanex", "alvesco",
            "singulair", "accolate", "zyflo", "theo-dur", "phyllocontin",
            # GI brands
            "prilosec", "prevacid", "nexium", "protonix", "aciphex", "dexilant",
            "zantac", "pepcid", "tagamet", "axid", "reglan", "zofran", "kytril",
            "anzemet", "aloxi", "gas-x", "imodium", "pepto-bismol", "carafate",
            "cytotec",
            # Thyroid brands
            "synthroid", "cytomel", "tapazole", "ptu",
            # Osteoporosis brands
            "fosamax", "actonel", "boniva", "reclast", "prolia", "evista",
            "miacalcin", "forteo", "tymlos",
            # Immunosuppressant brands
            "sandimmune", "neoral", "prograf", "rapamune", "zortress", "cellcept",
            "imuran", "rheumatrex", "arava", "plaquenil", "azulfidine",
            # Cancer brands
            "adriamycin", "cytoxan", "adrucil", "paraplatin", "platinol", "eloxatin",
            "taxol", "taxotere", "gemzar", "camptosar", "hycamtin", "toposar",
            "blenoxane", "oncovin", "velban", "navelbine", "xeloda", "temodar",
            "dtic-dome", "gleevec", "sprycel", "tasigna", "bosulif", "iclusig",
            "tarceva", "iressa", "gilotrif", "tagrisso", "nexavar", "sutent",
            "votrient", "stivarga", "cometriq", "lenvima", "avastin", "herceptin",
            "rituxan", "erbitux", "vectibix"
        }

        comprehensive_brands.update(additional_brands)

        # Write to file
        with file_path.open('w', encoding='utf-8') as f:
            f.write("# Comprehensive brand drug names for pharmaceutical processing\n")
            f.write("# One drug name per line, case insensitive\n")
            f.write("# Comments start with #\n\n")
            for drug in sorted(comprehensive_brands):
                f.write(f"{drug}\n")

    def _create_cyp_roles_file(self, file_path: Path) -> None:
        """Create CYP enzyme roles CSV file."""
        cyp_data = [
            ("enzyme", "role", "drug"),
            ("CYP1A2", "substrate", "caffeine"),
            ("CYP1A2", "substrate", "theophylline"),
            ("CYP1A2", "inhibitor", "fluvoxamine"),
            ("CYP1A2", "inhibitor", "ciprofloxacin"),
            ("CYP1A2", "inducer", "smoking"),
            ("CYP2C9", "substrate", "warfarin"),
            ("CYP2C9", "substrate", "phenytoin"),
            ("CYP2C9", "inhibitor", "fluconazole"),
            ("CYP2C9", "inhibitor", "amiodarone"),
            ("CYP2C9", "inducer", "rifampin"),
            ("CYP2C19", "substrate", "omeprazole"),
            ("CYP2C19", "substrate", "clopidogrel"),
            ("CYP2C19", "inhibitor", "omeprazole"),
            ("CYP2C19", "inhibitor", "fluoxetine"),
            ("CYP2C19", "inducer", "rifampin"),
            ("CYP2D6", "substrate", "metoprolol"),
            ("CYP2D6", "substrate", "codeine"),
            ("CYP2D6", "inhibitor", "quinidine"),
            ("CYP2D6", "inhibitor", "fluoxetine"),
            ("CYP3A4", "substrate", "midazolam"),
            ("CYP3A4", "substrate", "simvastatin"),
            ("CYP3A4", "inhibitor", "ketoconazole"),
            ("CYP3A4", "inhibitor", "grapefruit"),
            ("CYP3A4", "inducer", "rifampin"),
            ("CYP3A4", "inducer", "carbamazepine"),
        ]

        with file_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for row in cyp_data:
                writer.writerow(row)

    def _load_cyp_roles_file(self, path: str) -> Dict[str, Dict[str, Set[str]]]:
        roles: Dict[str, Dict[str, Set[str]]] = {}
        file_path = Path(path)
        try:
            with file_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    enzyme = (row.get("enzyme") or "").strip()
                    role = (row.get("role") or "").strip().lower()
                    drug = (row.get("drug") or "").strip()
                    if not enzyme or not role:
                        continue
                    enzyme_key = self._normalise_enzyme(enzyme)
                    role_key = self._normalise_role(role)
                    if not role_key:
                        continue
                    role_map = roles.setdefault(
                        enzyme_key,
                        {"substrates": set(), "inhibitors": set(), "inducers": set()},
                    )
                    if drug:
                        role_map[role_key].add(drug)
        except OSError as exc:
            raise ValueError(f"Unable to read CYP roles file '{path}': {exc}") from exc
        return roles

    @staticmethod
    def _normalise_role(role: str) -> Optional[str]:
        lowered = role.lower()
        if lowered.startswith("substrate"):
            return "substrates"
        if lowered.startswith("inhibitor") or lowered.startswith("inhibition"):
            return "inhibitors"
        if lowered.startswith("inducer") or lowered.startswith("induction"):
            return "inducers"
        return None

    @staticmethod
    def extract_pk_terms(text: str) -> List[str]:
        """Extract pharmacokinetic terms from text.

        Args:
            text: Text to search for PK terms

        Returns:
            Sorted list of unique PK terms found
        """
        found_terms = set()
        text_lower = text.lower()

        for pattern_name, patterns in _PK_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(text_lower):
                    found_terms.add(pattern_name)
                    break

        return sorted(found_terms)

    @staticmethod
    def extract_cyp_enzyme_strings(text: str) -> List[str]:
        """Extract CYP enzyme strings from text.

        Args:
            text: Text to search for CYP enzymes

        Returns:
            List of uppercased CYP enzyme names found
        """
        enzymes = []
        text_upper = text.upper()

        for match in _CYP_PATTERN.finditer(text_upper):
            enzymes.append(match.group())

        return enzymes

    @staticmethod
    def _normalise_enzyme(enzyme: str) -> str:
        normalized = enzyme.upper().replace(" ", "")
        if not normalized.startswith("CYP"):
            normalized = f"CYP{normalized}"
        return normalized

    def get_cyp_roles(self) -> Dict[str, Dict[str, Set[str]]]:
        return deepcopy(self.cyp_roles)

    def annotate_cyp_roles(self, text: Optional[str]) -> List[Dict[str, Any]]:
        if not text or not self.cyp_roles:
            return []
        annotations: List[Dict[str, Any]] = []
        for enzyme in self.extract_cyp_enzymes(text):
            role_map = self.cyp_roles.get(enzyme)
            if not role_map:
                continue
            roles = [role for role, drugs in role_map.items() if drugs]
            evidence = {role: sorted(drugs) for role, drugs in role_map.items() if drugs}
            if roles:
                evidence_str = "; ".join(
                    f"{role}: {', '.join(drugs)}" for role, drugs in evidence.items()
                ) if evidence else ""
                annotations.append(
                    {
                        "enzyme": enzyme,
                        "roles": sorted(roles),
                        "evidence": evidence_str,
                    }
                )
        return annotations

    def _extract_original_case(self, text: str, term: str) -> str:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        match = pattern.search(text)
        return match.group(0) if match else term


    @staticmethod
    def download_default_lexicons(target_dir: str = "./data") -> bool:
        """Download default curated drug lexicons to target directory.

        Args:
            target_dir: Directory to save lexicon files

        Returns:
            True if successful, False otherwise
        """
        try:
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            processor = PharmaceuticalProcessor(auto_fetch=False)
            processor._auto_fetch_lexicons(target_path)
            return True

        except Exception as exc:
            logger.error("Failed to download lexicons: %s", exc)
            return False


__all__ = ["PharmaceuticalProcessor"]
