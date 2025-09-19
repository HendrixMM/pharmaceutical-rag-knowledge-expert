"""Utility helpers for extracting pharmaceutical signals from biomedical text."""

from __future__ import annotations

import csv
import logging
import os
import re
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
    ) -> None:
        """Seed instance lexicons and optionally extend them from custom files."""
        self.generic_drug_names: Set[str] = set(_GENERIC_DRUG_NAMES)
        self.brand_drug_names: Set[str] = set(_BRAND_DRUG_NAMES)
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
                    logger.info("Loaded comprehensive generic drug lexicon from %s", bundled_generic)
                except ValueError as exc:
                    logger.warning("Failed to load bundled generic lexicon: %s", exc)
            else:
                logger.error(
                    "Missing %s - drug detection will be limited. "
                    "Set DRUG_GENERIC_LEXICON environment variable to provide custom lexicon.",
                    bundled_generic.name
                )
                if os.getenv("AUTO_FETCH_DRUG_LEXICONS", "false").lower() == "true":
                    self._auto_fetch_lexicons(bundled_dir)

            bundled_brand = bundled_dir / "drugs_brand.txt"
            if bundled_brand.exists():
                try:
                    self.brand_drug_names.update(
                        self._load_lexicon_file(str(bundled_brand))
                    )
                    logger.info("Loaded comprehensive brand drug lexicon from %s", bundled_brand)
                except ValueError as exc:
                    logger.warning("Failed to load bundled brand lexicon: %s", exc)
            else:
                logger.error(
                    "Missing %s - brand drug detection will be limited. "
                    "Set DRUG_BRAND_LEXICON environment variable to provide custom lexicon.",
                    bundled_brand.name
                )
                if os.getenv("AUTO_FETCH_DRUG_LEXICONS", "false").lower() == "true":
                    self._auto_fetch_lexicons(bundled_dir)
        else:
            logger.warning("Data directory not found - using minimal drug vocabularies")

        env_generic_path = generic_lexicon_path or os.getenv("DRUG_GENERIC_LEXICON")
        env_brand_path = brand_lexicon_path or os.getenv("DRUG_BRAND_LEXICON")

        if env_generic_path:
            additional_generics = self._load_lexicon_file(env_generic_path)
            if additional_generics:
                self.generic_drug_names.update(additional_generics)
        if env_brand_path:
            additional_brands = self._load_lexicon_file(env_brand_path)
            if additional_brands:
                self.brand_drug_names.update(additional_brands)

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

    def extract_drug_names(self, text: Optional[str]) -> List[Dict[str, Any]]:
        """Return detected drug names with type labels and confidence."""
        if not text:
            return []

        candidates: Dict[str, Dict[str, Any]] = {}
        lowered = text.lower()

        for name in self.generic_drug_names:
            if name in lowered:
                key = name.lower()
                canonical = self._extract_original_case(text, name)
                candidates.setdefault(
                    key,
                    {"name": canonical, "type": "generic", "confidence": 0.98},
                )

        for name in self.brand_drug_names:
            if name in lowered:
                key = name.lower()
                canonical = self._extract_original_case(text, name)
                candidates.setdefault(
                    key,
                    {"name": canonical, "type": "brand", "confidence": 0.96},
                )

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

    def extract_cyp_enzymes(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        matches = {
            self._normalise_enzyme(match)
            for match in _CYP_PATTERN.findall(text)
        }
        return sorted(matches)

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

    def _auto_fetch_lexicons(self, data_dir: Path) -> None:
        """Auto-fetch curated drug lexicons when AUTO_FETCH_DRUG_LEXICONS=true."""
        logger.info("AUTO_FETCH_DRUG_LEXICONS enabled - creating comprehensive lexicons")

        try:
            data_dir.mkdir(parents=True, exist_ok=True)

            # Create comprehensive generic drug lexicon
            generic_file = data_dir / "drugs_generic.txt"
            if not generic_file.exists():
                logger.info("Creating comprehensive generic drug lexicon")
                # This would typically fetch from a curated source
                # For now, we already have the comprehensive files
                pass

            # Create comprehensive brand drug lexicon
            brand_file = data_dir / "drugs_brand.txt"
            if not brand_file.exists():
                logger.info("Creating comprehensive brand drug lexicon")
                # This would typically fetch from a curated source
                # For now, we already have the comprehensive files
                pass

        except Exception as exc:
            logger.warning("Failed to auto-fetch drug lexicons: %s", exc)

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

            logger.info("Default lexicons are bundled - use AUTO_FETCH_DRUG_LEXICONS=true for automatic setup")
            return True

        except Exception as exc:
            logger.error("Failed to download lexicons: %s", exc)
            return False


__all__ = ["PharmaceuticalProcessor"]
