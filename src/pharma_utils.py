"""Shared pharmaceutical utilities for RAG template.

This module provides common utilities used across the pharmaceutical RAG system
to avoid code duplication and ensure consistent behavior.

Environment Variables:
    ENABLE_PK_FILTERING: Enables pharmacokinetics filtering when set to "true"
    QUERY_ENGINE_MAX_CACHE_MB: Maximum cache size in MB (default: 1000)
    QUERY_ENGINE_RUNTIME_EXTRACTION_CHAR_LIMIT: Character limit for runtime extraction

Common Constants:
    _SPECIES_KEYWORDS: Mapping of species categories to keyword sets
    _NEGATION_TERMS: Terms that indicate non-biological contexts
    _CLINICAL_STUDY_TAGS: Tags identifying clinical study types

Common Functions:
    normalize_text: Shared text normalization for consistent handling of Unicode,
        diacritics, and case across modules
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Module-level constants for PK filtering
_PK_FILTERING_ENABLED = os.getenv("ENABLE_PK_FILTERING", "false").lower() == "true"

# Species filtering defaults
# Default behavior for including documents without species information
SPECIES_UNKNOWN_DEFAULT = True
# Default species preference when none specified
SPECIES_PREFERENCE_DEFAULT = None

# Species-related constants
_SPECIES_KEYWORDS = {
    "human": {"human", "humans"},
    "mouse": {"mouse", "mice"},
    "rat": {"rat", "rats"},
    "dog": {"dog", "dogs", "canine", "canines"},
    "monkey": {"monkey", "monkeys", "nonhuman primate", "macaque"},
}

_CLINICAL_STUDY_TAGS = {
    "clinical trial",
    "randomized controlled trial",
    "controlled clinical trial",
    "phase i",
    "phase ii",
    "phase iii",
    "phase iv",
    "rct",
    "clinical study",
    "human study",
    "patient study",
    "clinical investigation",
}

_NEGATION_TERMS = {
    "in vitro",
    "cell culture",
    "cultured cells",
    "tissue culture",
    "cell line",
    "isolated cells",
    "artificial",
}

# Drug suffix patterns for fast detection
_DRUG_SUFFIXES = {
    "-azole",  # Antifungals
    "-statin",  # Lipid-lowering
    "-pril",  # ACE inhibitors
    "-sartan",  # ARBs
    "-vir",  # Antivirals
    "-cillin",  # Penicillins
    "-mycin",  # Macrolides
    "-oxetine",  # SSRIs
    "-prazole",  # PPIs
}

# Common drug names for lightweight checking
_COMMON_DRUG_NAMES = {
    "aspirin",
    "ibuprofen",
    "acetaminophen",
    "paracetamol",
    "warfarin",
    "lisinopril",
    "metoprolol",
    "atorvastatin",
    "simvastatin",
    "amlodipine",
    "metformin",
    "insulin",
    "glipizide",
    "sertraline",
    "escitalopram",
    "losartan",
    "valsartan",
    "hydrochlorothiazide",
    "furosemide",
    "omeprazole",
    "pantoprazole",
    "prednisone",
    "levothyroxine",
}


def normalize_text(text: str, *, remove_diacritics: bool = True, lowercase: bool = True) -> str:
    """Normalize text for consistent comparison across modules.

    Handles Unicode normalization, diacritic removal, case normalization,
    and whitespace cleanup. Useful for title/species normalization in
    QueryEngine and VectorDatabase.

    Args:
        text: Input text to normalize
        remove_diacritics: Whether to remove diacritical marks (default: True)
        lowercase: Whether to convert to lowercase (default: True)

    Returns:
        Normalized text string

    Examples:
        >>> normalize_text("Médecine Sans Frontières")
        'medecine sans frontieres'
        >>> normalize_text("CYP3A4 Inhibition Study")
        'cyp3a4 inhibition study'
        >>> normalize_text("Überprüfung", remove_diacritics=False)
        'überprüfung'
    """
    if not text:
        return ""

    # First normalize Unicode form (NFKC for compatibility)
    text = unicodedata.normalize("NFKC", text)

    # Remove diacritics if requested
    if remove_diacritics:
        text = "".join(c for c in text if not unicodedata.combining(c))

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()

    return text


def _tokenize_species_string(text: str) -> set[str]:
    """Normalize species strings by tokenizing on non-alphanumeric characters.

    Handles special cases like 'non-human' → 'nonhuman' and ensures consistent
    tokenization across modules.

    Args:
        text: Species string to tokenize

    Returns:
        Set of normalized tokens

    Examples:
        >>> _tokenize_species_string("Human studies")
        {'human'}
        >>> _tokenize_species_string("non-human primates")
        {'nonhuman', 'primates'}
        >>> _tokenize_species_string("In vitro")
        {'vitro'}
    """
    if not text:
        return set()

    # Handle 'non-human' as a special case
    text = text.replace("non-human", "nonhuman")
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))

    # Also add 'nonhuman' if 'non' and 'human' are both present
    if "non" in tokens and "human" in tokens:
        tokens.add("nonhuman")

    return tokens


@dataclass
class CacheSizeConfig:
    """Configuration for cache size management."""

    max_size_mb: int = int(os.getenv("QUERY_ENGINE_MAX_CACHE_MB", "1000"))
    cleanup_threshold_mb: int = int(os.getenv("QUERY_ENGINE_CACHE_CLEANUP_THRESHOLD_MB", "900"))
    check_frequency: int = int(os.getenv("QUERY_ENGINE_CACHE_CHECK_FREQUENCY", "50"))


class DrugNameChecker:
    """Lightweight utility for fast drug name detection in query enhancement.

    This class provides fast regex-based detection without the overhead of
    full PharmaceuticalProcessor initialization. Use when only drug name
    detection is needed (e.g., query enhancement).
    """

    def __init__(self) -> None:
        # Compile regex patterns for performance
        self.cyp_pattern = re.compile(r"\bCYP\d+[A-Z]*\b", re.IGNORECASE)
        self.suffix_patterns = [re.compile(r"\w+" + re.escape(suffix), re.IGNORECASE) for suffix in _DRUG_SUFFIXES]

    def is_drug_like(self, text: str) -> bool:
        """Check if text contains drug-like terms.

        Args:
            text: Text to analyze

        Returns:
            True if drug-like terms are detected
        """
        text_lower = text.lower()

        # Check CYP enzymes
        if self.cyp_pattern.search(text):
            return True

        # Check drug suffixes
        for pattern in self.suffix_patterns:
            if pattern.search(text):
                return True

        # Check common drug names
        words = set(re.findall(r"\b\w+\b", text_lower))
        if any(drug in words for drug in _COMMON_DRUG_NAMES):
            return True

        return False

    def extract_drug_signals(self, text: str) -> dict[str, Any]:
        """Extract drug-related signals from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with signal counts and detected terms
        """
        signals = {"cyp_enzymes": [], "drug_suffix_matches": [], "common_drug_names": [], "signal_count": 0}

        # Find CYP enzymes
        cyp_matches = self.cyp_pattern.findall(text)
        if cyp_matches:
            signals["cyp_enzymes"] = list(set(cyp_matches))
            signals["signal_count"] += len(signals["cyp_enzymes"])

        # Find drug suffix matches
        text_lower = text.lower()
        for suffix in _DRUG_SUFFIXES:
            if suffix in text_lower:
                # Find words ending with this suffix
                pattern = re.compile(r"\b\w+" + re.escape(suffix) + r"\b")
                matches = pattern.findall(text)
                if matches:
                    signals["drug_suffix_matches"].extend(matches)

        if signals["drug_suffix_matches"]:
            signals["signal_count"] += len(set(signals["drug_suffix_matches"]))

        # Find common drug names
        words = set(re.findall(r"\b\w+\b", text_lower))
        found_drugs = [drug for drug in _COMMON_DRUG_NAMES if drug in words]
        if found_drugs:
            signals["common_drug_names"] = found_drugs
            signals["signal_count"] += len(found_drugs)

        return signals


def get_cache_dir_size_mb(cache_dir_path: str | os.PathLike) -> float:
    """Calculate total size of cache directory in megabytes.

    Args:
        cache_dir_path: Path to cache directory

    Returns:
        Size in MB (float)
    """
    cache_dir = Path(cache_dir_path)
    if not cache_dir.exists():
        return 0.0

    total_size = 0
    for cache_file in cache_dir.glob("*.json"):
        try:
            total_size += cache_file.stat().st_size
        except OSError:
            continue

    return total_size / (1024 * 1024)


def cleanup_oldest_cache_files(cache_dir_path: str | os.PathLike, target_size_mb: float) -> dict[str, int]:
    """Remove oldest cache files to meet size target.

    Args:
        cache_dir_path: Path to cache directory
        target_size_mb: Target size in MB

    Returns:
        Dictionary with cleanup statistics
    """
    cache_dir = Path(cache_dir_path)
    if not cache_dir.exists():
        return {"files_removed": 0, "bytes_freed": 0}

    # Get all cache files with their timestamps
    cache_files = []
    for cache_file in cache_dir.glob("*.json"):
        try:
            stat = cache_file.stat()
            cache_files.append((cache_file, stat.st_mtime, stat.st_size))
        except OSError:
            continue

    # Sort by modification time (oldest first)
    cache_files.sort(key=lambda x: x[1])

    files_removed = 0
    bytes_freed = 0
    current_size = sum(size for _, _, size in cache_files)
    target_bytes = target_size_mb * 1024 * 1024

    # Remove oldest files until under target
    for cache_file, _, size in cache_files:
        if current_size <= target_bytes:
            break

        try:
            cache_file.unlink()
            files_removed += 1
            bytes_freed += size
            current_size -= size
        except OSError:
            continue

    return {"files_removed": files_removed, "bytes_freed": bytes_freed, "final_size_mb": current_size / (1024 * 1024)}


__all__ = [
    "_PK_FILTERING_ENABLED",
    "_SPECIES_KEYWORDS",
    "_CLINICAL_STUDY_TAGS",
    "_NEGATION_TERMS",
    "_tokenize_species_string",
    "CacheSizeConfig",
    "DrugNameChecker",
    "get_cache_dir_size_mb",
    "cleanup_oldest_cache_files",
]
