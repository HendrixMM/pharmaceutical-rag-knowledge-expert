"""
PDF Document Loader for RAG Agent
Handles loading and processing PDF documents from local folder
"""

import os
import logging
import json
import html
from io import BytesIO
import re
from typing import List, Optional, Dict, Callable, Any, Pattern
from pathlib import Path

logger = logging.getLogger(__name__)

HAS_REGEX: bool = False
_REGEX_MODULE: Optional[Any] = None
_RE_ENGINE: Any = re

SOFT_LINEBREAK_PATTERN: Optional[Pattern[str]] = None
DOI_REGEX: Optional[Pattern[str]] = None
PMID_REGEX: Optional[Pattern[str]] = None
PMID_EXTRACT_REGEX: Optional[Pattern[str]] = None
HEADER_TERMINATOR_PATTERN: Optional[Pattern[str]] = None
AFFILIATION_LINE_PATTERN: Optional[Pattern[str]] = None
AFFILIATION_PREFIX_PATTERN: Optional[Pattern[str]] = None
AUTHOR_PREFIX_PATTERN: Optional[Pattern[str]] = None
AUTHOR_LINE_PATTERN_UNICODE: Optional[Pattern[str]] = None
AUTHOR_LINE_PATTERN_ASCII: Optional[Pattern[str]] = None
AUTHOR_UNICODE_PREFIX_PATTERN: Optional[Pattern[str]] = None
AUTHOR_UNICODE_PREFIX_PATTERN_ASCII: Optional[Pattern[str]] = None
AUTHOR_NAME_CLUSTER_PATTERN_UNICODE: Optional[Pattern[str]] = None
AUTHOR_NAME_CLUSTER_PATTERN_ASCII: Optional[Pattern[str]] = None
JOURNAL_PATTERN_UNICODE: Optional[Pattern[str]] = None
JOURNAL_PATTERN_ASCII: Optional[Pattern[str]] = None
AFFILIATION_EXCLUSION_PATTERN: Optional[Pattern[str]] = None


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    """Return True when the named environment flag is set to a truthy value."""

    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    return default


def _setup_regex_support(regex_module: Optional[Any]) -> None:
    """Configure regex/regex fallbacks and precompiled patterns."""

    global HAS_REGEX
    global _REGEX_MODULE
    global _RE_ENGINE
    global SOFT_LINEBREAK_PATTERN
    global DOI_REGEX
    global PMID_REGEX
    global PMID_EXTRACT_REGEX
    global HEADER_TERMINATOR_PATTERN
    global AFFILIATION_LINE_PATTERN
    global AFFILIATION_PREFIX_PATTERN
    global AUTHOR_PREFIX_PATTERN
    global AUTHOR_LINE_PATTERN_UNICODE
    global AUTHOR_LINE_PATTERN_ASCII
    global AUTHOR_UNICODE_PREFIX_PATTERN
    global AUTHOR_UNICODE_PREFIX_PATTERN_ASCII
    global AUTHOR_NAME_CLUSTER_PATTERN_UNICODE
    global AUTHOR_NAME_CLUSTER_PATTERN_ASCII
    global JOURNAL_PATTERN_UNICODE
    global JOURNAL_PATTERN_ASCII
    global AFFILIATION_EXCLUSION_PATTERN

    unicode_module = regex_module
    engine = unicode_module if unicode_module is not None else re

    SOFT_LINEBREAK_PATTERN = engine.compile(r'(?<=\w)\n(?=\w)')
    DOI_REGEX = engine.compile(DOI_PATTERN, engine.IGNORECASE)
    PMID_REGEX = engine.compile(PMID_PATTERN, engine.IGNORECASE)
    PMID_EXTRACT_REGEX = engine.compile(PMID_PATTERN_EXTRACT, engine.IGNORECASE)
    HEADER_TERMINATOR_PATTERN = engine.compile(
        r'^(Abstract|Introduction|Background|Keywords|ABSTRACT|INTRODUCTION)\b',
        engine.IGNORECASE,
    )
    AFFILIATION_LINE_PATTERN = engine.compile(
        r'(@|University|Department|Institute|College|Hospital|School|Center|Centre|\.edu|\.org)',
        engine.IGNORECASE,
    )
    AFFILIATION_PREFIX_PATTERN = engine.compile(
        r'(Department|University|Hospital|Institute|College|Center|Centre)',
        engine.IGNORECASE,
    )
    AUTHOR_PREFIX_PATTERN = engine.compile(r'^Authors?:\s+(.+)', engine.IGNORECASE | engine.MULTILINE)

    # ASCII-safe fallback patterns that work with Python's stdlib `re`
    AUTHOR_LINE_PATTERN_ASCII = re.compile(
        r"^[\w'\-·]+(?:\s+[\w'\-·]+)*,\s*[\w'\-·]+(?:\s+[\w'\-·]+)*"
    )
    AUTHOR_UNICODE_PREFIX_PATTERN_ASCII = re.compile(
        r'^Author(?:s)?\s*[:：]\s*([\w\s\-·,.;.-]+)$',
        re.IGNORECASE,
    )
    AUTHOR_NAME_CLUSTER_PATTERN_ASCII = re.compile(
        r"\b[\w'\-·]+(?:\s+[\w'\-·]+)+"
    )
    JOURNAL_PATTERN_ASCII = re.compile(
        r'^[A-Za-z][A-Za-z\s\-:,\.]{3,}\s\d{4}(?:\s?[A-Za-z]{3})?;\s?\d{1,3}(?:\(\d{1,3}\))?:\d{1,5}(?:-\d{1,5})?',
    )

    unicode_patterns_success = False
    if unicode_module is not None:
        try:
            AUTHOR_LINE_PATTERN_UNICODE = unicode_module.compile(
                r'^\p{Lu}[\p{L}\p{M}\-·]*\.?(\s+\p{Lu}[\p{L}\p{M}\-·]*\.?)*,\s*\p{Lu}[\p{L}\p{M}\-·]*\.?(\s+\p{Lu}[\p{L}\p{M}\-·]*\.?)*'
            )
            AUTHOR_UNICODE_PREFIX_PATTERN = unicode_module.compile(
                r'^Author(?:s)?\s*[:：]\s*([\p{L}\p{M}\p{Zs}\p{Pd}·,.;.-]+)$',
                unicode_module.IGNORECASE,
            )
            AUTHOR_NAME_CLUSTER_PATTERN_UNICODE = unicode_module.compile(
                r'\b\p{Lu}[\p{L}\p{M}\-·]+(?:\s+\p{Lu}[\p{L}\p{M}\-·]+)+'
            )
            JOURNAL_PATTERN_UNICODE = unicode_module.compile(
                r'^[\p{L}\p{M}][\p{L}\p{M}\s\-:,\.]{3,}\s\d{4}(?:\s?[A-Za-z]{3})?;\s?\d{1,3}(?:\(\d{1,3}\))?:\d{1,5}(?:-\d{1,5})?'
            )
            unicode_patterns_success = True
        except Exception as err:
            logger.warning(
                "Unicode-aware regex patterns unavailable; falling back to ASCII-safe heuristics: %s",
                err,
            )
            unicode_module = None
            AUTHOR_LINE_PATTERN_UNICODE = None
            AUTHOR_UNICODE_PREFIX_PATTERN = None
            AUTHOR_NAME_CLUSTER_PATTERN_UNICODE = None
            JOURNAL_PATTERN_UNICODE = None
    if not unicode_patterns_success:
        AUTHOR_LINE_PATTERN_UNICODE = None
        AUTHOR_UNICODE_PREFIX_PATTERN = None
        AUTHOR_NAME_CLUSTER_PATTERN_UNICODE = None
        JOURNAL_PATTERN_UNICODE = None

    HAS_REGEX = unicode_patterns_success
    _REGEX_MODULE = unicode_module if unicode_patterns_success else None
    _RE_ENGINE = unicode_module if unicode_patterns_success else re

    AFFILIATION_EXCLUSION_PATTERN = re.compile(
        r'\d{4,}|[A-Z]{2,}\s+\d|PO\s+Box|\b(USA|UK|Canada|Germany|France|China|Japan)\b',
        re.IGNORECASE,
    )


try:
    import regex as _imported_regex  # type: ignore import
except ImportError:
    _imported_regex = None
    logger.warning(
        "The optional 'regex' module is not installed; Unicode-aware PubMed parsing will use "
        "ASCII fallbacks. Install it with 'pip install regex' to restore full functionality."
    )

_PENDING_REGEX_MODULE: Optional[Any] = _imported_regex

# Import unified DOI/PMID patterns and utilities
try:
    from .paper_schema import (
        DOI_PATTERN, PMID_PATTERN, PMID_PATTERN_EXTRACT,
        clean_identifier, normalize_doi, normalize_pmid
    )
    PAPER_SCHEMA_AVAILABLE = True
except ImportError:
    # Define conservative fallback patterns when paper_schema is not available
    PAPER_SCHEMA_AVAILABLE = False

    # Conservative DOI pattern - matches 10.xxxx/xxxx format
    DOI_PATTERN = r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b'

    # Conservative PMID pattern - matches PMIDs in various formats
    PMID_PATTERN = r'\b(PMID:\s*)?(\d{7,8})\b'
    PMID_PATTERN_EXTRACT = r'(\d{7,8})'

    def clean_identifier(id_str: str) -> str:
        """Minimal identifier cleaning - strip whitespace and common prefixes"""
        return id_str.strip().replace('doi:', '').replace('DOI:', '').replace('pmid:', '').replace('PMID:', '').strip()

    def normalize_doi(doi: str) -> str:
        """Minimal DOI normalization - strip whitespace and prefixes"""
        return clean_identifier(doi)

    def normalize_pmid(pmid: str) -> str:
        """Minimal PMID normalization - strip whitespace and prefixes"""
        return clean_identifier(pmid)

    logger.warning("paper_schema module not available - using conservative fallback patterns for DOI/PMID extraction")

# Verify patterns are available
if not PAPER_SCHEMA_AVAILABLE:
    logger.info("Using fallback DOI/PMID patterns - some functionality may be limited")

_setup_regex_support(_PENDING_REGEX_MODULE)


def _safe_sub(pattern: Optional[Pattern[str]], repl: str, text: str, label: str) -> str:
    """Safely apply substitution, returning original text on failure."""

    if pattern is None:
        return text

    try:
        return pattern.sub(repl, text)
    except Exception as err:
        logger.debug("Pattern %s substitution failed: %s", label, err)
        return text


def _safe_search(pattern: Optional[Pattern[str]], text: str, label: str) -> Optional[Any]:
    """Safely perform regex search and return match when available."""

    if pattern is None:
        return None

    try:
        return pattern.search(text)
    except Exception as err:
        logger.debug("Pattern %s search failed: %s", label, err)
        return None


def _safe_match(pattern: Optional[Pattern[str]], text: str, label: str) -> Optional[Any]:
    """Safely perform regex match and return match when available."""

    if pattern is None:
        return None

    try:
        return pattern.match(text)
    except Exception as err:
        logger.debug("Pattern %s match failed: %s", label, err)
        return None


def _safe_findall(pattern: Optional[Pattern[str]], text: str, label: str) -> List[str]:
    """Safely perform regex findall returning empty list on failure."""

    if pattern is None:
        return []

    try:
        return pattern.findall(text)
    except Exception as err:
        logger.debug("Pattern %s findall failed: %s", label, err)
        return []


def _prefer_unicode_pattern(unicode_pattern: Optional[Pattern[str]], ascii_pattern: Optional[Pattern[str]]) -> Optional[Pattern[str]]:
    """Return the Unicode-capable pattern when present, otherwise fall back to ASCII pattern."""

    if HAS_REGEX and unicode_pattern is not None:
        return unicode_pattern
    return ascii_pattern


JOURNAL_KEYWORDS = [
    'journal', 'nature', 'science', 'cell', 'lancet', 'nejm', 'bmj', 'jama',
    'plos', 'proceedings', 'annals', 'review', 'research', 'medicine',
    'biology', 'chemistry', 'physics', 'therapeutics', 'clinical'
]


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
    from langchain_community.document_loaders.parsers.pdf import PyPDFParser
except ImportError:
    PyPDFParser = None  # type: ignore[assignment]

from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain_core.documents.base import Blob
except ImportError:
    try:
        # Older langchain_core exposes Blob directly under documents
        from langchain_core.documents import Blob  # type: ignore
    except ImportError:
        Blob = None  # type: ignore[assignment]

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from dateutil.parser import parse as parse_date
except ImportError:
    parse_date = None

class PDFDocumentLoader:
    """Handles loading and processing PDF documents"""
    
    def __init__(self, docs_folder: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF document loader
        
        Args:
            docs_folder: Path to folder containing PDF documents
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.docs_folder = Path(docs_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Reuse parser and PDF reader class across loads to cut redundant work.
        self._use_parser = PyPDFParser is not None and Blob is not None
        self._pdf_parser = PyPDFParser(mode="page") if self._use_parser else None
        self._pdf_reader_cls = self._detect_pdf_reader()
        
        # Ensure docs folder exists
        self.docs_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized PDF loader for folder: {self.docs_folder}")

    def _detect_pdf_reader(self) -> Optional[Any]:
        """Return a PdfReader implementation if available."""
        try:
            from pypdf import PdfReader  # type: ignore import

            return PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore import

                return PdfReader
            except ImportError:
                logger.debug("pypdf/PyPDF2 not available for XMP metadata extraction.")
                return None

    def _create_pdf_reader(self, pdf_path: Path, pdf_bytes: Optional[bytes] = None) -> Optional[Any]:
        """Instantiate a PdfReader without reopening the file when bytes are provided."""
        if self._pdf_reader_cls is None:
            return None

        if pdf_bytes is not None:
            try:
                return self._pdf_reader_cls(BytesIO(pdf_bytes))
            except Exception as reader_error:
                logger.debug("Unable to create PdfReader from bytes for %s: %s", pdf_path, reader_error)

        try:
            return self._pdf_reader_cls(str(pdf_path))
        except Exception as reader_error:
            logger.debug("Unable to read %s for metadata: %s", pdf_path, reader_error)
            return None

    def _extract_pubmed_metadata_from_xmp(
        self,
        pdf_path: Path,
        doi_pattern: str,
        sanitize_author_line: Callable[[str], str],
        pdf_reader: Optional[Any] = None,
    ) -> Dict[str, str]:
        """Extract metadata from PDF XMP information when available."""
        extracted: Dict[str, str] = {}
        reader = pdf_reader
        if reader is None:
            reader = self._create_pdf_reader(pdf_path)
            if reader is None:
                return extracted

        xmp = getattr(reader, "xmp_metadata", None)
        doc_info = getattr(reader, "metadata", None)

        def _record_identifier(identifier_text: str) -> None:
            cleaned = str(identifier_text).strip()
            if not cleaned:
                return

            doi_match = re.search(doi_pattern, cleaned, re.IGNORECASE)
            if doi_match and 'doi' not in extracted:
                raw_doi = doi_match.group().replace('doi:', '').replace('DOI:', '').strip()
                clean_doi = clean_identifier(raw_doi.rstrip('.,;)'))
                extracted['doi'] = normalize_doi(clean_doi)
            elif cleaned.lower().startswith('10.') and 'doi' not in extracted:
                clean_doi = clean_identifier(cleaned.rstrip('.,;)'))
                extracted['doi'] = normalize_doi(clean_doi)

            pmid_match = re.search(PMID_PATTERN, cleaned, re.IGNORECASE)
            if pmid_match and 'pmid' not in extracted:
                clean_pmid = clean_identifier(pmid_match.group(1))
                extracted['pmid'] = normalize_pmid(clean_pmid)

        identifiers_to_check: List[str] = []

        if xmp:
            identifiers = getattr(xmp, "dc_identifier", None)
            if identifiers:
                if not isinstance(identifiers, (list, tuple)):
                    identifiers = [identifiers]
                identifiers_to_check.extend(str(item) for item in identifiers if str(item).strip())

            creators = getattr(xmp, "dc_creator", None)
            if creators and 'authors' not in extracted:
                if not isinstance(creators, (list, tuple)):
                    creators = [creators]
                authors = ', '.join(str(item).strip() for item in creators if str(item).strip())
                if authors:
                    extracted['authors'] = sanitize_author_line(authors)

            xmp_dates = getattr(xmp, "dc_date", None)
            if xmp_dates and 'publication_date' not in extracted:
                if not isinstance(xmp_dates, (list, tuple)):
                    xmp_dates = [xmp_dates]
                for item in xmp_dates:
                    date_text = str(item).strip()
                    if not date_text:
                        continue
                    if parse_date:
                        try:
                            parsed = parse_date(date_text)
                            if parsed:
                                extracted['publication_date'] = parsed.isoformat()
                                break
                        except Exception:
                            pass
                    extracted['publication_date'] = date_text
                    break

        for identifier in identifiers_to_check:
            _record_identifier(identifier)
            if 'doi' in extracted and 'pmid' in extracted:
                break

        doc_info_get = getattr(doc_info, "get", None)
        if doc_info_get:
            doc_author = doc_info_get('/Author')
            if doc_author and 'authors' not in extracted:
                extracted['authors'] = sanitize_author_line(str(doc_author))

            for key in ('/doi', '/DOI', '/Identifier', '/Subject', '/Keywords'):
                value = doc_info_get(key)
                if value:
                    _record_identifier(value)
                    if 'doi' in extracted and 'pmid' in extracted:
                        break

            if 'publication_date' not in extracted:
                for key in ('/CreationDate', '/ModDate'):
                    value = doc_info_get(key)
                    if not value:
                        continue

                    cleaned = str(value).strip()
                    if cleaned.startswith('D:'):
                        cleaned = cleaned[2:]

                    if parse_date:
                        try:
                            parsed = parse_date(cleaned)
                            if parsed:
                                extracted['publication_date'] = parsed.isoformat()
                                break
                        except Exception:
                            pass

                    extracted['publication_date'] = cleaned
                    break

        return extracted

    def _extract_pubmed_metadata_from_text(
        self,
        text: str,
        pdf_path: Optional[Path] = None,
        pdf_reader: Optional[Any] = None,
    ) -> Dict:
        """
        Extract PubMed metadata from PDF text content

        Args:
            text: Text content from first 1-3 pages
            pdf_path: Optional PDF path for XMP metadata inspection

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        enable_mesh_from_pdf = _env_flag_enabled("ENABLE_MESH_FROM_PDF", False)

        def sanitize_author_line(raw_line: str) -> str:
            """
            Remove affiliations, emails, and superscripts from detected author text.
            Authors are always stored as comma-separated strings for consistency with metadata standards.
            """
            sanitized = re.sub(r'\b\S+@\S+\b', '', raw_line)
            sanitized = re.sub(r'\[[^\]]*\]', '', sanitized)
            sanitized = re.sub(r'[\d*†]+', '', sanitized)
            sanitized = re.sub(r'\s+', ' ', sanitized).strip(' ,;')
            if len(sanitized) > 200:
                trimmed = sanitized[:197].rstrip(',; ')
                sanitized = f"{trimmed}..."
            return sanitized

        # Use shared DOI pattern
        doi_pattern = DOI_PATTERN

        if pdf_path:
            xmp_metadata = self._extract_pubmed_metadata_from_xmp(
                pdf_path,
                doi_pattern,
                sanitize_author_line,
                pdf_reader=pdf_reader,
            )
            for key, value in xmp_metadata.items():
                if key not in metadata and value:
                    metadata[key] = value

        # Apply DOI regex across text with soft line-breaks collapsed
        text_collapsed = _safe_sub(SOFT_LINEBREAK_PATTERN, ' ', text, 'soft_linebreak')
        doi_match = _safe_search(DOI_REGEX, text_collapsed, 'doi_extraction')
        if doi_match:
            raw_doi = doi_match.group().replace('doi:', '').replace('DOI:', '').strip()
            clean_doi = clean_identifier(raw_doi.rstrip('.,;)'))
            metadata['doi'] = normalize_doi(clean_doi)

        # Use shared PMID pattern for extraction
        pmid_match = _safe_search(PMID_EXTRACT_REGEX, text, 'pmid_extraction')
        if pmid_match:
            clean_pmid = clean_identifier(pmid_match.group(1))
            metadata['pmid'] = normalize_pmid(clean_pmid)

        # Enhanced author line heuristics with boundary detection and affiliation filtering
        lines = text.split('\n')
        cumulative_chars = 0
        author_line_pattern = _prefer_unicode_pattern(AUTHOR_LINE_PATTERN_UNICODE, AUTHOR_LINE_PATTERN_ASCII)
        author_prefix_pattern = _prefer_unicode_pattern(AUTHOR_UNICODE_PREFIX_PATTERN, AUTHOR_UNICODE_PREFIX_PATTERN_ASCII)
        author_cluster_pattern = _prefer_unicode_pattern(AUTHOR_NAME_CLUSTER_PATTERN_UNICODE, AUTHOR_NAME_CLUSTER_PATTERN_ASCII)

        for line in lines[:80]:  # Check first 80 lines with character bound
            cumulative_chars += len(line)
            if cumulative_chars > 2500:  # Upper bound on cumulative characters processed
                break
            line = line.strip()

            if _safe_match(HEADER_TERMINATOR_PATTERN, line, 'section_header'):
                break

            if _safe_search(AFFILIATION_LINE_PATTERN, line, 'affiliation_line'):
                continue

            tokens = line.split()
            if any(_safe_search(AFFILIATION_PREFIX_PATTERN, token, 'affiliation_prefix') for token in tokens[:2]):
                continue

            author_line_match = _safe_match(author_line_pattern, line, 'author_line') if author_line_pattern else None
            if author_line_match:
                if len(line.split(',')) >= 2 and len(line) < 200:
                    if not _safe_search(AFFILIATION_EXCLUSION_PATTERN, line, 'author_affiliation_exclusion'):
                        sanitized_authors = sanitize_author_line(line)
                        if sanitized_authors:
                            metadata['authors'] = sanitized_authors
                            break

        # Fallback check for explicit "Authors:" prefix if not found above
        if 'authors' not in metadata:
            prefix_match = _safe_search(AUTHOR_PREFIX_PATTERN, text, 'authors_prefix')
            if prefix_match:
                sanitized_authors = sanitize_author_line(prefix_match.group(1).strip())
                if sanitized_authors:
                    metadata['authors'] = sanitized_authors

        # Secondary Unicode-aware fallback for Author(s) lines that may include non-Latin scripts
        if 'authors' not in metadata and author_prefix_pattern is not None:
            cumulative_chars = 0
            for line in lines[:80]:
                cumulative_chars += len(line)
                if cumulative_chars > 2500:
                    break
                unicode_match = _safe_match(author_prefix_pattern, line, 'unicode_author_prefix')
                if unicode_match:
                    sanitized_authors = sanitize_author_line(unicode_match.group(1).strip())
                    if sanitized_authors:
                        metadata['authors'] = sanitized_authors
                        break

        # Relaxed fallback: allow potential affiliation words but require multiple capitalized names
        if 'authors' not in metadata and author_cluster_pattern is not None:
            for line in lines[:80]:
                candidate = line.strip()
                if not candidate or len(candidate) > 140:
                    continue
                name_matches = _safe_findall(author_cluster_pattern, candidate, 'author_name_cluster')
                if len(name_matches) >= 2:
                    sanitized_authors = sanitize_author_line(candidate)
                    if sanitized_authors:
                        metadata['authors'] = sanitized_authors
                        break

        # Extract date strings and parse them
        if parse_date:
            date_patterns = [
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
                r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, text)
                if date_match:
                    try:
                        parsed_date = parse_date(date_match.group())
                        metadata['publication_date'] = parsed_date.isoformat()
                        break
                    except:
                        continue

        # Attempt to capture journal information if still missing
        if 'journal' not in metadata:
            journal_pattern = _prefer_unicode_pattern(JOURNAL_PATTERN_UNICODE, JOURNAL_PATTERN_ASCII)
            if journal_pattern is not None:
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if len(stripped) < 12 or len(stripped) > 200:
                        continue

                    if not _safe_match(journal_pattern, stripped, 'journal_line'):
                        continue

                    validation_context = []
                    for j in range(max(0, i - 1), min(len(lines), i + 2)):
                        validation_context.append(lines[j].strip())
                    context_text = ' '.join(validation_context)
                    context_text_lower = context_text.lower()

                    has_doi_pmid = bool(
                        _safe_search(DOI_REGEX, context_text, 'journal_context_doi') or
                        _safe_search(PMID_REGEX, context_text, 'journal_context_pmid')
                    )

                    has_journal_keyword = any(keyword in context_text_lower for keyword in JOURNAL_KEYWORDS)

                    if has_doi_pmid or has_journal_keyword:
                        metadata['journal'] = stripped
                        break

        # Extract abstract content after detecting an Abstract header when missing
        if 'abstract' not in metadata:
            abstract_pattern = re.compile(r'^\s*Abstract\s*[:\-]?\s*$', re.IGNORECASE)
            lines = text.split('\n')
            abstract_lines: List[str] = []
            capture = False
            for line in lines:
                if not capture and abstract_pattern.match(line):
                    capture = True
                    continue
                if capture:
                    if re.match(r'^\s*(Introduction|Background|Methods|Materials|Patients|Results|Discussion|Conclusion)s?\b', line, re.IGNORECASE):
                        break
                    abstract_lines.append(line.strip())
            abstract_text = ' '.join(chunk for chunk in abstract_lines if chunk)
            if abstract_text:
                metadata['abstract'] = abstract_text

        # Extract MeSH terms
        mesh_terms: List[str] = []

        # Look for lines starting with "MeSH terms:" or "MeSH:"
        mesh_pattern1 = r'^(MeSH\s+terms?):\s*(.+)$'
        for line in text.split('\n'):
            line = line.strip()
            match = re.search(mesh_pattern1, line, re.IGNORECASE)
            if match:
                mesh_text = match.group(2)
                # Split on commas and semicolons, clean up terms
                terms = re.split(r'[,;]', mesh_text)
                mesh_terms.extend([term.strip() for term in terms if term.strip()])
                break

        # Alternative pattern: look for "MeSH terms" or "MeSH" followed by content
        if not mesh_terms:
            mesh_pattern2 = r'\b(MeSH\s+terms?)\b[:\s]+(.+?)(?=\n\n|\n[A-Z]|\Z)'
            match = re.search(mesh_pattern2, text, re.IGNORECASE | re.DOTALL)
            if match:
                mesh_text = match.group(2)
                # Split on commas and semicolons, clean up terms
                terms = re.split(r'[,;]', mesh_text)
                mesh_terms.extend([term.strip() for term in terms if term.strip()])

        extra_mesh_terms: List[str] = []
        if enable_mesh_from_pdf:
            raw_lines = text.split('\n')
            mesh_heading_pattern = re.compile(r'^(.+?)\s*\[MeSH\s+Terms\]\s*$', re.IGNORECASE)
            for raw_line in raw_lines:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                heading_match = mesh_heading_pattern.match(stripped)
                if heading_match:
                    extracted = heading_match.group(1)
                    extra_mesh_terms.extend(
                        [term.strip() for term in re.split(r'[,;/]', extracted) if term.strip()]
                    )

            if not extra_mesh_terms:
                heading_labels = {
                    'mesh terms',
                    'mesh headings',
                    'major mesh terms',
                }
                for index, raw_line in enumerate(raw_lines):
                    stripped = raw_line.strip()
                    if stripped.lower() not in heading_labels:
                        continue

                    collected: List[str] = []
                    for follower_raw in raw_lines[index + 1:index + 8]:
                        follower = follower_raw.strip()
                        if not follower:
                            break
                        if follower.endswith(':') and len(follower.split()) <= 4:
                            break
                        if re.match(r'^[A-Z][A-Za-z\s/-]+:$', follower):
                            break
                        cleaned = follower.lstrip('-*•').strip()
                        if not cleaned:
                            continue
                        collected.extend(
                            [term.strip() for term in re.split(r'[,;/]', cleaned) if term.strip()]
                        )
                    if collected:
                        extra_mesh_terms.extend(collected)
                        break

        if extra_mesh_terms:
            existing_lower = {existing.lower() for existing in mesh_terms}
            for term in extra_mesh_terms:
                lowered = term.lower()
                if lowered not in existing_lower:
                    mesh_terms.append(term)
                    existing_lower.add(lowered)

        # Add mesh_terms to metadata if found
        if mesh_terms:
            metadata['mesh_terms'] = self._normalize_mesh_terms(mesh_terms)

        return metadata

    def _normalize_authors(self, authors) -> str:
        """Normalize authors field to consistent string format.

        Examples:
            >>> loader = PDFDocumentLoader.__new__(PDFDocumentLoader)
            >>> loader._normalize_authors(["Jane Doe", {"first": "John", "last": "Smith"}])
            'Jane Doe, Smith, John'
            >>> loader._normalize_authors({"name": "Dr. Ada Lovelace"})
            'Dr. Ada Lovelace'

        Args:
            authors: Authors field (could be string, list, dict, or other)

        Returns:
            Normalized authors string
        """

        def _format_author(author_obj) -> str:
            if isinstance(author_obj, dict):
                name_value = author_obj.get("name")
                if isinstance(name_value, str) and name_value.strip():
                    return name_value.strip()

                last = author_obj.get("last") or author_obj.get("family")
                first = author_obj.get("first") or author_obj.get("given")

                if last and first:
                    return f"{str(last).strip()}, {str(first).strip()}"
                if last:
                    return str(last).strip()
                if first:
                    return str(first).strip()

                return str(author_obj).strip()

            if isinstance(author_obj, str):
                return author_obj.strip()

            return str(author_obj).strip()

        if isinstance(authors, list):
            normalized = [_format_author(author) for author in authors]
            return ", ".join(filter(None, normalized))

        if isinstance(authors, dict):
            return _format_author(authors)

        if isinstance(authors, str):
            return authors.strip()

        return str(authors).strip() if authors is not None else ""

    def _normalize_mesh_terms(self, mesh_terms) -> list:
        """
        Normalize mesh_terms field to consistent list format.
        Always returns a list of strings for consistency with metadata standards.

        Args:
            mesh_terms: MeSH terms field (could be string, list, or other)

        Returns:
            Normalized mesh_terms as list[str]
        """
        if isinstance(mesh_terms, list):
            terms_list = [str(term).strip() for term in mesh_terms if str(term).strip()]
        elif isinstance(mesh_terms, str):
            # If string, split by comma and clean
            terms_list = [term.strip() for term in mesh_terms.split(',') if term.strip()]
        else:
            return []

        seen = set()
        result = []
        for term in terms_list:
            t = term.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                result.append(t)
        return result

    def _truncate_metadata_fields(self, metadata: Dict[str, Any], source_label: str) -> None:
        """Cap oversized metadata fields to avoid excessive payloads."""

        authors_cap = _get_env_int('DOC_METADATA_AUTHORS_MAX_LEN', 500)
        abstract_cap = _get_env_int('DOC_METADATA_ABSTRACT_MAX_LEN', 4000)

        authors_value = metadata.get('authors')
        if isinstance(authors_value, str) and len(authors_value) > authors_cap:
            truncated = authors_value[:authors_cap].rstrip()
            metadata['authors'] = truncated
            logger.warning(
                "Truncated authors metadata for %s from %s to %s characters.",
                source_label,
                len(authors_value),
                authors_cap,
            )

        abstract_value = metadata.get('abstract')
        if isinstance(abstract_value, str) and len(abstract_value) > abstract_cap:
            truncated = abstract_value[:abstract_cap].rstrip()
            metadata['abstract'] = truncated
            logger.warning(
                "Truncated abstract metadata for %s from %s to %s characters.",
                source_label,
                len(abstract_value),
                abstract_cap,
            )

    def _extract_pubmed_metadata(self, pdf_path: Path) -> Dict:
        """
        Extract PubMed metadata from PDF file and optional sidecar JSON with deterministic precedence.

        Merge order:
        1. Start with extracted PDF/XMP metadata
        2. Overlay sidecar fields for: doi, pmid, publication_date, authors, mesh_terms, journal, abstract

        Sidecar JSON schema (produced by src.pubmed_scraper.write_sidecar_for_pdf):
            - doi: str
            - pmid: str
            - title: str
            - authors: str or list[str]
            - abstract: str
            - publication_date: ISO8601 string
            - journal: str
            - mesh_terms: list[str]

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with merged metadata
        """
        # Start with PDF/XMP extracted metadata
        metadata = self._extract_pubmed_metadata_from_xmp(pdf_path)

        # Check for sidecar JSON file
        json_path = pdf_path.with_suffix('.pubmed.json')
        if json_path.exists():
            sidecar_data = self._parse_pubmed_sidecar(json_path)
            if sidecar_data:
                logger.info(f"Loaded PubMed metadata from {json_path.name}")

                # Overlay specific sidecar fields with defined precedence
                overlay_fields = [
                    'doi', 'pmid', 'publication_date', 'authors',
                    'mesh_terms', 'journal', 'abstract'
                ]

                for field in overlay_fields:
                    if field in sidecar_data and sidecar_data[field]:
                        metadata[field] = sidecar_data[field]
                        logger.debug(f"Applied sidecar value for {field}: {sidecar_data[field]}")

        return metadata

    def _parse_pubmed_sidecar(self, json_path: Path) -> Dict[str, Any]:
        """
        Parse and normalize metadata from a PubMed sidecar JSON file.

        Args:
            json_path: Path to the .pubmed.json sidecar file

        Returns:
            Dictionary with normalized metadata fields.
            Returns empty dict if file cannot be parsed.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Extract and normalize metadata fields
            metadata = {}

            # Normalize authors field
            authors = json_data.get('authors')
            if authors:
                metadata['authors'] = self._normalize_authors(authors)

            # Normalize MeSH terms
            mesh_terms = json_data.get('mesh_terms')
            if mesh_terms:
                metadata['mesh_terms'] = self._normalize_mesh_terms(mesh_terms)

            # Normalize DOI
            raw_doi = json_data.get('doi')
            if raw_doi is not None and str(raw_doi).strip():
                doi_original = str(raw_doi).strip()
                cleaned_doi = clean_identifier(doi_original.rstrip('.,;)'))
                normalized_doi = normalize_doi(cleaned_doi)
                if normalized_doi:
                    if normalized_doi != doi_original:
                        logger.info(
                            "Normalized DOI value for %s: '%s' -> '%s'",
                            json_path.name,
                            doi_original,
                            normalized_doi,
                        )
                    metadata['doi'] = normalized_doi

            # Normalize PMID
            raw_pmid = json_data.get('pmid')
            if raw_pmid is not None and str(raw_pmid).strip():
                pmid_original = str(raw_pmid).strip()
                cleaned_pmid = clean_identifier(pmid_original)
                normalized_pmid = normalize_pmid(cleaned_pmid)
                if normalized_pmid:
                    if normalized_pmid != pmid_original:
                        logger.info(
                            "Normalized PMID value for %s: '%s' -> '%s'",
                            json_path.name,
                            pmid_original,
                            normalized_pmid,
                        )
                    metadata['pmid'] = normalized_pmid

            # Handle other fields, treating empty strings as missing
            for field in ['title', 'abstract', 'publication_date', 'journal']:
                value = json_data.get(field)
                if value is not None and str(value).strip():
                    metadata[field] = value

            # Optionally normalize publication_date via dateutil if it parses to a valid date
            if 'publication_date' in metadata and parse_date:
                try:
                    parsed_date = parse_date(str(metadata['publication_date']))
                    if parsed_date:
                        metadata['publication_date'] = parsed_date.isoformat()
                except Exception:
                    pass  # Keep original value if parsing fails

            return metadata

        except Exception as e:
            logger.warning(f"Failed to load JSON metadata from {json_path.name}: {str(e)}")
            return {}

    def read_pubmed_sidecar(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Read PubMed metadata from a PDF sidecar JSON file.

        This method provides a public API for accessing sidecar metadata without
        exposing private implementation details.

        Args:
            pdf_path: Path to the PDF file (should have corresponding .pubmed.json sidecar)

        Returns:
            Dictionary with PubMed metadata from the sidecar file.
            Returns empty dict if sidecar doesn't exist or cannot be parsed.

        Sidecar JSON schema:
            - doi: str
            - pmid: str
            - title: str
            - authors: str or list[str]
            - abstract: str
            - publication_date: ISO8601 string
            - journal: str
            - mesh_terms: list[str]
        """
        # Check for sidecar JSON file
        json_path = pdf_path.with_suffix('.pubmed.json')
        if not json_path.exists():
            logger.debug(f"No sidecar file found for {pdf_path.name}")
            return {}

        metadata = self._parse_pubmed_sidecar(json_path)
        if metadata:
            logger.debug(f"Loaded PubMed metadata from {json_path.name}")
        return metadata

    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the specified folder
        
        Returns:
            List of Document objects
        """
        if not self.docs_folder.exists():
            logger.error(f"Documents folder does not exist: {self.docs_folder}")
            return []
        
        # Get all PDF files with case-insensitive globbing
        pdf_patterns = ["*.pdf", "*.PDF"]
        pdf_files_set = set()
        for pattern in pdf_patterns:
            pdf_files_set.update(self.docs_folder.glob(pattern))
        pdf_files = sorted(pdf_files_set, key=lambda path: path.name.lower())
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.docs_folder}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        
        scan_pages_env = os.getenv("PUBMED_SCAN_PAGES")
        try:
            scan_pages = int(scan_pages_env) if scan_pages_env else 3
        except ValueError:
            logger.warning("Invalid PUBMED_SCAN_PAGES value '%s'; defaulting to 3", scan_pages_env)
            scan_pages = 3
        scan_pages = max(1, scan_pages)

        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading: {pdf_file.name}")

                pdf_reader: Optional[Any] = None
                pdf_documents: List[Document] = []

                if self._use_parser and self._pdf_parser and Blob is not None:
                    try:
                        pdf_bytes = pdf_file.read_bytes()
                    except Exception as read_error:
                        logger.error(f"Error reading {pdf_file.name}: {str(read_error)}")
                        continue

                    blob = Blob.from_data(pdf_bytes, path=str(pdf_file))

                    try:
                        pdf_documents = list(self._pdf_parser.lazy_parse(blob))
                    except Exception as parse_error:
                        logger.error(f"Error parsing {pdf_file.name}: {str(parse_error)}")
                        continue

                    pdf_reader = self._create_pdf_reader(pdf_file, pdf_bytes)
                else:
                    try:
                        loader = PyPDFLoader(str(pdf_file))
                        pdf_documents = loader.load()
                    except Exception as loader_error:
                        logger.error(f"Error loading {pdf_file.name} with PyPDFLoader: {str(loader_error)}")
                        continue

                    pdf_reader = self._create_pdf_reader(pdf_file)

                if pdf_documents:
                    first_doc = pdf_documents[0]
                    if not isinstance(first_doc, Document) or not hasattr(first_doc, "page_content"):
                        raise TypeError(
                            "Configured PDF loader did not return Document objects with page_content"
                        )

                # Extract PubMed metadata from JSON sidecar
                pubmed_metadata = self._extract_pubmed_metadata(pdf_file)

                # Extract metadata from PDF text if not already in JSON
                missing_fields = {
                    'doi',
                    'pmid',
                    'authors',
                    'publication_date',
                    'mesh_terms',
                    'journal',
                    'abstract',
                } - set(pubmed_metadata.keys())
                if missing_fields and pdf_documents:
                    # Get text from first 1-3 pages for metadata extraction
                    text_to_analyze = ""
                    pages_to_scan = min(len(pdf_documents), scan_pages)
                    for page in pdf_documents[:pages_to_scan]:
                        text_to_analyze += page.page_content + "\n"

                    # Extract missing metadata from text
                    text_metadata = self._extract_pubmed_metadata_from_text(
                        text_to_analyze,
                        pdf_file,
                        pdf_reader=pdf_reader,
                    )
                    # Only add fields that are missing from JSON metadata
                    for field in missing_fields:
                        if field in text_metadata:
                            pubmed_metadata[field] = text_metadata[field]

                self._truncate_metadata_fields(pubmed_metadata, pdf_file.name)

                # Add metadata to all pages
                for doc in pdf_documents:
                    doc.metadata.update({
                        "source_file": pdf_file.name,
                        "file_path": str(pdf_file)
                    })
                    doc.metadata.setdefault("source", doc.metadata.get("file_path", str(pdf_file)))
                    # Merge PubMed metadata
                    doc.metadata.update(pubmed_metadata)

                documents.extend(pdf_documents)
                logger.info(f"Loaded {len(pdf_documents)} pages from {pdf_file.name}")

            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {str(e)}")
                continue
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of split document chunks
        """
        if not documents:
            logger.warning("No documents to split")
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks")
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                "chunk_id": i,
                "chunk_size": len(doc.page_content)
            })
        
        logger.info(f"Created {len(split_docs)} document chunks")
        return split_docs
    
    def load_and_split(self) -> List[Document]:
        """
        Load all PDFs and split them into chunks
        
        Returns:
            List of split document chunks
        """
        documents = self.load_documents()
        return self.split_documents(documents)
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about loaded documents

        Args:
            documents: List of documents (chunks)

        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {
                "total_source_documents": 0,
                "total_chunks": 0,
                "total_characters": 0,
                "docs_with_doi": 0,
                "docs_with_pmid": 0
            }

        total_chars = sum(len(doc.page_content) for doc in documents)
        source_files = set(doc.metadata.get("source_file", "unknown") for doc in documents)

        # Count unique source files with DOI and PMID
        files_with_doi = set()
        files_with_pmid = set()
        for doc in documents:
            source_file = doc.metadata.get("source_file", "unknown")
            if doc.metadata.get("doi"):
                files_with_doi.add(source_file)
            if doc.metadata.get("pmid"):
                files_with_pmid.add(source_file)

        docs_with_doi = len(files_with_doi)
        docs_with_pmid = len(files_with_pmid)

        return {
            "total_source_documents": len(source_files),  # Unique source PDF files
            "total_chunks": len(documents),               # Total text chunks
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(documents) if documents else 0,
            "source_files": list(source_files),
            "num_source_files": len(source_files),        # Alias for backward compatibility
            "docs_with_doi": docs_with_doi,
            "docs_with_pmid": docs_with_pmid
        }


def main():
    """Test the document loader"""
    from dotenv import load_dotenv
    load_dotenv()
    
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    
    loader = PDFDocumentLoader(docs_folder)
    documents = loader.load_and_split()
    
    stats = loader.get_document_stats(documents)
    print("Document Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
