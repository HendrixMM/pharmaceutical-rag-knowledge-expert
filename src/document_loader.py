"""
PDF Document Loader for RAG Agent
Handles loading and processing PDF documents from local folder
"""

import os
import logging
import json
import html
from io import BytesIO
import regex as re
from typing import List, Optional, Dict, Callable, Any
from pathlib import Path

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
from langchain.schema import Document

try:
    from dateutil.parser import parse as parse_date
except ImportError:
    parse_date = None

# Set up logging
logger = logging.getLogger(__name__)


def _clean_identifier(identifier: str) -> str:
    """Clean DOI/PMID by stripping zero-width characters and decoding HTML entities."""
    if not identifier:
        return identifier

    # Strip zero-width characters
    cleaned = identifier.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\ufeff', '')

    # Decode common HTML entities
    cleaned = html.unescape(cleaned)

    return cleaned


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
                clean_doi = _clean_identifier(raw_doi.rstrip('.,;)'))
                extracted['doi'] = clean_doi
            elif cleaned.lower().startswith('10.') and 'doi' not in extracted:
                clean_doi = _clean_identifier(cleaned.rstrip('.,;)'))
                extracted['doi'] = clean_doi

            pmid_match = re.search(r'(?:\()?pmid[\s:]*[-]?(\d+)(?:\))?', cleaned, re.IGNORECASE)
            if pmid_match and 'pmid' not in extracted:
                clean_pmid = _clean_identifier(pmid_match.group(1))
                extracted['pmid'] = clean_pmid

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

        def sanitize_author_line(raw_line: str) -> str:
            """Remove affiliations, emails, and superscripts from detected author text."""
            sanitized = re.sub(r'\b\S+@\S+\b', '', raw_line)
            sanitized = re.sub(r'\[[^\]]*\]', '', sanitized)
            sanitized = re.sub(r'[\d*†]+', '', sanitized)
            sanitized = re.sub(r'\s+', ' ', sanitized).strip(' ,;')
            if len(sanitized) > 200:
                trimmed = sanitized[:197].rstrip(',; ')
                sanitized = f"{trimmed}..."
            return sanitized

        # Extract DOI using regex (tightened to avoid trailing bracketed text)
        doi_pattern = r'(?:doi[:\s]*)?10\.\d{4,9}/[^\s)>\]]+'

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
        text_collapsed = re.sub(r'(?<=\w)\n(?=\w)', ' ', text)
        doi_match = re.search(doi_pattern, text_collapsed, re.IGNORECASE)
        if doi_match:
            raw_doi = doi_match.group().replace('doi:', '').replace('DOI:', '').strip()
            # Strip trailing punctuation and clean identifier
            clean_doi = _clean_identifier(raw_doi.rstrip('.,;)'))
            metadata['doi'] = clean_doi

        # Extract PMID using enhanced regex to match PMID-\d+ and (PMID: \d+)
        pmid_pattern = r'(?:\()?PMID[\s:]*[-]?(\d+)(?:\))?'
        pmid_match = re.search(pmid_pattern, text, re.IGNORECASE)
        if pmid_match:
            clean_pmid = _clean_identifier(pmid_match.group(1))
            metadata['pmid'] = clean_pmid

        # Enhanced author line heuristics with boundary detection and affiliation filtering
        lines = text.split('\n')
        found_abstract = False
        cumulative_chars = 0
        for i, line in enumerate(lines[:80]):  # Check first 80 lines with character bound
            cumulative_chars += len(line)
            if cumulative_chars > 2500:  # Upper bound on cumulative characters processed
                break
            line = line.strip()

            # Stop scanning when encountering common headers
            if re.match(r'^(Abstract|Introduction|Background|Keywords|ABSTRACT|INTRODUCTION)\b', line, re.IGNORECASE):
                found_abstract = True
                break

            # Skip lines containing common affiliation tokens
            if re.search(r'(@|University|Department|Institute|College|Hospital|School|Center|Centre|\.edu|\.org)', line, re.IGNORECASE):
                continue

            # Enhanced heuristic: line with multiple comma-separated names
            # Support initials (A. B.) and hyphenated names (Smith-Jones)
            tokens = line.split()
            if any(
                re.search(r'(Department|University|Hospital|Institute|College|Center|Centre)', token, re.IGNORECASE)
                for token in tokens[:2]
            ):
                continue
            if re.match(r'^\p{Lu}[\p{L}-]*\.?(\s+\p{Lu}[\p{L}-]*\.?)*,\s*\p{Lu}[\p{L}-]*\.?(\s+\p{Lu}[\p{L}-]*\.?)*', line):
                if len(line.split(',')) >= 2 and len(line) < 200:
                    # Additional check: ensure it doesn't look like an affiliation line
                    if not re.search(r'\d{4,}|[A-Z]{2,}\s+\d|PO\s+Box|\b(USA|UK|Canada|Germany|France|China|Japan)\b', line):
                        sanitized_authors = sanitize_author_line(line)
                        if sanitized_authors:
                            metadata['authors'] = sanitized_authors
                            break

        # Fallback check for explicit "Authors:" prefix if not found above
        if 'authors' not in metadata:
            m = re.search(r'^Authors?:\s+(.+)', text, re.IGNORECASE | re.MULTILINE)
            if m:
                sanitized_authors = sanitize_author_line(m.group(1).strip())
                if sanitized_authors:
                    metadata['authors'] = sanitized_authors

        # Secondary Unicode-aware fallback for Author(s) lines that may include non-Latin scripts
        if 'authors' not in metadata:
            cumulative_chars = 0
            for line in lines[:80]:
                cumulative_chars += len(line)
                if cumulative_chars > 2500:  # Upper bound on cumulative characters processed
                    break
                unicode_match = re.match(r'^Author(?:s)?\s*[:：]\s*([\p{L}\p{M}\p{Zs}\p{Pd}·,.;.-]+)$', line, re.IGNORECASE)
                if unicode_match:
                    sanitized_authors = sanitize_author_line(unicode_match.group(1).strip())
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
            # Pattern examples: "Journal Name 2023;12(4):567-578" or "Journal Name. 2024 Jan;15(2):123-130"
            journal_pattern = r'\s*[\p{L}\p{M}\s\-:,\.]{5,}\s\d{4}(?:\s?[A-Za-z]{3})?;\s?\d{1,3}(?:\(\d{1,3}\))?:\d{1,5}(?:-\d{1,5})?'
            for line in text.split('\n'):
                stripped = line.strip()
                if len(stripped) < 12 or len(stripped) > 200:
                    continue
                if re.match(journal_pattern, stripped):
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
        mesh_terms = []

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
        Normalize mesh_terms field to consistent list format

        Args:
            mesh_terms: MeSH terms field (could be string, list, or other)

        Returns:
            Normalized mesh_terms list
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

    def _extract_pubmed_metadata(self, pdf_path: Path) -> Dict:
        """
        Extract PubMed metadata from PDF file and optional sidecar JSON

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}

        # Check for sidecar JSON file
        json_path = pdf_path.with_suffix('.pubmed.json')
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # Normalize fields from JSON with proper type handling
                authors = json_data.get('authors')
                if authors:
                    metadata['authors'] = self._normalize_authors(authors)

                mesh_terms = json_data.get('mesh_terms')
                if mesh_terms:
                    metadata['mesh_terms'] = self._normalize_mesh_terms(mesh_terms)

                # Handle other fields normally, treating empty strings as missing
                for field in ['doi', 'pmid', 'title', 'abstract', 'publication_date', 'journal']:
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

                logger.info(f"Loaded PubMed metadata from {json_path.name}")

            except Exception as e:
                logger.warning(f"Failed to load JSON metadata from {json_path.name}: {str(e)}")

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
