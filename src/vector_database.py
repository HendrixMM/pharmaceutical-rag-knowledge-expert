"""
Vector Database for RAG Agent
Handles local vector storage using FAISS for document embeddings
"""

import hashlib
import os
import pickle
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import faiss
except ImportError:
    faiss = None
    print("DEBUG: faiss import failed, some functionality may be limited")

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .nvidia_embeddings import NVIDIAEmbeddings
from .pharmaceutical_processor import PharmaceuticalProcessor
from . import pharma_utils
from .pharma_utils import (
    _tokenize_species_string,
    normalize_text,
)

# Enable species filtering without pharmaceutical enrichment
ALLOW_SPECIES_FILTER_WITHOUT_ENRICHMENT = os.getenv("ALLOW_SPECIES_FILTER_WITHOUT_ENRICHMENT", "false").lower() == "true"

# Maximum iterations for stats collection
VECTOR_DB_STATS_MAX_ITERATIONS = int(os.getenv("VECTOR_DB_STATS_MAX_ITERATIONS", "10000"))

# Vector search oversampling and capping configuration
# Maximum number of documents to fetch in a single batch (prevents memory issues)
VECTOR_DB_MAX_FETCH_SIZE = int(os.getenv("VECTOR_DB_MAX_FETCH_SIZE", "10000"))
# Maximum number of oversampling iterations (prevents infinite loops)
VECTOR_DB_MAX_OVERSAMPLE_ITERATIONS = int(os.getenv("VECTOR_DB_MAX_OVERSAMPLE_ITERATIONS", "3"))
# Default oversampling multiplier when filters are present
VECTOR_DB_DEFAULT_OVERSAMPLE_MULTIPLIER = int(os.getenv("VECTOR_DB_DEFAULT_OVERSAMPLE_MULTIPLIER", "5"))
# Maximum oversampling multiplier (prevents excessive fetching)
VECTOR_DB_MAX_OVERSAMPLE_MULTIPLIER = int(os.getenv("VECTOR_DB_MAX_OVERSAMPLE_MULTIPLIER", "20"))
# Minimum results to target before filtering (ensures we have enough candidates)
VECTOR_DB_MIN_TARGET_RESULTS = int(os.getenv("VECTOR_DB_MIN_TARGET_RESULTS", "10"))




_PHARMA_FILTER_KEYS = {
    "drug_names",
    "drug_annotations",
    "therapeutic_areas",
    "species",
    "cyp_enzymes",
    "interaction_types",
    "pharmacokinetics",
    "pharmaceutical_enriched",
}


class VectorDatabase:
    """Local vector database using FAISS for efficient similarity search"""
    
    def __init__(
        self,
        embeddings: NVIDIAEmbeddings,
        db_path: str = "./vector_db",
        index_name: str = "faiss_index",
        *,
        pharmaceutical_processor: Optional[PharmaceuticalProcessor] = None
    ):
        """
        Initialize vector database

        Args:
            embeddings: NVIDIA embeddings instance
            db_path: Path to store the vector database
            index_name: Name of the FAISS index
            pharmaceutical_processor: Optional PharmaceuticalProcessor instance to reuse.
                If not provided, a new one will be created when needed.
        """
        self.embeddings = embeddings
        self.db_path = Path(db_path)
        self.index_name = index_name
        self.vectorstore: Optional[FAISS] = None
        self.pharmaceutical_processor = pharmaceutical_processor
        self._pharma_metadata_enabled = False
        self._docstore_ids: List[Any] = []
        self._pharma_filter_warning_emitted = False

        # Optional: route to per-model directory
        per_model = os.getenv("VECTOR_DB_PER_MODEL", "false").strip().lower() in {"1", "true", "yes", "on"}
        if per_model:
            model_slug = str(getattr(embeddings, "model_name", "default")).replace("/", "_")
            self.db_path = self.db_path / model_slug

        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Paths for saving/loading
        self.index_path = self.db_path / f"{index_name}.faiss"
        self.docstore_path = self.db_path / f"{index_name}.pkl"
        
        logger.info(f"Initialized vector database at: {self.db_path}")

        # Startup validation: check index dimension compatibility when not per-model
        if not per_model:
            try:
                if self.index_path.exists() and faiss is not None:
                    # Load FAISS index header to read dimension
                    faiss_index = faiss.read_index(str(self.index_path))
                    faiss_dim = int(getattr(faiss_index, 'd', 0))
                    embed_dim = int(self.embeddings.get_embedding_dimension())
                    if faiss_dim and embed_dim and faiss_dim != embed_dim:
                        msg = (
                            f"FAISS index dimension ({faiss_dim}) does not match current embedding dimension ({embed_dim}). "
                            f"Either set VECTOR_DB_PER_MODEL=true to use per-model directories or rebuild the index at {self.db_path}."
                        )
                        logger.error(msg)
                        raise RuntimeError(msg)
            except Exception as e:
                logger.debug("Startup dimension validation notice: %s", e)

    def update_base_path(self, new_path: Union[str, Path]) -> None:
        """Update storage paths when the base directory changes."""
        self.db_path = Path(new_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.db_path / f"{self.index_name}.faiss"
        self.docstore_path = self.db_path / f"{self.index_name}.pkl"
        logger.info("Vector database storage path updated to: %s", self.db_path)

    def create_index(
        self,
        documents: List[Document],
        *,
        extract_pharmaceutical_metadata: bool = False,
    ) -> bool:
        """
        Create vector index from documents
        
        Args:
            documents: List of documents to index
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return False
        
        try:
            logger.info(f"Creating vector index from {len(documents)} documents...")

            processed_docs = (
                [self._prepare_pharmaceutical_document(doc) for doc in documents]
                if extract_pharmaceutical_metadata
                else [self._ensure_document_metadata(doc) for doc in documents]
            )

            texts = [doc.page_content for doc in processed_docs]
            metadatas = [doc.metadata for doc in processed_docs]
            
            # Create FAISS vectorstore
            self.vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            self._capture_doc_ids()
            self._pharma_metadata_enabled = (
                self._pharma_metadata_enabled or extract_pharmaceutical_metadata
            )
            if not self._pharma_metadata_enabled:
                self._pharma_metadata_enabled = self._detect_pharma_metadata()

            logger.info("✅ Vector index created successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}")
            return False

    def create_pharmaceutical_index(self, documents: List[Document]) -> bool:
        """Create an index with enriched pharmaceutical metadata."""
        logger.info("Creating pharmaceutical-aware vector index...")
        return self.create_index(documents, extract_pharmaceutical_metadata=True)
    
    def save_index(self) -> bool:
        """
        Save the vector index to disk
        
        Returns:
            True if successful, False otherwise
        """
        if not self.vectorstore:
            logger.error("No vector index to save")
            return False
        
        try:
            logger.info("Saving vector index to disk...")
            
            # Save FAISS index
            self.vectorstore.save_local(str(self.db_path), self.index_name)
            
            logger.info(f"✅ Vector index saved to: {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save vector index: {str(e)}")
            return False
    
    def load_index(self, *, allow_dangerous_deserialization: bool = False) -> bool:
        """
        Load vector index from disk.

        Args:
            allow_dangerous_deserialization: Forwarded to FAISS.load_local;
                defaults to False for safety.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.index_path.exists():
                logger.warning(f"No saved index found at: {self.index_path}")
                return False
            
            logger.info("Loading vector index from disk...")
            
            # Load FAISS index
            self.vectorstore = FAISS.load_local(
                str(self.db_path),
                self.embeddings,
                index_name=self.index_name,
                allow_dangerous_deserialization=allow_dangerous_deserialization,
            )
            self._capture_doc_ids()
            self._pharma_metadata_enabled = self._detect_pharma_metadata()

            # Optional dimension compatibility check after loading
            try:
                vs = self.vectorstore
                if vs is not None:
                    faiss_dim = getattr(getattr(vs, 'index', None), 'd', None)
                    embed_dim = self.embeddings.get_embedding_dimension()
                    if isinstance(faiss_dim, int) and isinstance(embed_dim, int) and faiss_dim != embed_dim:
                        logger.warning(
                            "FAISS index dimension %s != active embedder dimension %s; load failed.",
                            faiss_dim, embed_dim,
                        )
                        return False
            except Exception as dim_err:
                logger.debug("Dimension compatibility check failed: %s", dim_err)

            logger.info("✅ Vector index loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector index: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document], *, auto_enrich_on_add: bool = True) -> bool:
        """
        Add new documents to existing index

        Args:
            documents: List of documents to add
            auto_enrich_on_add: When pharmaceutical metadata enrichment is enabled,
                automatically extract metadata for newly added documents (default: True).
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided to add")
            return False
        
        try:
            if not self.vectorstore:
                logger.info("No existing index, creating new one...")
                return self.create_index(
                    documents,
                    extract_pharmaceutical_metadata=self._pharma_metadata_enabled and auto_enrich_on_add,
                )
            
            logger.info(f"Adding {len(documents)} documents to existing index...")
            
            enrich_on_add = self._pharma_metadata_enabled and auto_enrich_on_add
            if self._pharma_metadata_enabled and not auto_enrich_on_add:
                logger.warning(
                    "Pharmaceutical metadata is enabled but auto enrichment was disabled;"
                    " documents are being added without additional enrichment."
                )

            if enrich_on_add:
                prepared_docs = [self._prepare_pharmaceutical_document(doc) for doc in documents]
            else:
                prepared_docs = [self._ensure_document_metadata(doc) for doc in documents]

            texts = [doc.page_content for doc in prepared_docs]
            metadatas = [doc.metadata for doc in prepared_docs]
            
            # Add to vectorstore
            new_ids = self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self._record_doc_ids(new_ids)
            if not self._pharma_metadata_enabled:
                self._pharma_metadata_enabled = self._detect_pharma_metadata()
            
            logger.info("✅ Documents added successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False

    def add_pharmaceutical_documents(self, documents: List[Document]) -> bool:
        """Add documents with pharmaceutical metadata extraction."""
        if not documents:
            logger.warning("No documents provided to add")
            return False

        try:
            if not self.vectorstore:
                logger.info("No existing index, creating new pharmaceutical index...")
                return self.create_index(documents, extract_pharmaceutical_metadata=True)

            prepared_docs = [self._prepare_pharmaceutical_document(doc) for doc in documents]
            texts = [doc.page_content for doc in prepared_docs]
            metadatas = [doc.metadata for doc in prepared_docs]

            new_ids = self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self._record_doc_ids(new_ids)
            self._pharma_metadata_enabled = True
            logger.info("✅ Pharmaceutical documents added successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to add pharmaceutical documents: {str(e)}")
            return False
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Search query.
            k: Number of results to return.
            score_threshold: Optional maximum FAISS distance. Lower scores
                indicate closer matches, so only documents with distances at or
                below this threshold are returned.

        Returns:
            List of similar documents.
        """
        if not self.vectorstore:
            logger.error("No vector index available for search")
            return []
        
        try:
            logger.debug(f"Searching for: '{query}' (k={k})")
            
            if score_threshold is not None:
                # Search with score threshold
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                filtered_results = [
                    doc for doc, score in results
                    if score <= score_threshold
                ]
                logger.debug(
                    "Found %s results within distance threshold %s",
                    len(filtered_results),
                    score_threshold,
                )
                return filtered_results
            else:
                # Regular similarity search
                results = self.vectorstore.similarity_search(query, k=k)
                logger.debug(f"Found {len(results)} results")
                return results
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def similarity_search_with_pharmaceutical_filters(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
        oversample: Optional[int] = None,
    ) -> List[Document]:
        """Similarity search with post-filtering on pharmaceutical metadata.

        Supported filter keys:
            - drug_names: List of drug names to match
            - drug_annotations: Structured drug annotations (preferred)
            - therapeutic_areas: List of therapeutic areas
            - species: Species preference (alias for species_preference, backward compatibility)
            - species_preference: Species filter with more options (list/tuple/set supported)
            - include_unknown_species: Whether to include documents without species info
            - cyp_enzymes: List of CYP enzymes
            - interaction_types: Types of drug interactions
            - pharmacokinetics: PK filtering (only when ENABLE_PK_FILTERING=true)
                * True/False: Any PK data presence
                * Dict: {"half_life": True, "auc": True} for specific parameters
                * List: ["half_life", "auc"] for required parameters
            - pharmaceutical_enriched: Whether document has pharma metadata
            - min_ranking_score: Minimum ranking score threshold

        Oversampling widens the initial candidate pool (``k * oversample``) so
        more documents survive after metadata filters. The number of iterations
        is controlled by VECTOR_DB_MAX_OVERSAMPLE_ITERATIONS (default: 3), and
        the maximum fetch size is bounded by VECTOR_DB_MAX_FETCH_SIZE (default: 10000).
        The oversample multiplier is capped by VECTOR_DB_MAX_OVERSAMPLE_MULTIPLIER (default: 20).

        For species filtering, documents without species metadata are included
        by default (controlled by pharma_utils.SPECIES_UNKNOWN_DEFAULT).
        To exclude them, set filters["include_unknown_species"] = False.

        Note: PK filtering is only active when ENABLE_PK_FILTERING environment
        variable is set to "true".
        """
        if k <= 0:
            return []

        filters = filters or {}
        effective_filters = dict(filters)

        # Handle species alias for backward compatibility
        if effective_filters.get('species') and not effective_filters.get('species_preference'):
            effective_filters['species_preference'] = effective_filters['species']

        # Check if we should allow species filtering without enrichment
        species_only_filter = (
            ALLOW_SPECIES_FILTER_WITHOUT_ENRICHMENT and
            effective_filters.get('species_preference') and
            not self._pharma_metadata_enabled
        )

        # Remove pharma filters if metadata is not enabled, unless only species filtering
        if (
            effective_filters
            and not self._pharma_metadata_enabled
            and self._has_pharma_filter(effective_filters)
            and not species_only_filter
        ):
            if not self._pharma_filter_warning_emitted:
                logger.warning(
                    "Pharmaceutical filters requested without enriched metadata; "
                    "drug-specific filters will be ignored. Rebuild the index with "
                    "extract_pharmaceutical_metadata=True or re-add documents with "
                    "auto_enrich_on_add enabled."
                )
                self._pharma_filter_warning_emitted = True
            # When species-only filtering is enabled, keep species filters
            if species_only_filter:
                effective_filters = {
                    key: value
                    for key, value in effective_filters.items()
                    if key in {'species_preference', 'include_unknown_species'} or
                    key not in _PHARMA_FILTER_KEYS
                }
            else:
                effective_filters = {
                    key: value
                    for key, value in effective_filters.items()
                    if key not in _PHARMA_FILTER_KEYS
                }
        # Use configured oversample value if not provided
        if oversample is None:
            oversample = VECTOR_DB_DEFAULT_OVERSAMPLE_MULTIPLIER
        else:
            try:
                oversample = int(oversample)
            except (TypeError, ValueError):
                oversample = VECTOR_DB_DEFAULT_OVERSAMPLE_MULTIPLIER

        # Apply bounds to oversample multiplier
        oversample = max(1, min(oversample, VECTOR_DB_MAX_OVERSAMPLE_MULTIPLIER))

        # Use configured caps
        max_iterations = VECTOR_DB_MAX_OVERSAMPLE_ITERATIONS
        max_candidates = VECTOR_DB_MAX_FETCH_SIZE

        # Calculate initial fetch size with guards
        target_results = max(k * oversample, VECTOR_DB_MIN_TARGET_RESULTS)
        fetch_k = min(target_results, max_candidates)
        last_filtered: List[Document] = []

        for iteration in range(max_iterations):
            # Cap fetch size to prevent memory issues
            current_k = min(fetch_k, max_candidates)

            # Guard against excessive fetching
            if iteration > 0 and current_k > fetch_k * 2:
                logger.warning(
                    f"Oversampling growing too large: {current_k} (iteration {iteration}), "
                    f"capping at {max_candidates}"
                )
                current_k = max_candidates

            base_results = self.similarity_search(query, k=current_k)
            if not base_results:
                return []

            filtered_results = self._apply_pharmaceutical_filters(base_results, effective_filters)

            # Check if we have enough results or hit the cap
            if len(filtered_results) >= k or current_k >= max_candidates:
                return filtered_results[:k]

            last_filtered = filtered_results

            # If we got fewer results than requested, no need to continue
            if len(base_results) < current_k:
                logger.debug(f"Exhausted results: only {len(base_results)} of {current_k} available")
                break

            # Calculate next fetch size with exponential backoff cap
            # Use progressive oversampling to avoid excessive fetching
            iteration_multiplier = min(oversample ** (iteration + 1), VECTOR_DB_MAX_OVERSAMPLE_MULTIPLIER)
            fetch_k = min(
                max(current_k + k, current_k * iteration_multiplier),
                max_candidates
            )

            logger.debug(
                f"Iteration {iteration}: fetched {current_k}, "
                f"got {len(filtered_results)} filtered, "
                f"next fetch target: {fetch_k}"
            )

        logger.info(
            f"Oversampling completed after {iteration + 1} iterations with "
            f"{len(last_filtered)} final results (requested {k})"
        )
        return last_filtered[:k]
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with similarity scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vectorstore:
            logger.error("No vector index available for search")
            return []
        
        try:
            logger.debug(f"Searching with scores for: '{query}' (k={k})")
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.debug(f"Found {len(results)} results with scores")
            return results
            
        except Exception as e:
            logger.error(f"Search with scores failed: {str(e)}")
            return []

    def search_by_drug_name(self, drug_name: str, k: int = 10) -> List[Document]:
        """Targeted search for documents mentioning a specific drug."""
        if not drug_name:
            return []
        filters = {"drug_names": [drug_name]}
        results = self.similarity_search_with_pharmaceutical_filters(
            drug_name,
            k=max(k * 2, 10),
            filters=filters,
        )
        return results[:k]

    def search_with_info(self, query: str, k: int = 4, filters: Optional[Dict[str, Any]] = None) -> Tuple[List[Document], Dict[str, Any]]:
        """Search with additional information about pharmaceutical metadata status.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional pharmaceutical filters (only applied if pharma metadata is enabled)

        Returns:
            Tuple of (results, info) where info contains:
            - pharma_metadata_enabled: Whether pharmaceutical filters were applied
            - filter_results: Summary of filter effects
            - ignored_filters: List of filters that were ignored and why
            - warning: Any warnings about filter application
        """
        info = {
            "pharma_metadata_enabled": self.is_pharma_metadata_enabled(),
            "filter_results": None,
            "ignored_filters": [],
            "warning": None,
        }

        if filters and not self.is_pharma_metadata_enabled():
            info["warning"] = "Pharmaceutical filters provided but metadata extraction is disabled. Filters will be ignored."
            # Track which filters would be ignored
            ignored = []
            for key in filters:
                if key in _PHARMA_FILTER_KEYS:
                    ignored.append({"filter": key, "reason": "Pharmaceutical metadata not enabled"})
            info["ignored_filters"] = ignored
            # Fall back to basic search
            results = self.similarity_search(query, k=k)
        else:
            # Use the internal method to get detailed filter information
            results, detailed_info = self._similarity_search_with_pharmaceutical_filters_with_info(
                query,
                k=k,
                filters=filters,
            )
            # Merge the detailed info
            info["ignored_filters"] = detailed_info.get("ignored_filters", [])
            info["filter_stats"] = detailed_info.get("filter_stats", {})
            if filters:
                applied_count = detailed_info.get("effective_filters_count", 0)
                info["filter_results"] = f"Applied {applied_count} of {len(filters)} filter categories"

        return results, info

    def _similarity_search_with_pharmaceutical_filters_with_info(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
        oversample: int = 5,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """Internal version that returns detailed filter information.

        This method mirrors the behavior of similarity_search_with_pharmaceutical_filters
        but provides detailed tracking of which filters are ignored and why.
        """
        if k <= 0:
            return [], {}

        filters = filters or {}
        effective_filters = dict(filters)
        ignored_filters = set()

        # Track which filters are ignored
        def track_ignored_filter(key: str, reason: str) -> None:
            ignored_filters.add((key, reason))

        # Handle species alias for backward compatibility
        if effective_filters.get('species') and not effective_filters.get('species_preference'):
            effective_filters['species_preference'] = effective_filters['species']

        # Check if we should allow species filtering without enrichment
        species_only_filter = (
            ALLOW_SPECIES_FILTER_WITHOUT_ENRICHMENT and
            effective_filters.get('species_preference') and
            not self._pharma_metadata_enabled
        )

        # Remove pharma filters if metadata is not enabled, unless only species filtering
        if (
            effective_filters
            and not self._pharma_metadata_enabled
            and self._has_pharma_filter(effective_filters)
            and not species_only_filter
        ):
            if not self._pharma_filter_warning_emitted:
                logger.warning(
                    "Pharmaceutical filters requested without enriched metadata; "
                    "drug-specific filters will be ignored. Rebuild the index with "
                    "extract_pharmaceutical_metadata=True or re-add documents with "
                    "auto_enrich_on_add enabled."
                )
                self._pharma_filter_warning_emitted = True

            # Track ignored pharma filters
            for key in effective_filters:
                if key in _PHARMA_FILTER_KEYS:
                    track_ignored_filter(key, "Pharmaceutical metadata not enabled")

            # When species-only filtering is enabled, keep species filters
            if species_only_filter:
                effective_filters = {
                    key: value
                    for key, value in effective_filters.items()
                    if key in {'species_preference', 'include_unknown_species'} or
                    key not in _PHARMA_FILTER_KEYS
                }
            else:
                effective_filters = {
                    key: value
                    for key, value in effective_filters.items()
                    if key not in _PHARMA_FILTER_KEYS
                }

        # Track PK filtering ignored when disabled
        if filters.get("pharmacokinetics") and not pharma_utils._PK_FILTERING_ENABLED:
            track_ignored_filter("pharmacokinetics", "PK filtering disabled (set ENABLE_PK_FILTERING=true)")

        try:
            oversample = int(oversample)
        except (TypeError, ValueError):
            oversample = 5
        oversample = max(1, oversample)

        max_iterations = 3
        max_candidates = 10_000
        fetch_k = max(k * oversample, k, 1)
        last_filtered: List[Document] = []

        # Track filter application statistics
        total_before_filter = 0
        total_after_filter = 0

        for _ in range(max_iterations):
            current_k = min(fetch_k, max_candidates)
            base_results = self.similarity_search(query, k=current_k)
            if not base_results:
                break

            total_before_filter += len(base_results)
            filtered_results = self._apply_pharmaceutical_filters_with_info(
                base_results,
                effective_filters,
                ignored_filters,
                track_ignored_filter
            )
            total_after_filter += len(filtered_results)

            if len(filtered_results) >= k or current_k >= max_candidates:
                # Build comprehensive info
                info = {
                    "ignored_filters": [{"filter": key, "reason": reason} for key, reason in ignored_filters],
                    "pharma_metadata_enabled": self._pharma_metadata_enabled,
                    "effective_filters_count": len(effective_filters),
                    "original_filters_count": len(filters) if filters else 0,
                    "filter_stats": {
                        "documents_before_filter": total_before_filter,
                        "documents_after_filter": total_after_filter,
                        "filter_effectiveness": round(total_after_filter / total_before_filter, 3) if total_before_filter > 0 else 0.0,
                        "oversample_iterations": _ + 1,
                        "final_fetch_k": current_k,
                    }
                }
                return filtered_results[:k], info

            last_filtered = filtered_results
            if len(base_results) < current_k:
                break

            fetch_k = max(current_k + k, current_k * oversample)
            if fetch_k >= max_candidates:
                fetch_k = max_candidates

        # Build info for final results
        info = {
            "ignored_filters": [{"filter": key, "reason": reason} for key, reason in ignored_filters],
            "pharma_metadata_enabled": self._pharma_metadata_enabled,
            "effective_filters_count": len(effective_filters),
            "original_filters_count": len(filters) if filters else 0,
            "filter_stats": {
                "documents_before_filter": total_before_filter,
                "documents_after_filter": total_after_filter,
                "filter_effectiveness": round(total_after_filter / total_before_filter, 3) if total_before_filter > 0 else 0.0,
                "oversample_iterations": max_iterations,
                "final_fetch_k": fetch_k,
                "note": "Reached maximum iterations or candidate limit"
            }
        }
        return last_filtered[:k], info

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database
        
        Returns:
            Dictionary with database statistics
        """
        if not self.vectorstore:
            return {"status": "No index loaded", "document_count": 0}
        
        try:
            # Get document count from FAISS index
            doc_count = self.vectorstore.index.ntotal
            stats = {
                "status": "Index loaded",
                "document_count": doc_count,
                "index_path": str(self.index_path),
                "index_exists": self.index_path.exists()
            }
            if self._pharma_metadata_enabled:
                stats["pharmaceutical"] = self.get_pharmaceutical_stats()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {"status": "Error getting stats", "error": str(e)}

    def get_pharmaceutical_stats(self) -> Dict[str, Any]:
        """Return aggregated statistics about pharmaceutical annotations."""
        if not self.vectorstore:
            return {"status": "No index loaded"}

        documents = self._iter_documents()
        if not documents:
            return {"status": "No documents"}

        total_docs = len(documents)
        drug_counter = Counter()
        study_type_counter = Counter()
        therapeutic_counter = Counter()
        species_counter = Counter()
        cyp_counter = Counter()
        ranking_accumulator: List[float] = []
        annotated_docs = 0

        # Apply iteration limit to prevent performance issues
        iterations = min(total_docs, VECTOR_DB_STATS_MAX_ITERATIONS)
        if iterations < total_docs:
            logger.warning(
                "Stats collection limited to %d documents (total: %d). "
                "Adjust VECTOR_DB_STATS_MAX_ITERATIONS to process more.",
                iterations, total_docs
            )

        for i, doc in enumerate(documents):
            if i >= iterations:
                break
            metadata = doc.metadata or {}
            drug_names = self._ensure_string_list(metadata.get("drug_names", []))
            if drug_names:
                annotated_docs += 1
            drug_counter.update(drug_names)

            study_types = self._ensure_string_list(metadata.get("study_types", []))
            if metadata.get("study_type"):
                study_types.append(str(metadata["study_type"]))
            study_type_counter.update(self._ensure_string_list(study_types))

            therapeutic_counter.update(self._ensure_string_list(metadata.get("therapeutic_areas", [])))
            species_meta = metadata.get("species")
            species_values = self._ensure_string_list(species_meta if isinstance(species_meta, (list, tuple, set)) else [species_meta])
            species_counter.update(species_values)

            cyp_counter.update(self._ensure_string_list(metadata.get("cyp_enzymes", [])))

            if "ranking_score" in metadata:
                try:
                    ranking_accumulator.append(float(metadata["ranking_score"]))
                except (TypeError, ValueError):
                    continue

        average_ranking = round(sum(ranking_accumulator) / len(ranking_accumulator), 4) if ranking_accumulator else None
        coverage_ratio = round(annotated_docs / iterations, 4) if iterations else 0.0

        # Build result with sampling info if applicable
        result = {
            "status": "ok",
            "documents_indexed": total_docs,
            "documents_processed": iterations,
            "drug_annotation_ratio": coverage_ratio,
            "top_drug_names": drug_counter.most_common(10),
            "study_type_distribution": study_type_counter.most_common(10),
            "therapeutic_area_distribution": therapeutic_counter.most_common(10),
            "species_distribution": species_counter.most_common(10),
            "cyp_frequencies": cyp_counter.most_common(10),
            "average_ranking_score": average_ranking,
        }

        # Add sampling warning if not all documents were processed
        if iterations < total_docs:
            result["sampling_warning"] = f"Stats based on sample of {iterations:,} documents out of {total_docs:,} total"
            result["sampling_ratio"] = round(iterations / total_docs, 4)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_pharma_filter(self, filters: Dict[str, Any]) -> bool:
        # When species-only filtering is enabled, don't count species as a pharma filter
        if ALLOW_SPECIES_FILTER_WITHOUT_ENRICHMENT and filters.get('species_preference'):
            # Check if filters contain only species-related keys
            non_species_keys = [
                key for key in filters.keys()
                if key not in {'species_preference', 'include_unknown_species', 'species'}
            ]
            # If only species filters are present, return False
            if not non_species_keys:
                return False

        for key in _PHARMA_FILTER_KEYS:
            if key not in filters:
                continue
            value = filters[key]
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)) and not value:
                continue
            if isinstance(value, dict) and not value:
                continue
            return True
        return False

    def _apply_pharmaceutical_filters(
        self,
        documents: List[Document],
        filters: Dict[str, Any],
    ) -> List[Document]:
        if not filters:
            return list(documents)

        filtered: List[Document] = []
        for doc in documents:
            metadata = doc.metadata or {}
            # When species-only filtering is enabled, pass page_content for inference
            page_content = doc.page_content if ALLOW_SPECIES_FILTER_WITHOUT_ENRICHMENT else None
            if self._document_matches_pharma_filters(metadata, filters, page_content):
                filtered.append(doc)
        return filtered

    def _apply_pharmaceutical_filters_with_info(
        self,
        documents: List[Document],
        filters: Dict[str, Any],
        ignored_filters: Set[Tuple[str, str]],
        track_ignored_filter: callable,
    ) -> List[Document]:
        """Apply pharmaceutical filters with detailed tracking of ignored filters.

        This method mirrors _apply_pharmaceutical_filters but adds tracking
        for filters that are ignored during the application process.
        """
        if not filters:
            return list(documents)

        filtered: List[Document] = []
        for doc in documents:
            metadata = doc.metadata or {}
            # When species-only filtering is enabled, pass page_content for inference
            page_content = doc.page_content if ALLOW_SPECIES_FILTER_WITHOUT_ENRICHMENT else None
            if self._document_matches_pharma_filters_with_info(
                metadata, filters, page_content, ignored_filters, track_ignored_filter
            ):
                filtered.append(doc)
        return filtered

    def _document_matches_pharma_filters_with_info(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
        page_content: Optional[str] = None,
        ignored_filters: Optional[Set[Tuple[str, str]]] = None,
        track_ignored_filter: Optional[callable] = None,
    ) -> bool:
        """Check if document metadata matches pharmaceutical filters with tracking.

        This is a wrapper around _document_matches_pharma_filters that adds
        tracking for ignored filters during the matching process.
        """
        # Track PK filtering ignored when disabled at the document level
        if filters.get("pharmacokinetics") and not pharma_utils._PK_FILTERING_ENABLED:
            if ignored_filters is not None and track_ignored_filter is not None:
                track_ignored_filter("pharmacokinetics", "PK filtering disabled (set ENABLE_PK_FILTERING=true)")

        # Delegate to the original method
        return self._document_matches_pharma_filters(metadata, filters, page_content)

    def _document_matches_pharma_filters(
        self,
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
        page_content: Optional[str] = None,
    ) -> bool:
        """Check if document metadata matches pharmaceutical filters.

        For species filtering, documents without species metadata are included
        by default (include_unknown_species=True). Set include_unknown_species=False
        in filters to exclude documents lacking species information.

        When ALLOW_SPECIES_FILTER_WITHOUT_ENRICHMENT is enabled, species inference
        uses page_content for cheap text-based species detection.

        Pharmacokinetics filtering supports:
        - Boolean: filters["pharmacokinetics"] = True (any PK data present)
        - Dict: filters["pharmacokinetics"] = {"half_life": True, "auc": True}
        - List: filters["pharmacokinetics"] = ["half_life", "auc"]
        """
        study_types = filters.get("study_types")
        if study_types:
            # Use normalized study_types from metadata
            doc_study_types = set(metadata.get("study_types", []) or [])
            filter_study_types = {str(v).lower() for v in study_types}
            if not doc_study_types.intersection(filter_study_types):
                return False

        year_range = filters.get("year_range")
        if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
            start, end = year_range
            # Use normalized publication_year
            year = metadata.get("publication_year")
            if year is not None:
                if (start is not None and year < start) or (end is not None and year > end):
                    return False

        drug_names = filters.get("drug_names")
        if drug_names:
            doc_drugs = set()
            # Handle drug_names (simple strings)
            for entry in self._ensure_string_list(metadata.get("drug_names", [])):
                if entry:
                    doc_drugs.add(entry.lower())
            # Handle drug_annotations (structured annotations)
            for entry in metadata.get("drug_annotations", []) or []:
                if isinstance(entry, dict):
                    name = entry.get("name")
                    if name:
                        doc_drugs.add(str(name).lower())
                elif entry:
                    doc_drugs.add(str(entry).lower())
            # Check intersection with filter terms
            filter_drugs = {str(v).lower() for v in drug_names}
            if not doc_drugs.intersection(filter_drugs):
                return False

        therapeutic_areas = filters.get("therapeutic_areas")
        if therapeutic_areas:
            doc_areas = {str(area).lower() for area in metadata.get("therapeutic_areas", []) or []}
            if not doc_areas.intersection({str(v).lower() for v in therapeutic_areas}):
                return False

        min_score = filters.get("min_ranking_score")
        if isinstance(min_score, (int, float)):
            try:
                score = float(metadata.get("ranking_score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if score < float(min_score):
                return False

        species_preference = filters.get("species_preference")
        if species_preference:
            if isinstance(species_preference, (list, tuple, set)):
                preferred_values = [str(value).lower() for value in species_preference if value]
            else:
                preferred_values = [str(species_preference).lower()]

            if not preferred_values:
                return False

            # Use normalized species_list from metadata
            species_values = metadata.get("species_list", []) or []

            # When metadata lacks species, attempt cheap species inference
            if not species_values:
                inferred_species = self._resolve_species(metadata, page_content)
                if inferred_species:
                    # Normalize the inferred species for consistency
                    processor = self._get_pharmaceutical_processor()
                    species_values = processor.normalize_species([inferred_species])

            # Configurable include_unknown_species flag defaulting to True
            include_unknown_species = filters.get("include_unknown_species", True)
            if not species_values and not include_unknown_species:
                return False
            elif not species_values:
                # Include unknown species when flag is True (default)
                return True

            # Use tokenized matching to reduce false positives
            # Note: 'non-human' is normalized to 'nonhuman' to prevent false positives on human studies
            species_tokens = set()
            for value in species_values:
                species_tokens.update(_tokenize_species_string(value))
            preferred_tokens = set()
            for pref in preferred_values:
                preferred_tokens.update(_tokenize_species_string(pref))
            if not preferred_tokens.intersection(species_tokens):
                return False

        # Pharmacokinetics filtering (only when enabled)
        pharmacokinetics = filters.get("pharmacokinetics")
        if pharmacokinetics and not pharma_utils._PK_FILTERING_ENABLED:
            # Skip PK filtering when feature is disabled
            pass
        elif pharmacokinetics:
            # Use normalized pharmacokinetics data (both fields now contain the same normalized data)
            doc_pk = metadata.get("pharmacokinetics", {}) or {}

            if isinstance(pharmacokinetics, dict):
                # Support filtering by presence of PK parameters
                for pk_param, required in pharmacokinetics.items():
                    if required is True:
                        # Check if parameter exists in normalized PK data
                        if pk_param not in doc_pk:
                            return False
                    elif isinstance(required, (str, dict)):
                        # Support regex/value matching (for future enhancement)
                        # For now, just check presence
                        if pk_param not in doc_pk:
                            return False
            elif isinstance(pharmacokinetics, (list, tuple)):
                # Support list of required PK parameters
                for pk_param in pharmacokinetics:
                    if pk_param not in doc_pk:
                        return False
            else:
                # Boolean check - any PK data present
                if pharmacokinetics and not doc_pk:
                    return False

        return True

    def _prepare_pharmaceutical_document(self, document: Document) -> Document:
        doc = self._ensure_document_metadata(document)
        self._extract_pharmaceutical_metadata(doc)
        return doc

    def _ensure_document_metadata(self, document: Document) -> Document:
        if document.metadata is None:
            document.metadata = {}
        return document

    def _ensure_string_list(self, values: Any) -> List[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values_iterable: Iterable[Any] = [values]
        elif isinstance(values, (list, tuple, set)):
            values_iterable = values
        else:
            values_iterable = [values]
        dedup: Dict[str, str] = {}
        for value in values_iterable:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            key = text.lower()
            dedup.setdefault(key, text)
        return list(dedup.values())

    def _resolve_study_type(self, metadata: Dict[str, Any]) -> Optional[str]:
        candidates = [metadata.get("study_type"), metadata.get("study_design"), metadata.get("publication_type")]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        study_types = metadata.get("study_types")
        if isinstance(study_types, list) and study_types:
            for item in study_types:
                if isinstance(item, str) and item.strip():
                    return item
        tags = metadata.get("tags")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    return tag
        return None

    def _resolve_year(self, metadata: Dict[str, Any]) -> Optional[int]:
        for key in ("publication_year", "year"):
            value = metadata.get(key)
            if value is None:
                continue
            try:
                year = int(str(value)[:4])
                return year
            except (TypeError, ValueError):
                continue
        date_field = metadata.get("publication_date") or metadata.get("date")
        if isinstance(date_field, str):
            match = re.match(r"(\d{4})", date_field)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return None
        return None

    def _resolve_species(self, metadata: Dict[str, Any], page_content: Optional[str]) -> Optional[str]:
        species_meta = metadata.get("species") or metadata.get("organism")
        if isinstance(species_meta, str) and species_meta.strip():
            return species_meta
        if isinstance(species_meta, (list, tuple, set)):
            for value in species_meta:
                if isinstance(value, str) and value.strip():
                    return value
        text_sources = [
            metadata.get("title"),
            metadata.get("abstract"),
            metadata.get("summary"),
            page_content,
        ]
        combined = " ".join(filter(None, (str(fragment) for fragment in text_sources))).lower()

        # Use word-boundary regex for more accurate species matching
        import re
        species_patterns = [
            r'\bhuman\b',
            r'\bhumans\b',
            r'\bmouse\b',
            r'\bmice\b',
            r'\brat\b',
            r'\brats\b',
            r'\bdog\b',
            r'\bdogs\b',
            r'\bin vitro\b',
            r'\bnon-?human\b'  # Handles both 'non-human' and 'nonhuman'
        ]

        for pattern in species_patterns:
            match = re.search(pattern, combined)
            if match:
                species = match.group()
                # Normalize 'non-human' variants to 'nonhuman'
                if species.startswith('non'):
                    return 'nonhuman'
                return species
        return None

    def is_pharma_metadata_enabled(self) -> bool:
        """Check if pharmaceutical metadata extraction is enabled for this vector database.

        Returns:
            bool: True if pharmaceutical metadata is enabled and available, False otherwise.
            Callers should check this before passing pharmaceutical filters to avoid
            unexpected behavior when metadata is not available.
        """
        return self._pharma_metadata_enabled

    def get_pharmaceutical_metadata_status(self) -> Dict[str, Any]:
        """Get detailed status information about pharmaceutical metadata capabilities.

        Returns:
            Dict containing:
            - enabled: Whether pharma metadata is enabled
            - processor_available: Whether the processor is available
            - document_count: Number of documents with pharma metadata (if available)
        """
        status = {
            "enabled": self._pharma_metadata_enabled,
            "processor_available": self.pharmaceutical_processor is not None,
            "document_count": None,
        }

        # Try to get document count if metadata is enabled
        if self._pharma_metadata_enabled:
            self._capture_doc_ids()
            status["document_count"] = len(self._docstore_ids)

        return status

    def _get_pharmaceutical_processor(self) -> PharmaceuticalProcessor:
        """Lazy-initialize PharmaceuticalProcessor to avoid overhead when pharma features unused."""
        if self.pharmaceutical_processor is None:
            self.pharmaceutical_processor = PharmaceuticalProcessor()
        return self.pharmaceutical_processor

    def _extract_pharmaceutical_metadata(self, document: Document) -> Dict[str, Any]:
        base_metadata = dict(document.metadata or {})
        processor = self._get_pharmaceutical_processor()
        enriched = processor.enhance_document_metadata(
            {
                "page_content": document.page_content,
                "metadata": base_metadata,
            }
        )
        metadata = dict(enriched.get("metadata", base_metadata))

        # Normalize all metadata fields using the processor
        metadata = self._normalize_metadata_fields(metadata, processor)

        metadata["pharmaceutical_enriched"] = True
        document.metadata = metadata
        return metadata

    def _normalize_metadata_fields(self, metadata: Dict[str, Any], processor: Any) -> Dict[str, Any]:
        """Normalize all metadata fields using pharmaceutical processor.

        Args:
            metadata: Raw metadata dictionary
            processor: PharmaceuticalProcessor instance

        Returns:
            Normalized metadata dictionary
        """
        # Normalize drug names and annotations
        raw_drug_entries = metadata.get("drug_names", [])
        raw_annotations = metadata.get("drug_annotations", [])
        drug_annotations: List[Dict[str, Any]] = []
        drug_names: List[str] = []

        if isinstance(raw_drug_entries, list):
            for entry in raw_drug_entries:
                if isinstance(entry, dict):
                    drug_annotations.append(entry)
                    name = entry.get("name")
                    if name:
                        drug_names.append(str(name))
                elif entry:
                    drug_names.append(str(entry))
        elif isinstance(raw_drug_entries, str):
            drug_names.append(raw_drug_entries)

        if isinstance(raw_annotations, list):
            for entry in raw_annotations:
                if not isinstance(entry, dict):
                    continue
                drug_annotations.append(entry)
                name = entry.get("name")
                if name:
                    drug_names.append(str(name))

        if drug_annotations:
            metadata["drug_annotations"] = drug_annotations
        metadata["drug_names"] = self._ensure_string_list(drug_names)

        # Backward compatibility: create drug_entities alias if it exists in old indexes
        if drug_annotations and "drug_entities" in metadata:
            metadata["drug_entities"] = drug_annotations

        # Normalize MeSH terms and therapeutic areas
        mesh_terms = metadata.get("mesh_terms") or metadata.get("mesh") or []
        normalized_mesh_terms = processor.normalize_mesh_terms(mesh_terms)
        metadata["mesh_terms"] = normalized_mesh_terms
        metadata["therapeutic_areas"] = processor.identify_therapeutic_areas(normalized_mesh_terms)

        # Normalize CYP enzymes
        metadata["cyp_enzymes"] = self._ensure_string_list(
            [item.upper() for item in metadata.get("cyp_enzymes", []) or []]
        )

        # Normalize pharmacokinetic data
        pk_data = metadata.get("pharmacokinetics") or metadata.get("pharmacokinetic_parameters") or {}
        pk_values = metadata.get("pharmacokinetic_values", {})

        # Combine PK data for normalization
        combined_pk = {**pk_data}
        if pk_values:
            combined_pk.update(pk_values)

        normalized_pk = processor.normalize_pharmacokinetic_data(combined_pk)
        metadata["pharmacokinetics"] = normalized_pk
        # Keep pharmacokinetic_values for backward compatibility but normalize it too
        metadata["pharmacokinetic_values"] = normalized_pk

        # Normalize study types
        study_type_data = {
            "study_type": metadata.get("study_type"),
            "study_types": metadata.get("study_types")
        }
        normalized_study_types = processor.normalize_study_types(study_type_data)
        metadata["study_types"] = normalized_study_types
        # Set study_type to the first normalized type for backward compatibility
        if normalized_study_types:
            metadata["study_type"] = normalized_study_types[0]

        # Normalize publication year
        year_data = metadata.get("publication_year") or metadata.get("year")
        normalized_year = processor.normalize_publication_year(year_data)
        if normalized_year is not None:
            metadata["publication_year"] = normalized_year
            # Keep year for backward compatibility
            metadata["year"] = normalized_year

        # Normalize species
        species_data = metadata.get("species")
        normalized_species = processor.normalize_species(species_data)
        metadata["species"] = normalized_species[0] if normalized_species else None
        # Also store as list for consistent access
        metadata["species_list"] = normalized_species

        # Normalize ranking score
        if "ranking_score" in metadata:
            try:
                metadata["ranking_score"] = float(metadata["ranking_score"])
            except (TypeError, ValueError):
                metadata.pop("ranking_score", None)

        return metadata

    def _iter_documents(self, *, enable_similarity_fallback: bool = False) -> List[Document]:
        """
        Iterate through documents in the vectorstore.

        Args:
            enable_similarity_fallback: If True, use similarity_search as fallback
                when docstore enumeration fails. This may trigger embeddings calls.
        """
        if not self.vectorstore:
            return []
        store = getattr(self.vectorstore, "docstore", None)
        if not store:
            return []
        raw_dict = getattr(store, "_dict", None)
        doc_count = len(self._docstore_ids)
        if isinstance(raw_dict, dict):
            doc_count = max(doc_count, len(raw_dict))

        # Try direct docstore access first
        if isinstance(raw_dict, dict):
            docs = [doc for doc in raw_dict.values() if doc is not None]
            if docs:
                return self._deduplicate_documents(docs)

        # Only use similarity_search fallback if explicitly enabled and other methods fail
        if enable_similarity_fallback:
            similarity_search = getattr(self.vectorstore, "similarity_search", None)
            if callable(similarity_search):
                try:
                    # Reduce k to minimize embedding calls
                    k = min(max(doc_count, 1), 1000)  # Cap at 1000 to prevent excessive embeddings
                    results = similarity_search("*", k=k)
                except Exception as exc:
                    logger.debug("Vectorstore similarity_search iteration failed: %s", exc)
                else:
                    if results:
                        return self._deduplicate_documents(results)

        if not self._docstore_ids:
            self._capture_doc_ids()

        documents: List[Document] = []
        search_fn = getattr(store, "search", None)
        if callable(search_fn):
            lookup_ids = self._docstore_ids or [""]
            for doc_id in lookup_ids:
                try:
                    result = search_fn(doc_id)
                except Exception as exc:
                    logger.debug("Docstore search failed for id %s: %s", doc_id, exc)
                    continue
                self._append_from_docstore_result(result, documents)
            if documents:
                return self._deduplicate_documents(documents)

        get_fn = getattr(store, "get", None)
        if callable(get_fn) and self._docstore_ids:
            for doc_id in self._docstore_ids:
                try:
                    result = get_fn(doc_id)
                except Exception as exc:
                    logger.debug("Docstore get failed for id %s: %s", doc_id, exc)
                    continue
                self._append_from_docstore_result(result, documents)
            if documents:
                return self._deduplicate_documents(documents)

        logger.warning(
            "Unable to iterate documents from vectorstore; consider rebuilding or re-saving the index."
        )
        return []

    def _capture_doc_ids(self) -> None:
        if not self.vectorstore:
            return
        store = getattr(self.vectorstore, "docstore", None)
        if not store:
            return

        collected: List[Any] = []
        raw_dict = getattr(store, "_dict", None)
        if isinstance(raw_dict, dict):
            collected.extend(raw_dict.keys())

        index_mapping = getattr(self.vectorstore, "index_to_docstore_id", None)
        if isinstance(index_mapping, dict):
            collected.extend(index_mapping.values())

        if collected:
            self._docstore_ids = list(dict.fromkeys(collected))

    def _record_doc_ids(self, doc_ids: Optional[Iterable[Any]]) -> None:
        if not doc_ids:
            return
        updated = list(self._docstore_ids)
        updated.extend(doc_ids)
        self._docstore_ids = list(dict.fromkeys(updated))

    def _append_from_docstore_result(self, result: Any, accumulator: List[Document]) -> None:
        if not result:
            return
        if isinstance(result, Document):
            accumulator.append(result)
            return
        if isinstance(result, list):
            for item in result:
                if isinstance(item, Document):
                    accumulator.append(item)
            return
        if isinstance(result, dict):
            for item in result.values():
                if isinstance(item, Document):
                    accumulator.append(item)

    def _deduplicate_documents(self, documents: Iterable[Document]) -> List[Document]:
        seen = set()
        deduped: List[Document] = []
        for doc in documents:
            if doc is None:
                continue

            metadata = dict(getattr(doc, "metadata", {}) or {})
            doc_id = (
                metadata.get("doc_id")
                or metadata.get("document_id")
                or metadata.get("id")
                or metadata.get("pmid")
                or metadata.get("pmcid")
            )
            if doc_id:
                key = ("doc_id", str(doc_id).strip().lower())
            else:
                source = metadata.get("source")
                page = metadata.get("page")
                if page is None:
                    page = metadata.get("page_number")
                if page is None:
                    page = metadata.get("page_index")
                if source:
                    page_fragment = str(page).strip().lower() if page is not None else "__none__"
                    key = (
                        "source_page",
                        str(source).strip().lower(),
                        page_fragment,
                    )
                else:
                    content = doc.page_content or ""
                    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
                    key = ("content", digest)

            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)

        return deduped

    def _detect_pharma_metadata(self) -> bool:
        for doc in self._iter_documents():
            metadata = doc.metadata or {}
            if any(
                key in metadata
                for key in (
                    "drug_names",
                    "therapeutic_areas",
                    "cyp_enzymes",
                    "pharmacokinetics",
                    "pharmaceutical_enriched",
                )
            ):
                return True
        return False
    
    def delete_index(self) -> bool:
        """
        Delete the vector index from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.index_path.exists():
                self.index_path.unlink()
                logger.info("Vector index deleted from disk")
            
            if self.docstore_path.exists():
                self.docstore_path.unlink()
                logger.info("Docstore file deleted from disk")
            
            self.vectorstore = None
            self._pharma_metadata_enabled = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete index: {str(e)}")
            return False


def main():
    """Test the vector database"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize components
    embeddings = NVIDIAEmbeddings()
    vector_db = VectorDatabase(embeddings)
    
    # Test documents
    test_docs = [
        Document(
            page_content="Artificial intelligence is the simulation of human intelligence in machines.",
            metadata={"source": "test1.pdf", "page": 1}
        ),
        Document(
            page_content="Machine learning is a subset of AI that enables computers to learn without explicit programming.",
            metadata={"source": "test2.pdf", "page": 1}
        )
    ]
    
    # Create and save index
    if vector_db.create_index(test_docs):
        vector_db.save_index()
        
        # Test search
        results = vector_db.similarity_search("What is AI?", k=2)
        print(f"Search results: {len(results)}")
        for i, doc in enumerate(results):
            print(f"  {i+1}. {doc.page_content[:100]}...")
        
        # Get stats
        stats = vector_db.get_stats()
        print(f"Database stats: {stats}")


if __name__ == "__main__":
    main()
