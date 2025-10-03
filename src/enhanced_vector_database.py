"""
Enhanced Vector Database with FAISS and cuVS Support

Hybrid vector database that supports both legacy FAISS (CPU) and new cuVS (GPU)
backends for optimal performance across different deployment scenarios.

Features:
1. Automatic backend selection (GPU cuVS preferred, CPU FAISS fallback)
2. Seamless migration between FAISS and cuVS indices
3. Pharmaceutical metadata preservation and enhancement
4. Performance monitoring and optimization
5. Backward compatibility with existing FAISS indices
6. Advanced filtering and search capabilities

<<use_mcp microsoft-learn>>
"""
import logging
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document

# FAISS import (always available as fallback)
try:
    import faiss
    from langchain_community.vectorstores import FAISS

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS = None
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available")

# cuVS imports (GPU acceleration)
try:
    import cupy as cp
    import cuvs

    CUVS_AVAILABLE = True
except ImportError:
    cp = None
    cuvs = None
    CUVS_AVAILABLE = False
    logging.info("cuVS not available, using CPU-only mode")

from .embedding_performance_monitor import monitor_embedding_request

# Import existing components
from .nvidia_embeddings_v2 import NVIDIAEmbeddings
from .pharmaceutical_processor import PharmaceuticalProcessor

logger = logging.getLogger(__name__)


class VectorBackend(Enum):
    """Available vector database backends."""

    FAISS_CPU = "faiss_cpu"
    CUVS_GPU = "cuvs_gpu"
    AUTO = "auto"


class IndexType(Enum):
    """Vector index types."""

    FLAT = "flat"  # Exact search
    IVF = "ivf"  # Inverted file index
    HNSW = "hnsw"  # Hierarchical navigable small world


@dataclass
class VectorDatabaseConfig:
    """Configuration for enhanced vector database."""

    backend: VectorBackend = VectorBackend.AUTO
    index_type: IndexType = IndexType.FLAT
    dimension: Optional[int] = None
    enable_gpu_acceleration: bool = True
    enable_pharmaceutical_enhancement: bool = True
    enable_performance_monitoring: bool = True
    cache_size_mb: int = 512
    index_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from vector search."""

    documents: List[Document]
    scores: List[float]
    search_time_ms: float
    backend_used: str
    pharmaceutical_enhanced: bool


class EnhancedVectorDatabase:
    """
    Enhanced vector database with hybrid FAISS/cuVS support.

    Provides intelligent backend selection, seamless migration between
    CPU and GPU implementations, and pharmaceutical domain optimizations.
    """

    def __init__(
        self,
        embeddings: NVIDIAEmbeddings,
        config: Optional[VectorDatabaseConfig] = None,
        db_path: str = "./enhanced_vector_db",
        index_name: str = "hybrid_index",
        pharmaceutical_processor: Optional[PharmaceuticalProcessor] = None,
    ):
        """
        Initialize enhanced vector database.

        Args:
            embeddings: Embedding service (supports both legacy and v2)
            config: Vector database configuration
            db_path: Path to store database files
            index_name: Name of the index
            pharmaceutical_processor: Optional pharmaceutical processor
        """
        self.embeddings = embeddings
        self.config = config or VectorDatabaseConfig()
        self.db_path = Path(db_path)
        self.index_name = index_name
        self.pharmaceutical_processor = pharmaceutical_processor

        # Backend instances
        self.faiss_vectorstore: Optional[FAISS] = None
        self.cuvs_index: Optional[Any] = None
        self.current_backend: Optional[VectorBackend] = None

        # Index metadata
        self.documents: List[Document] = []
        self.embeddings_array: Optional[np.ndarray] = None
        self.dimension: Optional[int] = None

        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "faiss_searches": 0,
            "cuvs_searches": 0,
            "avg_search_time_ms": 0.0,
            "pharmaceutical_enhancements": 0,
        }

        # Create database directory
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Determine optimal backend
        self._initialize_backend()

        logger.info(f"Initialized Enhanced Vector Database at: {self.db_path}")
        logger.info(f"Backend: {self.current_backend.value if self.current_backend else 'not_selected'}")

    def _initialize_backend(self):
        """Initialize the optimal vector backend."""
        if self.config.backend == VectorBackend.AUTO:
            # Auto-select based on availability and capabilities
            if CUVS_AVAILABLE and self.config.enable_gpu_acceleration and self._check_gpu_availability():
                self.current_backend = VectorBackend.CUVS_GPU
                logger.info("Auto-selected cuVS GPU backend")
            elif FAISS_AVAILABLE:
                self.current_backend = VectorBackend.FAISS_CPU
                logger.info("Auto-selected FAISS CPU backend")
            else:
                raise RuntimeError("No vector backend available (neither FAISS nor cuVS)")
        else:
            # Use specified backend
            if self.config.backend == VectorBackend.CUVS_GPU and not CUVS_AVAILABLE:
                logger.warning("cuVS requested but not available, falling back to FAISS")
                self.current_backend = VectorBackend.FAISS_CPU
            elif self.config.backend == VectorBackend.FAISS_CPU and not FAISS_AVAILABLE:
                raise RuntimeError("FAISS requested but not available")
            else:
                self.current_backend = self.config.backend

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for cuVS."""
        try:
            if cp is None:
                return False

            # Try to create a simple cupy array
            test_array = cp.array([1, 2, 3])
            del test_array
            return True
        except Exception as e:
            logger.debug(f"GPU not available: {e}")
            return False

    def create_index(self, documents: List[Document], extract_pharmaceutical_metadata: bool = None) -> bool:
        """
        Create vector index from documents using optimal backend.

        Args:
            documents: List of documents to index
            extract_pharmaceutical_metadata: Whether to extract pharmaceutical metadata

        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return False

        # Use config default if not specified
        if extract_pharmaceutical_metadata is None:
            extract_pharmaceutical_metadata = self.config.enable_pharmaceutical_enhancement

        start_time = time.time()

        try:
            logger.info(f"Creating vector index from {len(documents)} documents using {self.current_backend.value}...")

            # Process documents with pharmaceutical enhancement if enabled
            if extract_pharmaceutical_metadata:
                processed_docs = self._enhance_pharmaceutical_documents(documents)
                self.search_stats["pharmaceutical_enhancements"] += len(documents)
            else:
                processed_docs = documents

            # Extract texts and embeddings
            texts = [doc.page_content for doc in processed_docs]

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Store for later use
            self.documents = processed_docs
            self.embeddings_array = embeddings_array
            self.dimension = embeddings_array.shape[1]

            # Create index using appropriate backend
            success = self._create_backend_index(embeddings_array, processed_docs)

            if success:
                processing_time = (time.time() - start_time) * 1000
                logger.info(f"✅ Vector index created successfully in {processing_time:.2f}ms!")

                # Record performance metrics
                if self.config.enable_performance_monitoring:
                    monitor_embedding_request(
                        service_name=f"enhanced_vector_db_{self.current_backend.value}",
                        response_time_ms=processing_time,
                        success=True,
                        is_pharmaceutical=extract_pharmaceutical_metadata,
                    )

            return success

        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            return False

    def _create_backend_index(self, embeddings_array: np.ndarray, documents: List[Document]) -> bool:
        """Create index using the current backend."""
        if self.current_backend == VectorBackend.FAISS_CPU:
            return self._create_faiss_index(embeddings_array, documents)
        elif self.current_backend == VectorBackend.CUVS_GPU:
            return self._create_cuvs_index(embeddings_array, documents)
        else:
            raise ValueError(f"Unsupported backend: {self.current_backend}")

    def _create_faiss_index(self, embeddings_array: np.ndarray, documents: List[Document]) -> bool:
        """Create FAISS index."""
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Create FAISS vectorstore
            self.faiss_vectorstore = FAISS.from_texts(texts=texts, embedding=self.embeddings, metadatas=metadatas)

            logger.info("FAISS index created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            return False

    def _create_cuvs_index(self, embeddings_array: np.ndarray, documents: List[Document]) -> bool:
        """Create cuVS index."""
        try:
            # Convert to GPU arrays
            gpu_embeddings = cp.asarray(embeddings_array)

            # Create cuVS index based on type
            if self.config.index_type == IndexType.FLAT:
                # Use brute force for exact search
                self.cuvs_index = {"embeddings": gpu_embeddings, "documents": documents, "type": "flat"}
            elif self.config.index_type == IndexType.IVF:
                # Create IVF index for faster approximate search
                nlist = min(int(np.sqrt(len(embeddings_array))), 1024)
                index_params = cuvs.IvfFlatSearchParams(nlist=nlist)

                # Build index
                index = cuvs.IvfFlat.build(params=index_params, dataset=gpu_embeddings)

                self.cuvs_index = {"index": index, "documents": documents, "type": "ivf"}
            else:
                logger.warning(f"Index type {self.config.index_type} not supported for cuVS, using flat")
                return self._create_cuvs_index(embeddings_array, documents)

            logger.info("cuVS GPU index created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create cuVS index: {e}")
            # Fallback to FAISS
            logger.info("Falling back to FAISS CPU backend")
            self.current_backend = VectorBackend.FAISS_CPU
            return self._create_faiss_index(embeddings_array, documents)

    def _enhance_pharmaceutical_documents(self, documents: List[Document]) -> List[Document]:
        """Enhance documents with pharmaceutical metadata."""
        if not self.pharmaceutical_processor:
            self.pharmaceutical_processor = PharmaceuticalProcessor()

        enhanced_docs = []
        for doc in documents:
            try:
                # Extract pharmaceutical metadata
                pharma_metadata = self.pharmaceutical_processor.extract_metadata(doc.page_content)

                # Create enhanced document
                enhanced_metadata = {**doc.metadata, **pharma_metadata}
                enhanced_doc = Document(page_content=doc.page_content, metadata=enhanced_metadata)
                enhanced_docs.append(enhanced_doc)

            except Exception as e:
                logger.warning(f"Failed to enhance document with pharmaceutical metadata: {e}")
                enhanced_docs.append(doc)

        return enhanced_docs

    def search(
        self, query: str, k: int = 5, filter_criteria: Optional[Dict[str, Any]] = None, search_type: str = "similarity"
    ) -> SearchResult:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return
            filter_criteria: Optional filtering criteria
            search_type: Type of search ("similarity", "mmr")

        Returns:
            Search results with performance metrics
        """
        start_time = time.time()
        self.search_stats["total_searches"] += 1

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Perform search using current backend
            if self.current_backend == VectorBackend.FAISS_CPU:
                documents, scores = self._search_faiss(query_embedding, k, filter_criteria, search_type)
                self.search_stats["faiss_searches"] += 1
            elif self.current_backend == VectorBackend.CUVS_GPU:
                documents, scores = self._search_cuvs(query_embedding, k, filter_criteria)
                self.search_stats["cuvs_searches"] += 1
            else:
                raise ValueError(f"Unsupported backend: {self.current_backend}")

            search_time_ms = (time.time() - start_time) * 1000

            # Update search statistics
            self.search_stats["avg_search_time_ms"] = (
                self.search_stats["avg_search_time_ms"] * (self.search_stats["total_searches"] - 1) + search_time_ms
            ) / self.search_stats["total_searches"]

            # Record performance metrics
            if self.config.enable_performance_monitoring:
                monitor_embedding_request(
                    service_name=f"enhanced_vector_db_{self.current_backend.value}",
                    response_time_ms=search_time_ms,
                    success=True,
                    is_pharmaceutical=self.config.enable_pharmaceutical_enhancement,
                )

            return SearchResult(
                documents=documents,
                scores=scores,
                search_time_ms=search_time_ms,
                backend_used=self.current_backend.value,
                pharmaceutical_enhanced=self.config.enable_pharmaceutical_enhancement,
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            search_time_ms = (time.time() - start_time) * 1000

            # Record failed search
            if self.config.enable_performance_monitoring:
                monitor_embedding_request(
                    service_name=f"enhanced_vector_db_{self.current_backend.value}",
                    response_time_ms=search_time_ms,
                    success=False,
                    is_pharmaceutical=self.config.enable_pharmaceutical_enhancement,
                )

            # Return empty result
            return SearchResult(
                documents=[],
                scores=[],
                search_time_ms=search_time_ms,
                backend_used=self.current_backend.value,
                pharmaceutical_enhanced=False,
            )

    def _search_faiss(
        self, query_embedding: List[float], k: int, filter_criteria: Optional[Dict[str, Any]], search_type: str
    ) -> Tuple[List[Document], List[float]]:
        """Search using FAISS backend."""
        if not self.faiss_vectorstore:
            raise RuntimeError("FAISS vectorstore not initialized")

        if filter_criteria:
            # Apply metadata filtering
            def filter_func(metadata: Dict[str, Any]) -> bool:
                for key, value in filter_criteria.items():
                    if key not in metadata:
                        return False
                    if isinstance(value, list):
                        if metadata[key] not in value:
                            return False
                    else:
                        if metadata[key] != value:
                            return False
                return True

            # Search with filtering
            docs_and_scores = self.faiss_vectorstore.similarity_search_with_score(
                query=query_embedding, k=k * 3, filter=filter_func  # Oversample to account for filtering
            )

            # Extract results
            documents = [doc for doc, score in docs_and_scores[:k]]
            scores = [score for doc, score in docs_and_scores[:k]]
        else:
            # Standard similarity search
            if search_type == "mmr":
                documents = self.faiss_vectorstore.max_marginal_relevance_search(query=query_embedding, k=k)
                scores = [0.0] * len(documents)  # MMR doesn't return scores
            else:
                docs_and_scores = self.faiss_vectorstore.similarity_search_with_score(query=query_embedding, k=k)
                documents = [doc for doc, score in docs_and_scores]
                scores = [score for doc, score in docs_and_scores]

        return documents, scores

    def _search_cuvs(
        self, query_embedding: List[float], k: int, filter_criteria: Optional[Dict[str, Any]]
    ) -> Tuple[List[Document], List[float]]:
        """Search using cuVS backend."""
        if not self.cuvs_index:
            raise RuntimeError("cuVS index not initialized")

        # Convert query to GPU array
        query_gpu = cp.asarray([query_embedding], dtype=cp.float32)

        if self.cuvs_index["type"] == "flat":
            # Brute force search
            embeddings_gpu = self.cuvs_index["embeddings"]

            # Compute similarities (cosine similarity)
            similarities = cp.dot(query_gpu, embeddings_gpu.T) / (
                cp.linalg.norm(query_gpu, axis=1, keepdims=True) * cp.linalg.norm(embeddings_gpu, axis=1)
            )

            # Get top k indices
            top_indices = cp.argsort(similarities[0])[::-1][:k]
            top_scores = similarities[0][top_indices]

            # Convert back to CPU
            indices_cpu = cp.asnumpy(top_indices)
            scores_cpu = cp.asnumpy(top_scores)

        elif self.cuvs_index["type"] == "ivf":
            # Use IVF index for search
            search_params = cuvs.IvfFlatSearchParams(nprobe=min(32, self.cuvs_index["index"].nlist))

            # Perform search
            scores, indices = cuvs.search(index=self.cuvs_index["index"], params=search_params, queries=query_gpu, k=k)

            # Convert to CPU
            indices_cpu = cp.asnumpy(indices[0])
            scores_cpu = cp.asnumpy(scores[0])

        else:
            raise ValueError(f"Unknown cuVS index type: {self.cuvs_index['type']}")

        # Retrieve documents
        documents = []
        final_scores = []

        for i, idx in enumerate(indices_cpu):
            if idx < len(self.cuvs_index["documents"]):
                doc = self.cuvs_index["documents"][idx]

                # Apply filtering if specified
                if filter_criteria:
                    if self._apply_filter(doc.metadata, filter_criteria):
                        documents.append(doc)
                        final_scores.append(float(scores_cpu[i]))
                else:
                    documents.append(doc)
                    final_scores.append(float(scores_cpu[i]))

        return documents[:k], final_scores[:k]

    def _apply_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Apply filtering criteria to document metadata."""
        for key, value in filter_criteria.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True

    def save_index(self) -> bool:
        """Save the vector index to disk."""
        try:
            logger.info(f"Saving {self.current_backend.value} index to disk...")

            # Save backend-specific index
            if self.current_backend == VectorBackend.FAISS_CPU and self.faiss_vectorstore:
                self.faiss_vectorstore.save_local(str(self.db_path), self.index_name)
            elif self.current_backend == VectorBackend.CUVS_GPU and self.cuvs_index:
                # Save cuVS index data
                index_data = {
                    "type": self.cuvs_index["type"],
                    "documents": self.cuvs_index["documents"],
                    "embeddings_cpu": cp.asnumpy(self.cuvs_index["embeddings"])
                    if "embeddings" in self.cuvs_index
                    else None,
                }

                with open(self.db_path / f"{self.index_name}_cuvs.pkl", "wb") as f:
                    pickle.dump(index_data, f)

            # Save metadata
            metadata = {
                "backend": self.current_backend.value,
                "config": self.config,
                "dimension": self.dimension,
                "num_documents": len(self.documents),
                "search_stats": self.search_stats,
            }

            with open(self.db_path / f"{self.index_name}_metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"✅ Vector index saved to: {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
            return False

    def load_index(self) -> bool:
        """Load vector index from disk."""
        try:
            logger.info("Loading vector index from disk...")

            # Load metadata
            metadata_path = self.db_path / f"{self.index_name}_metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)

                self.search_stats = metadata.get("search_stats", self.search_stats)
                self.dimension = metadata.get("dimension")
                saved_backend = VectorBackend(metadata["backend"])
            else:
                # Try to detect legacy FAISS index
                faiss_path = self.db_path / f"{self.index_name}.faiss"
                if faiss_path.exists():
                    saved_backend = VectorBackend.FAISS_CPU
                else:
                    logger.error("No saved index found")
                    return False

            # Load backend-specific index
            if saved_backend == VectorBackend.FAISS_CPU:
                if FAISS_AVAILABLE:
                    self.faiss_vectorstore = FAISS.load_local(
                        str(self.db_path), self.embeddings, self.index_name, allow_dangerous_deserialization=True
                    )
                    self.current_backend = VectorBackend.FAISS_CPU
                else:
                    logger.error("FAISS not available but required for saved index")
                    return False

            elif saved_backend == VectorBackend.CUVS_GPU:
                cuvs_path = self.db_path / f"{self.index_name}_cuvs.pkl"
                if cuvs_path.exists() and CUVS_AVAILABLE:
                    with open(cuvs_path, "rb") as f:
                        index_data = pickle.load(f)

                    # Reconstruct cuVS index
                    if index_data["embeddings_cpu"] is not None:
                        gpu_embeddings = cp.asarray(index_data["embeddings_cpu"])
                        self.cuvs_index = {
                            "embeddings": gpu_embeddings,
                            "documents": index_data["documents"],
                            "type": index_data["type"],
                        }
                        self.current_backend = VectorBackend.CUVS_GPU
                    else:
                        logger.error("Corrupted cuVS index data")
                        return False
                else:
                    logger.warning("cuVS index not available, falling back to FAISS")
                    return self.load_index()  # Retry with FAISS

            logger.info(f"✅ Vector index loaded successfully using {self.current_backend.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return False

    def migrate_to_backend(self, target_backend: VectorBackend) -> bool:
        """
        Migrate index to a different backend.

        Args:
            target_backend: Target backend to migrate to

        Returns:
            True if migration successful
        """
        if target_backend == self.current_backend:
            logger.info("Already using target backend")
            return True

        logger.info(f"Migrating from {self.current_backend.value} to {target_backend.value}")

        try:
            # Ensure we have documents and embeddings
            if not self.documents or self.embeddings_array is None:
                logger.error("No data available for migration")
                return False

            # Store current state
            old_backend = self.current_backend
            old_faiss = self.faiss_vectorstore
            old_cuvs = self.cuvs_index

            # Switch to target backend
            self.current_backend = target_backend

            # Create new index
            success = self._create_backend_index(self.embeddings_array, self.documents)

            if success:
                logger.info(f"✅ Successfully migrated to {target_backend.value}")
                # Clean up old backend resources if needed
                if old_backend == VectorBackend.CUVS_GPU and old_cuvs:
                    # Clean up GPU memory
                    pass
                return True
            else:
                # Restore previous state
                self.current_backend = old_backend
                self.faiss_vectorstore = old_faiss
                self.cuvs_index = old_cuvs
                logger.error(f"Migration to {target_backend.value} failed, restored previous state")
                return False

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        return {
            "backend": self.current_backend.value if self.current_backend else "none",
            "num_documents": len(self.documents),
            "dimension": self.dimension,
            "search_stats": self.search_stats,
            "config": {
                "backend": self.config.backend.value,
                "index_type": self.config.index_type.value,
                "enable_gpu_acceleration": self.config.enable_gpu_acceleration,
                "enable_pharmaceutical_enhancement": self.config.enable_pharmaceutical_enhancement,
            },
            "capabilities": {
                "faiss_available": FAISS_AVAILABLE,
                "cuvs_available": CUVS_AVAILABLE,
                "gpu_available": self._check_gpu_availability(),
            },
        }

    def optimize_performance(self) -> Dict[str, Any]:
        """
        Analyze and optimize database performance.

        Returns:
            Performance optimization report
        """
        report = {"current_performance": self.search_stats.copy(), "recommendations": [], "optimizations_applied": []}

        # Analyze search performance
        if self.search_stats["total_searches"] > 0:
            avg_search_time = self.search_stats["avg_search_time_ms"]

            # GPU acceleration recommendation
            if (
                self.current_backend == VectorBackend.FAISS_CPU
                and CUVS_AVAILABLE
                and self._check_gpu_availability()
                and avg_search_time > 100
            ):
                report["recommendations"].append(
                    {
                        "type": "backend_migration",
                        "recommendation": "Migrate to cuVS GPU backend for better performance",
                        "expected_improvement": "50-80% faster search times",
                        "action": "call migrate_to_backend(VectorBackend.CUVS_GPU)",
                    }
                )

            # Index optimization recommendation
            if len(self.documents) > 10000 and self.config.index_type == IndexType.FLAT:
                report["recommendations"].append(
                    {
                        "type": "index_optimization",
                        "recommendation": "Switch to IVF index for large document collections",
                        "expected_improvement": "Faster search with minimal accuracy loss",
                        "action": "recreate index with IndexType.IVF",
                    }
                )

        return report


# Factory functions for easy integration


def create_enhanced_vector_database(
    embeddings: NVIDIAEmbeddings, enable_gpu: bool = True, enable_pharmaceutical: bool = True, **kwargs
) -> EnhancedVectorDatabase:
    """
    Create enhanced vector database with optimal configuration.

    Args:
        embeddings: Embedding service
        enable_gpu: Enable GPU acceleration if available
        enable_pharmaceutical: Enable pharmaceutical enhancements
        **kwargs: Additional configuration parameters

    Returns:
        Configured EnhancedVectorDatabase
    """
    config = VectorDatabaseConfig(
        backend=VectorBackend.AUTO,
        enable_gpu_acceleration=enable_gpu,
        enable_pharmaceutical_enhancement=enable_pharmaceutical,
        **kwargs,
    )

    return EnhancedVectorDatabase(embeddings=embeddings, config=config)


def migrate_legacy_faiss_database(
    legacy_db_path: str, embeddings: NVIDIAEmbeddings, target_backend: VectorBackend = VectorBackend.AUTO
) -> EnhancedVectorDatabase:
    """
    Migrate legacy FAISS database to enhanced vector database.

    Args:
        legacy_db_path: Path to legacy FAISS database
        embeddings: Embedding service
        target_backend: Target backend for migration

    Returns:
        Enhanced vector database with migrated data
    """
    # Create enhanced database
    enhanced_db = create_enhanced_vector_database(embeddings)

    # Set path to legacy database
    enhanced_db.db_path = Path(legacy_db_path)

    # Load legacy index
    if enhanced_db.load_index():
        logger.info("Successfully loaded legacy FAISS database")

        # Migrate to target backend if different
        if target_backend != VectorBackend.AUTO and target_backend != enhanced_db.current_backend:
            enhanced_db.migrate_to_backend(target_backend)

        return enhanced_db
    else:
        raise RuntimeError(f"Failed to load legacy database from {legacy_db_path}")
