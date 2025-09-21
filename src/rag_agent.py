"""
RAG Agent - Main pipeline for Retrieval-Augmented Generation
Combines document retrieval with question answering using NVIDIA models
"""

import os
import json
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM

from .document_loader import PDFDocumentLoader
from .nvidia_embeddings import NVIDIAEmbeddings
from .vector_database import VectorDatabase

# Set up logging
logger = logging.getLogger(__name__)

# Import unified disclaimer constants
try:
    from guardrails.constants import MEDICAL_DISCLAIMER
except ImportError:
    # Fallback disclaimer if guardrails not available
    MEDICAL_DISCLAIMER = (
        "Medical Disclaimer: This information is for research and educational purposes only and is not intended as "
        "medical advice, diagnosis, or treatment. It is not a substitute for professional medical consultation. "
        "Always consult qualified healthcare professionals or licensed clinicians for any medical concerns or decisions. "
        "This system does not handle medical emergencies - seek immediate medical attention for urgent conditions. "
        "The information provided may contain inaccuracies and should be verified with authoritative medical sources."
    )


@dataclass
class RAGResponse:
    """Response from RAG agent"""
    answer: str
    source_documents: List[Document]
    confidence_scores: Optional[List[float]] = None
    query: str = ""
    processing_time: float = 0.0
    disclaimer: Optional[str] = None
    needs_rebuild: bool = False


class SimpleNVIDIALLM:
    """Simple NVIDIA LLM wrapper for basic text generation"""

    def __init__(self, api_key: str, model_name: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
        self.base_url = "https://integrate.api.nvidia.com/v1"

    def generate_response(self, prompt: str) -> str:
        """Generate a response using NVIDIA API"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.1
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return "I apologize, but I'm unable to generate a response at the moment."

        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."


class RAGAgent:
    """Main RAG Agent that combines retrieval and generation

    UIs should prefer get_disclaimer() or RAGResponse.disclaimer for display.
    """
    
    def __init__(
        self,
        docs_folder: str,
        api_key: str,
        vector_db_path: str = "./vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model_name: Optional[str] = None,
        enable_preflight_embedding: Optional[bool] = None,
        # WARNING: To avoid duplicate disclaimer rendering, UIs that render RAGResponse.disclaimer
        # MUST set append_disclaimer_in_answer=False. Default is True for backward compatibility.
        append_disclaimer_in_answer: Optional[bool] = None,
    ):
        """
        Initialize RAG Agent

        Args:
            docs_folder: Path to PDF documents folder
            api_key: NVIDIA API key
            vector_db_path: Path for vector database storage
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            embedding_model_name: Optional embedding model name
            enable_preflight_embedding: Forces question-time preflight embedding when True;
                defaults to False unless DISABLE_PREFLIGHT_EMBEDDING is explicitly set false in the environment.
            append_disclaimer_in_answer: Controls whether MEDICAL_DISCLAIMER is appended to returned answer text.
                When None, reads APPEND_DISCLAIMER_IN_ANSWER environment variable (defaults to True so answers include
                the disclaimer by default). UIs should avoid duplicating the disclaimer by either rendering only
                RAGResponse.disclaimer or setting APPEND_DISCLAIMER_IN_ANSWER=false when they present the separate field.

        Configuration Notes:
            - VECTOR_DB_PER_MODEL (default: "false") stores the FAISS index in a sanitized
              model-specific subdirectory under vector_db_path (e.g.,
              ./vector_db/nvidia_llama-3.2-nemoretriever-1b-vlm-embed-v1). Compatible legacy
              indexes in VECTOR_DB_PATH are migrated on first run; mismatched models or
              embedding dimensions trigger a rebuild at the destination path.
        """
        self.docs_folder = docs_folder
        self.api_key = api_key

        # Initialize components
        self.document_loader = PDFDocumentLoader(docs_folder, chunk_size, chunk_overlap)

        # Store base path and determine per-model behavior early so we can
        # finalize the embedding model before deriving a path.
        self.base_vector_db_path = vector_db_path
        self.use_per_model_path = os.getenv("VECTOR_DB_PER_MODEL", "false").lower() in ("true", "1", "yes", "on")
        disable_preflight_env = os.getenv("DISABLE_PREFLIGHT_EMBEDDING")
        enable_preflight_env = os.getenv("ENABLE_PREFLIGHT_EMBEDDING")
        env_truthy = ("true", "1", "yes", "on")
        env_falsey = ("false", "0", "no", "off")
        if enable_preflight_embedding is not None:
            self.enable_preflight_embedding = enable_preflight_embedding
        else:
            if enable_preflight_env is not None:
                normalized_enable = enable_preflight_env.strip().lower()
                self.enable_preflight_embedding = normalized_enable in env_truthy
            elif disable_preflight_env is not None:
                normalized_disable = disable_preflight_env.strip().lower()
                self.enable_preflight_embedding = normalized_disable not in env_truthy
            else:
                # default to disabled for latency/cost reasons
                self.enable_preflight_embedding = False

        # Set disclaimer append behavior from parameter or environment
        if append_disclaimer_in_answer is not None:
            self.append_disclaimer_in_answer = append_disclaimer_in_answer
        else:
            env_append_disclaimer = os.getenv("APPEND_DISCLAIMER_IN_ANSWER")
            if env_append_disclaimer is not None:
                self.append_disclaimer_in_answer = env_append_disclaimer.strip().lower() in ("true", "1", "yes", "on")
            else:
                self.append_disclaimer_in_answer = True  # Default to True to ensure answers carry the disclaimer
        self._guardrail_metadata: Dict[str, Any] = {}

        # Set force preflight on first query behavior from environment
        env_force_preflight_first = os.getenv("FORCE_PREFLIGHT_ON_FIRST_QUERY")
        if env_force_preflight_first is not None:
            self.force_preflight_on_first_query = env_force_preflight_first.strip().lower() in ("true", "1", "yes", "on")
        else:
            self.force_preflight_on_first_query = True  # Default to True to maintain existing behavior

        falsey_values = ("false", "0", "no", "off")
        env_probe_setting = os.getenv("EMBEDDING_PROBE_ON_INIT")
        # Probing reduces runtime fallbacks but adds a small startup latency.
        if env_probe_setting is None:
            probe_on_init_flag = True
        else:
            normalized_probe_setting = env_probe_setting.strip().lower()
            probe_on_init_flag = normalized_probe_setting not in falsey_values
            logger.debug(
                "EMBEDDING_PROBE_ON_INIT override detected; probe_on_init=%s",
                probe_on_init_flag,
            )

        self.embeddings = NVIDIAEmbeddings(
            api_key=self.api_key,
            embedding_model_name=embedding_model_name or os.getenv("EMBEDDING_MODEL_NAME"),
            probe_on_init=probe_on_init_flag,
        )

        # Track the model associated with the current vector DB path
        self._active_vector_db_model = self.embeddings.model_name

        if self.use_per_model_path:
            # Use per-model index directories with sanitization to avoid
            # compatibility issues when switching embedding models.
            sanitized_model_name = self._sanitize_model_name(self.embeddings.model_name)
            self.vector_db_path = os.path.join(self.base_vector_db_path, sanitized_model_name)
        else:
            # Use the provided vector_db_path unchanged for backward compatibility
            self.vector_db_path = self.base_vector_db_path

        self.vector_db = VectorDatabase(self.embeddings, self.vector_db_path)
        self._did_one_time_preflight = False
        self.llm = SimpleNVIDIALLM(api_key)

        # Custom prompt template
        self.prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Use research-based information only; do not provide medical advice; state when information is insufficient.

Context:
{context}

Question: {question}

Answer: """
        
        logger.info(f"RAG Agent initialized successfully with model: {self.embeddings.model_name}")
        if self.use_per_model_path:
            logger.info(
                "VECTOR_DB_PER_MODEL enabled: maintaining per-model indexes under '%s'."
                " Rebuild the knowledge base when the embedding model falls back or changes to keep the index compatible.",
                self.vector_db_path,
            )
        else:
            logger.info(f"Vector DB path (legacy mode): {self.vector_db_path}")

        if self.enable_preflight_embedding:
            if enable_preflight_embedding is not None:
                logger.info("Question preflight embedding enabled via constructor override.")
            elif disable_preflight_env is not None and disable_preflight_env.strip().lower() in env_falsey:
                logger.info(
                    "Question preflight embedding enabled because DISABLE_PREFLIGHT_EMBEDDING was set false in the environment."
                )
            else:
                logger.info("Question preflight embedding enabled.")
        else:
            if enable_preflight_embedding is False:
                logger.info("Question preflight embedding disabled via constructor override.")
            elif disable_preflight_env is None:
                logger.info(
                    "Question preflight embedding disabled by default to avoid extra embedding latency and cost per query."
                )
            else:
                logger.info(
                    "Question preflight embedding disabled because DISABLE_PREFLIGHT_EMBEDDING was set true in the environment."
                )
            logger.info(
                "Pass enable_preflight_embedding=True or set DISABLE_PREFLIGHT_EMBEDDING=false to re-enable the preflight check."
            )

    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name for use as directory name"""
        return model_name.replace('/', '_').replace(' ', '_').replace(':', '_')

    def _get_embeddings_metadata_path(self) -> Path:
        """Get path for embeddings metadata file"""
        return Path(self.vector_db_path) / "embeddings_meta.json"

    def _load_embeddings_metadata(self) -> Optional[Dict[str, Any]]:
        """Load stored embeddings metadata if present."""
        metadata_path = self._get_embeddings_metadata_path()
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                return json.load(metadata_file)
        except Exception as error:
            logger.warning("Failed to read embeddings metadata from %s: %s", metadata_path, error)
            return None

    def set_guardrail_metadata(self, metadata: Optional[Dict[str, Any]]) -> None:
        """Store guardrail metadata so disclaimer logic can honour prior augmentations."""
        self._guardrail_metadata = dict(metadata or {})

    def _apply_disclaimer(self, text: str, *, disclaimer_already_present: Optional[bool] = None) -> str:
        """Append the medical disclaimer when configured to do so."""
        if self.append_disclaimer_in_answer:
            # Keep callers from double-rendering the disclaimer: they can disable this flag and use
            # RAGResponse.disclaimer instead if their UI presents the disclaimer separately.
            guardrail_flag = False
            if disclaimer_already_present is None:
                guardrail_flag = bool(self._guardrail_metadata.get("disclaimer_added"))
                if guardrail_flag:
                    self._guardrail_metadata["disclaimer_added"] = False
            else:
                guardrail_flag = disclaimer_already_present

            if guardrail_flag:
                return text

            # Use standardized disclaimer detection
            try:
                from guardrails.actions import contains_medical_disclaimer
                if contains_medical_disclaimer(text):
                    return text
            except ImportError:
                # Fallback to original detection
                normalized = text.lower()
                if "medical disclaimer" in normalized or MEDICAL_DISCLAIMER.lower() in normalized:
                    return text
            separator = "\n\n---\n"
            if text.endswith("\n"):
                separator = "\n---\n"
            return f"{text}{separator}{MEDICAL_DISCLAIMER}"
        return text

    def get_answer_and_disclaimer(self, answer_text: str) -> tuple[str, str]:
        """
        Separate answer and disclaimer to prevent duplication in UI rendering.

        Args:
            answer_text: The answer text that may contain a disclaimer

        Returns:
            tuple[str, str]: (clean_answer, disclaimer)
                - clean_answer: Answer text with disclaimer removed
                - disclaimer: Disclaimer text if enabled, empty string otherwise
        """
        if not self.append_disclaimer_in_answer:
            # Disclaimer not enabled, return original answer and empty disclaimer
            return answer_text, ""

        # Check if disclaimer is already present using the same logic as _apply_disclaimer
        try:
            from guardrails.actions import contains_medical_disclaimer
            if contains_medical_disclaimer(answer_text):
                # Remove disclaimer - this is simplified since the actual disclaimer format may vary
                # In practice, UIs should use RAGResponse.disclaimer field instead
                return answer_text, MEDICAL_DISCLAIMER
        except ImportError:
            # Fallback to original detection
            normalized = answer_text.lower()
            if "medical disclaimer" in normalized or MEDICAL_DISCLAIMER.lower() in normalized:
                return answer_text, MEDICAL_DISCLAIMER

        # No disclaimer found
        return answer_text, ""

    def _write_embeddings_metadata(self) -> None:
        """Write embeddings metadata to file"""
        try:
            # Use cached dimension if available to avoid extra network call
            dimension = self.embeddings.get_embedding_dimension()
            metadata = {
                "model_name": self.embeddings.model_name,
                "dimension": dimension,
            }
            selection_reason = getattr(self.embeddings, "model_selection_reason", None)
            if selection_reason:
                metadata["model_selection_reason"] = selection_reason

            metadata_path = self._get_embeddings_metadata_path()
            metadata_path.parent.mkdir(parents=True, exist_ok=True)

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Embeddings metadata written: {metadata}")

        except Exception as e:
            logger.warning(f"Failed to write embeddings metadata: {str(e)}")

    def _check_embeddings_compatibility(self) -> bool:
        """Check if current embeddings are compatible with existing index"""
        metadata_path = self._get_embeddings_metadata_path()

        if not metadata_path.exists():
            logger.info("No embeddings metadata found, will create after building index")
            return True

        try:
            with open(metadata_path, 'r') as f:
                stored_metadata = json.load(f)

            stored_model = stored_metadata.get('model_name')
            current_model = self.embeddings.model_name

            if stored_model != current_model:
                logger.warning(f"⚠️  EMBEDDING MODEL CHANGE DETECTED!")
                logger.warning(f"   Stored model: {stored_model}")
                logger.warning(f"   Current model: {current_model}")
                logger.warning("   This will require a FULL INDEX REBUILD which may take considerable time.")
                logger.warning("   Consumer impact: All queries will be slower until rebuild completes.")
                logger.warning("   To avoid this, set VECTOR_DB_PER_MODEL=true or use the same embedding model.")
                return False

            # Optionally check dimension too (only if both stored and current are valid)
            stored_dimension = stored_metadata.get('dimension')
            if stored_dimension and isinstance(stored_dimension, int):
                try:
                    current_dimension = self.embeddings.get_embedding_dimension()
                    if current_dimension is not None and isinstance(current_dimension, int):
                        if stored_dimension != current_dimension:
                            logger.warning(f"⚠️  EMBEDDING DIMENSION MISMATCH DETECTED!")
                            logger.warning(f"   Stored dimension: {stored_dimension}")
                            logger.warning(f"   Current dimension: {current_dimension}")
                            logger.warning("   This will require a FULL INDEX REBUILD which may take considerable time.")
                            logger.warning("   Consumer impact: All queries will be slower until rebuild completes.")
                            logger.warning("   This typically occurs when switching between different embedding model families.")
                            return False
                except Exception as e:
                    logger.warning(f"Could not verify current dimension: {str(e)}")

            logger.info(f"Embeddings compatible: model='{current_model}'")
            return True

        except Exception as e:
            logger.warning(f"Failed to read embeddings metadata: {str(e)}")
            logger.warning("Will force rebuild of index")
            return False

    def _detect_embedding_metadata_mismatch(self) -> Optional[str]:
        """Return a rebuild advisory when stored metadata mismatches the active embedder."""
        stored_metadata = self._load_embeddings_metadata()
        if not stored_metadata:
            return None

        stored_model = stored_metadata.get("model_name")
        stored_dimension_raw = stored_metadata.get("dimension")

        try:
            stored_dimension = int(stored_dimension_raw)
        except (TypeError, ValueError):
            stored_dimension = None

        current_model = self.embeddings.model_name

        mismatch_details = []

        if stored_model and stored_model != current_model:
            mismatch_details.append(
                f"embedding model '{stored_model}' (stored) vs '{current_model}' (active)"
            )

        current_dimension: Optional[int] = None
        if stored_dimension is not None:
            try:
                current_dimension_value = self.embeddings.get_embedding_dimension()
                if isinstance(current_dimension_value, int):
                    current_dimension = current_dimension_value
            except Exception as dimension_error:
                logger.debug(
                    "Unable to resolve current embedding dimension for comparison: %s",
                    dimension_error,
                )

        if stored_dimension is not None and current_dimension is not None and stored_dimension != current_dimension:
            mismatch_details.append(
                f"embedding dimension {stored_dimension} (stored) vs {current_dimension} (active)"
            )

        if mismatch_details:
            detail_text = "; ".join(mismatch_details)
            return (
                "Stored embeddings metadata is incompatible with the active embedding runtime ("
                f"{detail_text}). Please rebuild the knowledge base to regenerate the vector index."
            )

        return None

    def _check_and_migrate_legacy_index(self) -> bool:
        """
        Check for legacy index at base vector_db_path and migrate if compatible

        Returns:
            True if legacy index was found and successfully migrated, False otherwise
        """
        # Only attempt migration when using per-model paths
        if not self.use_per_model_path:
            return False

        legacy_metadata_path = Path(self.base_vector_db_path) / "embeddings_meta.json"

        # Try to load legacy metadata if it exists
        legacy_metadata = None
        if legacy_metadata_path.exists():
            try:
                with open(legacy_metadata_path, 'r') as f:
                    legacy_metadata = json.load(f)

                legacy_model = legacy_metadata.get('model_name')
                current_model = self.embeddings.model_name

                if legacy_model != current_model:
                    logger.info(f"Legacy index found but model mismatch: legacy='{legacy_model}', current='{current_model}'")
                    logger.info("Proceeding with rebuild instead of migration")
                    return False
            except Exception as e:
                logger.warning(f"Failed to read legacy metadata: {str(e)}")
                legacy_metadata = None

        # Try to load legacy vector DB files, even without metadata
        try:
            from .vector_database import VectorDatabase
            legacy_vector_db = VectorDatabase(self.embeddings, self.base_vector_db_path)

            if legacy_vector_db.load_index():
                if legacy_metadata:
                    legacy_model = legacy_metadata.get('model_name')
                    logger.info(f"Legacy index found and compatible (model: {legacy_model})")
                else:
                    logger.info("Legacy index found without metadata, treating as compatible")

                logger.info(f"Migrating legacy index from {self.base_vector_db_path} to {self.vector_db_path}")

                # Copy the loaded vector store to current vector DB
                self.vector_db.vectorstore = legacy_vector_db.vectorstore

                # Save to new per-model location
                if self.vector_db.save_index():
                    # Write metadata for new location (with current model info)
                    self._write_embeddings_metadata()
                    if not legacy_metadata:
                        logger.info("Legacy index migration completed successfully (metadata was missing but index was migrated)")
                    else:
                        logger.info("Legacy index migration completed successfully")
                    return True
                else:
                    logger.error("Failed to save migrated index")
                    return False
            else:
                logger.debug("Legacy index files not found or unloadable")
                return False

        except Exception as e:
            logger.warning(f"Failed to check/migrate legacy index: {str(e)}")
            return False

    def _ensure_vector_db_path_current(self) -> Tuple[bool, bool]:
        """Ensure vector DB path matches the active embedding model.

        Returns:
            Tuple where the first element indicates whether a reconciliation
            occurred, and the second indicates if a rebuild is required.
        """
        if not self.use_per_model_path:
            return False, False

        current_model = self.embeddings.model_name
        expected_dirname = self._sanitize_model_name(current_model)
        expected_path = os.path.join(self.base_vector_db_path, expected_dirname)

        previous_model = getattr(self, "_active_vector_db_model", None)
        previous_path = getattr(self, "vector_db_path", expected_path)

        path_changed = (previous_path != expected_path) or (previous_model != current_model)

        if not path_changed:
            # Keep the active model tracker in sync even when nothing changed.
            self._active_vector_db_model = current_model
            return False, False

        logger.warning(
            "Embedding model changed from '%s' to '%s'; reconciling vector DB path.",
            previous_model,
            current_model,
        )

        rebuild_required = False
        previous_path_obj = Path(previous_path) if previous_path else None
        expected_path_obj = Path(expected_path)

        if previous_path == expected_path:
            # Same directory but different model; rebuild to avoid incompatible data reuse.
            rebuild_required = True
            logger.info(
                "Vector DB directory '%s' already matches expected path but indexes belong to '%s'. Forcing rebuild.",
                expected_path,
                previous_model,
            )
        else:
            if previous_path_obj and previous_path_obj.exists() and not expected_path_obj.exists():
                metadata_path = previous_path_obj / "embeddings_meta.json"
                stored_model = None
                try:
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            stored_metadata = json.load(f)
                            stored_model = stored_metadata.get('model_name')
                except Exception as metadata_error:
                    logger.debug(
                        "Could not inspect metadata at %s for migration: %s",
                        metadata_path,
                        metadata_error,
                    )

                if stored_model == current_model:
                    try:
                        expected_path_obj.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(previous_path_obj), str(expected_path_obj))
                        logger.info(
                            "Moved vector DB directory from '%s' to '%s' to align with embedding fallback.",
                            previous_path,
                            expected_path,
                        )
                    except Exception as move_error:
                        logger.warning(
                            "Failed to move vector DB directory from '%s' to '%s': %s",
                            previous_path,
                            expected_path,
                            move_error,
                        )
                        rebuild_required = True
                        logger.info(
                            "Move fallback unsuccessful; the index will be rebuilt at '%s' to align with the active embedding model.",
                            expected_path,
                        )
                else:
                    rebuild_required = True
                    logger.info(
                        "Existing vector DB at '%s' targets model '%s', which mismatches '%s'. Will rebuild at '%s'.",
                        previous_path,
                        stored_model,
                        current_model,
                        expected_path,
                    )
            elif (
                previous_path_obj
                and previous_path_obj.exists()
                and expected_path_obj.exists()
                and previous_path_obj.resolve() != expected_path_obj.resolve()
            ):
                rebuild_required = True
                logger.warning(
                    "Both '%s' and '%s' exist. Preferring expected path and forcing rebuild to avoid conflicts.",
                    previous_path,
                    expected_path,
                )

        # Update internal bookkeeping and vector DB paths
        self.vector_db_path = expected_path
        self._active_vector_db_model = current_model

        vector_db = getattr(self, "vector_db", None)
        if vector_db:
            vector_db.update_base_path(self.vector_db_path)
            logger.info(
                "Vector DB path reconciled to '%s' for embedding model '%s'.",
                self.vector_db_path,
                current_model,
            )

        if not expected_path_obj.exists():
            try:
                expected_path_obj.mkdir(parents=True, exist_ok=True)
            except Exception as mkdir_error:
                logger.debug(
                    "Unable to pre-create vector DB directory %s: %s",
                    expected_path,
                    mkdir_error,
                )

        if rebuild_required:
            logger.warning(
                "Rebuild required for vector DB at '%s' to align with embedding model '%s'. Run setup_knowledge_base(force_rebuild=True) after preparing documents.",
                self.vector_db_path,
                current_model,
            )

        return True, rebuild_required

    def _reconcile_before_query(self) -> Tuple[bool, bool, bool, Optional[str]]:
        """Reconcile vector DB state before query execution.

        Returns:
            Tuple of (reconciled, rebuild_required, load_success, rebuild_message)
        """
        reconciled, rebuild_required = self._ensure_vector_db_path_current()
        load_success = True
        rebuild_message = None

        if not rebuild_required:
            rebuild_message = self._detect_embedding_metadata_mismatch()
            if rebuild_message:
                rebuild_required = True
                load_success = False
                logger.warning(rebuild_message)

        if rebuild_required:
            return reconciled, True, load_success, rebuild_message

        if reconciled:
            load_success = self.vector_db.load_index()
            if not load_success:
                rebuild_required = True

        # Defensive dimension compatibility check after any vector DB load
        try:
            vs = self.vector_db.vectorstore
            if vs is not None:
                faiss_dim = getattr(getattr(vs, 'index', None), 'd', None)
                embed_dim = self.embeddings.get_embedding_dimension()
                if isinstance(faiss_dim, int) and isinstance(embed_dim, int) and faiss_dim != embed_dim:
                    logger.warning(
                        "FAISS index dimension %s != active embedder dimension %s; forcing rebuild.",
                        faiss_dim, embed_dim,
                    )
                    return reconciled, True, False, (
                        "Stored vector index dimensionality is incompatible with the active embedder. "
                        "Please rebuild the knowledge base."
                    )
        except Exception as dim_err:
            logger.debug("Dimension compatibility check failed: %s", dim_err)

        return reconciled, rebuild_required, load_success, rebuild_message

    def setup_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """
        Set up the knowledge base from PDF documents

        Args:
            force_rebuild: Whether to force rebuild the index even if it exists

        Returns:
            True if successful, False otherwise
        """
        try:
            _, rebuild_required = self._ensure_vector_db_path_current()
            if rebuild_required:
                force_rebuild = True

            # Check embeddings compatibility before loading index
            if not force_rebuild:
                if not self._check_embeddings_compatibility():
                    logger.info("Embedding model changed; rebuilding index for compatibility.")
                    force_rebuild = True

            if not force_rebuild:
                _, rebuild_after_probe = self._ensure_vector_db_path_current()
                if rebuild_after_probe:
                    force_rebuild = True

            # Try to load existing index first
            if not force_rebuild and self.vector_db.load_index():
                logger.info(f"✅ Loaded existing knowledge base (reused compatible index at {self.vector_db_path})")
                return True

            # Check for legacy index migration (only when using per-model paths)
            if not force_rebuild:
                legacy_migrated = self._check_and_migrate_legacy_index()
                if legacy_migrated:
                    logger.info("✅ Loaded existing knowledge base (migrated from legacy index)")
                    return True

            # If per-model mode is enabled but per-model path load failed,
            # attempt to load from the legacy base path even when no embeddings_meta.json is present
            if not force_rebuild and self.use_per_model_path and self.vector_db_path != self.base_vector_db_path:
                try:
                    from .vector_database import VectorDatabase
                    base_vector_db = VectorDatabase(self.embeddings, self.base_vector_db_path)

                    if base_vector_db.load_index():
                        logger.info(f"Loaded existing index from legacy base path: {self.base_vector_db_path}")

                        # Copy the loaded vector store to current vector DB
                        self.vector_db.vectorstore = base_vector_db.vectorstore

                        # Save to per-model location and write metadata
                        if self.vector_db.save_index():
                            self._write_embeddings_metadata()
                            logger.info(f"✅ Loaded existing knowledge base (migrated from legacy base path to {self.vector_db_path})")
                            return True
                        else:
                            logger.warning("Failed to save migrated index from legacy base path")
                except Exception as e:
                    logger.debug(f"Could not load from legacy base path: {str(e)}")
                    # Continue with rebuild
            
            logger.info("Building knowledge base from PDF documents...")
            
            # Load and process documents
            documents = self.document_loader.load_and_split()
            
            if not documents:
                logger.error("No documents found to build knowledge base")
                return False
            
            # Get document stats
            stats = self.document_loader.get_document_stats(documents)
            logger.info(f"Loaded {stats['num_source_files']} PDF files with {stats['total_chunks']} chunks")
            
            # Create vector index
            if not self.vector_db.create_index(documents):
                logger.error("Failed to create vector index")
                return False
            
            # Save index
            if not self.vector_db.save_index():
                logger.error("Failed to save vector index")
                return False

            # Write embeddings metadata after successful build
            self._write_embeddings_metadata()

            logger.info("✅ Knowledge base setup completed successfully (rebuilt index)!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup knowledge base: {str(e)}")
            return False
    

    
    def ask_question(
        self,
        question: str,
        k: int = 4,
        disclaimer_already_present: Optional[bool] = None,
    ) -> RAGResponse:
        """
        Ask a question and get an answer from the knowledge base

        Args:
            question: The question to ask
            k: Number of relevant documents to retrieve
            disclaimer_already_present: When True, skip adding disclaimer text to the answer

        Returns:
            RAGResponse with answer and source documents

        Note:
            Downstream consumers should prefer RAGResponse.disclaimer when displaying
            disclaimers to avoid duplicating MEDICAL_DISCLAIMER in the rendered text.
        """
        import time
        start_time = time.time()

        try:
            preflight_snippet = question[:64] if question else ""
            force_preflight = self.use_per_model_path or (not self.vector_db.vectorstore and self.force_preflight_on_first_query) or (not self._did_one_time_preflight and self.force_preflight_on_first_query)
            if preflight_snippet and (self.enable_preflight_embedding or force_preflight):
                preflight_model_name = self.embeddings.model_name
                try:
                    # Mandatory or opt-in preflight embedding to detect runtime fallback before querying the index.
                    if not self._did_one_time_preflight:
                        self._did_one_time_preflight = True
                    self.embeddings.embed_query(preflight_snippet)
                    if self.embeddings.model_name != preflight_model_name:
                        logger.info(
                            "Embedding fallback detected during question preflight; switching from '%s' to '%s'.",
                            preflight_model_name,
                            self.embeddings.model_name,
                        )
                        self._active_vector_db_model = self.embeddings.model_name
                        # Persist updated embedding model information
                        self._write_embeddings_metadata()
                        # If models diverge, reconcile paths and signal rebuild when necessary.
                        (
                            reconciled,
                            rebuild_required,
                            load_success,
                            rebuild_message,
                        ) = self._reconcile_before_query()
                        advisory = (
                            rebuild_message
                            or "Embedding model changed after fallback. Please rebuild the knowledge base to ensure dimension compatibility."
                        )
                        return RAGResponse(
                            answer=self._apply_disclaimer(
                                advisory,
                                disclaimer_already_present=disclaimer_already_present,
                            ),
                            source_documents=[],
                            query=question,
                            processing_time=time.time() - start_time,
                            disclaimer=MEDICAL_DISCLAIMER,
                            needs_rebuild=True,
                        )
                except Exception as preflight_error:
                    logger.debug("Preflight embedding failed: %s", preflight_error)
            elif preflight_snippet:
                logger.debug("Skipping embedding preflight because it is disabled and not forced.")

            previous_vector_model = getattr(self, "_active_vector_db_model", None)
            runtime_model_changed = self.embeddings.model_name != previous_vector_model
            (
                reconciled,
                rebuild_required,
                load_success,
                rebuild_message,
            ) = self._reconcile_before_query()

            if rebuild_message:
                logger.info(
                    "Returning rebuild guidance to caller instead of querying vector store due to metadata mismatch."
                )
                return RAGResponse(
                    answer=self._apply_disclaimer(
                        rebuild_message,
                        disclaimer_already_present=disclaimer_already_present,
                    ),
                    source_documents=[],
                    query=question,
                    processing_time=time.time() - start_time,
                    disclaimer=MEDICAL_DISCLAIMER,
                    needs_rebuild=True,
                )

            if runtime_model_changed:
                load_ok = load_success if reconciled else self.vector_db.load_index()
                if rebuild_required or not load_ok:
                    logger.warning(
                        "Embedding model changed from '%s' to '%s'. Rebuild the knowledge base to regenerate the vector index (VECTOR_DB_PER_MODEL guidance applies).",
                        previous_vector_model,
                        self.embeddings.model_name,
                    )
                    return RAGResponse(
                        answer=self._apply_disclaimer(
                            "The active embedding model changed at runtime. Please rebuild the knowledge base to ensure compatibility.",
                            disclaimer_already_present=disclaimer_already_present,
                        ),
                        source_documents=[],
                        query=question,
                        processing_time=time.time() - start_time,
                        disclaimer=MEDICAL_DISCLAIMER,
                        needs_rebuild=True,
                    )

            if not self.vector_db.vectorstore:
                # Try explicit lazy load when uninitialized but files exist
                logger.info("Vector database not initialized - attempting lazy load...")
                if self.vector_db.load_index():
                    logger.info("✅ Lazy load successful, proceeding with query")
                    # Continue with the query processing
                else:
                    # Respect rebuild advisories from _reconcile_before_query
                    if rebuild_required or rebuild_message:
                        guidance_text = rebuild_message or "Please rebuild the knowledge base to ensure compatibility."
                        logger.error("Lazy load failed and rebuild is required.")
                        return RAGResponse(
                            answer=self._apply_disclaimer(
                                guidance_text,
                                disclaimer_already_present=disclaimer_already_present,
                            ),
                            source_documents=[],
                            query=question,
                            processing_time=time.time() - start_time,
                            disclaimer=MEDICAL_DISCLAIMER,
                            needs_rebuild=True,
                        )
                    else:
                        logger.error("Vector database not initialized and lazy load failed. Please setup knowledge base first.")
                        return RAGResponse(
                            answer=self._apply_disclaimer(
                                "Knowledge base not initialized. Please setup the knowledge base first.",
                                disclaimer_already_present=disclaimer_already_present,
                            ),
                            source_documents=[],
                            query=question,
                            processing_time=time.time() - start_time,
                            disclaimer=MEDICAL_DISCLAIMER,
                        )

            logger.info(f"Processing question: {question}")

            # Get relevant documents
            try:
                scored_docs = self.vector_db.similarity_search_with_scores(question, k=k)
            except Exception as search_error:
                error_text = str(search_error)
                if "dimension" in error_text.lower() or "shape" in error_text.lower():
                    logger.error("Vector search failed due to potential embedding dimension mismatch: %s", error_text)
                    guidance = (
                        "Vector index appears incompatible with the active embedding model. "
                        "Please rebuild the knowledge base to regenerate FAISS indexes with the current embedding dimensions."
                    )
                    return RAGResponse(
                        answer=self._apply_disclaimer(
                            guidance,
                            disclaimer_already_present=disclaimer_already_present,
                        ),
                        source_documents=[],
                        query=question,
                        processing_time=time.time() - start_time,
                        disclaimer=MEDICAL_DISCLAIMER,
                        needs_rebuild=True,
                    )
                raise

            if not scored_docs:
                return RAGResponse(
                    answer=self._apply_disclaimer(
                        "I couldn't find any relevant information to answer your question.",
                        disclaimer_already_present=disclaimer_already_present,
                    ),
                    source_documents=[],
                    query=question,
                    processing_time=time.time() - start_time,
                    disclaimer=MEDICAL_DISCLAIMER,
                )

            # Extract documents and scores
            source_docs = [doc for doc, score in scored_docs]
            scores = [score for doc, score in scored_docs]

            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in source_docs])

            # Create prompt
            prompt = self.prompt_template.format(context=context, question=question)

            # Generate answer using LLM
            answer = self.llm.generate_response(prompt)
            answer = self._apply_disclaimer(
                answer,
                disclaimer_already_present=disclaimer_already_present,
            )

            processing_time = time.time() - start_time

            logger.info(f"Question answered in {processing_time:.2f} seconds")

            return RAGResponse(
                answer=answer,
                source_documents=source_docs,
                confidence_scores=scores,
                query=question,
                processing_time=processing_time,
                disclaimer=MEDICAL_DISCLAIMER,
            )

        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            return RAGResponse(
                answer=self._apply_disclaimer(
                    f"I encountered an error while processing your question: {str(e)}",
                    disclaimer_already_present=disclaimer_already_present,
                ),
                source_documents=[],
                query=question,
                processing_time=time.time() - start_time,
                disclaimer=MEDICAL_DISCLAIMER,
            )
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Get relevant documents for a query with similarity scores
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_db.similarity_search_with_scores(query, k=k)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary with knowledge base statistics
        """
        vector_stats = self.vector_db.get_stats()
        
        # Add document loader stats if available
        try:
            if os.path.exists(self.docs_folder):
                docs_path = Path(self.docs_folder)
                pdf_files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.PDF"))
                vector_stats["pdf_files_available"] = len(pdf_files)
                vector_stats["docs_folder"] = self.docs_folder
        except Exception:
            pass
        
        return vector_stats
    
    def add_documents_to_knowledge_base(self, new_docs_folder: Optional[str] = None) -> bool:
        """
        Add new documents to the existing knowledge base
        
        Args:
            new_docs_folder: Optional path to new documents folder
            
        Returns:
            True if successful, False otherwise
        """
        try:
            folder = new_docs_folder or self.docs_folder
            
            # Load new documents
            loader = PDFDocumentLoader(folder)
            new_documents = loader.load_and_split()
            
            if not new_documents:
                logger.warning("No new documents found to add")
                return False
            
            # Add to vector database
            if self.vector_db.add_documents(new_documents):
                # Save updated index
                self.vector_db.save_index()
                logger.info(f"✅ Added {len(new_documents)} new document chunks")
                return True
            else:
                logger.error("Failed to add new documents")
                return False
                
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False

    def get_disclaimer(self) -> str:
        """Get the medical disclaimer for UI display.

        Returns:
            The medical disclaimer text
        """
        return MEDICAL_DISCLAIMER

    def get_guardrail_metadata(self) -> Dict[str, Any]:
        """Get current guardrail metadata.

        Returns:
            Copy of current guardrail metadata dictionary
        """
        return dict(self._guardrail_metadata)

    def apply_disclaimer(self, text: str, *, disclaimer_already_present: Optional[bool] = None) -> str:
        """Apply medical disclaimer to text when configured to do so.

        Args:
            text: Text to potentially append disclaimer to
            disclaimer_already_present: If True, disclaimer won't be added; if None, checks metadata

        Returns:
            Text with disclaimer appended if needed
        """
        return self._apply_disclaimer(text, disclaimer_already_present=disclaimer_already_present)

    # ------------------------------------------------------------------
    # Pharmaceutical filter methods - delegate to VectorDatabase
    # ------------------------------------------------------------------

    def similarity_search_with_pharmaceutical_filters(
        self,
        query: str,
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
        oversample: int = 5,
    ) -> List[Document]:
        """
        Similarity search with pharmaceutical metadata filtering.

        Delegates to VectorDatabase's similarity_search_with_pharmaceutical_filters method.

        Args:
            query: Search query string
            k: Number of results to return
            filters: Pharmaceutical metadata filters (drug_names, species, etc.)
            oversample: Oversampling factor for filter refinement

        Returns:
            List of filtered documents
        """
        try:
            # Ensure vector database is initialized
            if not self.vector_db.vectorstore:
                if not self.vector_db.load_index():
                    logger.warning("Vector database not available for pharmaceutical search")
                    return []

            return self.vector_db.similarity_search_with_pharmaceutical_filters(
                query=query,
                k=k,
                filters=filters,
                oversample=oversample,
            )

        except Exception as e:
            logger.error(f"Pharmaceutical similarity search failed: {str(e)}")
            return []

    def search_by_drug_name(self, drug_name: str, k: int = 10) -> List[Document]:
        """
        Targeted search for documents mentioning a specific drug.

        Delegates to VectorDatabase's search_by_drug_name method.

        Args:
            drug_name: Name of the drug to search for
            k: Number of results to return

        Returns:
            List of documents mentioning the specified drug
        """
        try:
            # Ensure vector database is initialized
            if not self.vector_db.vectorstore:
                if not self.vector_db.load_index():
                    logger.warning("Vector database not available for drug name search")
                    return []

            return self.vector_db.search_by_drug_name(drug_name=drug_name, k=k)

        except Exception as e:
            logger.error(f"Drug name search failed: {str(e)}")
            return []

    def get_pharmaceutical_stats(self) -> Dict[str, Any]:
        """
        Get pharmaceutical statistics about the knowledge base.

        Delegates to VectorDatabase's get_pharmaceutical_stats method.

        Returns:
            Dictionary with pharmaceutical annotation statistics
        """
        try:
            # Ensure vector database is initialized
            if not self.vector_db.vectorstore:
                if not self.vector_db.load_index():
                    logger.warning("Vector database not available for pharmaceutical stats")
                    return {"status": "No index loaded"}

            return self.vector_db.get_pharmaceutical_stats()

        except Exception as e:
            logger.error(f"Pharmaceutical stats retrieval failed: {str(e)}")
            return {"status": "Error", "error": str(e)}


def main():
    """Test the RAG agent"""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get configuration
    api_key = os.getenv("NVIDIA_API_KEY")
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    
    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment variables")
        return
    
    # Initialize RAG agent
    rag_agent = RAGAgent(docs_folder, api_key)
    
    # Setup knowledge base
    if rag_agent.setup_knowledge_base():
        print("✅ Knowledge base setup successful!")
        
        # Get stats
        stats = rag_agent.get_knowledge_base_stats()
        print(f"Knowledge base stats: {stats}")
        
        # Test question
        test_question = "What is the main topic of the documents?"
        response = rag_agent.ask_question(test_question)
        
        print(f"\nQuestion: {test_question}")
        print(f"Answer: {response.answer}")
        print(f"Sources: {len(response.source_documents)} documents")
        print(f"Processing time: {response.processing_time:.2f} seconds")
        
    else:
        print("❌ Failed to setup knowledge base")


if __name__ == "__main__":
    main()
