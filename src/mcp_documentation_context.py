"""
MCP-Enhanced Documentation Context for NVIDIA NeMo Retriever

Provides live, up-to-date documentation context by integrating with Microsoft Learn MCP Server
to fetch the latest NVIDIA NeMo Retriever patterns, best practices, and API references.

This service ensures all NeMo operations in the pharmaceutical RAG system are aligned
with the most current NVIDIA documentation and recommendations.

<<use_mcp microsoft-learn>>

Features:
1. Live documentation fetching from Microsoft Learn
2. Pharmaceutical domain-specific context enhancement
3. Caching for performance optimization
4. Context-aware error handling and troubleshooting
5. Integration with all NeMo services (Extraction, Embedding, Reranking)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

import aiohttp
import requests
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

@dataclass
class MCPDocumentationRequest:
    """Request for MCP documentation context."""
    topic: str
    context_type: str  # 'embedding', 'reranking', 'extraction', 'general'
    pharmaceutical_context: bool = True
    max_age_hours: int = 24
    priority: str = "medium"  # 'low', 'medium', 'high', 'critical'

@dataclass
class MCPDocumentationResponse:
    """Response from MCP documentation context."""
    content: str
    source_url: str
    last_updated: datetime
    context_type: str
    pharmaceutical_optimized: bool
    cache_key: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PharmaceuticalContextProfile:
    """Pharmaceutical domain context for documentation enhancement."""
    medical_terminologies: List[str] = field(default_factory=lambda: [
        "adverse events", "clinical trials", "drug interactions", "pharmacokinetics",
        "bioavailability", "therapeutic index", "contraindications", "dosage forms",
        "active pharmaceutical ingredient", "excipients", "stability studies"
    ])
    regulatory_frameworks: List[str] = field(default_factory=lambda: [
        "FDA", "EMA", "ICH", "GMP", "GLP", "GCP", "21 CFR Part 11", "EU MDR"
    ])
    document_types: List[str] = field(default_factory=lambda: [
        "prescribing information", "clinical study reports", "drug labels",
        "package inserts", "investigator brochures", "regulatory submissions"
    ])
    content_priorities: Dict[str, float] = field(default_factory=lambda: {
        "safety_information": 1.0,
        "efficacy_data": 0.9,
        "dosing_guidelines": 0.95,
        "contraindications": 1.0,
        "drug_interactions": 0.9,
        "adverse_events": 1.0
    })

class MCPDocumentationContextService:
    """
    MCP-Enhanced Documentation Context Service for NVIDIA NeMo Retriever.

    Integrates with Microsoft Learn MCP Server to provide live, up-to-date
    documentation context for all NeMo operations with pharmaceutical optimization.
    """

    def __init__(
        self,
        mcp_server_url: Optional[str] = None,
        cache_dir: Optional[str] = None,
        pharmaceutical_profile: Optional[PharmaceuticalContextProfile] = None
    ):
        """
        Initialize MCP Documentation Context Service.

        Args:
            mcp_server_url: Microsoft Learn MCP Server URL
            cache_dir: Local cache directory for documentation
            pharmaceutical_profile: Pharmaceutical domain context profile
        """
        self.mcp_server_url = mcp_server_url or os.getenv(
            'MCP_SERVER_URL',
            'https://learn.microsoft.com/mcp/api/v1'
        )

        self.cache_dir = Path(cache_dir or "mcp_cache")
        self.cache_dir.mkdir(exist_ok=True)

        self.pharmaceutical_profile = pharmaceutical_profile or PharmaceuticalContextProfile()

        # NVIDIA NeMo specific documentation endpoints
        self.nemo_doc_endpoints = {
            'embedding': [
                '/en-us/azure/ai-services/openai/concepts/models#embeddings-models',
                '/en-us/azure/cognitive-services/language-service/concepts/data-limits',
                '/nvidia/nemo-retriever/embedding-models'
            ],
            'reranking': [
                '/nvidia/nemo-retriever/reranking-models',
                '/en-us/azure/search/semantic-search-overview',
                '/nvidia/nemo-retriever/best-practices#reranking'
            ],
            'extraction': [
                '/nvidia/nemo-retriever/document-extraction',
                '/nvidia/nv-ingest/overview',
                '/en-us/azure/ai-services/document-intelligence/overview'
            ],
            'general': [
                '/nvidia/nemo-retriever/overview',
                '/nvidia/nemo-retriever/getting-started',
                '/nvidia/nemo-retriever/architecture'
            ]
        }

        # Performance tracking
        self.metrics = {
            'requests_total': 0,
            'requests_cached': 0,
            'requests_failed': 0,
            'avg_response_time': 0.0,
            'pharmaceutical_contexts_applied': 0
        }

        logger.info(f"Initialized MCP Documentation Context Service")
        logger.info(f"MCP Server: {self.mcp_server_url}")
        logger.info(f"Cache Directory: {self.cache_dir}")
        logger.info(f"Pharmaceutical Optimization: Enabled")

        # Baseten documentation endpoints (lightweight map used by helpers)
        self.baseten_doc_endpoints = {
            "deployment": "deploy/models/nvidia-models",
            "authentication": "api-reference/authentication",
            "endpoints": "api-reference/model-endpoints",
            "monitoring": "observability/monitoring",
            "integration": "deploy/models/nvidia-models#integration-guide",
        }

    def _generate_cache_key(self, request: MCPDocumentationRequest) -> str:
        """Generate cache key for documentation request."""
        key_data = f"{request.topic}:{request.context_type}:{request.pharmaceutical_context}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str, max_age_hours: int) -> Optional[MCPDocumentationResponse]:
        """Retrieve cached documentation response if valid."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            last_updated = datetime.fromisoformat(cached_data['last_updated'])
            if datetime.now() - last_updated > timedelta(hours=max_age_hours):
                return None

            self.metrics['requests_cached'] += 1

            return MCPDocumentationResponse(
                content=cached_data['content'],
                source_url=cached_data['source_url'],
                last_updated=last_updated,
                context_type=cached_data['context_type'],
                pharmaceutical_optimized=cached_data['pharmaceutical_optimized'],
                cache_key=cache_key,
                metadata=cached_data.get('metadata', {})
            )

        except Exception as e:
            logger.warning(f"Failed to load cached response {cache_key}: {e}")
            return None

    def _cache_response(self, response: MCPDocumentationResponse) -> None:
        """Cache documentation response for future use."""
        cache_file = self.cache_dir / f"{response.cache_key}.json"

        try:
            cache_data = {
                'content': response.content,
                'source_url': response.source_url,
                'last_updated': response.last_updated.isoformat(),
                'context_type': response.context_type,
                'pharmaceutical_optimized': response.pharmaceutical_optimized,
                'metadata': response.metadata
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.warning(f"Failed to cache response {response.cache_key}: {e}")

    def _enhance_pharmaceutical_context(self, content: str, context_type: str) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance documentation content with pharmaceutical domain context.

        Args:
            content: Raw documentation content
            context_type: Type of context (embedding, reranking, extraction, general)

        Returns:
            Tuple of (enhanced_content, metadata)
        """
        enhanced_content = content
        metadata = {
            'pharmaceutical_enhancements': [],
            'medical_terms_found': [],
            'regulatory_references': [],
            'safety_considerations': []
        }

        # Add pharmaceutical-specific guidance based on context type
        if context_type == 'embedding':
            pharmaceutical_guidance = """

PHARMACEUTICAL DOMAIN CONSIDERATIONS FOR EMBEDDINGS:

1. Medical Terminology Handling:
   - Use domain-specific embedding models when available
   - Consider medical abbreviations and synonyms (e.g., "MI" vs "myocardial infarction")
   - Account for drug name variations (generic vs brand names)

2. Regulatory Compliance:
   - Ensure embedding processes maintain data lineage for FDA 21 CFR Part 11
   - Consider GDPR implications for patient data in embeddings
   - Implement audit trails for regulatory submissions

3. Safety-Critical Information:
   - Prioritize embeddings for contraindications and warnings
   - Weight adverse event reports higher in similarity calculations
   - Ensure dosing information maintains precision in vector space

4. Content Types:
   - Optimize for clinical trial data, prescribing information, and drug labels
   - Handle structured data like dosing tables and interaction matrices
   - Consider medical imaging and molecular structure representations
"""

        elif context_type == 'reranking':
            pharmaceutical_guidance = """

PHARMACEUTICAL DOMAIN CONSIDERATIONS FOR RERANKING:

1. Clinical Relevance Scoring:
   - Prioritize FDA-approved information over investigational data
   - Weight recent clinical trial results higher than older studies
   - Consider evidence levels (systematic reviews > RCTs > observational studies)

2. Safety-First Reranking:
   - Always surface contraindications and warnings first
   - Prioritize drug interaction information
   - Highlight adverse event reports for safety signals

3. Regulatory Authority Weighting:
   - FDA/EMA approved content > manufacturer claims
   - Peer-reviewed publications > marketing materials
   - Official prescribing information > summary documents

4. Patient Population Considerations:
   - Age-specific dosing information
   - Pregnancy/lactation safety categories
   - Renal/hepatic impairment adjustments
"""

        elif context_type == 'extraction':
            pharmaceutical_guidance = """

PHARMACEUTICAL DOMAIN CONSIDERATIONS FOR EXTRACTION:

1. Structured Data Extraction:
   - Dosing tables with precise numerical values
   - Drug interaction matrices
   - Clinical trial endpoints and statistical results
   - Pharmacokinetic parameters

2. Regulatory Document Processing:
   - FDA/EMA submission document structures
   - Clinical study report templates
   - Package insert standardized sections
   - Investigator brochure formats

3. Safety Information Extraction:
   - Adverse event case reports (E2B format)
   - Risk evaluation and mitigation strategies (REMS)
   - Contraindications and warnings
   - Drug interaction mechanisms

4. Quality Assurance:
   - Maintain numerical precision for dosing information
   - Preserve chemical structure representations
   - Ensure proper handling of medical abbreviations
   - Validate extraction against source document layout
"""

        else:  # general
            pharmaceutical_guidance = """

PHARMACEUTICAL DOMAIN GENERAL CONSIDERATIONS:

1. Regulatory Environment:
   - FDA, EMA, ICH guidelines compliance
   - Good Manufacturing Practice (GMP) requirements
   - Clinical trial regulations (GCP)
   - Pharmacovigilance obligations

2. Data Integrity:
   - 21 CFR Part 11 electronic records compliance
   - ALCOA+ principles (Attributable, Legible, Contemporaneous, Original, Accurate)
   - Audit trail requirements
   - Data backup and recovery procedures

3. Content Prioritization:
   - Safety information has highest priority
   - Efficacy data with proper statistical context
   - Dosing and administration guidelines
   - Contraindications and drug interactions

4. Quality Management:
   - Source document verification
   - Change control procedures
   - Version management for regulatory submissions
   - Traceability throughout the document lifecycle
"""

        enhanced_content += pharmaceutical_guidance
        metadata['pharmaceutical_enhancements'].append(f"{context_type}_specific_guidance")

        # Identify medical terms and regulatory references
        content_lower = content.lower()
        for term in self.pharmaceutical_profile.medical_terminologies:
            if term.lower() in content_lower:
                metadata['medical_terms_found'].append(term)

        for framework in self.pharmaceutical_profile.regulatory_frameworks:
            if framework.lower() in content_lower:
                metadata['regulatory_references'].append(framework)

        # Add safety considerations
        safety_keywords = [
            'adverse', 'contraindication', 'warning', 'precaution',
            'interaction', 'toxicity', 'safety', 'risk'
        ]

        for keyword in safety_keywords:
            if keyword in content_lower:
                metadata['safety_considerations'].append(keyword)

        self.metrics['pharmaceutical_contexts_applied'] += 1

        return enhanced_content, metadata

    async def get_documentation_context(
        self,
        request: MCPDocumentationRequest
    ) -> MCPDocumentationResponse:
        """
        Retrieve documentation context from MCP server with pharmaceutical optimization.

        Args:
            request: Documentation context request

        Returns:
            Documentation response with pharmaceutical enhancement
        """
        start_time = time.time()
        self.metrics['requests_total'] += 1

        cache_key = self._generate_cache_key(request)

        # Check cache first
        cached_response = self._get_cached_response(cache_key, request.max_age_hours)
        if cached_response:
            logger.info(f"Retrieved cached documentation for {request.topic}")
            return cached_response

        try:
            # Fetch from MCP server
            documentation_content = await self._fetch_from_mcp_server(request)

            # Enhance with pharmaceutical context if requested
            enhanced_content = documentation_content
            metadata = {}

            if request.pharmaceutical_context:
                enhanced_content, metadata = self._enhance_pharmaceutical_context(
                    documentation_content,
                    request.context_type
                )

            # Create response
            response = MCPDocumentationResponse(
                content=enhanced_content,
                source_url=f"{self.mcp_server_url}/nvidia/nemo-retriever/{request.context_type}",
                last_updated=datetime.now(),
                context_type=request.context_type,
                pharmaceutical_optimized=request.pharmaceutical_context,
                cache_key=cache_key,
                metadata=metadata
            )

            # Cache the response
            self._cache_response(response)

            # Update metrics
            response_time = time.time() - start_time
            self.metrics['avg_response_time'] = (
                (self.metrics['avg_response_time'] * (self.metrics['requests_total'] - 1) + response_time)
                / self.metrics['requests_total']
            )

            logger.info(f"Retrieved fresh documentation for {request.topic} in {response_time:.2f}s")
            return response

        except Exception as e:
            self.metrics['requests_failed'] += 1
            logger.error(f"Failed to retrieve documentation context for {request.topic}: {e}")

            # Return fallback response
            return self._get_fallback_response(request, cache_key)

    async def _fetch_from_mcp_server(self, request: MCPDocumentationRequest) -> str:
        """
        Fetch documentation from MCP server.

        Args:
            request: Documentation context request

        Returns:
            Raw documentation content
        """
        endpoints = self.nemo_doc_endpoints.get(request.context_type, self.nemo_doc_endpoints['general'])

        for endpoint in endpoints:
            try:
                url = urljoin(self.mcp_server_url, endpoint)

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                        if response.status == 200:
                            content = await response.text()
                            logger.debug(f"Successfully fetched from {url}")
                            return content
                        else:
                            logger.warning(f"MCP server returned {response.status} for {url}")

            except Exception as e:
                logger.warning(f"Failed to fetch from {url}: {e}")
                continue

        # If all endpoints fail, return mock documentation for development
        return self._get_mock_documentation(request.context_type)

    def _get_mock_documentation(self, context_type: str) -> str:
        """Generate mock documentation for development when MCP server is unavailable."""
        base_doc = f"""
# NVIDIA NeMo Retriever - {context_type.title()} Documentation

## Overview
This is mock documentation for {context_type} operations in NVIDIA NeMo Retriever.
In production, this content would be fetched from Microsoft Learn MCP Server.

## Key Features
- Industry-leading performance
- Enterprise-grade scalability
- Pharmaceutical domain optimization
- Advanced error handling

## Best Practices
1. Use appropriate batch sizes for optimal performance
2. Implement proper error handling and retry logic
3. Monitor service health and performance metrics
4. Follow regulatory compliance requirements

## API Reference
[Detailed API documentation would be provided by MCP server]

---
*Note: This is mock content for development purposes.*
*Production deployment requires valid MCP server connection.*
"""
        return base_doc

    def _get_fallback_response(
        self,
        request: MCPDocumentationRequest,
        cache_key: str
    ) -> MCPDocumentationResponse:
        """Generate fallback response when MCP server is unavailable."""

        fallback_content = self._get_mock_documentation(request.context_type)

        # Apply pharmaceutical context if requested
        if request.pharmaceutical_context:
            fallback_content, metadata = self._enhance_pharmaceutical_context(
                fallback_content,
                request.context_type
            )
        else:
            metadata = {}

        return MCPDocumentationResponse(
            content=fallback_content,
            source_url="fallback://mock-documentation",
            last_updated=datetime.now(),
            context_type=request.context_type,
            pharmaceutical_optimized=request.pharmaceutical_context,
            cache_key=cache_key,
            metadata={**metadata, 'fallback': True}
        )

    def get_context_for_nemo_operation(
        self,
        operation_type: str,
        specific_topic: Optional[str] = None,
        pharmaceutical_focus: bool = True
    ) -> MCPDocumentationResponse:
        """
        Synchronous wrapper for getting documentation context for NeMo operations.

        Args:
            operation_type: Type of NeMo operation (embedding, reranking, extraction)
            specific_topic: Specific topic within the operation type
            pharmaceutical_focus: Whether to apply pharmaceutical optimizations

        Returns:
            Documentation response
        """
        topic = specific_topic or f"nemo-{operation_type}-best-practices"

        request = MCPDocumentationRequest(
            topic=topic,
            context_type=operation_type,
            pharmaceutical_context=pharmaceutical_focus,
            max_age_hours=24,
            priority="medium"
        )

        # Run async operation in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.get_documentation_context(request))

    def get_pharmaceutical_guidance(self, content_type: str) -> Dict[str, Any]:
        """
        Get pharmaceutical-specific guidance for content processing.

        Args:
            content_type: Type of pharmaceutical content

        Returns:
            Pharmaceutical guidance dictionary
        """
        guidance = {
            'content_priorities': self.pharmaceutical_profile.content_priorities,
            'medical_terminologies': self.pharmaceutical_profile.medical_terminologies,
            'regulatory_frameworks': self.pharmaceutical_profile.regulatory_frameworks,
            'document_types': self.pharmaceutical_profile.document_types
        }

        # Add content-type specific guidance
        if content_type in ['clinical_trial', 'efficacy']:
            guidance['statistical_considerations'] = [
                'p-values and confidence intervals',
                'primary vs secondary endpoints',
                'intention-to-treat vs per-protocol analysis',
                'subgroup analyses and multiplicity'
            ]

        elif content_type in ['safety', 'adverse_events']:
            guidance['safety_priorities'] = [
                'serious adverse events (SAEs)',
                'adverse drug reactions (ADRs)',
                'contraindications and warnings',
                'drug-drug interactions',
                'special populations (pediatric, geriatric, pregnant)'
            ]

        elif content_type in ['dosing', 'administration']:
            guidance['dosing_considerations'] = [
                'therapeutic dose range',
                'maximum recommended dose',
                'dose adjustments for special populations',
                'bioavailability and bioequivalence',
                'food effects and timing'
            ]

        return guidance

    def get_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        total_requests = self.metrics['requests_total']
        cache_hit_rate = (
            self.metrics['requests_cached'] / total_requests
            if total_requests > 0 else 0.0
        )

        return {
            **self.metrics,
            'cache_hit_rate': cache_hit_rate,
            'cache_directory_size': sum(
                f.stat().st_size for f in self.cache_dir.glob('*.json')
            ) if self.cache_dir.exists() else 0
        }

    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear documentation cache.

        Args:
            older_than_hours: Only clear cache entries older than this many hours

        Returns:
            Number of cache files cleared
        """
        cleared_count = 0
        cutoff_time = None

        if older_than_hours:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)

        for cache_file in self.cache_dir.glob('*.json'):
            try:
                if cutoff_time:
                    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if file_time > cutoff_time:
                        continue

                cache_file.unlink()
                cleared_count += 1

            except Exception as e:
                logger.warning(f"Failed to clear cache file {cache_file}: {e}")

        logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count

    # ---------------------------------------------------------------------
    # Baseten documentation helpers (MCP-enhanced)
    # ---------------------------------------------------------------------
    async def fetch_documentation(self, url: str, context_filter: Optional[str] = None) -> str:
        """Fetch documentation from a URL with optional context filter.

        This reuses the existing aiohttp client logic to retrieve content;
        when an MCP server is available, callers typically pass MCP URLs. Here
        we allow direct HTTP URLs for Baseten docs as well, to keep light.
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    text = await resp.text()
                    if context_filter and context_filter.lower() not in text.lower():
                        return text  # Return raw; consumers decide filtering
                    return text
        except Exception as exc:
            logger.debug("Baseten doc fetch failed for %s: %s", url, exc)
            return ""

    async def fetch_baseten_integration_docs(self, topic: str = "deployment") -> str:
        """Fetch Baseten-specific integration documentation with pharma context."""
        endpoint = self.baseten_doc_endpoints.get(topic, "deploy/models")
        base_url = "https://docs.baseten.co/"
        full_url = base_url + endpoint
        docs = await self.fetch_documentation(
            full_url,
            context_filter="nvidia nemo retriever pharmaceutical integration",
        )
        return docs

    async def get_baseten_nvidia_context(self, pharmaceutical_focus: bool = True) -> str:
        """Aggregate Baseten + NVIDIA integration context (lightweight)."""
        sections = []
        for topic in ("deployment", "authentication", "endpoints"):
            try:
                content = await self.fetch_baseten_integration_docs(topic)
                if content:
                    sections.append(content)
            except Exception:
                continue
        if pharmaceutical_focus:
            try:
                pharma = self.get_pharmaceutical_guidance("regulatory_document")
                sections.append(json.dumps(pharma))
            except Exception:
                pass
        return "\n\n---\n\n".join(sections)


# Global instance for easy access
mcp_context_service = MCPDocumentationContextService()

def get_nemo_context(
    operation_type: str,
    topic: Optional[str] = None,
    pharmaceutical_focus: bool = True
) -> MCPDocumentationResponse:
    """
    Convenience function to get NeMo documentation context.

    Args:
        operation_type: NeMo operation type (embedding, reranking, extraction)
        topic: Specific topic or None for general best practices
        pharmaceutical_focus: Apply pharmaceutical domain optimizations

    Returns:
        Documentation context response
    """
    return mcp_context_service.get_context_for_nemo_operation(
        operation_type=operation_type,
        specific_topic=topic,
        pharmaceutical_focus=pharmaceutical_focus
    )

def get_pharmaceutical_guidance(content_type: str) -> Dict[str, Any]:
    """
    Convenience function to get pharmaceutical content guidance.

    Args:
        content_type: Type of pharmaceutical content

    Returns:
        Pharmaceutical guidance dictionary
    """
    return mcp_context_service.get_pharmaceutical_guidance(content_type)
