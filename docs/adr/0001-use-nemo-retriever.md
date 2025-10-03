# ADR-0001: Adopt NVIDIA NeMo Retriever for RAG Pipeline

---

Last Updated: 2025-10-03
Owner: Architecture Team
Review Cadence: Quarterly
Status: Accepted
Deciders: Architecture Team, Engineering Team
Date: 2024-09-15
Template Version: 1.0

---

## Status

**Accepted** - Implemented and operational as of January 2025

## Context

The project required a production-grade RAG system for pharmaceutical research with the following requirements:

- **High-quality embeddings** optimized for scientific and medical content
- **Reranking capabilities** to improve retrieval precision
- **Extraction services** for structured data from unstructured text
- **Pharmaceutical domain support** with specialized terminology handling
- **NGC deprecation immunity** (March 2026 deadline approaching)
- **Cost optimization** for research budgets
- **Regulatory compliance** considerations for pharmaceutical use cases

### Constraints

- **NGC API Deprecation**: NVIDIA announced NGC API deprecation for March 2026, requiring migration away from NGC-dependent systems
- **Budget Limitations**: Academic and research budgets necessitate free tier or cost-effective solutions
- **Data Sovereignty**: Pharmaceutical research may require local data processing options
- **Uptime Requirements**: System must maintain high availability for ongoing research
- **Domain Expertise**: Pharmaceutical terminology and context require specialized models

### Evaluation Period

September 2024 - January 2025

### Existing Landscape

Prior to this decision, the following options were commonly used in RAG systems:

- LangChain + FAISS with OpenAI embeddings
- Sentence Transformers with custom models
- Cloud provider solutions (Azure OpenAI, AWS Bedrock)
- Custom in-house RAG implementations

## Decision

**We will adopt NVIDIA NeMo Retriever as our core RAG technology** with the following implementation strategy:

1. **Use NVIDIA Build platform (integrate.api.nvidia.com) as primary endpoint**

   - Cloud-managed infrastructure
   - Free tier: 10,000 requests/month for development and small-scale research
   - Automatic updates and model improvements

2. **Implement cloud-first strategy with self-hosted NIM fallback**

   - Primary: NVIDIA Build cloud API
   - Secondary: Self-hosted NVIDIA Inference Microservices (NIMs)
   - Seamless fallback mechanism for data sovereignty requirements

3. **Leverage three-step NeMo Retriever pipeline**

   - **Embedding**: `llama-3.2-nemoretriever-1b-vlm-embed-v1` (1024 dimensions) or `nvidia/nv-embed-v1` (4096 dimensions)
   - **Reranking**: `llama-3_2-nemoretriever-500m-rerank-v2` for precision improvement
   - **Extraction**: Structured data extraction from retrieved documents

4. **Integrate pharmaceutical guardrails using NeMo Guardrails framework**

   - Input validation for medical queries
   - Retrieval filtering for credible sources
   - Output safety with medical disclaimers

5. **Maintain backward compatibility**
   - Support existing FAISS indexes
   - Preserve LangChain integration patterns
   - Enable gradual migration from legacy systems

## Consequences

### Positive Consequences

1. **NGC Deprecation Immunity** ✅

   - System architecture avoids deprecated NGC APIs entirely
   - 6+ months ahead of March 2026 deadline
   - Future-proof against NVIDIA infrastructure changes

2. **Access to State-of-the-Art Models**

   - NeMo Retriever models specifically optimized for RAG tasks
   - Superior performance on pharmaceutical and scientific content
   - Continuous model updates from NVIDIA

3. **Cost Efficiency**

   - Free tier: 10,000 requests/month (sufficient for development and small research projects)
   - Predictable pricing for scaled usage
   - Self-hosted option eliminates recurring API costs for high-volume use

4. **OpenAI SDK Compatibility**

   - Familiar API patterns reduce learning curve
   - Easy integration with existing LangChain/LlamaIndex workflows
   - Standardized interfaces across cloud and self-hosted deployments

5. **Pharmaceutical Domain Support**

   - Models trained on scientific and medical corpora
   - Better handling of pharmaceutical terminology
   - Improved accuracy on drug interaction and mechanism queries

6. **Flexibility**

   - Self-hosted fallback for data sovereignty requirements
   - Multi-cloud deployment options
   - No vendor lock-in (can migrate to self-hosted NIMs)

7. **Enterprise Support**
   - Active NVIDIA support channels
   - Regular model updates and improvements
   - Community and documentation resources

### Negative Consequences

1. **Vendor Dependency on NVIDIA**

   - Primary reliance on NVIDIA Build platform availability
   - Model updates controlled by NVIDIA schedule
   - Mitigated by: Self-hosted NIM fallback option

2. **Learning Curve**

   - Team must learn NeMo-specific features and configuration
   - Different from traditional OpenAI/Anthropic API patterns
   - Mitigated by: Comprehensive documentation and wrapper classes

3. **Free Tier Limitations**

   - 10,000 requests/month may be insufficient for production scale
   - Requires cost monitoring and optimization
   - Mitigated by: Credit tracking, batch optimization, caching strategies

4. **Model Availability**

   - Specific models subject to NVIDIA Build platform availability
   - Potential for model deprecations (though less frequent than NGC)
   - Mitigated by: Fallback model configuration, self-hosted option

5. **Infrastructure Complexity**
   - Self-hosted NIMs require GPU infrastructure for fallback scenarios
   - Increased operational burden for full self-hosted deployment
   - Mitigated by: Cloud-first strategy, optional self-hosting

### Trade-offs

| Chose            | Over                       | Rationale                                                   |
| ---------------- | -------------------------- | ----------------------------------------------------------- |
| Cloud-first      | Fully self-hosted          | Easier deployment, automatic updates, free tier access      |
| Free tier limits | Unlimited self-hosted      | Zero infrastructure costs during development                |
| NGC immunity     | Immediate feature parity   | Long-term viability more important than short-term features |
| NVIDIA ecosystem | Multi-provider flexibility | Deeper integration, better pharmaceutical performance       |
| Managed service  | Full control               | Reduced operational burden, faster time-to-value            |

## Alternatives Considered

### Alternative 1: Continue with LangChain + OpenAI Embeddings

**Description**: Maintain existing LangChain framework with OpenAI's `text-embedding-3-small` or `text-embedding-3-large` models

**Pros**:

- Familiar technology stack
- Excellent documentation and community support
- Large ecosystem of integrations
- Proven reliability

**Cons**:

- Higher ongoing costs ($0.00013-0.0001 per 1K tokens)
- No pharmaceutical domain specialization
- No NGC immunity (not applicable, but no strategic alignment)
- Generic embeddings not optimized for scientific content
- No integrated reranking solution

**Reason for Rejection**: Cost and lack of domain specialization. OpenAI embeddings are excellent for general use but lack pharmaceutical optimization and would incur ongoing costs for research projects.

### Alternative 2: Fully Self-Hosted Open Source (Sentence Transformers + FAISS)

**Description**: Deploy open-source models (e.g., `all-mpnet-base-v2`, `bge-large`) with local FAISS indexes

**Pros**:

- Complete control over infrastructure
- Zero API costs
- Data sovereignty by default
- Customizable models

**Cons**:

- GPU infrastructure required
- Model quality gaps compared to NVIDIA NeMo Retriever
- Maintenance burden (updates, monitoring, scaling)
- No pharmaceutical domain features
- Limited reranking capabilities

**Reason for Rejection**: Maintenance burden and quality gaps. While cost-effective, the operational overhead and performance trade-offs were deemed too significant for pharmaceutical research requirements.

### Alternative 3: Azure OpenAI + Custom Pharmaceutical Layer

**Description**: Use Azure OpenAI's enterprise-grade APIs with custom pharmaceutical safety and terminology handling

**Pros**:

- Enterprise-grade support and SLAs
- High availability and geographic distribution
- Familiar OpenAI API patterns
- Compliance certifications (HIPAA, SOC 2)

**Cons**:

- Significantly higher costs (no free tier)
- No NGC immunity (not applicable)
- Custom pharmaceutical layer increases complexity
- Still generic embeddings without domain optimization

**Reason for Rejection**: Cost and complexity. Azure's enterprise features are valuable but overkill for research projects, and pharmaceutical customization would still be required.

### Alternative 4: Hybrid Multi-Provider (OpenAI + Cohere + Custom)

**Description**: Use best-of-breed components: OpenAI for embeddings, Cohere for reranking, custom models for domain-specific tasks

**Pros**:

- Flexibility to choose optimal service for each task
- Can optimize for cost vs. quality per component
- Reduced single-vendor dependency

**Cons**:

- Significant integration complexity
- Multiple vendor dependencies and billing
- Cost unpredictability
- No cohesive pharmaceutical domain strategy
- Increased operational burden

**Reason for Rejection**: Operational complexity and cost unpredictability outweighed flexibility benefits.

## Implementation Notes

### Migration Status

✅ **Completed**: January 2025

### Key Implementation Details

1. **Wrapper Classes**

   - `NVIDIAEmbeddings` class compatible with LangChain `Embeddings` interface
   - Automatic fallback from llama-3.2-nemoretriever to nv-embed-v1
   - Exponential backoff and retry logic for resilience

2. **Configuration**

   - Environment variables: `NVIDIA_API_KEY`, `NVIDIA_BUILD_BASE_URL`
   - Model selection: `EMBEDDING_MODEL`, `RERANK_MODEL`
   - Fallback configuration: `EMBEDDING_FALLBACK_MODEL`

3. **Backward Compatibility**

   - Existing FAISS indexes preserved
   - LangChain integration maintained
   - No breaking changes to existing workflows

4. **Pharmaceutical Features**

   - Guardrails integrated via `guardrails/` directory
   - Medical terminology detection
   - Drug interaction awareness
   - Safety disclaimers on all outputs

5. **Cost Monitoring**
   - `src/monitoring/credit_tracker.py` for usage tracking
   - Budget enforcement with `PHARMA_BUDGET_LIMIT_USD`
   - Batch optimization to maximize free tier

### Timeline

- **September 2024**: Decision made, architecture designed
- **October-November 2024**: Implementation and testing
- **December 2024**: Pharmaceutical features integration
- **January 2025**: Production deployment, documentation completed

### Rollback Strategy

If NVIDIA Build platform experiences extended outages:

1. Switch to self-hosted NIMs via `NEMO_EMBEDDING_ENDPOINT` override
2. Use cached embeddings where available
3. Fall back to `nv-embed-v1` model if llama-3.2 unavailable

## References

### Official Documentation

- [NVIDIA NeMo Retriever Documentation](https://docs.nvidia.com/nemo-retriever/)
- [NVIDIA Build Platform](https://build.nvidia.com)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)

### Internal Documentation

- [Architecture Documentation](../ARCHITECTURE.md)
- [NGC Deprecation Immunity Guide](../NGC_DEPRECATION_IMMUNITY.md)
- [API Reference](../API_REFERENCE.md)
- [Free Tier Maximization](../FREE_TIER_MAXIMIZATION.md)
- [Cheapest Deployment Guide](../CHEAPEST_DEPLOYMENT.md)

### Code References

- `src/nvidia_embeddings.py` - Core embedding client implementation
- `src/nemo_reranking_service.py` - Reranking service
- `src/monitoring/credit_tracker.py` - Cost tracking
- `guardrails/` - Pharmaceutical safety guardrails

### Related ADRs

- (None yet - this is ADR 0001)

### External References

- [NGC Deprecation Announcement](https://docs.nvidia.com/ngc/) (March 2026 timeline)
- [OpenAI SDK Compatibility](https://github.com/openai/openai-python)

## Review and Updates

### Review Cadence

**Quarterly** - Review every 3 months to ensure decision remains valid

### Next Review Date

2025-04-15

### Update Triggers

This ADR should be updated if:

1. **NVIDIA Build platform changes significantly** (new pricing, deprecated models, major feature changes)
2. **NGC deprecation timeline changes** (accelerated or delayed from March 2026)
3. **Alternative solutions emerge** that significantly outperform NeMo Retriever for pharmaceutical RAG
4. **Cost structure changes** make NVIDIA Build uncompetitive
5. **Self-hosted NIMs become primary deployment** (would require new ADR on cloud-to-self-hosted migration)
6. **Pharmaceutical compliance requirements change** necessitating different architecture

### Change Log

| Date       | Version | Changes                                         | Author            |
| ---------- | ------- | ----------------------------------------------- | ----------------- |
| 2025-01-15 | 1.0     | Initial ADR documenting NeMo Retriever adoption | Architecture Team |

---

## Conclusion

The adoption of NVIDIA NeMo Retriever provides a future-proof, cost-effective, and pharmaceutically-optimized RAG foundation with NGC deprecation immunity, ensuring continuity of research operations beyond March 2026.
