---
Last Updated: 2025-10-03
Owner: Docs
Review Cadence: Quarterly
---

# Updated Migration & Build Plan - PubMed Pharmaceutical Expert RAG Chatbot

## Current Codebase State Summary

### 📊 Codebase Statistics

- **Total Python Files**: ~67 files
- **Core Source Code**: 26,033 lines in src/ directory
- **Main Application**: 4,217 lines in root directory
- **Test Coverage**: 32 test files ensuring system reliability
- **UI Interface**: 951-line Streamlit application

### 🏗️ Architecture Overview

#### Phase 1 ✅ COMPLETED - NeMo Foundation Services

- **nemo_retriever_client.py** (675 lines) - Universal NIM client
- **nemo_embedding_service.py** (643 lines) - Multi-model embedding service
- **nemo_extraction_service.py** (797 lines) - VLM-based document processing
- **nemo_reranking_service.py** (794 lines) - Cross-modal reranking
- **mcp_documentation_context.py** (690 lines) - Live documentation context

#### Existing Core Components (Preserved)

- **enhanced_rag_agent.py** (1,929 lines) - Main RAG orchestration
- **pubmed_scraper.py** (2,683 lines) - PubMed integration & scraping
- **medical_guardrails.py** (1,976 lines) - Safety & compliance system
- **pharmaceutical_processor.py** (1,553 lines) - Domain-specific processing
- **vector_database.py** (1,608 lines) - FAISS-based vector storage
- **synthesis_engine.py** (1,576 lines) - Response generation
- **query_engine.py** (1,424 lines) - Query processing pipeline

#### Supporting Infrastructure

- **streamlit_app.py** (951 lines) - Web interface
- **main.py** (266 lines) - CLI application
- **Guardrails system** (3 files) - Medical safety enforcement
- **Testing framework** (32 files) - Comprehensive validation
- **Cache management & rate limiting systems**

---

## 🚀 Complete Migration & Build Plan

### Phase 2: Enhanced Embeddings Migration (3-5 days)

#### Step 8-12: Backward-Compatible Embedding Integration

- Create nvidia_embeddings_v2.py with NeMo service integration
- Implement multi-model embedding strategy (E5-v5, Mistral7B-v2, Arctic-Embed-L)
- Add pharmaceutical domain embeddings optimization
- Build embedding performance monitoring & fallback system
- Update vector_database.py to support both legacy FAISS and new cuVS

#### Step 13-15: Vector Database Enhancement

- Implement hybrid FAISS/cuVS vector storage
- Add GPU-accelerated search capabilities
- Create migration utilities for existing vector indices
- Implement pharmaceutical metadata preservation during migration

### Phase 3: LangChain Integration Updates (2-3 days)

#### Step 16-20: Chain Integration & Compatibility

- Update enhanced_rag_agent.py to use NeMo services
- Integrate NeMo reranking into retrieval chains
- Update query_engine.py with NeMo embedding calls
- Implement LangChain NeMo chain wrappers
- Add chain performance monitoring & debugging

### Phase 4: Document Processing Pipeline (2-3 days)

#### Step 21-25: Advanced Document Ingestion

- Update document_loader.py with NeMo extraction service
- Implement VLM-based table/chart extraction for pharmaceutical docs
- Add chemical structure recognition capabilities
- Update pharmaceutical_processor.py with enhanced NeMo processing
- Implement batch document processing optimization

### Phase 5: PubMed Integration Optimization (2-3 days)

#### Step 26-30: Hybrid Search Enhancement

- Update pubmed_scraper.py with NeMo reranking
- Implement intelligent local vs PubMed routing
- Add pharmaceutical authority source weighting
- Update enhanced_pubmed_agent.py integration
- Implement PubMed result pharmaceutical optimization

### Phase 6: Medical Guardrails Integration (2-3 days)

#### Step 31-35: Safety & Compliance Enhancement

- Update medical_guardrails.py with NeMo-aware safety checks
- Implement NeMo content safety filtering
- Add pharmaceutical compliance validation for NeMo outputs
- Update safety monitoring for all NeMo operations
- Implement audit trail for regulatory compliance

### Phase 7: Streamlit UI Enhancement (2-3 days)

#### Step 36-40: User Interface Updates

- Update streamlit_app.py with NeMo service monitoring
- Add NeMo performance metrics dashboard
- Implement pharmaceutical workflow UI enhancements
- Add NeMo model selection interface
- Update caching visualization for NeMo operations

### Phase 8: Docker & Deployment Infrastructure (3-4 days)

#### Step 41-45: Production Deployment

- Create NeMo-enabled Dockerfile with GPU support
- Implement docker-compose with NeMo services
- Add Kubernetes deployment manifests
- Create environment validation automation
- Implement production health monitoring

### Phase 9: Testing & Validation Framework (2-3 days)

#### Step 46-50: Comprehensive Testing

- Update existing 32 test files with NeMo integration tests
- Add NeMo service performance benchmarks
- Implement pharmaceutical accuracy validation tests
- Create integration test suite for hybrid workflows
- Add load testing for production deployment

### Phase 10: Migration Utilities & Documentation (2-3 days)

#### Step 51-55: Final Migration & Docs

- Create automated migration scripts for existing data
- Update all documentation with NeMo integration details
- Create pharmaceutical deployment guide
- Implement monitoring dashboards
- Final system validation & performance optimization

---

## 🎯 Critical Success Factors

### Pharmaceutical Domain Priorities

1. **Safety-First**: All NeMo outputs undergo medical guardrails validation
2. **Regulatory Compliance**: Maintain FDA 21 CFR Part 11 compatibility
3. **Clinical Accuracy**: Preserve pharmaceutical metadata & context
4. **Source Authority**: Prioritize FDA/EMA over manufacturer content

### Technical Excellence

1. **Backward Compatibility**: Existing workflows continue during migration
2. **Performance**: GPU acceleration where available, CPU fallbacks
3. **Reliability**: Comprehensive fallback & error handling
4. **Monitoring**: Real-time performance & accuracy tracking

### Deployment Strategy

1. **Hybrid Deployment**: Support both cloud NIMs & self-hosted options
2. **Gradual Migration**: Phase-by-phase rollout with validation
3. **Environment Flexibility**: Development, staging, production configs
4. **Monitoring**: Comprehensive observability & alerting

---

## 📈 Project Timeline & Resources

- **Estimated Total Timeline**: 20-25 development days
- **Dependencies**: NVIDIA API access, GPU resources for optimal performance
- **Risk Mitigation**: Comprehensive fallback systems ensure continuous operation

---

Last Updated: 2025-10-03
Owner: Docs
Review Cadence: Quarterly

---
