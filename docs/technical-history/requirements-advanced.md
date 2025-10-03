---
Last Updated: 2025-10-03
Owner: Docs
Review Cadence: Quarterly
---

# Pharmaceutical RAG Knowledge Expert - Product Requirements Document

## Project Overview

### Product Vision

Build an AI-powered pharmaceutical knowledge expert that provides accurate, peer-reviewed information exclusively from PubMed research papers, designed for healthcare professionals, researchers, and pharmaceutical companies requiring evidence-based answers with full source attribution.

### Project Status

- **Base Template**: Cloned from `zainulabedeen123/RAG-Template-for-NVIDIA-nemoretriever`
- **Development Environment**: VS Code with Claude Code
- **Current Phase**: Requirements Definition & Architecture Planning

---

## Business Requirements

### Primary Objectives

1. **Exclusive PubMed Integration**: All research data must originate from PubMed API via Apify scraper
2. **Medical Accuracy**: Responses must include proper medical disclaimers and source attribution
3. **Regulatory Compliance**: System must align with FDA/EMA guidelines for medical information dissemination
4. **Production Deployment**: Deploy on Vercel with automated CI/CD pipeline

### Target Users

- **Primary**: Healthcare professionals, clinical researchers, pharmaceutical scientists
- **Secondary**: Academic researchers, medical students, regulatory affairs professionals
- **Tertiary**: Pharmaceutical companies conducting competitive intelligence

### Success Metrics

- **Accuracy**: >95% of responses include valid PubMed source citations
- **Response Time**: <2 minutes average query processing time (allowing for complex pharmaceutical analysis)
- **Coverage**: Support for 1,000+ pharmaceutical research papers minimum (scalable foundation)
- **User Satisfaction**: Medical disclaimer compliance in 100% of responses

---

## Technical Requirements

### Core Architecture (Based on Template)

#### Required Template Modifications

```
src/
├── document_loader.py          # MODIFY: Add PubMed metadata extraction
├── nvidia_embeddings.py        # ENHANCE: Upgrade to llama-3.2-nemoretriever
├── vector_database.py          # ENHANCE: Add pharmaceutical metadata indexing
├── rag_agent.py                # MODIFY: Add medical disclaimers & validation
├── pubmed_scraper.py           # NEW: PubMed API integration via Apify
├── medical_guardrails.py       # NEW: Medical accuracy validation
├── pharmaceutical_processor.py  # NEW: Drug name/compound processing
├── query_engine.py             # NEW: Advanced query processing with caching
├── ranking_filter.py           # NEW: Study ranking and filtering system
├── synthesis_engine.py         # NEW: Meta-summary generation
└── ddi_pk_processor.py         # NEW: Drug interaction and PK/PD analysis
```

#### Template Dependencies to Maintain

- **LangChain**: >=0.1.0 for RAG pipeline
- **FAISS**: >=1.7.0 for vector similarity search
- **Streamlit**: >=1.28.0 for web interface
- **Requests**: >=2.31.0 for API integration

### Data Requirements

#### PubMed Integration Specifications

- **Source**: Apify PubMed Articles Scraper (`scrapestorm/pubmed-articles-scraper`)
- **Initial Budget**: $20/month (~2,000 research papers for MVP)
- **Required Fields**: Title, Authors, DOI, Abstract, Publication Date, PubMed ID, MeSH terms
- **Update Frequency**: Monthly automated scraping for new pharmaceutical research

#### Document Processing Requirements

- **Input Format**: PDF files downloaded from PubMed links
- **Chunking Strategy**: 1000 tokens with 200 token overlap (template default)
- **Metadata Extraction**: DOI, authors, journal impact factor, MeSH terms, study type, sample size
- **Storage**: Local FAISS index with cloud backup capability

### AI Model Requirements

#### NVIDIA Integration (Template-Based)

- **Primary Embedding**: `llama-3.2-nemoretriever-1b-vlm-embed-v1` (2048 dimensions)
- **Fallback Embedding**: `nvidia/nv-embed-v1` (template current)
- **LLM**: `meta/llama-3.1-8b-instruct` for response generation
- **Guardrails**: NVIDIA NeMo Guardrails for medical content safety

#### Custom Medical Enhancements

- **Drug Name Recognition**: Enhanced NER for pharmaceutical compounds
- **Medical Terminology**: Specialized vocabulary for clinical terms
- **Citation Validation**: Automated DOI verification system
- **Bias Detection**: Pharmaceutical industry funding source analysis

---

## Functional Requirements

### Core Features (Template-Enhanced)

#### 1. Document Ingestion System

- **FR-001**: System must download PDFs from PubMed links via Apify scraper
- **FR-002**: Extract and validate DOI, publication date, author credentials
- **FR-003**: Process documents into searchable chunks with pharmaceutical metadata
- **FR-004**: Store in FAISS vector database with persistence

#### 2. Query Processing Engine

- **FR-005**: Accept natural language questions about pharmaceutical topics
- **FR-006**: Retrieve relevant research papers using semantic similarity
- **FR-007**: Generate comprehensive responses with mandatory medical disclaimers
- **FR-008**: Provide full PubMed source attribution for every claim

#### 3. Web Interface (Streamlit-Based)

- **FR-009**: User-friendly chat interface for pharmaceutical queries
- **FR-010**: Document statistics dashboard showing research coverage
- **FR-011**: Source visualization with publication dates and impact factors
- **FR-012**: Export functionality for research citations and chat history

### Advanced Pharmaceutical Features

#### 4. Query → Fetch (Apify-only)

- **FR-013**: **Query Parameters**: Must support keyword, maxitems, sort_by parameters
  - Default sort_by: "Best match"
  - Optional sort_by: "Most recent"
- **FR-014**: **Query Limits**:
  - Default maxitems = 50 (Exploratory mode)
  - Hard cap = 100 items per query
  - Daily limit = 600 items (configurable)
- **FR-015**: **Smart Caching**: 24-hour cache by {keyword, sort_by, mode}
  - Cache is mergeable for incremental expansion when maxitems increases
  - Reduces API costs and improves response times
- **FR-016**: **PII Protection**: Basic redaction before vendor API calls
- **FR-017**: **Deduplication**: Remove duplicates by DOI or PMID across entire dataset

#### 5. Ranking & Filtering System

- **FR-018**: **Study Quality Signals**: Rank studies using multiple criteria:
  - **Recency**: Newer studies weighted higher
  - **Study Hierarchy**: Systematic review/RCT > observational > case report
  - **Species Preference**: Human studies > animal studies
  - **Sample Size**: Larger N weighted higher
  - **Abstract Quality**: Length/structure assessment
  - **MeSH Terms**: Presence of relevant medical subject headings
  - **DDI/PK/PD Cues**: Drug interaction and pharmacokinetic indicators
- **FR-019**: **User Filtering Options**:
  - Year range selection
  - Article type filtering
  - Human vs. animal study selection
  - Minimum sample size (N) threshold
  - Clinical trial phase keywords
  - Top-k results per journal (optional)
- **FR-020**: **Study Diversity**: Ensure top-k results include diverse study types
  - Avoid returning 10 near-duplicate studies
  - Balance different methodologies and perspectives

#### 6. Synthesis & Answering Engine

- **FR-021**: **Meta-Summary Generation**: Create 3-7 bullet points aggregating key findings
  - Example: "Most RCTs (n=3) report AUC↑ ~1.8–2.3× with strong CYP3A4 inhibitors; n≈1.2k across trials"
  - Include quantitative effect sizes when available
- **FR-022**: **Comparative Analysis**: When literature conflicts, present both perspectives
  - Example: "2 studies show ↑ effect, 1 study shows ↔ (no change)"
  - Include study counts for each finding
- **FR-023**: **Per-Claim Citations**: Every claim must include proper citations
  - Format: [PMID: 123..., 456...] or author-year style
  - Link citations to expandable study cards
- **FR-024**: **Exploratory Leads**: Uncited notes appear separately (Exploratory mode only)
  - Clearly labeled as preliminary/uncited information
  - Never co-located with evidence-based claims

#### 7. DDI / PK-PD Specialized Handling

- **FR-025**: **Drug-Drug Interaction Queries**: Prioritize studies with measured effect sizes
  - Show AUC/Cmax/t½ ranges with sample sizes (N)
  - Highlight inhibitor classifications: strong/moderate/weak
  - Include mechanism of interaction when available
- **FR-026**: **Pharmacokinetic-Pharmacodynamic Analysis**: Comprehensive PK-PD summaries include:
  - Model type identification: noncompartmental vs. population PK
  - Key pharmacokinetic parameters reported
  - Study population demographics and characteristics
  - Clinical relevance and dosing implications

#### 8. Medical Safety & Compliance

- **FR-027**: Include FDA-compliant medical disclaimer in all responses
- **FR-028**: Flag potential drug interactions based on research data
- **FR-029**: Validate clinical trial phases and regulatory status
- **FR-030**: Highlight conflicts of interest in research funding

#### 9. Advanced Search Capabilities

- **FR-031**: Search by drug compound names (generic/brand)
- **FR-032**: Filter by publication date, clinical trial phase, study type
- **FR-033**: Cross-reference multiple studies for comprehensive insights
- **FR-034**: Identify research gaps in pharmaceutical literature

---

## Non-Functional Requirements

### Performance Requirements

- **NFR-001**: Query response time <2 minutes for complex pharmaceutical analysis (95th percentile)
- **NFR-002**: Support concurrent users up to 10 simultaneous sessions (MVP capacity)
- **NFR-003**: Vector database loading time <60 seconds on startup
- **NFR-004**: PubMed scraping rate limit compliance (respect API guidelines)
- **NFR-005**: Cache hit ratio >70% for repeated queries within 24 hours

### Security & Compliance

- **NFR-006**: HTTPS encryption for all API communications
- **NFR-007**: API key security via environment variables only
- **NFR-008**: Medical information disclaimers for regulatory compliance
- **NFR-009**: User query logging for audit trail (anonymized)
- **NFR-010**: PII redaction in all external API calls

### Reliability & Availability

- **NFR-011**: 95% uptime during development phase, 99% for production
- **NFR-012**: Automated backup of vector database weekly
- **NFR-013**: Graceful error handling for API failures with user-friendly messages
- **NFR-014**: Basic system health monitoring and error logging

### Scalability & Maintainability

- **NFR-015**: Support for 1,000+ research papers initially (scalable to 50,000+)
- **NFR-016**: Modular architecture for easy component updates
- **NFR-017**: Cloud-ready deployment for future scaling needs
- **NFR-018**: Comprehensive logging for debugging and optimization

---

## Development Phases

### Phase 1: Template Customization & Core Integration

1. **Modify document_loader.py** for PubMed PDF processing with enhanced metadata
2. **Enhance rag_agent.py** with medical disclaimers and pharmaceutical context
3. **Integrate Apify scraper** with query parameters and caching system
4. **Configure NVIDIA APIs** and test embedding functionality
5. **Initial dataset**: Process 100-500 pharmaceutical papers for testing

### Phase 2: Advanced Query & Ranking System

1. **Implement query_engine.py** with caching and deduplication
2. **Build ranking_filter.py** with study quality signals and user filters
3. **Add pharmaceutical_processor.py** for drug name recognition and MeSH processing
4. **Integrate NeMo Guardrails** for medical accuracy validation
5. **Expand dataset**: Scale to 1,000+ pharmaceutical papers

### Phase 3: Synthesis & DDI/PK-PD Features

1. **Develop synthesis_engine.py** for meta-summary generation and comparative analysis
2. **Implement ddi_pk_processor.py** for specialized drug interaction analysis
3. **Add citation management** with per-claim attribution system
4. **Build study diversity algorithms** to prevent near-duplicate results
5. **Enhanced metadata extraction** for clinical trial information

### Phase 4: Web Interface & Production Deployment

1. **Customize Streamlit interface** for pharmaceutical use cases with advanced filtering
2. **Add medical disclaimer components** and regulatory compliance features
3. **Implement user filtering UI** for year range, study type, sample size
4. **Create pharmaceutical analytics dashboard** with study quality metrics
5. **Vercel deployment** with Browser MCP integration for automated testing

---

## Risk Management

### Technical Risks

- **RISK-001**: NVIDIA API rate limits impacting response times
  - _Mitigation_: Implement caching, batch processing, and graceful degradation
- **RISK-002**: PubMed API changes breaking data ingestion
  - _Mitigation_: Version pinning, error handling, and fallback scraping methods
- **RISK-003**: Vector database performance with larger datasets
  - _Mitigation_: Optimize indexing, implement pagination, monitor performance
- **RISK-004**: Complex ranking algorithms affecting response times
  - _Mitigation_: Pre-compute rankings, optimize algorithms, implement caching

### Compliance Risks

- **RISK-005**: Medical information accuracy and liability concerns
  - _Mitigation_: Comprehensive disclaimers, clear limitations, expert consultation
- **RISK-006**: Copyright issues with PubMed content usage
  - _Mitigation_: Fair use compliance, proper attribution, academic research focus
- **RISK-007**: Data privacy and user query handling
  - _Mitigation_: Anonymize logs, secure storage, transparent privacy policy
- **RISK-008**: Regulatory compliance with medical device regulations
  - _Mitigation_: Legal review, FDA guidance consultation, clear disclaimers

---

## Success Criteria

### Minimum Viable Product (MVP)

- [ ] Successfully process 1,000 pharmaceutical research papers
- [ ] Implement basic query→fetch with caching and deduplication
- [ ] Generate meta-summaries with 3-7 key findings per query
- [ ] Deploy functional Streamlit interface on Vercel
- [ ] Include medical disclaimers in all pharmaceutical responses
- [ ] Achieve <2 minute average query response time
- [ ] Support basic DDI queries with effect size reporting

### Full Product Launch

- [ ] Support 5,000+ pharmaceutical research papers
- [ ] Advanced ranking system with study quality signals
- [ ] Complete DDI/PK-PD analysis capabilities
- [ ] User filtering by year, study type, sample size
- [ ] Integration with Browser MCP for automated testing
- [ ] Analytics dashboard with study diversity metrics
- [ ] 99% uptime with monitoring and alerting

### Long-term Goals

- [ ] Multimodal capability for research images and charts
- [ ] Integration with additional medical databases beyond PubMed
- [ ] API endpoints for third-party pharmaceutical applications
- [ ] Machine learning models for drug interaction predictions
- [ ] Scale to 50,000+ research papers with advanced meta-analysis
- [ ] Real-time literature monitoring and alert system

---

## Development Approach

### Methodology

- **Iterative Development**: Build and test incrementally with pharmaceutical domain experts
- **Continuous Integration**: Automated testing and deployment with medical accuracy validation
- **User-Centric Design**: Regular feedback from healthcare professionals
- **Evidence-Based**: All features validated against real pharmaceutical use cases

### Quality Assurance

- **Medical Review**: All pharmaceutical logic reviewed by domain experts
- **Testing Framework**: Unit tests, integration tests, and pharmaceutical-specific test cases
- **Performance Monitoring**: Response time tracking and query optimization
- **Medical Accuracy**: Regular validation of DDI/PK-PD analysis against published research

### Documentation Strategy

- **Technical Documentation**: Architecture decisions, API documentation, deployment guides
- **User Documentation**: Healthcare professional user guides, query examples, interpretation guidelines
- **Medical Documentation**: Disclaimer templates, compliance guidelines, regulatory considerations
- **Research Documentation**: Study ranking methodologies, meta-analysis algorithms, citation formats

---

## Appendices

### A. Template File Structure Analysis

```
Enhanced Template Structure:
├── src/document_loader.py      # PDF processing with pharmaceutical metadata
├── src/nvidia_embeddings.py    # NVIDIA API integration
├── src/vector_database.py      # FAISS vector storage with study rankings
├── src/rag_agent.py           # Main RAG pipeline with medical disclaimers
├── src/query_engine.py        # Query processing with caching/deduplication
├── src/ranking_filter.py      # Study quality ranking and filtering
├── src/synthesis_engine.py    # Meta-summary and comparative analysis
├── src/ddi_pk_processor.py    # Drug interaction and PK/PD analysis
├── streamlit_app.py           # Enhanced web interface
├── requirements.txt           # Core dependencies
├── requirements-medical.txt   # Medical safety validation dependencies (optional)
└── .env.template             # Configuration template
```

### B. Required Environment Variables

```
# NVIDIA API Configuration
NVIDIA_API_KEY=your_nvidia_api_key_here

# Apify PubMed Scraper
APIFY_API_TOKEN=your_apify_token_here

# Query Configuration
DEFAULT_MAX_ITEMS=50
HARD_CAP_MAX_ITEMS=100
DAILY_QUERY_LIMIT=600
CACHE_DURATION_HOURS=24

# Document Processing
DOCS_FOLDER=Data/Pharmaceutical_Papers
VECTOR_DB_PATH=./pharmaceutical_vector_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Deployment
VERCEL_PROJECT_ID=hendrix-majumdar-moreaus-projects
```

### C. Study Quality Ranking Weights

| Ranking Factor   | Weight | Description                                    |
| ---------------- | ------ | ---------------------------------------------- |
| Study Type       | 0.25   | RCT > Systematic Review > Cohort > Case Report |
| Recency          | 0.20   | Exponential decay from publication date        |
| Sample Size      | 0.15   | Log-scaled weighting for larger N              |
| Species          | 0.15   | Human > Primate > Mammal > Other               |
| Journal Impact   | 0.10   | Impact factor normalization                    |
| MeSH Relevance   | 0.10   | Target pharmaceutical term matching            |
| Abstract Quality | 0.05   | Structure and completeness assessment          |

### D. Medical Disclaimer Template

```
IMPORTANT MEDICAL DISCLAIMER:
This information is for educational and research purposes only.
It is not intended as medical advice, diagnosis, or treatment.
Drug interactions and pharmacokinetic data may vary by individual.
Always consult qualified healthcare professionals for medical decisions.
Sources: [PubMed citations with DOI links and study quality indicators]
```

### E. DDI/PK-PD Query Examples

```
Example Queries:
- "What is the drug interaction between warfarin and amiodarone?"
- "How does ketoconazole affect midazolam pharmacokinetics?"
- "What are the CYP3A4 inhibition effects on atorvastatin?"
- "Show me RCTs on drug interactions with over 100 subjects"
- "Recent systematic reviews on DDIs published after 2020"
```

---

**Document Version**: 2.0
**Last Updated**: September 16, 2025
**Next Review**: October 1, 2025

---

Last Updated: 2025-10-03
Owner: Docs
Review Cadence: Quarterly

---
