---
Last Updated: 2025-10-03
Owner: Docs
Review Cadence: Quarterly
---

# Summary of Changes to RAG Template for NVIDIA NemoRetriever

## Overview

The project has been significantly enhanced with medical/pharmaceutical features, including PubMed integration, drug interaction analysis, and medical safety guardrails.

## Major Enhancements

### 1. Medical/Pharmaceutical Features

- Added PubMed scraping and integration capabilities
- Implemented drug interaction (DDI) and pharmacokinetic (PK) analysis
- Added medical safety guardrails using NVIDIA NeMo Guardrails
- Enhanced document processing for medical literature
- Added pharmaceutical query enhancement capabilities

### 2. New Configuration Options

- Extensive environment variables for PubMed scraping
- Medical safety features configuration
- Rate limiting for API calls
- Enhanced embedding model management

### 3. New Source Files

- `ddi_pk_processor.py`: Drug-Drug Interaction and Pharmacokinetic analysis
- `enhanced_pubmed_scraper.py`: Enhanced PubMed scraping capabilities
- `medical_guardrails.py`: Medical safety validation features
- `paper_schema.py`: Schema for academic papers
- `pharmaceutical_processor.py`: Pharmaceutical entity processing
- `pharmaceutical_query_adapter.py`: Query enhancement for pharmaceutical terms
- `pubmed_scraper.py`: PubMed scraping functionality
- `query_engine.py`: Enhanced query processing
- `ranking_filter.py`: Study ranking and filtering
- `rate_limiting.py`: API rate limiting
- `synthesis_engine.py`: Information synthesis from multiple sources

### 4. New Data Files

- `cyp_roles.csv`: Cytochrome P450 enzyme roles
- `drugs_brand.txt`: Brand name drug lexicon
- `drugs_generic.txt`: Generic drug name lexicon
- `mesh_therapeutic_areas.json`: MeSH therapeutic area mappings

### 5. New Test Files

- Comprehensive test suite for all new medical features
- Tests for PubMed scraping, drug processing, medical guardrails, etc.

### 6. Documentation

- Extensive updates to README.md
- New SETUP_GUIDE.md with medical features
- WEB_INTERFACE_GUIDE.md with medical safety features
- Implementation summaries and optimization guides

## Key Configuration Changes

- Added extensive PubMed scraping configuration options
- Enhanced embedding model configuration with fallback support
- Medical safety features configuration
- Rate limiting configuration for NCBI compliance

---

Last Updated: 2025-10-03
Owner: Docs
Review Cadence: Quarterly

---
