# Pharmaceutical Benchmarks

---

Last Updated: 2025-10-03
Owner: Performance Team
Review Cadence: Monthly

---

<!-- TOC -->

- [Overview](#overview)
- [Dataset Categories](#dataset-categories)
- [Dataset Structure](#dataset-structure)
- [Versioning Strategy](#versioning-strategy)
- [Data Sources](#data-sources)
- [Running Benchmarks](#running-benchmarks)
  - [Basic Usage](#basic-usage)
  - [Advanced Options](#advanced-options)
  - [Preset Configurations](#preset-configurations)
- [Benchmark Execution Metadata](#benchmark-execution-metadata)
- [Performance Metrics](#performance-metrics)
- [Generating Reports](#generating-reports)
- [Quality Assurance](#quality-assurance)
- [Evaluation Metrics](#evaluation-metrics)
- [Integration](#integration)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting Benchmark Issues](#troubleshooting-benchmark-issues)
- [Maintenance](#maintenance)
- [Contributing](#contributing)
- [Files Reference](#files-reference)
- [Cross-References](#cross-references)
- [License](#license)

<!-- /TOC -->

## Overview

This directory contains versioned benchmark datasets for evaluating pharmaceutical RAG query performance, accuracy, and cost efficiency. These benchmarks test the system's ability to handle domain-specific pharmaceutical queries across multiple categories.

**Purpose:**
- Evaluate pharmaceutical RAG performance
- Track accuracy and quality metrics
- Monitor cost efficiency
- Detect performance regressions
- Validate pharmaceutical domain knowledge

**Location:** `benchmarks/` directory

## Dataset Categories

### 1. Drug Interactions (drug_interactions_v*.json)

Tests the system's ability to identify and explain drug-drug interactions, including:

- Mechanism-based interactions
- Pharmacokinetic interactions (absorption, metabolism, excretion)
- Pharmacodynamic interactions (additive, synergistic, antagonistic effects)
- Clinical significance assessment
- Contraindications and warnings

**Query Types:** Comparison, retrieval, reasoning
**Expected Response:** Detailed interaction mechanisms, clinical implications, monitoring requirements
**Example Queries:**
- "What are the interaction mechanisms between warfarin and aspirin?"
- "How does rifampin affect the metabolism of oral contraceptives?"
- "Explain the pharmacodynamic interaction between ACE inhibitors and NSAIDs"

### 2. Pharmacokinetics (pharmacokinetics_v*.json)

Evaluates understanding of ADME (Absorption, Distribution, Metabolism, Excretion) properties:

- Absorption characteristics and bioavailability
- Distribution volumes and protein binding
- Metabolic pathways (CYP450 enzymes)
- Elimination routes and half-life
- Renal/hepatic impairment adjustments

**Query Types:** Scientific, retrieval
**Expected Response:** Quantitative pharmacokinetic parameters, pathway details
**Example Queries:**
- "What is the bioavailability and half-life of metoprolol?"
- "Which CYP450 enzymes metabolize simvastatin?"
- "How does renal impairment affect digoxin clearance?"

### 3. Clinical Terminology (clinical_terminology_v*.json)

Tests medical and pharmaceutical terminology comprehension:

- Drug classification systems (ATC, therapeutic classes)
- Medical terminology and abbreviations
- Dosage forms and routes of administration
- Clinical indications and FDA approvals
- Regulatory and formulary terms

**Query Types:** Definition, classification
**Expected Response:** Accurate terminology definitions, proper classifications
**Example Queries:**
- "What is the ATC classification of metformin?"
- "Define the term 'first-pass metabolism'"
- "List common dosage forms for oral medications"

### 4. Mechanism of Action (mechanism_of_action_v*.json)

Assesses understanding of drug mechanisms and molecular targets:

- Receptor interactions and signaling pathways
- Enzyme inhibition/activation
- Ion channel modulation
- Cellular and molecular effects
- Downstream therapeutic effects

**Query Types:** Scientific, reasoning
**Expected Response:** Detailed molecular mechanisms, pathway diagrams (when applicable)
**Example Queries:**
- "Explain the mechanism of action of ACE inhibitors"
- "How do statins lower cholesterol levels?"
- "Describe the receptor targets of beta-blockers"

### 5. Adverse Reactions (adverse_reactions_v*.json)

Evaluates knowledge of drug safety profiles:

- Common and serious adverse effects
- Black box warnings
- Frequency and severity of reactions
- Risk factors and monitoring parameters
- Management strategies

**Query Types:** Safety, retrieval
**Expected Response:** Comprehensive safety information, clinical management guidance
**Example Queries:**
- "What are the black box warnings for warfarin?"
- "List common adverse effects of metformin"
- "How is statin-induced rhabdomyolysis managed?"

## Dataset Structure

Each benchmark dataset follows this JSON schema:

```json
{
  "metadata": {
    "version": "1",
    "category": "drug_interactions|pharmacokinetics|clinical_terminology|mechanism_of_action|adverse_reactions",
    "created_date": "YYYY-MM-DD",
    "total_queries": 50,
    "description": "Dataset description"
  },
  "queries": [
    {
      "id": "category_001",
      "query": "The user's question",
      "expected_type": "comparison|retrieval|scientific|reasoning|safety",
      "expected_content": ["key concept 1", "key concept 2"],
      "evaluation_criteria": {
        "accuracy_weight": 0.4,
        "completeness_weight": 0.3,
        "relevance_weight": 0.3
      },
      "tags": ["tag1", "tag2"]
    }
  ]
}
```

**Validation:**
```bash
# Validate benchmark structure
python scripts/config_validator.py benchmarks/drug_interactions_v1.json
```

## Versioning Strategy

- **v1**: Initial baseline dataset (50 queries per category)
- **v2**: Expanded dataset with edge cases (100 queries per category)
- **v3+**: Continuous additions based on production insights

**Immutability:** Versions are immutable once created. New queries are added in new versions to enable regression testing.

**Version Naming:**
```
drug_interactions_v1.json
drug_interactions_v2.json
pharmacokinetics_v1.json
...
```

## Data Sources

Benchmark queries are derived from:

- FDA-approved drug labels and package inserts
- Clinical pharmacology textbooks
- Peer-reviewed pharmaceutical literature
- Real-world clinical scenarios
- Drug databases (DrugBank, PubChem, RxNorm)

Drug names sourced from:

- [Data/drugs_brand.txt](../Data/drugs_brand.txt) (345 brand names)
- [Data/drugs_generic.txt](../Data/drugs_generic.txt) (344 generic names)

## Running Benchmarks

### Basic Usage

**Run All Benchmarks:**
```bash
python scripts/run_pharmaceutical_benchmarks.py
```

**Expected Output:**
```
🧪 Running Pharmaceutical Benchmarks
=====================================
Category: drug_interactions (v1)
  Progress: 50/50 queries [████████████████] 100%
  Accuracy: 0.87 ± 0.08
  Completeness: 0.82 ± 0.12
  Relevance: 0.89 ± 0.07
  Avg latency: 1,245ms
  Total cost: $0.50

Category: pharmacokinetics (v1)
  Progress: 50/50 queries [████████████████] 100%
  Accuracy: 0.84 ± 0.09
  ...

Overall Performance:
  Total queries: 250
  Overall accuracy: 0.85
  Total cost: $2.50
  Avg cost per query: $0.01
```

**Run Specific Category:**
```bash
python scripts/run_pharmaceutical_benchmarks.py --category drug_interactions

# Output saves to: results/drug_interactions_v1_YYYYMMDD_HHMMSS.json
```

**Run Specific Version:**
```bash
python scripts/run_pharmaceutical_benchmarks.py --version 1

# Runs v1 of all categories
```

**Run with Concurrency:**
```bash
python scripts/run_pharmaceutical_benchmarks.py --concurrency 4

# Process 4 queries in parallel (faster, higher API usage)
```

**Simulation Mode (No API Calls):**
```bash
python scripts/run_pharmaceutical_benchmarks.py --simulate

# Test benchmark infrastructure without consuming API credits
```

### Advanced Options

**Use Preset Configuration:**
```bash
# Quick test (small sample)
python scripts/run_pharmaceutical_benchmarks.py --preset quick-test

# Full production run
python scripts/run_pharmaceutical_benchmarks.py --preset production

# Cost-optimized run
python scripts/run_pharmaceutical_benchmarks.py --preset cost-optimized
```

**List Available Presets:**
```bash
python scripts/run_pharmaceutical_benchmarks.py --list-presets

# Output:
# Available presets:
#   quick-test: Sample 10 queries per category, concurrency=1
#   production: All queries, concurrency=2, full metrics
#   cost-optimized: Batch size 20, caching enabled
#   regression: Compare against baseline
```

**Adaptive Concurrency with Budget:**
```bash
python scripts/run_pharmaceutical_benchmarks.py \
  --adaptive-concurrency \
  --budget 100 \
  --cost-per-query 0.01

# Automatically adjusts concurrency to stay within $100 budget
```

**Compare with Baseline:**
```bash
python scripts/run_pharmaceutical_benchmarks.py \
  --category drug_interactions \
  --baseline results/baseline_v1.json

# Output includes regression detection:
# ⚠️  Accuracy dropped by 7% (regression threshold: 5%)
# ✅ Latency improved by 15%
```

**Validate Classifier Predictions:**
```bash
python scripts/run_pharmaceutical_benchmarks.py \
  --validate-classifier \
  --category all

# Validates pharmaceutical classifier accuracy
```

**Dual Mode (Cloud + Self-Hosted Comparison):**
```bash
python scripts/run_pharmaceutical_benchmarks.py \
  --mode dual \
  --category pharmacokinetics

# Runs benchmarks on both NVIDIA Build and self-hosted NIMs
# Compares performance, accuracy, and cost
```

### Preset Configurations

**Quick Test:**
- Sample: 10 queries per category
- Concurrency: 1
- Use case: Development, smoke testing

**Production:**
- All queries
- Concurrency: 2
- Full metrics collection
- Use case: Pre-release validation

**Cost-Optimized:**
- Batch size: 20
- Caching: Enabled
- Concurrency: 1
- Use case: Free tier conservation

**Regression:**
- Compare against baseline
- All queries
- Detailed diff report
- Use case: CI/CD regression testing

## Benchmark Execution Metadata

For reproducibility, record this metadata with every benchmark run:

### Metadata Template

```json
{
  "execution_metadata": {
    "dataset": "drug_interactions_v1",
    "hardware": {
      "cpu": "Apple M1 Pro",
      "memory": "32GB",
      "gpu": "None (cloud API)"
    },
    "date": "2025-10-03T14:30:00Z",
    "command": "python scripts/run_pharmaceutical_benchmarks.py --category drug_interactions",
    "environment": {
      "python_version": "3.11.5",
      "embedding_model": "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
      "rerank_model": "llama-3_2-nemoretriever-500m-rerank-v2",
      "endpoint": "https://integrate.api.nvidia.com/v1"
    },
    "configuration": {
      "concurrency": 1,
      "batch_size": 10,
      "enable_caching": true,
      "pharmaceutical_mode": true
    }
  },
  "results": {
    "total_queries": 50,
    "avg_accuracy": 0.87,
    "avg_completeness": 0.82,
    "avg_relevance": 0.89,
    "avg_latency_ms": 1250,
    "p50_latency_ms": 1100,
    "p95_latency_ms": 2300,
    "p99_latency_ms": 3100,
    "total_cost_usd": 0.50,
    "cost_per_query": 0.01
  }
}
```

### Hardware Compatibility Matrix

| Hardware | Embedding Speed | Reranking Speed | Memory Usage |
|----------|-----------------|-----------------|--------------|
| M1 Pro (cloud) | 0.8s/query | 0.3s/query | 2GB |
| Intel i7 (cloud) | 1.2s/query | 0.4s/query | 2GB |
| GPU (self-hosted) | 0.2s/query | 0.1s/query | 8GB VRAM |
| CPU (self-hosted) | 3.5s/query | 1.2s/query | 4GB |

## Performance Metrics

### Metrics Tracked

**Accuracy (0-1 scale):**
- Correctness of information
- Factual alignment with expected content
- Detection of hallucinations

**Completeness (0-1 scale):**
- Coverage of expected content points
- Comprehensiveness of response
- Missing critical information

**Relevance (0-1 scale):**
- Alignment with query intent
- Topical relevance
- Context appropriateness

**Overall Score:**
```
overall_score = (accuracy × 0.4) + (completeness × 0.3) + (relevance × 0.3)
```

**Latency Metrics:**
- **p50 (median)**: 50th percentile response time
- **p95**: 95th percentile (outlier threshold)
- **p99**: 99th percentile (worst-case scenario)
- **Mean**: Average response time
- **Stddev**: Standard deviation

**Cost Metrics:**
- Credits/USD per query
- Total cost per benchmark run
- Cost efficiency ratio (score/cost)

**Throughput:**
- Queries per second
- Queries per minute
- Daily query capacity

**Token Usage:**
- Input tokens per query
- Output tokens per query
- Total tokens per benchmark

### Percentile Calculations

```python
import numpy as np

latencies = [1100, 1200, 1150, 1250, 2300, 1180, ...]  # All query latencies

p50 = np.percentile(latencies, 50)  # Median
p95 = np.percentile(latencies, 95)  # 95th percentile
p99 = np.percentile(latencies, 99)  # 99th percentile

print(f"p50: {p50:.0f}ms")
print(f"p95: {p95:.0f}ms")
print(f"p99: {p99:.0f}ms")
```

## Generating Reports

### Regression Report

**Compare Two Benchmark Runs:**
```bash
python scripts/pharmaceutical_benchmark_report.py \
  --compare results/baseline_v1.json results/current_v1.json

# Output: regression_report_YYYYMMDD.html
```

**Expected Output:**
```
📊 Benchmark Comparison Report
==============================

Accuracy:
  Baseline: 0.87 ± 0.08
  Current:  0.82 ± 0.09
  Change:   -5.7% ⚠️  REGRESSION

Latency (p95):
  Baseline: 2100ms
  Current:  1850ms
  Change:   -11.9% ✅ IMPROVEMENT

Cost per Query:
  Baseline: $0.010
  Current:  $0.009
  Change:   -10.0% ✅ IMPROVEMENT

Regression Detected:
  ⚠️  Accuracy dropped below 5% threshold
  ✅ Cost and latency within acceptable ranges
```

### Detailed Report with Charts

```bash
python scripts/benchmarks_report.py \
  --input results/drug_interactions_v1.json \
  --output reports/drug_interactions_report.html

# Generates HTML report with:
# - Accuracy distribution chart
# - Latency percentile graph
# - Cost breakdown
# - Query-level details
```

### Orchestrate Multiple Benchmark Runs

```bash
python scripts/orchestrate_benchmarks.py \
  --categories all \
  --versions 1,2 \
  --output-dir results/

# Runs all categories for versions 1 and 2
# Generates consolidated report
```

**Example Output:**
```
🚀 Orchestrating Benchmarks
===========================

Running: drug_interactions_v1 ✅ (45s, accuracy=0.87)
Running: drug_interactions_v2 ✅ (52s, accuracy=0.85)
Running: pharmacokinetics_v1 ✅ (43s, accuracy=0.84)
...

Summary:
  Total runs: 10
  Success rate: 100%
  Total time: 8m 32s
  Avg accuracy: 0.85
  Reports saved to: results/
```

## Quality Assurance

All benchmark queries undergo:

### 1. Clinical Validation
- Review by pharmaceutical domain experts
- Verification against authoritative sources (FDA labels, clinical guidelines)
- Accuracy of expected content
- Clinical relevance

### 2. Technical Validation
- Ensure expected_content is achievable by RAG system
- Verify query complexity is appropriate
- Test for edge cases and ambiguity
- Validate JSON schema compliance

### 3. Diversity Check
- Balanced coverage across drug classes
- Multiple therapeutic areas represented
- Range of query difficulties (easy, medium, hard)
- Variety of query types (factual, reasoning, comparison)

### 4. Difficulty Calibration
- Mix of straightforward and complex queries
- 40% easy queries (factual retrieval)
- 40% medium queries (synthesis required)
- 20% hard queries (multi-step reasoning)

## Evaluation Metrics

### Quality Metrics

**Accuracy (0-1):**
- 0.9-1.0: Excellent (all facts correct)
- 0.8-0.9: Good (minor inaccuracies)
- 0.7-0.8: Acceptable (some errors)
- <0.7: Poor (major errors or hallucinations)

**Completeness (0-1):**
- 0.9-1.0: Comprehensive (all expected points covered)
- 0.8-0.9: Good (most points covered)
- 0.7-0.8: Partial (key points missing)
- <0.7: Incomplete (significant gaps)

**Relevance (0-1):**
- 0.9-1.0: Highly relevant (directly addresses query)
- 0.8-0.9: Relevant (on-topic with minor tangents)
- 0.7-0.8: Somewhat relevant (partially off-topic)
- <0.7: Irrelevant (doesn't address query)

### Regression Detection

Regression is flagged when:

**Accuracy Regression:**
- Accuracy drops > 5% from previous version
- Example: 0.87 → 0.82 (5.7% drop) ⚠️

**Cost Regression:**
- Cost per query increases > 20% without quality improvement
- Example: $0.01 → $0.013 with same accuracy ⚠️

**Latency Regression:**
- Response time increases > 50% from baseline
- Example: p95 latency 1500ms → 2300ms (53% increase) ⚠️

**Classifier Validation:**
- Classifier accuracy < 90% on query classification
- Example: 85% correct query type predictions ⚠️

## Integration

Benchmarks integrate with:

### PharmaceuticalCostAnalyzer
Track credits usage per query type:
```python
from src.monitoring.pharmaceutical_cost_analyzer import PharmaceuticalCostAnalyzer

analyzer = PharmaceuticalCostAnalyzer()
analyzer.track_benchmark_run(
    category="drug_interactions",
    queries=50,
    total_cost=0.50
)
```

### EnhancedNeMoClient
Execute queries against NeMo models:
```python
from src.enhanced_nemo_client import EnhancedNeMoClient

client = EnhancedNeMoClient()
response = client.query(
    benchmark_query="What are drug interactions?",
    benchmark_mode=True  # Enables detailed metrics
)
```

### Monitoring Dashboard
Real-time performance visualization:
- Live accuracy graphs
- Cost tracking
- Latency distribution
- Regression alerts

### CI/CD Pipeline
Automated regression testing:
```yaml
# .github/workflows/benchmarks.yml
- name: Run Pharmaceutical Benchmarks
  run: |
    python scripts/run_pharmaceutical_benchmarks.py \
      --preset regression \
      --baseline results/main_baseline.json \
      --fail-on-regression
```

## Interpreting Results

### Good Performance Indicators

**Quality:**
- Accuracy > 0.85
- Completeness > 0.80
- Relevance > 0.85
- Overall score > 0.83

**Performance:**
- Latency p50 < 1500ms
- Latency p95 < 2000ms
- Latency p99 < 3000ms

**Cost:**
- Cost per query < $0.02
- Cost efficiency > 40 (score/cost)

**Consistency:**
- Accuracy stddev < 0.10
- Low variance across categories

### Regression Indicators

**Quality Regression:**
- Accuracy drop > 5% from baseline
- Increased hallucination rate
- Lower completeness scores

**Performance Regression:**
- Latency increase > 50% from baseline
- Higher p99 latency (outliers)
- Throughput decrease

**Cost Regression:**
- Cost increase > 20% without quality improvement
- Higher token usage per query
- Inefficient batching

**Classifier Issues:**
- Validation accuracy < 0.90
- Misclassified query types
- Incorrect pharmaceutical entity extraction

### Example Interpretation

```
Results: drug_interactions_v1
=============================
Accuracy:      0.87 ✅ (target: >0.85)
Completeness:  0.82 ✅ (target: >0.80)
Relevance:     0.89 ✅ (target: >0.85)
Latency (p95): 1850ms ✅ (target: <2000ms)
Cost/query:    $0.009 ✅ (target: <$0.02)

Interpretation:
✅ All metrics within acceptable ranges
✅ No regressions detected
✅ Ready for production deployment
```

## Troubleshooting Benchmark Issues

### Issue 1: API Rate Limiting During Benchmarks

**Symptom:** HTTP 429 errors, incomplete benchmark runs

**Solution:**
```bash
# Enable rate limiting
export ENABLE_RATE_LIMITING=true
export MAX_REQUESTS_PER_SECOND=2.5

# Reduce concurrency
python scripts/run_pharmaceutical_benchmarks.py --concurrency 1

# Use preset with conservative settings
python scripts/run_pharmaceutical_benchmarks.py --preset cost-optimized
```

### Issue 2: Memory Issues with Large Datasets

**Symptom:** Out of memory errors, system slowdown

**Solution:**
```bash
# Reduce batch size
export EMBEDDING_BATCH_SIZE=5

# Process categories sequentially
for category in drug_interactions pharmacokinetics clinical_terminology; do
  python scripts/run_pharmaceutical_benchmarks.py --category $category
done

# Enable garbage collection
export ENABLE_AGGRESSIVE_GC=true
```

### Issue 3: Inconsistent Results (High Variance)

**Symptom:** Same query produces different scores across runs

**Solution:**
```bash
# Enable deterministic mode
export DETERMINISTIC_MODE=true

# Set random seed
export RANDOM_SEED=42

# Disable temperature randomness
export LLM_TEMPERATURE=0.0

# Run multiple times and average
python scripts/run_pharmaceutical_benchmarks.py --runs 3 --average
```

### Issue 4: Missing Benchmark Datasets

**Symptom:** FileNotFoundError for benchmark files

**Solution:**
```bash
# Generate missing benchmarks
python scripts/generate_pharmaceutical_benchmarks.py

# Verify benchmark files exist
ls benchmarks/

# Expected files:
#   drug_interactions_v1.json
#   pharmacokinetics_v1.json
#   clinical_terminology_v1.json
#   mechanism_of_action_v1.json
#   adverse_reactions_v1.json
```

### Issue 5: Classifier Validation Failures

**Symptom:** Low classifier accuracy, misclassified queries

**Solution:**
```bash
# Retrain classifier with updated data
python scripts/train_pharmaceutical_classifier.py

# Validate classifier performance
python scripts/run_pharmaceutical_benchmarks.py \
  --validate-classifier \
  --category all

# Check expected output:
# Classifier accuracy: 0.92 ✅ (target: >0.90)
```

## Maintenance

### Monthly Tasks
- Review new drug approvals and update datasets
- Check for outdated drug information
- Add queries for newly identified edge cases
- Update drug name lists from FDA

### Quarterly Tasks
- Generate new benchmark versions
- Run regression tests against all versions
- Update expected content based on new literature
- Review and update quality thresholds

### Annual Tasks
- Major dataset revision with clinical expert review
- Comprehensive coverage analysis
- Difficulty recalibration
- Remove deprecated drugs
- Add emerging therapeutic areas

## Contributing

When adding new benchmark queries:

1. **Follow JSON Schema:**
   ```json
   {
     "id": "unique_id",
     "query": "Clear, specific question",
     "expected_type": "comparison|retrieval|scientific|reasoning|safety",
     "expected_content": ["concept1", "concept2"],
     "evaluation_criteria": {
       "accuracy_weight": 0.4,
       "completeness_weight": 0.3,
       "relevance_weight": 0.3
     },
     "tags": ["drug_class", "therapeutic_area"]
   }
   ```

2. **Ensure Clinical Accuracy:**
   - Verify against FDA-approved labels
   - Cross-reference with multiple authoritative sources
   - Review by pharmaceutical domain expert

3. **Include Diverse Drug Examples:**
   - Cover multiple therapeutic classes
   - Include brand and generic names
   - Represent various mechanisms of action

4. **Tag Appropriately:**
   - Drug classes (anticoagulants, statins, etc.)
   - Therapeutic areas (cardiology, oncology, etc.)
   - Query difficulty (easy, medium, hard)
   - Special populations (geriatric, pediatric, renal impairment)

5. **Update Manifest:**
   ```bash
   # Update benchmarks_manifest.yaml
   vim benchmarks/benchmarks_manifest.yaml

   # Add new version entry:
   drug_interactions:
     - version: 2
       created: 2025-10-03
       total_queries: 100
       changes: "Added 50 new queries covering emerging drug interactions"
   ```

## Files Reference

**Benchmark Datasets:**
- `benchmarks/drug_interactions_v*.json` - Drug interaction benchmarks
- `benchmarks/pharmacokinetics_v*.json` - ADME property benchmarks
- `benchmarks/clinical_terminology_v*.json` - Medical terminology benchmarks
- `benchmarks/mechanism_of_action_v*.json` - Drug mechanism benchmarks
- `benchmarks/adverse_reactions_v*.json` - Safety profile benchmarks

**Scripts:**
- [scripts/run_pharmaceutical_benchmarks.py](../scripts/run_pharmaceutical_benchmarks.py) - Main benchmark runner
- [scripts/generate_pharmaceutical_benchmarks.py](../scripts/generate_pharmaceutical_benchmarks.py) - Generate new benchmarks
- [scripts/pharmaceutical_benchmark_report.py](../scripts/pharmaceutical_benchmark_report.py) - Generate reports
- [scripts/benchmarks_report.py](../scripts/benchmarks_report.py) - Detailed HTML reports
- [scripts/orchestrate_benchmarks.py](../scripts/orchestrate_benchmarks.py) - Orchestrate multiple runs

**Data Sources:**
- [Data/drugs_brand.txt](../Data/drugs_brand.txt) - 345 brand names
- [Data/drugs_generic.txt](../Data/drugs_generic.txt) - 344 generic names

**Configuration:**
- `benchmarks/benchmarks_manifest.yaml` - Dataset metadata and versioning

**Documentation:**
- `benchmarks/README.md` - Original benchmark documentation (consolidated into this file)
- [docs/BENCHMARKS.md](./BENCHMARKS.md) - This comprehensive guide

## Cross-References

### Related Documentation
- [API Reference](API_REFERENCE.md) - Configuration for benchmark execution
- [Examples](EXAMPLES.md) - Code examples for running benchmarks
- [Development Guide](DEVELOPMENT.md) - Testing integration
- [Troubleshooting Guide](TROUBLESHOOTING_GUIDE.md) - Diagnostic procedures

### Architecture
- [Architecture Documentation](ARCHITECTURE.md) - System design
- [ADR-0001: NeMo Retriever Adoption](adr/0001-use-nemo-retriever.md) - Technology decision

### Operations
- [Free Tier Maximization](FREE_TIER_MAXIMIZATION.md) - Cost optimization
- [Cheapest Deployment](CHEAPEST_DEPLOYMENT.md) - Budget deployment

### Pharmaceutical Domain
- [Pharmaceutical Best Practices](PHARMACEUTICAL_BEST_PRACTICES.md) - Domain guidelines
- [Features](FEATURES.md) - Pharmaceutical features

## License

These benchmarks are derived from publicly available pharmaceutical data sources and are intended for research and evaluation purposes only.

**⚠️ NOT FOR CLINICAL DECISION-MAKING**

All benchmark queries and expected responses are for system evaluation only. Do not use for:
- Clinical diagnosis
- Treatment decisions
- Drug prescribing
- Patient care

Consult licensed healthcare professionals for medical advice.

---

**Last Validated:** 2025-10-03
**Benchmark Count:** 250+ queries (5 categories × 50 queries × 1+ versions)
**Coverage:** Drug interactions, pharmacokinetics, clinical terminology, mechanisms, adverse reactions
**Quality:** Clinically validated, peer-reviewed sources
