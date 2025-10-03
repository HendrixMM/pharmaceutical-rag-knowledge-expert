# Pharmaceutical Benchmarking Datasets

## Overview

This directory contains versioned benchmark datasets for evaluating pharmaceutical RAG query performance, accuracy, and cost efficiency. These benchmarks are designed to test the system's ability to handle domain-specific pharmaceutical queries across multiple categories.

## Dataset Categories

### 1. Drug Interactions (drug_interactions_v\*.json)

Tests the system's ability to identify and explain drug-drug interactions, including:

- Mechanism-based interactions
- Pharmacokinetic interactions
- Pharmacodynamic interactions
- Clinical significance assessment
- Contraindications and warnings

**Query Types:** Comparison, retrieval, reasoning
**Expected Response:** Detailed interaction mechanisms, clinical implications, monitoring requirements

### 2. Pharmacokinetics (pharmacokinetics_v\*.json)

Evaluates understanding of ADME (Absorption, Distribution, Metabolism, Excretion) properties:

- Absorption characteristics and bioavailability
- Distribution volumes and protein binding
- Metabolic pathways (CYP450 enzymes)
- Elimination routes and half-life
- Renal/hepatic impairment adjustments

**Query Types:** Scientific, retrieval
**Expected Response:** Quantitative pharmacokinetic parameters, pathway details

### 3. Clinical Terminology (clinical_terminology_v\*.json)

Tests medical and pharmaceutical terminology comprehension:

- Drug classification systems (ATC, therapeutic classes)
- Medical terminology and abbreviations
- Dosage forms and routes of administration
- Clinical indications and FDA approvals
- Regulatory and formulary terms

**Query Types:** Definition, classification
**Expected Response:** Accurate terminology definitions, proper classifications

### 4. Mechanism of Action (mechanism_of_action_v\*.json)

Assesses understanding of drug mechanisms and molecular targets:

- Receptor interactions and signaling pathways
- Enzyme inhibition/activation
- Ion channel modulation
- Cellular and molecular effects
- Downstream therapeutic effects

**Query Types:** Scientific, reasoning
**Expected Response:** Detailed molecular mechanisms, pathway diagrams (when applicable)

### 5. Adverse Reactions (adverse_reactions_v\*.json)

Evaluates knowledge of drug safety profiles:

- Common and serious adverse effects
- Black box warnings
- Frequency and severity of reactions
- Risk factors and monitoring parameters
- Management strategies

**Query Types:** Safety, retrieval
**Expected Response:** Comprehensive safety information, clinical management guidance

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

## Versioning Strategy

- **v1**: Initial baseline dataset (50 queries per category)
- **v2**: Expanded dataset with edge cases (100 queries per category)
- **v3+**: Continuous additions based on production insights

Versions are immutable once created. New queries are added in new versions to enable regression testing.

## Data Sources

Benchmark queries are derived from:

- FDA-approved drug labels and package inserts
- Clinical pharmacology textbooks
- Peer-reviewed pharmaceutical literature
- Real-world clinical scenarios
- Drug databases (DrugBank, PubChem, RxNorm)

Drug names sourced from:

- `Data/drugs_brand.txt` (345 brand names)
- `Data/drugs_generic.txt` (344 generic names)

## Usage

### Running Benchmarks

```bash
# Run all benchmarks
python scripts/run_pharmaceutical_benchmarks.py

# Run specific category
python scripts/run_pharmaceutical_benchmarks.py --category drug_interactions

# Run specific version
python scripts/run_pharmaceutical_benchmarks.py --version 1

# Generate regression report
python scripts/pharmaceutical_benchmark_report.py --compare v1 v2
```

### Generating New Benchmarks

```bash
# Generate all categories
python scripts/generate_pharmaceutical_benchmarks.py

# Generate specific category
python scripts/generate_pharmaceutical_benchmarks.py --category pharmacokinetics

# Specify output version
python scripts/generate_pharmaceutical_benchmarks.py --version 2
```

## Quality Assurance

All benchmark queries undergo:

1. **Clinical Validation**: Review by pharmaceutical domain experts
2. **Technical Validation**: Ensure expected_content is achievable
3. **Diversity Check**: Balanced coverage across drug classes
4. **Difficulty Calibration**: Mix of straightforward and complex queries

## Evaluation Metrics

Benchmarks track:

- **Accuracy**: Correctness of information (0-1 scale)
- **Completeness**: Coverage of expected_content (0-1 scale)
- **Relevance**: Alignment with query intent (0-1 scale)
- **Cost Efficiency**: Credits per query vs. quality score
- **Response Time**: Latency in milliseconds

### Regression Detection

Regression is flagged when:

- Accuracy drops > 5% from previous version
- Cost per query increases > 20% without quality improvement
- Response time increases > 50% from baseline

## Integration

Benchmarks integrate with:

- **PharmaceuticalCostAnalyzer**: Track credits usage per query type
- **EnhancedNeMoClient**: Execute queries against NeMo models
- **Monitoring Dashboard**: Real-time performance visualization
- **CI/CD Pipeline**: Automated regression testing

## Maintenance

- **Monthly**: Review new drug approvals and update datasets
- **Quarterly**: Generate new benchmark versions
- **Annually**: Major dataset revision with clinical expert review

## Contributing

When adding new benchmark queries:

1. Follow the JSON schema strictly
2. Ensure clinical accuracy
3. Include diverse drug examples
4. Tag appropriately for filtering
5. Update benchmarks_manifest.yaml

## Files in This Directory

- `drug_interactions_v*.json` - Drug interaction benchmarks
- `pharmacokinetics_v*.json` - ADME property benchmarks
- `clinical_terminology_v*.json` - Medical terminology benchmarks
- `mechanism_of_action_v*.json` - Drug mechanism benchmarks
- `adverse_reactions_v*.json` - Safety profile benchmarks
- `benchmarks_manifest.yaml` - Dataset metadata and versioning
- `README.md` - This documentation file

## License

These benchmarks are derived from publicly available pharmaceutical data sources and are intended for research and evaluation purposes only. Not for clinical decision-making.
