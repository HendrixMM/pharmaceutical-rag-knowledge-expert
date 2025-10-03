#!/usr/bin/env python3
"""
Pharmaceutical Benchmark Generator

Generates versioned benchmark datasets for pharmaceutical queries.
Supports all five categories: drug interactions, pharmacokinetics, clinical terminology,
mechanism of action, and adverse reactions.

Usage:
    python scripts/generate_pharmaceutical_benchmarks.py
    python scripts/generate_pharmaceutical_benchmarks.py --category drug_interactions
    python scripts/generate_pharmaceutical_benchmarks.py --version 2
    python scripts/generate_pharmaceutical_benchmarks.py --output benchmarks/
"""
import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DrugDataLoader:
    """Loads drug names from project data files."""

    def __init__(self, data_dir: str = "Data"):
        self.data_dir = Path(data_dir)
        self.brand_drugs: List[str] = []
        self.generic_drugs: List[str] = []

    def load_drugs(self) -> None:
        """Load brand and generic drug names from data files."""
        # Load brand names
        brand_file = self.data_dir / "drugs_brand.txt"
        if brand_file.exists():
            with open(brand_file) as f:
                self.brand_drugs = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            logger.info(f"Loaded {len(self.brand_drugs)} brand drug names")

        # Load generic names
        generic_file = self.data_dir / "drugs_generic.txt"
        if generic_file.exists():
            with open(generic_file) as f:
                self.generic_drugs = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
            logger.info(f"Loaded {len(self.generic_drugs)} generic drug names")

    def get_random_drug(self, drug_type: str = "any") -> str:
        """Get a random drug name."""
        if drug_type == "brand":
            return random.choice(self.brand_drugs) if self.brand_drugs else "Drug"
        elif drug_type == "generic":
            return random.choice(self.generic_drugs) if self.generic_drugs else "drug"
        else:
            all_drugs = self.brand_drugs + self.generic_drugs
            return random.choice(all_drugs) if all_drugs else "Drug"


class BenchmarkGenerator:
    """Generates pharmaceutical benchmark datasets."""

    CATEGORIES = [
        "drug_interactions",
        "pharmacokinetics",
        "clinical_terminology",
        "mechanism_of_action",
        "adverse_reactions",
    ]

    def __init__(self, output_dir: str = "benchmarks", seed: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Optional random seed for reproducibility
        self.seed: Optional[int] = seed
        if seed is not None:
            try:
                random.seed(seed)  # Seed global RNG used by DrugDataLoader
            except Exception:
                pass
        # Local RNG
        self.rnd: random.Random = random.Random(seed)
        self.drug_loader = DrugDataLoader()
        self.drug_loader.load_drugs()

    def generate_baselines(self, category: str) -> Dict[str, Any]:
        """
        Generate baseline performance metrics for benchmark category.

        These are default/placeholder baselines that should be updated
        after running actual baseline performance measurements.
        """
        return {
            "cloud": {
                "average_latency_ms": 450.0,
                "success_rate": 0.98,
                "average_cost_per_query": 12.5,
                "average_accuracy": 0.85,
                "notes": "Based on NVIDIA Build cloud endpoints (integrate.api.nvidia.com). Update after baseline run.",
            },
            "self_hosted": {
                "average_latency_ms": 850.0,
                "success_rate": 0.95,
                "average_cost_per_query": 0.0,
                "average_accuracy": 0.82,
                "notes": "Based on local NIM containers with GPU acceleration. Update after baseline run.",
            },
            "regression_thresholds": {
                "accuracy_drop_percent": 5,
                "cost_increase_percent": 20,
                "latency_increase_percent": 50,
            },
            "classifier_validation": {
                "overall_accuracy": 0.95,
                "notes": "Expected PharmaceuticalQueryClassifier accuracy for domain, safety urgency, and research priority",
            },
        }

    def generate_metadata(self, category: str, version: int, total_queries: int) -> Dict[str, Any]:
        """Generate metadata for a benchmark dataset."""
        descriptions = {
            "drug_interactions": "Drug interaction queries covering mechanism-based, pharmacokinetic, and pharmacodynamic interactions",
            "pharmacokinetics": "ADME queries covering absorption, distribution, metabolism, and excretion parameters",
            "clinical_terminology": "Terminology queries covering drug classifications, medical terms, and pharmaceutical concepts",
            "mechanism_of_action": "Mechanism queries covering receptor interactions, enzyme targets, and signaling pathways",
            "adverse_reactions": "Adverse reaction queries covering common and serious side effects, warnings, and monitoring",
        }

        meta = {
            "version": str(version),
            "category": category,
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "total_queries": total_queries,
            "description": descriptions.get(category, "Pharmaceutical benchmark queries"),
            "baselines": self.generate_baselines(category),
        }
        if getattr(self, "seed", None) is not None:
            meta["generation_seed"] = int(self.seed)  # type: ignore[arg-type]
        return meta

    def _extract_drug_names_from_query(self, query_text: str) -> List[str]:
        """Extract drug names from query text by matching against drug list."""
        query_lower = query_text.lower()
        found_drugs = []

        # Check against loaded drugs (both brand and generic)
        all_drugs = self.drug_loader.brand_drugs + self.drug_loader.generic_drugs
        for drug in all_drugs:
            if drug.lower() in query_lower:
                found_drugs.append(drug)

        return found_drugs

    def generate_classification(
        self, category: str, query_text: str, expected_type: str, drug_names_in_query: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate expected_classification for a query.

        Args:
            category: Benchmark category (determines domain)
            query_text: Query text (for extracting drug names)
            expected_type: Expected response type (safety, comparison, etc.)
            drug_names_in_query: Explicit drug names if known

        Returns:
            Dictionary with domain, safety_urgency, research_priority, drug_names
        """
        # Map category to domain
        domain_mapping = {
            "drug_interactions": "drug_interactions",
            "pharmacokinetics": "pharmacokinetics",
            "clinical_terminology": "general_research",  # Most terminology is general
            "mechanism_of_action": "mechanism_of_action",
            "adverse_reactions": "adverse_reactions",
        }

        # Determine safety urgency based on expected_type and category
        safety_urgency = "low"  # default
        if expected_type == "safety" or category in ["adverse_reactions", "drug_interactions"]:
            safety_urgency = "high"
        elif expected_type == "definition":
            safety_urgency = "none"

        # Determine research priority
        research_priority = "normal"  # default
        if safety_urgency in ["critical", "high"]:
            research_priority = "high"
        elif expected_type == "definition":
            research_priority = "background"

        # Extract drug names from query if not provided
        if drug_names_in_query is None:
            drug_names_in_query = self._extract_drug_names_from_query(query_text)

        return {
            "domain": domain_mapping.get(category, "general_research"),
            "safety_urgency": safety_urgency,
            "research_priority": research_priority,
            "drug_names": drug_names_in_query,
        }

    def generate_query_template(
        self,
        query_id: str,
        query_text: str,
        expected_type: str,
        expected_content: List[str],
        tags: List[str],
        category: str = "general_research",
        drug_names: Optional[List[str]] = None,
        accuracy_weight: float = 0.4,
        completeness_weight: float = 0.3,
        relevance_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """Generate a query entry template."""
        # Generate classification
        classification = self.generate_classification(
            category=category, query_text=query_text, expected_type=expected_type, drug_names_in_query=drug_names
        )

        return {
            "id": query_id,
            "query": query_text,
            "expected_type": expected_type,
            "expected_content": expected_content,
            "expected_classification": classification,
            "evaluation_criteria": {
                "accuracy_weight": accuracy_weight,
                "completeness_weight": completeness_weight,
                "relevance_weight": relevance_weight,
            },
            "tags": tags,
        }

    def generate_sample_queries(self, category: str, count: int = 10) -> List[Dict[str, Any]]:
        """Generate varied sample queries for a category.

        Each iteration selects a different template and substitutes randomized
        brand/generic drug names (and therapeutic areas when relevant) to ensure
        the resulting list has varied question texts and populated expected metadata.
        """
        queries: List[Dict[str, Any]] = []

        # Therapeutic areas used to add variety when relevant
        therapeutic_areas = [
            "oncology",
            "cardiology",
            "neurology",
            "psychiatry",
            "endocrinology",
            "infectious disease",
            "dermatology",
            "gastroenterology",
            "nephrology",
            "pulmonology",
        ]

        rnd = random.Random()

        def rand_brand() -> str:
            return self.drug_loader.get_random_drug("brand")

        def rand_generic() -> str:
            return self.drug_loader.get_random_drug("generic")

        def rand_any() -> str:
            return self.drug_loader.get_random_drug("any")

        def rand_ta() -> str:
            return rnd.choice(therapeutic_areas)

        # Template definitions per category
        templates: Dict[str, List[Dict[str, Any]]] = {
            "drug_interactions": [
                {
                    "text": "What are the interactions between {drug1} and {drug2}?",
                    "type": "comparison",
                    "content": ["interaction mechanism", "clinical significance"],
                    "tags": ["interaction", "pharmacokinetic"],
                    "drug_count": 2,
                },
                {
                    "text": "Is it safe to combine {drug1} with {drug2} in {ta} patients?",
                    "type": "safety",
                    "content": ["contraindications", "monitoring"],
                    "tags": ["interaction", "pharmacodynamic"],
                    "drug_count": 2,
                    "needs_ta": True,
                },
                {
                    "text": "How does grapefruit juice affect {drug1}?",
                    "type": "safety",
                    "content": ["CYP3A4", "exposure", "myopathy"],
                    "tags": ["food-drug", "CYP3A4"],
                    "drug_count": 1,
                },
                {
                    "text": "What are known interactions between {brand} and {generic}?",
                    "type": "retrieval",
                    "content": ["dose adjustment", "mechanism"],
                    "tags": ["interaction", "monitoring"],
                    "brand_generic": True,
                },
            ],
            "pharmacokinetics": [
                {
                    "text": "What is the half-life of {drug1}?",
                    "type": "scientific",
                    "content": ["half-life", "elimination"],
                    "tags": ["pharmacokinetics", "elimination"],
                    "drug_count": 1,
                },
                {
                    "text": "Summarize metabolism and clearance for {drug1} in {ta} patients.",
                    "type": "scientific",
                    "content": ["metabolism", "clearance", "special population"],
                    "tags": ["ADME", "population"],
                    "drug_count": 1,
                    "needs_ta": True,
                },
                {
                    "text": "What factors affect the bioavailability of {drug1}?",
                    "type": "reasoning",
                    "content": ["bioavailability", "food", "formulation"],
                    "tags": ["absorption", "bioavailability"],
                    "drug_count": 1,
                },
            ],
            "clinical_terminology": [
                {
                    "text": "What does 'BID' mean in prescription writing?",
                    "type": "definition",
                    "content": ["twice daily", "dosing"],
                    "tags": ["abbreviation", "prescription"],
                },
                {
                    "text": "Define the term 'contraindication' and give an example.",
                    "type": "definition",
                    "content": ["contraindication", "example"],
                    "tags": ["terminology", "safety"],
                },
                {
                    "text": "What is an 'ACE inhibitor' and list one representative drug.",
                    "type": "definition",
                    "content": ["class", "example"],
                    "tags": ["classification", "cardiology"],
                },
            ],
            "mechanism_of_action": [
                {
                    "text": "How does {drug1} work at the molecular level?",
                    "type": "scientific",
                    "content": ["mechanism", "molecular target"],
                    "tags": ["mechanism", "molecular"],
                    "drug_count": 1,
                },
                {
                    "text": "Describe the receptor interactions involved in {drug1}'s action.",
                    "type": "scientific",
                    "content": ["receptor", "affinity"],
                    "tags": ["receptor", "binding"],
                    "drug_count": 1,
                },
                {
                    "text": "What signaling pathways are modulated by {drug1}?",
                    "type": "scientific",
                    "content": ["pathway", "downstream effects"],
                    "tags": ["pathway", "signal transduction"],
                    "drug_count": 1,
                },
            ],
            "adverse_reactions": [
                {
                    "text": "What are the common side effects of {drug1}?",
                    "type": "safety",
                    "content": ["side effects", "adverse events"],
                    "tags": ["safety", "adverse effects"],
                    "drug_count": 1,
                },
                {
                    "text": "List serious adverse reactions associated with {drug1} in {ta} patients.",
                    "type": "safety",
                    "content": ["serious AE", "monitoring"],
                    "tags": ["black box", "monitoring"],
                    "drug_count": 1,
                    "needs_ta": True,
                },
                {
                    "text": "What warnings and precautions are noted for {drug1}?",
                    "type": "safety",
                    "content": ["warnings", "precautions"],
                    "tags": ["label", "safety"],
                    "drug_count": 1,
                },
            ],
        }

        cat_templates = templates.get(category, [])
        if not cat_templates:
            return queries

        # Build a sequence of templates ensuring variety: cycle through shuffled templates
        seq: List[Dict[str, Any]] = []
        while len(seq) < count:
            shuffled = cat_templates[:]
            rnd.shuffle(shuffled)
            seq.extend(shuffled)
        seq = seq[:count]

        for i, tpl in enumerate(seq, start=1):
            query_id = f"{category[:3]}_{i:03d}"

            # Choose drugs according to template
            drug_names: List[str] = []
            placeholders: Dict[str, str] = {}
            if tpl.get("brand_generic"):
                b = rand_brand()
                g = rand_generic()
                placeholders.update({"brand": b, "generic": g})
                drug_names = [b, g]
            else:
                dcount = int(tpl.get("drug_count", 0))
                if dcount >= 1:
                    d1 = rand_any()
                    drug_names.append(d1)
                    placeholders["drug1"] = d1
                if dcount >= 2:
                    # ensure distinct drugs if possible
                    tries = 0
                    d2 = rand_any()
                    while d2 == drug_names[0] and tries < 5:
                        d2 = rand_any()
                        tries += 1
                    drug_names.append(d2)
                    placeholders["drug2"] = d2

            if tpl.get("needs_ta"):
                placeholders["ta"] = rand_ta()

            query_text = tpl["text"].format(**placeholders)

            # Expected content and tags
            expected_content = list(tpl.get("content", []))
            tags = list(tpl.get("tags", []))
            expected_type = tpl.get("type", "retrieval")

            # Slight randomization of weights for variety where safety gets higher accuracy by default
            acc_w, comp_w, rel_w = 0.4, 0.3, 0.3
            if expected_type == "safety":
                acc_w = 0.5

            entry = self.generate_query_template(
                query_id=query_id,
                query_text=query_text,
                expected_type=expected_type,
                expected_content=expected_content,
                tags=tags,
                category=category,
                drug_names=drug_names,
                accuracy_weight=acc_w,
                completeness_weight=comp_w,
                relevance_weight=rel_w,
            )

            queries.append(entry)

        return queries

    def generate_benchmark(self, category: str, version: int = 1, num_queries: int = 50) -> Dict[str, Any]:
        """Generate a complete benchmark dataset."""
        if category not in self.CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Must be one of {self.CATEGORIES}")

        logger.info(f"Generating {category} benchmark version {version} with {num_queries} queries")

        metadata = self.generate_metadata(category, version, num_queries)
        queries = self.generate_sample_queries(category, num_queries)

        return {"metadata": metadata, "queries": queries}

    def save_benchmark(self, benchmark: Dict[str, Any], category: str, version: int) -> Path:
        """Save benchmark to JSON file."""
        filename = f"{category}_v{version}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(benchmark, f, indent=2)

        logger.info(f"Saved benchmark to {filepath}")
        return filepath

    def generate_all_categories(self, version: int = 1, num_queries: int = 50) -> List[Path]:
        """Generate benchmarks for all categories."""
        generated_files = []

        for category in self.CATEGORIES:
            try:
                benchmark = self.generate_benchmark(category, version, num_queries)
                filepath = self.save_benchmark(benchmark, category, version)
                generated_files.append(filepath)
            except Exception as e:
                logger.error(f"Error generating {category}: {e}")

        return generated_files


def main():
    """Main entry point for benchmark generation."""
    parser = argparse.ArgumentParser(description="Generate pharmaceutical benchmark datasets")
    parser.add_argument(
        "--category",
        choices=BenchmarkGenerator.CATEGORIES + ["all"],
        default="all",
        help="Benchmark category to generate",
    )
    parser.add_argument("--version", type=int, default=1, help="Benchmark version number")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of queries to generate")
    parser.add_argument("--output", default="benchmarks", help="Output directory for benchmark files")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible dataset generation")

    args = parser.parse_args()

    generator = BenchmarkGenerator(output_dir=args.output, seed=getattr(args, "seed", None))

    try:
        if args.category == "all":
            logger.info("Generating benchmarks for all categories")
            files = generator.generate_all_categories(version=args.version, num_queries=args.num_queries)
            logger.info(f"Generated {len(files)} benchmark files")
            for file in files:
                print(f"  - {file}")
        else:
            logger.info(f"Generating benchmark for category: {args.category}")
            benchmark = generator.generate_benchmark(
                category=args.category, version=args.version, num_queries=args.num_queries
            )
            filepath = generator.save_benchmark(benchmark, args.category, args.version)
            print(f"Generated: {filepath}")

        logger.info("Benchmark generation completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Benchmark generation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
