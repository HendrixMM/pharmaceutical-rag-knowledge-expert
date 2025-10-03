#!/usr/bin/env python3
"""
Example usage of the pharmaceutical filter methods in RAG Template

This script demonstrates how to use the newly implemented pharmaceutical-aware search methods.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_agent import RAGAgent


def main():
    """Example usage of pharmaceutical filter methods"""

    # Initialize RAG Agent
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Please set NVIDIA_API_KEY environment variable")
        return

    rag_agent = RAGAgent(docs_folder="Data/Docs", api_key=api_key, vector_db_path="./example_vector_db")

    # Setup knowledge base
    if not rag_agent.setup_knowledge_base():
        print("Failed to setup knowledge base")
        return

    print("=== Pharmaceutical Search Examples ===\n")

    # Example 1: Basic similarity search with pharmaceutical filters
    print("1. Search for drug interactions with human studies only:")
    results = rag_agent.similarity_search_with_pharmaceutical_filters(
        query="drug interactions side effects",
        k=5,
        filters={"species_preference": "human", "study_types": ["clinical trial", "randomized controlled trial"]},
    )
    print(f"   Found {len(results)} human clinical studies\n")

    # Example 2: Search by specific drug
    print("2. Search for documents about metformin:")
    metformin_docs = rag_agent.search_by_drug_name("metformin", k=3)
    print(f"   Found {len(metformin_docs)} documents about metformin")
    if metformin_docs:
        first_doc = metformin_docs[0]
        print(f"   First document mentions {len(first_doc.metadata.get('drug_names', []))} drugs\n")

    # Example 3: Filter by therapeutic area and year range
    print("3. Search for cardiovascular studies from 2020-2024:")
    cardio_results = rag_agent.similarity_search_with_pharmaceutical_filters(
        query="treatment outcomes",
        k=5,
        filters={"therapeutic_areas": ["cardiology"], "year_range": [2020, 2024], "min_ranking_score": 0.7},
    )
    print(f"   Found {len(cardio_results)} recent cardiovascular studies\n")

    # Example 4: Get pharmaceutical statistics
    print("4. Pharmaceutical statistics:")
    stats = rag_agent.get_pharmaceutical_stats()
    print(f"   Drug annotation ratio: {stats.get('drug_annotation_ratio', 'N/A')}")
    print(f"   Top 3 drugs: {[drug[0] for drug in stats.get('top_drug_names', [])[:3]]}")
    print(f"   Most common species: {[species[0] for species in stats.get('species_distribution', [])[:3]]}")
    print()

    # Example 5: Complex filter with multiple criteria
    print("5. Complex search: diabetes drugs in human trials:")
    complex_results = rag_agent.similarity_search_with_pharmaceutical_filters(
        query="glycemic control",
        k=5,
        filters={
            "drug_names": ["metformin", "insulin", "glipizide"],
            "species_preference": "human",
            "therapeutic_areas": ["endocrinology"],
            "study_types": ["clinical trial"],
            "year_range": [2018, 2024],
            "include_unknown_species": False,
        },
    )
    print(f"   Found {len(complex_results)} matching documents")

    # Show details of first result
    if complex_results:
        first = complex_results[0]
        meta = first.metadata
        print(f"\n   First result details:")
        print(f"   - Drugs: {meta.get('drug_names', [])}")
        print(f"   - Species: {meta.get('species', 'Unknown')}")
        print(f"   - Study type: {meta.get('study_type', 'Unknown')}")
        print(f"   - Year: {meta.get('publication_year', 'Unknown')}")
        print(f"   - Ranking score: {meta.get('ranking_score', 'N/A')}")

    print("\n=== Examples completed! ===")


if __name__ == "__main__":
    main()
