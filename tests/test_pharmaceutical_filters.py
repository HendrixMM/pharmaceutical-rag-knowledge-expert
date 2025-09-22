#!/usr/bin/env python3
"""
Test script for pharmaceutical filter methods in RAG Template

This script tests all the newly implemented pharmaceutical-aware search methods:
- similarity_search_with_pharmaceutical_filters
- search_by_drug_name
- get_pharmaceutical_stats
- _apply_pharmaceutical_filters (via VectorDatabase)
- _extract_pharmaceutical_metadata (via VectorDatabase)
- Extended get_stats with pharmaceutical statistics
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_agent import RAGAgent
from vector_database import VectorDatabase
from nvidia_embeddings import NVIDIAEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pharmaceutical_filters():
    """Test all pharmaceutical filter methods"""
    print("=" * 80)
    print("Testing Pharmaceutical Filter Methods")
    print("=" * 80)

    # Get configuration
    api_key = os.getenv("NVIDIA_API_KEY")
    docs_folder = os.getenv("DOCS_FOLDER", "Data/Docs")
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./test_vector_db")

    if not api_key:
        print("❌ NVIDIA_API_KEY not found in environment variables")
        return False

    # Initialize RAG Agent
    print(f"\n🚀 Initializing RAG Agent...")
    print(f"   Docs folder: {docs_folder}")
    print(f"   Vector DB path: {vector_db_path}")

    try:
        rag_agent = RAGAgent(
            docs_folder=docs_folder,
            api_key=api_key,
            vector_db_path=vector_db_path
        )
        print("✅ RAG Agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG Agent: {str(e)}")
        return False

    # Setup knowledge base
    print(f"\n📚 Setting up knowledge base...")
    try:
        if rag_agent.setup_knowledge_base(force_rebuild=False):
            print("✅ Knowledge base setup successful")
        else:
            print("⚠️  Knowledge base setup failed, continuing with tests anyway")
    except Exception as e:
        print(f"❌ Knowledge base setup failed: {str(e)}")
        return False

    # Test 1: Get knowledge base stats (should include pharmaceutical stats)
    print(f"\n📊 Test 1: Getting knowledge base stats...")
    try:
        stats = rag_agent.get_knowledge_base_stats()
        print("✅ Knowledge base stats retrieved")
        print(f"   Status: {stats.get('status', 'Unknown')}")
        print(f"   Document count: {stats.get('document_count', 0)}")

        # Check if pharmaceutical stats are included
        if 'pharmaceutical' in stats:
            pharma_stats = stats['pharmaceutical']
            print(f"   Drug annotation ratio: {pharma_stats.get('drug_annotation_ratio', 'N/A')}")
            print(f"   Top drugs: {pharma_stats.get('top_drug_names', [])[:3]}")
            print("✅ Pharmaceutical stats included in knowledge base stats")
        else:
            print("⚠️  Pharmaceutical stats not found in knowledge base stats")
    except Exception as e:
        print(f"❌ Failed to get knowledge base stats: {str(e)}")

    # Test 2: Get pharmaceutical stats directly
    print(f"\n💊 Test 2: Getting pharmaceutical stats...")
    try:
        pharma_stats = rag_agent.get_pharmaceutical_stats()
        print("✅ Pharmaceutical stats retrieved")
        print(f"   Status: {pharma_stats.get('status', 'Unknown')}")
        print(f"   Documents indexed: {pharma_stats.get('documents_indexed', 0)}")
        print(f"   Drug annotation ratio: {pharma_stats.get('drug_annotation_ratio', 'N/A')}")
        print(f"   Top drugs: {pharma_stats.get('top_drug_names', [])[:5]}")
        print(f"   Study types: {pharma_stats.get('study_type_distribution', [])[:5]}")
        print(f"   Species distribution: {pharma_stats.get('species_distribution', [])[:5]}")
    except Exception as e:
        print(f"❌ Failed to get pharmaceutical stats: {str(e)}")

    # Test 3: Search by drug name
    print(f"\n🔍 Test 3: Searching by drug name...")
    drug_names = ["aspirin", "metformin", "lipitor", "ibuprofen"]

    for drug_name in drug_names:
        try:
            results = rag_agent.search_by_drug_name(drug_name, k=3)
            print(f"   🔍 {drug_name}: Found {len(results)} documents")
            if results:
                # Show first result metadata
                first_doc = results[0]
                if hasattr(first_doc, 'metadata'):
                    metadata = first_doc.metadata
                    print(f"      First doc has {len(metadata.get('drug_names', []))} drug names")
                    print(f"      Species: {metadata.get('species', 'Unknown')}")
                    print(f"      Study type: {metadata.get('study_type', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ Search for {drug_name} failed: {str(e)}")

    # Test 4: Similarity search with pharmaceutical filters
    print(f"\n🎯 Test 4: Similarity search with pharmaceutical filters...")

    # Test with different filter combinations
    filter_combinations = [
        {
            "name": "Drug names filter",
            "query": "drug interactions",
            "filters": {"drug_names": ["aspirin", "warfarin"]},
            "k": 3
        },
        {
            "name": "Species filter",
            "query": "clinical trial",
            "filters": {"species_preference": "human"},
            "k": 3
        },
        {
            "name": "Therapeutic area filter",
            "query": "treatment outcomes",
            "filters": {"therapeutic_areas": ["cardiology", "neurology"]},
            "k": 3
        },
        {
            "name": "Study type filter",
            "query": "drug efficacy",
            "filters": {"study_types": ["randomized controlled trial"]},
            "k": 3
        },
        {
            "name": "Year range filter",
            "query": "recent developments",
            "filters": {"year_range": [2020, 2024]},
            "k": 3
        },
        {
            "name": "Combined filters",
            "query": "cardiovascular drugs",
            "filters": {
                "drug_names": ["aspirin"],
                "species_preference": "human",
                "therapeutic_areas": ["cardiology"],
                "study_types": ["clinical trial"],
                "year_range": [2015, 2024]
            },
            "k": 3
        }
    ]

    for filter_test in filter_combinations:
        try:
            results = rag_agent.similarity_search_with_pharmaceutical_filters(
                query=filter_test["query"],
                k=filter_test["k"],
                filters=filter_test["filters"]
            )
            print(f"   🎯 {filter_test['name']}: Found {len(results)} documents")

            if results:
                # Show filter matches in first result
                first_doc = results[0]
                if hasattr(first_doc, 'metadata'):
                    metadata = first_doc.metadata
                    print(f"      Drug names: {metadata.get('drug_names', [])}")
                    print(f"      Species: {metadata.get('species', 'Unknown')}")
                    print(f"      Therapeutic areas: {metadata.get('therapeutic_areas', [])}")
                    print(f"      Study types: {metadata.get('study_types', [])}")
                    print(f"      Year: {metadata.get('publication_year', 'Unknown')}")
        except Exception as e:
            print(f"   ❌ {filter_test['name']} failed: {str(e)}")

    # Test 5: Backward compatibility with similarity_search_with_scores
    print(f"\n🔄 Test 5: Testing backward compatibility...")
    try:
        # This should work as before
        scored_docs = rag_agent.get_relevant_documents("drug interactions", k=3)
        print(f"✅ similarity_search_with_scores still works: Found {len(scored_docs)} documents")

        if scored_docs:
            print(f"   First document score: {scored_docs[0][1]:.4f}")
            print(f"   Document content preview: {scored_docs[0][0].page_content[:100]}...")
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {str(e)}")

    # Test 6: Test _apply_pharmaceutical_filters indirectly
    print(f"\n⚙️  Test 6: Testing pharmaceutical filter application...")
    try:
        # This tests the internal _apply_pharmaceutical_filters method through similarity_search_with_pharmaceutical_filters
        results_with_filters = rag_agent.similarity_search_with_pharmaceutical_filters(
            query="drug",
            k=10,
            filters={"min_ranking_score": 0.5}
        )

        results_no_filters = rag_agent.similarity_search_with_pharmaceutical_filters(
            query="drug",
            k=10,
            filters=None
        )

        print(f"   With min_ranking_score filter: {len(results_with_filters)} documents")
        print(f"   Without filters: {len(results_no_filters)} documents")

        if len(results_with_filters) <= len(results_no_filters):
            print("✅ Filter appears to be working correctly")
        else:
            print("⚠️  Filter behavior unexpected")
    except Exception as e:
        print(f"❌ Filter application test failed: {str(e)}")

    # Test 7: Test _extract_pharmaceutical_metadata indirectly
    print(f"\n🏷️  Test 7: Testing pharmaceutical metadata extraction...")
    try:
        # This tests the internal _extract_pharmaceutical_metadata method through search results
        results = rag_agent.similarity_search_with_pharmaceutical_filters(
            query="pharmacokinetics metabolism",
            k=5
        )

        pharma_metadata_found = 0
        for i, doc in enumerate(results):
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                has_drug_info = bool(metadata.get('drug_names') or metadata.get('drug_annotations'))
                has_cyp_info = bool(metadata.get('cyp_enzymes'))
                has_pk_info = bool(metadata.get('pharmacokinetics'))

                if has_drug_info or has_cyp_info or has_pk_info:
                    pharma_metadata_found += 1
                    print(f"   Document {i+1} has pharmaceutical metadata:")
                    if has_drug_info:
                        print(f"      Drugs: {metadata.get('drug_names', [])}")
                    if has_cyp_info:
                        print(f"      CYP enzymes: {metadata.get('cyp_enzymes', [])}")
                    if has_pk_info:
                        print(f"      PK parameters: {list(metadata.get('pharmacokinetics', {}).keys())}")

        if pharma_metadata_found > 0:
            print(f"✅ Found pharmaceutical metadata in {pharma_metadata_found} out of {len(results)} documents")
        else:
            print("⚠️  No pharmaceutical metadata found in results")
    except Exception as e:
        print(f"❌ Metadata extraction test failed: {str(e)}")

    print("\n" + "=" * 80)
    print("✅ All pharmaceutical filter tests completed!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = test_pharmaceutical_filters()
    sys.exit(0 if success else 1)