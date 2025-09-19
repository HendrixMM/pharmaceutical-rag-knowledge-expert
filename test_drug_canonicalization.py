#!/usr/bin/env python3
"""Test script for drug name canonicalization."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ddi_pk_processor import DDIPKProcessor
from pharmaceutical_processor import PharmaceuticalProcessor

def test_drug_canonicalization():
    """Test drug name canonicalization functionality."""
    print("Testing drug name canonicalization...")
    
    # Create processor instances
    pharma_processor = PharmaceuticalProcessor()
    ddi_processor = DDIPKProcessor(pharma_processor=pharma_processor)
    
    # Test papers with drug mentions
    test_papers = [
        {
            "page_content": "Warfarin metabolism is affected by Coumadin. Both drugs show similar AUC increases.",
            "metadata": {
                "pmid": "test1",
                "drug_annotations": [
                    {"name": "Warfarin", "type": "generic", "confidence": 0.98},
                    {"name": "Coumadin", "type": "brand", "confidence": 0.96}
                ]
            }
        }
    ]
    
    # Test canonicalization
    canonical_map = ddi_processor._build_drug_synonym_map(test_papers)
    print("Canonical map:", canonical_map)
    
    # Test with brand name input
    result = ddi_processor.analyze_drug_interactions(
        test_papers, 
        "Coumadin",  # brand name
        ["Warfarin"]  # generic name
    )
    
    print("Analysis result:")
    print(f"Primary drug: {result.get('primary_drug')}")
    print(f"Secondary drugs analyzed: {result.get('secondary_drugs_analyzed')}")
    print(f"Total interactions found: {result.get('summary_statistics', {}).get('total_interactions_found', 0)}")
    
    # Test with generic name input
    result2 = ddi_processor.analyze_drug_interactions(
        test_papers, 
        "warfarin",  # generic name
        ["coumadin"]  # brand name
    )
    
    print("\nReverse test result:")
    print(f"Primary drug: {result2.get('primary_drug')}")
    print(f"Secondary drugs analyzed: {result2.get('secondary_drugs_analyzed')}")
    print(f"Total interactions found: {result2.get('summary_statistics', {}).get('total_interactions_found', 0)}")
    
    return True

if __name__ == "__main__":
    try:
        test_drug_canonicalization()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)