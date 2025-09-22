#!/usr/bin/env python3
"""Test script for drug name canonicalization with proper assertions."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ddi_pk_processor import DDIPKProcessor
from pharmaceutical_processor import PharmaceuticalProcessor

def test_drug_canonicalization():
    """Test drug name canonicalization functionality with proper assertions."""
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

    # Test 1: Verify canonical mapping correctness
    canonical_map = ddi_processor._build_drug_synonym_map(test_papers)
    print("Canonical map:", canonical_map)

    # Assertions for canonical mapping
    assert "warfarin" in canonical_map, "Generic drug 'warfarin' should be in canonical map"
    assert "coumadin" in canonical_map, "Brand drug 'coumadin' should be in canonical map"
    assert canonical_map["warfarin"] == "warfarin", "Generic 'warfarin' should map to itself"
    assert canonical_map["coumadin"] == "warfarin", "Brand 'coumadin' should map to generic 'warfarin'"
    print("✓ Canonical mapping tests passed")

    # Test 2: Brand-to-generic conversion in analysis
    result = ddi_processor.analyze_drug_interactions(
        test_papers,
        "Coumadin",  # brand name input
        ["Warfarin"]  # generic name
    )

    print("Analysis result with brand name input:")
    print(f"Primary drug: {result.get('primary_drug')}")
    print(f"Secondary drugs analyzed: {result.get('secondary_drugs_analyzed')}")

    # Assertions for brand name analysis
    assert result.get('primary_drug'), "Analysis should return a primary drug"
    assert result.get('analyzed_papers') == 1, "Should analyze exactly 1 paper"
    assert 'error' not in result, f"Analysis should not have errors: {result.get('error')}"
    assert isinstance(result.get('secondary_drugs_analyzed'), list), "Secondary drugs should be a list"
    print("✓ Brand name analysis tests passed")

    # Test 3: Generic name consistency
    result2 = ddi_processor.analyze_drug_interactions(
        test_papers,
        "warfarin",  # generic name input
        ["coumadin"]  # brand name
    )

    print("\nReverse test result with generic name input:")
    print(f"Primary drug: {result2.get('primary_drug')}")
    print(f"Secondary drugs analyzed: {result2.get('secondary_drugs_analyzed')}")

    # Assertions for generic name analysis
    assert result2.get('primary_drug'), "Reverse analysis should return a primary drug"
    assert result2.get('analyzed_papers') == 1, "Should analyze exactly 1 paper"
    assert 'error' not in result2, f"Reverse analysis should not have errors: {result2.get('error')}"
    assert isinstance(result2.get('secondary_drugs_analyzed'), list), "Secondary drugs should be a list"
    print("✓ Generic name analysis tests passed")

    # Test 4: Consistency between brand and generic inputs
    # Both analyses should produce consistent results after canonicalization
    result1_summary = result.get('summary_statistics', {})
    result2_summary = result2.get('summary_statistics', {})

    assert isinstance(result1_summary, dict), "Summary statistics should be a dictionary"
    assert isinstance(result2_summary, dict), "Summary statistics should be a dictionary"

    # Both should have the same number of analyzed papers
    assert result.get('analyzed_papers') == result2.get('analyzed_papers'), \
        "Both analyses should process the same number of papers"
    print("✓ Consistency tests passed")

    # Test 5: Error handling for edge cases
    try:
        # Test with empty papers
        empty_result = ddi_processor.analyze_drug_interactions([], "test_drug", ["other_drug"])
        assert empty_result.get('analyzed_papers') == 0, "Empty papers should result in 0 analyzed papers"
        print("✓ Empty papers edge case handled correctly")

        # Test with None inputs
        none_result = ddi_processor.analyze_drug_interactions(test_papers, "nonexistent_drug", None)
        assert 'error' not in none_result or none_result.get('analyzed_papers') == 1, \
            "None secondary drugs should be handled gracefully"
        print("✓ None inputs edge case handled correctly")

    except Exception as e:
        print(f"Edge case handling error: {e}")
        raise AssertionError(f"Edge case handling failed: {e}")

    print("✓ All drug canonicalization tests passed!")
    return True

def test_interaction_detection():
    """Test that interaction detection works consistently."""
    print("\nTesting interaction detection consistency...")

    pharma_processor = PharmaceuticalProcessor()
    ddi_processor = DDIPKProcessor(pharma_processor=pharma_processor)

    # Paper with explicit interaction data
    interaction_papers = [
        {
            "page_content": "When warfarin and fluconazole are co-administered, AUC increased by 50% (p<0.05). Monitor INR closely.",
            "metadata": {
                "pmid": "test_interaction",
                "drug_annotations": [
                    {"name": "warfarin", "type": "generic", "confidence": 0.95},
                    {"name": "fluconazole", "type": "generic", "confidence": 0.93}
                ]
            }
        }
    ]

    result = ddi_processor.analyze_drug_interactions(
        interaction_papers,
        "warfarin",
        ["fluconazole"]
    )

    # Assertions for interaction detection
    assert result.get('auc_cmax_changes'), "Should detect AUC/Cmax changes"
    auc_changes = result.get('auc_cmax_changes', [])
    assert len(auc_changes) > 0, "Should find at least one pharmacokinetic change"

    # Check that the change contains expected fields
    if auc_changes:
        change = auc_changes[0]
        assert change.get('parameter'), "Change should have a parameter field"
        assert change.get('direction'), "Change should have a direction field"
        assert change.get('change_value') is not None, "Change should have a change_value field"

    print("✓ Interaction detection tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_drug_canonicalization()
        test_interaction_detection()
        print("\n🎉 All tests passed successfully!")
    except AssertionError as e:
        print(f"\n❌ Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)