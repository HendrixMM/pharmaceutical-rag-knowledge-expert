"""Regression tests for DDIPKProcessor drug canonicalization behaviour."""

from __future__ import annotations

import pytest

from src.ddi_pk_processor import DDIPKProcessor
from src.pharmaceutical_processor import PharmaceuticalProcessor


@pytest.fixture(scope="module")
def ddi_processor() -> DDIPKProcessor:
    """Provide a processor wired with the default pharmaceutical lexicons."""
    return DDIPKProcessor(pharma_processor=PharmaceuticalProcessor())


@pytest.fixture
def annotated_papers() -> list[dict[str, object]]:
    """Minimal paper set containing both generic and brand annotations."""
    return [
        {
            "page_content": (
                "Warfarin metabolism is affected by Coumadin. Both drugs show similar AUC increases."
            ),
            "metadata": {
                "pmid": "test1",
                "drug_annotations": [
                    {"name": "Warfarin", "type": "generic", "confidence": 0.98},
                    {"name": "Coumadin", "type": "brand", "confidence": 0.96},
                ],
            },
        }
    ]


def test_synonym_map_normalises_brand_aliases(ddi_processor: DDIPKProcessor, annotated_papers: list[dict[str, object]]) -> None:
    """Brand names should resolve to the canonical generic form for downstream analysis."""
    canonical_map = ddi_processor._build_drug_synonym_map(annotated_papers)

    # Retain self-mapping for canonical generic names to keep summaries predictable.
    assert canonical_map["warfarin"] == "warfarin"
    # Brand aliases must collapse to the same key so severity aggregation stays consistent.
    assert canonical_map["coumadin"] == "warfarin"


def test_analysis_handles_brand_and_generic_inputs(ddi_processor: DDIPKProcessor, annotated_papers: list[dict[str, object]]) -> None:
    """Switching between brand and generic names should not change canonicalised metrics."""
    canonical_map = ddi_processor._build_drug_synonym_map(annotated_papers)

    # Analyse with brand as the user-supplied primary drug.
    brand_first = ddi_processor.analyze_drug_interactions(annotated_papers, "Coumadin", ["Warfarin"])
    # Analyse with the canonical generic as the primary drug for parity checks.
    generic_first = ddi_processor.analyze_drug_interactions(annotated_papers, "warfarin", ["coumadin"])

    assert "error" not in brand_first
    assert "error" not in generic_first

    # Compare using canonical forms so formatting preferences do not affect the assertions.
    brand_primary_canon = canonical_map[brand_first["primary_drug"].lower()]
    generic_primary_canon = canonical_map[generic_first["primary_drug"].lower()]
    assert brand_primary_canon == generic_primary_canon == "warfarin"

    # Both analyses should process the same paper corpus after deduplication.
    assert brand_first["analyzed_papers"] == generic_first["analyzed_papers"] == 1

    brand_secondary_canon = sorted(canonical_map[name.lower()] for name in brand_first["secondary_drugs_analyzed"])
    generic_secondary_canon = sorted(canonical_map[name.lower()] for name in generic_first["secondary_drugs_analyzed"])
    assert brand_secondary_canon == generic_secondary_canon == ["warfarin"]
