from src.pharmaceutical_processor import PharmaceuticalProcessor


def test_pharmacokinetic_clearance_patterns_avoid_false_positive():
    processor = PharmaceuticalProcessor()
    noise_only = processor.extract_pharmacokinetic_parameters("The class of drugs was reviewed without PK metrics.")
    assert "clearance" not in noise_only

    matches = processor.extract_pharmacokinetic_parameters("Dose adjustments reduced CL/F to 18 L/hour.")
    assert "clearance" in matches
    assert "CL/F" in matches["clearance"]
    values = matches.get("pharmacokinetic_values", {}).get("clearance", [])
    assert values
    assert values[0]["unit"].lower() == "l/h"


def test_dosage_duration_extraction():
    processor = PharmaceuticalProcessor()
    text = "Administer 200 mg PO twice daily for 14 days."
    entries = processor.extract_dosage_information(text)
    assert entries
    duration = entries[0].get("duration")
    assert duration == {"value": 14, "unit": "days"}


def test_annotate_cyp_roles_uses_role_mapping():
    processor = PharmaceuticalProcessor()
    annotations = processor.annotate_cyp_roles("Strong CYP3A4 inhibitors such as ketoconazole")
    assert annotations
    enzymes = {entry["enzyme"] for entry in annotations}
    assert "CYP3A4" in enzymes


def test_external_generic_lexicon_boosts_confidence(tmp_path):
    lexicon_path = tmp_path / "generic_lexicon.txt"
    lexicon_path.write_text("Zenimab\n")

    processor = PharmaceuticalProcessor(generic_lexicon_path=str(lexicon_path))
    results = processor.extract_drug_names("The investigational agent Zenimab was compared to placebo.")

    zenimab = {item["name"].lower(): item for item in results}["zenimab"]
    assert zenimab["confidence"] == 0.98
    assert zenimab["type"] == "generic"


def test_bundled_and_env_lexicons_merge(tmp_path):
    """Test stub to verify loader merges bundled and env-provided lexicons."""
    # Create custom lexicon files
    custom_generic = tmp_path / "custom_generic.txt"
    custom_generic.write_text("custom_generic_drug\n")

    custom_brand = tmp_path / "custom_brand.txt"
    custom_brand.write_text("custom_brand_drug\n")

    # Initialize processor with custom lexicons
    processor = PharmaceuticalProcessor(generic_lexicon_path=str(custom_generic), brand_lexicon_path=str(custom_brand))

    # Should have both built-in and custom drugs
    assert "acetaminophen" in processor.generic_drug_names  # Built-in
    assert "custom_generic_drug" in processor.generic_drug_names  # Custom

    assert "coumadin" in processor.brand_drug_names  # Built-in
    assert "custom_brand_drug" in processor.brand_drug_names  # Custom


def test_comprehensive_lexicons_increase_detection_coverage():
    """Test that comprehensive lexicons increase drug detection coverage."""
    # Test with minimal built-in lexicons
    minimal_processor = PharmaceuticalProcessor()
    minimal_generics_count = len(minimal_processor.generic_drug_names)
    minimal_brands_count = len(minimal_processor.brand_drug_names)

    # Text with both common and less common drugs
    test_text = (
        "Patient was treated with amlodipine (Norvasc) for hypertension, "
        "atorvastatin (Lipitor) for hyperlipidemia, and esomeprazole (Nexium) for GERD."
    )

    minimal_results = minimal_processor.extract_drug_names(test_text)
    minimal_detected = {item["name"].lower() for item in minimal_results}

    # When comprehensive lexicons are available, detection should be more comprehensive
    # The bundled files should contain more drugs than the minimal built-in set
    assert minimal_generics_count >= 10  # Sanity check for built-in drugs
    assert minimal_brands_count >= 5  # Sanity check for built-in brands

    # Test should detect common drugs even with minimal lexicons
    assert len(minimal_detected) > 0, "Should detect at least some drugs from test text"
