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
