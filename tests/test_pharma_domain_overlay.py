import importlib.util
from pathlib import Path

import pytest


def _load_pharmaceutical_processor() -> type:
    """Load PharmaceuticalProcessor without importing src package (to avoid heavy deps)."""
    file_path = Path("src/pharmaceutical_processor.py").resolve()
    spec = importlib.util.spec_from_file_location("pharmaceutical_processor", str(file_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return getattr(module, "PharmaceuticalProcessor")


@pytest.fixture()
def tmp_synonyms_csv(tmp_path: Path) -> Path:
    p = tmp_path / "drug_synonyms.csv"
    p.write_text(
        "brand,generic,iupac,aliases\n" "Advil,ibuprofen,2-(4-isobutylphenyl)propionic acid,Motrin|Nurofen\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def tmp_regulatory_csv(tmp_path: Path) -> Path:
    p = tmp_path / "regulatory_status.csv"
    p.write_text(
        "drug,agency,status,date,notes\n" "ibuprofen,FDA,Approved,2000-01-01,\n",
        encoding="utf-8",
    )
    return p


def _build_processor(monkeypatch: pytest.MonkeyPatch, synonyms: Path | None, regulatory: Path | None):
    PharmaceuticalProcessor = _load_pharmaceutical_processor()
    # Enable overlay and sub-features
    monkeypatch.setenv("PHARMA_DOMAIN_OVERLAY", "true")
    monkeypatch.setenv("PHARMA_ENABLE_REGULATORY_TAGS", "true")
    monkeypatch.setenv("PHARMA_ENABLE_SPECIES_INFERENCE", "true")
    monkeypatch.setenv("PHARMA_ENABLE_EVIDENCE_LEVEL", "true")
    # Point to optional CSVs
    if synonyms is not None:
        monkeypatch.setenv("DRUG_SYNONYMS_CSV", str(synonyms))
        monkeypatch.delenv("DRUG_SYNONYMS_JSON", raising=False)
    if regulatory is not None:
        monkeypatch.setenv("REGULATORY_STATUS_CSV", str(regulatory))
    # Build processor fresh to read env
    return PharmaceuticalProcessor()


def test_overlay_synonyms_and_regulatory(
    monkeypatch: pytest.MonkeyPatch, tmp_synonyms_csv: Path, tmp_regulatory_csv: Path
) -> None:
    proc = _build_processor(monkeypatch, tmp_synonyms_csv, tmp_regulatory_csv)

    # Prepare a simple document mentioning a brand name only
    doc = {
        "page_content": "Advil is a common analgesic.",
        "metadata": {"study_types": ["randomized controlled trial"]},
    }

    enhanced = proc.enhance_document_metadata(doc)
    meta = enhanced["metadata"]

    # Synonym canonicalization
    assert "drug_names" in meta and meta["drug_names"], "expected drug_names to be populated"
    assert "drug_canonical_names" in meta, "expected canonical names"
    assert "ibuprofen" in [n.lower() for n in meta["drug_canonical_names"]]
    assert meta.get("drug_name_map"), "expected mapping of original to canonical"

    # Regulatory annotations
    # Entries include FDA approval row; tags include approved:FDA
    regs = meta.get("regulatory_status")
    assert isinstance(regs, list) and regs, "expected regulatory entries"
    tags = meta.get("regulatory_tags") or []
    assert any(t.startswith("approved:FDA") or t == "approved:fda" for t in tags)
    agencies = meta.get("regulatory_agencies") or []
    assert any(a.upper() == "FDA" for a in agencies)

    # Evidence level (from provided study_types)
    assert meta.get("evidence_level") in {"high", "very_high"}


def test_overlay_cyp_species_pk(monkeypatch: pytest.MonkeyPatch, tmp_synonyms_csv: Path) -> None:
    # No regulatory file here to ensure graceful degradation
    proc = _build_processor(monkeypatch, tmp_synonyms_csv, None)

    # Content contains CYP inhibitor cue, human species, and PK values
    content = (
        "Fluconazole is a CYP3A4 inhibitor in human studies. " "Reported AUC 20 ng*h/mL and Cmax 50 ng/mL in trial."
    )
    doc = {"page_content": content, "metadata": {}}

    enhanced = proc.enhance_document_metadata(doc)
    meta = enhanced["metadata"]

    # CYP risk scoring present
    assert meta.get("cyp_risk_label") in {"low", "moderate", "high"}
    assert isinstance(meta.get("cyp_risk_score"), int)

    # Species inference fallback
    assert meta.get("species") or meta.get("species_list"), "expected species inference"
    assert meta.get("species_inferred") is True

    # PK signals summary
    signals = meta.get("pk_signals_present") or []
    assert "auc" in [s.lower() for s in signals]
    assert "cmax" in [s.lower() for s in signals]
    assert isinstance(meta.get("pk_summary_score"), int) and meta["pk_summary_score"] >= 2


def test_overlay_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    PharmaceuticalProcessor = _load_pharmaceutical_processor()
    # Ensure overlay off yields no overlay-only fields
    monkeypatch.delenv("PHARMA_DOMAIN_OVERLAY", raising=False)
    monkeypatch.delenv("DRUG_SYNONYMS_CSV", raising=False)
    monkeypatch.delenv("REGULATORY_STATUS_CSV", raising=False)
    proc = PharmaceuticalProcessor()

    doc = {"page_content": "Advil is a brand.", "metadata": {}}
    meta = proc.enhance_document_metadata(doc)["metadata"]

    assert "drug_canonical_names" not in meta
    assert "regulatory_status" not in meta
    assert "cyp_risk_label" not in meta
    assert "evidence_level" not in meta
    assert "species_inferred" not in meta
    assert "pk_signals_present" not in meta
