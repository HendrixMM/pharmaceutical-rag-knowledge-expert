import datetime

from src.ranking_filter import StudyRankingFilter


def test_estimate_sample_size_prioritizes_explicit_n_equals():
    ranking_filter = StudyRankingFilter()
    text = "The cohort included n=48 participants and an additional 120 patients in registries."
    assert ranking_filter._estimate_sample_size_from_text(text) == 48


def test_estimate_sample_size_ignores_four_digit_years():
    ranking_filter = StudyRankingFilter()
    text = "A retrospective review followed 2020 participants from the 2020 outbreak."
    assert ranking_filter._estimate_sample_size_from_text(text) is None


def test_estimate_sample_size_with_enrolled_context():
    ranking_filter = StudyRankingFilter()
    text = "A total of 180 patients were enrolled across sites in 2020."
    assert ranking_filter._estimate_sample_size_from_text(text) == 180


def test_estimate_sample_size_with_randomized_prefix():
    ranking_filter = StudyRankingFilter()
    text = "Randomized 95 participants to receive either treatment or placebo."
    assert ranking_filter._estimate_sample_size_from_text(text) == 95


def test_recency_decay_years_parameter_adjusts_decay():
    current_year = datetime.datetime.now(datetime.timezone.utc).year
    default_filter = StudyRankingFilter()
    fast_decay_filter = StudyRankingFilter(recency_decay_years=5)

    year = current_year - 5
    default_score = default_filter._calculate_recency_score(year)
    fast_decay_score = fast_decay_filter._calculate_recency_score(year)

    assert fast_decay_score < default_score


def test_verbose_ranking_explanation_includes_terms():
    ranking_filter = StudyRankingFilter()
    paper = {
        "title": "Pharmacokinetics of drug interaction",
        "abstract": "CYP3A4 inhibitors such as ketoconazole reduce clearance.",
        "drug_names": ["ketoconazole"],
    }
    ranked = ranking_filter.rank_studies([paper], query="ketoconazole interaction")
    explanation = ranking_filter.get_ranking_explanation(ranked[0], verbose=True)
    assert "pharma keywords" in explanation
    assert "ketoconazole" in explanation


def test_rank_studies_populates_pharma_details():
    ranking_filter = StudyRankingFilter()
    paper = {
        "title": "CYP3A4 interaction study",
        "abstract": "Ketoconazole shows strong pharmacokinetics interaction with CYP3A4.",
        "drug_names": ["ketoconazole"],
        "cyp_enzymes": ["CYP3A4"],
        "mesh_terms": ["humans"],
    }
    ranked = ranking_filter.rank_studies([paper], query="ketoconazole interaction")
    details = ranked[0].get("ranking_pharma_details")
    assert details is not None
    assert "keywords" in details and "pharmacokinetics" in details["keywords"]
    assert details.get("cyp_overlaps") == ["cyp3a4"]


def test_quality_score_handles_randomised_spelling():
    ranking_filter = StudyRankingFilter()
    score = ranking_filter._calculate_study_quality_score(["Randomised Controlled Trial"])
    assert score >= 0.88


def test_lower_quality_synonyms_reduce_score():
    ranking_filter = StudyRankingFilter()
    score = ranking_filter._calculate_study_quality_score(["Case Report"])
    assert score <= 0.45


def test_diversity_filter_short_circuits_exact_titles():
    ranking_filter = StudyRankingFilter()
    papers = [
        {"title": "Study One", "abstract": "Group A"},
        {"title": "Study One", "abstract": "Group B variant"},
    ]
    filtered = ranking_filter.apply_diversity_filter(papers)
    assert len(filtered) == 1


def test_observational_study_pattern_matching():
    """Test that observational study patterns match correctly with fixed regex."""
    ranking_filter = StudyRankingFilter()

    # Test direct observational match
    score1 = ranking_filter._match_tag_with_patterns("observational study")
    assert score1 == 0.7

    # Test observational with studies
    score2 = ranking_filter._match_tag_with_patterns("observational studies")
    assert score2 == 0.7

    # Test case insensitive
    score3 = ranking_filter._match_tag_with_patterns("OBSERVATIONAL STUDY")
    assert score3 == 0.7


def test_clinical_trial_phase_numeric_matching():
    """Test that clinical trial phases accept both roman and numeric values."""
    ranking_filter = StudyRankingFilter()

    # Test roman numerals (existing behavior)
    score1 = ranking_filter._match_tag_with_patterns("phase i clinical trial")
    assert score1 == 0.78

    score2 = ranking_filter._match_tag_with_patterns("phase iii clinical trial")
    assert score2 == 0.88

    # Test numeric values (new behavior)
    score3 = ranking_filter._match_tag_with_patterns("phase 1 clinical trial")
    assert score3 == 0.78

    score4 = ranking_filter._match_tag_with_patterns("phase 3 clinical trial")
    assert score4 == 0.88

    # Test case insensitive
    score5 = ranking_filter._match_tag_with_patterns("PHASE II CLINICAL TRIAL")
    assert score5 == 0.82

    score6 = ranking_filter._match_tag_with_patterns("PHASE 2 CLINICAL TRIAL")
    assert score6 == 0.82
