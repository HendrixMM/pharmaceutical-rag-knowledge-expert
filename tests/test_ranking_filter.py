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


def test_recency_decay_years_parameter_adjusts_decay():
    current_year = datetime.datetime.utcnow().year
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
    assert "terms" in explanation
