"""Unit tests for sample size extraction year guardrails."""
import unittest

from src.ranking_filter import StudyRankingFilter


class TestSampleSizeYearGuardrails(unittest.TestCase):
    """Test cases for year guardrails in sample size extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = StudyRankingFilter()

    def test_rejects_years_with_date_context(self):
        """Test that numbers 1900-2099 are rejected when near date context words."""
        # Test cases with year and date context
        test_cases = [
            ("Study period from 2010 to 2015", None),
            ("Follow-up until 2020", None),
            ("Years between 1995 and 2005", None),
            ("During the year 2018", None),
            ("From 2012 to present", None),
            ("Study duration 2019-2021", None),
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.filter._estimate_sample_size_from_text(text)
                self.assertEqual(result, expected, f"Failed for text: {text}")

    def test_accepts_years_without_date_context(self):
        """Test that numbers 1900-2099 are accepted when NOT in date context."""
        # These should be treated as sample sizes, not years
        test_cases = [
            ("We enrolled 2015 patients in the study", 2015),
            ("Total sample size was 2005 subjects", 2005),
            ("N=1999 participants were included", 1999),
            ("The study included 2100 individuals", 2100),
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.filter._estimate_sample_size_from_text(text)
                self.assertEqual(result, expected, f"Failed for text: {text}")

    def test_parses_sample_size_with_year_nearby(self):
        """Test that sample sizes are correctly parsed even when years appear nearby."""
        test_cases = [
            ("In 2015, we studied 500 patients", 500),
            ("The 2020 study enrolled 100 subjects", 100),
            ("During 2018-2019, 50 participants were enrolled", 50),
            ("Published in 2016, n=3000", 3000),
        ]

        for text, expected in test_cases:
            with self.subTest(text=text):
                result = self.filter._estimate_sample_size_from_text(text)
                self.assertEqual(result, expected, f"Failed for text: {text}")

    def test_date_context_keywords_variations(self):
        """Test various date context keyword variations."""
        date_context_patterns = [
            ("follow-up period: 2010", None),
            ("study year 2015", None),
            ("years 2000-2005", None),
            ("duration from 2018", None),
            ("between 2012 and 2014", None),
            ("during 2019", None),
            ("until 2021", None),
        ]

        for text, expected in date_context_patterns:
            with self.subTest(text=text):
                result = self.filter._estimate_sample_size_from_text(text)
                self.assertEqual(result, expected, f"Failed for text: {text}")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        edge_cases = [
            # Exact year boundaries
            ("Study in 1899", 1899),  # Before 1900 - should be accepted
            ("Study in 2101", 2101),  # After 2100 - should be accepted
            # Year-like numbers in clear sample size context
            ("n=1900 patients", 1900),
            ("Total 2100 subjects", 2100),
            # Multiple numbers with date context
            ("From 2010 to 2015 (n=500)", 500),
            ("Years 2018-2020: 1000 patients enrolled", 1000),
        ]

        for text, expected in edge_cases:
            with self.subTest(text=text):
                result = self.filter._estimate_sample_size_from_text(text)
                self.assertEqual(result, expected, f"Failed for text: {text}")


if __name__ == "__main__":
    unittest.main()
