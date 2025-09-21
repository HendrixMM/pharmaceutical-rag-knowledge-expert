"""Unit tests for MinHash dependency guards in ranking filter."""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Import the module after potentially mocking dependencies
with patch.dict('sys.modules', {'numpy': None, 'datasketch': None}):
    from src.ranking_filter import StudyRankingFilter, ENABLE_MINHASH_DIVERSITY, HAS_DATASKETCH, HAS_NUMPY


class TestMinHashDependencyGuards(unittest.TestCase):
    """Test that MinHash dependencies are properly guarded and fallback gracefully."""

    def setUp(self):
        """Set up test fixtures."""
        self.filter = StudyRankingFilter()

    def test_missing_datasketch_fallback(self):
        """Test that missing datasketch falls back to pure Python implementation."""
        # Mock datasketch as missing
        with patch.dict('sys.modules', {'datasketch': None}):
            # Reload module to trigger missing import
            import importlib
            import src.ranking_filter
            importlib.reload(src.ranking_filter)

            # Verify flags are set correctly
            self.assertFalse(src.ranking_filter.HAS_DATASKETCH)
            self.assertTrue(src.ranking_filter.HAS_NUMPY)  # Still available

            # Create test papers
            papers = [
                {"title": "Study 1", "abstract": "This is about drugs"},
                {"title": "Study 2", "abstract": "This is also about drugs"},
            ]

            # Should not raise error
            result = src.ranking_filter.StudyRankingFilter().apply_diversity_filter(
                papers, method="minhash", threshold=0.8
            )
            self.assertEqual(len(result), 2)

    def test_missing_numpy_fallback(self):
        """Test that missing numpy falls back to pure Python implementation."""
        # Mock both as missing
        with patch.dict('sys.modules', {'numpy': None, 'datasketch': None}):
            import importlib
            import src.ranking_filter
            importlib.reload(src.ranking_filter)

            # Verify flags
            self.assertFalse(src.ranking_filter.HAS_NUMPY)
            self.assertFalse(src.ranking_filter.HAS_DATASKETCH)

            # Should still work
            papers = [
                {"title": "Study 1", "abstract": "This is about drugs"},
                {"title": "Study 2", "abstract": "This is also about drugs"},
            ]

            result = src.ranking_filter.StudyRankingFilter().apply_diversity_filter(
                papers, method="minhash", threshold=0.8
            )
            self.assertEqual(len(result), 2)

    def test_enable_minhash_disabled(self):
        """Test behavior when ENABLE_MINHASH_DIVERSITY is disabled."""
        with patch('src.ranking_filter.ENABLE_MINHASH_DIVERSITY', False):
            papers = [
                {"title": "Study 1", "abstract": "This is about drugs"},
                {"title": "Study 2", "abstract": "This is also about drugs"},
            ]

            result = self.filter.apply_diversity_filter(
                papers, method="minhash", threshold=0.8
            )
            # Should fall back to signature method
            self.assertEqual(len(result), 2)

    def test_datasketch_exception_handling(self):
        """Test that exceptions in datasketch are caught and fallback works."""
        papers = [
            {"title": "Study 1", "abstract": "This is about drugs"},
            {"title": "Study 2", "abstract": "This is also about drugs"},
        ]

        # Mock datasketch to raise an exception
        with patch.object(self.filter, '_apply_minhash_datasketch', side_effect=Exception("Test error")):
            result = self.filter.apply_diversity_filter(
                papers, method="minhash", threshold=0.8
            )
            # Should fall back successfully
            self.assertEqual(len(result), 2)

    def test_small_input_size_fallback(self):
        """Test that small input sizes use pure Python regardless of MinHash availability."""
        papers = [
            {"title": "Study 1", "abstract": "Short"},
        ]

        # Even with MinHash enabled, small input should use pure Python
        with patch.object(self.filter, '_apply_minhash_datasketch') as mock_minhash:
            result = self.filter.apply_diversity_filter(
                papers, method="minhash", threshold=0.8
            )
            # Should not call MinHash implementation
            mock_minhash.assert_not_called()
            self.assertEqual(len(result), 1)

    def test_configuration_validation(self):
        """Test that MinHash configuration values are validated with fallbacks."""
        # Test with invalid configuration
        with patch.dict(os.environ, {
            "MINHASH_NUM_PERMUTATIONS": "50",  # Below minimum
            "MINHASH_LSH_THRESHOLD": "1.5",   # Above maximum
        }):
            import importlib
            import src.ranking_filter
            importlib.reload(src.ranking_filter)

            # Should fallback to defaults
            self.assertEqual(src.ranking_filter.MINHASH_NUM_PERMUTATIONS, 128)
            self.assertEqual(src.ranking_filter.MINHASH_LSH_THRESHOLD, 0.8)


if __name__ == "__main__":
    unittest.main()