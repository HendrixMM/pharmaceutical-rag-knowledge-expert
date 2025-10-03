"""Unit tests for pharmaceutical utilities."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pharma_utils import _tokenize_species_string
from src.pharma_utils import CacheSizeConfig
from src.pharma_utils import cleanup_oldest_cache_files
from src.pharma_utils import DrugNameChecker
from src.pharma_utils import get_cache_dir_size_mb


class TestSpeciesTokenization(unittest.TestCase):
    """Test species string tokenization."""

    def test_empty_string(self):
        """Test empty string handling."""
        result = _tokenize_species_string("")
        self.assertEqual(result, set())

    def test_none_input(self):
        """Test None input handling."""
        result = _tokenize_species_string(None)
        self.assertEqual(result, set())

    def test_simple_species(self):
        """Test simple species names."""
        result = _tokenize_species_string("Human")
        self.assertEqual(result, {"human"})

    def test_case_insensitive(self):
        """Test case insensitivity."""
        result = _tokenize_species_string("HuMaN")
        self.assertEqual(result, {"human"})

    def test_multiple_words(self):
        """Test multi-word species strings."""
        result = _tokenize_species_string("Human and mouse studies")
        self.assertEqual(result, {"human", "and", "mouse", "studies"})

    def test_non_human_normalization(self):
        """Test non-human to nonhuman normalization."""
        result = _tokenize_species_string("non-human primates")
        self.assertEqual(result, {"nonhuman", "primates"})

    def test_separate_non_human(self):
        """Test separate non and human tokens."""
        result = _tokenize_species_string("non human")
        self.assertEqual(result, {"non", "human", "nonhuman"})

    def test_in_vitro(self):
        """Test in vitro tokenization."""
        result = _tokenize_species_string("In vitro studies")
        self.assertEqual(result, {"in", "vitro", "studies"})

    def test_special_characters(self):
        """Test handling of special characters."""
        result = _tokenize_species_string("Mouse, rat; and dog!")
        self.assertEqual(result, {"mouse", "rat", "and", "dog"})


class TestDrugNameChecker(unittest.TestCase):
    """Test lightweight drug name detection."""

    def setUp(self):
        """Set up DrugNameChecker instance."""
        self.checker = DrugNameChecker()

    def test_cyp_enzymes(self):
        """Test CYP enzyme detection."""
        result = self.checker.is_drug_like("CYP3A4 inhibition")
        self.assertTrue(result)

        result = self.checker.is_drug_like("cyp2d6 metabolizer")
        self.assertTrue(result)

        result = self.checker.is_drug_like("CYP1A2")
        self.assertTrue(result)

    def test_drug_suffixes(self):
        """Test drug suffix detection."""
        result = self.checker.is_drug_like("fluconazole treatment")
        self.assertTrue(result)

        result = self.checker.is_drug_like("atorvastatin therapy")
        self.assertTrue(result)

        result = self.checker.is_drug_like("lisinopril dosage")
        self.assertTrue(result)

    def test_common_drug_names(self):
        """Test common drug name detection."""
        result = self.checker.is_drug_like("aspirin for pain")
        self.assertTrue(result)

        result = self.checker.is_drug_like("metformin and diabetes")
        self.assertTrue(result)

    def test_no_drug_terms(self):
        """Test text without drug terms."""
        result = self.checker.is_drug_like("general medical condition")
        self.assertFalse(result)

    def test_extract_signals(self):
        """Test signal extraction."""
        signals = self.checker.extract_drug_signals("CYP3A4 and fluconazole study")

        self.assertIn("cyp_enzymes", signals)
        self.assertIn("CYP3A4", signals["cyp_enzymes"])

        self.assertIn("drug_suffix_matches", signals)
        self.assertTrue(any("fluconazole" in match for match in signals["drug_suffix_matches"]))

        self.assertEqual(signals["signal_count"], 2)

    def test_signal_count_accuracy(self):
        """Test signal count accuracy."""
        signals = self.checker.extract_drug_signals("aspirin CYP3A4 fluconazole")
        self.assertEqual(signals["signal_count"], 3)

        signals = self.checker.extract_drug_signals("no drugs here")
        self.assertEqual(signals["signal_count"], 0)


class TestCacheSizeConfig(unittest.TestCase):
    """Test cache size configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheSizeConfig()
        self.assertEqual(config.max_size_mb, 1000)
        self.assertEqual(config.cleanup_threshold_mb, 900)
        self.assertEqual(config.check_frequency, 50)

    @patch.dict(
        os.environ,
        {
            "QUERY_ENGINE_MAX_CACHE_MB": "500",
            "QUERY_ENGINE_CACHE_CLEANUP_THRESHOLD_MB": "450",
            "QUERY_ENGINE_CACHE_CHECK_FREQUENCY": "25",
        },
    )
    def test_env_override(self):
        """Test environment variable overrides."""
        config = CacheSizeConfig()
        self.assertEqual(config.max_size_mb, 500)
        self.assertEqual(config.cleanup_threshold_mb, 450)
        self.assertEqual(config.check_frequency, 25)


class TestCacheSizeManagement(unittest.TestCase):
    """Test cache size management utilities."""

    def setUp(self):
        """Set up temporary cache directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_empty_cache_size(self):
        """Test size calculation for empty cache."""
        size_mb = get_cache_dir_size_mb(self.cache_dir)
        self.assertEqual(size_mb, 0.0)

    def test_cache_size_with_files(self):
        """Test size calculation with cache files."""
        # Create test files
        (self.cache_dir / "file1.json").write_text('{"data": "x" * 1000}')
        (self.cache_dir / "file2.json").write_text('{"data": "y" * 2000}')

        size_mb = get_cache_dir_size_mb(self.cache_dir)
        # Should be approximately 3KB
        self.assertGreater(size_mb, 0.0)
        self.assertLess(size_mb, 0.1)

    def test_cleanup_no_files(self):
        """Test cleanup with no files."""
        stats = cleanup_oldest_cache_files(self.cache_dir, 1.0)
        self.assertEqual(stats["files_removed"], 0)
        self.assertEqual(stats["bytes_freed"], 0)

    def test_cleanup_within_limit(self):
        """Test cleanup when under size limit."""
        # Create small file
        (self.cache_dir / "file1.json").write_text('{"small": "data"}')

        stats = cleanup_oldest_cache_files(self.cache_dir, 10.0)
        self.assertEqual(stats["files_removed"], 0)

    def test_cleanup_over_limit(self):
        """Test cleanup when over size limit."""
        # Create test files with different timestamps
        files = []
        for i in range(5):
            file_path = self.cache_dir / f"file{i}.json"
            file_path.write_text('{"data": "x" * 1000}')  # 1KB each
            # Set different modification times
            file_path.touch()
            files.append(file_path)

        # Set target size very low to force cleanup
        stats = cleanup_oldest_cache_files(self.cache_dir, 0.001)

        # Should have removed some files
        self.assertGreater(stats["files_removed"], 0)
        self.assertGreater(stats["bytes_freed"], 0)

    def test_cleanup_ignores_non_json(self):
        """Test that cleanup only considers JSON files."""
        # Create mixed files
        (self.cache_dir / "data.json").write_text('{"data": "x" * 1000}')
        (self.cache_dir / "temp.txt").write_text("temporary file")
        (self.cache_dir / "backup").write_text("backup data")

        stats = cleanup_oldest_cache_files(self.cache_dir, 0.001)
        # Should only remove JSON files
        self.assertEqual(stats["files_removed"], 1)


class TestPKFilteringFlag(unittest.TestCase):
    """Test PK filtering environment variable handling."""

    def test_default_disabled(self):
        """Test that PK filtering is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload module to pick up env
            from importlib import reload

            import src.pharma_utils

            reload(src.pharma_utils)
            self.assertFalse(src.pharma_utils._PK_FILTERING_ENABLED)

    @patch.dict(os.environ, {"ENABLE_PK_FILTERING": "true"})
    def test_enabled_true(self):
        """Test enabling PK filtering."""
        from importlib import reload

        import src.pharma_utils

        reload(src.pharma_utils)
        self.assertTrue(src.pharma_utils._PK_FILTERING_ENABLED)

    @patch.dict(os.environ, {"ENABLE_PK_FILTERING": "TRUE"})
    def test_enabled_case_insensitive(self):
        """Test case insensitive enable."""
        from importlib import reload

        import src.pharma_utils

        reload(src.pharma_utils)
        self.assertTrue(src.pharma_utils._PK_FILTERING_ENABLED)

    @patch.dict(os.environ, {"ENABLE_PK_FILTERING": "false"})
    def test_explicitly_disabled(self):
        """Test explicitly disabling PK filtering."""
        from importlib import reload

        import src.pharma_utils

        reload(src.pharma_utils)
        self.assertFalse(src.pharma_utils._PK_FILTERING_ENABLED)


if __name__ == "__main__":
    unittest.main()
