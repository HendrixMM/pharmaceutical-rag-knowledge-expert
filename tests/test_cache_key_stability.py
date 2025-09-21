"""Unit tests for cache key stability in QueryEngine."""

import unittest
from unittest.mock import Mock, patch
from src.query_engine import EnhancedQueryEngine
from src.pubmed_scraper import PubMedScraper


class TestCacheKeyStability(unittest.TestCase):
    """Test cache key generation across different filter configurations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_scraper = Mock(spec=PubMedScraper)
        self.engine = EnhancedQueryEngine(self.mock_scraper)

    def test_cache_key_stability_filter_order(self):
        """Test that semantically equivalent filters produce identical cache keys regardless of order."""
        filters1 = {
            "drug_names": ["aspirin", "ibuprofen"],
            "species": "human",
            "year_range": [2020, 2023]
        }
        filters2 = {
            "species": "human",
            "year_range": [2020, 2023],
            "drug_names": ["aspirin", "ibuprofen"]
        }

        key1 = self.engine._generate_cache_key(
            query="test query",
            enhanced_query="test enhanced",
            max_items=10,
            sort_by="relevance",
            filters=filters1
        )

        key2 = self.engine._generate_cache_key(
            query="test query",
            enhanced_query="test enhanced",
            max_items=10,
            sort_by="relevance",
            filters=filters2
        )

        self.assertEqual(key1, key2, "Cache keys should be identical for semantically equivalent filters")

    def test_cache_key_stability_filter_types(self):
        """Test that different filter types with same values produce consistent keys."""
        filters_list = {"drug_names": ["aspirin", "ibuprofen"]}
        filters_tuple = {"drug_names": ("aspirin", "ibuprofen")}
        filters_set = {"drug_names": {"aspirin", "ibuprofen"}}

        key1 = self.engine._generate_cache_key(
            query="test query",
            enhanced_query="test enhanced",
            max_items=10,
            sort_by="relevance",
            filters=filters_list
        )

        key2 = self.engine._generate_cache_key(
            query="test query",
            enhanced_query="test enhanced",
            max_items=10,
            sort_by="relevance",
            filters=filters_tuple
        )

        key3 = self.engine._generate_cache_key(
            query="test query",
            enhanced_query="test enhanced",
            max_items=10,
            sort_by="relevance",
            filters=filters_set
        )

        # All should produce the same key after normalization
        self.assertEqual(key1, key2, "List and tuple filters should normalize to same key")
        self.assertEqual(key1, key3, "Set filters should normalize to same key as list/tuple")

    def test_cache_key_json_compactness(self):
        """Test that cache keys use compact JSON separators."""
        filters = {"drug_names": ["aspirin"], "species": "human"}

        with patch('json.dumps') as mock_dumps:
            mock_dumps.return_value = '{"query":"test","filters":{"drug_names":["aspirin"],"species":"human"}}'

            self.engine._generate_cache_key(
                query="test",
                enhanced_query="test enhanced",
                max_items=10,
                sort_by="relevance",
                filters=filters
            )

            # Verify compact separators are used
            args, kwargs = mock_dumps.call_args
            self.assertEqual(kwargs.get('separators'), (',', ':'))
            self.assertTrue(kwargs.get('sort_keys'))

    def test_cache_key_different_queries(self):
        """Test that different queries produce different cache keys."""
        filters = {"species": "human"}

        key1 = self.engine._generate_cache_key(
            query="query one",
            enhanced_query="enhanced one",
            max_items=10,
            sort_by="relevance",
            filters=filters
        )

        key2 = self.engine._generate_cache_key(
            query="query two",
            enhanced_query="enhanced two",
            max_items=10,
            sort_by="relevance",
            filters=filters
        )

        self.assertNotEqual(key1, key2, "Different queries should produce different cache keys")

    def test_cache_key_none_values(self):
        """Test that None values in filters are handled consistently."""
        filters1 = {"drug_names": ["aspirin"], "species": None}
        filters2 = {"drug_names": ["aspirin"]}

        key1 = self.engine._generate_cache_key(
            query="test query",
            enhanced_query="test enhanced",
            max_items=10,
            sort_by="relevance",
            filters=filters1
        )

        key2 = self.engine._generate_cache_key(
            query="test query",
            enhanced_query="test enhanced",
            max_items=10,
            sort_by="relevance",
            filters=filters2
        )

        self.assertEqual(key1, key2, "None values should be normalized consistently")


if __name__ == "__main__":
    unittest.main()