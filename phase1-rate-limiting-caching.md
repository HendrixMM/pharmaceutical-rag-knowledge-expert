# Phase 1: Critical Foundation Implementation Guide

## **Rate Limiting & Compliance + Advanced Caching Enhancement**

This implementation guide enhances your existing PubMedScraper class with NCBI compliance and intelligent caching while maintaining backward compatibility with your current medical disclaimer and safety systems.

---

## **Overview**

**Goal**: Add production-grade rate limiting and caching to your existing pharmaceutical RAG system without breaking current functionality.

**Approach**: Enhance existing PubMedScraper class with new methods and features while preserving current interface and logging patterns.

---

## **Enhanced PubMedScraper Class Implementation**

### **Core Enhancement Strategy**

```python
# src/enhanced_pubmed_scraper.py
import time
import asyncio
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Import your existing scraper
from .pubmed_scraper import EasyAPIPubMedScraper  # Your current implementation

logger = logging.getLogger(__name__)

class EnhancedPubMedScraper(EasyAPIPubMedScraper):
    """
    Enhanced PubMed scraper with rate limiting and advanced caching
    Extends existing EasyAPIPubMedScraper while maintaining compatibility
    """
    
    def __init__(self, api_token: str, cache_dir: str = "./pubmed_cache", 
                 enable_rate_limiting: bool = True, enable_caching: bool = True):
        # Initialize parent class (your existing implementation)
        super().__init__(api_token)
        
        # New enhancement features
        self.cache_dir = Path(cache_dir)
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_caching = enable_caching
        self.cache_duration = timedelta(hours=24)  # NCBI recommended
        
        # Rate limiting state
        self.request_timestamps = []
        self.max_requests_per_second = 3  # NCBI limit without API key
        self.daily_request_count = 0
        self.daily_limit = 500  # Conservative daily limit
        self.last_request_date = datetime.now().date()
        
        # Initialize cache directory
        self._initialize_cache_directory()
        
        logger.info("✅ Enhanced PubMed scraper initialized with rate limiting and caching")
    
    def _initialize_cache_directory(self):
        """Initialize cache directory structure"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "queries").mkdir(exist_ok=True)
            (self.cache_dir / "metadata").mkdir(exist_ok=True)
            logger.debug(f"Cache directory initialized: {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize cache directory: {e}")
            self.enable_caching = False
```

### **Rate Limiting Implementation**

```python
    async def _enforce_rate_limit(self):
        """Enforce NCBI rate limiting (3 requests per second)"""
        if not self.enable_rate_limiting:
            return
        
        current_time = time.time()
        
        # Reset daily count if new day
        current_date = datetime.now().date()
        if current_date != self.last_request_date:
            self.daily_request_count = 0
            self.last_request_date = current_date
            logger.info(f"Daily request count reset for {current_date}")
        
        # Check daily limit
        if self.daily_request_count >= self.daily_limit:
            raise Exception(f"Daily request limit ({self.daily_limit}) exceeded")
        
        # Remove timestamps older than 1 second
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 1.0
        ]
        
        # Check rate limit (3 requests per second)
        if len(self.request_timestamps) >= self.max_requests_per_second:
            # Calculate wait time
            oldest_relevant_request = min(self.request_timestamps)
            wait_time = 1.0 - (current_time - oldest_relevant_request)
            
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_timestamps.append(current_time)
        self.daily_request_count += 1
        
        logger.debug(f"Request {self.daily_request_count}/{self.daily_limit} for {current_date}")
    
    def _is_optimal_time(self) -> bool:
        """Check if current time is optimal for large PubMed jobs (9 PM - 5 AM EST)"""
        current_hour = datetime.now().hour
        is_weekend = datetime.now().weekday() >= 5
        
        if is_weekend:
            return True
        
        # Weekday optimal hours: 9 PM to 5 AM EST
        return current_hour >= 21 or current_hour <= 5
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status for monitoring"""
        current_time = time.time()
        recent_requests = [
            ts for ts in self.request_timestamps 
            if current_time - ts < 1.0
        ]
        
        return {
            "daily_requests_used": self.daily_request_count,
            "daily_limit": self.daily_limit,
            "requests_last_second": len(recent_requests),
            "rate_limit": self.max_requests_per_second,
            "is_optimal_time": self._is_optimal_time(),
            "requests_remaining_today": self.daily_limit - self.daily_request_count
        }
```

### **Advanced Caching System**

```python
    def _generate_cache_key(self, query: str, max_items: int, 
                          additional_params: Dict[str, Any] = None) -> str:
        """Generate unique cache key for query parameters"""
        cache_data = {
            "query": query.lower().strip(),
            "max_items": max_items,
            "scraper_version": "enhanced_v1"
        }
        
        if additional_params:
            cache_data.update(additional_params)
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()[:16]
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get cache file path for given key"""
        return self.cache_dir / "queries" / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cached result is still valid (24-hour policy)"""
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            age = datetime.now() - cached_time
            
            is_valid = age < self.cache_duration
            if not is_valid:
                logger.debug(f"Cache expired for {cache_file.name}, age: {age}")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"Invalid cache file {cache_file}: {e}")
            return False
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if valid"""
        if not self.enable_caching:
            return None
        
        cache_file = self._get_cache_file_path(cache_key)
        
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                logger.info(f"✅ Cache hit for query (key: {cache_key})")
                cached_data['cache_hit'] = True
                return cached_data
                
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_file}: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache query result with metadata"""
        if not self.enable_caching:
            return
        
        cache_file = self._get_cache_file_path(cache_key)
        
        # Add caching metadata
        cached_result = result.copy()
        cached_result.update({
            "timestamp": datetime.now().isoformat(),
            "cache_key": cache_key,
            "cached_by": "enhanced_pubmed_scraper"
        })
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cached_result, f, indent=2)
            
            logger.debug(f"💾 Results cached with key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
```

### **Enhanced Search Method with Caching & Rate Limiting**

```python
    async def search_papers_enhanced(self, query: str, max_items: int = 30,
                                   bypass_cache: bool = False,
                                   additional_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhanced paper search with rate limiting and caching
        Maintains compatibility with existing search_papers method
        
        Args:
            query: Search query string
            max_items: Maximum papers to retrieve (default: 30, max: 100)
            bypass_cache: Force fresh API call even if cached result exists
            additional_params: Additional parameters for cache key generation
        
        Returns:
            Enhanced results dictionary with caching and rate limit metadata
        """
        
        # Validate input
        max_items = min(max_items, 100)  # EasyAPI hard limit
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, max_items, additional_params)
        
        # Try to get cached result first
        if not bypass_cache:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
        
        # Rate limiting before API call
        await self._enforce_rate_limit()
        
        logger.info(f"🔍 Executing fresh PubMed search: {query[:50]}...")
        
        try:
            # Call parent class method (your existing implementation)
            results = await super().search_papers(query, max_items)
            
            # Enhance results with new metadata
            enhanced_results = {
                **results,
                "enhanced_metadata": {
                    "cache_hit": False,
                    "rate_limited": True,
                    "optimal_time": self._is_optimal_time(),
                    "daily_requests_used": self.daily_request_count,
                    "query_optimized": False  # Will be True in Phase 2
                },
                "search_params": {
                    "query": query,
                    "max_items": max_items,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Cache the results
            await self._cache_result(cache_key, enhanced_results)
            
            logger.info(f"✅ Retrieved {len(results.get('papers', []))} papers from PubMed")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ PubMed search failed: {e}")
            
            # Return safe fallback response
            return {
                "papers": [],
                "error": str(e),
                "enhanced_metadata": {
                    "cache_hit": False,
                    "rate_limited": True,
                    "error_occurred": True,
                    "daily_requests_used": self.daily_request_count
                },
                "search_params": {
                    "query": query,
                    "max_items": max_items,
                    "timestamp": datetime.now().isoformat()
                }
            }
```

### **Cache Management Utilities**

```python
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self.enable_caching:
            return {"caching_enabled": False}
        
        cache_files = list((self.cache_dir / "queries").glob("*.json"))
        valid_caches = []
        expired_caches = []
        invalid_caches = []
        
        for cache_file in cache_files:
            try:
                if self._is_cache_valid(cache_file):
                    valid_caches.append(cache_file)
                else:
                    expired_caches.append(cache_file)
            except:
                invalid_caches.append(cache_file)
        
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())
        
        return {
            "caching_enabled": True,
            "cache_directory": str(self.cache_dir),
            "total_cache_files": len(cache_files),
            "valid_caches": len(valid_caches),
            "expired_caches": len(expired_caches),
            "invalid_caches": len(invalid_caches),
            "total_cache_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_duration_hours": self.cache_duration.total_seconds() / 3600
        }
    
    def cleanup_expired_cache(self) -> Dict[str, int]:
        """Clean up expired cache files"""
        if not self.enable_caching:
            return {"cleaned": 0, "errors": 0}
        
        cache_files = list((self.cache_dir / "queries").glob("*.json"))
        cleaned_count = 0
        error_count = 0
        
        for cache_file in cache_files:
            try:
                if not self._is_cache_valid(cache_file):
                    cache_file.unlink()
                    cleaned_count += 1
                    logger.debug(f"Cleaned expired cache: {cache_file.name}")
            except Exception as e:
                error_count += 1
                logger.warning(f"Failed to clean cache file {cache_file}: {e}")
        
        logger.info(f"🧹 Cache cleanup: {cleaned_count} files removed, {error_count} errors")
        
        return {
            "cleaned": cleaned_count,
            "errors": error_count,
            "remaining_files": len(list((self.cache_dir / "queries").glob("*.json")))
        }
    
    async def warm_cache_for_common_queries(self, common_queries: List[str]):
        """Pre-populate cache with common pharmaceutical queries"""
        if not self.enable_caching:
            logger.warning("Caching disabled, cannot warm cache")
            return
        
        logger.info(f"🔥 Warming cache for {len(common_queries)} common queries")
        
        for query in common_queries:
            cache_key = self._generate_cache_key(query, 30)
            cached_result = await self._get_cached_result(cache_key)
            
            if not cached_result:
                try:
                    await self.search_papers_enhanced(query, max_items=30)
                    logger.debug(f"Cache warmed for query: {query[:30]}...")
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(1.2)
                    
                except Exception as e:
                    logger.warning(f"Failed to warm cache for query '{query}': {e}")
        
        logger.info("✅ Cache warming completed")
```

---

## **Integration with Existing RAG Agent**

### **Backward Compatible Integration**

```python
# src/enhanced_rag_agent.py
from .enhanced_pubmed_scraper import EnhancedPubMedScraper
from .rag_agent import RAGAgent  # Your existing implementation

class EnhancedRAGAgent(RAGAgent):
    """Enhanced RAG Agent with rate limiting and caching, maintains compatibility"""
    
    def __init__(self, docs_folder: str, api_key: str, 
                 enable_enhancements: bool = True, cache_dir: str = "./pubmed_cache"):
        # Initialize parent RAG agent
        super().__init__(docs_folder, api_key)
        
        # Replace scraper with enhanced version if enabled
        if enable_enhancements:
            self.scraper = EnhancedPubMedScraper(
                api_token=api_key,  # Assuming api_key works for both NVIDIA and Apify
                cache_dir=cache_dir,
                enable_rate_limiting=True,
                enable_caching=True
            )
            logger.info("✅ Enhanced RAG Agent initialized with caching and rate limiting")
        else:
            logger.info("Enhanced features disabled, using standard RAG agent")
    
    async def ask_question_enhanced(self, question: str, k: int = 4,
                                  use_cached_papers: bool = True) -> Dict[str, Any]:
        """
        Enhanced question answering with caching and rate limiting
        Falls back to parent method if enhancements not available
        """
        
        if hasattr(self.scraper, 'search_papers_enhanced'):
            # Use enhanced scraper if available
            try:
                # Get relevant papers using enhanced search
                papers_response = await self.scraper.search_papers_enhanced(
                    query=question,
                    max_items=k * 5,  # Get more papers for better selection
                    bypass_cache=not use_cached_papers
                )
                
                # Process papers through existing RAG pipeline
                base_response = super().ask_question(question, k)
                
                # Combine enhanced metadata with RAG response
                enhanced_response = {
                    **base_response,
                    "enhanced_metadata": papers_response.get("enhanced_metadata", {}),
                    "cache_info": {
                        "cache_hit": papers_response.get("enhanced_metadata", {}).get("cache_hit", False),
                        "papers_retrieved": len(papers_response.get("papers", [])),
                        "rate_limited": papers_response.get("enhanced_metadata", {}).get("rate_limited", False)
                    }
                }
                
                return enhanced_response
                
            except Exception as e:
                logger.warning(f"Enhanced search failed, falling back to standard: {e}")
        
        # Fallback to parent implementation
        return super().ask_question(question, k)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including enhancements"""
        base_status = {
            "rag_agent_active": True,
            "vector_db_status": self.vector_db.get_stats() if hasattr(self, 'vector_db') else {},
            "timestamp": datetime.now().isoformat()
        }
        
        if hasattr(self.scraper, 'get_rate_limit_status'):
            # Add enhanced scraper status
            base_status.update({
                "enhanced_features": True,
                "rate_limiting": self.scraper.get_rate_limit_status(),
                "caching": self.scraper.get_cache_stats()
            })
        else:
            base_status["enhanced_features"] = False
        
        return base_status
```

---

## **Configuration and Environment Setup**

### **Enhanced Environment Variables**

```python
# Add to your .env file
# Rate Limiting Configuration
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_SECOND=3
DAILY_REQUEST_LIMIT=500

# Caching Configuration  
ENABLE_CACHING=true
CACHE_DIRECTORY=./pubmed_cache
CACHE_DURATION_HOURS=24

# Enhancement Features
ENABLE_ENHANCED_SCRAPING=true
LOG_LEVEL=INFO
```

### **Configuration Loading**

```python
# src/config.py
import os
from typing import Dict, Any

def load_enhanced_config() -> Dict[str, Any]:
    """Load configuration for enhanced PubMed scraper"""
    return {
        # Rate limiting
        "enable_rate_limiting": os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
        "max_requests_per_second": int(os.getenv("MAX_REQUESTS_PER_SECOND", "3")),
        "daily_request_limit": int(os.getenv("DAILY_REQUEST_LIMIT", "500")),
        
        # Caching
        "enable_caching": os.getenv("ENABLE_CACHING", "true").lower() == "true",
        "cache_directory": os.getenv("CACHE_DIRECTORY", "./pubmed_cache"),
        "cache_duration_hours": int(os.getenv("CACHE_DURATION_HOURS", "24")),
        
        # Features
        "enable_enhanced_features": os.getenv("ENABLE_ENHANCED_SCRAPING", "true").lower() == "true",
        "log_level": os.getenv("LOG_LEVEL", "INFO")
    }
```

---

## **Testing and Validation**

### **Rate Limiting Tests**

```python
# tests/test_rate_limiting.py
import pytest
import asyncio
import time
from src.enhanced_pubmed_scraper import EnhancedPubMedScraper

@pytest.mark.asyncio
class TestRateLimiting:
    
    @pytest.fixture
    def scraper(self):
        return EnhancedPubMedScraper(
            api_token="test-token",
            enable_rate_limiting=True,
            enable_caching=False  # Disable for pure rate limit testing
        )
    
    async def test_rate_limit_enforcement(self, scraper):
        """Test that rate limiting prevents exceeding 3 req/sec"""
        start_time = time.time()
        
        # Try to make 5 requests rapidly
        tasks = []
        for i in range(5):
            tasks.append(scraper._enforce_rate_limit())
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should take at least 1.33 seconds for 5 requests at 3 req/sec
        assert elapsed >= 1.2, f"Rate limiting too permissive: {elapsed} seconds"
        
    async def test_daily_limit_enforcement(self, scraper):
        """Test daily request limit enforcement"""
        # Simulate hitting daily limit
        scraper.daily_request_count = scraper.daily_limit - 1
        
        # This should work
        await scraper._enforce_rate_limit()
        
        # This should fail
        with pytest.raises(Exception, match="Daily request limit"):
            await scraper._enforce_rate_limit()
    
    def test_optimal_time_detection(self, scraper):
        """Test optimal time detection logic"""
        # This test would need to mock datetime to test different times
        result = scraper._is_optimal_time()
        assert isinstance(result, bool)
```

### **Caching Tests**

```python
# tests/test_caching.py
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from src.enhanced_pubmed_scraper import EnhancedPubMedScraper

@pytest.mark.asyncio  
class TestCaching:
    
    @pytest.fixture
    def scraper_with_temp_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scraper = EnhancedPubMedScraper(
                api_token="test-token",
                cache_dir=temp_dir,
                enable_caching=True,
                enable_rate_limiting=False
            )
            yield scraper
    
    def test_cache_key_generation(self, scraper_with_temp_cache):
        """Test cache key generation consistency"""
        query = "warfarin drug interactions"
        max_items = 30
        
        key1 = scraper_with_temp_cache._generate_cache_key(query, max_items)
        key2 = scraper_with_temp_cache._generate_cache_key(query, max_items)
        
        assert key1 == key2, "Cache keys should be consistent"
        assert len(key1) == 16, "Cache key should be 16 characters"
    
    async def test_cache_storage_and_retrieval(self, scraper_with_temp_cache):
        """Test basic cache storage and retrieval"""
        cache_key = "test_key_123"
        test_data = {
            "papers": [{"title": "Test Paper", "pmid": "12345"}],
            "total_count": 1
        }
        
        # Store in cache
        await scraper_with_temp_cache._cache_result(cache_key, test_data)
        
        # Retrieve from cache
        cached_result = await scraper_with_temp_cache._get_cached_result(cache_key)
        
        assert cached_result is not None
        assert cached_result["papers"][0]["title"] == "Test Paper"
        assert "timestamp" in cached_result
    
    def test_cache_expiration(self, scraper_with_temp_cache):
        """Test cache expiration logic"""
        cache_file = scraper_with_temp_cache._get_cache_file_path("expired_test")
        
        # Create expired cache file
        expired_data = {
            "timestamp": (datetime.now() - timedelta(hours=25)).isoformat(),
            "papers": []
        }
        
        with open(cache_file, 'w') as f:
            json.dump(expired_data, f)
        
        # Should be invalid due to expiration
        assert not scraper_with_temp_cache._is_cache_valid(cache_file)
    
    def test_cache_cleanup(self, scraper_with_temp_cache):
        """Test expired cache cleanup"""
        # Create some expired cache files
        for i in range(3):
            cache_file = scraper_with_temp_cache._get_cache_file_path(f"expired_{i}")
            expired_data = {
                "timestamp": (datetime.now() - timedelta(hours=25)).isoformat(),
                "papers": []
            }
            with open(cache_file, 'w') as f:
                json.dump(expired_data, f)
        
        # Create one valid cache file
        valid_cache_file = scraper_with_temp_cache._get_cache_file_path("valid")
        valid_data = {
            "timestamp": datetime.now().isoformat(),
            "papers": []
        }
        with open(valid_cache_file, 'w') as f:
            json.dump(valid_data, f)
        
        # Run cleanup
        cleanup_result = scraper_with_temp_cache.cleanup_expired_cache()
        
        assert cleanup_result["cleaned"] == 3
        assert cleanup_result["remaining_files"] == 1
```

---

## **Monitoring and Logging Enhancements**

### **Enhanced Logging Configuration**

```python
# src/logging_config.py
import logging
import sys
from pathlib import Path

def setup_enhanced_logging(log_level: str = "INFO", log_file: str = None):
    """Set up enhanced logging for pharmaceutical RAG system"""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Specific loggers for enhanced features
    enhanced_logger = logging.getLogger('enhanced_pubmed_scraper')
    enhanced_logger.setLevel(logging.DEBUG)
    
    rate_limit_logger = logging.getLogger('rate_limiting')
    cache_logger = logging.getLogger('caching')
    
    logging.info("✅ Enhanced logging configured")
```

### **Status Dashboard Integration**

```python
# For Streamlit integration
def display_enhanced_status_dashboard(rag_agent):
    """Display enhanced status dashboard in Streamlit sidebar"""
    
    if hasattr(rag_agent, 'get_system_status'):
        status = rag_agent.get_system_status()
        
        st.sidebar.markdown("## 🚀 Enhanced Features")
        
        if status.get("enhanced_features", False):
            st.sidebar.success("✅ Enhancements Active")
            
            # Rate limiting status
            rate_status = status.get("rate_limiting", {})
            st.sidebar.markdown("### ⏱️ Rate Limiting")
            st.sidebar.metric(
                "Daily Requests", 
                f"{rate_status.get('daily_requests_used', 0)}/{rate_status.get('daily_limit', 500)}"
            )
            
            if rate_status.get("is_optimal_time", False):
                st.sidebar.info("🌙 Optimal time for large queries")
            
            # Caching status  
            cache_status = status.get("caching", {})
            if cache_status.get("caching_enabled", False):
                st.sidebar.markdown("### 💾 Cache Status")
                st.sidebar.metric("Valid Caches", cache_status.get("valid_caches", 0))
                st.sidebar.metric("Cache Size", f"{cache_status.get('total_cache_size_mb', 0):.1f} MB")
        else:
            st.sidebar.info("Standard mode (enhancements disabled)")
```

---

## **Deployment Integration**

### **Gradual Feature Rollout**

```python
# src/feature_flags.py
import os
from typing import Dict, Any

class FeatureFlags:
    """Feature flags for gradual rollout of enhancements"""
    
    @staticmethod
    def get_flags() -> Dict[str, bool]:
        return {
            "rate_limiting_enabled": os.getenv("FF_RATE_LIMITING", "true").lower() == "true",
            "caching_enabled": os.getenv("FF_CACHING", "true").lower() == "true", 
            "cache_warming_enabled": os.getenv("FF_CACHE_WARMING", "false").lower() == "true",
            "enhanced_logging_enabled": os.getenv("FF_ENHANCED_LOGGING", "true").lower() == "true",
            "status_dashboard_enabled": os.getenv("FF_STATUS_DASHBOARD", "true").lower() == "true"
        }
    
    @staticmethod
    def is_enabled(flag_name: str) -> bool:
        flags = FeatureFlags.get_flags()
        return flags.get(flag_name, False)

# Usage in your enhanced scraper
def __init__(self, api_token: str, **kwargs):
    # Check feature flags
    enable_rate_limiting = FeatureFlags.is_enabled("rate_limiting_enabled")
    enable_caching = FeatureFlags.is_enabled("caching_enabled")
    
    super().__init__(api_token, 
                    enable_rate_limiting=enable_rate_limiting,
                    enable_caching=enable_caching)
```

---

## **Next Steps After Implementation**

Once this Phase 1 implementation is complete and tested:

1. **Validate Compliance**: Ensure rate limiting works correctly with real API calls
2. **Monitor Cache Performance**: Check cache hit rates and storage usage
3. **Test Error Handling**: Verify fallback mechanisms work properly
4. **Measure Performance**: Compare response times with and without caching
5. **Prepare for Phase 2**: Query optimization will build on this foundation

This Phase 1 implementation provides:
- ✅ **NCBI Compliance**: Respects 3 req/sec limits and daily quotas
- ✅ **Intelligent Caching**: 24-hour cache with automatic cleanup
- ✅ **Backward Compatibility**: Existing code continues to work
- ✅ **Production Monitoring**: Status dashboards and comprehensive logging
- ✅ **Gradual Rollout**: Feature flags for safe deployment

The enhanced system maintains your existing medical disclaimer and safety features while adding production-grade reliability and compliance.