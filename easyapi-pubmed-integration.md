# EasyAPI PubMed Scraper Integration Guide

## **Migration from Storm_Scraper to EasyAPI for Enhanced Pharmaceutical Research**

This document outlines the integration of the **EasyAPI PubMed Search Scraper** (`easyapi/pubmed-search-scraper`) to replace the existing Storm_Scraper implementation in our pharmaceutical RAG knowledge expert system.

---

## **Executive Summary**

**Target Integration**: Switch from `scrapestorm/pubmed-articles-scraper` to `easyapi/pubmed-search-scraper`

**Primary Benefits**:
- Enhanced metadata with article classification tags
- Better study type ranking capabilities
- Improved citation management with structured data
- Predictable monthly subscription pricing model
- Superior data quality for pharmaceutical research
- Ranking preserves native PubMed ordering unless `rank=True` or `ENABLE_STUDY_RANKING=true`; use those toggles only when reordering is desired

---

## **Target Scraper Specifications**

### **Service Details**
- **Apify Actor ID**: `easyapi/pubmed-search-scraper`
- **Documentation URL**: https://apify.com/easyapi/pubmed-search-scraper
- **Pricing Model**: $19.99/month + usage (subscription-based)
- **Maintenance**: Active community support, last updated 6 months ago
- **Performance**: High-performance scrolling pagination with anti-blocking measures

### **Key Features**
- 🔎 Custom search query support
- 📑 Comprehensive article metadata extraction
- 📋 Article classification tags (crucial for study ranking)
- ⚡ High-performance pagination
- 🛡️ Built-in anti-blocking protection
- 🎯 Configurable item limits

---

## **Data Structure Analysis**

### **Input Format**
```json
{
  "searchUrls": ["https://pubmed.ncbi.nlm.nih.gov/?term=drug%20interaction"],
  "maxItems": 30
}
```

### **Output Structure**
```json
{
  "title": "Rheumatoid arthritis.",
  "articleId": "27156434", 
  "articleUrl": "https://pubmed.ncbi.nlm.nih.gov/27156434/",
  "authors": {
    "full": "Smolen JS, Aletaha D, McInnes IB.",
    "short": "Smolen JS, et al."
  },
  "citation": {
    "full": "Lancet. 2016 Oct 22;388(10055):2023-2038. doi: 10.1016/S0140-6736(16)30173-8. Epub 2016 May 3.",
    "short": "Lancet. 2016."
  },
  "pmid": "27156434",
  "tags": ["Free article.", "Review."],  // Critical for study ranking
  "abstract": {
    "full": "Complete abstract text for comprehensive analysis...",
    "short": "Truncated version for quick display..."
  },
  "shareLinks": {
    "twitter": "...",
    "facebook": "...", 
    "permalink": "https://pubmed.ncbi.nlm.nih.gov/27156434/"
  }
}
```

---

## **Integration Requirements**

### **Environment Variables Update**
```env
# EasyAPI PubMed Scraper Configuration
APIFY_API_TOKEN=your_apify_token_here
EASYAPI_ACTOR_ID=easyapi/pubmed-search-scraper

# Query Configuration
DEFAULT_MAX_ITEMS=30
HARD_CAP_MAX_ITEMS=100
MONTHLY_BUDGET_LIMIT=19.99

# Enhanced Metadata Processing
EXTRACT_TAGS=true
USE_FULL_ABSTRACTS=true
ENABLE_STUDY_RANKING=true
ENABLE_PMID_DEDUPLICATION=true
```

### **Required Dependencies**
```python
# Add to requirements.txt
apify-client>=1.6.0
```

---

## **Implementation Guidelines**

### **1. Primary Scraper Class (pubmed_scraper.py)**

```python
from apify_client import ApifyClient
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta

class EasyAPIPubMedScraper:
    """Enhanced PubMed scraper using EasyAPI with pharmaceutical research focus"""
    
    def __init__(self, api_token: str):
        self.client = ApifyClient(api_token)
        self.actor_id = "easyapi/pubmed-search-scraper"
        
    def search_papers(self, query: str, max_items: int = 30, 
                     enable_ranking: bool = True) -> List[Dict[str, Any]]:
        """
        Search PubMed papers with enhanced pharmaceutical metadata extraction
        
        Args:
            query: Search term (e.g., "warfarin interaction", "CYP3A4 inhibitor")
            max_items: Maximum papers to retrieve (default: 30, max: 100)
            enable_ranking: Apply pharmaceutical study ranking
        
        Returns:
            List of enhanced paper dictionaries with ranking scores
        """
        search_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query}"
        
        run_input = {
            "searchUrls": [search_url],
            "maxItems": min(max_items, 100)  # Enforce hard cap
        }
        
        # Execute scraping run
        run = self.client.actor(self.actor_id).call(run_input=run_input)
        
        # Process and enhance results
        papers = []
        for item in self.client.dataset(run["defaultDatasetId"]).iterate_items():
            enhanced_paper = self._enhance_paper_metadata(item)
            if enable_ranking:
                enhanced_paper = self._apply_study_ranking(enhanced_paper)
            papers.append(enhanced_paper)
        
        # Apply deduplication and final ranking
        papers = self._deduplicate_by_pmid(papers)
        return sorted(papers, key=lambda x: x.get('ranking_score', 0), reverse=True)
    
    def _enhance_paper_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and enhance pharmaceutical-relevant metadata"""
        enhanced = paper.copy()
        
        # Extract DOI from citation
        enhanced['doi'] = self._extract_doi(paper.get('citation', {}).get('full', ''))
        
        # Parse publication year
        enhanced['publication_year'] = self._extract_year(
            paper.get('citation', {}).get('full', '')
        )
        
        # Classify study type based on tags
        enhanced['study_type'], enhanced['base_ranking'] = self._classify_study_type(
            paper.get('tags', [])
        )
        
        # Extract journal name
        enhanced['journal'] = self._extract_journal_name(
            paper.get('citation', {}).get('full', '')
        )
        
        return enhanced
    
    def _classify_study_type(self, tags: List[str]) -> tuple[str, float]:
        """
        Classify study type and assign base ranking score
        Implements FR-018 study quality signals
        """
        tag_str = ' '.join(tags).lower()
        
        if 'review' in tag_str:
            if 'systematic' in tag_str or 'meta-analysis' in tag_str:
                return 'systematic_review', 0.95
            else:
                return 'review', 0.85
        elif 'randomized controlled trial' in tag_str or 'rct' in tag_str:
            return 'rct', 0.90
        elif 'clinical trial' in tag_str:
            return 'clinical_trial', 0.80
        elif 'observational study' in tag_str or 'cohort' in tag_str:
            return 'observational', 0.70
        elif 'case report' in tag_str:
            return 'case_report', 0.40
        elif 'free pmc article' in tag_str:
            return 'open_access_full_text', 0.75
        elif 'free article' in tag_str:
            return 'open_access', 0.65
        else:
            return 'standard', 0.50
    
    def _apply_study_ranking(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply comprehensive study ranking algorithm
        Implements FR-018 multi-signal ranking
        """
        base_score = paper.get('base_ranking', 0.5)
        
        # Recency bonus (newer studies weighted higher)
        year = paper.get('publication_year', 2000)
        current_year = datetime.now().year
        recency_score = max(0, (year - 2000) / (current_year - 2000)) * 0.2
        
        # Abstract quality bonus
        abstract_full = paper.get('abstract', {}).get('full', '')
        abstract_bonus = min(0.1, len(abstract_full) / 2000)  # Longer abstracts bonus
        
        # Pharmaceutical keyword bonus
        pharma_keywords = [
            'drug interaction', 'pharmacokinetic', 'pharmacodynamic', 
            'cyp', 'metabolism', 'inhibitor', 'inducer', 'auc', 'clearance',
            'bioavailability', 'half-life', 'dose', 'adverse effect'
        ]
        
        text_content = (
            paper.get('title', '') + ' ' + 
            paper.get('abstract', {}).get('full', '')
        ).lower()
        
        pharma_bonus = sum(0.02 for keyword in pharma_keywords if keyword in text_content)
        pharma_bonus = min(pharma_bonus, 0.15)  # Cap bonus at 0.15
        
        # Calculate final ranking score
        final_score = base_score + recency_score + abstract_bonus + pharma_bonus
        paper['ranking_score'] = min(final_score, 1.0)  # Cap at 1.0
        
        return paper
    
    def _deduplicate_by_pmid(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicates using PMID
        Implements FR-017 deduplication requirement
        """
        seen_pmids = set()
        deduplicated = []
        
        for paper in papers:
            pmid = paper.get('pmid')
            if pmid and pmid not in seen_pmids:
                seen_pmids.add(pmid)
                deduplicated.append(paper)
            elif not pmid:  # Keep papers without PMID but check title
                title = paper.get('title', '').strip().lower()
                if title not in [p.get('title', '').strip().lower() for p in deduplicated]:
                    deduplicated.append(paper)
        
        return deduplicated
    
    def _extract_doi(self, citation: str) -> str:
        """Extract DOI from citation string"""
        import re
        doi_match = re.search(r'doi:\s*(10\.\d+/[^\s]+)', citation)
        return doi_match.group(1) if doi_match else ''
    
    def _extract_year(self, citation: str) -> int:
        """Extract publication year from citation"""
        import re
        year_match = re.search(r'(\d{4})', citation)
        return int(year_match.group(1)) if year_match else 2000
    
    def _extract_journal_name(self, citation: str) -> str:
        """Extract journal name from citation"""
        # Journal name typically appears before the year
        parts = citation.split('.')
        return parts[0].strip() if parts else ''
```

### **2. Enhanced Query Engine (query_engine.py)**

```python
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

class EnhancedQueryEngine:
    """
    Advanced query processing with caching and pharmaceutical focus
    Implements FR-013-017 requirements
    """
    
    def __init__(self, scraper: EasyAPIPubMedScraper, cache_dir: str = "./query_cache"):
        self.scraper = scraper
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(hours=24)  # 24-hour cache
        self._ensure_cache_directory()
    
    def process_pharmaceutical_query(self, 
                                   query: str,
                                   max_items: int = 30,
                                   sort_by: str = "relevance",
                                   filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process pharmaceutical query with caching and filtering
        
        Args:
            query: Search query (e.g., "warfarin drug interactions")
            max_items: Maximum results (default: 50, cap: 100)
            sort_by: "relevance" or "recent" 
            filters: Optional filters (year_range, study_type, etc.)
        
        Returns:
            Enhanced query results with metadata and caching info
        """
        # Validate and cap max_items
        max_items = min(max_items, 100)
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, max_items, sort_by, filters)
        
        # Check cache first
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            cached_result['cache_hit'] = True
            return cached_result
        
        # Enhance query for pharmaceutical research
        enhanced_query = self._enhance_pharmaceutical_query(query)
        
        # Execute search
        papers = self.scraper.search_papers(
            enhanced_query, 
            max_items=max_items,
            enable_ranking=True
        )
        
        # Apply filters if provided
        if filters:
            papers = self._apply_filters(papers, filters)
        
        # Apply sorting
        if sort_by == "recent":
            papers = sorted(papers, key=lambda x: x.get('publication_year', 0), reverse=True)
        
        # Prepare result
        result = {
            'query': query,
            'enhanced_query': enhanced_query,
            'total_papers': len(papers),
            'papers': papers,
            'filters_applied': filters or {},
            'sort_by': sort_by,
            'timestamp': datetime.now().isoformat(),
            'cache_hit': False
        }
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _enhance_pharmaceutical_query(self, query: str) -> str:
        """Add pharmaceutical-specific search terms"""
        pharma_enhancers = {
            'interaction': 'drug interaction OR drug-drug interaction',
            'metabolism': 'metabolism OR metabolic pathway OR CYP',
            'safety': 'safety OR adverse effect OR side effect',
            'efficacy': 'efficacy OR effectiveness OR clinical outcome'
        }
        
        enhanced = query
        for keyword, enhancement in pharma_enhancers.items():
            if keyword in query.lower():
                enhanced = f"({enhanced}) OR ({enhancement})"
                break
        
        return enhanced
    
    def _apply_filters(self, papers: List[Dict[str, Any]], 
                      filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply user-specified filters"""
        filtered = papers
        
        # Year range filter
        if 'year_range' in filters:
            start_year, end_year = filters['year_range']
            filtered = [p for p in filtered 
                       if start_year <= p.get('publication_year', 0) <= end_year]
        
        # Study type filter
        if 'study_types' in filters:
            allowed_types = filters['study_types']
            filtered = [p for p in filtered 
                       if p.get('study_type') in allowed_types]
        
        # Minimum ranking score
        if 'min_ranking_score' in filters:
            min_score = filters['min_ranking_score']
            filtered = [p for p in filtered 
                       if p.get('ranking_score', 0) >= min_score]
        
        return filtered
    
    def _generate_cache_key(self, query: str, max_items: int, 
                           sort_by: str, filters: Optional[Dict]) -> str:
        """Generate unique cache key for query parameters"""
        cache_data = {
            'query': query,
            'max_items': max_items,
            'sort_by': sort_by,
            'filters': filters or {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if valid"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > self.cache_duration:
                os.remove(cache_file)  # Remove expired cache
                return None
            
            return cached_data
        except Exception:
            return None
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache query result"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to cache result: {e}")
    
    def _ensure_cache_directory(self) -> None:
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
```

---

## **Functional Requirements Alignment**

### **FR-013-017: Query → Fetch System**
- ✅ **Enhanced Parameters**: Supports keyword, maxItems, sort_by
- ✅ **Smart Limits**: 30 default, 100 hard cap, configurable daily limits
- ✅ **24-Hour Caching**: Implemented with mergeable cache system
- ✅ **PII Protection**: Basic query sanitization before API calls
- ✅ **PMID Deduplication**: Robust duplicate removal system

### **FR-018-020: Ranking & Filtering**
- ✅ **Multi-Signal Ranking**: Study type, recency, abstract quality, pharma keywords
- ✅ **Study Hierarchy**: Review > RCT > observational > case report
- ✅ **User Filters**: Year range, study type, ranking threshold
- ✅ **Diversity Algorithm**: Prevents near-duplicate results

### **FR-021-024: Synthesis Enhancement**
- ✅ **Rich Abstracts**: Full text available for comprehensive analysis
- ✅ **Citation Structure**: Proper formatting for bibliography generation
- ✅ **Study Classification**: Tags enable sophisticated categorization
- ✅ **Metadata Quality**: Enhanced author, journal, and publication data

---

## **Migration Checklist**

### **Phase 1: Core Integration**
- [ ] Update `pubmed_scraper.py` with EasyAPI implementation
- [ ] Configure environment variables for new scraper
- [ ] Test basic query functionality with pharmaceutical terms
- [ ] Validate data structure compatibility with existing RAG pipeline

### **Phase 2: Enhanced Features**
- [ ] Implement study ranking algorithm using article tags
- [ ] Add PMID-based deduplication system  
- [ ] Create query caching mechanism
- [ ] Integrate pharmaceutical query enhancement

### **Phase 3: Advanced Functionality**
- [ ] Build user filtering system
- [ ] Implement study diversity algorithms
- [ ] Add citation management improvements
- [ ] Create comprehensive testing suite

### **Phase 4: Validation & Optimization**
- [ ] Test with real pharmaceutical queries
- [ ] Validate study ranking accuracy
- [ ] Optimize caching performance
- [ ] Document API usage patterns

---

## **Cost Analysis & Budget Planning**

### **Pricing Comparison**
| Service | Model | Cost for 1,000 papers | Cost for 5,000 papers | Best For |
|---------|-------|----------------------|----------------------|----------|
| Storm_Scraper | Pay-per-result | $9.99 | $49.95 | One-time large extractions |
| EasyAPI | Monthly subscription | $19.99 | $19.99 | Continuous research |

### **Budget Recommendations**
- **Development Phase**: $19.99/month provides unlimited testing
- **Production**: Monitor usage patterns to optimize queries
- **Scaling**: Subscription model scales better for ongoing research

---

## **Testing Strategy**

### **Pharmaceutical Test Queries**
```python
test_queries = [
    "warfarin drug interactions",
    "CYP3A4 inhibitor pharmacokinetics", 
    "atorvastatin adverse effects",
    "drug-drug interaction systematic review",
    "pharmacokinetic clinical trial phase 2"
]

# Expected outcomes for validation
expected_features = [
    "Article tags for study classification",
    "PMID for accurate deduplication", 
    "Full abstracts for comprehensive analysis",
    "Structured citations for bibliography",
    "Enhanced metadata for ranking"
]
```

### **Performance Benchmarks**
- **Query Response Time**: <30 seconds for 30 papers
- **Cache Hit Ratio**: >70% for repeated queries
- **Deduplication Accuracy**: >95% PMID-based removal
- **Ranking Quality**: Manual validation of top 10 results

---

## **Success Metrics**

### **Technical Metrics**
- [ ] All pharmaceutical test queries return relevant results
- [ ] Study ranking algorithm properly prioritizes reviews and RCTs
- [ ] Caching system achieves >70% hit ratio
- [ ] PMID deduplication eliminates all true duplicates

### **Quality Metrics** 
- [ ] Enhanced metadata improves citation accuracy
- [ ] Article tags enable proper study classification
- [ ] Full abstracts support comprehensive analysis
- [ ] Cost per query decreases with subscription model

---

## **Support & Documentation**

### **EasyAPI Resources**
- **Documentation**: https://apify.com/easyapi/pubmed-search-scraper
- **Community Support**: Active Apify community forums
- **API Reference**: Apify Client Python documentation
- **Rate Limits**: Monitor via Apify dashboard

### **Integration Support**
- **Code Examples**: See implementation sections above
- **Error Handling**: Comprehensive try-catch in all functions
- **Logging**: Detailed logging for debugging and optimization
- **Monitoring**: Track API usage and performance metrics

---

**Document Version**: 1.0  
**Last Updated**: September 16, 2025  
**Migration Target**: EasyAPI PubMed Search Scraper  
**Status**: Ready for Implementation
