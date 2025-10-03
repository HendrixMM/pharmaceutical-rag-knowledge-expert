# EasyAPI PubMed Scraper - Best Practices & Optimization Guide

## **Comprehensive Query Optimization for Pharmaceutical Research**

This guide provides advanced strategies for optimizing your pharmaceutical RAG system's interaction with the EasyAPI PubMed Search Scraper, implementing NCBI guidelines and pharmaceutical research best practices.

---

## **Executive Summary**

Based on the EasyAPI PubMed Search Scraper specifications and NCBI E-utilities guidelines, this guide optimizes your pharmaceutical RAG system for maximum efficiency, compliance, and research quality.

**Key Optimization Areas**:

- Query construction using PubMed search tags and field tags
- Rate limiting and caching strategies for sustainable API usage
- Pharmaceutical-specific search optimization techniques
- NCBI compliance and best practices implementation

---

## **NCBI API Guidelines & Rate Limiting**

### **Core Usage Requirements**[118][115]

**Rate Limits (Critical for Implementation)**:

- **Without API Key**: Maximum 3 requests per second[118]
- **With NCBI API Key**: Up to 10 requests per second[118]
- **Enhanced Access**: Available by request for higher rates[112]
- **Optimal Timing**: Large jobs between 9 PM - 5 AM EST weekdays[118]

**EasyAPI Implementation Strategy**:

```python
import time
import asyncio
from typing import Dict, List, Any
from datetime import datetime, timedelta

class OptimizedPubMedScraper:
    """Optimized EasyAPI PubMed scraper with NCBI compliance"""

    def __init__(self, apify_token: str, max_concurrent: int = 2):
        self.apify_token = apify_token
        self.max_concurrent = max_concurrent
        self.request_interval = 1.0  # 1 second between requests (conservative)
        self.last_request_time = 0
        self.daily_request_count = 0
        self.daily_limit = 500  # Conservative daily limit

    async def rate_limited_request(self):
        """Implement rate limiting for NCBI compliance"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)

        self.last_request_time = time.time()
        self.daily_request_count += 1

        if self.daily_request_count >= self.daily_limit:
            raise Exception("Daily request limit reached")
```

### **Caching Strategy for 12-24 Hour Periods**[115]

```python
import json
import hashlib
from datetime import datetime, timedelta

class PubMedQueryCache:
    """12-24 hour caching system for PubMed queries"""

    def __init__(self, cache_duration_hours: int = 24):
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_dir = "./pubmed_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self, query: str, max_items: int, sort_by: str) -> str:
        """Generate unique cache key for query parameters"""
        cache_data = {
            "query": query.lower().strip(),
            "max_items": max_items,
            "sort_by": sort_by
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

    def is_cache_valid(self, cache_file: str) -> bool:
        """Check if cached result is still valid"""
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            return datetime.now() - cached_time < self.cache_duration
        except:
            return False
```

---

## **PubMed Search Tags & Field Optimization**

### **Essential Field Tags for Pharmaceutical Research**[113][110]

Based on PubMed's field tag system, optimize queries for pharmaceutical content:

#### **Core Pharmaceutical Field Tags**

```python
class PharmaceuticalSearchTags:
    """PubMed field tags optimized for pharmaceutical research"""

    # Primary content fields
    TITLE_ABSTRACT = "[tiab]"           # Title/Abstract - most comprehensive
    TITLE_ONLY = "[ti]"                 # Title only - high precision
    MESH_TERMS = "[mh]"                 # Medical Subject Headings
    MESH_MAJOR = "[majr]"               # Major MeSH topics only

    # Drug-specific fields
    PHARMACOLOGICAL_ACTION = "[pa]"      # Drug mechanisms and effects
    SUPPLEMENTARY_CONCEPT = "[nm]"       # Drug names and chemicals

    # Publication type filters
    PUBLICATION_TYPE = "[pt]"            # Clinical trial, review, etc.

    # Quality indicators
    JOURNAL = "[ta]"                     # Journal name for impact factor filtering
    AUTHOR = "[au]"                      # Author expertise filtering

    # Date and recency
    DATE_PUBLICATION = "[dp]"            # Publication date
    DATE_ENTRY = "[edat]"               # PubMed entry date

    # Study design indicators
    TEXT_WORDS = "[tw]"                 # All text fields
```

#### **Pharmaceutical Query Enhancement Examples**

```python
class PharmaceuticalQueryOptimizer:
    """Optimize queries for pharmaceutical research using PubMed field tags"""

    def optimize_drug_interaction_query(self, drug1: str, drug2: str) -> str:
        """Optimize drug-drug interaction queries"""
        # Use multiple approaches for comprehensive coverage
        base_query = f"({drug1} AND {drug2})"

        # Add pharmaceutical-specific terms
        interaction_terms = [
            "drug interaction[tiab]",
            "drug-drug interaction[tiab]",
            "pharmacokinetic interaction[tiab]",
            "CYP[tiab]",
            "metabolism[tiab]"
        ]

        # Combine with OR for broader coverage
        enhanced_query = f"{base_query} AND ({' OR '.join(interaction_terms)})"

        # Focus on high-quality sources
        quality_filters = [
            "systematic review[pt]",
            "randomized controlled trial[pt]",
            "clinical trial[pt]"
        ]

        return f"({enhanced_query}) AND ({' OR '.join(quality_filters)})"

    def optimize_drug_safety_query(self, drug_name: str) -> str:
        """Optimize drug safety and adverse effect queries"""
        safety_terms = [
            "adverse effects[sh]",      # MeSH subheading
            "toxicity[sh]",             # MeSH subheading
            "side effect[tiab]",
            "adverse event[tiab]",
            "safety[tiab]",
            "contraindication[tiab]"
        ]

        return f"{drug_name}[tiab] AND ({' OR '.join(safety_terms)})"

    def optimize_mechanism_query(self, drug_name: str) -> str:
        """Optimize mechanism of action queries"""
        mechanism_terms = [
            "mechanism of action[tiab]",
            "pharmacology[sh]",         # MeSH subheading
            "pharmacokinetics[sh]",     # MeSH subheading
            "pharmacodynamics[tiab]",
            "molecular mechanism[tiab]",
            "target[tiab]",
            "receptor[tiab]"
        ]

        return f"{drug_name}[tiab] AND ({' OR '.join(mechanism_terms)})"
```

### **Advanced Search Construction Strategies**[113][119]

#### **Boolean Logic Optimization for Pharmaceutical Queries**

```python
class AdvancedQueryBuilder:
    """Build complex pharmaceutical queries using PubMed Boolean logic"""

    def build_comprehensive_drug_query(self, drug_name: str,
                                     aspects: List[str] = None) -> str:
        """Build comprehensive drug research query"""
        if aspects is None:
            aspects = ["safety", "efficacy", "mechanism", "interactions"]

        # Base drug identification (multiple approaches)
        drug_identification = [
            f'"{drug_name}"[tiab]',           # Exact phrase in title/abstract
            f'{drug_name}[nm]',               # Supplementary concept (drug names)
            f'{drug_name}[pa]'                # Pharmacological action
        ]

        drug_base = f"({' OR '.join(drug_identification)})"

        # Aspect-specific terms
        aspect_queries = []

        if "safety" in aspects:
            safety_query = self._build_safety_query()
            aspect_queries.append(f"({safety_query})")

        if "efficacy" in aspects:
            efficacy_query = self._build_efficacy_query()
            aspect_queries.append(f"({efficacy_query})")

        if "mechanism" in aspects:
            mechanism_query = self._build_mechanism_query()
            aspect_queries.append(f"({mechanism_query})")

        if "interactions" in aspects:
            interaction_query = self._build_interaction_query()
            aspect_queries.append(f"({interaction_query})")

        # Combine all aspects with OR
        combined_aspects = f"({' OR '.join(aspect_queries)})"

        return f"{drug_base} AND {combined_aspects}"

    def _build_safety_query(self) -> str:
        """Build safety-focused sub-query"""
        return ' OR '.join([
            '"adverse effects"[sh]',        # MeSH subheading
            '"toxicity"[sh]',               # MeSH subheading
            '"side effects"[tiab]',
            '"adverse events"[tiab]',
            '"contraindications"[tiab]',
            '"warnings"[tiab]',
            '"black box"[tiab]'
        ])

    def _build_efficacy_query(self) -> str:
        """Build efficacy-focused sub-query"""
        return ' OR '.join([
            '"therapeutic use"[sh]',        # MeSH subheading
            '"treatment outcome"[mh]',      # MeSH term
            '"efficacy"[tiab]',
            '"effectiveness"[tiab]',
            '"clinical outcome"[tiab]',
            '"therapeutic effect"[tiab]'
        ])
```

---

## **Specific Search Term Optimization**

### **Pharmaceutical Research Term Strategies**[110][114]

#### **Truncation and Wildcard Usage for Drug Names**

```python
class PharmaceuticalTermOptimizer:
    """Optimize pharmaceutical terms using PubMed wildcards and truncation"""

    def optimize_drug_name_variations(self, base_drug_name: str) -> List[str]:
        """Generate drug name variations for comprehensive searching"""
        optimized_terms = []

        # Exact drug name
        optimized_terms.append(f'"{base_drug_name}"[tiab]')

        # Truncation for drug name variations (minimum 4 characters)
        if len(base_drug_name) >= 4:
            optimized_terms.append(f'{base_drug_name}*[tiab]')

        # Generic vs brand name considerations
        # Add common suffixes/prefixes for pharmaceutical compounds
        pharma_suffixes = ['ate', 'ine', 'ide', 'ium', 'acid']
        pharma_prefixes = ['meta', 'para', 'ortho', 'anti', 'pro']

        for suffix in pharma_suffixes:
            if base_drug_name.endswith(suffix[:3]):
                optimized_terms.append(f'{base_drug_name[:-3]}*[tiab]')

        return optimized_terms

    def build_dosage_form_query(self, drug_name: str) -> str:
        """Include dosage forms in drug queries"""
        dosage_forms = [
            "tablet*", "capsule*", "injection*", "solution*",
            "suspension*", "cream*", "ointment*", "patch*",
            "inhalation*", "suppository*", "oral", "topical",
            "intravenous", "intramuscular", "subcutaneous"
        ]

        dosage_query = ' OR '.join([f'{form}[tiab]' for form in dosage_forms])
        return f'({drug_name}[tiab]) AND ({dosage_query})'
```

#### **MeSH Term Integration for Precision**[113][122]

```python
class MeSHOptimizer:
    """Optimize queries using Medical Subject Headings (MeSH) terms"""

    def integrate_mesh_terms(self, drug_query: str,
                           include_subheadings: bool = True) -> str:
        """Integrate relevant MeSH terms for pharmaceutical queries"""

        # Common pharmaceutical MeSH terms
        pharma_mesh_terms = [
            '"Pharmaceutical Preparations"[mh]',
            '"Drug Therapy"[mh]',
            '"Pharmacokinetics"[mh]',
            '"Pharmacodynamics"[mh]',
            '"Drug Interactions"[mh]',
            '"Adverse Drug Reaction Reporting Systems"[mh]'
        ]

        if include_subheadings:
            # Add relevant subheadings for comprehensive coverage
            mesh_with_subheadings = [
                '"Pharmaceutical Preparations/adverse effects"[majr]',
                '"Pharmaceutical Preparations/pharmacokinetics"[majr]',
                '"Pharmaceutical Preparations/therapeutic use"[majr]',
                '"Drug Therapy/adverse effects"[majr]',
                '"Drug Therapy/methods"[majr]'
            ]
            pharma_mesh_terms.extend(mesh_with_subheadings)

        mesh_query = f"({' OR '.join(pharma_mesh_terms)})"
        return f"({drug_query}) AND {mesh_query}"
```

---

## **Query Size & Batch Processing Optimization**

### **Breaking Large Searches into Smaller Queries**[1]

Based on EasyAPI recommendations and NCBI guidelines:

```python
class QueryBatchProcessor:
    """Process large pharmaceutical queries in optimal batches"""

    def __init__(self, max_items_per_query: int = 30, max_total_items: int = 200):
        self.max_items_per_query = max_items_per_query  # EasyAPI example uses 30
        self.max_total_items = max_total_items
        self.batch_delay = 2.0  # 2 second delay between batches

    async def process_large_pharmaceutical_search(self,
                                                base_query: str,
                                                total_items_needed: int) -> List[Dict]:
        """Process large pharmaceutical searches in batches"""

        if total_items_needed <= self.max_items_per_query:
            return await self._single_query(base_query, total_items_needed)

        # Break into multiple targeted queries
        batches = self._create_query_batches(base_query, total_items_needed)

        all_results = []
        for i, (query, max_items) in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}: {query[:50]}...")

            batch_results = await self._single_query(query, max_items)
            all_results.extend(batch_results)

            # Rate limiting delay
            if i < len(batches) - 1:  # Don't delay after last batch
                await asyncio.sleep(self.batch_delay)

        # Remove duplicates based on PMID
        return self._deduplicate_results(all_results)

    def _create_query_batches(self, base_query: str,
                            total_items: int) -> List[Tuple[str, int]]:
        """Create focused query batches for pharmaceutical research"""

        # Strategy: Use different aspects/filters for each batch
        batch_strategies = [
            ("systematic review[pt] OR meta-analysis[pt]", "reviews"),
            ("randomized controlled trial[pt]", "rcts"),
            ("clinical trial[pt]", "clinical_trials"),
            ("humans[mh] AND last 5 years[dp]", "recent_human"),
            ("adverse effects[sh] OR toxicity[sh]", "safety"),
            ("therapeutic use[sh] OR pharmacology[sh]", "efficacy")
        ]

        batches = []
        items_per_batch = min(self.max_items_per_query,
                            total_items // len(batch_strategies))

        for filter_term, description in batch_strategies:
            enhanced_query = f"({base_query}) AND ({filter_term})"
            batches.append((enhanced_query, items_per_batch))

            if len(batches) * items_per_batch >= total_items:
                break

        return batches
```

### **Optimal Runtime Allocation**[1]

```python
class RuntimeOptimizer:
    """Optimize scraping runtime for pharmaceutical research"""

    def __init__(self):
        self.optimal_times = {
            "large_jobs_start": "21:00",  # 9 PM EST
            "large_jobs_end": "05:00",    # 5 AM EST
            "weekend_anytime": True
        }

    def is_optimal_time(self) -> bool:
        """Check if current time is optimal for large PubMed jobs"""
        from datetime import datetime

        now = datetime.now()
        current_hour = now.hour
        is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6

        if is_weekend:
            return True

        # Weekday: optimal between 9 PM and 5 AM
        if current_hour >= 21 or current_hour <= 5:
            return True

        return False

    def calculate_optimal_batch_size(self, total_queries: int) -> Dict[str, int]:
        """Calculate optimal batch sizes based on current time"""

        if self.is_optimal_time():
            # Larger batches during optimal hours
            return {
                "batch_size": 50,
                "concurrent_requests": 3,
                "delay_between_batches": 1.0
            }
        else:
            # Smaller, more conservative batches during peak hours
            return {
                "batch_size": 20,
                "concurrent_requests": 1,
                "delay_between_batches": 3.0
            }
```

---

## **Pharmaceutical-Specific Query Patterns**

### **Clinical Research Query Templates**

```python
class ClinicalQueryTemplates:
    """Pre-optimized query templates for common pharmaceutical research"""

    @staticmethod
    def drug_safety_profile(drug_name: str, years: int = 10) -> str:
        """Comprehensive drug safety profile query"""
        return f'''
        ({drug_name}[tiab] OR {drug_name}[nm]) AND
        (
            "adverse effects"[sh] OR "toxicity"[sh] OR
            "side effects"[tiab] OR "adverse events"[tiab] OR
            "contraindications"[tiab] OR "black box warning"[tiab] OR
            "drug safety"[tiab] OR "pharmacovigilance"[tiab]
        ) AND
        (
            "systematic review"[pt] OR "meta-analysis"[pt] OR
            "randomized controlled trial"[pt] OR "clinical trial"[pt]
        ) AND
        last {years} years[dp] AND humans[mh]
        '''

    @staticmethod
    def drug_drug_interactions(drug1: str, drug2: str) -> str:
        """Drug-drug interaction focused query"""
        return f'''
        (({drug1}[tiab] OR {drug1}[nm]) AND ({drug2}[tiab] OR {drug2}[nm])) AND
        (
            "drug interactions"[mh] OR "drug interaction"[tiab] OR
            "CYP"[tiab] OR "cytochrome"[tiab] OR "metabolism"[tiab] OR
            "pharmacokinetic interaction"[tiab] OR "pharmacodynamic interaction"[tiab]
        ) AND
        (
            "case reports"[pt] OR "clinical trial"[pt] OR
            "pharmacokinetics"[sh] OR "metabolism"[sh]
        )
        '''

    @staticmethod
    def clinical_efficacy_studies(drug_name: str, condition: str) -> str:
        """Clinical efficacy studies query"""
        return f'''
        ({drug_name}[tiab] OR {drug_name}[nm]) AND
        ({condition}[tiab] OR {condition}[mh]) AND
        (
            "therapeutic use"[sh] OR "treatment outcome"[mh] OR
            "efficacy"[tiab] OR "effectiveness"[tiab] OR
            "clinical response"[tiab] OR "treatment response"[tiab]
        ) AND
        (
            "randomized controlled trial"[pt] OR "controlled clinical trial"[pt] OR
            "clinical trial, phase ii"[pt] OR "clinical trial, phase iii"[pt]
        ) AND
        humans[mh] AND english[la]
        '''
```

### **Pharmaceutical Company Research Patterns**

```python
class PharmaCompanyQueries:
    """Specialized queries for pharmaceutical industry research"""

    @staticmethod
    def competitive_intelligence(drug_class: str, exclude_company: str = None) -> str:
        """Competitive intelligence query for drug classes"""
        base_query = f'''
        {drug_class}[tiab] AND
        (
            "clinical trial"[pt] OR "clinical trial, phase i"[pt] OR
            "clinical trial, phase ii"[pt] OR "clinical trial, phase iii"[pt] OR
            "clinical trial, phase iv"[pt]
        ) AND
        (
            "sponsor"[tiab] OR "funded by"[tiab] OR "supported by"[tiab] OR
            "pharmaceutical"[ad] OR "pharma"[ad] OR "inc"[ad] OR "ltd"[ad]
        ) AND
        last 3 years[dp]
        '''

        if exclude_company:
            base_query += f' NOT "{exclude_company}"[ad]'

        return base_query

    @staticmethod
    def patent_landscape(drug_name: str) -> str:
        """Patent and IP landscape query"""
        return f'''
        {drug_name}[tiab] AND
        (
            "patent"[tiab] OR "intellectual property"[tiab] OR
            "exclusivity"[tiab] OR "generic"[tiab] OR
            "biosimilar"[tiab] OR "market entry"[tiab]
        ) AND
        (
            "review"[pt] OR "editorial"[pt] OR "news"[pt] OR
            "pharmaceutical industry"[mh] OR "patents as topic"[mh]
        )
        '''

    @staticmethod
    def regulatory_filing_research(drug_name: str) -> str:
        """Regulatory filing and approval research"""
        return f'''
        {drug_name}[tiab] AND
        (
            "FDA approval"[tiab] OR "drug approval"[tiab] OR
            "regulatory"[tiab] OR "ANDA"[tiab] OR "NDA"[tiab] OR
            "BLA"[tiab] OR "breakthrough therapy"[tiab] OR
            "fast track"[tiab] OR "priority review"[tiab]
        ) AND
        (
            "United States Food and Drug Administration"[mh] OR
            "Drug Approval"[mh] OR "legislation and jurisprudence"[sh]
        )
        '''
```

---

## **Integration with Your RAG System**

### **Enhanced EasyAPI Implementation**

```python
# Complete implementation combining all optimization strategies

class OptimizedPharmaceuticalPubMedScraper:
    """Production-ready optimized PubMed scraper for pharmaceutical RAG"""

    def __init__(self, apify_token: str, cache_duration_hours: int = 24):
        self.apify_token = apify_token
        self.cache = PubMedQueryCache(cache_duration_hours)
        self.rate_limiter = OptimizedPubMedScraper(apify_token)
        self.query_optimizer = PharmaceuticalQueryOptimizer()
        self.batch_processor = QueryBatchProcessor()
        self.template_engine = ClinicalQueryTemplates()

    async def search_pharmaceutical_papers(self,
                                         query: str,
                                         max_items: int = 30,
                                         query_type: str = "general",
                                         enable_optimization: bool = True) -> Dict[str, Any]:
        """
        Optimized pharmaceutical paper search with all best practices

        Args:
            query: Base search query
            max_items: Maximum papers to retrieve (EasyAPI limit: 100)
            query_type: Type of pharmaceutical query for optimization
            enable_optimization: Whether to apply pharmaceutical optimizations

        Returns:
            Enhanced search results with pharmaceutical metadata
        """

        # 1. Query Optimization
        if enable_optimization:
            optimized_query = await self._optimize_pharmaceutical_query(
                query, query_type
            )
        else:
            optimized_query = query

        # 2. Cache Check
        cache_key = self.cache.get_cache_key(optimized_query, max_items, "relevance")
        cached_result = await self._get_cached_result(cache_key)

        if cached_result:
            logger.info("Retrieved results from cache")
            return cached_result

        # 3. Rate-Limited Execution
        await self.rate_limiter.rate_limited_request()

        # 4. Batch Processing for Large Queries
        if max_items > 30:
            results = await self.batch_processor.process_large_pharmaceutical_search(
                optimized_query, max_items
            )
        else:
            results = await self._execute_single_query(optimized_query, max_items)

        # 5. Pharmaceutical Enhancement
        enhanced_results = await self._enhance_pharmaceutical_results(results)

        # 6. Cache Results
        await self._cache_results(cache_key, enhanced_results)

        return enhanced_results

    async def _optimize_pharmaceutical_query(self, query: str,
                                           query_type: str) -> str:
        """Apply pharmaceutical-specific query optimizations"""

        optimization_map = {
            "drug_safety": self.query_optimizer.optimize_drug_safety_query,
            "drug_interactions": self._extract_and_optimize_interaction_query,
            "mechanism": self.query_optimizer.optimize_mechanism_query,
            "clinical_efficacy": self._optimize_efficacy_query,
            "competitive_intel": self._optimize_competitive_query
        }

        if query_type in optimization_map:
            return optimization_map[query_type](query)
        else:
            # General pharmaceutical enhancement
            return self._apply_general_pharma_filters(query)

    def _apply_general_pharma_filters(self, query: str) -> str:
        """Apply general pharmaceutical research filters"""
        pharma_filters = [
            "humans[mh]",  # Human studies preferred
            "(english[la] OR has abstract[filter])",  # Language accessibility
            "last 20 years[dp]"  # Recent and relevant
        ]

        return f"({query}) AND ({' AND '.join(pharma_filters)})"

    async def _enhance_pharmaceutical_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Add pharmaceutical-specific metadata to results"""

        enhanced_results = []

        for paper in results:
            enhanced_paper = paper.copy()

            # Extract pharmaceutical metadata
            enhanced_paper.update({
                "pharmaceutical_relevance_score": self._calculate_pharma_relevance(paper),
                "study_type_classification": self._classify_study_type(paper),
                "drug_mentions": self._extract_drug_mentions(paper),
                "clinical_phase": self._identify_clinical_phase(paper),
                "regulatory_mentions": self._identify_regulatory_content(paper)
            })

            enhanced_results.append(enhanced_paper)

        # Sort by pharmaceutical relevance
        enhanced_results.sort(
            key=lambda x: x["pharmaceutical_relevance_score"],
            reverse=True
        )

        return {
            "papers": enhanced_results,
            "total_count": len(enhanced_results),
            "pharmaceutical_metadata": {
                "avg_relevance_score": sum(p["pharmaceutical_relevance_score"]
                                         for p in enhanced_results) / len(enhanced_results),
                "study_type_distribution": self._analyze_study_types(enhanced_results),
                "top_drug_mentions": self._analyze_drug_mentions(enhanced_results)
            },
            "query_optimization_applied": True,
            "cache_status": "fresh"
        }
```

---

## **Monitoring & Usage Analytics**

### **PubMed Guidelines Compliance Monitoring**

```python
class PubMedComplianceMonitor:
    """Monitor compliance with PubMed/NCBI usage guidelines"""

    def __init__(self):
        self.daily_requests = 0
        self.request_timestamps = []
        self.compliance_log = []

    def log_request(self, query_type: str, items_requested: int):
        """Log request for compliance monitoring"""
        timestamp = datetime.now()

        self.daily_requests += 1
        self.request_timestamps.append(timestamp)

        # Check rate compliance (3 requests/second limit)
        recent_requests = [
            ts for ts in self.request_timestamps
            if timestamp - ts < timedelta(seconds=1)
        ]

        compliance_status = {
            "timestamp": timestamp.isoformat(),
            "query_type": query_type,
            "items_requested": items_requested,
            "daily_total": self.daily_requests,
            "requests_last_second": len(recent_requests),
            "rate_compliant": len(recent_requests) <= 3,
            "timing_optimal": self._is_optimal_timing()
        }

        self.compliance_log.append(compliance_status)

        if not compliance_status["rate_compliant"]:
            logger.warning("Rate limit compliance violation detected")

        return compliance_status

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report for monitoring"""
        if not self.compliance_log:
            return {"status": "no_data"}

        recent_violations = [
            entry for entry in self.compliance_log[-100:]  # Last 100 requests
            if not entry["rate_compliant"]
        ]

        return {
            "total_requests_today": self.daily_requests,
            "rate_violations_count": len(recent_violations),
            "compliance_rate": (len(self.compliance_log) - len(recent_violations)) / len(self.compliance_log),
            "optimal_timing_usage": sum(1 for entry in self.compliance_log if entry["timing_optimal"]) / len(self.compliance_log),
            "recommendations": self._generate_compliance_recommendations()
        }
```

---

## **Production Deployment Checklist**

### **EasyAPI Integration Verification**

- [ ] **Query Optimization**

  - [ ] Pharmaceutical field tags implemented
  - [ ] MeSH term integration active
  - [ ] Boolean logic optimized for drug research
  - [ ] Truncation and wildcards properly used

- [ ] **Rate Limiting & Caching**

  - [ ] 24-hour cache system implemented
  - [ ] Rate limiting respects NCBI guidelines (≤3 req/sec)
  - [ ] Optimal timing detection for large jobs
  - [ ] Daily usage monitoring active

- [ ] **Batch Processing**

  - [ ] Large queries split into focused batches
  - [ ] Deduplication by PMID implemented
  - [ ] Concurrent request limiting configured
  - [ ] Error handling for API failures

- [ ] **Pharmaceutical Enhancement**

  - [ ] Study type classification working
  - [ ] Drug mention extraction active
  - [ ] Clinical phase identification implemented
  - [ ] Regulatory content detection functional

- [ ] **Compliance Monitoring**
  - [ ] Usage analytics dashboard active
  - [ ] Compliance violation alerts configured
  - [ ] Performance metrics tracking enabled
  - [ ] Error logging comprehensive

---

## **Cost Optimization Strategies**

### **EasyAPI Budget Management**

Based on EasyAPI's $19.99/month + usage model:

```python
class EasyAPIBudgetOptimizer:
    """Optimize costs for EasyAPI PubMed scraper usage"""

    def __init__(self, monthly_budget: float = 50.0):
        self.monthly_budget = monthly_budget
        self.base_cost = 19.99  # Monthly subscription
        self.available_usage_budget = monthly_budget - self.base_cost

    def calculate_optimal_usage_pattern(self, research_priorities: List[str]) -> Dict[str, int]:
        """Calculate optimal monthly usage pattern"""

        # Priority-based allocation
        priority_weights = {
            "drug_safety": 0.3,       # High priority - safety critical
            "drug_interactions": 0.25, # High priority - clinical relevance
            "clinical_efficacy": 0.2,  # Medium-high priority
            "mechanism_research": 0.15, # Medium priority
            "competitive_intel": 0.1   # Lower priority
        }

        # Estimate queries per priority area
        base_queries_per_month = 200  # Conservative estimate

        allocation = {}
        for priority in research_priorities:
            weight = priority_weights.get(priority, 0.1)
            allocation[priority] = int(base_queries_per_month * weight)

        return {
            "monthly_query_budget": allocation,
            "estimated_monthly_papers": sum(allocation.values()) * 25,  # ~25 papers per query
            "cost_per_query": self.available_usage_budget / sum(allocation.values()),
            "optimization_recommendations": self._generate_cost_recommendations()
        }

    def _generate_cost_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations"""
        return [
            "Use 24-hour caching to reduce duplicate queries",
            "Batch related queries to maximize paper retrieval per request",
            "Focus on high-impact journals for better ROI",
            "Schedule large research jobs during optimal hours",
            "Implement query result sharing across research teams"
        ]
```

---

## **Conclusion**

This comprehensive optimization guide ensures your pharmaceutical RAG system maximizes the value of EasyAPI PubMed Search Scraper while maintaining full compliance with NCBI guidelines. Key implementation priorities:

1. **Query Optimization**: Use pharmaceutical-specific field tags and MeSH terms
2. **Rate Limiting**: Respect NCBI's 3 requests/second limit with proper caching
3. **Batch Processing**: Break large searches into focused, manageable queries
4. **Cost Management**: Optimize usage patterns for maximum research value
5. **Compliance Monitoring**: Track usage patterns and maintain NCBI guidelines

By implementing these strategies, your pharmaceutical RAG system will deliver high-quality, cost-effective research results while maintaining sustainable and compliant API usage patterns.

---

**Document Version**: 1.0
**Integration Target**: EasyAPI PubMed Search Scraper
**Compliance**: NCBI E-utilities Guidelines
**Focus**: Pharmaceutical Research Optimization
