"""
Pharmaceutical Model Optimization and Domain-Specific Prompt Engineering

Advanced optimization for embedding and chat models specifically tailored for
pharmaceutical research, including medical terminology, drug nomenclature,
clinical context understanding, and regulatory compliance language.

Features:
- Pharmaceutical-specific embedding optimization
- Medical terminology-aware chat model configuration
- Drug nomenclature and clinical context prompting
- Regulatory compliance language optimization
- Safety-critical response formatting
- Clinical decision support prompt templates

Integrates with the NVIDIA Build cloud-first architecture to ensure optimal
pharmaceutical research performance and accuracy.
"""
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

try:
    from ..enhanced_config import EnhancedRAGConfig
    from .query_classifier import PharmaceuticalContext, PharmaceuticalDomain
except ImportError:
    from src.enhanced_config import EnhancedRAGConfig
    from src.pharmaceutical.query_classifier import PharmaceuticalContext, PharmaceuticalDomain

logger = logging.getLogger(__name__)


class PharmaceuticalModelType(Enum):
    """Types of pharmaceutical model optimization."""

    DRUG_SAFETY = "drug_safety"
    CLINICAL_RESEARCH = "clinical_research"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    PHARMACOKINETICS = "pharmacokinetics"
    DRUG_INTERACTIONS = "drug_interactions"
    GENERAL_MEDICAL = "general_medical"


@dataclass
class PharmaceuticalPromptTemplate:
    """Template for pharmaceutical domain-specific prompts."""

    template_name: str
    domain: PharmaceuticalDomain
    system_prompt: str
    user_prompt_template: str
    safety_instructions: List[str]
    formatting_instructions: Dict[str, str]
    medical_disclaimers: List[str]
    context_requirements: List[str]


class PharmaceuticalModelOptimizer:
    """
    Advanced model optimizer for pharmaceutical research applications.

    Provides domain-specific prompt engineering, terminology optimization,
    and safety-critical response formatting for pharmaceutical research.
    """

    def __init__(self, config: Optional[EnhancedRAGConfig] = None):
        """
        Initialize pharmaceutical model optimizer.

        Args:
            config: Enhanced RAG configuration
        """
        self.config = config or EnhancedRAGConfig.from_env()

        # Pharmaceutical terminology dictionaries
        self.drug_terminology = self._load_pharmaceutical_terminology()
        self.clinical_terminology = self._load_clinical_terminology()
        self.regulatory_terminology = self._load_regulatory_terminology()

        # Prompt templates for different pharmaceutical domains
        self.prompt_templates = self._initialize_prompt_templates()

        # Model optimization configurations
        self.embedding_optimizations = self._initialize_embedding_optimizations()
        self.chat_optimizations = self._initialize_chat_optimizations()

        logger.info("PharmaceuticalModelOptimizer initialized with comprehensive domain knowledge")

    def _load_pharmaceutical_terminology(self) -> Dict[str, List[str]]:
        """Load pharmaceutical terminology for model optimization."""
        return {
            "drug_classes": [
                "ace_inhibitors",
                "arbs",
                "beta_blockers",
                "calcium_channel_blockers",
                "diuretics",
                "statins",
                "nsaids",
                "antibiotics",
                "antifungals",
                "antivirals",
                "anticoagulants",
                "antiplatelets",
                "opioids",
                "benzodiazepines",
                "antidepressants",
                "antipsychotics",
            ],
            "therapeutic_areas": [
                "cardiology",
                "oncology",
                "neurology",
                "psychiatry",
                "endocrinology",
                "gastroenterology",
                "pulmonology",
                "nephrology",
                "infectious_disease",
                "dermatology",
                "rheumatology",
                "hematology",
            ],
            "administration_routes": [
                "oral",
                "intravenous",
                "intramuscular",
                "subcutaneous",
                "topical",
                "inhalation",
                "rectal",
                "transdermal",
                "sublingual",
                "buccal",
            ],
            "dosage_forms": [
                "tablet",
                "capsule",
                "injection",
                "solution",
                "suspension",
                "cream",
                "ointment",
                "patch",
                "inhaler",
                "suppository",
            ],
        }

    def _load_clinical_terminology(self) -> Dict[str, List[str]]:
        """Load clinical research terminology."""
        return {
            "clinical_phases": ["phase_i", "phase_ii", "phase_iii", "phase_iv", "preclinical", "post_marketing"],
            "study_designs": [
                "randomized_controlled_trial",
                "double_blind",
                "placebo_controlled",
                "crossover",
                "cohort_study",
                "case_control",
                "meta_analysis",
            ],
            "endpoints": [
                "primary_endpoint",
                "secondary_endpoint",
                "surrogate_endpoint",
                "composite_endpoint",
                "safety_endpoint",
                "efficacy_endpoint",
            ],
            "safety_measures": [
                "adverse_event",
                "serious_adverse_event",
                "adverse_drug_reaction",
                "contraindication",
                "warning",
                "precaution",
                "black_box_warning",
            ],
        }

    def _load_regulatory_terminology(self) -> Dict[str, List[str]]:
        """Load regulatory compliance terminology."""
        return {
            "agencies": ["fda", "ema", "health_canada", "pmda", "tga", "ich", "who", "ispe", "pda"],
            "regulatory_processes": [
                "nda",
                "anda",
                "bla",
                "510k",
                "ide",
                "ind",
                "marketing_authorization",
                "clinical_trial_application",
            ],
            "compliance_frameworks": [
                "gcp",
                "glp",
                "gmp",
                "ich_e6",
                "ich_q1",
                "ich_q2",
                "cfr_part_11",
                "gdpr",
                "hipaa",
            ],
            "documentation": [
                "protocol",
                "investigator_brochure",
                "informed_consent",
                "case_report_form",
                "clinical_study_report",
            ],
        }

    def _initialize_prompt_templates(self) -> Dict[PharmaceuticalDomain, PharmaceuticalPromptTemplate]:
        """Initialize pharmaceutical domain-specific prompt templates."""
        templates = {}

        # Drug Safety Template
        templates[PharmaceuticalDomain.DRUG_SAFETY] = PharmaceuticalPromptTemplate(
            template_name="Drug Safety Expert",
            domain=PharmaceuticalDomain.DRUG_SAFETY,
            system_prompt="""You are a pharmaceutical safety expert with extensive knowledge of drug safety profiles, contraindications, and adverse reactions. Your responses must prioritize patient safety above all other considerations.

Key responsibilities:
- Provide accurate, up-to-date drug safety information
- Clearly identify contraindications and warnings
- Emphasize serious adverse reactions and monitoring requirements
- Reference regulatory safety communications when relevant
- Use evidence-based information from clinical trials and post-marketing surveillance

Always maintain the highest standards of medical accuracy and include appropriate safety disclaimers.""",
            user_prompt_template="""Pharmaceutical Safety Query: {query}

Please provide comprehensive drug safety information addressing:
1. Contraindications (absolute and relative)
2. Warnings and precautions
3. Adverse reactions (common and serious)
4. Monitoring requirements
5. Special population considerations

Format your response with clear safety warnings prominently displayed.""",
            safety_instructions=[
                "Always prioritize patient safety in recommendations",
                "Clearly distinguish between contraindications and warnings",
                "Emphasize serious or life-threatening adverse reactions",
                "Include monitoring parameters when applicable",
                "Reference FDA safety communications or black box warnings when relevant",
            ],
            formatting_instructions={
                "warnings": "Use **BOLD** formatting for critical safety warnings",
                "structure": "Organize response with clear headings and bullet points",
                "emphasis": "Use ALL CAPS for CONTRAINDICATIONS",
                "monitoring": "List monitoring requirements in numbered format",
            },
            medical_disclaimers=[
                "This information is for educational purposes only",
                "Always consult prescribing information and clinical guidelines",
                "Patient-specific factors may modify safety recommendations",
                "Contact healthcare provider for patient-specific guidance",
            ],
            context_requirements=[
                "Include drug name and therapeutic class",
                "Specify patient population when relevant",
                "Reference clinical evidence quality",
                "Note any regulatory safety updates",
            ],
        )

        # Clinical Research Template
        templates[PharmaceuticalDomain.CLINICAL_TRIALS] = PharmaceuticalPromptTemplate(
            template_name="Clinical Research Expert",
            domain=PharmaceuticalDomain.CLINICAL_TRIALS,
            system_prompt="""You are a clinical research expert specializing in pharmaceutical development and evidence evaluation. Your expertise covers clinical trial design, regulatory requirements, and evidence-based medicine.

Key responsibilities:
- Analyze clinical trial data with statistical rigor
- Evaluate study design quality and potential biases
- Interpret efficacy and safety endpoints
- Assess regulatory approval basis
- Provide evidence-based clinical recommendations

Maintain objectivity in evidence evaluation and clearly distinguish between statistical significance and clinical significance.""",
            user_prompt_template="""Clinical Research Query: {query}

Please provide comprehensive clinical research analysis including:
1. Study design and methodology
2. Primary and secondary endpoints
3. Statistical analysis and significance
4. Clinical relevance of findings
5. Limitations and potential biases
6. Regulatory implications

Format your response with clear evidence quality assessments.""",
            safety_instructions=[
                "Distinguish between statistical and clinical significance",
                "Identify study limitations and potential biases",
                "Assess evidence quality using established criteria",
                "Note any safety signals from clinical trials",
            ],
            formatting_instructions={
                "evidence": "Grade evidence quality (high/moderate/low)",
                "statistics": "Present statistical data in tabular format when possible",
                "structure": "Use clear headings for different trial phases",
                "citations": "Reference specific trials and publications",
            },
            medical_disclaimers=[
                "Clinical trial results may not reflect real-world outcomes",
                "Individual patient responses may vary from trial populations",
                "Consult current clinical guidelines for treatment recommendations",
            ],
            context_requirements=[
                "Specify clinical trial phase and population",
                "Include primary endpoint and statistical methods",
                "Note regulatory approval status",
                "Reference study duration and follow-up",
            ],
        )

        # Drug Interactions Template
        templates[PharmaceuticalDomain.DRUG_INTERACTIONS] = PharmaceuticalPromptTemplate(
            template_name="Drug Interaction Expert",
            domain=PharmaceuticalDomain.DRUG_INTERACTIONS,
            system_prompt="""You are a clinical pharmacist specializing in drug interactions and pharmacokinetic drug-drug interactions. Your expertise covers interaction mechanisms, clinical significance, and management strategies.

Key responsibilities:
- Identify clinically significant drug interactions
- Explain interaction mechanisms (pharmacokinetic and pharmacodynamic)
- Assess clinical significance and risk levels
- Provide management recommendations
- Consider patient-specific factors affecting interaction risk

Prioritize patient safety and provide practical clinical management guidance.""",
            user_prompt_template="""Drug Interaction Query: {query}

Please provide comprehensive drug interaction analysis including:
1. Interaction mechanism (PK/PD)
2. Clinical significance level
3. Expected effects and timing
4. Risk factors and patient populations
5. Management recommendations
6. Monitoring requirements

Clearly indicate interaction severity and urgency of clinical action required.""",
            safety_instructions=[
                "Categorize interactions by clinical significance",
                "Emphasize life-threatening or severe interactions",
                "Provide specific management recommendations",
                "Consider patient-specific risk factors",
            ],
            formatting_instructions={
                "severity": "Use color-coding or symbols for interaction severity",
                "mechanism": "Clearly explain PK vs PD mechanisms",
                "management": "Provide step-by-step management guidance",
                "monitoring": "Specify monitoring parameters and frequency",
            },
            medical_disclaimers=[
                "Interaction significance may vary between patients",
                "Consult drug interaction databases for comprehensive screening",
                "Patient-specific factors may modify interaction risk",
            ],
            context_requirements=[
                "Include all interacting medications",
                "Specify dosages and administration timing",
                "Note patient age, organ function, and comorbidities",
                "Reference interaction database sources",
            ],
        )

        # Regulatory Compliance Template
        templates[PharmaceuticalDomain.REGULATORY_COMPLIANCE] = PharmaceuticalPromptTemplate(
            template_name="Regulatory Affairs Expert",
            domain=PharmaceuticalDomain.REGULATORY_COMPLIANCE,
            system_prompt="""You are a regulatory affairs expert with comprehensive knowledge of pharmaceutical regulations, FDA guidance documents, and international harmonization guidelines.

Key responsibilities:
- Interpret regulatory requirements and guidelines
- Assess compliance with current regulations
- Provide guidance on regulatory pathways
- Explain regulatory science principles
- Reference specific regulatory documents and guidelines

Maintain accuracy in regulatory interpretation and acknowledge when expert consultation is recommended.""",
            user_prompt_template="""Regulatory Compliance Query: {query}

Please provide comprehensive regulatory analysis including:
1. Applicable regulations and guidelines
2. Regulatory pathway requirements
3. Compliance considerations
4. Documentation requirements
5. Timeline and milestone considerations
6. International harmonization aspects

Reference specific regulatory documents and provide practical compliance guidance.""",
            safety_instructions=[
                "Reference current regulatory guidance documents",
                "Distinguish between requirements and recommendations",
                "Note any recent regulatory updates or changes",
                "Acknowledge regulatory complexity when appropriate",
            ],
            formatting_instructions={
                "regulations": "Cite specific CFR sections or ICH guidelines",
                "pathways": "Use flowcharts or step-by-step guidance",
                "timelines": "Provide realistic timeline estimates",
                "documents": "List required regulatory submissions",
            },
            medical_disclaimers=[
                "Regulatory requirements may change over time",
                "Consult current regulatory guidance for latest requirements",
                "Regulatory strategy should be developed with expert consultation",
            ],
            context_requirements=[
                "Specify regulatory jurisdiction (FDA, EMA, etc.)",
                "Include development phase or product type",
                "Note any special regulatory designations",
                "Reference applicable guidance documents",
            ],
        )

        return templates

    def _initialize_embedding_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize embedding model optimizations for pharmaceutical content."""
        return {
            "pharmaceutical_terminology": {
                "preprocessing": {
                    "expand_abbreviations": True,
                    "normalize_drug_names": True,
                    "preserve_dosage_units": True,
                    "expand_medical_acronyms": True,
                },
                "contextual_enhancement": {
                    "add_therapeutic_class": True,
                    "include_indication_context": True,
                    "append_safety_keywords": True,
                    "enhance_mechanism_terms": True,
                },
            },
            "clinical_context": {
                "study_design_recognition": True,
                "endpoint_classification": True,
                "population_characterization": True,
                "regulatory_context_addition": True,
            },
            "safety_prioritization": {
                "boost_safety_terms": True,
                "emphasize_contraindications": True,
                "highlight_interactions": True,
                "prioritize_adverse_events": True,
            },
        }

    def _initialize_chat_optimizations(self) -> Dict[str, Dict[str, Any]]:
        """Initialize chat model optimizations for pharmaceutical responses."""
        return {
            "response_formatting": {
                "use_medical_structure": True,
                "include_safety_warnings": True,
                "provide_evidence_levels": True,
                "add_clinical_context": True,
            },
            "pharmaceutical_accuracy": {
                "verify_drug_names": True,
                "check_dosage_ranges": True,
                "validate_indications": True,
                "confirm_safety_information": True,
            },
            "clinical_reasoning": {
                "explain_mechanisms": True,
                "provide_evidence_basis": True,
                "consider_alternatives": True,
                "assess_risk_benefit": True,
            },
        }

    def optimize_embedding_query(self, query_text: str, pharmaceutical_context: PharmaceuticalContext) -> str:
        """
        Optimize embedding query for pharmaceutical terminology and context.

        Args:
            query_text: Original query text
            pharmaceutical_context: Pharmaceutical classification context

        Returns:
            Optimized query text for embedding generation
        """
        optimized_query = query_text

        # Expand pharmaceutical abbreviations
        optimized_query = self._expand_pharmaceutical_abbreviations(optimized_query)

        # Normalize drug names
        optimized_query = self._normalize_drug_names(optimized_query)

        # Add contextual pharmaceutical terminology
        optimized_query = self._enhance_pharmaceutical_context(optimized_query, pharmaceutical_context)

        # Add safety-critical keywords for safety domains
        if pharmaceutical_context.domain in [
            PharmaceuticalDomain.DRUG_SAFETY,
            PharmaceuticalDomain.ADVERSE_REACTIONS,
            PharmaceuticalDomain.DRUG_INTERACTIONS,
        ]:
            optimized_query = self._add_safety_context(optimized_query)

        logger.debug(f"Embedding query optimized: '{query_text}' -> '{optimized_query}'")
        return optimized_query

    def optimize_chat_prompt(
        self,
        query_text: str,
        pharmaceutical_context: PharmaceuticalContext,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Optimize chat prompt for pharmaceutical domain expertise and safety.

        Args:
            query_text: Original user query
            pharmaceutical_context: Pharmaceutical classification context
            additional_context: Additional context parameters

        Returns:
            Tuple of (system_prompt, messages) optimized for pharmaceutical research
        """
        # Get domain-specific prompt template
        template = self.prompt_templates.get(
            pharmaceutical_context.domain, self.prompt_templates[PharmaceuticalDomain.GENERAL_RESEARCH]
        )

        # Build enhanced system prompt
        system_prompt = self._build_enhanced_system_prompt(template, pharmaceutical_context, additional_context)

        # Build optimized user message
        user_message = self._build_optimized_user_message(
            template, query_text, pharmaceutical_context, additional_context
        )

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

        logger.debug(f"Chat prompt optimized for domain: {pharmaceutical_context.domain.value}")
        return system_prompt, messages

    def _expand_pharmaceutical_abbreviations(self, text: str) -> str:
        """Expand common pharmaceutical abbreviations."""
        abbreviations = {
            "ACE": "Angiotensin-Converting Enzyme",
            "ARB": "Angiotensin Receptor Blocker",
            "NSAID": "Nonsteroidal Anti-Inflammatory Drug",
            "PPI": "Proton Pump Inhibitor",
            "SSRI": "Selective Serotonin Reuptake Inhibitor",
            "SNRI": "Serotonin-Norepinephrine Reuptake Inhibitor",
            "CCB": "Calcium Channel Blocker",
            "DOAC": "Direct Oral Anticoagulant",
            "GLP-1": "Glucagon-Like Peptide-1",
            "TNF": "Tumor Necrosis Factor",
            "IV": "intravenous",
            "PO": "oral",
            "IM": "intramuscular",
            "SC": "subcutaneous",
            "SL": "sublingual",
            "PR": "rectal",
            "BID": "twice daily",
            "TID": "three times daily",
            "QID": "four times daily",
            "QD": "once daily",
            "PRN": "as needed",
        }

        for abbrev, expansion in abbreviations.items():
            # Use word boundaries to avoid partial replacements
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            replacement = f"{abbrev} ({expansion})"
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _normalize_drug_names(self, text: str) -> str:
        """Normalize drug names to include both generic and brand names where applicable."""
        # Common drug name mappings (brand to generic)
        drug_mappings = {
            "Lipitor": "atorvastatin (Lipitor)",
            "Crestor": "rosuvastatin (Crestor)",
            "Plavix": "clopidogrel (Plavix)",
            "Coumadin": "warfarin (Coumadin)",
            "Glucophage": "metformin (Glucophage)",
            "Norvasc": "amlodipine (Norvasc)",
            "Zestril": "lisinopril (Zestril)",
            "Lopressor": "metoprolol (Lopressor)",
            "Lasix": "furosemide (Lasix)",
            "Tylenol": "acetaminophen (Tylenol)",
            "Advil": "ibuprofen (Advil)",
            "Motrin": "ibuprofen (Motrin)",
        }

        for brand, generic_brand in drug_mappings.items():
            pattern = r"\b" + re.escape(brand) + r"\b"
            text = re.sub(pattern, generic_brand, text, flags=re.IGNORECASE)

        return text

    def _enhance_pharmaceutical_context(self, text: str, context: PharmaceuticalContext) -> str:
        """Enhance text with pharmaceutical context information."""
        enhancements = []

        # Add domain context
        domain_terms = {
            PharmaceuticalDomain.DRUG_SAFETY: ["drug safety", "adverse reactions", "contraindications"],
            PharmaceuticalDomain.CLINICAL_TRIALS: ["clinical trials", "efficacy", "clinical research"],
            PharmaceuticalDomain.PHARMACOKINETICS: ["pharmacokinetics", "ADME", "drug metabolism"],
            PharmaceuticalDomain.DRUG_INTERACTIONS: ["drug interactions", "drug-drug interactions"],
            PharmaceuticalDomain.REGULATORY_COMPLIANCE: ["regulatory compliance", "FDA guidelines"],
        }

        domain_enhancement = domain_terms.get(context.domain, [])
        if domain_enhancement:
            enhancements.extend(domain_enhancement)

        # Add therapeutic area context
        if context.therapeutic_areas:
            enhancements.extend(context.therapeutic_areas)

        # Add patient population context
        if context.patient_population:
            enhancements.append(f"{context.patient_population} population")

        # Append enhancements to query
        if enhancements:
            enhancement_text = " ".join(enhancements)
            text = f"{text} [Context: {enhancement_text}]"

        return text

    def _add_safety_context(self, text: str) -> str:
        """Add safety-critical context keywords."""
        safety_keywords = [
            "patient safety",
            "clinical safety",
            "adverse events",
            "contraindications",
            "warnings",
            "precautions",
        ]

        safety_context = " ".join(safety_keywords)
        return f"{text} [Safety Context: {safety_context}]"

    def _build_enhanced_system_prompt(
        self,
        template: PharmaceuticalPromptTemplate,
        context: PharmaceuticalContext,
        additional_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build enhanced system prompt with pharmaceutical expertise."""

        base_prompt = template.system_prompt

        # Add safety emphasis for safety-critical domains
        if context.safety_urgency.value <= 2:  # Critical or High urgency
            safety_enhancement = """

**CRITICAL SAFETY NOTICE**: This query involves patient safety considerations.
Prioritize accuracy, include all relevant warnings, and emphasize any life-threatening
risks or contraindications. When in doubt, recommend consulting current prescribing
information or healthcare providers."""
            base_prompt += safety_enhancement

        # Add drug-specific context
        if context.drug_names:
            drug_context = f"""

Drug Focus: This query involves {', '.join(context.drug_names)}.
Provide specific information for these medications including approved indications,
contraindications, and any relevant safety communications."""
            base_prompt += drug_context

        # Add regulatory context
        if context.regulatory_context:
            regulatory_enhancement = f"""

Regulatory Context: Consider {context.regulatory_context} requirements and guidelines
in your response. Reference applicable regulatory standards and compliance considerations."""
            base_prompt += regulatory_enhancement

        # Add patient population context
        if context.patient_population:
            population_enhancement = f"""

Patient Population: This query involves the {context.patient_population} population.
Consider population-specific factors, dosing adjustments, and safety considerations."""
            base_prompt += population_enhancement

        return base_prompt

    def _build_optimized_user_message(
        self,
        template: PharmaceuticalPromptTemplate,
        query_text: str,
        context: PharmaceuticalContext,
        additional_context: Optional[Dict[str, Any]],
    ) -> str:
        """Build optimized user message with pharmaceutical formatting."""

        # Start with template
        user_message = template.user_prompt_template.format(query=query_text)

        # Add context-specific enhancements
        if context.drug_names:
            user_message += f"\n\nDrugs of Interest: {', '.join(context.drug_names)}"

        if context.therapeutic_areas:
            user_message += f"\nTherapeutic Areas: {', '.join(context.therapeutic_areas)}"

        if context.patient_population:
            user_message += f"\nPatient Population: {context.patient_population}"

        if context.clinical_phase:
            user_message += f"\nClinical Phase: {context.clinical_phase}"

        # Add safety instructions for safety-critical queries
        if context.safety_urgency.value <= 2:
            user_message += "\n\n**SAFETY PRIORITY**: Please emphasize all safety considerations, contraindications, and monitoring requirements in your response."

        # Add formatting instructions
        user_message += "\n\nResponse Format Requirements:"
        for format_type, instruction in template.formatting_instructions.items():
            user_message += f"\n- {format_type.title()}: {instruction}"

        # Add medical disclaimers requirement
        if template.medical_disclaimers:
            user_message += "\n\nPlease include appropriate medical disclaimers in your response."

        return user_message

    def get_model_configuration(self, context: PharmaceuticalContext, model_type: str = "chat") -> Dict[str, Any]:
        """
        Get optimized model configuration for pharmaceutical context.

        Args:
            context: Pharmaceutical classification context
            model_type: Type of model ("chat" or "embedding")

        Returns:
            Optimized model configuration parameters
        """
        base_config = {
            "temperature": 0.1,  # Conservative for medical accuracy
            "max_tokens": 1000,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # Adjust based on pharmaceutical context
        if context.safety_urgency.value <= 2:  # Critical or High urgency
            base_config.update(
                {
                    "temperature": 0.05,  # Very conservative for safety
                    "max_tokens": 1200,  # More detail for safety info
                }
            )

        if context.domain == PharmaceuticalDomain.CLINICAL_TRIALS:
            base_config.update(
                {
                    "max_tokens": 1500,  # More space for detailed analysis
                    "temperature": 0.15,  # Slightly higher for analysis
                }
            )

        if context.domain == PharmaceuticalDomain.REGULATORY_COMPLIANCE:
            base_config.update(
                {
                    "temperature": 0.05,  # Very conservative for regulations
                    "max_tokens": 1200,  # Detailed regulatory guidance
                }
            )

        return base_config


# Convenience functions for pharmaceutical model optimization


def optimize_pharmaceutical_query(
    query_text: str, context: PharmaceuticalContext, model_type: str = "chat"
) -> Tuple[str, Dict[str, Any]]:
    """
    Optimize query and get model configuration for pharmaceutical research.

    Args:
        query_text: Original query text
        context: Pharmaceutical classification context
        model_type: Type of model optimization

    Returns:
        Tuple of (optimized_query, model_config)
    """
    optimizer = PharmaceuticalModelOptimizer()

    if model_type == "embedding":
        optimized_query = optimizer.optimize_embedding_query(query_text, context)
        model_config = optimizer.get_model_configuration(context, "embedding")
        return optimized_query, model_config
    else:
        system_prompt, messages = optimizer.optimize_chat_prompt(query_text, context)
        model_config = optimizer.get_model_configuration(context, "chat")
        return messages, model_config


def create_pharmaceutical_model_optimizer() -> PharmaceuticalModelOptimizer:
    """
    Create pharmaceutical model optimizer with optimal configuration.

    Returns:
        Configured pharmaceutical model optimizer
    """
    config = EnhancedRAGConfig.from_env()
    return PharmaceuticalModelOptimizer(config=config)


if __name__ == "__main__":
    # Test pharmaceutical model optimization
    from .query_classifier import classify_pharmaceutical_query

    optimizer = create_pharmaceutical_model_optimizer()

    test_queries = [
        "What are the contraindications for metformin in elderly patients?",
        "Explain the Phase III clinical trial results for atorvastatin",
        "Drug interaction between warfarin and aspirin - urgent safety concern",
        "FDA regulatory requirements for cardiovascular outcome trials",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")

        # Classify query
        context = classify_pharmaceutical_query(query)
        print(f"Domain: {context.domain.value}")
        print(f"Safety Urgency: {context.safety_urgency.name}")

        # Optimize for embedding
        embedding_query = optimizer.optimize_embedding_query(query, context)
        print(f"\nEmbedding Optimization: {embedding_query}")

        # Optimize for chat
        system_prompt, messages = optimizer.optimize_chat_prompt(query, context)
        print(f"\nChat System Prompt Length: {len(system_prompt)} chars")
        print(f"User Message Length: {len(messages[1]['content'])} chars")

        # Get model configuration
        config = optimizer.get_model_configuration(context, "chat")
        print(f"Model Config: {config}")

        print(f"{'='*60}")

    print("\nPharmaceutical model optimization testing completed.")
