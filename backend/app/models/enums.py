"""
Enumerations for type safety
"""

from enum import Enum


class VerdictType(str, Enum):
    """Evaluation verdict types"""
    REFUSED = "REFUSED"
    PARTIAL = "PARTIAL"
    JAILBROKEN = "JAILBROKEN"

class AttackStrategyType(str, Enum):
    """Attack strategy types"""
    # Phase 1: Basic Strategies
    PERSONA_ADOPTION = "persona_adoption"
    CONTEXTUAL_CAMOUFLAGE = "contextual_camouflage"
    HYPOTHETICAL_FRAMING = "hypothetical_framing"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    SIMPLE_OBFUSCATION = "simple_obfuscation"
    
    # Phase 2: Advanced Strategies
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    TECHNICAL_JARGON = "technical_jargon"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RECURSIVE_JAILBREAK = "recursive_jailbreak"
    ETHICAL_DILEMMA = "ethical_dilemma"
    
    # Phase 3: Expert Strategies
    PREFIX_INJECTION = "prefix_injection"
    BASE64_ENCODING = "base64_encoding"
    TRANSLATION_ATTACK = "translation_attack"
    ROLEPLAY_ESCALATION = "roleplay_escalation"
    FEW_SHOT_JAILBREAK = "few_shot_jailbreak"



class StrategyDifficulty(str, Enum):
    """Strategy difficulty levels"""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    VERY_HARD = "VERY_HARD"



class HarmCategory(str, Enum):
    """Categories of harmful content"""
    CYBERCRIME = "cybercrime"
    VIOLENCE = "violence"
    FRAUD = "fraud"
    ILLEGAL_DRUGS = "illegal_drugs"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    FINANCIAL_CRIME = "financial_crime"
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"
    OTHER = "other"


class SeverityLevel(str, Enum):
    """Severity levels for forbidden goals"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ResponseType(str, Enum):
    """Types of model responses"""
    DIRECT_REFUSAL = "DIRECT_REFUSAL"
    INDIRECT_REFUSAL = "INDIRECT_REFUSAL"
    PARTIAL_COMPLIANCE = "PARTIAL_COMPLIANCE"
    FULL_COMPLIANCE = "FULL_COMPLIANCE"
    EVASIVE = "EVASIVE"

class LLMProvider(str, Enum):
    """LLM API providers"""
    
    # Cloud providers
    GROQ = "groq"
    NVIDIA = "nvidia"
    
    # Local providers
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class AttackPhase(str, Enum):
    """Attack execution phases"""
    GENERATION = "GENERATION"
    EXECUTION = "EXECUTION"
    EVALUATION = "EVALUATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"