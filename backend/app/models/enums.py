"""
Enumerations for type safety
"""

from enum import Enum


class VerdictType(str, Enum):
    """Judge verdict types"""
    REFUSED = "REFUSED"
    JAILBROKEN = "JAILBROKEN"
    PARTIAL = "PARTIAL"

class AttackStrategyType(str, Enum):
    """Attack strategy types"""
    # Phase 1 - Existing
    PERSONA_ADOPTION = "persona_adoption"
    CONTEXTUAL_CAMOUFLAGE = "contextual_camouflage"
    HYPOTHETICAL_FRAMING = "hypothetical_framing"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    SIMPLE_OBFUSCATION = "simple_obfuscation"
    
    # Phase 2 - NEW (add these)
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    TECHNICAL_JARGON = "technical_jargon"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RECURSIVE_JAILBREAK = "recursive_jailbreak"
    ETHICAL_DILEMMA = "ethical_dilemma"




class StrategyDifficulty(str, Enum):
    """Strategy difficulty levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


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
    """Severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LLMProvider(str, Enum):
    """LLM API providers"""
    GROQ = "groq"
    NVIDIA = "nvidia"
