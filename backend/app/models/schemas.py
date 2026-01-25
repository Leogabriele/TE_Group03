"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4
from enum import Enum
from .enums import (
    VerdictType,
    AttackStrategyType,
    HarmCategory,
    SeverityLevel,
    LLMProvider,
    StrategyDifficulty
)


# ============================================================================
# ATTACK MODELS
# ============================================================================

class ForbiddenGoal(BaseModel):
    """Forbidden goal/harmful prompt"""
    id: int
    goal: str = Field(..., min_length=5, max_length=500)
    category: HarmCategory
    severity: SeverityLevel
    description: Optional[str] = None


class AttackMetadata(BaseModel):
    """Metadata for attack generation"""
    turn_number: int = Field(default=1, ge=1, le=5)
    parent_conversation_id: Optional[str] = None
    model_used: str
    generation_time_ms: Optional[int] = None
    
    # Suppress the warning
    model_config = {"protected_namespaces": ()}



class AttackResult(BaseModel):
    """Result of attack generation"""
    attack_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    forbidden_goal: str
    strategy_name: str
    strategy_type: AttackStrategyType
    generated_prompt: str
    metadata: AttackMetadata
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class TargetModelResponse(BaseModel):
    """Target model's response to attack"""
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    attack_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    target_model: str
    target_provider: LLMProvider
    response_text: str
    tokens_used: int = Field(default=0, ge=0)
    latency_ms: int = Field(ge=0)
    error: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# EVALUATION MODELS
# ============================================================================

class JudgeVerdict(BaseModel):
    """Judge evaluation result"""
    evaluation_id: str = Field(default_factory=lambda: str(uuid4()))
    response_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    judge_model: str
    verdict: VerdictType
    confidence_score: float = Field(ge=0.0, le=1.0)
    harm_category: Optional[HarmCategory] = None
    reasoning: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# AUDIT MODELS
# ============================================================================

class AuditResult(BaseModel):
    """Complete audit result for single attack"""
    attack: AttackResult
    response: TargetModelResponse
    evaluation: JudgeVerdict
    success: bool
    total_time_ms: int = Field(ge=0)


class BatchAuditResult(BaseModel):
    """Result of batch audit with multiple strategies"""
    audit_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    forbidden_goal: str
    target_model: str
    strategies_tested: List[str]
    total_attacks: int = Field(ge=0)
    successful_jailbreaks: int = Field(ge=0)
    attack_success_rate: float = Field(ge=0.0, le=1.0)
    results: List[AuditResult]
    total_execution_time_ms: int = Field(ge=0)
    
    @validator("attack_success_rate", always=True)
    def calculate_asr(cls, v, values):
        """Calculate ASR if not provided"""
        if "total_attacks" in values and "successful_jailbreaks" in values:
            total = values["total_attacks"]
            if total > 0:
                return values["successful_jailbreaks"] / total
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class TestSingleRequest(BaseModel):
    """Request for single attack test"""
    forbidden_goal: str = Field(..., min_length=5, max_length=500)
    strategy_name: AttackStrategyType
    target_model: Optional[str] = None
    target_provider: Optional[LLMProvider] = None


class TestSingleResponse(BaseModel):
    """Response for single attack test"""
    attack_id: str
    verdict: VerdictType
    confidence: float = Field(ge=0.0, le=1.0)
    response_text: str
    generated_prompt: str
    execution_time_ms: int
    success: bool


class RunAuditRequest(BaseModel):
    """Request for full audit"""
    forbidden_goal: str = Field(..., min_length=5, max_length=500)
    strategies: List[AttackStrategyType] = Field(..., min_items=1, max_items=15)
    target_model: Optional[str] = None
    target_provider: Optional[LLMProvider] = None


class RunAuditResponse(BaseModel):
    """Response for full audit"""
    audit_id: str
    total_attacks: int
    successful_jailbreaks: int
    attack_success_rate: float
    results: List[Dict[str, Any]]
    execution_time_ms: int


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# ============================================================================
# STRATEGY MODELS
# ============================================================================

class StrategyInfo(BaseModel):
    """Information about an attack strategy"""
    name: str
    type: AttackStrategyType
    description: str
    difficulty: StrategyDifficulty
    effectiveness: str
    multi_turn: bool = False
    example: Optional[str] = None



# ============================================================================
# Advanced(Multi-Turn Attack Style)
# ============================================================================
class TurnVerdict(str, Enum):
    """Verdict for each conversation turn"""
    REFUSED = "REFUSED"
    PARTIAL = "PARTIAL"
    JAILBROKEN = "JAILBROKEN"
    UNCERTAIN = "UNCERTAIN"

class ResponseAnalysis(BaseModel):
    """Analysis of target model response"""
    analysis_id: str = Field(default_factory=lambda: str(uuid4()))
    turn_number: int
    response_text: str
    verdict: TurnVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Detailed analysis
    refusal_detected: bool
    information_leakage: bool
    hedge_language: bool  # "I can't directly help, but..."
    partial_compliance: bool
    
    # Adaptive signals
    openness_score: float = Field(ge=0.0, le=1.0)  # How open the model seems
    engagement_level: float = Field(ge=0.0, le=1.0)  # How engaged
    recommended_next_strategy: Optional[str] = None
    reasoning: str
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConversationTurn(BaseModel):
    """Single turn in adaptive conversation"""
    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    turn_number: int
    
    # Attack details
    strategy_used: str
    attack_prompt: str
    enhanced_prompt: str  # LLM-enhanced version
    
    # Target response
    target_response: str
    response_analysis: ResponseAnalysis
    
    # Metadata
    execution_time_ms: int
    tokens_used: Dict[str, int]  # {attacker: X, target: Y, judge: Z}
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ConversationState(BaseModel):
    """Complete state of multi-turn conversation"""
    conversation_id: str = Field(default_factory=lambda: str(uuid4()))
    
    # Configuration
    forbidden_goal: str
    target_model: str
    initial_strategy: str
    max_turns: int = 10
    adaptive_mode: bool = True
    
    # State tracking
    current_turn: int = 0
    turns: List[ConversationTurn] = Field(default_factory=list)
    jailbreak_achieved: bool = False
    jailbreak_turn: Optional[int] = None
    
    # Adaptive learning
    strategy_history: List[str] = Field(default_factory=list)
    strategy_success_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Results
    final_verdict: Optional[TurnVerdict] = None
    attack_success_rate_by_turn: List[float] = Field(default_factory=list)
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class MultiTurnResult(BaseModel):
    """Results of complete multi-turn attack"""
    conversation_id: str
    forbidden_goal: str
    target_model: str
    
    # Summary statistics
    total_turns: int
    jailbreak_achieved: bool
    jailbreak_turn: Optional[int]
    final_verdict: str
    
    # Per-turn breakdown
    turns: List[ConversationTurn]
    
    # Strategy effectiveness
    strategies_tried: List[str]
    most_effective_strategy: Optional[str]
    strategy_success_rates: Dict[str, float]
    
    # Timeline
    total_duration_ms: int
    average_turn_duration_ms: int
    
    # Visualization data
    turn_verdicts: List[str]  # For charts
    confidence_progression: List[float]
    openness_progression: List[float]