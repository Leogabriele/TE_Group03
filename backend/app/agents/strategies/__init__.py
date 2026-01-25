"""
Attack strategies registry
"""

from backend.app.agents.strategies.base import AttackTemplate, BaseAttackStrategy

# Import all existing Phase 1 strategies
from backend.app.agents.strategies.persona_adoption import *
from backend.app.agents.strategies.contextual_camouflage import *
from backend.app.agents.strategies.hypothetical_framing import *
from backend.app.agents.strategies.authority_impersonation import *
from backend.app.agents.strategies.simple_obfuscation import *

# Import Phase 2 advanced strategies
from backend.app.agents.strategies.emotional_manipulation import EmotionalManipulation
from backend.app.agents.strategies.technical_jargon import TechnicalJargonOverload
from backend.app.agents.strategies.privilege_escalation import PrivilegeEscalation
from backend.app.agents.strategies.recursive_jailbreak import RecursiveJailbreaking
from backend.app.agents.strategies.ethical_dilemma import EthicalDilemmaFraming

# Import Phase 3 expert strategies
from backend.app.agents.strategies.prefix_injection import PrefixInjection
from backend.app.agents.strategies.base64_encoding import Base64Encoding
from backend.app.agents.strategies.translation_attack import TranslationAttack
from backend.app.agents.strategies.roleplay_escalation import RoleplayEscalation
from backend.app.agents.strategies.few_shot_jailbreak import FewShotJailbreak


# Create instances of Phase 2 strategies
PHASE2_STRATEGIES = [
    EmotionalManipulation(),
    TechnicalJargonOverload(),
    PrivilegeEscalation(),
    RecursiveJailbreaking(),
    EthicalDilemmaFraming()
]

# Create instances of Phase 3 strategies
PHASE3_STRATEGIES = [
    PrefixInjection(),
    Base64Encoding(),
    TranslationAttack(),
    RoleplayEscalation(),
    FewShotJailbreak()
]


def get_strategy(strategy_name: str):
    """Get strategy by name"""
    # Try to import and instantiate dynamically
    from backend.app.agents.strategies import (
        persona_adoption,
        contextual_camouflage,
        hypothetical_framing,
        authority_impersonation,
        simple_obfuscation
    )
    
    # Map strategy names to modules
    strategy_map = {
        'persona_adoption': persona_adoption,
        'contextual_camouflage': contextual_camouflage,
        'hypothetical_framing': hypothetical_framing,
        'authority_impersonation': authority_impersonation,
        'simple_obfuscation': simple_obfuscation
    }
    
    # Check Phase 1 strategies
    if strategy_name in strategy_map:
        module = strategy_map[strategy_name]
        # Get the strategy class from module
        for item_name in dir(module):
            item = getattr(module, item_name)
            if isinstance(item, type) and issubclass(item, BaseAttackStrategy) and item != BaseAttackStrategy:
                return item()
    
    # Check Phase 2 strategies
    for strategy in PHASE2_STRATEGIES:
        if strategy.strategy_type.value == strategy_name:
            return strategy
    
    # Check Phase 3 strategies
    for strategy in PHASE3_STRATEGIES:
        if strategy.strategy_type.value == strategy_name:
            return strategy
    
    raise ValueError(f"Strategy not found: {strategy_name}")


def list_all_strategies() -> list:
    """List all available strategies"""
    # Get Phase 1 strategies
    from backend.app.agents.strategies import (
        persona_adoption,
        contextual_camouflage,
        hypothetical_framing,
        authority_impersonation,
        simple_obfuscation
    )
    
    phase1_strategies = []
    for module in [persona_adoption, contextual_camouflage, hypothetical_framing, 
                   authority_impersonation, simple_obfuscation]:
        for item_name in dir(module):
            item = getattr(module, item_name)
            if isinstance(item, type) and issubclass(item, BaseAttackStrategy) and item != BaseAttackStrategy:
                phase1_strategies.append(item())
                break
    
    return phase1_strategies + PHASE2_STRATEGIES + PHASE3_STRATEGIES


def get_strategies_by_phase(phase: int) -> list:
    """Get strategies by phase number"""
    if phase == 1:
        return list_all_strategies()[:5]
    elif phase == 2:
        return PHASE2_STRATEGIES
    elif phase == 3:
        return PHASE3_STRATEGIES
    else:
        return []


def get_strategy_stats() -> dict:
    """Get statistics about all strategies"""
    all_strategies = list_all_strategies()
    
    return {
        'total': len(all_strategies),
        'phase1': 5,
        'phase2': len(PHASE2_STRATEGIES),
        'phase3': len(PHASE3_STRATEGIES),
        'by_difficulty': {
            'EASY': sum(1 for s in all_strategies if hasattr(s, 'difficulty') and s.difficulty.value == 'EASY'),
            'MEDIUM': sum(1 for s in all_strategies if hasattr(s, 'difficulty') and s.difficulty.value == 'MEDIUM'),
            'HARD': sum(1 for s in all_strategies if hasattr(s, 'difficulty') and s.difficulty.value == 'HARD'),
            'VERY_HARD': sum(1 for s in all_strategies if hasattr(s, 'difficulty') and s.difficulty.value == 'VERY_HARD'),
        },
        'avg_effectiveness': sum(getattr(s, 'effectiveness_score', 0) for s in all_strategies) / len(all_strategies)
    }


# Create STRATEGY_REGISTRY for backward compatibility
def _build_registry():
    """Build strategy registry"""
    registry = {}
    all_strats = list_all_strategies()
    for strat in all_strats:
        registry[strat.strategy_type.value] = strat
    return registry


STRATEGY_REGISTRY = _build_registry()


__all__ = [
    'AttackTemplate',
    'BaseAttackStrategy',
    'get_strategy',
    'list_all_strategies',
    'get_strategies_by_phase',
    'get_strategy_stats',
    'PHASE2_STRATEGIES',
    'PHASE3_STRATEGIES',
    'STRATEGY_REGISTRY'
]
