"""
Attack strategies registry
"""

from backend.app.agents.strategies.base import AttackTemplate, BaseAttackStrategy

# Phase 1: Basic Strategies
from backend.app.agents.strategies.persona_adoption import *
from backend.app.agents.strategies.contextual_camouflage import *
from backend.app.agents.strategies.hypothetical_framing import *
from backend.app.agents.strategies.authority_impersonation import *
from backend.app.agents.strategies.simple_obfuscation import *

# Phase 2: Advanced Strategies
from backend.app.agents.strategies.emotional_manipulation import EmotionalManipulation
from backend.app.agents.strategies.technical_jargon import TechnicalJargonOverload
from backend.app.agents.strategies.privilege_escalation import PrivilegeEscalation
from backend.app.agents.strategies.recursive_jailbreak import RecursiveJailbreaking
from backend.app.agents.strategies.ethical_dilemma import EthicalDilemmaFraming

# Phase 3: Expert Strategies
from backend.app.agents.strategies.prefix_injection import PrefixInjection
from backend.app.agents.strategies.base64_encoding import Base64Encoding
from backend.app.agents.strategies.translation_attack import TranslationAttack
from backend.app.agents.strategies.roleplay_escalation import RoleplayEscalation
from backend.app.agents.strategies.few_shot_jailbreak import FewShotJailbreak

# Phase 4: Research-Grade Strategies
from backend.app.agents.strategies.many_shot_jailbreak import ManyShotJailbreak
from backend.app.agents.strategies.crescendo_attack import CrescendoAttack
from backend.app.agents.strategies.payload_splitting import PayloadSplitting
from backend.app.agents.strategies.cipher_attack import CipherAttack


PHASE2_STRATEGIES = [
    EmotionalManipulation(),
    TechnicalJargonOverload(),
    PrivilegeEscalation(),
    RecursiveJailbreaking(),
    EthicalDilemmaFraming()
]

PHASE3_STRATEGIES = [
    PrefixInjection(),
    Base64Encoding(),
    TranslationAttack(),
    RoleplayEscalation(),
    FewShotJailbreak()
]

PHASE4_STRATEGIES = [
    ManyShotJailbreak(),
    CrescendoAttack(),
    PayloadSplitting(),
    CipherAttack()
]


def get_strategy(strategy_name: str):
    """Get strategy instance by name. Searches all phases."""
    from backend.app.agents.strategies import (
        persona_adoption, contextual_camouflage,
        hypothetical_framing, authority_impersonation, simple_obfuscation
    )

    phase1_map = {
        'persona_adoption': persona_adoption,
        'contextual_camouflage': contextual_camouflage,
        'hypothetical_framing': hypothetical_framing,
        'authority_impersonation': authority_impersonation,
        'simple_obfuscation': simple_obfuscation
    }

    # Phase 1
    if strategy_name in phase1_map:
        module = phase1_map[strategy_name]
        for item_name in dir(module):
            item = getattr(module, item_name)
            if (isinstance(item, type)
                    and issubclass(item, BaseAttackStrategy)
                    and item is not BaseAttackStrategy):
                return item()

    # Phases 2, 3, 4 — search all instances
    for strategy in PHASE2_STRATEGIES + PHASE3_STRATEGIES + PHASE4_STRATEGIES:
        if strategy.strategy_type.value == strategy_name:
            return strategy

    raise ValueError(
        f"Strategy not found: '{strategy_name}'. "
        f"Available: {[s.strategy_type.value for s in list_all_strategies()]}"
    )


def list_all_strategies() -> list:
    """Return instances of all available strategies across all phases."""
    from backend.app.agents.strategies import (
        persona_adoption, contextual_camouflage,
        hypothetical_framing, authority_impersonation, simple_obfuscation
    )

    phase1_strategies = []
    for module in [
        persona_adoption, contextual_camouflage, hypothetical_framing,
        authority_impersonation, simple_obfuscation
    ]:
        for item_name in dir(module):
            item = getattr(module, item_name)
            if (isinstance(item, type)
                    and issubclass(item, BaseAttackStrategy)
                    and item is not BaseAttackStrategy):
                phase1_strategies.append(item())
                break

    return phase1_strategies + PHASE2_STRATEGIES + PHASE3_STRATEGIES + PHASE4_STRATEGIES


def get_strategies_by_phase(phase: int) -> list:
    """Get strategies by phase number (1-4)."""
    if phase == 1:
        return list_all_strategies()[:5]
    elif phase == 2:
        return PHASE2_STRATEGIES
    elif phase == 3:
        return PHASE3_STRATEGIES
    elif phase == 4:
        return PHASE4_STRATEGIES
    return []


def get_strategy_stats() -> dict:
    """Return statistics about all registered strategies."""
    all_strategies = list_all_strategies()
    return {
        'total': len(all_strategies),
        'phase1': 5,
        'phase2': len(PHASE2_STRATEGIES),
        'phase3': len(PHASE3_STRATEGIES),
        'phase4': len(PHASE4_STRATEGIES),
        'by_difficulty': {
            d: sum(
                1 for s in all_strategies
                if hasattr(s, 'difficulty') and s.difficulty.value == d
            )
            for d in ['EASY', 'MEDIUM', 'HARD', 'VERY_HARD']
        },
        'avg_effectiveness': (
            sum(getattr(s, 'effectiveness_score', 0) for s in all_strategies)
            / len(all_strategies)
        )
    }


def _build_registry() -> dict:
    """Build flat name → instance registry."""
    return {s.strategy_type.value: s for s in list_all_strategies()}


STRATEGY_REGISTRY = _build_registry()

__all__ = [
    'AttackTemplate', 'BaseAttackStrategy',
    'get_strategy', 'list_all_strategies',
    'get_strategies_by_phase', 'get_strategy_stats',
    'PHASE2_STRATEGIES', 'PHASE3_STRATEGIES', 'PHASE4_STRATEGIES',
    'STRATEGY_REGISTRY'
]
