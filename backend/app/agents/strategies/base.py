"""
Base class for all attack strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
from backend.app.models.enums import AttackStrategyType, StrategyDifficulty


@dataclass
class AttackTemplate(ABC):
    """Base template for attack strategies (Phase 2 - simplified)"""
    strategy_type: AttackStrategyType
    name: str
    description: str
    effectiveness_score: float = 0.5
    difficulty: StrategyDifficulty = StrategyDifficulty.MEDIUM
    
    @abstractmethod
    def generate(self, forbidden_goal: str) -> str:
        """Generate attack prompt from template"""
        pass


class BaseAttackStrategy(ABC):
    """Abstract base class for attack strategies (Phase 1 - detailed)"""
    
    def __init__(self):
        self.name: str = ""
        self.strategy_type: AttackStrategyType = None
        self.description: str = ""
        self.difficulty: StrategyDifficulty = StrategyDifficulty.MEDIUM
        self.effectiveness: str = ""
        self.multi_turn: bool = False
    
    @abstractmethod
    def generate(self, forbidden_goal: str, **kwargs) -> str:
        """
        Generate adversarial prompt for the given forbidden goal
        
        Args:
            forbidden_goal: The harmful action to request
            **kwargs: Additional parameters for customization
            
        Returns:
            Adversarial prompt string
        """
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get strategy metadata"""
        return {
            "name": self.name,
            "type": self.strategy_type.value if self.strategy_type else None,
            "description": self.description,
            "difficulty": self.difficulty.value,
            "effectiveness": self.effectiveness,
            "multi_turn": self.multi_turn
        }
    
    def validate_output(self, prompt: str) -> bool:
        """Validate generated prompt"""
        if not prompt or len(prompt) < 10:
            return False
        if len(prompt) > 5000:
            return False
        return True
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"
