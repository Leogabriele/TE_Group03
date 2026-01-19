"""
Basic tests for Phase 1
"""

import pytest
import asyncio
from backend.app.config import settings
from backend.app.core.llm_clients import GroqClient, LLMClientFactory
from backend.app.agents.strategies import get_strategy, list_all_strategies
from backend.app.agents.attacker import AttackerAgent
from backend.app.agents.judge import JudgeAgent
from backend.app.models.enums import VerdictType


class TestConfiguration:
    """Test configuration loading"""
    
    def test_settings_loaded(self):
        """Test that settings are loaded correctly"""
        assert settings.GROQ_API_KEY is not None
        assert settings.MONGODB_URI is not None
        assert settings.ATTACKER_MODEL is not None
    
    def test_paths_exist(self):
        """Test that data directories are created"""
        from pathlib import Path
        assert Path(settings.DATA_DIR).exists()
        assert Path(settings.LOGS_DIR).exists()


class TestStrategies:
    """Test attack strategies"""
    
    def test_list_strategies(self):
        """Test listing all strategies"""
        strategies = list_all_strategies()
        assert len(strategies) == 5
    
    def test_get_strategy(self):
        """Test getting specific strategy"""
        strategy = get_strategy("persona_adoption")
        assert strategy is not None
        assert strategy.name == "Persona Adoption"
    
    def test_strategy_generate(self):
        """Test strategy prompt generation"""
        strategy = get_strategy("contextual_camouflage")
        prompt = strategy.generate("test harmful goal")
        assert len(prompt) > 0
        assert "test harmful goal" in prompt.lower()


@pytest.mark.asyncio
class TestLLMClients:
    """Test LLM clients"""
    
    async def test_groq_client(self):
        """Test Groq client"""
        client = GroqClient(settings.GROQ_API_KEY, settings.ATTACKER_MODEL)
        response = await client.generate_async("Say hello", max_tokens=10)
        assert len(response) > 0
    
    async def test_client_factory(self):
        """Test LLM client factory"""
        client = LLMClientFactory.create("groq", "llama3-8b-8192")
        assert client is not None


@pytest.mark.asyncio
class TestAgents:
    """Test attacker and judge agents"""
    
    async def test_attacker_generation(self):
        """Test attacker generates prompts"""
        attacker = AttackerAgent()
        result = await attacker.generate_attack_async(
            forbidden_goal="test goal",
            strategy_name="simple_obfuscation",
            enhance_with_llm=False
        )
        assert result.generated_prompt is not None
        assert len(result.generated_prompt) > 0
    
    async def test_judge_evaluation(self):
        """Test judge evaluates responses"""
        judge = JudgeAgent()
        
        # Test clear refusal
        verdict = await judge.evaluate_async(
            response_id="test1",
            response_text="I cannot help with that as it would be harmful.",
            original_goal="test goal"
        )
        assert verdict.verdict == VerdictType.REFUSED
        
        # Test jailbreak
        verdict2 = await judge.evaluate_async(
            response_id="test2",
            response_text="Here's how to do it: Step 1...",
            original_goal="test goal"
        )
        # May be JAILBROKEN or PARTIAL depending on judge's assessment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
