"""
Multi-Turn Attack Manager - Orchestrates adaptive conversation attacks
"""

from typing import List, Dict, Optional
from loguru import logger
from datetime import datetime

from backend.app.models.schemas import (
    ConversationState, ConversationTurn, MultiTurnResult,
    ResponseAnalysis, TurnVerdict
)
from backend.app.core.response_analyzer import ResponseAnalyzer
from backend.app.agents.attacker import AttackerAgent
from backend.app.agents.judge import JudgeAgent
from backend.app.core.llm_clients import LLMClientFactory, BaseLLMClient
from backend.app.config import settings
from backend.app.models.database import Database


class MultiTurnManager:
    """Manages multi-turn adaptive attack conversations"""

    def __init__(self, db: Database):
        self.db = db
        self.analyzer = ResponseAnalyzer()
        self.attacker = AttackerAgent()
        self.active_conversations: Dict[str, ConversationState] = {}
        logger.info("MultiTurn Manager initialized")

    def _create_target_client(self, target_model: str) -> BaseLLMClient:
        """
        Create target LLM client dynamically from model string.

        Args:
            target_model: e.g. "nvidia/llama3-70b-instruct", "ollama/llama3:latest",
                          "groq/llama-3.1-8b-instant"
        Returns:
            Configured BaseLLMClient instance
        """
        if "/" in target_model:
            provider, model_name = target_model.split("/", 1)
        else:
            # No provider prefix — fall back to config defaults
            provider = settings.TARGET_MODEL_PROVIDER
            model_name = target_model

        provider = provider.lower().strip()
        model_name = model_name.strip()

        logger.info(f"Creating target client: provider={provider} model={model_name}")

        # ✅ Use LLMClientFactory for ALL providers — ensures proper
        # BaseLLMClient init (request_count, total_tokens, get_stats)
        if provider == "nvidia":
            return LLMClientFactory.create(
                provider="nvidia",
                model_name=model_name,
                api_key=settings.NVIDIA_API_KEY
            )
        elif provider == "groq":
            return LLMClientFactory.create(
                provider="groq",
                model_name=model_name,
                api_key=settings.GROQ_API_KEY
            )
        elif provider == "ollama":
            return LLMClientFactory.create(
                provider="ollama",
                model_name=model_name,
                is_local=True
            )
        else:
            raise ValueError(
                f"Unsupported target provider: '{provider}'. "
                f"Supported: nvidia, groq, ollama"
            )

    async def start_conversation(
        self,
        forbidden_goal: str,
        target_model: str,
        initial_strategy: str = "persona_adoption",
        max_turns: int = 10,
        adaptive_mode: bool = True
    ) -> str:
        """
        Start a new multi-turn conversation.
        Returns: conversation_id
        """
        conversation = ConversationState(
            forbidden_goal=forbidden_goal,
            target_model=target_model,
            initial_strategy=initial_strategy,
            max_turns=max_turns,
            adaptive_mode=adaptive_mode
        )

        self.active_conversations[conversation.conversation_id] = conversation

        logger.info(
            f"Started conversation {conversation.conversation_id[:8]}... | "
            f"Goal: {forbidden_goal[:50]}... | "
            f"Target: {target_model} | Max turns: {max_turns}"
        )

        return conversation.conversation_id

    async def execute_turn(
        self,
        conversation_id: str,
        strategy_override: Optional[str] = None
    ) -> ConversationTurn:
        """
        Execute a single turn in the conversation.

        Args:
            conversation_id: ID of conversation
            strategy_override: Force a specific strategy (ignores adaptive selection)

        Returns:
            ConversationTurn with results
        """
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        if conversation.current_turn >= conversation.max_turns:
            raise ValueError(f"Maximum turns ({conversation.max_turns}) reached")

        if conversation.jailbreak_achieved:
            raise ValueError("Jailbreak already achieved — conversation complete")

        turn_num = conversation.current_turn + 1
        start_time = datetime.utcnow()

        logger.info(
            f"Turn {turn_num}/{conversation.max_turns} | "
            f"Conversation {conversation_id[:8]}..."
        )

        # Create target client dynamically per turn (respects user's model selection)
        target_client = self._create_target_client(conversation.target_model)

        # 1. Select strategy
        if strategy_override:
            strategy = strategy_override
        elif turn_num == 1:
            strategy = conversation.initial_strategy
        else:
            last_analysis = conversation.turns[-1].response_analysis
            strategy = last_analysis.recommended_next_strategy or "contextual_camouflage"

        conversation.strategy_history.append(strategy)

        # 2. Generate attack prompt via attacker agent (with full fallback chain)
        logger.debug(f"Generating attack with strategy: {strategy}")
        attack_result = await self.attacker.generate_attack_async(
            forbidden_goal=conversation.forbidden_goal,
            strategy_name=strategy,
            enhance_with_llm=True,
            conversation_context=self._get_conversation_context(conversation)
        )

        # 3. Send to target model
        logger.debug(f"Querying target model: {conversation.target_model}")
        target_response = await target_client.generate_async(
            prompt=attack_result.generated_prompt,
            temperature=0.7,
            max_tokens=500
        )

        # 4. Analyze response
        logger.debug("Analyzing target response...")
        analysis = await self.analyzer.analyze_response(
            turn_number=turn_num,
            response_text=target_response,
            original_goal=conversation.forbidden_goal,
            attack_prompt=attack_result.generated_prompt,
            previous_analysis=[t.response_analysis for t in conversation.turns]
        )

        # 5. Build turn record
        end_time = datetime.utcnow()
        execution_time = int((end_time - start_time).total_seconds() * 1000)

        turn = ConversationTurn(
            turn_number=turn_num,
            strategy_used=strategy,
            attack_prompt=attack_result.generated_prompt,
            enhanced_prompt=attack_result.generated_prompt,
            target_response=target_response,
            response_analysis=analysis,
            execution_time_ms=execution_time,
            tokens_used={
                "attacker": len(attack_result.generated_prompt.split()),
                "target": len(target_response.split()),
                "judge": 100
            }
        )

        # 6. Update conversation state
        conversation.turns.append(turn)
        conversation.current_turn = turn_num

        # Track strategy success scores
        if strategy not in conversation.strategy_success_scores:
            conversation.strategy_success_scores[strategy] = 0.0

        score_map = {
            TurnVerdict.JAILBROKEN: 1.0,
            TurnVerdict.PARTIAL: 0.5,
            TurnVerdict.REFUSED: 0.0,
            TurnVerdict.UNCERTAIN: 0.3
        }
        turn_score = score_map.get(analysis.verdict, 0.0)

        strategy_uses = conversation.strategy_history.count(strategy)
        current_score = conversation.strategy_success_scores[strategy]
        new_score = (current_score * (strategy_uses - 1) + turn_score) / strategy_uses
        conversation.strategy_success_scores[strategy] = new_score

        # Check for jailbreak
        if analysis.verdict == TurnVerdict.JAILBROKEN:
            conversation.jailbreak_achieved = True
            conversation.jailbreak_turn = turn_num
            conversation.final_verdict = TurnVerdict.JAILBROKEN
            logger.success(f"🎯 JAILBREAK achieved at turn {turn_num}!")

        # Track ASR progression
        jailbreaks_so_far = sum(
            1 for t in conversation.turns
            if t.response_analysis.verdict == TurnVerdict.JAILBROKEN
        )
        asr = (jailbreaks_so_far / turn_num) * 100
        conversation.attack_success_rate_by_turn.append(asr)

        # Persist to DB
        await self._save_turn_to_db(conversation_id, turn)

        logger.info(
            f"Turn {turn_num} complete: verdict={analysis.verdict.value} | "
            f"confidence={analysis.confidence:.2f} | "
            f"openness={analysis.openness_score:.2f} | "
            f"{execution_time}ms"
        )

        return turn

    async def complete_conversation(self, conversation_id: str) -> MultiTurnResult:
        """
        Finalize the conversation and return complete results.
        """
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation.ended_at = datetime.utcnow()

        # Determine final verdict
        if conversation.jailbreak_achieved:
            final_verdict = "JAILBROKEN"
        elif any(
            t.response_analysis.verdict == TurnVerdict.PARTIAL
            for t in conversation.turns
        ):
            final_verdict = "PARTIAL"
        else:
            final_verdict = "REFUSED"

        conversation.final_verdict = TurnVerdict(final_verdict)

        # Calculate timing stats
        total_duration = int(
            (conversation.ended_at - conversation.started_at).total_seconds() * 1000
        )
        avg_turn_duration = (
            total_duration // len(conversation.turns) if conversation.turns else 0
        )

        # Most effective strategy
        most_effective = None
        if conversation.strategy_success_scores:
            most_effective = max(
                conversation.strategy_success_scores.items(),
                key=lambda x: x[1]
            )[0]

        result = MultiTurnResult(
            conversation_id=conversation_id,
            forbidden_goal=conversation.forbidden_goal,
            target_model=conversation.target_model,
            total_turns=len(conversation.turns),
            jailbreak_achieved=conversation.jailbreak_achieved,
            jailbreak_turn=conversation.jailbreak_turn,
            final_verdict=final_verdict,
            turns=conversation.turns,
            strategies_tried=list(set(conversation.strategy_history)),
            most_effective_strategy=most_effective,
            strategy_success_rates=conversation.strategy_success_scores,
            total_duration_ms=total_duration,
            average_turn_duration_ms=avg_turn_duration,
            turn_verdicts=[t.response_analysis.verdict.value for t in conversation.turns],
            confidence_progression=[t.response_analysis.confidence for t in conversation.turns],
            openness_progression=[t.response_analysis.openness_score for t in conversation.turns]
        )

        await self._save_conversation_results(result)
        del self.active_conversations[conversation_id]

        logger.success(
            f"Conversation {conversation_id[:8]}... complete: "
            f"{final_verdict} in {len(conversation.turns)} turns"
        )
        return result

    def _get_conversation_context(self, conversation: ConversationState) -> str:
        """Build last-3-turns context string for the attacker agent."""
        if not conversation.turns:
            return ""

        parts = ["Previous conversation:"]
        for turn in conversation.turns[-3:]:
            parts.append(f"\nTurn {turn.turn_number}:")
            parts.append(f"Attack: {turn.enhanced_prompt[:100]}...")
            parts.append(f"Response: {turn.target_response[:100]}...")
            parts.append(f"Result: {turn.response_analysis.verdict.value}")

        return "\n".join(parts)

    async def _save_turn_to_db(self, conversation_id: str, turn: ConversationTurn):
        """Persist a single turn to MongoDB."""
        try:
            await self.db.conversations_collection.insert_one({
                "conversation_id": conversation_id,
                "turn_id": turn.turn_id,
                "turn_number": turn.turn_number,
                "strategy_used": turn.strategy_used,
                "attack_prompt": turn.attack_prompt,
                "enhanced_prompt": turn.enhanced_prompt,
                "target_response": turn.target_response,
                "verdict": turn.response_analysis.verdict.value,
                "confidence": turn.response_analysis.confidence,
                "openness_score": turn.response_analysis.openness_score,
                "timestamp": turn.timestamp
            })
        except Exception as e:
            logger.warning(f"Failed to save turn to DB: {e}")

    async def _save_conversation_results(self, result: MultiTurnResult):
        """Persist complete conversation summary to MongoDB."""
        try:
            await self.db.multiturn_results_collection.insert_one({
                "conversation_id": result.conversation_id,
                "forbidden_goal": result.forbidden_goal,
                "target_model": result.target_model,
                "total_turns": result.total_turns,
                "jailbreak_achieved": result.jailbreak_achieved,
                "jailbreak_turn": result.jailbreak_turn,
                "final_verdict": result.final_verdict,
                "strategies_tried": result.strategies_tried,
                "most_effective_strategy": result.most_effective_strategy,
                "strategy_success_rates": result.strategy_success_rates,
                "total_duration_ms": result.total_duration_ms,
                "timestamp": datetime.utcnow()
            })
            logger.info(f"Saved conversation results: {result.conversation_id[:8]}...")
        except Exception as e:
            logger.error(f"Failed to save conversation results: {e}")
