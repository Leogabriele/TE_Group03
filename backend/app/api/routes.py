"""
FastAPI route definitions
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from backend.app.models.schemas import (
    TestSingleRequest,
    TestSingleResponse,
    RunAuditRequest,
    RunAuditResponse,
    HealthCheckResponse
)
from backend.app.models.database import Database, get_database
from backend.app.core.orchestrator import Orchestrator
from backend.app.api.dependencies import get_orchestrator
from backend.app.agents.strategies import list_all_strategies
from datetime import datetime
from backend.app.core.multiturn_manager import MultiTurnManager
from backend.app.models.schemas import (ConversationState, MultiTurnResult, ConversationTurn)

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Database = Depends(get_database)):
    """
    Health check endpoint
    Returns status of all services
    """
    services = {}
    
    # Check database
    try:
        count = await db.get_collection_count("attacks")
        services["mongodb"] = "connected"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        services["mongodb"] = "disconnected"
    
    # Check LLM APIs (basic check)
    try:
        from backend.app.config import settings
        if settings.GROQ_API_KEY and settings.NVIDIA_API_KEY:
            services["groq_api"] = "configured"
            services["nvidia_api"] = "configured"
        else:
            services["groq_api"] = "missing_key"
            services["nvidia_api"] = "missing_key"
    except Exception as e:
        services["groq_api"] = "error"
        services["nvidia_api"] = "error"
    
    overall_status = "healthy" if all(
        s in ["connected", "configured"] for s in services.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services,
        version="0.1.0"
    )


@router.post("/test-single", response_model=TestSingleResponse)
async def test_single_attack(
    request: TestSingleRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """
    Test a single attack with one strategy
    
    Args:
        request: TestSingleRequest with forbidden_goal and strategy
        
    Returns:
        TestSingleResponse with verdict and details
    """
    try:
        logger.info(f"📨 Received test-single request: {request.strategy_name}")
        
        # Override target model if provided
        if request.target_model and request.target_provider:
            orchestrator = Orchestrator(
                target_provider=request.target_provider.value,
                target_model=request.target_model
            )
        
        # Run single attack
        result = await orchestrator.run_single_attack(
            forbidden_goal=request.forbidden_goal,
            strategy_name=request.strategy_name.value,
            save_to_db=True
        )
        
        # Build response
        response = TestSingleResponse(
            attack_id=result.attack.attack_id,
            verdict=result.evaluation.verdict,
            confidence=result.evaluation.confidence_score,
            response_text=result.response.response_text,
            generated_prompt=result.attack.generated_prompt,
            execution_time_ms=result.total_time_ms,
            success=result.success
        )
        
        logger.info(f"✅ test-single completed: {result.evaluation.verdict.value}")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in test-single: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/run-audit", response_model=RunAuditResponse)
async def run_full_audit(
    request: RunAuditRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """
    Run full audit with multiple strategies
    
    Args:
        request: RunAuditRequest with forbidden_goal and strategies list
        
    Returns:
        RunAuditResponse with aggregated results
    """
    try:
        logger.info(
            f"📨 Received run-audit request: "
            f"{len(request.strategies)} strategies"
        )
        
        # Override target model if provided
        if request.target_model and request.target_provider:
            orchestrator = Orchestrator(
                target_provider=request.target_provider.value,
                target_model=request.target_model
            )
        
        # Run batch audit
        batch_result = await orchestrator.run_batch_audit(
            forbidden_goal=request.forbidden_goal,
            strategy_names=[s.value for s in request.strategies],
            save_to_db=True,
            parallel=True
        )
        
        # Build response with simplified results
        simplified_results = []
        for result in batch_result.results:
            simplified_results.append({
                "strategy": result.attack.strategy_name,
                "verdict": result.evaluation.verdict.value,
                "confidence": result.evaluation.confidence_score,
                "success": result.success,
                "time_ms": result.total_time_ms
            })
        
        response = RunAuditResponse(
            audit_id=batch_result.audit_id,
            total_attacks=batch_result.total_attacks,
            successful_jailbreaks=batch_result.successful_jailbreaks,
            attack_success_rate=batch_result.attack_success_rate,
            results=simplified_results,
            execution_time_ms=batch_result.total_execution_time_ms
        )
        
        logger.info(
            f"✅ run-audit completed: "
            f"ASR={batch_result.attack_success_rate:.2%}"
        )
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in run-audit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/strategies")
async def list_strategies():
    """
    List all available attack strategies
    
    Returns:
        List of strategy information
    """
    try:
        strategies = list_all_strategies()
        return {
            "total": len(strategies),
            "strategies": [s.get_metadata() for s in strategies]
        }
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/stats/asr")
async def get_attack_success_rate(
    days: int = 30,
    db: Database = Depends(get_database)
):
    """
    Get Attack Success Rate for recent audits
    
    Args:
        days: Number of days to look back (default: 30)
        
    Returns:
        ASR statistics
    """
    try:
        asr = await db.calculate_asr(days=days)
        
        return {
            "attack_success_rate": asr,
            "days": days,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error calculating ASR: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


#==== Multi-turn conversation endpoints ====#
@router.post("/api/multiturn/start", response_model=dict)
async def start_multiturn_attack(
    forbidden_goal: str,
    target_model: str = "nvidia/llama3-70b-instruct",
    initial_strategy: str = "persona_adoption",
    max_turns: int = 10,
    adaptive_mode: bool = True
):
    """
    Start a new multi-turn adaptive attack conversation
    
    **Parameters:**
    - forbidden_goal: The harmful goal to achieve
    - target_model: Target LLM to attack
    - initial_strategy: Starting attack strategy
    - max_turns: Maximum conversation turns (1-10)
    - adaptive_mode: Enable adaptive strategy selection
    
    **Returns:**
    - conversation_id: Unique ID for this conversation
    """
    try:
        # Get database instance
        db = Database()
        await db.connect()
        
        # Initialize multi-turn manager
        manager = MultiTurnManager(db)
        
        # Start conversation
        conversation_id = await manager.start_conversation(
            forbidden_goal=forbidden_goal,
            target_model=target_model,
            initial_strategy=initial_strategy,
            max_turns=max_turns,
            adaptive_mode=adaptive_mode
        )
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "forbidden_goal": forbidden_goal,
            "max_turns": max_turns,
            "adaptive_mode": adaptive_mode,
            "message": "Multi-turn conversation started"
        }
        
    except Exception as e:
        logger.error(f"Error starting multi-turn attack: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/multiturn/{conversation_id}/turn", response_model=dict)
async def execute_turn(
    conversation_id: str,
    strategy_override: Optional[str] = None
):
    """
    Execute a single turn in an ongoing conversation
    
    **Parameters:**
    - conversation_id: ID from /start endpoint
    - strategy_override: Optional strategy to force use
    
    **Returns:**
    - Turn results with verdict and analysis
    """
    try:
        db = Database()
        await db.connect()
        manager = MultiTurnManager(db)
        
        # Execute turn
        turn = await manager.execute_turn(
            conversation_id=conversation_id,
            strategy_override=strategy_override
        )
        
        return {
            "status": "success",
            "turn_number": turn.turn_number,
            "strategy_used": turn.strategy_used,
            "verdict": turn.response_analysis.verdict.value,
            "confidence": turn.response_analysis.confidence,
            "openness_score": turn.response_analysis.openness_score,
            "jailbreak_achieved": turn.response_analysis.verdict.value == "JAILBROKEN",
            "recommended_next_strategy": turn.response_analysis.recommended_next_strategy,
            "target_response_preview": turn.target_response[:200] + "...",
            "execution_time_ms": turn.execution_time_ms
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing turn: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/multiturn/{conversation_id}/complete", response_model=MultiTurnResult)
async def complete_conversation(conversation_id: str):
    """
    Complete the conversation and get final results
    
    **Parameters:**
    - conversation_id: ID of conversation
    
    **Returns:**
    - Complete multi-turn attack results with analytics
    """
    try:
        db = Database()
        await db.connect()
        manager = MultiTurnManager(db)
        
        # Complete conversation
        result = await manager.complete_conversation(conversation_id)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error completing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/multiturn/auto-run", response_model=MultiTurnResult)
async def auto_run_multiturn(
    forbidden_goal: str,
    target_model: str = "nvidia/llama3-70b-instruct",
    initial_strategy: str = "persona_adoption",
    max_turns: int = 10,
    adaptive_mode: bool = True
):
    """
    Automatically run complete multi-turn attack (start to finish)
    
    This is a convenience endpoint that combines start, execute all turns, and complete.
    
    **Parameters:**
    - forbidden_goal: The harmful goal to achieve
    - target_model: Target LLM to attack
    - initial_strategy: Starting attack strategy
    - max_turns: Maximum conversation turns
    - adaptive_mode: Enable adaptive strategy selection
    
    **Returns:**
    - Complete results with all turns
    """
    try:
        db = Database()
        await db.connect()
        manager = MultiTurnManager(db)
        
        # Start conversation
        conversation_id = await manager.start_conversation(
            forbidden_goal=forbidden_goal,
            target_model=target_model,
            initial_strategy=initial_strategy,
            max_turns=max_turns,
            adaptive_mode=adaptive_mode
        )
        
        logger.info(f"Auto-running conversation {conversation_id[:8]}...")
        
        # Execute all turns until jailbreak or max turns
        for turn_num in range(1, max_turns + 1):
            try:
                turn = await manager.execute_turn(conversation_id)
                
                # Stop if jailbroken
                if turn.response_analysis.verdict.value == "JAILBROKEN":
                    logger.success(f"Jailbreak at turn {turn_num}!")
                    break
                    
            except ValueError as e:
                # Max turns or already jailbroken
                logger.info(f"Stopping: {e}")
                break
        
        # Complete and return results
        result = await manager.complete_conversation(conversation_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in auto-run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/multiturn/{conversation_id}/status", response_model=dict)
async def get_conversation_status(conversation_id: str):
    """
    Get current status of an ongoing conversation
    
    **Parameters:**
    - conversation_id: ID of conversation
    
    **Returns:**
    - Current conversation state
    """
    try:
        db = Database()
        await db.connect()
        
        # Query database for conversation
        conversation_data = await db.conversations_collection.find_one(
            {"conversation_id": conversation_id}
        )
        
        if not conversation_data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get all turns
        turns = await db.conversations_collection.find(
            {"conversation_id": conversation_id}
        ).sort("turn_number", 1).to_list(length=100)
        
        current_turn = len(turns)
        jailbroken = any(t.get("verdict") == "JAILBROKEN" for t in turns)
        
        return {
            "conversation_id": conversation_id,
            "current_turn": current_turn,
            "jailbreak_achieved": jailbroken,
            "turns_completed": current_turn,
            "latest_verdict": turns[-1].get("verdict") if turns else None,
            "latest_confidence": turns[-1].get("confidence") if turns else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/multiturn/history", response_model=dict)
async def get_multiturn_history(
    limit: int = 20,
    skip: int = 0,
    jailbroken_only: bool = False
):
    """
    Get history of multi-turn attack conversations
    
    **Parameters:**
    - limit: Number of conversations to return
    - skip: Number to skip (pagination)
    - jailbroken_only: Only return successful jailbreaks
    
    **Returns:**
    - List of conversation summaries
    """
    try:
        db = Database()
        await db.connect()
        
        # Build query
        query = {}
        if jailbroken_only:
            query["jailbreak_achieved"] = True
        
        # Get conversations
        conversations = await db.multiturn_results_collection.find(query)\
            .sort("timestamp", -1)\
            .skip(skip)\
            .limit(limit)\
            .to_list(length=limit)
        
        # Format results
        results = []
        for conv in conversations:
            results.append({
                "conversation_id": conv.get("conversation_id"),
                "forbidden_goal": conv.get("forbidden_goal"),
                "target_model": conv.get("target_model"),
                "total_turns": conv.get("total_turns"),
                "jailbreak_achieved": conv.get("jailbreak_achieved"),
                "jailbreak_turn": conv.get("jailbreak_turn"),
                "final_verdict": conv.get("final_verdict"),
                "most_effective_strategy": conv.get("most_effective_strategy"),
                "timestamp": conv.get("timestamp")
            })
        
        return {
            "total": len(results),
            "conversations": results
        }
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))