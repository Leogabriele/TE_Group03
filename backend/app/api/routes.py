"""
FastAPI route definitions
"""

from typing import List
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
