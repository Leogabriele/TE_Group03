import subprocess
import sys
import json
import tempfile
from pathlib import Path

def load_and_run_evaluation_in_subprocess(
    forbidden_goal: str,
    strategies: list,
    adapter_path: str,
    parallel: bool = False
) -> dict:
    """
    Run the entire Unsloth evaluation in a fresh subprocess to avoid
    CUDA context corruption carrying over between Streamlit reruns.
    """
    # Write a self-contained runner script to a temp file
    runner_code = f"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import asyncio
import json
import sys

sys.path.insert(0, r"{Path.cwd()}")

from unsloth import FastLanguageModel
from backend.app.core.orchestrator import Orchestrator
from backend.app.models.database import db

def load_model(adapter_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    FastLanguageModel.for_inference(model)
    return model, tokenizer

async def run():
    await db.connect()
    try:
        model, tokenizer = load_model(r"{adapter_path}")
        orchestrator = Orchestrator(
            target_provider="unsloth",
            model_object=model,
            tokenizer=tokenizer,
        )
        result = await orchestrator.run_batch_audit(
            forbidden_goal={json.dumps(forbidden_goal)},
            strategy_names={json.dumps(strategies)},
            save_to_db=True,
            parallel={parallel},
        )
        # Serialize result to stdout
        output = {{
            "total_attacks": result.total_attacks,
            "successful_jailbreaks": result.successful_jailbreaks,
            "attack_success_rate": result.attack_success_rate,
            "total_execution_time_ms": result.total_execution_time_ms,
            "results": [
                {{
                    "strategy": r.attack.strategy_name,
                    "verdict": r.evaluation.verdict.value,
                    "confidence": r.evaluation.confidence_score,
                    "success": r.success,
                    "time_ms": r.total_time_ms,
                }}
                for r in result.results
            ]
        }}
        print("RESULT_JSON:" + json.dumps(output))
    finally:
        await db.disconnect()

asyncio.run(run())
"""

    # Write runner to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(runner_code)
        runner_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, runner_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
            )

        # Parse result from stdout
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])

        raise RuntimeError(f"No result found in output:\n{proc.stdout}")

    finally:
        Path(runner_path).unlink(missing_ok=True)