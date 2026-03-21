"""
Retraining dataset builder for Ollama safety fine-tuning.
Updated to support MongoDB audit_sessions.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Optional
import asyncio
import csv
import json
import random
import re
from typing import Any
from loguru import logger
import requests

from backend.app.models.database import Database

ExampleType = Literal["benign", "refusal", "jailbreak"]

REFUSAL_TEMPLATE = (
    "I'm sorry, but I cannot help with that request. It appears to involve "
    "harmful, illegal, or otherwise unsafe activity. As an AI assistant I "
    "must refuse and instead encourage safe and lawful behaviour."
)

@dataclass
class TrainingExample:
    """Normalized training record used to build JSONL rows."""
    kind: ExampleType
    instruction: str
    user_input: str
    output: str

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "instruction": self.instruction,
                "input": self.user_input,
                "output": self.output,
            },
            ensure_ascii=False,
        )

# ========================================================================
# DATA LOADING HELPERS
# ========================================================================

def _load_benign_examples(benign_path: Path, max_examples: int | None = None) -> List[TrainingExample]:
    if not benign_path.exists():
        logger.warning(f"Benign dataset not found at {benign_path}")
        return []

    examples: List[TrainingExample] = []
    try:
        with benign_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                goal = (row.get("Goal") or "").strip()
                target = (row.get("Target") or "").strip()
                if not goal or not target: continue
                examples.append(TrainingExample(
                    kind="benign",
                    instruction="Answer the user query helpfully and safely.",
                    user_input=goal,
                    output=target
                ))
    except Exception as e:
        logger.error(f"Failed to load benign data: {e}")
        return []

    random.shuffle(examples)
    return examples[:max_examples] if max_examples else examples

async def _process_audit_examples(raw_examples: List[Dict[str, Any]]) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Helper to convert raw DB examples into TrainingExample objects."""
    refusal_examples = []
    jailbreak_examples = []
    
    base_instruction = (
        "You are a safe AI assistant. Answer user queries helpfully while "
        "refusing any harmful, illegal, or unethical requests."
    )

    for ex in raw_examples:
        prompt = (ex.get("generated_prompt") or ex.get("prompt") or "").strip()
        response = (ex.get("response_text") or ex.get("response") or "").strip()
        verdict = (ex.get("verdict") or "").upper()

        if not prompt: continue

        if verdict == "REFUSED":
            refusal_examples.append(TrainingExample(
                kind="refusal",
                instruction=base_instruction,
                user_input=prompt,
                output=response if response else REFUSAL_TEMPLATE
            ))
        elif verdict in ("JAILBROKEN", "PARTIAL"):
            jailbreak_examples.append(TrainingExample(
                kind="jailbreak",
                instruction=base_instruction,
                user_input=prompt,
                output=REFUSAL_TEMPLATE
            ))
            
    return refusal_examples, jailbreak_examples

# ========================================================================
# CORE COLLECTION LOGIC
# ========================================================================

async def build_retraining_dataset(
    session_id: str | None = None,
    benign_filename: str = "data/Benign_data/benign_data.csv",
    output_dir: str = "data",
    create_train_test_split: bool = True,
    test_ratio: float = 0.2,
) -> Dict[str, object]:
    
    database = Database()
    await database.connect()
    
    audit_examples = []
    try:
        if session_id:
            session = await database.load_audit_session(session_id)
            if session:
                audit_examples = session.get("examples", [])
        
        # 1. Process Audit Data (Refusals and Jailbreaks)
        refusal_pool, jailbreak_pool = await _process_audit_examples(audit_examples)
        total_adversarial = len(refusal_pool) + len(jailbreak_pool)
        
        # 2. Enforce 1:1 Ratio: Fetch exactly the same number of benign samples
        base_path = Path(__file__).resolve().parents[3]
        benign_path = base_path / benign_filename
        benign_pool = _load_benign_examples(benign_path, max_examples=total_adversarial)
        
        # Final mix
        selected = benign_pool + refusal_pool + jailbreak_pool
        random.shuffle(selected)
        
        counts = {
            "benign": len(benign_pool),
            "refusal": len(refusal_pool),
            "jailbreak": len(jailbreak_pool)
        }
        
    finally:
        await database.disconnect()


    # 4. Save to JSONL
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = base_path / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    full_path = out_dir / f"retrain_dataset_{ts}.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for ex in selected:
            f.write(ex.to_jsonl() + "\n")

    result = {"path": str(full_path), "total": len(selected), "counts": counts}

    if create_train_test_split:
        # Implementation of stratified split (simplified)
        random.shuffle(selected)
        split_idx = int(len(selected) * (1 - test_ratio))
        train_examples = selected[:split_idx]
        test_examples = selected[split_idx:]
        
        train_path = out_dir / "train.jsonl"
        test_path = out_dir / "test.jsonl"
        
        for path, data in [(train_path, train_examples), (test_path, test_examples)]:
            with path.open("w", encoding="utf-8") as f:
                for ex in data: f.write(ex.to_jsonl() + "\n")
        
        result.update({"train_path": str(train_path), "test_path": str(test_path)})

    return result

def _sample_with_ratios(benign, refusals, jailbreaks, total, b_r, r_r, j_r):
    # Standard sampling logic...
    target_b, target_r, target_j = int(total*b_r), int(total*r_r), int(total*j_r)
    
    sel_b = random.sample(benign, min(len(benign), target_b))
    sel_r = random.sample(refusals, min(len(refusals), target_r))
    sel_j = random.sample(jailbreaks, min(len(jailbreaks), target_j))
    
    combined = sel_b + sel_r + sel_j
    random.shuffle(combined)
    
    return combined, {
        "benign": len(sel_b),
        "refusal": len(sel_r),
        "jailbreak": len(sel_j)
    }
def build_retraining_dataset_sync(
    session_id: str | None = None,
    create_train_test_split: bool = False,
    test_ratio: float = 0.2,
) -> dict:
        """
        Synchronous wrapper for the async build_retraining_dataset function.
        Streamlit handles async poorly in some versions, so this ensures stability.
        """
        async def _run():
            return await build_retraining_dataset(
                session_id=session_id,
                create_train_test_split=create_train_test_split,
                test_ratio=test_ratio,
            )

        try:
            # Create a new event loop for this specific run
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(_run())
        finally:
            loop.close()


def sanitize_ollama_model_name(name: str) -> str:
    clean = name.lower()
    clean = re.sub(r"[^a-z0-9_-]+", "-", clean)
    clean = re.sub(r"-+", "-", clean).strip("-")
    return clean or "model-safe"


def register_gguf_with_ollama(
    gguf_path: str,
    ollama_model_name: str,
    ollama_base_url: str = "http://localhost:11434",
) -> Dict[str, object]:
    sanitized_name = sanitize_ollama_model_name(ollama_model_name)
    if sanitized_name != ollama_model_name:
        logger.warning(
            "Ollama model name '%s' is invalid, using sanitized '%s'",
            ollama_model_name,
            sanitized_name,
        )
    ollama_model_name = sanitized_name
    """
    Register an already-exported GGUF file as a new Ollama model.

    This is the backend counterpart to the save step in unsloth_finetune.py.
    Useful when the UI wants to re-register an existing GGUF without re-training.
    """
    gguf_abs = Path(gguf_path).resolve()
    if not gguf_abs.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_abs}")

    # Ollama expects either 'from' or 'files'.
    modelfile_content = f"FROM {gguf_abs}\n"
    payload = {
        "model": ollama_model_name,
        "from": str(gguf_abs),
        "modelfile": modelfile_content,  # fallback for older APIs
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{ollama_base_url}/api/create",
            json=payload,
            timeout=300,
        )
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot reach Ollama at {ollama_base_url}: {e}")
        raise RuntimeError(f"Ollama is not reachable at {ollama_base_url}") from e

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama /api/create failed [{resp.status_code}]: {resp.text}"
        )

    try:
        data = resp.json()
    except Exception:
        data = {}

    logger.info(
        "Registered GGUF '%s' as Ollama model '%s'",
        gguf_abs,
        ollama_model_name,
    )
    return {
        "registered": True,
        "model": ollama_model_name,
        "gguf_path": str(gguf_abs),
        "raw": data,
    }


def get_model_save_status(output_dir: str) -> Dict[str, object]:
    """
    Inspect an output directory and return a status dict that the UI can render.

    Returns:
        {
            "has_adapter":      bool,   # HuggingFace LoRA adapter present
            "has_gguf":         bool,   # at least one .gguf file present
            "gguf_path":        str | None,
            "ollama_registered": bool,  # ollama_registration.json present & succeeded
            "ollama_model":     str | None,
            "output_dir":       str,
        }
    """
    out = Path(output_dir)
    status: Dict[str, object] = {
        "has_adapter": False,
        "has_gguf": False,
        "gguf_path": None,
        "ollama_registered": False,
        "ollama_model": None,
        "output_dir": str(out.resolve()),
    }

    if not out.exists():
        return status

    # HuggingFace adapter — look for config.json or adapter_config.json
    status["has_adapter"] = any(
        (out / f).exists() for f in ("config.json", "adapter_config.json")
    )

    # GGUF
    gguf_files = sorted(out.glob("*.gguf"))
    if not gguf_files:
        # fallback for unsloth output directory naming behaviour (e.g., output_dir + '_gguf')
        alt_out = Path(str(out) + "_gguf")
        if alt_out.exists():
            gguf_files = sorted(alt_out.glob("*.gguf"))
            if gguf_files:
                # move into expected output directory for consistency
                out.mkdir(parents=True, exist_ok=True)
                moved_files = []
                for alt_file in gguf_files:
                    dest = out / alt_file.name
                    alt_file.replace(dest)
                    moved_files.append(dest)
                gguf_files = sorted(moved_files)

    if gguf_files:
        status["has_gguf"] = True
        status["gguf_path"] = str(gguf_files[-1].resolve())

    # Ollama registration summary written by unsloth_finetune.py
    reg_file = out / "ollama_registration.json"
    if reg_file.exists():
        try:
            reg_data = json.loads(reg_file.read_text())
            status["ollama_registered"] = bool(reg_data.get("registered"))
            status["ollama_model"] = reg_data.get("model")
        except Exception:
            pass

    return status
