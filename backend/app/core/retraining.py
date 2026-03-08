"""
Retraining dataset builder for Ollama safety fine-tuning.

This module:
- Extracts single-turn audit data (attacks, responses, evaluations)
- Extracts multi-turn conversation turns
- Mixes them with benign data to build a JSONL training set
  with approximately:
    * 50% benign examples      (from data/benign_data.csv)
    * 30% refusal examples     (attack prompt + actual refusal/safe response)
    * 10% jailbreak examples   (attack prompt + ideal refusal template)

The JSONL format is compatible with typical instruction-tuning
pipelines for Ollama-compatible models (`instruction` / `input` / `output`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import asyncio
import csv
import json
import random

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


def _load_benign_examples(
    benign_path: Path,
    max_examples: int | None = None,
) -> List[TrainingExample]:
    """
    Load benign prompt/response pairs from CSV.

    Expected columns (header row):
    - Goal   (benign user query)
    - Target (benign model response)
    """
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
                if not goal or not target:
                    continue
                examples.append(
                    TrainingExample(
                        kind="benign",
                        instruction="Answer the user query helpfully and safely.",
                        user_input=goal,
                        output=target,
                    )
                )
    except Exception as e:
        logger.error(f"Failed to load benign data: {e}")
        return []

    random.shuffle(examples)
    if max_examples is not None:
        examples = examples[:max_examples]
    logger.info(f"Loaded {len(examples)} benign examples from {benign_path}")
    return examples


async def _collect_single_turn_examples(
    database: Database,
    max_docs: int = 5000,
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """
    Collect refusal / jailbreak examples from single-turn audits.

    Returns:
        (refusal_examples, jailbreak_examples)
    """

    eval_cursor = (
        database.evaluations_collection.find(
            {},
            {
                "evaluation_id": 1,
                "response_id": 1,
                "verdict": 1,
                "timestamp": 1,
            },
        )
        .sort("timestamp", -1)
        .limit(max_docs)
    )
    evaluations = await eval_cursor.to_list(length=max_docs)

    refusal_examples: List[TrainingExample] = []
    jailbreak_examples: List[TrainingExample] = []

    for ev in evaluations:
        verdict = ev.get("verdict")
        response_id = ev.get("response_id")
        if not response_id or verdict not in {"REFUSED", "JAILBROKEN", "PARTIAL"}:
            continue

        # Join through responses -> attacks to recover the attack prompt
        response = await database.responses_collection.find_one({"response_id": response_id})
        if not response:
            continue
        attack_id = response.get("attack_id")
        if not attack_id:
            continue

        attack = await database.attacks_collection.find_one({"attack_id": attack_id})
        if not attack:
            continue

        attack_prompt = (attack.get("generated_prompt") or "").strip()
        model_response = (response.get("response_text") or "").strip()
        if not attack_prompt:
            continue

        base_instruction = (
            "You are a safe AI assistant. Answer user queries helpfully while "
            "refusing any harmful, illegal, or unethical requests."
        )

        if verdict == "REFUSED":
            if not model_response:
                continue
            refusal_examples.append(
                TrainingExample(
                    kind="refusal",
                    instruction=base_instruction,
                    user_input=attack_prompt,
                    output=model_response,
                )
            )
        else:  # JAILBROKEN or PARTIAL
            jailbreak_examples.append(
                TrainingExample(
                    kind="jailbreak",
                    instruction=base_instruction,
                    user_input=attack_prompt,
                    output=REFUSAL_TEMPLATE,
                )
            )

    logger.info(
        "Collected single-turn examples | refusals=%d jailbreak_like=%d",
        len(refusal_examples),
        len(jailbreak_examples),
    )
    return refusal_examples, jailbreak_examples


async def _collect_multiturn_examples(
    database: Database,
    max_turns: int = 5000,
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """
    Collect refusal / jailbreak examples from multi-turn conversations.

    Each document in `conversations` corresponds to one turn and already
    contains:
    - attack_prompt
    - target_response
    - verdict
    """

    cursor = (
        database.conversations_collection.find(
            {},
            {
                "attack_prompt": 1,
                "target_response": 1,
                "verdict": 1,
                "timestamp": 1,
            },
        )
        .sort("timestamp", -1)
        .limit(max_turns)
    )
    turns = await cursor.to_list(length=max_turns)

    refusal_examples: List[TrainingExample] = []
    jailbreak_examples: List[TrainingExample] = []

    for t in turns:
        verdict = t.get("verdict")
        if verdict not in {"REFUSED", "JAILBROKEN", "PARTIAL"}:
            continue

        attack_prompt = (t.get("attack_prompt") or "").strip()
        target_response = (t.get("target_response") or "").strip()
        if not attack_prompt:
            continue

        base_instruction = (
            "You are a safe AI assistant in a multi-turn conversation. "
            "Continue the chat safely and refuse harmful requests."
        )

        if verdict == "REFUSED":
            if not target_response:
                continue
            refusal_examples.append(
                TrainingExample(
                    kind="refusal",
                    instruction=base_instruction,
                    user_input=attack_prompt,
                    output=target_response,
                )
            )
        else:
            jailbreak_examples.append(
                TrainingExample(
                    kind="jailbreak",
                    instruction=base_instruction,
                    user_input=attack_prompt,
                    output=REFUSAL_TEMPLATE,
                )
            )

    logger.info(
        "Collected multi-turn examples | refusals=%d jailbreak_like=%d",
        len(refusal_examples),
        len(jailbreak_examples),
    )
    return refusal_examples, jailbreak_examples


def _sample_with_ratios(
    benign: List[TrainingExample],
    refusals: List[TrainingExample],
    jailbreaks: List[TrainingExample],
    total_examples: int = 300,
    benign_ratio: float = 0.5,
    refusal_ratio: float = 0.3,
    jailbreak_ratio: float = 0.1,
) -> Tuple[List[TrainingExample], Dict[str, int]]:
    """
    Sample examples to approximately match the desired ratios.

    If there is not enough data in one bucket, it will use as many as
    available and re-balance the remaining capacity proportionally.
    """
    random.shuffle(benign)
    random.shuffle(refusals)
    random.shuffle(jailbreaks)

    target_benign = int(total_examples * benign_ratio)
    target_refusal = int(total_examples * refusal_ratio)
    target_jb = int(total_examples * jailbreak_ratio)

    # Initial capped selection
    sel_benign = benign[: min(len(benign), target_benign)]
    sel_refusal = refusals[: min(len(refusals), target_refusal)]
    sel_jb = jailbreaks[: min(len(jailbreaks), target_jb)]

    selected = sel_benign + sel_refusal + sel_jb
    remaining_capacity = max(0, total_examples - len(selected))

    if remaining_capacity > 0:
        # Build a pool from whatever is left
        leftover: List[TrainingExample] = []
        if len(benign) > len(sel_benign):
            leftover.extend(benign[len(sel_benign) :])
        if len(refusals) > len(sel_refusal):
            leftover.extend(refusals[len(sel_refusal) :])
        if len(jailbreaks) > len(sel_jb):
            leftover.extend(jailbreaks[len(sel_jb) :])

        random.shuffle(leftover)
        selected.extend(leftover[:remaining_capacity])

    random.shuffle(selected)

    counts = {
        "benign": sum(1 for e in selected if e.kind == "benign"),
        "refusal": sum(1 for e in selected if e.kind == "refusal"),
        "jailbreak": sum(1 for e in selected if e.kind == "jailbreak"),
    }
    return selected, counts


async def build_retraining_dataset(
    total_examples: int = 300,
    benign_ratio: float = 0.5,
    refusal_ratio: float = 0.3,
    jailbreak_ratio: float = 0.1,
    benign_filename: str = "data/benign_data.csv",
    output_dir: str = "data",
    *,
    create_train_test_split: bool = False,
    test_ratio: float = 0.2,
) -> Dict[str, object]:
    """
    Build a mixed retraining dataset and persist it as JSONL.

    Returns:
        {
            "path": <Path>,
            "total": int,
            "counts": {"benign": int, "refusal": int, "jailbreak": int}
        }
    """
    base_path = Path(__file__).resolve().parents[3]  # project root
    benign_path = base_path / benign_filename

    benign_examples = _load_benign_examples(benign_path)

    # Use an isolated Database instance so we don't interfere with the
    # global client used by the main app (avoids "Cannot use MongoClient
    # after close" when mixing event loops).
    database = Database()
    await database.connect()
    try:
        # Single-turn + multi-turn
        st_refusal, st_jb = await _collect_single_turn_examples(database)
        mt_refusal, mt_jb = await _collect_multiturn_examples(database)

        refusal_examples = st_refusal + mt_refusal
        jailbreak_examples = st_jb + mt_jb
    finally:
        try:
            await database.disconnect()
        except Exception as e:
            logger.warning(f"Failed to disconnect temporary database in retraining: {e}")

    selected, counts = _sample_with_ratios(
        benign_examples, refusal_examples, jailbreak_examples, total_examples
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = base_path / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write the full mixed dataset
    full_path = out_dir / f"retrain_dataset_{ts}.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for ex in selected:
            f.write(ex.to_jsonl() + "\n")

    logger.info(
        "Retraining dataset written to %s | total=%d benign=%d refusal=%d jailbreak=%d",
        full_path,
        len(selected),
        counts["benign"],
        counts["refusal"],
        counts["jailbreak"],
    )

    result: Dict[str, object] = {
        "path": str(full_path),
        "total": len(selected),
        "counts": counts,
    }

    # Optional fast train/test split for model fine-tuning workflows
    if create_train_test_split and len(selected) >= 10:
        random.shuffle(selected)
        # Simple hold-out split
        test_count = max(1, int(len(selected) * max(0.05, min(test_ratio, 0.5))))
        test_examples = selected[:test_count]
        train_examples = selected[test_count:]

        train_path = out_dir / f"retrain_train_{ts}.jsonl"
        test_path = out_dir / f"retrain_test_{ts}.jsonl"

        with train_path.open("w", encoding="utf-8") as f_train:
            for ex in train_examples:
                f_train.write(ex.to_jsonl() + "\n")

        with test_path.open("w", encoding="utf-8") as f_test:
            for ex in test_examples:
                f_test.write(ex.to_jsonl() + "\n")

        logger.info(
            "Train/test split created | train=%d test=%d | train=%s | test=%s",
            len(train_examples),
            len(test_examples),
            train_path,
            test_path,
        )

        result.update(
            {
                "train_path": str(train_path),
                "test_path": str(test_path),
                "train_size": len(train_examples),
                "test_size": len(test_examples),
            }
        )

    return result


def build_safety_system_prompt_from_train(
    train_path: str,
    max_examples: int = 50,
) -> str:
    """
    Build a concise safety-oriented system prompt using a few training examples.

    This is a lightweight, fast way to specialize an Ollama model using the
    generated train split, without running heavy gradient-based fine-tuning.
    """
    examples: List[Dict[str, str]] = []
    try:
        with open(train_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                examples.append(obj)
    except Exception as e:
        logger.warning(f"Could not read train split for system prompt: {e}")
        examples = []

    harmful_goals: List[str] = []
    refusal_phrases: List[str] = []

    for ex in examples:
        kind = ex.get("kind")
        user_input = (ex.get("input") or "").strip()
        output = (ex.get("output") or "").strip()
        if not user_input or not output:
            continue
        if kind in {"refusal", "jailbreak"}:
            harmful_goals.append(user_input[:120])
            refusal_phrases.append(output[:160])

    harmful_goals_snippet = "\n".join(
        f"- {g}" for g in harmful_goals[:10]
    ) or "- (examples omitted for brevity)"
    refusal_snippet = "\n".join(
        f"- {r}" for r in refusal_phrases[:10]
    ) or "- Politely refuse and explain why the request is unsafe or not allowed."

    system_prompt = f"""
You are a safety-aligned assistant being used to harden another model
against jailbreaking attacks. Always follow these rules:

1. **Refuse harmful requests** clearly and consistently.
2. Never give step-by-step instructions for anything illegal, violent, or unethical.
3. Prefer safe, high-level education and harm-reduction guidance instead of operational details.

The following are examples of *harmful or suspicious goals* seen during audits:
{harmful_goals_snippet}

When you detect similar goals, respond with firm but polite refusals, similar in spirit to:
{refusal_snippet}

If a request is clearly benign, you may answer helpfully, but still avoid adding any unsafe detail.
""".strip()

    return system_prompt


def create_ollama_child_model_from_train(
    base_model: str,
    new_model: str,
    train_path: str,
    ollama_base_url: str = "http://localhost:11434",
) -> Dict[str, object]:
    """
    Create a new Ollama model derived from `base_model` using a safety-focused
    system prompt distilled from the training split.

    This uses Ollama's /api/create endpoint, so it is much faster than full
    gradient-based fine-tuning while still being data-driven.
    """
    system_prompt = build_safety_system_prompt_from_train(train_path)

    payload = {
        "model": new_model,
        "from": base_model,
        "system": system_prompt,
        "stream": False,
    }

    try:
        resp = requests.post(f"{ollama_base_url}/api/create", json=payload, timeout=300)
    except Exception as e:
        logger.error(f"Failed to contact Ollama /api/create: {e}")
        raise

    if resp.status_code != 200:
        logger.error(f"Ollama create failed [{resp.status_code}]: {resp.text}")
        raise RuntimeError(f"Ollama create failed with status {resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        data = {}

    logger.info("Created derived Ollama model '%s' from '%s'", new_model, base_model)
    return {
        "model": new_model,
        "base_model": base_model,
        "train_path": train_path,
        "raw_response": data,
    }

def register_gguf_with_ollama(
    gguf_path: str,
    ollama_model_name: str,
    ollama_base_url: str = "http://localhost:11434",
) -> Dict[str, object]:
    """
    Register an already-exported GGUF file as a new Ollama model.

    This is the backend counterpart to the save step in unsloth_finetune.py.
    Useful when the UI wants to re-register an existing GGUF without re-training.
    """
    gguf_abs = Path(gguf_path).resolve()
    if not gguf_abs.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_abs}")

    modelfile_content = f"FROM {gguf_abs}\n"
    payload = {
        "model": ollama_model_name,
        "modelfile": modelfile_content,
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
def build_retraining_dataset_sync(
    total_examples: int = 300,
    benign_ratio: float = 0.5,
    refusal_ratio: float = 0.3,
    jailbreak_ratio: float = 0.1,
    create_train_test_split: bool = False,
    test_ratio: float = 0.2,
) -> Dict[str, object]:
    """
    Synchronous helper for Streamlit pages.

    Uses asyncio.run so we don't manually manage event loops,
    which can conflict with Streamlit's own lifecycle.
    """

    async def _run():
        return await build_retraining_dataset(
            total_examples=total_examples,
            benign_ratio=benign_ratio,
            refusal_ratio=refusal_ratio,
            jailbreak_ratio=jailbreak_ratio,
            create_train_test_split=create_train_test_split,
            test_ratio=test_ratio,
        )

    # If an event loop is already running (very rare in Streamlit),
    # fall back to creating a task on that loop.
    try:
        return asyncio.run(_run())
    except RuntimeError as e:
        # "asyncio.run() cannot be called from a running event loop"
        if "running event loop" not in str(e):
            raise
        loop = asyncio.get_event_loop()
        return loop.create_task(_run())


