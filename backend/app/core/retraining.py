"""
Retraining dataset builder for Ollama safety fine-tuning.
Updated to support MongoDB audit_sessions and per-model chat templates.
"""

from __future__ import annotations
from dataclasses import dataclass, field
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


import random

REFUSAL_STYLES = [
    "I can't assist with that request.",
    "I'm unable to help with that.",
    "That request involves unsafe or unethical activity, so I must decline.",
    "I cannot provide that information.",
    "That goes against safety guidelines."
]

REFUSAL_SUFFIX = [
    "",
    " Let me know if you'd like help with something safe.",
    " I can help with a different topic instead.",
    " Feel free to ask another question."
]

def generate_refusal(prompt: str):
    base = random.choice(REFUSAL_STYLES)

    # context-aware hint (VERY POWERFUL)
    if "hack" in prompt.lower():
        base = "I can't help with hacking or unauthorized access."
    elif "fraud" in prompt.lower():
        base = "I can't assist with fraudulent or deceptive activities."

    return base + random.choice(REFUSAL_SUFFIX)




# ========================================================================
# CHAT TEMPLATE DETECTION & FORMATTING
# ========================================================================

# Human-readable registry: template_key → (display_name, example_model_tags)
# Used by the UI to populate a model selector and by tests to enumerate templates.
SUPPORTED_TEMPLATES: dict[str, dict[str, str]] = {
    "llama3":  {"name": "Llama 3 Instruct",   "example": "meta-llama/Llama-3-8B-Instruct"},
    "llama2":  {"name": "Llama 2 Chat",        "example": "meta-llama/Llama-2-7b-chat-hf"},
    "mistral": {"name": "Mistral / Mixtral",   "example": "mistralai/Mistral-7B-Instruct-v0.3"},
    "phi3":    {"name": "Phi-3 / Phi-4",       "example": "microsoft/Phi-3-mini-4k-instruct"},
    "gemma2":  {"name": "Gemma 2",             "example": "google/gemma-2-9b-it"},
    "tinyllama":{"name": "TinyLlama",          "example": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
    "chatml":  {"name": "ChatML (generic)",    "example": "any unrecognised model"},
}

# Each entry: (compiled regex matched against the lowercased model name, template key)
# Order matters — more specific patterns must come before broader ones.
_TEMPLATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"llama-?3"),           "llama3"),
    (re.compile(r"llama-?2"),           "llama2"),
    (re.compile(r"mistral|mixtral"),    "mistral"),
    (re.compile(r"phi-?4"),             "phi3"),    # Phi-4 reuses Phi-3 tokens
    (re.compile(r"phi-?3"),             "phi3"),
    (re.compile(r"gemma-?2|gemma2"),    "gemma2"),
    (re.compile(r"gemma"),              "gemma2"),
    # TinyLlama uses ChatML tokens identical to the default fallback, but we
    # match it explicitly so callers get a clean log line and the right label.
    (re.compile(r"tinyllama"),          "tinyllama"),
]

_DEFAULT_TEMPLATE = "chatml"


def get_template_for_model(model_name: str | None) -> str:
    """
    Public helper — resolve and return the chat-template key for *model_name*.

    Call this from the Streamlit UI, the fine-tune script, or any other entry
    point **before** building the dataset, so the template can be displayed to
    the user or stored alongside the run metadata.

    Parameters
    ----------
    model_name:
        Raw model identifier — short tag (``"llama3"``) or a full HuggingFace /
        Ollama path (``"meta-llama/Llama-3-8B-Instruct"``).  ``None`` or an
        empty string both fall back to ChatML with a warning.

    Returns
    -------
    str
        One of the keys in :data:`SUPPORTED_TEMPLATES`.
    """
    return _detect_template(model_name or "")


def _detect_template(model_name: str) -> str:
    """Internal implementation — prefer calling ``get_template_for_model()``."""
    if not model_name:
        logger.warning("No model name provided — using default '%s' template.", _DEFAULT_TEMPLATE)
        return _DEFAULT_TEMPLATE

    lower = model_name.lower()
    for pattern, key in _TEMPLATE_PATTERNS:
        if pattern.search(lower):
            display = SUPPORTED_TEMPLATES.get(key, {}).get("name", key)
            logger.info("Model '%s' → template '%s' (%s).", model_name, key, display)
            return key

    logger.warning(
        "Model '%s' did not match any known template — falling back to '%s'. "
        "Pass the correct model name or choose a template explicitly.",
        model_name,
        _DEFAULT_TEMPLATE,
    )
    return _DEFAULT_TEMPLATE


# Keep the old name as a private alias so existing internal callers still work.
detect_template = _detect_template


def format_prompt(instruction: str, user_input: str, output: str, template: str) -> str:
    """
    Render a fully-formed training string using the appropriate chat template.

    The rendered string includes the assistant turn so the fine-tuning loss is
    computed over the response tokens only (standard SFT behaviour).

    Supported templates
    -------------------
    chatml     — OpenAI / Unsloth default  (<|im_start|> … <|im_end|>)
    llama3     — Meta Llama-3 instruct     (<|begin_of_text|> … <|eot_id|>)
    llama2     — Meta Llama-2 chat         ([INST] … [/INST])
    mistral    — Mistral / Mixtral         ([INST] … [/INST], no system token)
    phi3       — Microsoft Phi-3 / Phi-4   (<|system|> … <|end|>)
    gemma2     — Google Gemma-2            (<start_of_turn> … <end_of_turn>)
    tinyllama  — TinyLlama-1.1B-Chat       (<|im_start|> … <|im_end|>, identical to ChatML)
    """
    system_block = instruction.strip()
    user_block   = user_input.strip()
    resp_block   = output.strip()

    if template == "llama3":
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_block}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_block}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{resp_block}<|eot_id|>"
        )

    if template == "llama2":
        # Llama-2 encodes the system prompt inside the first [INST] block.
        return (
            f"[INST] <<SYS>>\n{system_block}\n<</SYS>>\n\n"
            f"{user_block} [/INST] {resp_block} </s>"
        )

    if template == "mistral":
        # Mistral has no dedicated system token; prepend to user turn.
        combined_user = f"{system_block}\n\n{user_block}" if system_block else user_block
        return f"[INST] {combined_user} [/INST]{resp_block}</s>"

    if template == "phi3":
        return (
            f"<|system|>\n{system_block}<|end|>\n"
            f"<|user|>\n{user_block}<|end|>\n"
            f"<|assistant|>\n{resp_block}<|end|>"
        )

    if template == "gemma2":
        # Gemma has no system role; prepend instruction to user turn.
        combined_user = f"{system_block}\n\n{user_block}" if system_block else user_block
        return (
            f"<start_of_turn>user\n{combined_user}<end_of_turn>\n"
            f"<start_of_turn>model\n{resp_block}<end_of_turn>"
        )

    if template == "tinyllama":
        # TinyLlama-1.1B-Chat-v1.0 was trained with ChatML tokens — the template
        # is token-for-token identical to the ChatML default below.  We keep it
        # as an explicit branch so log output clearly names the model family.
        return (
            f"<|im_start|>system\n{system_block}<|im_end|>\n"
            f"<|im_start|>user\n{user_block}<|im_end|>\n"
            f"<|im_start|>assistant\n{resp_block}<|im_end|>"
        )

    # Default: ChatML (OpenAI / Unsloth)
    return (
        f"<|im_start|>system\n{system_block}<|im_end|>\n"
        f"<|im_start|>user\n{user_block}<|im_end|>\n"
        f"<|im_start|>assistant\n{resp_block}<|im_end|>"
    )


# ========================================================================
# DATA STRUCTURES
# ========================================================================

@dataclass
class TrainingExample:
    """Normalized training record used to build JSONL rows."""
    kind: ExampleType
    instruction: str
    user_input: str
    output: str
    # Populated by apply_template(); empty string means "not yet formatted".
    formatted_text: str = field(default="", compare=False)

    def apply_template(self, template: str) -> "TrainingExample":
        """Render the chat template in-place and return self for chaining."""
        self.formatted_text = format_prompt(
            self.instruction, self.user_input, self.output, template
        )
        return self

    def to_jsonl(self) -> str:
        row: dict[str, Any] = {
            "kind":           self.kind,
            "instruction":    self.instruction,
            "input":          self.user_input,
            "output":         self.output,
        }
        # Only include the rendered field when it has actually been set,
        # so legacy callers that don't pass a model name still get clean rows.
        if self.formatted_text:
            row["text"] = self.formatted_text
        return json.dumps(row, ensure_ascii=False)


# ========================================================================
# DATA LOADING HELPERS
# ========================================================================

def _load_benign_examples(benign_path: Path, max_examples: int | None = None) -> List[TrainingExample]:
    if not benign_path.exists():
        logger.warning(f"Benign dataset not found at {benign_path}")
        return []

    examples: List[TrainingExample] = []
    try:
        print("Loading benign from:", benign_path)
        with benign_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                goal   = (row.get("input")   or "").strip()
                target = (row.get("output") or "").strip()
                if not goal or not target:
                    continue
                examples.append(TrainingExample(
                    kind="benign",
                    instruction="Answer the user query helpfully and safely.",
                    user_input=goal,
                    output=target,
                ))
    except Exception as e:
        logger.error(f"Failed to load benign data: {e}")
        return []

    random.shuffle(examples)
    return examples[:max_examples] if max_examples else examples


async def _process_audit_examples(
    raw_examples: List[Dict[str, Any]],
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Convert raw DB examples into TrainingExample objects."""
    refusal_examples:  list[TrainingExample] = []
    jailbreak_examples: list[TrainingExample] = []

    base_instruction = (
        "You are a safe AI assistant. Answer user queries helpfully while "
        "refusing any harmful, illegal, or unethical requests."
    )

    for ex in raw_examples:
        prompt   = (ex.get("generated_prompt") or ex.get("prompt")        or "").strip()
        response = (ex.get("response_text")    or ex.get("response")      or "").strip()
        verdict  = (ex.get("verdict")          or "").upper()

        if not prompt:
            continue

        if verdict == "REFUSED":
            refusal_examples.append(TrainingExample(
                kind="refusal",
                instruction=base_instruction,
                user_input=prompt,
                output=response if response else generate_refusal(prompt),
            ))
        elif verdict in ("JAILBROKEN", "PARTIAL"):
            jailbreak_examples.append(TrainingExample(
                kind="jailbreak",
                instruction=base_instruction,
                user_input=prompt,
                output=generate_refusal(prompt),
            ))

    return refusal_examples, jailbreak_examples


# ========================================================================
# CORE COLLECTION LOGIC
# ========================================================================

async def build_retraining_dataset(
    session_id: str | None = None,
    model_name: str | None = None,
    benign_filename: str = "data/Benign_data/dolly_cleaned.csv",
    output_dir: str = "data",
) -> Dict[str, object]:

    template = detect_template(model_name or "")
    logger.info(f"Using template: {template}")

    # 🔥 CONSTANTS
    TARGET_BENIGN_RATIO = 0.6
    MIN_TOTAL = 100
    MAX_TOTAL = 500

    database = Database()
    await database.connect()

    try:
        # ─────────────────────────────────────────
        # 1. Load audit data
        # ─────────────────────────────────────────
        audit_examples = []
        if session_id:
            session = await database.load_audit_session(session_id)
            if session:
                audit_examples = session.get("examples", [])

        refusal_pool, jailbreak_pool = await _process_audit_examples(audit_examples)

        total_adversarial = len(refusal_pool) + len(jailbreak_pool)

        # ─────────────────────────────────────────
        # 2. Limit adversarial size
        # ─────────────────────────────────────────
        max_adversarial = int(MAX_TOTAL * (1 - TARGET_BENIGN_RATIO))

        if total_adversarial > max_adversarial:
            random.shuffle(refusal_pool)
            random.shuffle(jailbreak_pool)

            total = len(refusal_pool) + len(jailbreak_pool)

            keep_refusal = int(len(refusal_pool) / total * max_adversarial)
            keep_jailbreak = max_adversarial - keep_refusal

            refusal_pool = refusal_pool[:keep_refusal]
            jailbreak_pool = jailbreak_pool[:keep_jailbreak]

        total_adversarial = len(refusal_pool) + len(jailbreak_pool)

        # ─────────────────────────────────────────
        # 3. Compute dataset size
        # ─────────────────────────────────────────
        if total_adversarial == 0:
            total_dataset_size = MIN_TOTAL
        else:
            total_dataset_size = int(total_adversarial / (1 - TARGET_BENIGN_RATIO))

        total_dataset_size = max(MIN_TOTAL, min(MAX_TOTAL, total_dataset_size))

        target_benign = int(total_dataset_size * TARGET_BENIGN_RATIO)

        # ─────────────────────────────────────────
        # 4. Load Dolly (benign)
        # ─────────────────────────────────────────
        base_path = Path(__file__).resolve().parents[3]
        benign_path = base_path / benign_filename

        benign_pool = _load_benign_examples(
            benign_path,
            max_examples=target_benign
        )

        # ─────────────────────────────────────────
        # 5. Combine dataset
        # ─────────────────────────────────────────
        all_data = benign_pool + refusal_pool + jailbreak_pool
        if len(all_data) < MIN_TOTAL:
            print(f"⚠️ Expanding dataset: {len(all_data)} → {MIN_TOTAL}")

            needed = MIN_TOTAL - len(all_data)

            extra_benign = _load_benign_examples(
                benign_path,
                max_examples=needed
            )
            benign_pool.extend(extra_benign)  # Add to refusal pool to maintain some balance

            all_data.extend(extra_benign)

        for ex in all_data:
            ex.apply_template(template)

        random.shuffle(all_data)

        # ─────────────────────────────────────────
        # 6. Debug info
        # ─────────────────────────────────────────
        print(f"""
        Dataset Summary:
        Total: {len(all_data)}
        Benign: {len(benign_pool)}
        Refusal: {len(refusal_pool)}
        Jailbreak: {len(jailbreak_pool)}
        """)

        # ─────────────────────────────────────────
        # 7. Save SINGLE dataset
        # ─────────────────────────────────────────
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = base_path / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = out_dir / f"train_dataset_{ts}.jsonl"

        with dataset_path.open("w", encoding="utf-8") as f:
            for ex in all_data:
                f.write(ex.to_jsonl() + "\n")

        return {
            "path": str(dataset_path),
            "total": len(all_data),
            "counts": {
                "benign": len(benign_pool),
                "refusal": len(refusal_pool),
                "jailbreak": len(jailbreak_pool),
            },
            "template": template,
        }

    finally:
        await database.disconnect()


# ========================================================================
# SAMPLING HELPER
# ========================================================================

def _sample_with_ratios(
    benign:    list[TrainingExample],
    refusals:  list[TrainingExample],
    jailbreaks: list[TrainingExample],
    total: int,
    b_r: float,
    r_r: float,
    j_r: float,
) -> tuple[list[TrainingExample], dict[str, int]]:
    target_b = int(total * b_r)
    target_r = int(total * r_r)
    target_j = int(total * j_r)

    sel_b = random.sample(benign,     min(len(benign),     target_b))
    sel_r = random.sample(refusals,   min(len(refusals),   target_r))
    sel_j = random.sample(jailbreaks, min(len(jailbreaks), target_j))

    combined = sel_b + sel_r + sel_j
    random.shuffle(combined)

    return combined, {"benign": len(sel_b), "refusal": len(sel_r), "jailbreak": len(sel_j)}


# ========================================================================
# SYNC WRAPPER
# ========================================================================

def build_retraining_dataset_sync(
    session_id: str | None = None,
    model_name: str | None = None,
) -> dict:
    """
    Synchronous wrapper for :func:`build_retraining_dataset`.

    Streamlit handles async poorly in some versions, so this ensures stability.
    Pass *model_name* to select the correct chat template automatically.
    """
    async def _run() -> dict:
        return await build_retraining_dataset(
            session_id=session_id,
            model_name=model_name
        )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()


# ========================================================================
# OLLAMA HELPERS  (unchanged)
# ========================================================================

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
    """
    gguf_abs = Path(gguf_path).resolve()
    if not gguf_abs.exists():
        raise FileNotFoundError(f"GGUF file not found: {gguf_abs}")

    modelfile_content = f"FROM {gguf_abs}\n"
    payload = {
        "model":     ollama_model_name,
        "from":      str(gguf_abs),
        "modelfile": modelfile_content,
        "stream":    False,
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
        "Registered GGUF '%s' as Ollama model '%s'", gguf_abs, ollama_model_name
    )
    return {
        "registered": True,
        "model":      ollama_model_name,
        "gguf_path":  str(gguf_abs),
        "raw":        data,
    }


def get_model_save_status(output_dir: str) -> Dict[str, object]:
    """
    Inspect an output directory and return a status dict that the UI can render.
    """
    out = Path(output_dir)
    status: Dict[str, object] = {
        "has_adapter":       False,
        "has_gguf":          False,
        "gguf_path":         None,
        "ollama_registered": False,
        "ollama_model":      None,
        "output_dir":        str(out.resolve()),
    }

    if not out.exists():
        return status

    status["has_adapter"] = any(
        (out / f).exists() for f in ("config.json", "adapter_config.json")
    )

    gguf_files = sorted(out.glob("*.gguf"))
    if not gguf_files:
        alt_out = Path(str(out) + "_gguf")
        if alt_out.exists():
            gguf_files = sorted(alt_out.glob("*.gguf"))
            if gguf_files:
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

    reg_file = out / "ollama_registration.json"
    if reg_file.exists():
        try:
            reg_data = json.loads(reg_file.read_text())
            status["ollama_registered"] = bool(reg_data.get("registered"))
            status["ollama_model"]      = reg_data.get("model")
        except Exception:
            pass

    return status