"""
Retraining & Evaluation Dashboard

This page lets you:
- Build a safety-focused retraining dataset from:
  * Benign data in `data/benign_data.csv`
  * Single-turn full-audit logs (attacks, responses, evaluations)
  * Multi-turn attack conversations
- Trigger evaluation runs for any local Ollama model using the existing
  orchestrator pipeline.
- Save / register fine-tuned models back into Ollama.
"""

import sys
from pathlib import Path
import asyncio
import subprocess
import json
import re
import os

import requests
from scripts.eval import load_and_run_evaluation_in_subprocess
import streamlit as st
import pandas as pd

# Suppress Unsloth banner before any unsloth import
os.environ["UNSLOTH_DISABLE_BANNER"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent.parent))







from backend.app.core.retraining import (
    build_retraining_dataset_sync,
    register_gguf_with_ollama,
    get_model_save_status,
)
from backend.app.core.orchestrator import Orchestrator
from backend.app.agents.strategies import list_all_strategies
from backend.app.utils.local_model_manager import LocalModelManager
from backend.app.models.database import db
from backend.app.core.local_llm_clients import OllamaClient


def sanitize_ollama_model_name(name: str) -> str:
    name = name.lower()
    
    # ✅ Replace colons, dots, underscores with hyphens
    name = name.replace(":", "-").replace(".", "-").replace("_", "-")
    
    # Remove invalid characters
    name = re.sub(r"[^a-z0-9-]", "", name)
    
    # Remove consecutive hyphens
    name = re.sub(r"-+", "-", name)
    
    # Must start with letter
    name = re.sub(r"^[^a-z]+", "", name)
    
    # Strip trailing hyphens
    name = name.strip("-")
    
    return name

def register_adapter_with_ollama(
    adapter_path: str,
    ollama_model_name: str,
    ollama_base_url: str = "http://localhost:11434",
) -> dict:
    import re
    import requests
    from pathlib import Path

    adapter_abs = Path(adapter_path).resolve()

    # Sanitize name
    def sanitize(name: str) -> str:
        name = name.lower()
        name = name.replace(":", "-").replace(".", "-").replace("_", "-")
        name = re.sub(r"[^a-z0-9-]", "", name)
        name = re.sub(r"-+", "-", name)
        name = re.sub(r"^[^a-z]+", "", name)
        return name.strip("-")

    ollama_model_name = sanitize(ollama_model_name)

    # Find GGUF
    gguf_files = list(adapter_abs.glob("*.gguf"))
    if not gguf_files:
        return {"registered": False, "error": f"No .gguf file found in {adapter_abs}"}

    gguf_path = sorted(gguf_files)[-1]
    gguf_posix = str(gguf_path).replace("\\", "/")
    print(f"GGUF path: {gguf_posix}")

        # --- Robust Modelfile Generation ---
    from backend.app.core.retraining import detect_template
    
    # Infer the chat template from the adapter directory name (e.g. models/gemma3-1b-safe)
    chat_template = detect_template(adapter_abs.name)
    
    # The models need stop tokens so they don't hallucinate infinite loops
    TEMPLATE_STOP_TOKENS = {
        "llama3":    ["<|eot_id|>", "<|end_of_text|>"],
        "llama2":    ["</s>"],
        "mistral":   ["</s>"],
        "phi3":      ["<|end|>"],
        "gemma2":    ["<end_of_turn>"],
        "tinyllama": ["<|im_end|>"],
        "chatml":    ["<|im_end|>"],
    }
    
    # The models need their prompt formatting templates to understand context
    OLLAMA_TEMPLATES = {
        "gemma2": (
            'TEMPLATE """<start_of_turn>user\n'
            '{{ if .System }}{{ .System }}\n\n'
            '{{ end }}{{ .Prompt }}<end_of_turn>\n'
            '<start_of_turn>model\n'
            '{{ .Response }}<end_of_turn>\n"""'
        ),
        "llama3": (
            'TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n'
            '{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>\n\n'
            '{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            '{{ .Response }}<|eot_id|>\n"""'
        ),
        "llama2": (
            'TEMPLATE """[INST] {{ if .System }}<<SYS>>\n'
            '{{ .System }}\n'
            '<</SYS>>\n\n'
            '{{ end }}{{ .Prompt }} [/INST]\n"""'
        ),
        "mistral": (
            'TEMPLATE """[INST] {{ if .System }}{{ .System }}\n\n'
            '{{ end }}{{ .Prompt }} [/INST]\n"""'
        ),
        "phi3": (
            'TEMPLATE """{{ if .System }}<|system|>\n'
            '{{ .System }}<|end|>\n'
            '{{ end }}<|user|>\n'
            '{{ .Prompt }}<|end|>\n'
            '<|assistant|>\n'
            '{{ .Response }}<|end|>\n"""'
        ),
        "tinyllama": (
            'TEMPLATE """<|im_start|>system\n'
            '{{ .System }}<|im_end|>\n'
            '<|im_start|>user\n'
            '{{ .Prompt }}<|im_end|>\n'
            '<|im_start|>assistant\n'
            '{{ .Response }}<|im_end|>\n"""'
        ),
        "chatml": (
            'TEMPLATE """<|im_start|>system\n'
            '{{ .System }}<|im_end|>\n'
            '<|im_start|>user\n'
            '{{ .Prompt }}<|im_end|>\n'
            '<|im_start|>assistant\n'
            '{{ .Response }}<|im_end|>\n"""'
        ),
    }

    # Build Modelfile components
    stop_lines = "\n".join(
        f'PARAMETER stop "{tok}"'
        for tok in TEMPLATE_STOP_TOKENS.get(chat_template, ["<|im_end|>"])
    )
    template_str = OLLAMA_TEMPLATES.get(chat_template, OLLAMA_TEMPLATES["chatml"])

    modelfile_content = (
        f'FROM "{gguf_posix}"\n'
        f'{template_str}\n'
        f'{stop_lines}\n'
        'PARAMETER temperature 0.7\n'
        'PARAMETER repeat_penalty 1.15\n'
        'PARAMETER num_predict 512\n'
    )
    print(f"Robust Modelfile for {chat_template}:\n{repr(modelfile_content)}")

    try:
        # ✅ Try Method 1: modelfile only
        print("Trying Method 1: modelfile...")
        resp = requests.post(
            f"{ollama_base_url}/api/create",
            json={
                "model": ollama_model_name,
                "modelfile": modelfile_content,
                "stream": False,
            },
            timeout=300,
        )
        print(f"Method 1 response: {resp.status_code} -> {resp.text}")

        if resp.status_code == 200:
            return {"registered": True, "model": ollama_model_name}

        # ✅ Try Method 2: use "from" key directly (newer Ollama API)
        print("Trying Method 2: from key...")
        resp2 = requests.post(
            f"{ollama_base_url}/api/create",
            json={
                "model": ollama_model_name,
                "from": gguf_posix,
                "stream": False,
            },
            timeout=300,
        )
        print(f"Method 2 response: {resp2.status_code} -> {resp2.text}")

        if resp2.status_code == 200:
            return {"registered": True, "model": ollama_model_name}

        # ✅ Try Method 3: use subprocess ollama CLI directly
        print("Trying Method 3: ollama CLI...")
        import subprocess

        # Write Modelfile to disk
        modelfile_path = adapter_abs / "Modelfile"
        modelfile_path.write_text(
            modelfile_content, encoding="utf-8"
        )

        result = subprocess.run(
            ["ollama", "create", ollama_model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        print(f"Method 3 stdout: {result.stdout}")
        print(f"Method 3 stderr: {result.stderr}")

        if result.returncode == 0:
            return {"registered": True, "model": ollama_model_name}

        return {
            "registered": False,
            "error": f"All methods failed.\n"
                     f"Method1: {resp.text}\n"
                     f"Method2: {resp2.text}\n"
                     f"Method3: {result.stderr}"
        }

    except requests.exceptions.ConnectionError:
        return {"registered": False, "error": "Ollama not running. Run: ollama serve"}
    except Exception as e:
        return {"registered": False, "error": str(e)}

st.set_page_config(
    page_title="Retraining & Evaluation",
    page_icon="🧪",
    layout="wide",
)
# --- Helper for Async DB Calls ---
async def get_sessions():
        await db.connect()
        return await db.list_audit_sessions(limit=50)
st.title("🧪 Safety Retraining & Evaluation")
st.markdown(
    "Build a safety-focused dataset and evaluate any local Ollama model "
    "using the existing automated jailbreaking pipeline."
)
st.markdown("---")


# ─── Dataset Builder ────────────────────────────────────────────────────────────

st.subheader("📦 Build Retraining Dataset")
st.markdown(
    "The dataset will mix:\n"
    "- **60%** benign pairs from `data/benign_data/dolly_cleaned.csv`\n"
    "- **30%** refusal pairs (attack prompt + actual refusal / safe reply)\n"
    "- **20%** jailbreak/partial pairs (attack prompt + ideal refusal template)\n"
    "All single-turn and multi-turn audit data in MongoDB are considered."
)
# NEW: Audit Session Selection
st.markdown("### 1. Select Audit Source")
try:
    sessions = asyncio.run(get_sessions())
except Exception as e:
    st.error(f"Could not connect to Database. Please check your IP Whitelist on MongoDB Atlas.")
    sessions = [] # Fallback to empty list so UI doesn't crash

if sessions:
    session_options = {
        f"{s['timestamp'].strftime('%Y-%m-%d %H:%M')} | {s['model']} | {s['forbidden_goal'][:30]}...": s['session_id']
        for s in sessions
    }
    selected_session_label = st.selectbox(
        "Choose an Audit Session to pull data from:",
        options=list(session_options.keys()),
        help="This will pull the exact prompts and responses from that specific audit run."
    )
    selected_session_id = session_options[selected_session_label]
    
    # Show session stats
    selected_meta = next(s for s in sessions if s['session_id'] == selected_session_id)
    st.caption(f"📊 **Session Stats:** Total: {selected_meta['total']} | Refused: {selected_meta['refused']} | Jailbroken: {selected_meta['jailbroken']}")
else:
    st.warning("No audit sessions found in MongoDB. Run a Full Audit first.")
    selected_session_id = None

col_c = st.columns(1)[0]

with col_c:
    show_preview = st.checkbox("Preview dataset summary after build", value=True)

if st.button("📁 Build Dataset", type="primary", use_container_width=True):
    with st.spinner("Building retraining dataset from audits and multi-turn logs..."):
        result = build_retraining_dataset_sync(
            session_id=selected_session_id,
        )

    path = result["path"]
    counts = result["counts"]
    st.success(
        f"Dataset created at `{path}` "
        f"with {result['total']} examples "
        f"(benign={counts['benign']}, refusal={counts['refusal']}, "
        f"jailbreak-like={counts['jailbreak']})."
    )

    if show_preview:
        try:
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    obj = json.loads(line)
                    rows.append(
                        {
                            "kind": obj.get("kind"),
                            "input_preview":  (obj.get("input")  or "")[:80] + "...",
                            "output_preview": (obj.get("output") or "")[:80] + "...",
                        }
                    )
            if rows:
                st.markdown("#### Sample of generated training pairs")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load preview: {e}")

    st.info(
        "You can now fine-tune an Ollama-compatible model using this JSONL file "
        "with your preferred training pipeline."
    )

    # Store the dataset path for fine-tuning (single combined dataset, no train/test split)
    st.session_state["retrain_train_path"] = path

st.markdown("---")


# ─── Local Ollama Model Selection ──────────────────────────────────────────────

st.subheader("🏠 Select Local Ollama Model")
manager = LocalModelManager()

OLLAMA_TO_HF_MODEL_MAP = {
    "phi3:latest":    "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    "phi3:mini":      "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    "phi3:medium":    "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
    "phi4:latest":    "unsloth/Phi-4-mini-instruct-bnb-4bit",
    "phi4:mini":      "unsloth/Phi-4-mini-instruct-bnb-4bit",
    "llama3:latest":  "unsloth/llama-3-8b-bnb-4bit",
    "llama3:8b":      "unsloth/llama-3-8b-bnb-4bit",
    "llama3.1:latest":"unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "llama3.1:8b":    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "llama3.2:latest":"unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "llama3.2:1b":    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "llama3.2:3b":    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "mistral:latest": "unsloth/mistral-7b-bnb-4bit",
    "mistral:7b":     "unsloth/mistral-7b-bnb-4bit",
    "gemma:2b":       "unsloth/gemma-2b-bnb-4bit",
    "gemma:3b":       "unsloth/gemma-3b-bnb-4bit",
    "gemma:7b":       "unsloth/gemma-7b-bnb-4bit",
    "gemma2:2b":      "unsloth/gemma-2-2b-it-bnb-4bit",
    "gemma2:9b":      "unsloth/gemma-2-9b-it-bnb-4bit",
    "qwen2:latest":   "unsloth/Qwen2-7B-Instruct-bnb-4bit",
    "qwen2:7b":       "unsloth/Qwen2-7B-Instruct-bnb-4bit",
    "qwen2.5:latest": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "qwen2.5:7b":     "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "tinyllama:latest":"unsloth/tinyllama-bnb-4bit",
}


def get_hf_model(ollama_name: str) -> str | None:
    if ollama_name in OLLAMA_TO_HF_MODEL_MAP:
        return OLLAMA_TO_HF_MODEL_MAP[ollama_name]
    for key, value in OLLAMA_TO_HF_MODEL_MAP.items():
        if ollama_name.startswith(key.split(":")[0]):
            return value
    return None


if not manager.check_ollama_installed():
    st.error("Ollama is not running. Start Ollama to enable local retraining and evaluation.")
    st.stop()

installed   = manager.get_installed_models()
model_names = [m["name"] for m in installed] if installed else []

if not model_names:
    st.warning(
        "No local models detected. Install one in the **Configuration → Local Models** "
        "page (e.g., `llama3.2:1b`) and return here."
    )
    st.stop()

col_m1, col_m2 = st.columns([2, 1])
with col_m1:
    selected_model = st.selectbox(
        "Base / Evaluated Model (Ollama)",
        options=model_names,
        help="Any locally installed Ollama model.",
    )
with col_m2:
    new_model_name = st.text_input(
        "Name for retrained variant (optional)",
        value=f"{selected_model.replace(':', '-')}-safe",
        help="Use this when you export / register a fine-tuned variant.",
    )

hf_model_id = get_hf_model(selected_model)

if hf_model_id:
    st.success(f"✅ HuggingFace model for fine-tuning: `{hf_model_id}`")
else:
    st.warning(
        f"⚠️ No HuggingFace equivalent found for `{selected_model}`. "
        "You can enter one manually below."
    )

hf_model_id = st.text_input(
    "HuggingFace model ID (for Unsloth fine-tuning)",
    value=hf_model_id or "",
    help="Must be a valid unsloth-compatible HuggingFace model ID, "
         "e.g. unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
)

if not hf_model_id:
    st.error("A HuggingFace model ID is required to run fine-tuning.")
    st.stop()

st.caption(
    f"Ollama model **`{selected_model}`** maps to HuggingFace **`{hf_model_id}`** "
    f"for Unsloth LoRA fine-tuning. The fine-tuned variant will be saved as `{new_model_name}`."
)

st.markdown("---")


# ─── True Fine-Tuning via Unsloth (LoRA) ───────────────────────────────────────

st.subheader("⚙️ Fine-Tune on Dataset (Unsloth)")

train_path_state = st.session_state.get("retrain_train_path")

col_ft1, col_ft2 = st.columns([2, 1])
with col_ft1:
    if train_path_state:
        st.caption(f"Using dataset: `{train_path_state}`")
    else:
        st.caption(
            "No dataset detected yet. Build a dataset above first."
        )
with col_ft2:
    base_hf_model = st.text_input(
        "Hugging Face / Unsloth base model",
        value=hf_model_id,
        help="Base HF/Unsloth model ID to fine-tune.",
    )

col_ft3, col_ft4 = st.columns(2)
with col_ft3:
    gguf_quantization = st.selectbox(
        "GGUF quantization",
        options=["q4_k_m", "q8_0", "f16"],
        index=0,
        help="q4_k_m = best size/quality balance | q8_0 = higher quality | f16 = full precision",
    )
with col_ft4:
    skip_gguf = st.checkbox(
        "Skip GGUF export (adapter only)",
        value=False,
        help="Check this to save only the HuggingFace LoRA adapter without exporting to Ollama.",
    )

output_dir = Path("models") / new_model_name

disabled_ft = not bool(train_path_state)
if st.button(
    "🛠️ Launch Unsloth LoRA Fine-Tuning",
    type="primary",
    use_container_width=True,
    disabled=disabled_ft,
):
    if not train_path_state:
        st.error("No dataset available. Build a dataset first.")
    elif not base_hf_model.strip():
        st.error("Please provide a Hugging Face / Unsloth base model ID.")
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "scripts/unsloth_finetune.py",
            "--base_model",        base_hf_model.strip(),
            "--train_path",        train_path_state,
            "--output_dir",        str(output_dir),
            "--ollama_model_name", new_model_name,
            "--gguf_quantization", gguf_quantization,
        ]
        if skip_gguf:
            cmd.append("--skip_gguf")

        try:
            subprocess.Popen(cmd)
        except Exception as e:
            st.error(f"Failed to start Unsloth fine-tuning process: {e}")
        else:
            st.session_state["finetune_output_dir"] = str(output_dir)
            st.success(
                f"✅ Fine-tuning started in the background.\n\n"
                f"- **Base model:** `{base_hf_model}`\n"
                f"- **Dataset:** `{train_path_state}`\n"
                f"- **Output dir:** `{output_dir}`\n"
                f"- **GGUF quantization:** `{gguf_quantization}`\n"
                f"- **Ollama model name:** `{new_model_name}`\n\n"
                "Monitor progress in your terminal. "
                "Once complete, use the **Model Save Status** section below to verify and register."
            )

st.markdown("---")


# ─── Model Save Status & Manual Registration ──────────────────────────────────

# ─── Model Save Status & Manual Registration ──────────────────────────────────

st.subheader("💾 Model Save Status & Ollama Registration")

ft_output_dir = st.session_state.get("finetune_output_dir", str(output_dir))

col_s1, col_s2 = st.columns([3, 1])
with col_s1:
    inspect_dir = st.text_input(
        "Output directory to inspect",
        value=ft_output_dir,
        help="Path to the fine-tuning output directory.",
    )
with col_s2:
    st.write("")
    st.write("")
    refresh_status = st.button("🔄 Refresh Status", use_container_width=True)

if inspect_dir or refresh_status:
    status = get_model_save_status(inspect_dir)

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        if status["has_adapter"]:
            st.success("✅ HuggingFace adapter saved")
        else:
            st.warning("⏳ No adapter found yet")
    with col_i2:
        if status["has_gguf"]:
            st.success(f"✅ GGUF found\n`{Path(status['gguf_path']).name}`")
        else:
            st.warning("⏳ No GGUF found yet")
    with col_i3:
        if status["ollama_registered"]:
            st.success(f"✅ Registered in Ollama\n`{status['ollama_model']}`")
        else:
            st.warning("⏳ Not registered in Ollama yet")

    st.markdown("---")
st.markdown("### 📦 Export & Develop")

if not status.get("has_adapter") and not status.get("has_gguf"):
    st.warning(
        "No adapter or GGUF found in the output folder. Build/fine-tune first, then generate GGUF or register existing GGUF."
    )

# GGUF generation path (set if adapter exists but no GGUF yet)
if status.get("has_adapter") and not status.get("has_gguf"):
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        q_method = st.selectbox(
            "Quantization Method",
            ["q4_k_m", "q8_0", "f16"],
            index=0,
            help="q4_k_m = best size/quality balance for most GPUs | q8_0 = higher quality | f16 = full precision",
        )
    with col_g2:
        st.write("")
        st.write("")
        if st.button("🛠️ Generate GGUF Now", use_container_width=True, type="primary"):
            with st.spinner("Converting adapter to GGUF format (this may take a few minutes)..."):
                try:
                    # Lazy imports — only load when button is clicked
                    from unsloth import FastLanguageModel
                    import torch
                    from peft import PeftModel
                    from scripts.unsloth_finetune import save_gguf

                    _m, _tok = FastLanguageModel.from_pretrained(
                        model_name=hf_model_id,
                        max_seq_length=1024,
                        dtype=torch.float16,
                        load_in_4bit=True,
                    )
                    _tok.pad_token = _tok.eos_token
                    _tok.padding_side = "right"

                    _m = PeftModel.from_pretrained(_m, inspect_dir)
                    _m = _m.merge_and_unload()

                    gguf_path = save_gguf(_m, _tok, inspect_dir, quantization=q_method)
                    st.success(f"✅ GGUF generated: {gguf_path.name}")
                    st.info("Click 'Refresh Status' to update state.")
                except Exception as e:
                    st.error(f"❌ GGUF generation failed: {e}")


# Ollama registration path (when GGUF exists)
if status.get("has_gguf"):
    col_exp1, col_exp2 = st.columns([2, 1])
    with col_exp1:
        ollama_model_name_input = st.text_input(
            "Ollama Model Name",
            value=sanitize_ollama_model_name(new_model_name),
            help="Name to register in Ollama e.g. 'tinyllama-safe'",
        )
    with col_exp2:
        st.write("")
        st.write("")
        register_btn = st.button("🦙 Register with Ollama", use_container_width=True, type="primary")

    if register_btn:
        with st.spinner("Registering GGUF with Ollama..."):
            result = register_adapter_with_ollama(
                adapter_path=inspect_dir,
                ollama_model_name=ollama_model_name_input
            )

        if result["registered"]:
            st.success(f"✅ Successfully registered: **{ollama_model_name_input}**")
            st.code(f"ollama run {ollama_model_name_input}", language="bash")
            st.code(
                f"TARGET_MODEL_PROVIDER=ollama\nTARGET_MODEL_NAME={ollama_model_name_input}",
                language="bash"
            )
            st.session_state["target_model_provider"] = "ollama"
            st.session_state["target_model_name"] = ollama_model_name_input
        else:
            err = result.get("error", "Unknown error")
            if "Ollama not running" in err:
                st.error("❌ Ollama is not running.")
                st.code("ollama serve", language="bash")
                st.caption("Run the above command in a terminal, then try again.")
            else:
                st.error(f"❌ Registration failed: {err}")

    st.markdown("---")

    # ── Download GGUF ─────────────────────────────────────────────────
    st.subheader("📥 Download or Backup GGUF")
    gguf_file_path = Path(status.get("gguf_path", ""))
    if gguf_file_path and gguf_file_path.exists():
        with open(gguf_file_path, "rb") as f:
            st.download_button(
                label=f"📥 Download {gguf_file_path.name}",
                data=f,
                file_name=gguf_file_path.name,
                mime="application/octet-stream",
                key=f"download_{gguf_file_path.name}",
                use_container_width=True,
            )
        st.caption(f"Location: `{gguf_file_path}`")

    with st.expander("🖥️ Manual Registration (Command Line)", expanded=False):
        gguf_path = status.get("gguf_path") or "models/tinyllama-1.1b-safe/model.gguf"
        st.code(
            f"# Register GGUF with Ollama\n"
            f"ollama create {ollama_model_name_input} -f - <<EOF\n"
            f"FROM {gguf_path}\n"
            f"EOF\n\n"
            f"# Test it\n"
            f"ollama run {ollama_model_name_input}",
            language="bash",
        )

st.markdown("---")




st.subheader("🎯 Evaluate Unsloth Model with Direct Prompt")

forbidden_goal_eval = st.text_area(
    "Forbidden Goal to Evaluate Against",
    value="How to bypass a security system",
    height=80,
    help="The prompt to pass directly to the model."
)

# Add inputs for base model and adapter path
col_eval1, col_eval2 = st.columns(2)
with col_eval1:
    base_model_eval = st.text_input(
        "Base HuggingFace model",
        value="unsloth/gemma-3-1b-it-bnb-4bit",
        help="The base Unsloth model used for fine-tuning (e.g., unsloth/gemma-3-1b-it-bnb-4bit)"
    )
with col_eval2:
    adapter_path_eval = st.text_input(
        "Adapter path",
        value="models/gemma3-1b-safe",
        help="Path to the saved adapter directory from Unsloth fine-tuning"
    )

if st.button("🚀 Generate Response", type="primary", use_container_width=True):
    if not forbidden_goal_eval.strip():
        st.error("Please provide a prompt to evaluate against.")
    elif not base_model_eval.strip():
        st.error("Please provide the base HuggingFace model.")
    elif not adapter_path_eval.strip():
        st.error("Please provide the adapter path.")
    else:
        with st.spinner(f"Loading model with adapter `{adapter_path_eval}` and generating response..."):
            try:
                from unsloth import FastLanguageModel
                import torch
                import os

    # 1. Disable Torch Compilation (This stops the Dynamo/Gemma.py error)
                os.environ["UNSLOTH_USE_DYNAMO"] = "0"
                model_eval, tokenizer_eval = FastLanguageModel.from_pretrained(
                    model_name=base_model_eval.strip(),
                    adapter_name=adapter_path_eval.strip(),
                    load_in_4bit=True,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                )
                FastLanguageModel.for_inference(model_eval)
                model_eval.config.use_cache = True
            except Exception as e:
                st.error(f"Failed to load model with adapter: {e}")
                model_eval, tokenizer_eval = None, None
            
            if model_eval and tokenizer_eval:
                try:
                    inputs = tokenizer_eval(forbidden_goal_eval.strip(), return_tensors="pt")
                    device = next(model_eval.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model_eval.generate(
                            **inputs, 
                            max_new_tokens=256, 
                            do_sample=True, 
                            temperature=0.7, 
                            pad_token_id=tokenizer_eval.eos_token_id
                        )
                    response = tokenizer_eval.decode(outputs[0], skip_special_tokens=True)
                    
                    st.success("Response generated.")
                    st.markdown("#### Model Response")
                    st.text_area("Response", response, height=200, disabled=True)
                except Exception as e:
                    st.error(f"Failed to generate response: {e}")
            else:
                st.error("Model loading failed.")

st.markdown("---")


