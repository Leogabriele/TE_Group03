"""
Lightweight Unsloth fine-tuning script.

FIX SUMMARY (repetition fixes applied):
  1. format_prompt: EOS/stop token appended to every training example so
     the model learns when to stop generating (primary repetition fix).
  2. register_with_ollama: PARAMETER stop directives added to Modelfile
     so Ollama enforces stop tokens at inference time.
  3. LoRA: lora_alpha raised to 32 (= 2*r) for better generalisation.
"""

import argparse
import json
import re
import os
from pathlib import Path
import requests

# NOTE: unsloth, datasets, transformers, trl, torch are imported
# lazily inside main() so that importing save_gguf from this module
# does NOT trigger Unsloth to load on every Streamlit page reload.


def detect_template(model_name: str) -> str:
    """Infer the chat template key from the base model name."""
    lower = (model_name or "").lower()
    patterns = [
        (r"llama-?3",        "llama3"),
        (r"llama-?2",        "llama2"),
        (r"mistral|mixtral", "mistral"),
        (r"phi-?4|phi-?3",   "phi3"),
        (r"gemma-?2|gemma",  "gemma2"),
        (r"tinyllama",       "tinyllama"),
    ]
    for pattern, key in patterns:
        if re.search(pattern, lower):
            return key
    return "chatml"


# FIX 2 — stop tokens used when building the Ollama Modelfile
TEMPLATE_STOP_TOKENS = {
    "llama3":    ["<|eot_id|>", "<|end_of_text|>"],
    "llama2":    ["</s>"],
    "mistral":   ["</s>"],
    "phi3":      ["<|end|>"],
    "gemma2":    ["<end_of_turn>"],
    "tinyllama": ["<|im_end|>"],
    "chatml":    ["<|im_end|>"],
}


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


# Special token constants (avoids angle-bracket parsing issues)
_BOS       = "\N{LESS-THAN SIGN}|begin_of_text|\N{GREATER-THAN SIGN}"
_HDR_S     = "\N{LESS-THAN SIGN}|start_header_id|\N{GREATER-THAN SIGN}"
_HDR_E     = "\N{LESS-THAN SIGN}|end_header_id|\N{GREATER-THAN SIGN}"
_EOT       = "\N{LESS-THAN SIGN}|eot_id|\N{GREATER-THAN SIGN}"
_IM_START  = "\N{LESS-THAN SIGN}|im_start|\N{GREATER-THAN SIGN}"
_IM_END    = "\N{LESS-THAN SIGN}|im_end|\N{GREATER-THAN SIGN}"
_PHI_SYS   = "\N{LESS-THAN SIGN}|system|\N{GREATER-THAN SIGN}"
_PHI_USR   = "\N{LESS-THAN SIGN}|user|\N{GREATER-THAN SIGN}"
_PHI_AST   = "\N{LESS-THAN SIGN}|assistant|\N{GREATER-THAN SIGN}"
_PHI_END   = "\N{LESS-THAN SIGN}|end|\N{GREATER-THAN SIGN}"
_SOT       = "\N{LESS-THAN SIGN}start_of_turn\N{GREATER-THAN SIGN}"
_EOT_GEMMA = "\N{LESS-THAN SIGN}end_of_turn\N{GREATER-THAN SIGN}"
_EOS_S     = "</s>"


def format_prompt(user_input: str, output: str, template: str) -> str:
    """
    FIX 1 - Append EOS/stop token to every training example.

    Without a terminal stop token the model never learns WHEN to stop,
    producing the repetitive output loop seen after Ollama registration.
    
    Note: System instructions are not included; this uses the model's default.
    """
    sys_  = ""  # No system instruction in fine-tuning data
    user_ = user_input.strip()
    out_  = output.strip()
    NL    = "\n"

    if template == "llama3":
        return (
            _BOS
            + _HDR_S + "system" + _HDR_E + NL + NL + sys_ + _EOT
            + _HDR_S + "user"   + _HDR_E + NL + NL + user_ + _EOT
            + _HDR_S + "assistant" + _HDR_E + NL + NL + out_ + _EOT  # EOS
        )
    if template == "llama2":
        return (
            "[INST] <<SYS>>" + NL + sys_ + NL + "<</SYS>>" + NL + NL
            + user_ + " [/INST] " + out_ + " " + _EOS_S  # EOS
        )
    if template == "mistral":
        cu = sys_ + NL + NL + user_ if sys_ else user_
        return "[INST] " + cu + " [/INST]" + out_ + _EOS_S  # EOS
    if template == "phi3":
        return (
            _PHI_SYS + NL + sys_ + _PHI_END + NL
            + _PHI_USR + NL + user_ + _PHI_END + NL
            + _PHI_AST + NL + out_ + _PHI_END  # EOS
        )
    if template == "gemma2":
        cu = sys_ + NL + NL + user_ if sys_ else user_
        return (
            _SOT + "user" + NL + cu + _EOT_GEMMA + NL
            + _SOT + "model" + NL + out_ + _EOT_GEMMA  # EOS
        )
    if template == "tinyllama":
        return (
            _IM_START + "system" + NL + sys_ + _IM_END + NL
            + _IM_START + "user"   + NL + user_ + _IM_END + NL
            + _IM_START + "assistant" + NL + out_ + _IM_END  # EOS
        )
    # Default: ChatML
    return (
        _IM_START + "system" + NL + sys_ + _IM_END + NL
        + _IM_START + "user"   + NL + user_ + _IM_END + NL
        + _IM_START + "assistant" + NL + out_ + _IM_END  # EOS
    )


# ========================================================================
# DATASET LOADING
# ========================================================================

def load_jsonl_as_hf_dataset(path: str, chat_template: str = "chatml"):
    """
    Convert our JSONL format to a supervised fine-tuning dataset.

    Each row must have ``input`` and ``output`` fields.
    The text is formatted using the correct chat template for the target
    model so the token sequences match what the model expects at inference.
    """
    data = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            user  = (obj.get("input")       or "").strip()
            out   = (obj.get("output")      or "").strip()
            if not user or not out:
                continue
            text = format_prompt(user, out, chat_template)
            data.append({"text": text})

    tmp_jsonl = p.parent / (p.stem + "_unsloth_tmp.jsonl")
    with tmp_jsonl.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    from datasets import load_dataset
    dataset = load_dataset("json", data_files=str(tmp_jsonl))["train"]
    return dataset


# ─── GGUF export ──────────────────────────────────────────────────────────────

def save_gguf(model, tokenizer, output_dir: str, quantization: str = "q4_k_m") -> Path:
    """
    Export the model to GGUF using Unsloth's built-in save_pretrained_gguf().
    
    This replaces the manual subprocess calls to llama.cpp with the standard
    Unsloth exporter.
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Save model in HF format first.
    # Unsloth's save_pretrained_gguf reads config.json from the output directory
    # BEFORE running the converter, so HF files must already be there.
    print(f"Saving merged HF model to {output_path} ...")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Step 2: Convert to GGUF
    print(f"Converting to GGUF ({quantization}) ...")
    model.save_pretrained_gguf(
        str(output_path),
        tokenizer,
        quantization_method=quantization,
    )

    # Step 3: Find the output GGUF file
    gguf_files = sorted(output_path.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {output_path} after export.")

    final_gguf = gguf_files[-1]
    print(f"GGUF saved -> {final_gguf}")
    return final_gguf


# ─── Ollama registration ──────────────────────────────────────────────────────

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",        type=str,  required=True)
    parser.add_argument("--train_path",        type=str,  required=True)
    parser.add_argument("--output_dir",        type=str,  required=True)
    parser.add_argument("--max_steps",         type=int,  default=60)
    parser.add_argument("--ollama_model_name", type=str,  help="Name for Ollama")
    parser.add_argument("--gguf_quantization", type=str,  default="q4_k_m")
    parser.add_argument("--skip_gguf",         action="store_true")
    parser.add_argument("--ollama_base_url",   type=str,  default="http://localhost:11434")
    parser.add_argument("--epochs",            type=int,  default=1)
    args = parser.parse_args()

    # Lazy imports — only load when training is actually invoked
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import torch

    chat_template = detect_template(args.base_model)
    print(f"[INFO] Detected chat template '{chat_template}' for model '{args.base_model}'.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=1024,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # FIX 3: lora_alpha=32 (2 * r=16) for better generalisation
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,   # was 16 — raised to 2*r for better regularisation
        lora_dropout=0.05,
        bias="none",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    if args.max_steps > 0:
        if not args.train_path or args.train_path == ".":
            raise ValueError("Training requires a valid --train_path")

        train_dataset = load_jsonl_as_hf_dataset(args.train_path, chat_template=chat_template)
        dataset_size = len(train_dataset)

        if dataset_size < 200:
            args.max_steps = 80
        elif dataset_size < 500:
            args.max_steps = 120
        else:
            args.max_steps = 200

        training_args = TrainingArguments(
            output_dir=args.output_dir,

            per_device_train_batch_size=2,   # ↑ from 1
            gradient_accumulation_steps=8,   # ↓ from 16

            learning_rate=1e-4,              # 🔥 LOWER (critical)

            max_steps=args.max_steps,
            logging_steps=10,

            save_strategy="steps",
            save_steps=args.max_steps,

            fp16=True,
            bf16=False,

            optim="adamw_8bit",

            lr_scheduler_type="cosine",      # 🔥 smoother training
            warmup_ratio=0.1,                # 🔥 prevents instability
            max_grad_norm=1.0,
            report_to=[],
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            args=training_args,
            dataset_text_field="text",
        )

        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        model.config.save_pretrained(args.output_dir)

        meta = {
            "base_model":    args.base_model,
            "chat_template": chat_template,
            "lora_r":        16,
            "lora_alpha":    32,
            "max_seq_length": 1024,
            "dtype":         "float16",
            "trained_on":    str(Path(args.train_path).name),
            "max_steps":     args.max_steps,
        }
        (Path(args.output_dir) / "training_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Training complete. Adapter saved to {args.output_dir}")

    else:
        print(f"  Skipping training (max_steps=0). Loading existing adapter from {args.output_dir}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.output_dir)
        print(f"Adapter loaded from {args.output_dir}")

    # ── GGUF export ───────────────────────────────────────────────────────────
    gguf_path = None
    if not args.skip_gguf:
        try:
            gguf_path = save_gguf(
                model, tokenizer,
                args.output_dir,
                quantization=args.gguf_quantization,
            )
        except Exception as e:
            print(f"[WARNING] GGUF export failed: {e}")
            print(f"[INFO] HuggingFace adapter is still available at {args.output_dir}")

    # ── Ollama registration ───────────────────────────────────────────────────
    if gguf_path and args.ollama_model_name:
        result = register_with_ollama(
            gguf_path=gguf_path,
            ollama_model_name=args.ollama_model_name,
            chat_template=chat_template,
            ollama_base_url=args.ollama_base_url,
        )
        summary_path = Path(args.output_dir) / "ollama_registration.json"
        summary_path.write_text(json.dumps(result, indent=2))
        print(f"\nRegistration summary -> {summary_path}")

    elif not args.skip_gguf and not args.ollama_model_name:
        print(
            "\n --ollama_model_name not provided — skipping Ollama registration.\n"
            "    To register manually after this script finishes:\n"
            "    echo 'FROM <path>.gguf' > Modelfile && ollama create <name> -f Modelfile"
        )

    print("\nFine-tuning pipeline complete.")


if __name__ == "__main__":
    main()
