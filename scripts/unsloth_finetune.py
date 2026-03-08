"""
Lightweight Unsloth fine-tuning script.

This script is started from the Streamlit UI and performs a real LoRA
fine-tune on a Hugging Face base model using the JSONL training split
created by the retraining pipeline.

Requirements (install in your venv before running):
    pip install "unsloth[torch]" datasets accelerate

Usage (called programmatically):
    python scripts/unsloth_finetune.py \\
        --base_model unsloth/llama-3-8b-bnb-4bit \\
        --train_path data/retrain_train_*.jsonl \\
        --output_dir models/my-safe-model
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer


def load_jsonl_as_hf_dataset(path: str):
    """
    Convert our JSONL format to a simple supervised fine-tuning dataset.

    We flatten each row into a single text field:
        <instruction>\n\nUser: <input>\n\nAssistant: <output>
    """
    data = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            instr = (obj.get("instruction") or "").strip()
            user = (obj.get("input") or "").strip()
            out = (obj.get("output") or "").strip()
            if not user or not out:
                continue
            if instr:
                text = f"{instr}\n\nUser: {user}\n\nAssistant: {out}"
            else:
                text = f"User: {user}\n\nAssistant: {out}"
            data.append({"text": text})

    # Save to a temporary JSONL for datasets to read
    tmp_jsonl = p.parent / (p.stem + "_unsloth_tmp.jsonl")
    with tmp_jsonl.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dataset = load_dataset("json", data_files=str(tmp_jsonl))["train"]
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    train_dataset = load_jsonl_as_hf_dataset(args.train_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.base_model,
    max_seq_length=1024,
    dtype="bfloat16",
    load_in_4bit=True 
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=1,      # ✅ Down from 4 → 1
    gradient_accumulation_steps=16,     # ✅ Up from 4 → 16 (keeps effective batch size = 16)
    learning_rate=2e-4,
    max_steps=args.max_steps,
    logging_steps=10,
    save_strategy="steps",
    save_steps=args.max_steps,
    bf16=True,
    optim="adamw_8bit",                 # ✅ Change from adamw_torch → adamw_8bit (saves ~1GB)
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
            print(f"⚠️  GGUF export failed: {e}")
            print(f"    HuggingFace adapter is still available at {args.output_dir}")

    # ── Ollama registration ───────────────────────────────────────────────────
    if gguf_path and args.ollama_model_name:
        result = register_with_ollama(
            gguf_path=gguf_path,
            ollama_model_name=args.ollama_model_name,
            ollama_base_url=args.ollama_base_url,
        )
        # Write a JSON summary next to the adapter so the UI can read it
        summary_path = Path(args.output_dir) / "ollama_registration.json"
        summary_path.write_text(json.dumps(result, indent=2))
        print(f"\n📝  Registration summary → {summary_path}")

    elif not args.skip_gguf and not args.ollama_model_name:
        print(
            "\nℹ️  --ollama_model_name not provided — skipping Ollama registration.\n"
            "    To register manually after this script finishes:\n"
            f"    echo 'FROM <path>.gguf' > Modelfile && ollama create <name> -f Modelfile"
        )

    print("\n🎉  Fine-tuning pipeline complete.")



if __name__ == "__main__":
    main()

