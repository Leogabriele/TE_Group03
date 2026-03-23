# 🚀 How to Start — Developer Setup Guide

This guide walks you through setting up the **TE_Group03 Safety Red-Teaming Platform** from scratch on a Windows machine with an NVIDIA GPU.

---

## 📋 Prerequisites

Before you begin, make sure the following are installed on your system:

| Requirement | Version | Notes |
|---|---|---|
| Python | **3.11.x** | Must be exactly 3.11 — unsloth is not compatible with 3.12+ |
| NVIDIA GPU | Any CUDA-capable | Required for Unsloth fine-tuning |
| CUDA Toolkit | 12.8 or 12.9 | Used by PyTorch |
| Ollama | Latest | Download from [ollama.ai](https://ollama.ai) |
| MongoDB | Latest | Running locally on default port 27017 |
| Git | Any | For cloning the repository |

> ⚠️ **Python version matters.** Unsloth strictly requires Python 3.11. Using 3.12 or higher will cause installation failures.

---

## 🛠️ Step-by-Step Setup

### Step 1 — Create a Virtual Environment

Open a terminal in the project root directory and run:

```bash
python3.11 -m venv venv
```

This creates an isolated Python 3.11 environment in the `venv/` folder. All packages will be installed here, keeping your global Python installation clean.

> 💡 **Windows users:** If `python3.11` is not recognised, try `py -3.11 -m venv venv` instead.

---

### Step 2 — Activate the Virtual Environment

**Windows (Command Prompt / PowerShell):**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt, confirming the environment is active.

> ⚠️ You must activate the venv **every time** you open a new terminal before running any project commands.

---

### Step 3 — Install Python Dependencies

With the venv active, install all required packages:

```bash
pip install -r requirements.txt
```

This installs the full dependency stack including FastAPI, Streamlit, LangChain, MongoDB drivers, and all supporting libraries.

**Expected time:** 3–10 minutes depending on your internet speed.

> 💡 If you see dependency conflict warnings (not errors), these are usually safe to ignore. Errors that say `ERROR: ResolutionImpossible` need to be fixed — check the `requirements.txt` file.

---

### Step 4 — Install Unsloth (Fine-Tuning Engine)

Unsloth must be installed separately after the base requirements:

```bash
pip install "unsloth[torch]"
```

This installs Unsloth with its PyTorch integration for GPU-accelerated LoRA fine-tuning.

> ⚠️ Unsloth is **not** pinned in `requirements.txt` because it releases updates very frequently. Always install the latest version.

---

### Step 5 — Install PyTorch with CUDA Support

Install PyTorch built for your CUDA version:

**For CUDA 12.8 (recommended — widely tested with Unsloth):**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**For CUDA 12.9:**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

**Verify GPU is detected correctly after install:**
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
2.x.x+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX XXXX
```

> ❌ If `CUDA available: False`, PyTorch cannot see your GPU. Common fixes:
> - Make sure your NVIDIA drivers are up to date
> - Reinstall PyTorch using the exact command above
> - Check `nvidia-smi` runs successfully in your terminal

---
### Step 6 — GGUF Export Setup (for Fine-Tuned Model Deployment)
 
After fine-tuning a model, the platform can export it to GGUF format so it can be registered and run via Ollama. This requires **llama.cpp** — but you do **not** need to compile it from source.
 
#### Option A — Automatic Setup (Recommended)
 
The platform will **automatically download and set up llama.cpp** the first time you click **Generate GGUF** in the Retraining page. No manual steps required.
 
What happens automatically:
1. The script checks if `C:\llama.cpp\` exists
2. If not, it fetches the latest pre-built Windows binary from the [llama.cpp GitHub releases](https://github.com/ggerganov/llama.cpp/releases/latest)
3. It downloads the correct CUDA build for your system
4. It extracts the binary to `C:\llama.cpp\`
5. It downloads the `convert_hf_to_gguf.py` conversion script
 
**Expected download size:** ~50–100 MB  
**Expected time:** 1–3 minutes depending on internet speed
 
> 💡 This only happens **once**. Subsequent GGUF exports use the cached binary and are instant.
 
#### Option B — Manual Setup
 
If automatic setup fails (e.g. no internet access, firewall restrictions), follow these steps manually:
 
**Step 6a — Download the pre-built binary:**
 
Go to: [https://github.com/ggerganov/llama.cpp/releases/latest](https://github.com/ggerganov/llama.cpp/releases/latest)
 
Download the file matching your system:
 
| GPU | File to download |
|---|---|
| NVIDIA GPU (CUDA 12.x) | `llama-*-bin-win-cuda-cu12.x.x-x64.zip` |
| CPU only (no GPU) | `llama-*-bin-win-x64.zip` |
 
**Step 6b — Extract to the correct location:**
 
1. Create the folder `C:\llama.cpp\`
2. Extract the **contents** of the zip directly into `C:\llama.cpp\`
 
After extraction, verify these files exist:
```
C:\llama.cpp\
    llama-quantize.exe       ← required
    llama-cli.exe
    llama.dll
    ggml.dll
    ... other files
```
 
**Step 6c — Download the conversion script:**
 
The zip does not always include `convert_hf_to_gguf.py`. Download it separately:
 
```bash
curl -o C:\llama.cpp\convert_hf_to_gguf.py ^
  https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py
```
 
Or open the URL in your browser and save the file to `C:\llama.cpp\convert_hf_to_gguf.py`.
 
**Step 6d — Verify setup:**
 
```bash
dir C:\llama.cpp\convert_hf_to_gguf.py
dir C:\llama.cpp\llama-quantize.exe
```
 
Both files should be listed. You are now ready to export GGUF from the Retraining page.
 
> ⚠️ **Do not use winget or cmake.** Earlier versions of Unsloth tried to compile llama.cpp from source using winget and cmake — this fails on most Windows machines without admin rights and Visual Studio. The pre-built binary approach above bypasses this entirely.
 
---
### Step 6 — Start the Application

Make sure Ollama and MongoDB are running, then launch the Streamlit frontend:

```bash
streamlit run frontend/app.py
```

The app will open automatically in your browser at **http://localhost:8501**.

---

## 🔧 Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB_NAME=te_group03

# Groq API (for cloud LLM evaluation)
GROQ_API_KEY=your_groq_api_key_here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
```

> 🔑 Get a free Groq API key at [console.groq.com](https://console.groq.com). The platform uses Groq for fast LLM-based evaluation of attack responses.

---

## 🦙 Setting Up Ollama Models

With Ollama running, pull at least one local model for red-team evaluation:

```bash
# Lightweight (recommended for testing)
ollama pull llama3.2:1b

# Better quality
ollama pull llama3.2:3b
ollama pull phi3:latest
```

Verify Ollama is running and models are available:
```bash
ollama list
```

---

## 📁 Project Structure Overview

```
TE_Group03/
├── frontend/
│   └── app.py                      # Main Streamlit UI entry point
├── backend/
│   └── app/
│       ├── core/
│       │   ├── orchestrator.py     # Attack orchestration engine
│       │   ├── retraining.py       # Dataset builder + model saving
│       │   └── llm_clients.py      # Groq / Ollama API clients
│       ├── agents/
│       │   └── strategies.py       # Jailbreak attack strategies
│       ├── models/
│       │   └── database.py         # MongoDB connection manager
│       └── utils/
│           └── local_model_manager.py
├── scripts/
│   └── unsloth_finetune.py         # LoRA fine-tuning + GGUF export script
├── data/
│   └── benign_data.csv             # Benign prompt/response pairs for training
├── models/                         # Fine-tuned model output directory
├── requirements.txt
├── .env
└── How_to_start.md
```

---

## ⚡ Quick Start Checklist

Run through this checklist before your first launch:

- [ ] Python 3.11 installed (`python --version` shows `3.11.x`)
- [ ] NVIDIA GPU visible (`nvidia-smi` runs without error)
- [ ] MongoDB running (`mongosh` connects successfully)
- [ ] Ollama running (`ollama list` shows installed models)
- [ ] `.env` file created with correct values
- [ ] Virtual environment activated (`(venv)` in terminal prompt)
- [ ] All 5 installation steps completed
- [ ] `torch.cuda.is_available()` returns `True`

---

## 🐛 Common Issues & Fixes

### `ModuleNotFoundError: No module named 'unsloth'`
The venv is not activated, or unsloth was not installed. Run:
```bash
venv\Scripts\activate
pip install "unsloth[torch]"
```

### `NotImplementedError: Unsloth cannot find any torch accelerator`
Unsloth requires a GPU and cannot run on CPU. Check:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, reinstall PyTorch with CUDA (Step 5 above).

### `pymongo.errors.InvalidOperation: Cannot use MongoClient after close`
MongoDB client is being reused after it was closed. This is a race condition on first run — the second run usually succeeds. Ensure you are using the latest `database.py` which resets the client to `None` after close.

### `groq.BadRequestError: model decommissioned`
The Groq model `gemma2-9b-it` has been retired. Update your model config to use `llama-3.1-8b-instant` instead. Check available models at [console.groq.com/docs/models](https://console.groq.com/docs/models).

### `RuntimeError: Unsloth: No or negligible GPU memory available`
Not enough VRAM for fine-tuning. Try these in order:
```bash
# 1. Stop Ollama to free GPU memory before fine-tuning
ollama stop

# 2. Check available VRAM
python -c "import torch; free,total=torch.cuda.mem_get_info(); print(f'Free: {free/1e9:.1f}GB / Total: {total/1e9:.1f}GB')"
```
If VRAM is below 4 GB free, use a smaller model (e.g. `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`).

### `ERROR: ResolutionImpossible` during `pip install -r requirements.txt`
A package version conflict exists. Check that you are using the latest `requirements.txt` from the repo. Key pinned versions that must not be changed:
- `packaging==24.1`
- `streamlit==1.55.0`
- `langchain-core==1.2.14`
- `trl==0.24.0` (required by unsloth)
- `datasets==3.4.1` (required by unsloth)

### Port 8501 already in use
Another Streamlit instance is running. Kill it or specify a different port:
```bash
streamlit run frontend/app.py --server.port 8502
```

---

## 📞 Getting Help

- Check the terminal output when launching — most errors print clear messages
- MongoDB logs: check your MongoDB installation's log directory
- Ollama logs: run `ollama serve` manually to see verbose output
- For Unsloth issues: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)
- For Groq model availability: [console.groq.com/docs/models](https://console.groq.com/docs/models)