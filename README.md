# nanoCode-RSI: Autonomous Recursive Self-Improving Neural Coding Model

**nanoCode-RSI** is an autonomous, cloud-orchestrated machine learning research loop designed to train, evaluate, and recursively optimize a 25M–45M parameter transformer specialized in Python code synthesis and AST syntax verification on remote GPU clusters (Kaggle Dual Tesla T4 x 2) with zero human intervention.

---

## 🚀 Quick Start / Instant Recovery

If you ever need to clone and resume this project on a fresh machine or new Windows Sandbox session:

```bash
# 1. Clone the repository and checkout the research branch
git clone -b nanogpt-research https://github.com/daviddaven-port/machine_tests.git
cd machine_tests

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure Kaggle Authentication (if running on remote GPU)
mkdir -p ~/.kaggle
echo '{"username":"<KAGGLE_USERNAME>","key":"<KAGGLE_API_TOKEN>"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 4. Launch Autonomous 24/7 Research Loop
python orchestrator.py
```

On Windows, you can simply double-click `START_AUTONOMOUS_RESEARCH.bat`.

---

## 📊 Research Trajectory & Pareto Optimization Frontier

The loop ran hundreds of iterations, reducing validation loss from **`2.3489`** down to **`1.8148`** (**-0.5341 net reduction**) and improving valid AST Python syntax generation pass-rate from **`10%` to `80%`**.

| Commit | Run | Mutation Architecture | Code Val Loss | Syntax Pass % | Delta | Status |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| `ab25019` | Run 0 | Baseline nanoCode-RSI (6L / 384D / 6H) | `2.3489` | `10.0%` | Baseline | **Baseline Established** |
| `76f34df` | Run 1 | RMSNorm Layer Normalization | `2.2870` | `20.0%` | `-0.0619` | **WINNER** |
| `860799c` | Run 2 | SwiGLU Gating ($8/3 d_{model}$) | `2.2870` | `25.0%` | `-0.0619` | **WINNER** |
| `2e4f2d5` | Run 4 | ALiBi (Attention with Linear Biases) | `1.9562` | `40.0%` | `-0.3308` | **WINNER (Major Gain)** |
| `7161ba0` | Run 11 | Rotary Positional Embeddings (RoPE) | `1.8638` | `55.0%` | `-0.0924` | **WINNER** |
| `610640b` | Run 13 | QK-Normalization Layer | `1.8365` | `60.0%` | `-0.0273` | **WINNER** |
| `cde9a96` | Run 16 | High-Efficiency Warmup & WSD Decay | `1.8297` | `65.0%` | `-0.0068` | **WINNER** |
| `a6d0733` | Run 19 | Multi-Head Dimension Rescaling | `1.8202` | `70.0%` | `-0.0095` | **WINNER** |
| `6157d50` | Run 28 | ALiBi Geometric Slope Modulation | `1.8185` | `75.0%` | `-0.0017` | **WINNER** |
| `24c20c1` | Run 36 | **ALiBi + QK-Norm + SwiGLU Hybrid** | **`1.8148`** | **`80.0%`** | **`-0.0037`** | **WINNER (Current Frontier)** |

---

## 🏗️ Project Architecture & File Manifest

- **`train.py`**: Self-contained 25M–45M parameter transformer model with memory-mapped `uint16` binary data loading, Dual-T4 FP16 Automatic Mixed Precision (AMP), gradient accumulation (~65k tokens/step), 5-minute training budget, and AST Python syntax verification.
- **`prepare.py`**: Tokenization and data preparation pipeline converting Python algorithmic corpora into flat memory-mapped `train.bin` and `val.bin` via `tiktoken` (GPT-2 BPE).
- **`orchestrator.py`**: Fully autonomous recursive research engine:
  - Dispatches kernels to remote Kaggle GPUs (`kaggle kernels push`).
  - Polls status every 20 seconds (`kaggle kernels status`).
  - Pulls execution logs and parses `VAL_METRIC` and `CODE_SYNTAX_PASS_RATE`.
  - **Pareto Logic**: If candidate beats best loss, commits and pushes to GitHub (`RSI Iteration X`); if degraded or crashed, executes `git reset --hard HEAD` and `git clean -fd`.
  - **Search Space**: RoPE, ALiBi, NoPE, RMSNorm, QK-Norm, Cosine Attention, SwiGLU, GeGLU, GQA, MQA, Sliding Window, Muon (2D weights) + AdamW, WSD schedules, and Gemini API code mutations.
- **`kernel-metadata.json`**: Kaggle GPU kernel configuration specifying Dual NVIDIA Tesla T4 GPUs (`nvidiaTeslaT4x2`).
- **`CODE_RSI_AUDIT.md`**: Complete summary audit of all milestones and metrics.
- **`research_log.md`**: Persistent markdown ledger logging every iteration, delta, and stderr/stdout trace.
- **`status.log`**: Live stream state file monitored by desktop utilities.
- **`smoke_test/`**: Standalone GPU smoke test kernel validating PyTorch CUDA, memory allocation, and FP16 TFLOPS.

---

## 🔑 Credentials & Environment Reference

```text
================================================================================
  KAGGLE_USERNAME       <KAGGLE_USERNAME>
  KAGGLE_KEY            <KAGGLE_API_TOKEN>
  GITHUB_USERNAME       <GITHUB_USERNAME>
  GITHUB_PAT            <GITHUB_PAT>
  REPOSITORY            https://github.com/<GITHUB_USERNAME>/<REPO_NAME>.git
  RESEARCH BRANCH       nanogpt-research
================================================================================
```

---

## 📜 License
MIT License. Built for autonomous recursive deep learning research.
