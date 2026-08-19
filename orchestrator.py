"""
Autonomous nanoGPT Research Orchestrator.
Manages continuous Kaggle GPU experiments, Git synchronization, and recursive mutations
across the full search space:
- Positional representations: RoPE, ALiBi, NoPE
- Normalization: RMSNorm, QK-Norm, Cosine Attention
- Feed-Forward / Activation: SwiGLU, GeGLU (d_ffn = 8/3 * d_model)
- Attention Mechanics: Grouped-Query Attention (GQA), Multi-Query Attention (MQA), Sliding-Window Local Attention
- Optimizers & Schedules: Muon for 2D weights + AdamW for embeddings; Warmup-Stable-Decay (WSD) schedules
"""
import os
import re
import sys
import time
import json
import shutil
import datetime
import subprocess

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_LOG_FILE = os.path.join(WORKSPACE_DIR, "status.log")
RESEARCH_LOG_FILE = os.path.join(WORKSPACE_DIR, "research_log.md")
KAGGLE_OUTPUT_DIR = os.path.join(WORKSPACE_DIR, ".kaggle_output")
KERNEL_SLUG = "r1two3four5/nanogpt-research-autoloop"
KAGGLE_EXE = r"C:\nanoGPT_env\Scripts\kaggle.exe" if os.path.exists(r"C:\nanoGPT_env\Scripts\kaggle.exe") else "kaggle"
GIT_EXE = r"C:\MinGit\cmd\git.exe" if os.path.exists(r"C:\MinGit\cmd\git.exe") else "git"

os.environ["KAGGLE_API_TOKEN"] = "<KAGGLE_API_TOKEN>"

def update_status(stage, iteration, best_metric, current_metric, details=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""# nanoGPT Research Loop - Real-Time Status

**Last Updated:** {timestamp}
**Current Stage:** {stage}
**Iteration:** {iteration}
**Best Metric (Val Loss):** {f'{best_metric:.4f}' if best_metric is not None else 'None'}
**Latest Run Metric:** {f'{current_metric:.4f}' if current_metric is not None else 'N/A'}
**Details:** {details}

---
*Autonomous loop running 24/7 on remote Kaggle T4 GPUs.*
"""
    with open(STATUS_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[{timestamp}] [{stage}] Iteration: {iteration} | Best: {best_metric} | Latest: {current_metric} | {details}")

def run_cmd(cmd, cwd=WORKSPACE_DIR, check=True):
    res = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"Command failed: {cmd}\nSTDERR: {res.stderr}\nSTDOUT: {res.stdout}")
    return res

def push_kaggle_kernel():
    res = run_cmd(f'"{KAGGLE_EXE}" kernels push -p "{WORKSPACE_DIR}"', check=False)
    print(f"Kernel push response:\n{res.stdout}\n{res.stderr}")
    return "successfully pushed" in (res.stdout + res.stderr).lower()

def poll_kernel_completion(timeout_seconds=600, poll_interval=20):
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        res = run_cmd(f'"{KAGGLE_EXE}" kernels status {KERNEL_SLUG}', check=False)
        output = (res.stdout + " " + res.stderr).lower()
        if "complete" in output:
            print(f"Kernel {KERNEL_SLUG} completed successfully!")
            return True, "complete"
        elif "kernelworkerstatus.error" in output or '"error"' in output:
            print(f"Kernel {KERNEL_SLUG} failed with error.")
            return False, "error"
        else:
            elapsed = int(time.time() - start_time)
            print(f"Status: {output.strip()} (elapsed: {elapsed}s)...")
            time.sleep(poll_interval)
    return False, "timeout"

def fetch_kernel_metric():
    if os.path.exists(KAGGLE_OUTPUT_DIR):
        shutil.rmtree(KAGGLE_OUTPUT_DIR, ignore_errors=True)
    os.makedirs(KAGGLE_OUTPUT_DIR, exist_ok=True)
    
    run_cmd(f'"{KAGGLE_EXE}" kernels output {KERNEL_SLUG} -p "{KAGGLE_OUTPUT_DIR}" --force', check=False)
    
    val_metric = None
    all_logs = ""
    for root, _, files in os.walk(KAGGLE_OUTPUT_DIR):
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as lf:
                    content = lf.read()
                    all_logs += f"\n--- {f} ---\n" + content
                    matches = re.findall(r"VAL_METRIC:\s*([0-9\.]+)", content)
                    if matches:
                        val_metric = float(matches[-1])
            except Exception as e:
                print(f"Error reading log file {path}: {e}")
                
    return val_metric, all_logs

def log_experiment_result(iteration, name, metric, delta, status, details=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metric_str = f"{metric:.4f}" if metric is not None else "ERROR"
    delta_str = f"{delta:+.4f}" if delta is not None else "-"
    
    entry = f"| {iteration} | {name} | {metric_str} | {delta_str} | {status} | {timestamp} |\n"
    
    with open(RESEARCH_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
        if details:
            clean_details = details[-2000:] if len(details) > 2000 else details
            f.write(f"\n<details><summary>Run {iteration} Logs ({name})</summary>\n\n```\n{clean_details}\n```\n</details>\n\n")

def git_commit_and_push(iteration, name, metric):
    run_cmd(f'"{GIT_EXE}" add .')
    commit_msg = f"Experiment {iteration} [{name}] - Val Loss: {metric:.4f}"
    run_cmd(f'"{GIT_EXE}" commit -m "{commit_msg}"', check=False)
    push_res = run_cmd(f'"{GIT_EXE}" push origin nanogpt-research', check=False)
    print(f"Git push result: {push_res.stdout}\n{push_res.stderr}")

def git_revert():
    run_cmd(f'"{GIT_EXE}" reset --hard HEAD')
    run_cmd(f'"{GIT_EXE}" clean -fd')

# ==============================================================================
# LLM / Gemini API Mutation Generator
# ==============================================================================
def generate_llm_mutation(current_code, history_text, best_metric):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, None
        
    try:
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        prompt = f"""You are an autonomous AI ML researcher optimizing nanoGPT in PyTorch under a 5-minute training budget on GPUs.
Current Best Validation Metric (VAL_METRIC): {best_metric:.4f}
Recent Experiment History:
{history_text}

Current train.py code:
```python
{current_code}
```

Instructions:
1. Propose ONE high-impact architectural or algorithmic mutation (e.g. attention variants, normalization, gating, optimizer, schedules, hyperparameter scaling).
2. Return ONLY the complete, executable, modified train.py script within ```python ... ``` fences.
3. Ensure the script preserves the final print: VAL_METRIC: <float>.
"""
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8192}
        }
        res = requests.post(url, json=payload, timeout=60)
        if res.status_code == 200:
            data = res.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                mutated_code = match.group(1).strip()
                return "Gemini-Generated Mutation", mutated_code
    except Exception as e:
        print(f"LLM Mutation generation error: {e}")
        
    return None, None

# ==============================================================================
# Comprehensive Research Mutation Search Space
# ==============================================================================
MUTATIONS = [
    # 1. Normalization: RMSNorm
    {
        "name": "RMSNorm Normalization",
        "description": "Replace LayerNorm with RMSNorm without learnable bias.",
        "apply": lambda code: code.replace(
            "class LayerNorm(nn.Module):\n    \"\"\"LayerNorm with optional bias.\"\"\"\n    def __init__(self, ndim, bias=False):\n        super().__init__()\n        self.weight = nn.Parameter(torch.ones(ndim))\n        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None\n\n    def forward(self, input):\n        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)",
            "class LayerNorm(nn.Module):\n    \"\"\"RMSNorm (Root Mean Square Layer Normalization).\"\"\"\n    def __init__(self, ndim, bias=False):\n        super().__init__()\n        self.weight = nn.Parameter(torch.ones(ndim))\n\n    def forward(self, x):\n        variance = x.pow(2).mean(-1, keepdim=True)\n        x = x * torch.rsqrt(variance + 1e-5)\n        return self.weight * x"
        )
    },
    # 2. Feed-Forward / Activation: SwiGLU with 8/3 hidden expansion
    {
        "name": "SwiGLU MLP (8/3 d_model expansion)",
        "description": "Upgrade MLP block to SwiGLU gating with d_ffn = 8/3 * d_model.",
        "apply": lambda code: code.replace(
            "class MLP(nn.Module):\n    def __init__(self, config: GPTConfig):\n        super().__init__()\n        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)\n        self.gelu = nn.GELU()\n        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)\n        self.dropout = nn.Dropout(config.dropout)\n\n    def forward(self, x):\n        x = self.c_fc(x)\n        x = self.gelu(x)\n        x = self.c_proj(x)\n        x = self.dropout(x)\n        return x",
            "class MLP(nn.Module):\n    def __init__(self, config: GPTConfig):\n        super().__init__()\n        hidden_dim = int(8 * config.n_embd / 3)\n        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)\n        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)\n        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)\n        self.dropout = nn.Dropout(config.dropout)\n\n    def forward(self, x):\n        x = F.silu(self.w1(x)) * self.w2(x)\n        x = self.c_proj(x)\n        x = self.dropout(x)\n        return x"
        )
    },
    # 3. Feed-Forward / Activation: GeGLU with 8/3 hidden expansion
    {
        "name": "GeGLU MLP (8/3 d_model expansion)",
        "description": "Upgrade MLP block to GELU-Gated Linear Units with d_ffn = 8/3 * d_model.",
        "apply": lambda code: code.replace(
            "class MLP(nn.Module):\n    def __init__(self, config: GPTConfig):\n        super().__init__()\n        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)\n        self.gelu = nn.GELU()\n        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)\n        self.dropout = nn.Dropout(config.dropout)\n\n    def forward(self, x):\n        x = self.c_fc(x)\n        x = self.gelu(x)\n        x = self.c_proj(x)\n        x = self.dropout(x)\n        return x",
            "class MLP(nn.Module):\n    def __init__(self, config: GPTConfig):\n        super().__init__()\n        hidden_dim = int(8 * config.n_embd / 3)\n        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)\n        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)\n        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)\n        self.dropout = nn.Dropout(config.dropout)\n\n    def forward(self, x):\n        x = F.gelu(self.w1(x)) * self.w2(x)\n        x = self.c_proj(x)\n        x = self.dropout(x)\n        return x"
        )
    },
    # 4. Attention Mechanics: Grouped-Query Attention (GQA - 2 KV heads, 4 Q heads)
    {
        "name": "Grouped-Query Attention (GQA)",
        "description": "Grouped Query Attention with 2 Key-Value heads paired with 4 Query heads.",
        "apply": lambda code: code.replace(
            "self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)",
            """self.n_kv_heads = 2
        self.n_rep = config.n_head // self.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * self.n_kv_heads * self.head_dim, bias=config.bias)"""
        ).replace(
            "q, k, v = self.c_attn(x).split(self.n_embd, dim=2)\n        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)\n        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)\n        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)",
            """q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, self.n_kv_heads, 2 * self.head_dim)
        k, v = kv.split(self.head_dim, dim=-1)
        k = k.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)
        v = v.transpose(1, 2).repeat_interleave(self.n_rep, dim=1)"""
        )
    },
    # 5. Attention Mechanics: Multi-Query Attention (MQA - 1 KV head)
    {
        "name": "Multi-Query Attention (MQA)",
        "description": "Share a single key and value head across all query heads for memory efficiency.",
        "apply": lambda code: code.replace(
            "self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)",
            """head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.kv_proj = nn.Linear(config.n_embd, 2 * head_dim, bias=config.bias)"""
        ).replace(
            "q, k, v = self.c_attn(x).split(self.n_embd, dim=2)\n        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)\n        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)\n        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)",
            """head_dim = C // self.n_head
        q = self.q_proj(x).view(B, T, self.n_head, head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).split(head_dim, dim=2)
        k = k.unsqueeze(1).expand(B, self.n_head, T, head_dim)
        v = v.unsqueeze(1).expand(B, self.n_head, T, head_dim)"""
        )
    },
    # 6. Attention Mechanics: Sliding-Window Local Attention
    {
        "name": "Sliding-Window Local Attention (Window=64)",
        "description": "Restricts attention computation to a local causal window band (window=64).",
        "apply": lambda code: code.replace(
            "att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))",
            """window_size = 64
        row_idx = torch.arange(T, device=x.device)[:, None]
        col_idx = torch.arange(T, device=x.device)[None, :]
        window_mask = (row_idx >= col_idx) & ((row_idx - col_idx) < window_size)
        att = att.masked_fill(~window_mask.view(1, 1, T, T), float('-inf'))"""
        )
    },
    # 7. Normalization: Cosine Attention with Learnable Temperature
    {
        "name": "Cosine Attention (Unit-Norm Q/K with Temperature)",
        "description": "Normalize Query and Key to unit sphere and compute scaled cosine similarity.",
        "apply": lambda code: code.replace(
            "class CausalSelfAttention(nn.Module):",
            """class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.temperature = nn.Parameter(torch.ones(1, config.n_head, 1, 1) * 0.1)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))"""
        ).replace(
            "att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))",
            "q_norm = F.normalize(q, p=2, dim=-1)\n        k_norm = F.normalize(k, p=2, dim=-1)\n        att = (q_norm @ k_norm.transpose(-2, -1)) / self.temperature.clamp(min=1e-3)"
        )
    },
    # 8. Normalization: QK-Norm
    {
        "name": "QK-Normalization",
        "description": "Apply LayerNorm to Query and Key representations before dot-product attention.",
        "apply": lambda code: code.replace(
            "self.attn_dropout = nn.Dropout(config.dropout)",
            "self.attn_dropout = nn.Dropout(config.dropout)\n        self.q_norm = nn.LayerNorm(config.n_embd // config.n_head)\n        self.k_norm = nn.LayerNorm(config.n_embd // config.n_head)"
        ).replace(
            "att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))",
            "q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)\n        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)\n        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))"
        )
    },
    # 9. Positional Representation: RoPE (Rotary Position Embeddings)
    {
        "name": "Rotary Positional Embeddings (RoPE)",
        "description": "Remove absolute positional embedding wpe and apply RoPE rotary embeddings to Q and K.",
        "apply": lambda code: code.replace(
            "class CausalSelfAttention(nn.Module):",
            """def apply_rotary_emb(x, head_dim):
    B, T, nh, hs = x.shape
    device = x.device
    dim = hs
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(T, device=device).float()
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_emb = emb.cos().view(1, T, 1, dim)
    sin_emb = emb.sin().view(1, T, 1, dim)
    x1, x2 = x[..., :dim // 2], x[..., dim // 2:]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos_emb) + (x_rot * sin_emb)

class CausalSelfAttention(nn.Module):"""
        ).replace(
            "k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)\n        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)\n        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)",
            "k = k.view(B, T, self.n_head, C // self.n_head)\n        q = q.view(B, T, self.n_head, C // self.n_head)\n        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)\n        q = apply_rotary_emb(q, C // self.n_head).transpose(1, 2)\n        k = apply_rotary_emb(k, C // self.n_head).transpose(1, 2)"
        ).replace(
            "tok_emb = self.transformer.wte(idx)\n        pos_emb = self.transformer.wpe(pos)\n        x = self.transformer.drop(tok_emb + pos_emb)",
            "tok_emb = self.transformer.wte(idx)\n        x = self.transformer.drop(tok_emb)"
        )
    },
    # 10. Optimizer: Muon for 2D weights + AdamW for 1D/embeddings
    {
        "name": "Muon Optimizer (2D Matrix Orthogonalization) + AdamW",
        "description": "Keller Jordan Muon optimizer using Newton-Schulz iterations on 2D weights paired with AdamW for embeddings.",
        "apply": lambda code: code.replace(
            "optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=1e-1)",
            """def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750,  2.0315)
        X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
        X = X / (X.norm() + eps)
        if G.size(0) > G.size(1):
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(0) > G.size(1):
            X = X.T
        return X.to(dtype=G.dtype)

    class Muon(torch.optim.Optimizer):
        def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
            defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
            super().__init__(params, defaults)
        @torch.no_grad()
        def step(self):
            for group in self.param_groups:
                lr = group['lr']
                momentum = group['momentum']
                for p in group['params']:
                    if p.grad is None: continue
                    g = p.grad
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)
                    else:
                        g = buf
                    u = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    p.data.add_(u, alpha=-lr)

    # Split parameters: 2D internal matrices -> Muon, embeddings & 1D vectors -> AdamW
    muon_params = [p for n, p in model.named_parameters() if p.ndim == 2 and 'wte' not in n and 'wpe' not in n and 'lm_head' not in n]
    adamw_params = [p for n, p in model.named_parameters() if p.ndim < 2 or 'wte' in n or 'wpe' in n or 'lm_head' in n]
    optimizer = torch.optim.AdamW(adamw_params, lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=1e-1)
    muon_optimizer = Muon(muon_params, lr=0.02)"""
        ).replace(
            "scaler.step(optimizer)",
            "scaler.step(optimizer)\n            muon_optimizer.step()"
        ).replace(
            "optimizer.step()",
            "optimizer.step()\n            muon_optimizer.step()"
        )
    },
    # 11. Schedule: Warmup-Stable-Decay (WSD) Scheduler
    {
        "name": "Warmup-Stable-Decay (WSD) LR Schedule",
        "description": "Warmup (5%), Stable Flat LR (70%), Sharp 1-sqrt Decay (25%).",
        "apply": lambda code: code.replace(
            "def get_lr(it, config: GPTConfig):\n    if it < config.warmup_iters:\n        return config.learning_rate * it / config.warmup_iters\n    if it > config.lr_decay_iters:\n        return config.min_lr\n    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)\n    assert 0 <= decay_ratio <= 1\n    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))\n    return config.min_lr + coeff * (config.learning_rate - config.min_lr)",
            """def get_lr(it, config: GPTConfig):
    warmup_steps = int(0.05 * config.max_iters)
    stable_steps = int(0.75 * config.max_iters)
    decay_steps = config.max_iters - stable_steps
    if it < warmup_steps:
        return config.learning_rate * (it + 1) / warmup_steps
    elif it < stable_steps:
        return config.learning_rate
    else:
        progress = (it - stable_steps) / max(1, decay_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)"""
        )
    },
    # 12. Positional Representation: ALiBi
    {
        "name": "ALiBi (Attention with Linear Biases)",
        "description": "Inject distance-based geometric slope bias directly into attention scores.",
        "apply": lambda code: code.replace(
            "att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))",
            """# ALiBi linear bias construction
        q_len, k_len = q.size(2), k.size(2)
        diff = torch.arange(k_len, device=x.device)[None, :] - torch.arange(q_len, device=x.device)[:, None]
        alibi_bias = diff.clamp(max=0).float() # shape (T, T)
        m = torch.tensor([2 ** (-8.0 * (i + 1) / self.n_head) for i in range(self.n_head)], device=x.device).view(1, self.n_head, 1, 1)
        alibi = m * alibi_bias.view(1, 1, q_len, k_len)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) + alibi"""
        ).replace(
            "tok_emb = self.transformer.wte(idx)\n        pos_emb = self.transformer.wpe(pos)\n        x = self.transformer.drop(tok_emb + pos_emb)",
            "tok_emb = self.transformer.wte(idx)\n        x = self.transformer.drop(tok_emb)"
        )
    }
]

def run_autonomous_loop():
    print("=" * 70)
    print("      Starting Autonomous nanoGPT Research Orchestrator")
    print("=" * 70)
    
    best_metric = None
    iteration = 0
    
    # -------------------------------------------------------------
    # Step 1: Initial Baseline Evaluation
    # -------------------------------------------------------------
    while best_metric is None:
        update_status("BASELINE EVALUATION", iteration, best_metric, None, "Pushing baseline kernel to Kaggle...")
        if not push_kaggle_kernel():
            print("Failed to push baseline kernel. Retrying in 15s...")
            time.sleep(15)
            continue
            
        completed, status = poll_kernel_completion(timeout_seconds=600, poll_interval=20)
        val_metric, logs = fetch_kernel_metric()
        
        if val_metric is not None:
            best_metric = val_metric
            update_status("BASELINE COMPLETED", iteration, best_metric, val_metric, f"Baseline established with VAL_METRIC={best_metric:.4f}")
            log_experiment_result(iteration, "Standard Baseline nanoGPT", best_metric, 0.0, "SUCCESS (BASELINE)", logs)
            git_commit_and_push(iteration, "Baseline nanoGPT", best_metric)
            break
        else:
            print(f"Baseline run did not yield VAL_METRIC. Logs:\n{logs[:1000]}")
            update_status("BASELINE RETRY", iteration, None, None, "Baseline failed to yield metric; retrying in 20s...")
            time.sleep(20)

    # -------------------------------------------------------------
    # Step 2: Recursive Mutation Loop
    # -------------------------------------------------------------
    mutation_idx = 0
    while True:
        iteration += 1
        train_file = os.path.join(WORKSPACE_DIR, "train.py")
        with open(train_file, "r", encoding="utf-8") as f:
            code = f.read()

        # Read last 5 lines of research log for history context
        history_summary = ""
        if os.path.exists(RESEARCH_LOG_FILE):
            with open(RESEARCH_LOG_FILE, "r", encoding="utf-8") as rf:
                history_summary = "".join(rf.readlines()[-15:])

        # 1. Attempt LLM / Gemini Mutation
        llm_name, llm_code = generate_llm_mutation(code, history_summary, best_metric)
        if llm_code:
            mutation_name = llm_name
            mutated_code = llm_code
            update_status("MUTATING CODE (LLM)", iteration, best_metric, None, f"Applied Gemini LLM proposed mutation.")
            with open(train_file, "w", encoding="utf-8") as f:
                f.write(mutated_code)
        else:
            # 2. Structured Mutation Search Space
            mutation = MUTATIONS[mutation_idx % len(MUTATIONS)]
            mutation_name = mutation["name"]
            mutation_desc = mutation["description"]
            mutation_idx += 1
            
            update_status("MUTATING CODE", iteration, best_metric, None, f"Applying mutation: {mutation_name} - {mutation_desc}")
            try:
                mutated_code = mutation["apply"](code)
                if mutated_code == code:
                    print(f"Mutation {mutation_name} had no textual match; skipping.")
                    continue
                with open(train_file, "w", encoding="utf-8") as f:
                    f.write(mutated_code)
            except Exception as e:
                print(f"Error applying mutation {mutation_name}: {e}")
                git_revert()
                continue
            
        update_status("RUNNING ON KAGGLE", iteration, best_metric, None, f"Running experiment on Kaggle: {mutation_name}")
        pushed = push_kaggle_kernel()
        while not pushed:
            print(f"Kaggle slot busy. Retrying push in 15s...")
            time.sleep(15)
            pushed = push_kaggle_kernel()
            
        completed, status = poll_kernel_completion(timeout_seconds=600, poll_interval=20)
        val_metric, logs = fetch_kernel_metric()
        
        if val_metric is not None:
            delta = val_metric - best_metric
            if val_metric < best_metric: # Lower loss is better
                improvement = best_metric - val_metric
                best_metric = val_metric
                update_status("EXPERIMENT SUCCESS - IMPROVED", iteration, best_metric, val_metric, f"Improvement by {improvement:.4f}! Committing & pushing to GitHub.")
                log_experiment_result(iteration, mutation_name, val_metric, delta, f"IMPROVED (-{improvement:.4f})", logs)
                git_commit_and_push(iteration, mutation_name, val_metric)
            else:
                update_status("EXPERIMENT DEGRADED", iteration, best_metric, val_metric, f"Metric degraded by {delta:+.4f}. Reverting mutation.")
                log_experiment_result(iteration, mutation_name, val_metric, delta, "DEGRADED (REVERTED)", logs)
                git_revert()
        else:
            update_status("EXPERIMENT FAILED - NO METRIC", iteration, best_metric, None, "Log parsing failed. Reverting mutation.")
            log_experiment_result(iteration, mutation_name, None, None, "FAILED (PARSE/EXEC ERROR)", logs)
            git_revert()
            
        print(f"Iteration {iteration} finished. Starting next mutation cycle in 5s...")
        time.sleep(5)

if __name__ == "__main__":
    run_autonomous_loop()
