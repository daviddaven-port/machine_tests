"""
nanoCode-RSI: PyTorch/XLA Training Baseline for Google Cloud TPU VM v3-8 on Kaggle.
- Target Architecture: N_layer=6, d_model=384, N_head=6, T_seq=512.
- Native TPU bfloat16 mixed precision execution.
- Optimized with torch_xla PJRT runtime (xm.xla_device, xm.optimizer_step).
- Formatted stdout metric markers:
    VAL_METRIC: <float>
    CODE_SYNTAX_PASS_RATE: <float>%
"""
import os
import sys
import ast
import math
import time
import urllib.request
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# PyTorch / XLA Initialization for Google Cloud TPU
# ==============================================================================
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

# ==============================================================================
# Model & TPU Training Configuration
# ==============================================================================
@dataclass
class GPTConfig:
    block_size: int = 512              # Sequence length T_seq = 512
    vocab_size: int = 50304            # Padded GPT-2 BPE vocab (aligned for TPU Matrix Units)
    n_layer: int = 6                   # 6 Transformer layers
    n_head: int = 6                    # 6 Query heads (64 dim per head)
    n_embd: int = 384                  # d_model = 384
    dropout: float = 0.0
    bias: bool = False
    batch_size: int = 64               # TPU v3-8 optimized micro-batch size (high HBM throughput)
    gradient_accumulation_steps: int = 4 # Global batch = 64 * 512 * 4 = 131,072 tokens/step
    learning_rate: float = 8e-4
    max_iters: int = 2500
    eval_interval: int = 150
    eval_iters: int = 25
    warmup_iters: int = 60
    lr_decay_iters: int = 2500
    min_lr: float = 8e-5
    max_time_seconds: int = 270        # 4.5-minute execution window per iteration

# ==============================================================================
# Fast Dataset Loader (uint16 flat binary memmap for TPU)
# ==============================================================================
def get_dataset():
    data_dir = "."
    train_bin = os.path.join(data_dir, "train.bin")
    val_bin = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        print("Generating Python code dataset buffers for TPU...")
        code_corpus = """
def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop() if not self.is_empty() else None
    def is_empty(self):
        return len(self.items) == 0

def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""
        code_corpus = code_corpus * 300
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            train_ids = np.array(enc.encode_ordinary(code_corpus[:int(len(code_corpus)*0.9)]), dtype=np.uint16)
            val_ids = np.array(enc.encode_ordinary(code_corpus[int(len(code_corpus)*0.9):]), dtype=np.uint16)
        except Exception:
            chars = sorted(list(set(code_corpus)))
            stoi = { ch:i for i,ch in enumerate(chars) }
            train_ids = np.array([stoi[c] for c in code_corpus[:int(len(code_corpus)*0.9)]], dtype=np.uint16)
            val_ids = np.array([stoi[c] for c in code_corpus[int(len(code_corpus)*0.9):]], dtype=np.uint16)

        train_ids.tofile(train_bin)
        val_ids.tofile(val_bin)

    train_data = np.memmap(train_bin, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_bin, dtype=np.uint16, mode='r')
    return train_data, val_data

def get_batch(data, config: GPTConfig, device):
    max_idx = len(data) - config.block_size
    if max_idx <= 0:
        ix = torch.zeros((config.batch_size,), dtype=torch.long)
    else:
        ix = torch.randint(max_idx, (config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# ==============================================================================
# Model Architecture Components (Optimized for TPU XLA Execution Graph)
# ==============================================================================
class LayerNorm(nn.Module):
    """RMSNorm (Root Mean Square Layer Normalization)."""
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-5)
        return self.weight * x

def get_alibi_slopes(n_heads: int):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(n_heads))
    else:
        closest_pow_2 = 2 ** math.floor(math.log2(n_heads))
        return torch.tensor(get_slopes_power_of_2(closest_pow_2) + 
                            get_slopes_power_of_2(2 * closest_pow_2)[0::2][:n_heads - closest_pow_2])

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.q_norm = LayerNorm(config.n_embd // config.n_head)
        self.k_norm = LayerNorm(config.n_embd // config.n_head)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Precompute ALiBi linear slope bias
        slopes = get_alibi_slopes(config.n_head).view(1, config.n_head, 1, 1)
        positions = torch.arange(config.block_size).view(1, 1, 1, config.block_size)
        distance = positions - positions.transpose(-1, -2) # [1, 1, T, T]
        alibi_bias = slopes * distance
        self.register_buffer("alibi_bias", alibi_bias)
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # QK Normalization for attention stability
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Attention with ALiBi relative position bias (XLA-compiled matmul)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att + self.alibi_bias[:, :, :T, :T].to(x.dtype)
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = int(8 * config.n_embd / 3) # SwiGLU 8/3 expansion
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # Weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        tok_emb = self.transformer.wte(idx)
        x = tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# Python Code Syntax AST Evaluation Benchmark
# ==============================================================================
@torch.no_grad()
def evaluate_code_syntax_benchmark(model, config: GPTConfig, device):
    prompts = [
        "def binary_search(arr, target):\n    ",
        "def quick_sort(arr):\n    ",
        "def is_prime(n: int) -> bool:\n    ",
        "class Node:\n    def __init__(self, val):\n        ",
        "def factorial(n: int):\n    ",
    ]
    valid_syntax_count = 0
    total_prompts = len(prompts)
    
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        decode_fn = lambda tokens: enc.decode(tokens)
        encode_fn = lambda text: enc.encode_ordinary(text)
    except Exception:
        decode_fn = lambda tokens: "".join([chr(t) for t in tokens if t < 256])
        encode_fn = lambda text: [ord(c) % 256 for c in text]

    for prompt in prompts:
        tokens = encode_fn(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        out_tokens = model.generate(x, max_new_tokens=40, temperature=0.7)[0].cpu().tolist()
        generated_code = decode_fn(out_tokens)
        
        try:
            ast.parse(generated_code)
            valid_syntax_count += 1
        except SyntaxError:
            try:
                ast.parse(generated_code + "\n    pass")
                valid_syntax_count += 0.5
            except Exception:
                pass

    syntax_pass_rate = (valid_syntax_count / total_prompts) * 100.0
    return syntax_pass_rate

# ==============================================================================
# Training & Loss Evaluation
# ==============================================================================
@torch.no_grad()
def estimate_loss(model, train_data, val_data, config: GPTConfig, device):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, config, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it, config: GPTConfig):
    if it < config.warmup_iters:
        return config.learning_rate * (it + 1) / (config.warmup_iters + 1)
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def main():
    config = GPTConfig()
    start_time = time.time()

    # Hardware & PyTorch/XLA TPU Initialization
    if XLA_AVAILABLE:
        device = xm.xla_device()
        print(f"Initialized Google Cloud TPU via PyTorch/XLA PJRT (Device: {device})")
    else:
        device = torch.device('cpu')
        print("PyTorch/XLA unavailable. Running on CPU fallback.")

    print(f"Active training device: {device}")
    print("Setting default precision to native TPU bfloat16...")

    # Load Python Code Dataset
    train_data, val_data = get_dataset()
    
    # Instantiate Model in bfloat16 on TPU
    model = GPT(config).to(device=device, dtype=torch.bfloat16)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"nanoCode-RSI Parameter Count: {param_count:,} (bfloat16 on TPU)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=1e-1)

    iter_num = 0
    while iter_num < config.max_iters:
        # Check 4.5-minute time budget
        elapsed = time.time() - start_time
        if elapsed > config.max_time_seconds:
            print(f"Time budget of {config.max_time_seconds}s reached at step {iter_num}. Exiting training loop.")
            break

        # Adjust LR
        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward & Backward pass with Gradient Accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(config.gradient_accumulation_steps):
            X, Y = get_batch(train_data, config, device)
            logits, loss = model(X, Y)
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

        # TPU/XLA Optimizer Step (gradient all-reduce & execution graph compilation)
        if XLA_AVAILABLE:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            xm.optimizer_step(optimizer)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Periodic Validation (avoiding inner-loop sync stalls)
        if iter_num > 0 and iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config, device)
            print(f"step {iter_num:4d} (elapsed {elapsed:.1f}s): code train loss {losses['train']:.4f}, code val loss {losses['val']:.4f}")

        iter_num += 1

    # Final Evaluation & Code AST Benchmark
    final_losses = estimate_loss(model, train_data, val_data, config, device)
    final_val_loss = final_losses['val']
    syntax_pass_rate = evaluate_code_syntax_benchmark(model, config, device)
    
    print("=" * 60)
    print(f"TPU_FINAL_RESULTS: code_val_loss={final_val_loss:.4f}, syntax_pass_rate={syntax_pass_rate:.1f}%, total_time={time.time()-start_time:.1f}s")
    print(f"VAL_METRIC: {final_val_loss:.4f}")
    print(f"CODE_SYNTAX_PASS_RATE: {syntax_pass_rate:.1f}%")
    print("=" * 60)

if __name__ == '__main__':
    main()
