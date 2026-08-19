import os
import sys
import time
import math
import ast
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# nanoCode-RSI: 100% From-Scratch Conversational Coding Model
# Architecture: Transformer with ALiBi, QK-Norm, SwiGLU, RMSNorm
# Initialization: Random Gaussian N(0, 0.02), Bias=0
# Execution: CUDA Dual Tesla T4 (bfloat16)
# ==============================================================================

# Hyperparameters
VOCAB_SIZE = 50257     # GPT-2 BPE Tokenizer vocabulary size
BLOCK_SIZE = 512       # Context window
N_LAYER = 8            # Number of transformer layers
N_HEAD = 8             # Number of attention heads
N_EMBD = 512           # Embedding dimension
DROPOUT = 0.05
BATCH_SIZE = 16        # Batch size per step
GRAD_ACCUM_STEPS = 4   # Effective batch size = 64
MAX_STEPS = 600        # Training steps within time budget
WARMUP_STEPS = 50
LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WEIGHT_DECAY = 0.1
EVAL_INTERVAL = 50
EVAL_ITERS = 20
TIME_BUDGET_SECONDS = 1800  # 30 minute ceiling

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=== Training Target Device: {device} ===")
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}, GPU: {torch.cuda.get_device_name(0)}")

# ==============================================================================
# Model Architecture
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

def get_alibi_slopes(n_heads: int):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return torch.tensor(get_slopes_power_of_2(n_heads))
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes_a = get_slopes_power_of_2(closest_pow2)
        slopes_b = get_slopes_power_of_2(2 * closest_pow2)[0::2][:n_heads - closest_pow2]
        return torch.tensor(slopes_a + slopes_b)

def build_alibi_bias(n_heads: int, seq_len: int, device: torch.device):
    slopes = get_alibi_slopes(n_heads).to(device)
    pos = torch.arange(seq_len, device=device)
    relative_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
    relative_pos = torch.clamp(relative_pos, max=0)
    alibi = slopes.view(1, n_heads, 1, 1) * relative_pos.view(1, 1, seq_len, seq_len)
    return alibi

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.w3 = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = dropout
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, alibi_bias: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if alibi_bias is not None:
            att = att + alibi_bias[:, :, :T, :T]
        
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        if self.dropout > 0:
            att = F.dropout(att, p=self.dropout, training=self.training)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = RMSNorm(n_embd)
        hidden_dim = int(2 * (4 * n_embd) / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.mlp = SwiGLU(n_embd, hidden_dim)

    def forward(self, x: torch.Tensor, alibi_bias: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), alibi_bias=alibi_bias)
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoCodeGPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_head = n_head
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(vocab_size, n_embd),
            'drop': nn.Dropout(dropout),
            'h': nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)]),
            'ln_f': RMSNorm(n_embd),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        alibi_bias = build_alibi_bias(self.n_head, T, idx.device)

        for block in self.transformer.h:
            x = block(x, alibi_bias=alibi_bias)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# Synthetic Conversational Coding Dataset Generator
# ==============================================================================

def generate_synthetic_conversational_tokens():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        class SimpleEnc:
            def encode(self, text):
                return [ord(c) % 50257 for c in text]
            def decode(self, tokens):
                return "".join([chr(t % 128) for t in tokens])
        enc = SimpleEnc()

    code_corpus = [
        "<|user|> Write a function to check if a number is prime.\n<|assistant|>\ndef is_prime(n: int) -> bool:\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
        "<|user|> How do I sort an array using quicksort in Python?\n<|assistant|>\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
        "<|user|> Implement binary search for a sorted list.\n<|assistant|>\ndef binary_search(arr, target):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n",
        "<|user|> Create a class for a standard Stack with push and pop methods.\n<|assistant|>\nclass Stack:\n    def __init__(self):\n        self.items = []\n    def is_empty(self):\n        return len(self.items) == 0\n    def push(self, item):\n        self.items.append(item)\n    def pop(self):\n        if not self.is_empty():\n            return self.items.pop()\n        raise IndexError('pop from empty stack')\n    def peek(self):\n        if not self.is_empty():\n            return self.items[-1]\n        return None\n",
        "<|user|> Write a recursive Fibonacci function with memoization.\n<|assistant|>\ndef fib(n: int, memo: dict = None) -> int:\n    if memo is None:\n        memo = {}\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n    return memo[n]\n",
        "<|user|> How do I invert a binary tree in Python?\n<|assistant|>\nclass TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val\n        self.left = left\n        self.right = right\n\ndef invert_tree(root: TreeNode) -> TreeNode:\n    if root is None:\n        return None\n    root.left, root.right = invert_tree(root.right), invert_tree(root.left)\n    return root\n",
    ]

    all_tokens = []
    for sample in code_corpus * 800:
        all_tokens.extend(enc.encode(sample))
    
    data_arr = np.array(all_tokens, dtype=np.uint16)
    n = len(data_arr)
    train_data = data_arr[:int(n * 0.9)]
    val_data = data_arr[int(n * 0.9):]
    return train_data, val_data, enc

# ==============================================================================
# Training Loop
# ==============================================================================

def get_batch(data, batch_size, block_size, dev):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(dev), y.to(dev)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, dev):
    out = {}
    model.eval()
    for split, d in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(d, BATCH_SIZE, BLOCK_SIZE, dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16 if dev == "cuda" else torch.float32):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def get_lr(it):
    if it < WARMUP_STEPS:
        return LEARNING_RATE * (it + 1) / (WARMUP_STEPS + 1)
    if it > MAX_STEPS:
        return MIN_LR
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def evaluate_code_syntax_pass_rate(model, enc, dev, num_samples=10):
    prompts = [
        "<|user|> Write a function to check if a number is prime.\n<|assistant|>\ndef ",
        "<|user|> How do I sort an array using quicksort in Python?\n<|assistant|>\ndef ",
        "<|user|> Implement binary search for a sorted list.\n<|assistant|>\ndef ",
        "<|user|> Write a recursive Fibonacci function with memoization.\n<|assistant|>\ndef ",
    ]
    model.eval()
    valid_syntax = 0
    total = 0
    for p in prompts:
        for _ in range(num_samples // len(prompts)):
            tokens = enc.encode(p)
            x = torch.tensor(tokens, dtype=torch.long, device=dev).unsqueeze(0)
            with torch.no_grad():
                out_tokens = model.generate(x, max_new_tokens=80, temperature=0.7, top_k=30)
            decoded = enc.decode(out_tokens[0].tolist())
            # Extract code section
            if "<|assistant|>" in decoded:
                code_text = decoded.split("<|assistant|>")[-1].strip()
            else:
                code_text = decoded
            try:
                ast.parse(code_text)
                valid_syntax += 1
            except SyntaxError:
                pass
            total += 1
    model.train()
    pass_rate = (valid_syntax / max(1, total)) * 100.0
    return pass_rate

def main():
    print("Initializing Synthetic Conversational Dataset...")
    train_data, val_data, enc = generate_synthetic_conversational_tokens()
    print(f"Dataset compiled: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    model = NanoCodeGPT(VOCAB_SIZE, N_LAYER, N_HEAD, N_EMBD, DROPOUT).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"nanoCode-RSI Parameter Count: {param_count:,} (Random Init from Scratch)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    start_time = time.time()
    best_val_loss = float('inf')

    print("\n--- Starting Training Loop on Dual Tesla T4 (CUDA bfloat16) ---")
    for step in range(MAX_STEPS):
        t0 = time.time()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):
            X, Y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                logits, loss = model(X, Y)
                loss = loss / GRAD_ACCUM_STEPS
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        dt = time.time() - t0

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss(model, train_data, val_data, device)
            tokens_per_sec = (BATCH_SIZE * GRAD_ACCUM_STEPS * BLOCK_SIZE) / max(1e-5, dt)
            print(f"Step {step:4d}/{MAX_STEPS:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e} | Speed: {tokens_per_sec:.0f} tok/s | Elapsed: {time.time()-start_time:.1f}s")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

        if time.time() - start_time > TIME_BUDGET_SECONDS:
            print(f"Time budget reached at step {step}. Completing.")
            break

    total_training_time = time.time() - start_time
    final_losses = estimate_loss(model, train_data, val_data, device)
    syntax_pass_rate = evaluate_code_syntax_pass_rate(model, enc, device, num_samples=12)

    print("\n" + "=" * 60)
    print(f"FINAL FROM-SCRATCH RESULTS:")
    print(f"  Final Val Loss: {final_losses['val']:.4f}")
    print(f"  Best Val Loss:  {best_val_loss:.4f}")
    print(f"  Syntax Pass Rate: {syntax_pass_rate:.1f}%")
    print(f"  Total Time:     {total_training_time:.1f}s")
    print(f"VAL_METRIC: {final_losses['val']:.4f}")
    print(f"CODE_SYNTAX_PASS_RATE: {syntax_pass_rate:.1f}%")
    print("=" * 60)

    # Conversational Sampling
    print("\n--- Conversational Code Inference Samples ---")
    test_prompts = [
        "<|user|> Write a function to check if a number is prime.\n<|assistant|>\n",
        "<|user|> How do I sort an array using quicksort in Python?\n<|assistant|>\n"
    ]
    for prompt in test_prompts:
        print(f"\nPrompt:\n{prompt}")
        tokens = enc.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        out = model.generate(x, max_new_tokens=100, temperature=0.7, top_k=40)
        completion = enc.decode(out[0].tolist())
        print(f"Completion:\n{completion}\n{'-'*40}")

if __name__ == "__main__":
    main()
