# Research Log: nanoCode-RSI

## Experiment Summary & Pareto Frontier

### Milestone: From-Scratch Conversational Coding LLM (51.4M Parameters)
- **Date**: August 19, 2026
- **Architecture**: Causal Transformer with ALiBi, QK-Norm, SwiGLU, RMSNorm
- **Initialization**: 100% Random Gaussian $\mathcal{N}(0, 0.02)$, zero bias (NO pretrained weights)
- **Hardware**: Dual Nvidia Tesla T4 GPUs (`2x Tesla T4`) with CUDA `bfloat16` mixed precision
- **Training Tokens**: 686,880 train tokens across multi-turn `<|user|>` and `<|assistant|>` dialogue pairs
- **Final Validation Loss**: **`0.0085`** (down from $10.5043$ at step 0)

### Training Loss Trajectory
| Step | Train Loss | Val Loss | Learning Rate | Speed |
| :--- | :--- | :--- | :--- | :--- |
| **0** | 10.5069 | 10.5043 | 1.18e-05 | 5,395 tok/s |
| **50** | 0.2507 | 0.2472 | 6.00e-04 | 5,743 tok/s |
| **100** | 0.0250 | 0.0246 | 5.89e-04 | 5,760 tok/s |
| **150** | 0.0226 | 0.0228 | 5.57e-04 | 5,787 tok/s |
| **200** | 0.0146 | 0.0144 | 5.07e-04 | 5,769 tok/s |
| **250** | 0.0095 | 0.0093 | 4.42e-04 | 5,745 tok/s |
| **296** | **0.0085** | **0.0085** | 3.80e-04 | 5,750 tok/s |

### Verified Sample Output
```python
<|user|> Write a function to check if a number is prime.
<|assistant|>
def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```
