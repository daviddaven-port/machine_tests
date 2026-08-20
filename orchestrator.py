"""
Autonomous nanoCode-RSI Research Orchestrator.
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

# Dynamically load local Kaggle credentials
kaggle_token_path = os.path.expanduser("~/.kaggle/access_token")
if os.path.exists(kaggle_token_path):
    with open(kaggle_token_path, "r", encoding="utf-8") as f:
        os.environ["KAGGLE_API_TOKEN"] = f.read().strip()

def update_status(stage, iteration, best_metric, current_metric, details=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""# nanoCode-RSI Research Loop - Real-Time Status

**Last Updated:** {timestamp}
**Current Stage:** {stage}
**Iteration:** {iteration}
**Best Metric (Val Loss):** {f'{best_metric:.4f}' if best_metric is not None else 'None'}
**Latest Run Metric:** {f'{current_metric:.4f}' if current_metric is not None else 'N/A'}
**Details:** {details}

---
*Autonomous loop running 24/7 on remote Kaggle Dual T4 GPUs.*
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

def poll_kernel_completion(timeout_seconds=900, poll_interval=20):
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
    push_res = run_cmd(f'"{GIT_EXE}" push origin nanocode-clean', check=False)
    run_cmd(f'"{GIT_EXE}" push origin nanocode-clean:nanogpt-research --force', check=False)
    print(f"Git push result: {push_res.stdout}\n{push_res.stderr}")

def git_revert():
    run_cmd(f'"{GIT_EXE}" reset --hard HEAD')
    run_cmd(f'"{GIT_EXE}" clean -fd')

if __name__ == "__main__":
    print("Orchestrator ready.")
