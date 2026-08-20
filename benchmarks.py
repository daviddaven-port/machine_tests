"""
Execution-Based Code Synthesis Benchmark Suite for nanoCode-RSI.
Evaluates functional correctness (Pass@1) across unseen algorithmic and systems programming tasks
inspired by HumanEval and MBPP.
"""
import ast
import sys
import multiprocessing
import traceback

BENCHMARK_TASKS = [
    {
        "task_id": "HumanEval/0",
        "prompt": "<|user|>\nWrite a Python function `has_close_elements(numbers: list[float], threshold: float) -> bool` to check if in given list of numbers, are any two numbers closer to each other than given threshold.\n<|assistant|>\n",
        "function_name": "has_close_elements",
        "test_cases": [
            (([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3), True),
            (([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05), False),
            (([1.0, 2.0, 5.9, 4.0, 5.0], 0.95), True),
            (([1.0, 2.0, 5.9, 4.0, 5.0], 0.8), False),
            (([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1), True),
        ]
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "<|user|>\nWrite a Python function `separate_paren_groups(paren_string: str) -> list[str]` that separates nested groups of parentheses into separate strings.\n<|assistant|>\n",
        "function_name": "separate_paren_groups",
        "test_cases": [
            (("(()()) ((())) () ((())()())",), ['(()())', '((()))', '()', '((())()())']),
            (("() (()) ((())) (((())))",), ['()', '(())', '((()))', '((((()))))']),
            (("(()(())((())))",), ['(()(())((())))']),
        ]
    },
    {
        "task_id": "HumanEval/2",
        "prompt": "<|user|>\nWrite a Python function `truncate_number(number: float) -> float` that returns the decimal part of the number.\n<|assistant|>\n",
        "function_name": "truncate_number",
        "test_cases": [
            ((3.5,), 0.5),
            ((1.25,), 0.25),
            ((123.0,), 0.0),
        ]
    },
    {
        "task_id": "HumanEval/3",
        "prompt": "<|user|>\nWrite a Python function `below_zero(operations: list[int]) -> bool` that returns True if bank balance falls below zero at any point.\n<|assistant|>\n",
        "function_name": "below_zero",
        "test_cases": [
            (([1, 2, 3],), False),
            (([1, 2, -4, 5],), True),
            (([1, -1, 2, -2, 5, -5, 4, -4],), False),
            (([1, -1, 2, -2, 5, -5, 4, -5],), True),
        ]
    },
    {
        "task_id": "MBPP/1",
        "prompt": "<|user|>\nWrite a Python function `min_cost(cost: list[list[int]], m: int, n: int) -> int` to find the minimum cost path to reach (m, n) from (0, 0) in a grid.\n<|assistant|>\n",
        "function_name": "min_cost",
        "test_cases": [
            (([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2), 8),
            (([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2), 12),
        ]
    }
]

def evaluate_functional_correctness(code_str: str, function_name: str, test_cases: list):
    local_scope = {}
    try:
        # Check AST syntax
        ast.parse(code_str)
        exec(code_str, {}, local_scope)
        fn = local_scope.get(function_name)
        if not callable(fn):
            return False, "Function not found in scope"
        
        for args, expected in test_cases:
            res = fn(*args)
            if isinstance(expected, float):
                if abs(res - expected) > 1e-5:
                    return False, f"Output mismatch: got {res}, expected {expected}"
            else:
                if res != expected:
                    return False, f"Output mismatch: got {res}, expected {expected}"
        return True, "All tests passed"
    except Exception as e:
        return False, str(e)

def run_benchmark_suite(generate_fn):
    print("=" * 70)
    print("      Running General Execution-Based Code Synthesis Benchmark")
    print("=" * 70)
    passed_tasks = 0
    total_tasks = len(BENCHMARK_TASKS)

    for task in BENCHMARK_TASKS:
        t_id = task["task_id"]
        prompt = task["prompt"]
        fn_name = task["function_name"]
        tests = task["test_cases"]

        print(f"\n[Evaluating {t_id}] ({fn_name})...")
        generated_raw = generate_fn(prompt)
        
        # Extract assistant response
        if "<|assistant|>" in generated_raw:
            code = generated_raw.split("<|assistant|>")[-1].strip()
        else:
            code = generated_raw
        
        # Clean trailing tokens
        clean_lines = []
        for line in code.splitlines():
            if line.startswith("<|user|>") or line.startswith("<|end|>"):
                break
            clean_lines.append(line)
        clean_code = "\n".join(clean_lines)

        passed, reason = evaluate_functional_correctness(clean_code, fn_name, tests)
        status = "PASSED" if passed else f"FAILED ({reason})"
        print(f"  Result: {status}")
        if passed:
            passed_tasks += 1

    pass_rate = (passed_tasks / total_tasks) * 100.0
    print("\n" + "=" * 70)
    print(f"Benchmark Pass@1 Rate: {passed_tasks}/{total_tasks} ({pass_rate:.1f}%)")
    print("=" * 70)
    return pass_rate
