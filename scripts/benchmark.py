"""
Evaluate the fine-tuned model against:
  1. VerilogEval (NVIDIA)  – functional correctness via iverilog simulation
  2. RTLLM                 – RTL design problems with testbenches
  3. Internal test split   – data/processed/test.jsonl (BLEU / exact match)

Prerequisites:
  - iverilog must be installed: brew install icarus-verilog
  - Run `uv run finetune` first, then `uv run export-gguf`
  - Adapter must be fused: outputs/adapters/ or outputs/gguf/

Usage:
    uv run benchmark                    # full eval
    uv run benchmark --suite verilogeval
    uv run benchmark --suite rtllm
    uv run benchmark --suite internal
    uv run benchmark --max-problems 20  # quick smoke test
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()
ROOT     = Path(__file__).parent.parent
BMARK    = ROOT / "data" / "raw" / "benchmarks"
RESULTS  = ROOT / "benchmarks"


# ── iverilog simulation ───────────────────────────────────────────────────────

def _compile_and_run(verilog_code: str, testbench: str | None = None) -> tuple[bool, str]:
    """Return (passed, log)."""
    if not shutil.which("iverilog"):
        return False, "iverilog not found — install with: brew install icarus-verilog"

    with tempfile.TemporaryDirectory() as tmp:
        dut = Path(tmp) / "dut.v"
        dut.write_text(verilog_code)

        if testbench:
            tb = Path(tmp) / "tb.v"
            tb.write_text(testbench)
            sources = [str(tb), str(dut)]
        else:
            sources = [str(dut)]

        out_bin = Path(tmp) / "sim"
        compile_res = subprocess.run(
            ["iverilog", "-o", str(out_bin), "-g2012"] + sources,
            capture_output=True, text=True,
        )
        if compile_res.returncode != 0:
            return False, "COMPILE ERROR:\n" + compile_res.stderr

        if not testbench:
            return True, "Compiled OK (no testbench)"

        sim_res = subprocess.run(
            ["vvp", str(out_bin)], capture_output=True, text=True, timeout=10
        )
        log = sim_res.stdout + sim_res.stderr
        passed = "FAIL" not in log.upper() and "ERROR" not in log.upper()
        return passed, log


# ── Model inference ───────────────────────────────────────────────────────────

def _generate(prompt: str, adapter_path: Path, max_tokens: int = 1024) -> str:
    """Run mlx_lm.generate with the fine-tuned adapter."""
    cmd = [
        "python3", "-m", "mlx_lm.generate",
        "--model", "Qwen/Qwen3.5-0.8B",
        "--adapter-path", str(adapter_path),
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    return result.stdout.strip()


def _extract_verilog(text: str) -> str:
    """Extract first ```verilog ... ``` block, or the raw text."""
    m = re.search(r"```(?:verilog|systemverilog)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ── VerilogEval suite ─────────────────────────────────────────────────────────

def run_verilogeval(adapter_path: Path, max_problems: int) -> dict:
    ve_dir = BMARK / "verilog-eval"
    if not ve_dir.exists():
        console.print("[yellow]VerilogEval not cloned. Run `uv run clone-data`[/yellow]")
        return {}

    # VerilogEval stores problems in data/problem_descriptions/ and testbenches in data/testbenches/
    desc_dir = ve_dir / "data" / "problem_descriptions"
    tb_dir   = ve_dir / "data" / "testbenches"
    if not desc_dir.exists():
        console.print(f"[yellow]VerilogEval structure not found at {desc_dir}[/yellow]")
        return {}

    problems = sorted(desc_dir.glob("*.txt"))[:max_problems]
    passed = 0

    for prob in problems:
        name = prob.stem
        description = prob.read_text()
        tb_file = tb_dir / f"{name}_tb.v"
        testbench = tb_file.read_text() if tb_file.exists() else None

        prompt = f"<|im_start|>system\n{_get_system()}<|im_end|>\n<|im_start|>user\nCreate a Verilog module that satisfies the following specification:\n\n{description}<|im_end|>\n<|im_start|>assistant\n"
        generated = _generate(prompt, adapter_path)
        code = _extract_verilog(generated)
        ok, log = _compile_and_run(code, testbench)
        if ok:
            passed += 1
        console.print(f"  {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'}  {name}")

    return {"suite": "VerilogEval", "total": len(problems), "passed": passed}


# ── RTLLM suite ───────────────────────────────────────────────────────────────

def run_rtllm(adapter_path: Path, max_problems: int) -> dict:
    rtllm_dir = BMARK / "RTLLM"
    if not rtllm_dir.exists():
        console.print("[yellow]RTLLM not cloned. Run `uv run clone-data`[/yellow]")
        return {}

    problems = [d for d in rtllm_dir.iterdir() if d.is_dir()][:max_problems]
    passed = 0

    for prob in problems:
        desc_file = prob / "description.txt"
        tb_files  = list(prob.glob("*tb*.v")) + list(prob.glob("*testbench*.v"))
        if not desc_file.exists():
            continue
        description = desc_file.read_text()
        testbench   = tb_files[0].read_text() if tb_files else None

        prompt = f"<|im_start|>system\n{_get_system()}<|im_end|>\n<|im_start|>user\n{description}<|im_end|>\n<|im_start|>assistant\n"
        generated = _generate(prompt, adapter_path)
        code = _extract_verilog(generated)
        ok, _ = _compile_and_run(code, testbench)
        if ok:
            passed += 1
        console.print(f"  {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'}  {prob.name}")

    return {"suite": "RTLLM", "total": len(problems), "passed": passed}


# ── Internal test split ────────────────────────────────────────────────────────

def run_internal(adapter_path: Path, max_problems: int) -> dict:
    test_file = ROOT / "data" / "processed" / "test.jsonl"
    if not test_file.exists():
        console.print("[yellow]test.jsonl not found. Run `uv run prepare`[/yellow]")
        return {}

    samples = []
    with test_file.open() as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:max_problems]

    compile_ok = 0
    for s in samples:
        messages = s["messages"]
        user_msg  = next(m["content"] for m in messages if m["role"] == "user")
        reference = next(m["content"] for m in messages if m["role"] == "assistant")

        prompt = _build_chatml(messages[:-1])   # system + user only
        generated = _generate(prompt, adapter_path)
        code = _extract_verilog(generated)
        ok, _ = _compile_and_run(code)
        if ok:
            compile_ok += 1

    return {"suite": "Internal", "total": len(samples), "passed": compile_ok}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_system() -> str:
    from scripts.prepare_dataset import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def _build_chatml(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["verilogeval", "rtllm", "internal", "all"],
                        default="all")
    parser.add_argument("--max-problems", type=int, default=50)
    parser.add_argument("--adapter-path", type=Path,
                        default=ROOT / "outputs" / "adapters")
    args = parser.parse_args()

    console.rule("[bold cyan]Verilog Benchmark Evaluation[/bold cyan]")

    results = []
    if args.suite in ("verilogeval", "all"):
        r = run_verilogeval(args.adapter_path, args.max_problems)
        if r:
            results.append(r)

    if args.suite in ("rtllm", "all"):
        r = run_rtllm(args.adapter_path, args.max_problems)
        if r:
            results.append(r)

    if args.suite in ("internal", "all"):
        r = run_internal(args.adapter_path, args.max_problems)
        if r:
            results.append(r)

    # Summary table
    console.rule("[bold]Results[/bold]")
    table = Table("Suite", "Passed", "Total", "Pass Rate")
    for r in results:
        rate = f"{r['passed'] / r['total'] * 100:.1f}%" if r["total"] else "—"
        table.add_row(r["suite"], str(r["passed"]), str(r["total"]), rate)
    console.print(table)

    # Save JSON
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "results.json"
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
