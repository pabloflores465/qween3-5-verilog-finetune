"""
Convert raw Verilog sources into instruction-tuning JSONL.

Three task types are generated:
  1. CREATE  – "Create a file <name>.v that implements <desc>"
  2. MODIFY  – "Modify the following Verilog to <change>"
  3. EXPLAIN – "Explain what this module does"  (optional, helps generalization)

Output:  data/processed/train.jsonl
         data/processed/valid.jsonl
         data/processed/test.jsonl   ← reserved for benchmark eval
"""

import json
import random
import re
import textwrap
from pathlib import Path
from typing import Iterator

from rich.console import Console
from tqdm import tqdm
from transformers import AutoTokenizer

console = Console()

# Cargado una sola vez al inicio del script
_tokenizer = None

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        console.print("  [dim]Cargando tokenizador Qwen/Qwen3.5-0.8B...[/dim]")
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    return _tokenizer

def _token_count(sample: dict) -> int:
    """Cuenta tokens reales usando el tokenizador del modelo."""
    tok = _get_tokenizer()
    text = "".join(m["content"] for m in sample["messages"])
    return len(tok.encode(text, add_special_tokens=False))

RAW = Path(__file__).parent.parent / "data" / "raw"
OUT = Path(__file__).parent.parent / "data" / "processed"

MAX_TOKENS = 512  # M1 8GB: backward pass necesita recomputar activaciones, 512 es seguro

# ChatML system prompt used during fine-tuning
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert RTL designer specialised in synthesisable Verilog and SystemVerilog.
    When asked to CREATE a file, output ONLY the complete file content starting with the
    module declaration. When asked to MODIFY code, output the complete modified file.
    Always follow these rules:
      - Use non-blocking assignments (<=) in always @(posedge clk) blocks.
      - Declare all ports explicitly. Use parameter/localparam for constants.
      - Write synthesisable code unless explicitly asked for a testbench.
      - Filenames follow snake_case (e.g. axi_lite_slave.v).
""")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_verilog_files(root: Path, exts=(".v", ".sv")) -> Iterator[Path]:
    for ext in exts:
        yield from root.rglob(f"*{ext}")


def _module_name(src: str) -> str | None:
    m = re.search(r"\bmodule\s+(\w+)", src)
    return m.group(1) if m else None


def _strip_comments_header(src: str) -> str:
    """Remove leading license/comment block to save tokens."""
    lines = src.splitlines()
    in_header = True
    out = []
    for line in lines:
        stripped = line.strip()
        if in_header and (stripped.startswith("//") or stripped.startswith("/*") or stripped == ""):
            continue
        in_header = False
        out.append(line)
    return "\n".join(out)


def _infer_description(path: Path, module: str) -> str:
    """Heuristic: turn filename + module name into a human description."""
    stem = path.stem.replace("_", " ").replace("-", " ")
    tag_map = {
        "uart": "UART serial communication",
        "spi":  "SPI (Serial Peripheral Interface)",
        "i2c":  "I2C (Inter-Integrated Circuit)",
        "axi":  "AXI bus interface",
        "axil": "AXI-Lite bus interface",
        "fifo": "synchronous FIFO",
        "arb":  "round-robin arbiter",
        "mux":  "parameterised multiplexer",
        "dec":  "binary decoder",
        "enc":  "priority encoder",
        "fsm":  "finite state machine",
        "wb":   "Wishbone bus interface",
        "eth":  "Ethernet MAC/PHY interface",
    }
    for kw, desc in tag_map.items():
        if kw in stem or kw in module.lower():
            return desc
    return stem


def _build_create_sample(path: Path, src: str) -> dict | None:
    module = _module_name(src)
    if not module:
        return None
    desc = _infer_description(path, module)
    instruction = (
        f"Create a file named `{path.name}` that implements a synthesisable {desc} "
        f"module called `{module}` in Verilog."
    )
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": src.strip()},
        ],
        "task": "create",
        "file": path.name,
        "module": module,
    }


def _build_modify_sample(path: Path, src: str) -> dict | None:
    """Synthetic modification task: add a configurable reset polarity."""
    if "posedge clk" not in src or len(src) > 8000:
        return None
    module = _module_name(src)
    if not module:
        return None
    instruction = (
        f"The following Verilog file is `{path.name}`. "
        "Modify it so that the reset signal is active-low (negedge rst_n) "
        "instead of active-high. Keep all other logic unchanged. "
        "Output the complete modified file."
    )
    modified = src.replace("posedge rst", "negedge rst_n").replace(
        "if (rst)", "if (!rst_n)"
    )
    if modified == src:
        return None
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": instruction + "\n\n```verilog\n" + src.strip() + "\n```"},
            {"role": "assistant", "content": modified.strip()},
        ],
        "task": "modify",
        "file": path.name,
        "module": module,
    }


def _load_hf_dataset(name: str) -> list[dict]:
    from datasets import load_from_disk

    dest = RAW / "hf" / name.replace("/", "__")
    if not dest.exists():
        console.print(f"  [yellow]skip[/yellow] HF dataset {name} (not downloaded)")
        return []
    ds = load_from_disk(str(dest))
    samples = []
    split = ds["train"] if hasattr(ds, "__getitem__") and "train" in ds else ds
    for row in split:
        instruction = row.get("instruction") or row.get("prompt") or ""
        output = row.get("output") or row.get("completion") or ""
        if not instruction or not output:
            continue
        if not any(kw in output for kw in ("module ", "endmodule")):
            continue
        samples.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": instruction.strip()},
                {"role": "assistant", "content": output.strip()},
            ],
            "task": "hf_instruct",
        })
    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []

    # 1. Scan all cloned git repos
    console.rule("[cyan]Processing cloned repositories[/cyan]")
    repo_dirs = list(RAW.glob("comm/*")) + list(RAW.glob("general/*"))
    for repo in repo_dirs:
        if not repo.is_dir():
            continue
        files = list(_read_verilog_files(repo))
        console.print(f"  {repo.name}: {len(files)} files")
        for f in tqdm(files, desc=repo.name, leave=False):
            try:
                src = f.read_text(errors="ignore")
            except Exception:
                continue
            if len(src) < 100 or len(src) > 12000:
                continue
            src = _strip_comments_header(src)
            s = _build_create_sample(f, src)
            if s:
                samples.append(s)
            s = _build_modify_sample(f, src)
            if s:
                samples.append(s)

    # 2. HuggingFace curated datasets
    console.rule("[cyan]Processing HuggingFace datasets[/cyan]")
    for name in ["shailja/Verilog_GitHub"]:
        hf_samples = _load_hf_dataset(name)
        console.print(f"  {name}: {len(hf_samples)} samples")
        samples.extend(hf_samples)

    # 3. Filtrar por tokens reales (tokenizador del modelo)
    console.print("\n[bold]Filtrando por longitud con tokenizador real...[/bold]")
    before = len(samples)
    samples = [s for s in tqdm(samples, desc="Contando tokens") if _token_count(s) <= MAX_TOKENS]
    console.print(f"  {before} → {len(samples)} samples (descartados {before - len(samples)} con > {MAX_TOKENS} tokens)")

    # 4. Split
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    n_valid = max(50, int(n * 0.05))
    n_test  = max(50, int(n * 0.05))
    test    = samples[:n_test]
    valid   = samples[n_test : n_test + n_valid]
    train   = samples[n_test + n_valid :]

    def _write(path: Path, data: list[dict]) -> None:
        with path.open("w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        console.print(f"  wrote {len(data):>5} samples → {path}")

    console.rule("[cyan]Writing splits[/cyan]")
    _write(OUT / "train.jsonl", train)
    _write(OUT / "valid.jsonl", valid)
    _write(OUT / "test.jsonl",  test)

    console.rule(f"[green]Total: {n} samples[/green]")


if __name__ == "__main__":
    main()
