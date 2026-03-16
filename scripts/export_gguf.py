"""
Fuse LoRA adapters into the base model and export to GGUF for LM Studio.

Steps:
  1. mlx_lm.fuse  → merge adapter weights into base model (HF safetensors)
  2. llama.cpp convert_hf_to_gguf.py → create .gguf file
  3. llama-quantize → quantize to Q4_K_M (matches original LM Studio model)

Prerequisites:
  - llama.cpp must be built: brew install llama.cpp   OR   build from source
    (the convert_hf_to_gguf.py script is needed for step 2)

Usage:
    uv run export-gguf                     # defaults
    uv run export-gguf --quant Q8_0       # 8-bit
    uv run export-gguf --no-quantize       # keep f16
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()
ROOT    = Path(__file__).parent.parent
FUSED   = ROOT / "outputs" / "fused"
GGUF    = ROOT / "outputs" / "gguf"


def _find_llama_cpp() -> Path | None:
    """Locate llama.cpp convert script."""
    candidates = [
        Path("/opt/homebrew/bin/llama-quantize"),          # brew
        Path.home() / "llama.cpp" / "llama-quantize",
        Path("/usr/local/bin/llama-quantize"),
    ]
    for p in candidates:
        if p.exists():
            return p.parent
    return None


def _find_convert_script(llama_dir: Path | None) -> Path | None:
    if llama_dir is None:
        return None
    script = llama_dir / "convert_hf_to_gguf.py"
    if not script.exists():
        script = llama_dir.parent / "convert_hf_to_gguf.py"
    return script if script.exists() else None


def step1_fuse(adapter_path: Path, model_id: str) -> None:
    console.print("\n[bold cyan]Step 1 — Fusing LoRA adapter into base model[/bold cyan]")
    FUSED.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", model_id,
            "--adapter-path", str(adapter_path),
            "--save-path", str(FUSED),
            "--de-quantize",   # convert back to f16 for GGUF conversion
        ],
        check=True,
        cwd=str(ROOT),
    )
    console.print(f"  Fused model saved → {FUSED}")


def step2_convert(convert_script: Path) -> Path:
    console.print("\n[bold cyan]Step 2 — Converting to GGUF (f16)[/bold cyan]")
    GGUF.mkdir(parents=True, exist_ok=True)
    out_f16 = GGUF / "qwen35-verilog-f16.gguf"
    subprocess.run(
        [
            sys.executable, str(convert_script),
            str(FUSED),
            "--outfile", str(out_f16),
            "--outtype", "f16",
        ],
        check=True,
        cwd=str(ROOT),
    )
    console.print(f"  F16 GGUF → {out_f16}")
    return out_f16


def step3_quantize(llama_dir: Path, f16_path: Path, quant: str) -> Path:
    console.print(f"\n[bold cyan]Step 3 — Quantizing to {quant}[/bold cyan]")
    out_q = GGUF / f"qwen35-verilog-{quant.lower()}.gguf"
    quantize_bin = llama_dir / "llama-quantize"
    subprocess.run(
        [str(quantize_bin), str(f16_path), str(out_q), quant],
        check=True,
    )
    console.print(f"  Quantized GGUF → {out_q}")
    return out_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", type=Path,
                        default=ROOT / "outputs" / "adapters")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                        help="Base HF model id used during fine-tuning")
    parser.add_argument("--quant", default="Q4_K_M",
                        help="llama.cpp quantization type (Q4_K_M, Q8_0, Q5_K_M, …)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Stop after f16 GGUF, skip quantization")
    args = parser.parse_args()

    console.rule("[bold cyan]Export Fine-tuned Model → GGUF[/bold cyan]")

    llama_dir    = _find_llama_cpp()
    convert_script = _find_convert_script(llama_dir)

    if not convert_script:
        console.print(
            "[red]llama.cpp not found.[/red]\n"
            "Install with:  [bold]brew install llama.cpp[/bold]\n"
            "Or build from source and ensure convert_hf_to_gguf.py is on PATH."
        )
        sys.exit(1)

    step1_fuse(args.adapter_path, args.model)
    f16_path = step2_convert(convert_script)

    if not args.no_quantize:
        final = step3_quantize(llama_dir, f16_path, args.quant)
        console.print(
            f"\n[bold green]Done![/bold green] Copy to LM Studio models folder:\n"
            f"  cp {final} ~/Library/Application\\ Support/LM\\ Studio/models/lmstudio-community/"
        )
    else:
        console.print(f"\n[bold green]Done![/bold green] F16 GGUF at: {f16_path}")


if __name__ == "__main__":
    main()
