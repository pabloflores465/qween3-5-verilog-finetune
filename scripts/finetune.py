"""
Launch fine-tuning via mlx-lm.

Usage:
    uv run finetune                          # uses configs/lora.yaml
    uv run finetune --full                   # force full fine-tuning
    uv run finetune --lora --layers 16       # LoRA on last 16 layers

mlx-lm handles the actual training; this script just validates the environment,
checks that the dataset exists, and delegates to `mlx_lm.lora`.
"""

import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

console = Console()
ROOT = Path(__file__).parent.parent
CONFIG = ROOT / "configs" / "lora.yaml"


def _check_dataset() -> None:
    processed = ROOT / "data" / "processed"
    for split in ("train.jsonl", "valid.jsonl"):
        p = processed / split
        if not p.exists():
            console.print(f"[red]Missing {p}. Run `uv run prepare` first.[/red]")
            sys.exit(1)
        n = sum(1 for _ in p.open())
        console.print(f"  {split}: {n} samples")


def _check_memory() -> None:
    import subprocess
    result = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
    )
    if result.returncode == 0:
        gb = int(result.stdout.strip()) / 1024**3
        console.print(f"  Unified memory: {gb:.1f} GB")
        if gb < 16:
            console.print(
                "[yellow]Warning: < 16 GB RAM. Full fine-tuning may OOM. "
                "Consider setting num_layers: 16 in configs/lora.yaml[/yellow]"
            )


def main(full: bool = False, lora: bool = False, layers: int | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full",   action="store_true", help="Force full fine-tuning")
    parser.add_argument("--lora",   action="store_true", help="Force LoRA mode")
    parser.add_argument("--layers", type=int, default=None, help="LoRA layers (overrides config)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    console.rule("[bold cyan]Qwen3.5 2B — Verilog Fine-Tuning[/bold cyan]")

    # Patch config if flags provided
    with CONFIG.open() as f:
        cfg = yaml.safe_load(f)

    if args.full:
        cfg["num_layers"] = -1
    if args.lora and args.layers:
        cfg["num_layers"] = args.layers
    elif args.lora:
        cfg["num_layers"] = 16

    # Write patched config to a temp file
    tmp_config = ROOT / "configs" / "_run.yaml"
    with tmp_config.open("w") as f:
        yaml.dump(cfg, f)

    num_layers = cfg["num_layers"]
    mode = "full fine-tuning" if num_layers == -1 else f"LoRA (last {num_layers} layers)"
    console.print(Panel(
        f"Model:      {cfg['model']}\n"
        f"Mode:       {mode}\n"
        f"Iters:      {cfg['iters']}\n"
        f"Batch size: {cfg['batch_size']}\n"
        f"Max seq:    {cfg['max_seq_length']}\n"
        f"Output:     {cfg['adapter_path']}",
        title="Config",
    ))

    console.print("\n[bold]Dataset[/bold]")
    _check_dataset()

    console.print("\n[bold]System[/bold]")
    _check_memory()

    console.print("\n[bold green]Starting training...[/bold green]\n")

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--config", str(tmp_config),
    ]
    if args.resume:
        cmd += ["--resume-adapter-file", str(ROOT / cfg["adapter_path"] / "adapters.safetensors")]

    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
