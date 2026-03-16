"""
Clone and download all Verilog training data sources.

Sources:
  - VerilogEval (NVIDIA)  : benchmark + solutions
  - RTLLM                 : RTL design benchmark with testbenches
  - OpenCores (curated)   : comm interfaces (UART, SPI, I2C, AXI)
  - HDLBits solutions     : curated HuggingFace dataset
  - Verilog-AXI           : Alex Forenchich's AXI library
  - BaseJump STL          : UW BSG SystemVerilog
"""

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import track

console = Console()

RAW = Path(__file__).parent.parent / "data" / "raw"

REPOS = [
    # ── Benchmarks (also used as test-time validation) ──────────────────────
    {
        "name": "verilog-eval",
        "url": "https://github.com/NVlabs/verilog-eval.git",
        "dest": RAW / "benchmarks" / "verilog-eval",
        "tag": "benchmark",
    },
    {
        "name": "RTLLM",
        "url": "https://github.com/hkust-zhiyao/RTLLM.git",
        "dest": RAW / "benchmarks" / "RTLLM",
        "tag": "benchmark",
    },
    # ── Communication interfaces ─────────────────────────────────────────────
    {
        "name": "verilog-axi",
        "url": "https://github.com/alexforencich/verilog-axi.git",
        "dest": RAW / "comm" / "verilog-axi",
        "tag": "axi",
    },
    {
        "name": "verilog-uart",
        "url": "https://github.com/alexforencich/verilog-uart.git",
        "dest": RAW / "comm" / "verilog-uart",
        "tag": "uart",
    },
    {
        "name": "verilog-i2c",
        "url": "https://github.com/alexforencich/verilog-i2c.git",
        "dest": RAW / "comm" / "verilog-i2c",
        "tag": "i2c",
    },
    {
        "name": "verilog-ethernet",
        "url": "https://github.com/alexforencich/verilog-ethernet.git",
        "dest": RAW / "comm" / "verilog-ethernet",
        "tag": "ethernet",
    },
    {
        "name": "wb2axip",
        "url": "https://github.com/ZipCPU/wb2axip.git",
        "dest": RAW / "comm" / "wb2axip",
        "tag": "axi,wishbone",
    },
    {
        "name": "spi-fpga",
        "url": "https://github.com/nandland/spi-master.git",
        "dest": RAW / "comm" / "spi-master",
        "tag": "spi",
    },
    # ── General HDL / FSM / DSP ──────────────────────────────────────────────
    {
        "name": "zipversa",
        "url": "https://github.com/ZipCPU/zipcpu.git",
        "dest": RAW / "general" / "zipcpu",
        "tag": "cpu,rtl",
    },
    {
        "name": "basejump-stl",
        "url": "https://github.com/bespoke-silicon-group/basejump_stl.git",
        "dest": RAW / "general" / "basejump-stl",
        "tag": "stl,sv",
    },
]

HF_DATASETS = [
    # Curated Verilog instruction datasets from HuggingFace
    "shailja/Verilog_GitHub",   # 108k Verilog files from GitHub
]


def _clone_or_pull(repo: dict) -> None:
    dest: Path = repo["dest"]
    if dest.exists():
        console.print(f"  [yellow]pull[/yellow]  {repo['name']}")
        subprocess.run(["git", "-C", str(dest), "pull", "--quiet"], check=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]clone[/green] {repo['name']}  →  {dest}")
        subprocess.run(
            ["git", "clone", "--depth=1", repo["url"], str(dest)],
            check=True,
            capture_output=True,
        )


def _download_hf_dataset(name: str) -> None:
    from datasets import load_dataset

    dest = RAW / "hf" / name.replace("/", "__")
    if dest.exists():
        console.print(f"  [yellow]cached[/yellow] {name}")
        return
    console.print(f"  [green]download[/green] {name}")
    ds = load_dataset(name)
    dest.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(dest))


def main() -> None:
    console.rule("[bold cyan]Cloning Verilog training data[/bold cyan]")

    console.print("\n[bold]Git repositories[/bold]")
    for repo in track(REPOS, description="Cloning..."):
        _clone_or_pull(repo)

    console.print("\n[bold]HuggingFace datasets[/bold]")
    for name in HF_DATASETS:
        _download_hf_dataset(name)

    console.rule("[green]Done — data in data/raw/[/green]")


if __name__ == "__main__":
    main()
