"""Entry point — shows the workflow summary."""
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    console.print(Panel(
        "[bold]Qwen3.5 2B — Verilog Fine-Tuning Pipeline[/bold]\n\n"
        "1. [cyan]uv run clone-data[/cyan]   — clone repos + download HF datasets\n"
        "2. [cyan]uv run prepare[/cyan]      — build train/valid/test.jsonl\n"
        "3. [cyan]uv run finetune[/cyan]     — full fine-tuning via mlx-lm\n"
        "4. [cyan]uv run benchmark[/cyan]    — evaluate with VerilogEval + RTLLM + iverilog\n"
        "5. [cyan]uv run export-gguf[/cyan]  — fuse + convert to GGUF for LM Studio\n\n"
        "[dim]See configs/lora.yaml to switch between full fine-tuning and LoRA.[/dim]",
        title="[bold green]Workflow[/bold green]",
    ))


if __name__ == "__main__":
    main()
