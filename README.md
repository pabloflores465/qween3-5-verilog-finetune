# Qwen3.5 0.8B — Verilog Fine-Tuning

Full fine-tuning de **Qwen/Qwen3.5-0.8B** especializado en:
- Generación de código Verilog para interfaces de comunicación (AXI, AXI-Lite, UART, SPI, I2C, Ethernet, Wishbone)
- **Creación de archivos** `.v` / `.sv` con nombre y estructura correcta
- **Modificación de código** existente según instrucciones
- Validación automática con benchmarks reales (VerilogEval, RTLLM) via `iverilog`

> **Nota:** El GGUF de LM Studio no se puede fine-tunear directamente.
> El pipeline descarga los pesos originales desde HuggingFace (`Qwen/Qwen3.5-0.8B`),
> entrena en Apple Silicon con MLX, y exporta de vuelta a GGUF Q8_0 para LM Studio.

---

## Requisitos previos

```bash
# uv (gestor de paquetes)
brew install uv

# iverilog (simulador para benchmarks)
brew install icarus-verilog

# llama.cpp (para exportar a GGUF)
brew install llama.cpp

# Token de HuggingFace (para descargar el modelo base)
huggingface-cli login
# o: export HF_TOKEN=hf_xxxxxxxxxxxx
```

Instalar dependencias del proyecto:

```bash
uv sync
```

---

## Flujo completo

### 1. Clonar datos de entrenamiento

```bash
uv run clone-data
```

Clona los siguientes repositorios en `data/raw/` y descarga datasets de HuggingFace:

| Fuente | Tipo | Tags |
|--------|------|------|
| alexforencich/verilog-axi | Git | AXI, AXI-Lite |
| alexforencich/verilog-uart | Git | UART |
| alexforencich/verilog-i2c | Git | I2C |
| alexforencich/verilog-ethernet | Git | Ethernet |
| ZipCPU/wb2axip | Git | AXI, Wishbone |
| nandland/spi-master | Git | SPI |
| ZipCPU/zipcpu | Git | CPU, RTL |
| bespoke-silicon-group/basejump_stl | Git | SystemVerilog |
| NVlabs/verilog-eval | Git | Benchmark |
| hkust-zhiyao/RTLLM | Git | Benchmark |
| shailja/Verilog_GitHub | HuggingFace | Instrucciones |
| BetsyTang/RTLCoder-Instruct | HuggingFace | Instrucciones |

---

### 2. Preparar el dataset

```bash
uv run prepare
```

Genera tres splits en `data/processed/`:
- `train.jsonl` — 90% de los samples
- `valid.jsonl` — 5% (usado durante entrenamiento para loss de validación)
- `test.jsonl`  — 5% (reservado exclusivamente para benchmark)

Cada sample tiene uno de estos formatos:

**CREATE** — crear un archivo nuevo:
```
User: Create a file named `uart_tx.v` that implements a synthesisable UART serial
      communication module called `uart_tx` in Verilog.
Assistant: module uart_tx ( ... ) ... endmodule
```

**MODIFY** — modificar código existente:
```
User: The following Verilog file is `axi_slave.v`. Modify it so that the reset
      signal is active-low (negedge rst_n)... [código original]
Assistant: [código modificado completo]
```

**HF_INSTRUCT** — instrucciones curadas de HuggingFace.

---

### 3. Full fine-tuning

```bash
uv run finetune
```

Usa `mlx-lm` optimizado para Apple Silicon. Configuración en `configs/lora.yaml`:

```yaml
model: "Qwen/Qwen3.5-0.8B"
num_layers: -1        # -1 = full fine-tuning (todos los pesos)
iters: 2000
batch_size: 2
learning_rate: 1e-5
max_seq_length: 4096
grad_checkpoint: true
adapter_path: "outputs/adapters"
```

> Si tienes < 8 GB de RAM, cambia `num_layers: 16` para hacer LoRA en las últimas 16 capas.

Opciones adicionales:

```bash
# Forzar LoRA en las últimas 16 capas (más rápido, menos RAM)
uv run finetune --lora --layers 16

# Reanudar desde el último checkpoint
uv run finetune --resume

# Monitorear el entrenamiento (en otra terminal)
tail -f outputs/adapters/training.log
```

Los checkpoints se guardan en `outputs/adapters/` cada 500 iteraciones.

---

### 4. Evaluar con benchmarks

```bash
# Benchmark completo (VerilogEval + RTLLM + split interno)
uv run benchmark

# Solo VerilogEval (NVIDIA)
uv run benchmark --suite verilogeval

# Solo RTLLM
uv run benchmark --suite rtllm

# Solo test split interno
uv run benchmark --suite internal

# Smoke test rápido (primeros 20 problemas)
uv run benchmark --max-problems 20
```

El benchmark:
1. Genera código Verilog con el modelo fine-tuneado
2. Compila con `iverilog`
3. Si hay testbench, ejecuta la simulación con `vvp`
4. Reporta `PASS` / `FAIL` por problema y tasa global

Resultados guardados en `benchmarks/results.json`.

---

### 5. Exportar a GGUF para LM Studio

```bash
# Q8_0 (igual que el modelo original en LM Studio)
uv run export-gguf --quant Q8_0

# Q4_K_M (más ligero)
uv run export-gguf --quant Q4_K_M

# Sin cuantizar (f16)
uv run export-gguf --no-quantize
```

El proceso:
1. **Fuse** — fusiona los pesos del adaptador en el modelo base (`outputs/fused/`)
2. **Convert** — convierte a GGUF f16 con `convert_hf_to_gguf.py` de llama.cpp
3. **Quantize** — cuantiza al formato elegido (`outputs/gguf/qwen35-verilog-q8_0.gguf`)

Copiar a LM Studio:

```bash
cp outputs/gguf/qwen35-verilog-q8_0.gguf \
   ~/Library/Application\ Support/LM\ Studio/models/lmstudio-community/
```

---

## Tiempo estimado (Apple Silicon)

| Chip | RAM | Modo | 2000 iters |
|------|-----|------|-----------|
| M1 | 8 GB | Full fine-tuning ⚠️ | ~70 min |
| M2 Pro | 16 GB | Full fine-tuning | ~45 min |
| M3 Max | 36 GB | Full fine-tuning | ~20 min |
| M4 Pro | 24 GB | Full fine-tuning | ~25 min |

> ⚠️ **M1 8 GB:** el modelo ocupa ~6.4 GB con `grad_checkpoint: true` (ya activado).
> Funciona, pero al límite — macOS puede hacer swapping y ralentizar el entrenamiento.
> Si falla por OOM, usa `uv run finetune --lora --layers 16` (~3 GB, ~25 min).
>
> Si el dataset generado en el paso 2 tiene más de 5000 samples, aumenta `iters` a 4000–6000.

---

## Estructura del proyecto

```
qween3.5/
├── pyproject.toml              # dependencias (mlx-lm, datasets, gitpython…)
├── configs/
│   └── lora.yaml               # hiperparámetros de entrenamiento
├── scripts/
│   ├── clone_data.py           # clona repos + descarga HF datasets
│   ├── prepare_dataset.py      # genera train/valid/test.jsonl
│   ├── finetune.py             # lanza mlx-lm con el config
│   ├── benchmark.py            # evalúa con VerilogEval + RTLLM + iverilog
│   └── export_gguf.py          # fuse → GGUF → cuantización
├── data/
│   ├── raw/                    # repos clonados y datasets HF
│   └── processed/              # train.jsonl, valid.jsonl, test.jsonl
├── outputs/
│   ├── adapters/               # checkpoints del entrenamiento
│   ├── fused/                  # modelo fusionado (HF safetensors)
│   └── gguf/                   # archivo .gguf final
└── benchmarks/
    └── results.json            # resultados de evaluación
```

---

## Referencia rápida

```bash
uv sync                              # instalar dependencias
uv run clone-data                    # paso 1: clonar datos
uv run prepare                       # paso 2: preparar dataset
uv run finetune                      # paso 3: entrenar (full fine-tuning)
uv run finetune --lora --layers 16   # paso 3 alternativo (LoRA, menos RAM)
uv run finetune --resume             # paso 3 reanudar desde checkpoint
uv run benchmark --max-problems 20   # paso 4: evaluar rápido
uv run benchmark                     # paso 4: evaluar completo
uv run export-gguf --quant Q8_0      # paso 5: exportar para LM Studio
```
