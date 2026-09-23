# Artifact: Efficient Training on Multiple Consumer GPUs with RoundPipe

**Paper:** Yibin Luo, Shiwei Gao, Huichuan Zheng, Youyou Lu, Jiwu Shu.
"Efficient Training on Multiple Consumer GPUs with RoundPipe."

**Source code:** <https://github.com/thustorage/RoundPipe>
([PyPI package](https://pypi.org/project/roundpipe/))

**License:** MIT (this artifact repository), Apache 2.0 (RoundPipe library)

## Overview

RoundPipe is a large DNN training framework that enables training huge models on consumer-grade GPUs.
It combines host-device pipelining with an asymmetric forward/backward stage partition and an asynchronous optimizer step to minimize pipeline bubbles while keeping the programming interface sequential.

This artifact contains all scripts and raw data needed to reproduce the experimental results in the paper.
It covers:

- **Theoretical analysis** (Sections 2.1.1, 3.3): Recomputation-vs-offload analysis and roofline model.
- **End-to-end training benchmarks** (Sections 5.2–5.3): Throughput and maximum sequence length on 8×RTX 4090 and 8×A800 across five models (Qwen3-1.7B, Llama-3.1-8B, GPT-OSS-20B, Qwen3-32B, Qwen3-235B-LoRA), comparing RoundPipe against DeepSpeed ZeRO-2, ZeRO-Infinity, PyTorch FSDP, Megatron-LM (PP and TP), and Mobius.
- **Cost efficiency** (Section 5.5): Training cost (USD per billion tokens) comparison.
- **Scalability** (Section 5.6): Multi-GPU scaling from 1 to 8 RTX 4090 GPUs.
- **Sequence length sensitivity** (Section 5.7): Throughput vs. sequence length for Qwen3-1.7B on 8×4090.
- **Pipeline simulation** (Section 5.8.1): Bubble-ratio comparison of RoundPipe against 1F1B, GPipe, Interleaved-1F1B, and Looped-BFS.

## Artifact Structure

```
RoundPipe-artifact/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies (pip install)
│
├── theoretical-analysis/        # §2.1.1, §3.3: Roofline & recomputation analysis
│   ├── OI.py                    #   → operational_intensity.pdf (Figure 5)
│   ├── recompute.py             #   → recompute_vs_offload_time.pdf (Figure 2)
│   └── README.md                #   Detailed derivations and notation
│
├── pp-simulate/                 # §5.8.1: Pipeline schedule simulator
│   ├── simulator.py             #   Core simulator (1F1B, GPipe, I-1F1B, Looped-BFS, RoundPipe)
│   ├── run.sh                   #   Run all model × scheduler combinations → log
│   ├── plot_bubble.py           #   → bubble_bar.pdf (Figure 17)
│   ├── plot_ideal.py            #   → bubble_ideal_compare.pdf (Figure 3)
│   └── README.md                #   Simulator parameters and semantics
│
├── roundpipe/                   # RoundPipe training scripts
│   ├── train.py                 #   Full fine-tuning benchmark
│   ├── train_lora.py            #   LoRA fine-tuning benchmark
│   ├── train-sync.py            #   Synchronous variant (no async optimizer)
│   ├── train_lora-sync.py       #   LoRA synchronous variant
│   └── run.sh                   #   All RoundPipe experiments (4090/A800/W7800/910B)
│
├── deepspeed/                   # DeepSpeed baseline (ZeRO-2, ZeRO-Infinity)
│   ├── train.py                 #   Full fine-tuning
│   ├── train-lora.py            #   LoRA fine-tuning
│   ├── zero2.json               #   ZeRO Stage 2 config
│   ├── zero-inf.json            #   ZeRO-Infinity config
│   └── run.sh                   #   All DeepSpeed experiments
│
├── fsdp/                        # PyTorch FSDP baseline (Zero3, offload)
│   ├── train.py                 #   Full fine-tuning
│   ├── train-lora.py            #   LoRA fine-tuning
│   └── run.sh                   #   All FSDP experiments
│
├── megatron/                    # Megatron-LM baseline (PP and TP)
│   ├── train.py                 #   Pipeline parallel (NeMo)
│   ├── train-tp.py              #   Tensor parallel (NeMo)
│   ├── train-lora.py            #   PP LoRA
│   ├── train-lora-tp.py         #   TP LoRA
│   └── run.sh                   #   All Megatron experiments
│
├── roundpipe-timing/            # §5.8.1: Layer timing data collection for simulator
│   ├── train.py                 #   Full fine-tuning with verbose layer timing
│   ├── train_lora.py            #   LoRA fine-tuning with verbose layer timing
│   ├── time_schedule.patch      #   Patch to roundpipe for layer timing output
│   └── run.sh                   #   Collect per-layer fwd/bwd times for all models
│
└── exp/                         # Figure generation from raw data
    ├── raw-data.csv             #   All measured throughput & max-seqlen data
    ├── config.py                #   Shared plotting style
    ├── plot_4090_throughput.py   #   → 4090_throughput.pdf (Figure 10)
    ├── plot_4090_maxseqlen.py   #   → 4090_maxseqlen.pdf (Figure 11)
    ├── plot_a800_throughput.py   #   → a800_throughput.pdf (Figure 12)
    ├── plot_a800_maxseqlen.py   #   → a800_maxseqlen.pdf (Figure 13)
    ├── plot_scalability.py      #   → scalability.pdf (Figure 15)
    ├── plot_seqlen_sensitivity.py # → seqlen_sensitivity.pdf (Figure 16)
    └── plot_cost_efficiency.py  #   → cost_efficiency.pdf (Figure 14)
```

## Claims and Corresponding Experiments

The table below maps each paper claim to the artifact component that supports it.

| Paper Claim | Section | Artifact Component | Output |
|---|---|---|---|
| Consumer GPUs are compute-bound even at small batch sizes (roofline analysis) | §3.3 | `theoretical-analysis/OI.py` | `operational_intensity.pdf` (Figure 5) |
| Recomputation is faster than PCIe activation reload | §2.1.1 | `theoretical-analysis/recompute.py` | `recompute_vs_offload_time.pdf` (Figure 2) |
| Interleaved schedulers approach but don't reach ideal bubble ratio | §2.3 | `pp-simulate/plot_ideal.py` | `bubble_ideal_compare.pdf` (Figure 3) |
| RoundPipe achieves higher throughput than baselines on 4090 | §5.2 | `roundpipe/run.sh`, `deepspeed/run.sh`, `fsdp/run.sh`, `megatron/run.sh` | `4090_throughput.pdf` (Figure 10) |
| RoundPipe supports longer sequences than baselines on 4090 | §5.2 | Same as above | `4090_maxseqlen.pdf` (Figure 11) |
| RoundPipe achieves competitive throughput on A800 | §5.3 | Same as above (A800 sections in `run.sh`) | `a800_throughput.pdf` (Figure 12) |
| RoundPipe supports longer sequences on A800 | §5.3 | Same as above | `a800_maxseqlen.pdf` (Figure 13) |
| RoundPipe is cost-efficient (USD per billion tokens) | §5.5 | `exp/plot_cost_efficiency.py` | `cost_efficiency.pdf` (Figure 14) |
| Linear multi-GPU scaling | §5.6 | `roundpipe/run.sh` (4090 multi-GPU) | `scalability.pdf` (Figure 15) |
| Throughput degrades gracefully with sequence length | §5.7 | `roundpipe/run.sh` (vary sequence length) | `seqlen_sensitivity.pdf` (Figure 16) |
| RoundPipe has lower pipeline bubble ratio than existing schedulers | §5.8.1 | `pp-simulate/run.sh` | `bubble_bar.pdf` (Figure 17), schedule PDFs |

## Hardware Requirements

Reproducing the full evaluation requires:

| Platform | Hardware | Purpose |
|---|---|---|
| 8× NVIDIA RTX 4090 (24 GB) | PCIe 4.0 interconnect | Main evaluation (§5.2, 5.5–5.8) |
| 8× NVIDIA A800 (80 GB) | NVLink interconnect | Datacenter GPU comparison (§5.3, 5.5) |

The pipeline simulator and theoretical analysis scripts require **no GPU** — they run on CPU only.

**CPU & Memory:** Intel Xeon or AMD EPYC CPU with 800+ GB available system RAM recommended.

## Software Dependencies

- **OS:** Ubuntu 22.04 with Linux 6.8
- **Python:** 3.10
- **PyTorch:** 2.7.0
- **CUDA:** 12.6
- **Flash Attention:** 2.8.3

This is the environment we used for all experiments. Other versions may work but are not guaranteed.

### Installation

In a python environment having pytorch installed, run the following commands:

```bash
pip install -r requirements.txt
```

### Baseline-Specific Dependencies

- **DeepSpeed** (≥ 0.18.6): `pip install deepspeed` (included in requirements.txt)
- **Megatron-LM:** Requires NVIDIA NeMo. We recommend pulling the official container: `docker pull nvcr.io/nvidia/nemo:25.11`. See [NeMo Framework Megatron Backend](https://catalog.ngc.nvidia.com/orgs/nvidia/-/containers/nemo/-).
- **Mobius:** Mobius is not open-sourced. We obtained the source code directly from the Mobius authors. Reviewers may contact the Mobius authors for access or skip the Mobius comparison.

### Model Weights

All models are loaded from Hugging Face Hub:

| Model | Hugging Face ID | Parameters |
|---|---|---|
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | 1.7B |
| Llama-3.1-8B | `meta-llama/Meta-Llama-3.1-8B` | 8B |
| GPT-OSS-20B | `openai/gpt-oss-20b` | 20B (MoE) |
| Qwen3-32B | `Qwen/Qwen3-32B` | 32B |
| Qwen3-235B (LoRA) | `Qwen/Qwen3-235B-A22B` | 235B (MoE) |

To use Llama-3.1-8B, you need to have access to the model on Hugging Face Hub. Ensure you have the necessary permissions and have logged in using huggingface CLI.

## Reproduction

### 1. Theoretical Analysis (CPU only)

Reproduces Figure 5 (roofline) and Figure 2 (recompute vs. offload):

```bash
cd theoretical-analysis
python OI.py          # → operational_intensity.pdf
python recompute.py   # → recompute_vs_offload_time.pdf
```

### 2. Pipeline Simulation (CPU only)

Reproduces Figures 3 and 17 (bubble ratio analysis) and schedule diagrams:

```bash
cd pp-simulate
bash run.sh           # runs all schedulers → log file + schedule PDFs
python plot_bubble.py # → bubble_bar.pdf
python plot_ideal.py  # → bubble_ideal_compare.pdf
```

### 3. End-to-End Training Benchmarks (8× GPU)

Each `run.sh` script contains clearly commented sections for each platform (4090, A800, etc.).
Run the relevant section for your hardware.
For scalability analysis, run the same script with different `CUDA_VISIBLE_DEVICES` environment variable settings to vary the number of GPUs.

Each script runs 5 warmup iterations followed by 10 measured iterations, reporting throughput in tokens/s.

### 4. Layer Timing Collection for Simulator (1× GPU)

Collects per-layer forward/backward timing data used to calibrate the pipeline simulator (Section 5.8.1).
The measured times are the `--fwd-time-per-layer` and `--bwd-time-per-layer` parameters in `pp-simulate/run.sh`.

```bash
cd roundpipe-timing
# Apply the timing instrumentation patch to your roundpipe installation:
pip install -e /path/to/RoundPipe   # install from source
cd /path/to/RoundPipe && git apply /path/to/roundpipe-timing/time_schedule.patch
# Then run:
cd roundpipe-timing
bash run.sh
```

The output prints per-layer forward and backward times for each model.

### 5. Generate Figures from Raw Data (CPU only)

If you want to regenerate the paper's figures from the provided raw data (or from your own measurements after updating `exp/raw-data.csv`):

```bash
cd exp
python plot_4090_throughput.py      # → 4090_throughput.pdf   (Figure 10)
python plot_4090_maxseqlen.py       # → 4090_maxseqlen.pdf   (Figure 11)
python plot_a800_throughput.py      # → a800_throughput.pdf   (Figure 12)
python plot_a800_maxseqlen.py       # → a800_maxseqlen.pdf   (Figure 13)
python plot_cost_efficiency.py      # → cost_efficiency.pdf   (Figure 14)
python plot_scalability.py          # → scalability.pdf       (Figure 15)
python plot_seqlen_sensitivity.py   # → seqlen_sensitivity.pdf (Figure 16)
```

## Notes

- The `run.sh` scripts reference local model paths (e.g., `/public/huggingface-models/...`). Replace these with your local paths or Hugging Face model IDs.
- Some baseline configurations are marked `# OOM` in the run scripts — these are expected to fail with out-of-memory errors, confirming the paper's OOM entries.
