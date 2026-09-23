# Theoretical Analysis

This directory contains the theoretical derivations and scripts for the roofline and recomputation analyses in the paper.

## Files

| File | Description |
|---|---|
| `OI.py` | Computes the operational intensity (OI) of transformer layer forward passes and plots OI against batch size for five models, with GPU roofline thresholds overlaid. Also prints per-sequence training FLOPS (including the vocabulary projection). Produces `operational_intensity.pdf`. |
| `recompute.py` | Compares the per-layer activation recomputation time against the PCIe activation reload time on an RTX 4090. Produces `recompute_vs_offload_time.pdf`. |
| `operational_intensity.pdf` | Plot of operational intensity vs. batch size ($s=2048$, batch $1$–$256$) for five models. Horizontal lines mark each GPU's roofline threshold, showing that consumer-grade GPU training is compute-bound even at small batch sizes. |
| `recompute_vs_offload_time.pdf` | Grouped bar chart comparing recomputation time and activation reload time per layer on the 4090 ($b=4, s=2048$). Recomputation is faster than PCIe reload for every tested model. |

## Summary of Notations

The table below defines the notation used throughout this analysis.

| Symbol | Meaning |
|---|---|
| $s$ | Sequence length |
| $b$ | Micro-batch size |
| $h$ | Hidden dimension |
| $m$ | Intermediate dimension in MLP |
| $a$ | Number of attention heads |
| $k$ | Number of key-value heads |
| $E_{\text{act}}$ | Number of active experts |
| $E$ | Number of experts in total |

## Hardware Specs

Hardware specs of consumer-grade and datacenter GPUs used in the following analysis.

| | **4090** | **5090** | **A100** | **H100** |
|---|---|---|---|---|
| **FP16 (TFLOPS)** | 330 | 419 | 312 | 989.5 |
| **Memory (GB)** | 24 | 32 | 80 | 80 |
| **Interconnect** | PCIe4 | PCIe5 | NVLink3 | NVLink4 |
| **Speed (GB/s)** | 32 | 64 | 300 | 450 |

## Model Configs

Configs of models used in the following analysis.
These configs are consistent with their open-source weights.

| Model | $h$ | $a$ | $k$ | $m$ | $E_{\text{act}}$ | $E$ |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 2048 | 16 | 8 | 6144 | 1 | 1 |
| Llama-3.1-8B | 4096 | 32 | 8 | 14336 | 1 | 1 |
| GPT-OSS-20B | 2880 | 64 | 8 | 2880 | 4 | 32 |
| Qwen3-32B | 5120 | 64 | 8 | 25600 | 1 | 1 |
| Qwen3-235B | 4096 | 64 | 4 | 1536 | 8 | 128 |

## Recomputation Analysis

### Activation Size

We derive an approximate formula for the memory required to store activations in the forward pass of a single GQA [1, 2, 3] transformer layer.
We use the same calculation method as in Korthikanti et al. [4], considering only the main contributors to the memory and ignoring small buffers.
We also assume that the network and the activations are stored in a 16-bit floating point format, and therefore each element requires 2 bytes for storage.

Each transformer layer consists of an attention and an MLP block connected with two layer norms.
Below, we derive the memory required to store activations for each of these elements:

**Attention block.** It includes a self-attention followed by a linear projection.

- Query (Q), Key (K), and Value (V) matrix multiplies: We only need to store their shared input with size $2sbh$.
- FlashAttention [5, 6]: It requires storage of Q, K, V with a total size $2sbh + 4\frac{k}{a}sbh$.
- Output projection: It has an input size of $2sbh$.

**MLP.**
Dense models compute a single MLP while MoE models compute $E_{\text{act}}$ MLPs.
The two linear layers store their inputs with sizes $2sbh$ and $2sbmE_{\text{act}}$.
The SwiGLU non-linearity also needs its input with size $4sbmE_{\text{act}}$ for back-propagation.
In total, the MLP block requires $2sbh + 6sbmE_{\text{act}}$ bytes of storage.

**Layer norm.**
Each layer norm stores its input with size $2sbh$ and therefore in total, we will need $4sbh$ of storage.

Summing the memory required for attention, MLP, and the layer-norms, the memory required to store the activations for a single layer of a transformer network is:

$$\text{Activations per layer} = \left(12 + \frac{4k}{a}\right) sbh + 6sbmE_{\text{act}} \text{ bytes}$$

By substituting the actual configuration of a LLaMA-3.1-8B [2] model ($s=16384, b=1, h=4096, m=14336, a=32, k=8, E_{\text{act}}=1$, and 32 layers), training with a single 16k-token sequence generates **68 GB** of activations from all layers.

### Activation Recompute vs. Reload Analysis

Consider a single transformer layer.
The forward/recompute pass performs four self-attention projections ($Q$, $K$, $V$, output), the attention computation, and three feed-forward network (FFN) projections (gate, up, down).
The total FP16 forward FLOPS are (we count only matrix-multiplication FLOPS, which dominate both compute and transfer time; each multiply-add accounts for two FLOPS; element-wise operations such as LayerNorm, softmax, and activation functions are negligible compared to matrix operations and therefore omitted):

$$\text{FLOPS}_{\text{fwd}} = \underbrace{4sbh^2}_{\text{Q, out proj.}} + \underbrace{4sbh^2\frac{k}{a}}_{\text{K, V proj.}} + \underbrace{4sb^2h}_{\text{attention}} + \underbrace{6sbhmE_{\text{act}}}_{\text{FFN}}$$

For dense transformer models, it is equivalent to $E_{\text{act}}=1$.

`recompute.py` substitutes micro-batch size $b=4$, sequence length $s=2048$, model configs in the table above, and hardware specs of the 4090 to calculate and compare the activation recompute time and reload time per layer.

## Roofline Analysis

### OI of Dense Transformer Layer Forwarding

The data movement consists of uploading the layer's parameters and the input activation and downloading the output activation.
Because PCIe is full-duplex (uploads and downloads proceed simultaneously), the effective transfer time is determined by the larger direction.

The upload volume is:

$$\text{Bytes}_{\text{fwd upload}} = \underbrace{4h^2}_{\text{Q, out proj.}} + \underbrace{4h^2\frac{k}{a}}_{\text{K, V proj.}} + \underbrace{6hm}_{\text{FFN}} + \underbrace{2bsh}_{\text{input act.}}$$

The download volume is $2bsh$ bytes for the output activation.
Since the upload side always exceeds the download side, the operational intensity of the dense transformer layer forward pass is therefore:

$$\text{OI}_{\text{fwd}} = \frac{4sbh^2 + 4sbh^2\frac{k}{a} + 4sb^2h + 6sbhm}{4h^2 + 4h^2\frac{k}{a} + 6hm + 2bsh}$$

### OI of a Mixture-of-Experts Layer

For MoE transformer layers [7], the attention computation is identical, but each token is routed to $E_{\text{act}}$ active experts out of $E$ total.
Computation scales with $E_{\text{act}}$ while the weight transfer must cover all $E$ expert matrices, since all experts are expected to be active when processing multiple tokens.
The operational intensity becomes:

$$\text{OI}_{\text{moe}} = \frac{4sbh^2 + 4sbh^2\frac{k}{a} + 4sb^2h + 6sbhmE_{\text{act}}}{4h^2 + 4h^2\frac{k}{a} + 6hmE + 2bsh}$$

The key distinction is that all $E$ expert weight sets ($6hmE$ bytes) must be transferred, but only $E_{\text{act}}$ experts contribute FLOPS in the numerator.
This makes MoE layers lower in OI than their dense counterparts of comparable parameter count.

### Backward Pass Has Even Higher OI

The analysis above covers the forward pass.
During the backward pass with activation recomputation, the total computation consists of three components: the recomputed forward pass, the gradient with respect to activations, and the gradient with respect to weights, each contributing approximately $1\times$ the forward FLOPS.
The total backward FLOPS are therefore approximately $3\times$ the forward FLOPS [8].

The data movement in the backward pass, accounting for full-duplex PCIe, is as follows.
The upload direction carries parameters, the input activation, and the output gradient.
The download direction carries the parameter gradient and the input gradient.
The effective transfer is the maximum of the two directions, which is the upload side:

$$\text{Bytes}_{\text{bwd}} = \text{Bytes}_{\text{fwd upload}} + 2bsh$$

With $2bsh < \text{Bytes}_{\text{fwd upload}}$, the ratio of backward to forward transfer lies strictly between 1 and 2.
Therefore, we have:

$$\text{OI}_{\text{bwd}} > \frac{3 \cdot \text{FLOPS}_{\text{fwd}}}{2 \cdot \text{Bytes}_{\text{fwd upload}}} > \text{OI}_{\text{fwd}}$$

If the forward pass is compute-bound, the backward pass is guaranteed to be compute-bound as well.

`OI.py` plots the forward-pass OI against batch size for all five models, with horizontal lines marking each GPU's roofline threshold.

## Reproducing

```bash
python OI.py          # generates operational_intensity.pdf
python recompute.py   # generates recompute_vs_offload_time.pdf
```

## References

[1] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388. https://arxiv.org/abs/2505.09388

[2] Llama Team, AI @ Meta. 2024. The Llama 3 Herd of Models. arXiv:2407.21783. https://arxiv.org/abs/2407.21783

[3] OpenAI. 2025. gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925. https://arxiv.org/abs/2508.10925

[4] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. 2022. Reducing Activation Recomputation in Large Transformer Models. arXiv:2205.05198. https://arxiv.org/abs/2205.05198

[5] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. arXiv:2205.14135. https://arxiv.org/abs/2205.14135

[6] Tri Dao. 2023. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691. https://arxiv.org/abs/2307.08691

[7] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2024. Mixtral of Experts. arXiv:2401.04088. https://arxiv.org/abs/2401.04088

[8] Penghui Qi, Xinyi Wan, Guangxing Huang, and Min Lin. 2023. Zero Bubble Pipeline Parallelism. arXiv:2401.10241. https://arxiv.org/abs/2401.10241
