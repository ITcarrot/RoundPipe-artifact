import numpy as np
import matplotlib.pyplot as plt

# Model hyperparameters sourced from the local /public/huggingface-models configs.
MODEL = ["Qwen3-1.7B", "Llama-3.1-8B", "gpt-oss-20B", "Qwen3-32B", "Qwen3-235B"]
H = [2048, 4096, 2880, 5120, 4096]
HEAD = [16, 32, 64, 64, 64]
KVHEAD = [8, 8, 8, 8, 4]
I = [6144, 14336, 2880, 25600, 1536]
Eact = [1, 1, 4, 1, 8]
E = [1, 1, 32, 1, 128]
NUM_LAYERS = [28, 32, 24, 64, 94]
VOCAB_SIZE = [151936, 128256, 201088, 151936, 151936]
S = 2048
GPU = ["4090", "A800", "5090", "H100"]
ROI = [10322, 9750, 6547, 15461]
MODEL_LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1))]

def operational_intensity_dense(batch, seq_len, hidden_size, intermediate_size, GQA_ratio):
    numerator = (
        4 * batch * seq_len * hidden_size**2
        + 4 * batch * seq_len * hidden_size**2 * GQA_ratio
        + 4 * batch * seq_len**2 * hidden_size
        + 6 * batch * seq_len * intermediate_size * hidden_size
    )
    denominator = (
        4 * hidden_size**2
        + 4 * hidden_size**2 * GQA_ratio
        + 6 * intermediate_size * hidden_size
        + 2 * batch * seq_len * hidden_size
    )
    return numerator / denominator


def operational_intensity_moe(
    batch, seq_len, hidden_size, intermediate_size, active_experts, total_experts, GQA_ratio
):
    numerator = (
        4 * batch * seq_len * hidden_size**2
        + 4 * batch * seq_len * hidden_size**2 * GQA_ratio
        + 4 * batch * seq_len**2 * hidden_size
        + 2 * batch * seq_len * hidden_size * total_experts
        + 6 * batch * seq_len * active_experts * intermediate_size * hidden_size
    )
    denominator = (
        4 * hidden_size**2
        + 4 * hidden_size**2 * GQA_ratio
        + 2 * hidden_size * total_experts
        + 6 * total_experts * intermediate_size * hidden_size
        + 2 * batch * seq_len * hidden_size
    )
    return numerator / denominator


def forward_flops_per_layer_dense(batch, seq_len, hidden_size, intermediate_size, GQA_ratio):
    return (
        4 * batch * seq_len * hidden_size**2
        + 4 * batch * seq_len * hidden_size**2 * GQA_ratio
        + 4 * batch * seq_len**2 * hidden_size
        + 6 * batch * seq_len * intermediate_size * hidden_size
    )


def forward_flops_per_layer_moe(
    batch, seq_len, hidden_size, intermediate_size, active_experts, total_experts, GQA_ratio
):
    return (
        4 * batch * seq_len * hidden_size**2
        + 4 * batch * seq_len * hidden_size**2 * GQA_ratio
        + 4 * batch * seq_len**2 * hidden_size
        + 2 * batch * seq_len * hidden_size * total_experts
        + 6 * batch * seq_len * active_experts * intermediate_size * hidden_size
    )


def output_projection_flops(batch, seq_len, hidden_size, vocab_size):
    # LM head matmul FLOPs: (batch * seq_len, hidden) x (hidden, vocab)
    return 2 * batch * seq_len * hidden_size * vocab_size


def _format_flops(flops):
    units = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
    value = float(flops)
    unit_idx = 0
    while value >= 1e3 and unit_idx < len(units) - 1:
        value /= 1e3
        unit_idx += 1
    return f"{value:.3f} {units[unit_idx]}"


def print_training_flops_per_sequence():
    batch = 1
    print(f"Training FLOPs per sequence (S={S}, batch={batch}, training=3x forward):")
    print("(includes transformer stack + vocab projection)")
    for idx, name in enumerate(MODEL):
        gqa_ratio = KVHEAD[idx] / HEAD[idx]
        if E[idx] == 1:
            forward_per_layer = forward_flops_per_layer_dense(batch, S, H[idx], I[idx], gqa_ratio)
        else:
            forward_per_layer = forward_flops_per_layer_moe(batch, S, H[idx], I[idx], Eact[idx], E[idx], gqa_ratio)
        transformer_forward_per_sequence = forward_per_layer * NUM_LAYERS[idx]
        vocab_forward_per_sequence = output_projection_flops(batch, S, H[idx], VOCAB_SIZE[idx])
        forward_per_sequence = transformer_forward_per_sequence + vocab_forward_per_sequence
        training_per_sequence = 3 * forward_per_sequence
        print(
            f"- {name}: forward={_format_flops(forward_per_sequence)}, "
            f"training={_format_flops(training_per_sequence)}"
        )


def plot_oi(batch_sizes, y_range):
    plt.figure(figsize=(9, 6))
    for idx, name in enumerate(MODEL):
        if E[idx] == 1:
            oi = operational_intensity_dense(batch_sizes, S, H[idx], I[idx], KVHEAD[idx] / HEAD[idx])
        else:
            oi = operational_intensity_moe(batch_sizes, S, H[idx], I[idx], Eact[idx], E[idx], KVHEAD[idx] / HEAD[idx])
        plt.plot(batch_sizes, oi, label=name, linestyle=MODEL_LINESTYLES[idx % len(MODEL_LINESTYLES)], linewidth=2.0)

    max_batch = float(np.max(batch_sizes))

    # Annotate ROI thresholds as gray solid lines and place labels inside the plot.
    y_span = y_range[1] - y_range[0]
    label_y_offset = {
        "H100": 0.01 * y_span,
        "4090": 0.01 * y_span,
        "A800": -0.01 * y_span,
        "5090": -0.01 * y_span,
    }
    for gpu, roi in zip(GPU, ROI):
        line = plt.axhline(
            y=roi,
            linestyle="-",
            color="gray",
            linewidth=1.8,
            alpha=0.8,
        )
        color = line.get_color()
        text_y = roi + label_y_offset.get(gpu, 0.0)
        va = "center" if label_y_offset.get(gpu, 0.0) == 0.0 else ("bottom" if label_y_offset.get(gpu, 0.0) > 0 else "top")
        plt.text(max_batch * 0.97, text_y, gpu, va=va, ha="right", fontsize=18, color=color)

    plt.ylim(y_range)
    plt.xlim(batch_sizes.min(), max_batch)
    plt.xlabel("Batch Size", fontsize=18)
    plt.ylabel("Operational Intensity (FLOPS/Byte)", fontsize=18)
    plt.title(f"Operational Intensity vs Batch Size (S={S})", fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.legend(ncol=2, loc="upper right", fontsize=18)
    plt.tight_layout()
    plt.savefig("operational_intensity.pdf", dpi=300)


if __name__ == "__main__":
    batch_range = np.arange(1, 257)
    y_range = (0, 30000)
    print_training_flops_per_sequence()
    plot_oi(batch_range, y_range)