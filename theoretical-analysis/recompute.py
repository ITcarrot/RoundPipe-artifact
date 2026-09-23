import numpy as np
import matplotlib.pyplot as plt

MODEL = ["Qwen3-1.7B", "Llama-3.1-8B", "gpt-oss-20B", "Qwen3-32B", "Qwen3-235B"]
H = [2048, 4096, 2880, 5120, 4096]
HEAD = [16, 32, 64, 64, 64]
KVHEAD = [8, 8, 8, 8, 4]
I = [6144, 14336, 2880, 25600, 1536]
EACT = [1, 1, 4, 1, 8]
E = [1, 1, 32, 1, 128]

S = 2048
BATCH = 4
TFLOPS = 330
PCIE_GBS = 32

# Keep the two bars' style settings at the top for quick tweaking.
RECOMPUTE_COLOR = "#DDEBF7"
RECOMPUTE_HATCH = "//"
OFFLOAD_COLOR = "#F8CBAD"
OFFLOAD_HATCH = "\\\\"

FONT_SIZE_PT = 18
TITLE_SIZE_PT = 20


def format_model_label(model_name):
	# Put parameter size on the second line for cleaner x-axis labels.
	if "-" in model_name:
		name, params = model_name.rsplit("-", 1)
		return f"{name}\n{params}"
	return model_name


def recompute_flops_dense(batch, seq_len, hidden_size, intermediate_size, GQA_ratio):
	return (
        4 * batch * seq_len * hidden_size**2
        + 4 * batch * seq_len * hidden_size**2 * GQA_ratio
        + 4 * batch * seq_len**2 * hidden_size
        + 6 * batch * seq_len * intermediate_size * hidden_size
    )


def recompute_flops_moe(
	batch, seq_len, hidden_size, intermediate_size, active_experts, GQA_ratio
):
	return (
        4 * batch * seq_len * hidden_size**2
        + 4 * batch * seq_len * hidden_size**2 * GQA_ratio
        + 4 * batch * seq_len**2 * hidden_size
        + 6 * batch * seq_len * active_experts * intermediate_size * hidden_size
	)


def activation_bytes_single_direction(batch, seq_len, hidden_size, intermediate_size, head, kv_head, active_experts):
	intermediate_effective = intermediate_size * active_experts
	return (
		# 4 * head * (seq_len**2) * batch
		+ (12 + 4 * kv_head / head) * seq_len * batch * hidden_size
		+ 6 * seq_len * batch * intermediate_effective
	)


def compute_times_ms():
	recompute_ms = []
	offload_ms = []

	for idx, _ in enumerate(MODEL):
		if E[idx] == 1:
			flops = recompute_flops_dense(BATCH, S, H[idx], I[idx], KVHEAD[idx] / HEAD[idx])
		else:
			flops = recompute_flops_moe(BATCH, S, H[idx], I[idx], EACT[idx], KVHEAD[idx] / HEAD[idx])

		act_bytes = activation_bytes_single_direction(
			BATCH,
			S,
			H[idx],
			I[idx],
			HEAD[idx],
			KVHEAD[idx],
			EACT[idx],
		)

		recompute_time_s = flops / (TFLOPS * 1e12)
		offload_time_s = act_bytes / (PCIE_GBS * 1e9)

		recompute_ms.append(recompute_time_s * 1e3)
		offload_ms.append(offload_time_s * 1e3)
		print(f"{MODEL[idx]}: {offload_ms[-1] / recompute_ms[-1]:.2f}x offload time vs recompute time")

	return np.array(recompute_ms), np.array(offload_ms)


def plot_times():
	recompute_ms, offload_ms = compute_times_ms()

	x = np.arange(len(MODEL))
	width = 0.30
	offset = 0.20

	plt.figure(figsize=(10, 4))
	plt.bar(
		x - offset,
		recompute_ms,
		width=width,
		label="Recompute Time",
		color=RECOMPUTE_COLOR,
		hatch=RECOMPUTE_HATCH,
	)
	plt.bar(
		x + offset,
		offload_ms,
		width=width,
		label="Activation Reload Time",
		color=OFFLOAD_COLOR,
		hatch=OFFLOAD_HATCH,
	)

	x_labels = [format_model_label(m) for m in MODEL]
	plt.xticks(x, x_labels, rotation=0, fontsize=FONT_SIZE_PT)
	plt.yticks(fontsize=FONT_SIZE_PT)
	plt.ylabel("Time (ms)", fontsize=FONT_SIZE_PT)
	plt.title(
		f"Recompute vs Activation Reload Time on 4090",
		fontsize=TITLE_SIZE_PT,
	)
	plt.grid(axis="y", linestyle="--", linewidth=0.6)
	plt.legend(fontsize=FONT_SIZE_PT)
	plt.tight_layout()
	plt.savefig("recompute_vs_offload_time.pdf", dpi=300)


if __name__ == "__main__":
	plot_times()
