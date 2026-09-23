#!/usr/bin/env python3
"""Plot RoundPipe throughput scaling from 1 to 8 RTX 4090 GPUs."""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'raw-data.csv')
FIG_PATH = os.path.join(SCRIPT_DIR, 'scalability.pdf')

MODEL_NAMES = ['Qwen3-1.7B', 'LLaMA-3.1-8B', 'GPT-OSS-20B', 'Qwen3-32B', 'Qwen3-235B-LoRA']
MARKERS = ['o', 's', '^', 'D', 'v']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

SCALE_ROW_START = 17  # GPU=0 (0-based index)
SCALE_ROW_END = 25    # GPU=8 (inclusive, 0-based)


def main():
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))

    gpu_counts = []
    throughputs = {m: [] for m in MODEL_NAMES}

    for row_idx in range(SCALE_ROW_START, SCALE_ROW_END + 1):
        row = rows[row_idx]
        n_gpu = int(row[1].strip())
        if n_gpu == 0:
            continue
        gpu_counts.append(n_gpu)
        for mi, m in enumerate(MODEL_NAMES):
            throughputs[m].append(float(row[2 + mi].strip()))

    gpu_counts = np.array(gpu_counts)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for mi, m in enumerate(MODEL_NAMES):
        vals = np.array(throughputs[m])
        ax.plot(gpu_counts, vals, marker=MARKERS[mi], color=COLORS[mi],
                label=m, markersize=5, linewidth=1.5)
        # Subtle cue: connect each curve's first point to origin.
        ax.plot([0, gpu_counts[0]], [0, vals[0]], '--', color=COLORS[mi],
            alpha=0.4, linewidth=1)

    ax.set_xlabel('Number of RTX 4090 GPUs')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_xlim(left=0)
    ax.set_xticks(np.insert(gpu_counts, 0, 0))
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    fig.savefig(FIG_PATH)
    plt.close(fig)
    print(f'Saved: {FIG_PATH}')


if __name__ == '__main__':
    main()
