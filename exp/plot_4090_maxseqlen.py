#!/usr/bin/env python3
"""Plot maximum supported sequence length on 8x RTX 4090."""

import csv
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
from config import COLORS, HATCHES

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 9.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'raw-data.csv')
FIG_PATH = os.path.join(SCRIPT_DIR, '4090_maxseqlen.pdf')

MODELS = ['Qwen3-1.7B', 'LLaMA-3.1-8B', 'GPT-OSS-20B', 'Qwen3-32B', 'Qwen3-235B-LoRA']


def model_label(model):
    """Split model string into two-line label: name and parameter scale."""
    m = re.match(r'^(.*?)-(\d+(?:\.\d+)?B(?:-LoRA)?)$', model)
    if m:
        return f"{m.group(1)}\n{m.group(2)}"
    return model

# Row indices (0-based) and display names for 4090 frameworks
# Seqlen columns are 9-13 (0-based)
FRAMEWORK_ROWS_4090 = [
    (1,  'ZeRO-2'),
    (3,  'FSDP'),
    (2,  'ZeRO-Infinity'),
    (5,  'Megatron-TP'),
    (4,  'Megatron-PP'),
    (6,  'Mobius'),
    (7,  'RoundPipe'),
]

def parse_seqlen(s):
    """Parse a seqlen value like '7k', '73k'; return None for OOM/N/A."""
    s = s.strip().lower()
    if s in ('oom', 'n/a', ''):
        return None
    m = re.match(r'^([\d.]+)k$', s)
    if m:
        return float(m.group(1))
    return float(s)


def main():
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))

    frameworks = []
    data = {}
    for row_idx, name in FRAMEWORK_ROWS_4090:
        row = rows[row_idx]
        vals = [parse_seqlen(row[c]) for c in range(9, 14)]
        frameworks.append(name)
        data[name] = vals

    n_models = len(MODELS)
    n_fw = len(frameworks)
    x = np.arange(n_models)
    width = 0.8 / n_fw

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, fw in enumerate(frameworks):
        vals = data[fw]
        bar_vals = [v if v is not None else 0 for v in vals]
        offset = (i - n_fw / 2 + 0.5) * width
        bars = ax.bar(x + offset, bar_vals, width * 0.92,
                      label=fw, color=COLORS.get(fw, '#999'),
                      hatch=HATCHES.get(fw, ''),
                      edgecolor='black', linewidth=0.4)
        for j, v in enumerate(vals):
            if v is None:
                tag = rows[FRAMEWORK_ROWS_4090[i][0]][9 + j].strip()
                ax.text(x[j] + offset, 0.3, tag,
                        ha='center', va='bottom', fontsize=7,
                        rotation=90, color='black')

    ax.set_ylabel('Max Sequence Length')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}K'))
    ax.set_xticks(x)
    ax.set_xticklabels([model_label(m) for m in MODELS], fontsize=9.5, rotation=0, ha='center')
    ax.set_xlim(-0.5, n_models - 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(ncol=3, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title('Max Sequence Length on 8×RTX 4090', fontsize=13)

    fig.savefig(FIG_PATH)
    plt.close(fig)
    print(f'Saved: {FIG_PATH}')


if __name__ == '__main__':
    main()
