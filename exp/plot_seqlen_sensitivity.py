#!/usr/bin/env python3
"""Plot throughput vs sequence length for Qwen3-1.7B on 8x RTX 4090."""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import ScalarFormatter

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 13,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'raw-data.csv')
FIG_PATH = os.path.join(SCRIPT_DIR, 'seqlen_sensitivity.pdf')

SEQLEN_HDR_ROW = 26  # 0-based
SEQLEN_DATA_ROW = 27


def main():
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        rows = list(csv.reader(f))

    hdr = rows[SEQLEN_HDR_ROW]
    data_row = rows[SEQLEN_DATA_ROW]

    seq_lengths = []
    throughputs = []
    for c in range(1, len(hdr)):
        val = hdr[c].strip()
        if not val:
            break
        seq_lengths.append(int(val))
        throughputs.append(float(data_row[c].strip()))

    seq_lengths = np.array(seq_lengths)
    throughputs = np.array(throughputs)

    fig, ax = plt.subplots(figsize=(5, 2.8))

    ax.plot(seq_lengths, throughputs, 'o-', color='#ff7f00',
            markersize=6, linewidth=2, label='Qwen3-1.7B')

    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels([f'{s//1024}K' if s >= 1024 else str(s)
                        for s in seq_lengths], fontsize=12)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title('Qwen3-1.7B on 8×RTX 4090', fontsize=14)

    fig.savefig(FIG_PATH)
    plt.close(fig)
    print(f'Saved: {FIG_PATH}')


if __name__ == '__main__':
    main()
