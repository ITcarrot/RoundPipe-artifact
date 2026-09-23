#!/usr/bin/env python3
"""Plot training cost (USD per billion tokens) on 8x RTX 4090 and 8x A800.

cost = hourly price of the 8-GPU server / (throughput tok/s * 3600) * 1e9

For each platform and model we compare RoundPipe, RoundPipe-sync and the best
(highest-throughput, i.e. cheapest) of the other systems measured on that
platform. Lower is better.
"""

import csv
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from config import COLORS

matplotlib.rcParams.update({
	'font.family': 'serif',
	'font.size': 11,
	'axes.labelsize': 12,
	'legend.fontsize': 9.5,
	'figure.dpi': 300,
	'savefig.bbox': 'tight',
	'savefig.pad_inches': 0.02,
	'hatch.linewidth': 0.6,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'raw-data.csv')
FIG_PATH = os.path.join(SCRIPT_DIR, 'cost_efficiency.pdf')

# Rental price of one 8-GPU server, USD per hour.
PRICE_PER_HOUR = {'4090': 2.32, 'A800': 8.16}
PLATFORMS = ['4090', 'A800']
PLATFORM_LABELS = {'4090': '8×RTX 4090', 'A800': '8×A800'}
PLATFORM_HATCH = {'4090': '', 'A800': '////'}

MODEL_KEYS = ['qwen3-1.7b', 'llama3.1-8b', 'gpt-oss-20b', 'qwen3-32b', 'qwen3-235b-lora']
MODELS = ['Qwen3-1.7B', 'LLaMA-3.1-8B', 'GPT-OSS-20B', 'Qwen3-32B', 'Qwen3-235B-LoRA']

OURS = ['RoundPipe', 'RoundPipe-sync']
BASELINE_NAMES = {
	'Deepspeed-Zero2': 'ZeRO-2',
	'Deepspeed-Zero-inf': 'ZeRO-Inf',
	'FSDP-Zero3': 'FSDP',
	'Megatron-PP': 'Megatron-PP',
	'Megatron-TP': 'Megatron-TP',
	'Mobius': 'Mobius',
}
SERIES = ['RoundPipe', 'RoundPipe-sync', 'Best baseline']
SERIES_COLORS = {
	'RoundPipe': COLORS['RoundPipe'],
	'RoundPipe-sync': COLORS['RoundPipe-sync'],
	'Best baseline': '#9e9e9e',
}


def model_label(model):
	m = re.match(r'^(.*?)-(\d+(?:\.\d+)?B(?:-LoRA)?)$', model)
	if m:
		return f"{m.group(1)}\n{m.group(2)}"
	return model


def parse_val(s):
	s = s.strip()
	if s in ('OOM', 'N/A', ''):
		return None
	return float(s)


def load_throughput():
	"""Return {platform: {framework: [tok/s or None per model]}} from the throughput table."""
	with open(CSV_PATH, 'r', encoding='utf-8-sig') as f:
		rows = list(csv.reader(f))
	header_idx = next(i for i, r in enumerate(rows) if r and r[0].strip() == '框架')
	cols = [rows[header_idx].index(k) for k in MODEL_KEYS]
	data = {p: {} for p in PLATFORMS}
	for row in rows[header_idx + 1:]:
		if len(row) < 2 or not row[0].strip():
			break
		framework, platform = row[0].strip(), row[1].strip()
		if platform in data:
			data[platform][framework] = [parse_val(row[c]) for c in cols]
	return data


def cost_per_billion(platform, tput):
	if tput is None:
		return None
	return PRICE_PER_HOUR[platform] / (tput * 3600) * 1e9


def build_costs(tput):
	"""Return {(platform, series): [(cost, best_baseline_name) per model]}."""
	costs = {}
	for platform in PLATFORMS:
		fw = tput[platform]
		for name in OURS:
			costs[(platform, name)] = [(cost_per_billion(platform, v), None) for v in fw[name]]
		best = []
		for j in range(len(MODEL_KEYS)):
			candidates = [(fw[k][j], BASELINE_NAMES[k]) for k in BASELINE_NAMES if k in fw and fw[k][j] is not None]
			if candidates:
				v, n = max(candidates)
				best.append((cost_per_billion(platform, v), n))
			else:
				best.append((None, None))
		costs[(platform, 'Best baseline')] = best
	return costs


def print_table(costs):
	print(f"{'model':<16}" + ''.join(f"{p + ' ' + s:>26}" for p in PLATFORMS for s in SERIES))
	for j, model in enumerate(MODELS):
		cells = []
		for p in PLATFORMS:
			for s in SERIES:
				c, n = costs[(p, s)][j]
				txt = 'OOM' if c is None else f'{c:.1f}' + (f' ({n})' if n else '')
				cells.append(f'{txt:>26}')
		print(f'{model:<16}' + ''.join(cells))
	print('(USD per billion tokens; lower is better)')


# Models split across two panels, each with its own linear y axis, so the
# small models are not flattened by the 32B / 235B costs.
PANELS = [
	{'models': [0, 1, 2], 'ylim': 160, 'side_label_last': True},
	{'models': [3, 4], 'ylim': 2200, 'side_label_last': True},
]


def draw_panel(ax, costs, model_idx, ylim, side_label_last=False):
	bars = [(p, s) for p in PLATFORMS for s in SERIES]
	n_bars = len(bars)
	x = np.arange(len(model_idx))
	width = 0.8 / n_bars
	for i, (platform, series) in enumerate(bars):
		# Small gap between the two platform halves of a group.
		offset = (i - n_bars / 2 + 0.5) * width + (0.02 if platform == 'A800' else -0.02)
		for k, j in enumerate(model_idx):
			cost, best_name = costs[(platform, series)][j]
			xpos = x[k] + offset
			if cost is None:
				ax.annotate('OOM', (xpos, 0), xytext=(2, 3), textcoords='offset points',
							ha='center', va='bottom', fontsize=9.5, rotation=90)
				continue
			ax.bar(xpos, cost, width * 0.92,
				   color=SERIES_COLORS[series], hatch=PLATFORM_HATCH[platform],
				   edgecolor='black', linewidth=0.4)
			if best_name and side_label_last and k == len(model_idx) - 1 and i == n_bars - 1:
				# Tallest bar of the panel: label runs down its right side so the
				# y axis does not need headroom for it.
				ax.annotate(best_name, (xpos + width * 0.46, cost), xytext=(1, 0),
							textcoords='offset points', ha='left', va='top',
							fontsize=9.5, rotation=90)
			elif best_name:
				ax.annotate(best_name, (xpos, cost), xytext=(2, 3), textcoords='offset points',
							ha='center', va='bottom', fontsize=9.5, rotation=90)
	ax.set_xlim(-0.5, len(model_idx) - 0.5 + (0.08 if side_label_last else 0))
	ax.set_ylim(0, ylim)
	ax.set_xticks(x)
	ax.set_xticklabels([model_label(MODELS[j]) for j in model_idx], fontsize=9.5)
	ax.grid(axis='y', alpha=0.3, linewidth=0.5)
	ax.set_axisbelow(True)


def main():
	costs = build_costs(load_throughput())
	print_table(costs)

	fig, axes = plt.subplots(
		1, len(PANELS), figsize=(7, 3.4),
		gridspec_kw={'width_ratios': [len(p['models']) for p in PANELS], 'wspace': 0.06},
	)
	for ax, panel in zip(axes, PANELS):
		draw_panel(ax, costs, panel['models'], panel['ylim'], panel.get('side_label_last', False))
	axes[0].set_ylabel('USD / Billion Tokens')
	# Right panel has its own scale; put its ticks on the outer edge.
	axes[-1].yaxis.tick_right()

	handles = [Patch(facecolor=SERIES_COLORS[s], edgecolor='black', linewidth=0.4, label=s) for s in SERIES]
	handles += [Patch(facecolor='white', edgecolor='black', linewidth=0.4, hatch=PLATFORM_HATCH[p],
					  label=f'{PLATFORM_LABELS[p]}') for p in PLATFORMS]
	# Blank slot so the columns read: ours | baseline | platforms.
	handles.insert(3, Patch(visible=False, label=' '))
	fig.subplots_adjust(top=0.76)
	fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.77),
			   frameon=False, columnspacing=1.0, handlelength=1.8)
	fig.suptitle('Training Cost', fontsize=13, y=0.955)

	fig.savefig(FIG_PATH)
	plt.close(fig)
	print(f'Saved: {FIG_PATH}')


if __name__ == '__main__':
	main()
