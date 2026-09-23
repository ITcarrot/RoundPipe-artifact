#!/usr/bin/env python3
"""Figure 16: pipeline bubble ratio broken down into fill / switch / drain / intra.

Reads the simulator log written by `run.sh` (one `key=value` per line, records
separated by `scheduler=`). Each bar is one schedule on one model, stacked by
bubble category; the categories sum to the schedule's total bubble ratio.

RoundPipe appears twice:
  * RoundPipe-Sync: one isolated iteration, i.e. the simulator's full bubble.
  * RoundPipe: with asynchronous optimizer updates the next iteration starts
    while the current one drains, so the fill and drain bubbles overlap with
    neighbouring iterations. Only switch + intra remain.

Exposed communication is a slice of the bubble that overlaps the four
categories, so it is not stacked; it is printed in the summary table instead.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SCHEDULE_PLOT_ORDER = [
	"gpipe",
	"1f1b",
	"interleaved-1f1b",
	"looped-bfs",
	"roundpipe-sync",
	"roundpipe",
]
# One-letter names printed under every bar; the caption spells them out.
SCHEDULE_SHORT_NAMES = {
	"gpipe": "G",
	"1f1b": "1",
	"interleaved-1f1b": "I",
	"looped-bfs": "B",
	"roundpipe-sync": "S",
	"roundpipe": "R",
}
SCHEDULE_DISPLAY_NAMES = {
	"gpipe": "GPipe",
	"1f1b": "1F1B",
	"interleaved-1f1b": "Interleaved-1F1B",
	"looped-bfs": "Looped-BFS",
	"roundpipe-sync": "RoundPipe-Sync",
	"roundpipe": "RoundPipe",
}
MODEL_PLOT_ORDER = ["1.7b", "8b", "20b", "32b", "235b"]
MODEL_DISPLAY_NAMES = ["Qwen3-1.7B", "Llama-3.1-8B", "GPT-OSS-20B", "Qwen3-32B", "Qwen3-235B"]

# Stack order, bottom to top. Intra sits on the baseline so the steady-state
# bubble is directly comparable across every bar, including async RoundPipe.
CATEGORY_ORDER = ["intra", "switch", "fill", "drain"]
CATEGORY_DISPLAY_NAMES = {
	"intra": "Intra (steady state)",
	"switch": "Switch (fwd→bwd)",
	"fill": "Fill",
	"drain": "Drain",
}
# Categorical slots 1-4 of the validated default palette, plus a hatch as a
# second cue for grayscale print and color-vision deficiency.
CATEGORY_STYLES = {
	"intra": {"color": "#2a78d6", "hatch": ""},
	"switch": {"color": "#eb6834", "hatch": "//"},
	"fill": {"color": "#1baf7a", "hatch": ".."},
	"drain": {"color": "#eda100", "hatch": "\\\\"},
}
# Categories hidden by overlapping consecutive iterations in async RoundPipe.
ASYNC_OVERLAPPED = {"fill", "drain"}

PERCENT_KEYS = {
	"bubble_ratio": "total",
	"fill_bubble_ratio": "fill",
	"switch_bubble_ratio": "switch",
	"drain_bubble_ratio": "drain",
	"intra_bubble_ratio": "intra",
	"comm_exposed_ratio": "comm",
}


def _parse_percent(key: str, value: str) -> float:
	match = re.fullmatch(r"(-?[0-9.]+)%", value)
	if not match:
		raise ValueError(f"Invalid {key} format: {value}")
	return float(match.group(1))


def parse_log(log_path: Path):
	"""Return {model: {schedule: {category: percent, 'total': ..., 'comm': ...}}}."""
	records = []
	current = {}
	for line in log_path.read_text().splitlines():
		line = line.strip()
		if "=" not in line:
			continue
		key, value = (part.strip() for part in line.split("=", 1))
		if key == "scheduler":
			if current:
				records.append(current)
			current = {"scheduler": value}
		elif key == "output":
			current["output"] = value
		elif key in PERCENT_KEYS:
			current[PERCENT_KEYS[key]] = _parse_percent(key, value)
	if current:
		records.append(current)

	data = {}
	for record in records:
		missing = [k for k in ("scheduler", "output", "total", *CATEGORY_ORDER) if k not in record]
		if missing:
			raise ValueError(f"Log record {record.get('output', '?')} lacks {missing}; rerun run.sh")
		model = Path(record["output"]).name.split("_", 1)[0]
		breakdown = {k: record[k] for k in ("total", *CATEGORY_ORDER)}
		breakdown["comm"] = record.get("comm", 0.0)

		if record["scheduler"] == "roundpipe":
			data.setdefault(model, {})["roundpipe-sync"] = breakdown
			async_breakdown = dict(breakdown)
			for category in ASYNC_OVERLAPPED:
				async_breakdown[category] = 0.0
			async_breakdown["total"] = sum(async_breakdown[c] for c in CATEGORY_ORDER)
			data[model]["roundpipe"] = async_breakdown
		else:
			data.setdefault(model, {})[record["scheduler"]] = breakdown

	return data


def print_summary(data, models):
	header = f"{'model':<6} {'schedule':<17}" + "".join(f"{c:>8}" for c in (*CATEGORY_ORDER, "total", "comm"))
	print(header)
	print("-" * len(header))
	for model in models:
		for scheduler in SCHEDULE_PLOT_ORDER:
			row = data[model][scheduler]
			values = "".join(f"{row[c]:8.2f}" for c in (*CATEGORY_ORDER, "total", "comm"))
			print(f"{model:<6} {SCHEDULE_DISPLAY_NAMES[scheduler]:<17}{values}")
	print("(percent of makespan x num_gpus; comm = exposed communication, already inside the other columns)")


def plot_bars(data, output_path: Path):
	plt.rcParams.update({"font.size": 20, "hatch.linewidth": 0.8})

	def format_model_label(label: str) -> str:
		parts = label.rsplit("-", 1)
		if len(parts) == 2:
			return f"{parts[0]}\n{parts[1]}"
		return label

	available_models = [model for model in MODEL_PLOT_ORDER if model in data]
	if not available_models:
		raise ValueError("No model data found for plotting.")
	for model in available_models:
		for scheduler in SCHEDULE_PLOT_ORDER:
			if scheduler not in data[model]:
				raise ValueError(f"Missing data for model={model}, scheduler={scheduler}")

	x = np.arange(len(available_models))
	n_schedules = len(SCHEDULE_PLOT_ORDER)
	width = 0.145
	bar_step = width + 0.012

	fig, ax = plt.subplots(figsize=(11, 5.8))
	ax.set_axisbelow(True)
	ax.grid(axis="y", alpha=0.25)

	bar_positions = []
	bar_labels = []
	for idx, scheduler in enumerate(SCHEDULE_PLOT_ORDER):
		offset = (idx - (n_schedules - 1) / 2) * bar_step
		positions = x + offset
		bottoms = np.zeros(len(available_models))
		for category in CATEGORY_ORDER:
			values = np.array([data[model][scheduler][category] for model in available_models])
			style = CATEGORY_STYLES[category]
			# White edges give the 2px-style gap between stacked segments and
			# draw the hatch in white so it stays subtle on the fill color.
			ax.bar(
				positions,
				values,
				width=width,
				bottom=bottoms,
				color=style["color"],
				hatch=style["hatch"],
				edgecolor="white",
				linewidth=0.6,
			)
			bottoms += values
		bar_positions.extend(positions)
		bar_labels.extend([SCHEDULE_SHORT_NAMES[scheduler]] * len(available_models))

	ax.set_ylabel("Bubble Ratio (%)")
	ax.set_ylim(0, None)
	ax.set_xlim(x[0] - 0.49, x[-1] + 0.49)
	ax.spines[["top", "right"]].set_visible(False)

	# Two-level x axis: schedule under each bar, model under each group.
	ax.set_xticks(bar_positions)
	ax.set_xticklabels(bar_labels, rotation=0, fontsize=20, va="top", linespacing=0.9)
	ax.tick_params(axis="x", length=0, pad=3)
	model_label_map = dict(zip(MODEL_PLOT_ORDER, MODEL_DISPLAY_NAMES))
	for xi, model in zip(x, available_models):
		ax.annotate(
			format_model_label(model_label_map.get(model, model)),
			xy=(xi, 0),
			xycoords=("data", "axes fraction"),
			xytext=(0, -28),
			textcoords="offset points",
			ha="center",
			va="top",
			fontsize=20,
		)

	handles = [
		Patch(
			facecolor=CATEGORY_STYLES[c]["color"],
			hatch=CATEGORY_STYLES[c]["hatch"],
			edgecolor="white",
			label=CATEGORY_DISPLAY_NAMES[c],
		)
		for c in reversed(CATEGORY_ORDER)
	]
	ax.legend(
		handles=handles[::-1],  # one row, left to right in stack order
		loc="lower center",
		bbox_to_anchor=(0.5, 1.0),
		ncol=len(handles),
		fontsize=20,
		frameon=False,
		handlelength=1.4,
		columnspacing=0.8, handletextpad=0.4,
	)

	fig.suptitle("Bubble Ratio Breakdown by Model and Schedule", fontsize=22, y=0.985)
	fig.subplots_adjust(left=0.085, right=0.99, top=0.815, bottom=0.22)
	fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
	plt.close(fig)
	return available_models


def main():
	parser = argparse.ArgumentParser(description="Plot bubble breakdown bar chart from simulator log.")
	parser.add_argument("--log", type=Path, default=Path("log"), help="Path to the simulator log file")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("bubble_bar.pdf"),
		help="Path to save bar chart image",
	)
	args = parser.parse_args()

	data = parse_log(args.log)
	models = plot_bars(data, args.output)
	print_summary(data, models)
	print(f"Saved chart to: {args.output}")


if __name__ == "__main__":
	main()
