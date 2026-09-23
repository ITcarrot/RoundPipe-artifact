import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL = ["Qwen3-1.7B", "Llama-3.1-8B", "gpt-oss-20B", "Qwen3-32B", "Qwen3-235B"]
LAYERS = [28, 32, 24, 64, 94]
MODEL_KEYS = ["1.7b", "8b", "20b", "32b", "235b"]
TARGET_SCHEDULES = ["interleaved-1f1b", "looped-bfs"]

# Keep the two real bars' style definitions at the top for quick tuning.
INTERLEAVED_COLOR = "#DDEBF7"
INTERLEAVED_HATCH = "//"
LOOPED_BFS_COLOR = "#F8CBAD"
LOOPED_BFS_HATCH = "\\\\"

FONT_SIZE_PT = 18
TITLE_SIZE_PT = 20

def bubble_ideal(layer, microbatch):
    stage_per_dev = math.ceil(layer / 8)
    return 7 / (stage_per_dev * microbatch + 7)


def parse_log(log_path: Path):
    lines = [line.strip() for line in log_path.read_text().splitlines() if line.strip()]

    records = []
    current = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "scheduler":
            if current:
                records.append(current)
                current = {}
            current["scheduler"] = value
        elif key == "output":
            current["output"] = value
        elif key == "bubble_ratio":
            match = re.match(r"([0-9.]+)%", value)
            if not match:
                raise ValueError(f"Invalid bubble_ratio format: {value}")
            current["bubble_ratio"] = float(match.group(1)) / 100.0

    if current:
        records.append(current)

    data = {}
    for record in records:
        if "scheduler" not in record or "output" not in record or "bubble_ratio" not in record:
            continue
        model_key = record["output"].split("_", 1)[0]
        scheduler = record["scheduler"]
        data.setdefault(model_key, {})[scheduler] = record["bubble_ratio"]

    return data


def plot_compare(log_path: Path, output_path: Path, microbatch: int):
    data = parse_log(log_path)

    ideal = [bubble_ideal(layer, microbatch) for layer in LAYERS]
    interleaved = [data[key]["interleaved-1f1b"] for key in MODEL_KEYS]
    looped_bfs = [data[key]["looped-bfs"] for key in MODEL_KEYS]

    x = np.arange(len(MODEL))
    width = 0.18
    offset = 0.22

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - offset, [v * 100 for v in ideal], width=width, label="ideal", color="#C6E0B4")
    ax.bar(
        x,
        [v * 100 for v in interleaved],
        width=width,
        label="interleaved-1f1b",
        color=INTERLEAVED_COLOR,
        hatch=INTERLEAVED_HATCH,
    )
    ax.bar(
        x + offset,
        [v * 100 for v in looped_bfs],
        width=width,
        label="looped-bfs",
        color=LOOPED_BFS_COLOR,
        hatch=LOOPED_BFS_HATCH,
    )

    ax.set_ylabel("Bubble Ratio (%)", fontsize=FONT_SIZE_PT)
    ax.set_title(f"Ideal vs Real Bubble Ratio", fontsize=TITLE_SIZE_PT)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL, fontsize=FONT_SIZE_PT)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_PT)
    ax.legend(fontsize=FONT_SIZE_PT)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    plot_compare(
        log_path=Path("log"),
        output_path=Path("bubble_ideal_compare.pdf"),
        microbatch=16,
    )
    print("Saved chart to: bubble_ideal_compare.pdf")

