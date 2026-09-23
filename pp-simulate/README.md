# PP Pipeline Simulator (1F1B / GPipe / Interleaved-1F1B / Looped-BFS / RoundPipe)

This directory provides a pipeline-parallel (PP) simulator for large model training. Currently implemented schedulers:

- `1f1b`
- `gpipe`
- `interleaved-1f1b`
- `looped-bfs`
- `roundpipe`

## Parameters

- Number of GPUs: `--num-gpus`
- Number of microbatches: `--num-microbatches`
- Stage partition: `--stage-partition`
  - `1f1b` / `gpipe` / `interleaved-1f1b` / `looped-bfs`: comma-separated list giving the number of transformer layers per **virtual stage**
  - `roundpipe`: `fwd0,fwd1,...|bwd0,bwd1,...`, specifying forward and backward stage partitions simultaneously
- Total transformer layers: `--num-layers`
- Forward time per layer per microbatch: `--fwd-time-per-layer`
- Backward time per layer per microbatch: `--bwd-time-per-layer`
- LM head forward time: `--lmhead-fwd-time`
  - `1f1b` / `gpipe` / `interleaved-1f1b` / `looped-bfs`: applied to the last stage
  - `roundpipe`: executed as a pre-step before backward in the first backward stage
- LM head backward time: `--lmhead-bwd-time` (in `roundpipe`, applied to the first backward stage)
- Point-to-point communication time between adjacent stages: `--comm-time` (default `0`)
  - Charged only when adjacent stages reside on different GPUs; intra-GPU handoffs (e.g. activations saved from forward to backward within the same stage) cost 0
  - Charged edges match the simulator's existing dependency edges: forward stage `s-1 → s`, backward stage `s+1 → s`; RoundPipe's backward also depends on the last forward stage's output

Implementation assumptions:

- In `roundpipe`, all backward stages except the one containing the LM head recompute their transformer layers before backward.
- The output diagram uses three color categories: `fwd`, `recompute`, and `bwd`.

Scheduling and stage count constraints:

- `--scheduler 1f1b`: `len(stage_partition) == num_gpus` and `num_microbatches >= num_stages`
- `--scheduler gpipe`: `len(stage_partition) == num_gpus` and `num_microbatches >= num_stages`
- `--scheduler interleaved-1f1b`: `len(stage_partition) % num_gpus == 0`
- `--scheduler looped-bfs`: `len(stage_partition) % num_gpus == 0`
- `--scheduler roundpipe`: `stage_partition` must be `fwd|bwd` format, with `sum(fwd) == sum(bwd) == num_layers`

### GPipe semantics

- All microbatches complete their forward pass across all stages before any backward pass begins.

### Interleaved-1F1B semantics

- Scheduling logic aligns with PyTorch `ScheduleInterleaved1F1B` (v2.10.0): per-rank compute actions are generated first, then lowered via `send/recv`, and finally advanced by runtime timestep.
- This preserves the same bubble/wait behavior as PyTorch (rather than doing global greedy scheduling by earliest-available time).

### Looped-BFS semantics

- Each GPU processes all microbatches for one virtual stage before moving to the next (breadth-first), looping over all its local virtual stages for forward, then backward in reverse order.

### RoundPipe scheduling semantics

- Forward and backward use separate stage partitions; forward and backward stages need not reside on the same GPU.
- Stages are mapped to GPUs via round-robin (`stage_idx % num_gpus`).
- The backward stage round-robin offset follows immediately after the forward stages (`(num_fwd_stages + bwd_stage_idx) % num_gpus`).
- Within a single backward stage, execution order is: recompute all layers in the stage first, then execute backward for all layers in the stage.
- Recompute placement is backward-aligned: recompute does not need to wait for the previous backward stage to complete — it only requires that backward dependencies are satisfied and the GPU is idle during recompute.
- The backward stage containing the LM head does not recompute transformer layers, but executes LM head forward first, then backward for both the LM head and stage layers.
- Microbatch dispatch proceeds in rounds: each round dispatches up to `num_gpus` microbatches, across multiple rounds until all are dispatched.
- Within each round, forward and backward operations complete in stage order; the next round begins dispatch from the first available GPU.
- Only per-microbatch stage execution order dependencies are enforced.

## Usage

```bash
cd pp-simulate
python simulator.py \
  --scheduler 1f1b \
  --num-gpus 4 \
  --num-microbatches 8 \
  --stage-partition 8,8,8,8 \
  --num-layers 32 \
  --fwd-time-per-layer 1.0 \
  --bwd-time-per-layer 1.5 \
  --lmhead-fwd-time 2.0 \
  --lmhead-bwd-time 3.0 \
  --output schedule_1f1b.png

python simulator.py \
  --scheduler interleaved-1f1b \
  --num-gpus 4 \
  --num-microbatches 8 \
  --stage-partition 4,4,4,4,4,4,4,4 \
  --num-layers 32 \
  --fwd-time-per-layer 1.0 \
  --bwd-time-per-layer 1.5 \
  --lmhead-fwd-time 2.0 \
  --lmhead-bwd-time 3.0 \
  --output schedule_interleaved_1f1b.png

python simulator.py \
  --scheduler roundpipe \
  --num-gpus 4 \
  --num-microbatches 8 \
  --stage-partition '8,8,8,8|8,8,8,8' \
  --num-layers 32 \
  --fwd-time-per-layer 1.0 \
  --bwd-time-per-layer 1.5 \
  --lmhead-fwd-time 2.0 \
  --lmhead-bwd-time 3.0 \
  --output schedule_roundpipe.png
```

Or simply:

```bash
bash run.sh
```

In the generated diagrams:

- X-axis: time
- Y-axis: GPU (stage)
- Colored blocks: computation phases (`fwd` / `recompute` / `bwd`)

### Statistics output

`run.sh` automatically writes statistics to a `log` file (change the filename with `LOG=other_name bash run.sh`; progress messages go to stderr and are not mixed into the log):

- `makespan` / `bubble_time` / `bubble_ratio`: total time and total bubble (denominator is `makespan × num_gpus`)
- Bubble is decomposed into four categories whose sum equals `bubble_time` / `bubble_ratio`:
  - `fill_bubble_*`: idle time on each GPU from `t=0` to its first operation
  - `switch_bubble_*`: idle time on each GPU from the end of warmup forward (the `num_gpus - gpu_id`-th forward) to its first backward, i.e. waiting for gradient propagation. Forward/recompute operations scheduled into this window count as busy time, not bubble. RoundPipe has no forward/backward phase switch (`has_warmup_phase = False`), so the window starts at the last forward before the first backward, making this term 0
  - `drain_bubble_*`: idle time on each GPU from its last backward to makespan
  - `intra_bubble_*`: total bubble minus the above three, i.e. steady-state bubble
- Communication statistics (reported separately; communication is a slice of the four bubble categories, not a fifth):
  - `comm_time_per_hop`: per-hop communication time
  - `comm_total_time`: sum of all inter-GPU transfer times
  - `comm_exposed_time` / `comm_exposed_ratio`: the portion that actually stalls the pipeline, i.e. idle time on a GPU caused by waiting for a transfer. Transfers that overlap with ongoing computation on that GPU are fully hidden and count as 0

### Debug output

- `--print-schedule`: print operation-level events (gpu / virtual-stage / kind / microbatch / time interval)
- `--print-schedule-limit N`: print at most the first `N` events; `N<=0` prints all

## Plotting scripts

### `plot_bubble.py` — Bubble ratio breakdown chart

Generates a grouped bar chart (saved as `bubble_bar.pdf` by default) showing the pipeline bubble ratio broken down into four categories: fill, switch, drain, and intra (steady state). Each group represents a model, and each bar within a group represents a scheduler.

RoundPipe appears in two variants:
- **RoundPipe-Sync**: one isolated iteration (the simulator's full bubble)
- **RoundPipe**: with asynchronous optimizer updates, the next iteration starts while the current one drains, so fill and drain bubbles overlap with neighbouring iterations — only switch and intra remain

```bash
python plot_bubble.py --log log --output bubble_bar.pdf
```

### `plot_ideal.py` — Ideal vs real bubble ratio comparison

Generates a grouped bar chart (saved as `bubble_ideal_compare.pdf` by default) comparing the ideal (theoretical minimum) bubble ratio against the simulated bubble ratios for Interleaved-1F1B and Looped-BFS across multiple models. The ideal bubble ratio is computed as `(p-1) / (stages_per_device × microbatches + p-1)` where `p` is the number of GPUs.

```bash
python plot_ideal.py
```

## Extensibility

- Schedulers use a registry pattern; current implementations are `1f1b`, `gpipe`, `interleaved-1f1b`, `looped-bfs`, and `roundpipe`.
- New schedulers can be added to `simulator.py` and registered in `SCHEDULER_REGISTRY`, reusing the same input parsing and plotting pipeline.
