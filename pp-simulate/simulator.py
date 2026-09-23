from __future__ import annotations

import argparse
import colorsys
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass(frozen=True)
class SimulationConfig:
    num_gpus: int
    num_microbatches: int
    stage_partition: Tuple[int, ...]
    num_layers: int
    fwd_time_per_layer: float
    bwd_time_per_layer: float
    lmhead_fwd_time: float
    lmhead_bwd_time: float
    comm_time: float = 0.0
    backward_stage_partition: Optional[Tuple[int, ...]] = None

    @property
    def num_stages(self) -> int:
        return len(self.stage_partition)

    def validate(self) -> None:
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be > 0")
        if self.num_microbatches <= 0:
            raise ValueError("num_microbatches must be > 0")
        if self.num_layers < 0:
            raise ValueError("num_layers must be >= 0")
        if self.comm_time < 0:
            raise ValueError("comm_time must be >= 0")
        if len(self.stage_partition) == 0:
            raise ValueError("stage_partition must not be empty")
        if any(layer_count < 0 for layer_count in self.stage_partition):
            raise ValueError("stage_partition contains negative layer count")
        if sum(self.stage_partition) != self.num_layers:
            raise ValueError(
                "sum(stage_partition) must equal num_layers, "
                f"got {sum(self.stage_partition)} and {self.num_layers}"
            )
        if self.backward_stage_partition is not None:
            if len(self.backward_stage_partition) == 0:
                raise ValueError("backward_stage_partition must not be empty")
            if any(layer_count < 0 for layer_count in self.backward_stage_partition):
                raise ValueError("backward_stage_partition contains negative layer count")
            if sum(self.backward_stage_partition) != self.num_layers:
                raise ValueError(
                    "sum(backward_stage_partition) must equal num_layers, "
                    f"got {sum(self.backward_stage_partition)} and {self.num_layers}"
                )
        durations = (
            self.fwd_time_per_layer,
            self.bwd_time_per_layer,
            self.lmhead_fwd_time,
            self.lmhead_bwd_time,
        )
        if any(duration < 0 for duration in durations):
            raise ValueError("all durations must be >= 0")

    def fwd_duration(self, stage_idx: int) -> float:
        duration = self.stage_partition[stage_idx] * self.fwd_time_per_layer
        if stage_idx == self.num_stages - 1:
            duration += self.lmhead_fwd_time
        return duration

    def recompute_duration(self, stage_idx: int) -> float:
        return self.stage_partition[stage_idx] * self.fwd_time_per_layer

    def bwd_duration(self, stage_idx: int) -> float:
        duration = self.stage_partition[stage_idx] * self.bwd_time_per_layer
        if stage_idx == self.num_stages - 1:
            duration += self.lmhead_bwd_time
        return duration


@dataclass(frozen=True)
class LayerTimelineEvent:
    stage: int
    virtual_stage: int
    lane: int
    microbatch: int
    kind: str
    start: float
    end: float
    is_lmhead: bool


@dataclass(frozen=True)
class CommStats:
    """Point-to-point communication between consecutive stages.

    `total` sums every cross-GPU transfer on the modeled dependency edges;
    `exposed` is the part of it that actually stalled the pipeline, i.e. the
    extra idle a GPU incurred because a transfer had to finish before its next
    op could start. A transfer that lands while the GPU is still busy is fully
    overlapped and contributes 0. `exposed` is a slice of the total bubble, not
    a fifth bubble category, so it overlaps fill/switch/drain/intra.
    """
    comm_time: float
    total: float
    exposed: float

    @property
    def overlapped(self) -> float:
        return max(self.total - self.exposed, 0.0)


def _resolve_start(
    comm_time: float,
    gpu_idx: int,
    gpu_ready: float,
    deps: Sequence[Tuple[float, Optional[int]]],
) -> Tuple[float, float, float]:
    """Start time of an op given its dependencies, plus its comm accounting.

    `deps` are `(producer_end, producer_gpu)` pairs; a `None` gpu (or the same
    gpu) is a local hand-off and costs nothing. Returns the start time, the
    stall caused by communication, and the transfer time issued by this op.
    """
    ready = gpu_ready
    ready_without_comm = gpu_ready
    issued = 0.0
    for dep_end, dep_gpu in deps:
        hop = comm_time if dep_gpu is not None and dep_gpu != gpu_idx else 0.0
        issued += hop
        ready = max(ready, dep_end + hop)
        ready_without_comm = max(ready_without_comm, dep_end)
    return ready, max(ready - ready_without_comm, 0.0), issued


@dataclass(frozen=True)
class OperationEvent:
    gpu: int
    virtual_stage: int
    microbatch: int
    kind: str
    start: float
    end: float


class PipelineScheduler(ABC):
    name: str
    # True for schedulers with a warm-up-forward phase followed by a backward
    # phase, where each GPU runs `num_gpus - gpu_id` forwards before its first
    # backward. RoundPipe has no such phase boundary.
    has_warmup_phase: bool = True

    @abstractmethod
    def simulate(self, config: SimulationConfig) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
        raise NotImplementedError


class OneFOneBScheduler(PipelineScheduler):
    name = "1f1b"

    def simulate(self, config: SimulationConfig) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
        if (
            config.backward_stage_partition is not None
            and config.backward_stage_partition != config.stage_partition
        ):
            raise ValueError(
                "scheduler=1f1b does not support separate backward stage partition; "
                "use scheduler=roundpipe for fwd|bwd partition input"
            )
        if config.num_stages != config.num_gpus:
            raise ValueError(
                "scheduler=1f1b requires len(stage_partition) == num_gpus, "
                f"got {config.num_stages} and {config.num_gpus}. "
                "Use scheduler=interleaved-1f1b for multiple virtual stages per GPU."
            )
        if config.num_microbatches < config.num_stages:
            raise ValueError(
                "scheduler=1f1b follows PyTorch Schedule1F1B and requires "
                "num_microbatches >= num_stages, "
                f"got {config.num_microbatches} and {config.num_stages}"
            )
        return _simulate_1f1b_standard(config)


class GPipeScheduler(PipelineScheduler):
    name = "gpipe"

    def simulate(self, config: SimulationConfig) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
        if (
            config.backward_stage_partition is not None
            and config.backward_stage_partition != config.stage_partition
        ):
            raise ValueError(
                "scheduler=gpipe does not support separate backward stage partition; "
                "use scheduler=roundpipe for fwd|bwd partition input"
            )
        if config.num_stages != config.num_gpus:
            raise ValueError(
                "scheduler=gpipe requires len(stage_partition) == num_gpus, "
                f"got {config.num_stages} and {config.num_gpus}. "
                "Use scheduler=looped-bfs or scheduler=interleaved-1f1b for multiple virtual stages per GPU."
            )
        if config.num_microbatches < config.num_stages:
            raise ValueError(
                "scheduler=gpipe requires num_microbatches >= num_stages, "
                f"got {config.num_microbatches} and {config.num_stages}"
            )
        return _simulate_gpipe_standard(config)


class InterleavedOneFOneBScheduler(PipelineScheduler):
    name = "interleaved-1f1b"

    def simulate(self, config: SimulationConfig) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
        if (
            config.backward_stage_partition is not None
            and config.backward_stage_partition != config.stage_partition
        ):
            raise ValueError(
                "scheduler=interleaved-1f1b does not support separate backward stage partition; "
                "use scheduler=roundpipe for fwd|bwd partition input"
            )
        if config.num_stages % config.num_gpus != 0:
            raise ValueError(
                "scheduler=interleaved-1f1b requires len(stage_partition) to be a multiple of num_gpus, "
                f"got {config.num_stages} and {config.num_gpus}"
            )
        return _simulate_interleaved_1f1b_standard(config)


class LoopedBFSScheduler(PipelineScheduler):
    name = "looped-bfs"

    def simulate(self, config: SimulationConfig) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
        if (
            config.backward_stage_partition is not None
            and config.backward_stage_partition != config.stage_partition
        ):
            raise ValueError(
                "scheduler=looped-bfs does not support separate backward stage partition; "
                "use scheduler=roundpipe for fwd|bwd partition input"
            )
        if config.num_stages % config.num_gpus != 0:
            raise ValueError(
                "scheduler=looped-bfs requires len(stage_partition) to be a multiple of num_gpus, "
                f"got {config.num_stages} and {config.num_gpus}"
            )
        return _simulate_looped_bfs_standard(config)


class RoundPipeScheduler(PipelineScheduler):
    name = "roundpipe"
    has_warmup_phase = False

    def simulate(self, config: SimulationConfig) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
        if config.backward_stage_partition is None:
            raise ValueError(
                "scheduler=roundpipe requires dual stage partitions: "
                "--stage-partition 'fwd_stage0,...|bwd_stage0,...'"
            )
        return _simulate_roundpipe(config)


def _get_1f1b_rank_ops_single_stage_standard(
    rank: int,
    pp_group_size: int,
    n_microbatches: int,
) -> List[Optional[_PlannedAction]]:
    rank_ops: List[Optional[_PlannedAction]] = []

    warmup_chunks = min(n_microbatches, pp_group_size - rank)

    fwd_mb_index = 0
    bwd_mb_index = 0

    for _ in range(warmup_chunks):
        rank_ops.append(_PlannedAction(rank, "fwd", fwd_mb_index))
        fwd_mb_index += 1

    while True:
        rank_ops.append(_PlannedAction(rank, "bwd", bwd_mb_index))
        bwd_mb_index += 1

        if fwd_mb_index == n_microbatches:
            break

        rank_ops.append(_PlannedAction(rank, "fwd", fwd_mb_index))
        fwd_mb_index += 1

    while bwd_mb_index < n_microbatches:
        rank_ops.append(_PlannedAction(rank, "bwd", bwd_mb_index))
        bwd_mb_index += 1

    return rank_ops


def _build_1f1b_rank_plans_standard(
    config: SimulationConfig,
) -> Dict[int, List[Optional[_PlannedAction]]]:
    rank_plans: Dict[int, List[Optional[_PlannedAction]]] = {}
    for rank in range(config.num_gpus):
        rank_plans[rank] = _get_1f1b_rank_ops_single_stage_standard(
            rank=rank,
            pp_group_size=config.num_gpus,
            n_microbatches=config.num_microbatches,
        )
    return rank_plans


def _build_gpipe_rank_plans_standard(
    config: SimulationConfig,
) -> Dict[int, List[Optional[_PlannedAction]]]:
    rank_plans: Dict[int, List[Optional[_PlannedAction]]] = {}
    for rank in range(config.num_gpus):
        rank_plans[rank] = [
            _PlannedAction(rank, "fwd", microbatch_idx)
            for microbatch_idx in range(config.num_microbatches)
        ]
        rank_plans[rank].extend(
            _PlannedAction(rank, "bwd", microbatch_idx)
            for microbatch_idx in range(config.num_microbatches)
        )
    return rank_plans


def _build_looped_bfs_rank_plans_standard(
    config: SimulationConfig,
) -> Dict[int, List[Optional[_PlannedAction]]]:
    if config.num_stages % config.num_gpus != 0:
        raise ValueError(
            "looped-bfs requires len(stage_partition) to be a multiple of num_gpus"
        )

    n_local_stages = config.num_stages // config.num_gpus
    rank_plans: Dict[int, List[Optional[_PlannedAction]]] = {}

    for rank in range(config.num_gpus):
        stage_indices = list(
            range(rank, config.num_gpus * n_local_stages, config.num_gpus)
        )
        rank_ops: List[Optional[_PlannedAction]] = [None for _ in range(rank)]

        for stage_index in stage_indices:
            rank_ops.extend(
                _PlannedAction(stage_index, "fwd", microbatch_idx)
                for microbatch_idx in range(config.num_microbatches)
            )

        post_warmup_ops = 2 * (config.num_gpus - 1 - rank)
        rank_ops.extend([None] * post_warmup_ops)

        for stage_index in reversed(stage_indices):
            rank_ops.extend(
                _PlannedAction(stage_index, "bwd", microbatch_idx)
                for microbatch_idx in reversed(range(config.num_microbatches))
            )

        rank_plans[rank] = rank_ops

    return rank_plans


def _simulate_gpipe_standard(
    config: SimulationConfig,
) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
    config.validate()

    if config.num_stages != config.num_gpus:
        raise ValueError(
            "scheduler=gpipe requires len(stage_partition) == num_gpus, "
            f"got {config.num_stages} and {config.num_gpus}"
        )

    rank_plans = _build_gpipe_rank_plans_standard(config)

    return _simulate_planned_actions(
        config=config,
        rank_plans=rank_plans,
        schedule_name="gpipe",
        enforce_stage_equals_rank=True,
    )


def _simulate_looped_bfs_standard(
    config: SimulationConfig,
) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
    config.validate()

    if config.num_stages % config.num_gpus != 0:
        raise ValueError(
            "scheduler=looped-bfs requires len(stage_partition) to be a multiple of num_gpus, "
            f"got {config.num_stages} and {config.num_gpus}"
        )

    rank_plans = _build_looped_bfs_rank_plans_standard(config)

    return _simulate_planned_actions(
        config=config,
        rank_plans=rank_plans,
        schedule_name="looped-bfs",
        enforce_stage_equals_rank=False,
    )


def _simulate_1f1b_standard(
    config: SimulationConfig,
) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
    config.validate()

    if config.num_stages != config.num_gpus:
        raise ValueError(
            "scheduler=1f1b requires len(stage_partition) == num_gpus, "
            f"got {config.num_stages} and {config.num_gpus}"
        )

    rank_plans = _build_1f1b_rank_plans_standard(config)

    return _simulate_planned_actions(
        config=config,
        rank_plans=rank_plans,
        schedule_name="1f1b",
        enforce_stage_equals_rank=True,
    )


def _append_forward_layer_events_roundpipe(
    config: SimulationConfig,
    layer_events: List[LayerTimelineEvent],
    stage_idx: int,
    stage_layers: int,
    gpu_idx: int,
    microbatch_idx: int,
    start_time: float,
) -> None:
    cursor = start_time
    for layer_idx in range(stage_layers):
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=layer_idx,
                microbatch=microbatch_idx,
                kind="fwd",
                start=cursor,
                end=cursor + config.fwd_time_per_layer,
                is_lmhead=False,
            )
        )
        cursor += config.fwd_time_per_layer


def _append_recompute_bwd_layer_events_roundpipe(
    config: SimulationConfig,
    layer_events: List[LayerTimelineEvent],
    stage_idx: int,
    stage_layers: int,
    has_lmhead_bwd: bool,
    include_recompute: bool,
    gpu_idx: int,
    microbatch_idx: int,
    recompute_start: float,
    expected_end: float,
) -> None:
    cursor = recompute_start

    if has_lmhead_bwd and config.lmhead_fwd_time > 0:
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=stage_layers,
                microbatch=microbatch_idx,
                kind="fwd",
                start=cursor,
                end=cursor + config.lmhead_fwd_time,
                is_lmhead=True,
            )
        )
        cursor += config.lmhead_fwd_time

    if include_recompute:
        for layer_idx in range(stage_layers - 1, -1, -1):
            rec_start = cursor
            rec_end = rec_start + config.fwd_time_per_layer
            layer_events.append(
                LayerTimelineEvent(
                    stage=gpu_idx,
                    virtual_stage=stage_idx,
                    lane=layer_idx,
                    microbatch=microbatch_idx,
                    kind="recompute",
                    start=rec_start,
                    end=rec_end,
                    is_lmhead=False,
                )
            )
            cursor = rec_end

    if has_lmhead_bwd and config.lmhead_bwd_time > 0:
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=stage_layers,
                microbatch=microbatch_idx,
                kind="bwd",
                start=cursor,
                end=cursor + config.lmhead_bwd_time,
                is_lmhead=True,
            )
        )
        cursor += config.lmhead_bwd_time

    for layer_idx in range(stage_layers - 1, -1, -1):
        bwd_start = cursor
        bwd_end = bwd_start + config.bwd_time_per_layer
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=layer_idx,
                microbatch=microbatch_idx,
                kind="bwd",
                start=bwd_start,
                end=bwd_end,
                is_lmhead=False,
            )
        )
        cursor = bwd_end

    if abs(cursor - expected_end) > 1e-8:
        raise RuntimeError(
            "roundpipe layer timeline generation mismatch: "
            f"gpu={gpu_idx}, microbatch={microbatch_idx}, "
            f"generated_end={cursor}, expected_end={expected_end}"
        )


def _simulate_roundpipe(
    config: SimulationConfig,
) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
    config.validate()
    if config.backward_stage_partition is None:
        raise ValueError("roundpipe requires backward_stage_partition")

    fwd_partition = config.stage_partition
    bwd_partition = config.backward_stage_partition
    num_fwd_stages = len(fwd_partition)
    num_bwd_stages = len(bwd_partition)

    gpu_free_time = [0.0 for _ in range(config.num_gpus)]
    fwd_done_end: Dict[Tuple[int, int], float] = {}
    bwd_done_end: Dict[Tuple[int, int], float] = {}
    fwd_done_gpu: Dict[Tuple[int, int], int] = {}
    bwd_done_gpu: Dict[Tuple[int, int], int] = {}

    layer_events: List[LayerTimelineEvent] = []
    op_events: List[OperationEvent] = []
    comm_total = 0.0
    comm_exposed = 0.0

    launch_rounds: List[Tuple[int, int]] = []
    for round_start in range(0, config.num_microbatches, config.num_gpus):
        round_end = min(round_start + config.num_gpus, config.num_microbatches)
        launch_rounds.append((round_start, round_end))

    round_start_gpu = 0

    for round_start, round_end in launch_rounds:
        for stage_idx, stage_layers in enumerate(fwd_partition):
            gpu_idx = (round_start_gpu + stage_idx) % config.num_gpus
            stage_duration = stage_layers * config.fwd_time_per_layer

            for microbatch_idx in range(round_start, round_end):
                deps: List[Tuple[float, Optional[int]]] = []
                if stage_idx > 0:
                    key = (stage_idx - 1, microbatch_idx)
                    deps.append((fwd_done_end[key], fwd_done_gpu[key]))

                start_time, comm_stall, comm_hops = _resolve_start(
                    config.comm_time, gpu_idx, gpu_free_time[gpu_idx], deps
                )
                comm_total += comm_hops
                comm_exposed += comm_stall
                end_time = start_time + stage_duration
                _append_forward_layer_events_roundpipe(
                    config=config,
                    layer_events=layer_events,
                    stage_idx=stage_idx,
                    stage_layers=stage_layers,
                    gpu_idx=gpu_idx,
                    microbatch_idx=microbatch_idx,
                    start_time=start_time,
                )
                op_events.append(
                    OperationEvent(
                        gpu=gpu_idx,
                        virtual_stage=stage_idx,
                        microbatch=microbatch_idx,
                        kind="fwd",
                        start=start_time,
                        end=end_time,
                    )
                )

                fwd_done_end[(stage_idx, microbatch_idx)] = end_time
                fwd_done_gpu[(stage_idx, microbatch_idx)] = gpu_idx
                gpu_free_time[gpu_idx] = end_time

        for stage_idx, stage_layers in enumerate(bwd_partition):
            gpu_idx = (round_start_gpu + num_fwd_stages + stage_idx) % config.num_gpus
            has_lmhead_bwd = stage_idx == 0
            include_recompute = not has_lmhead_bwd
            recompute_duration = stage_layers * config.fwd_time_per_layer if include_recompute else 0.0
            lmhead_fwd_duration = config.lmhead_fwd_time if has_lmhead_bwd else 0.0
            pre_bwd_duration = lmhead_fwd_duration + recompute_duration
            bwd_duration = stage_layers * config.bwd_time_per_layer
            if has_lmhead_bwd:
                bwd_duration += config.lmhead_bwd_time

            for microbatch_idx in range(round_start, round_end):
                fwd_key = (num_fwd_stages - 1, microbatch_idx)
                deps = [(fwd_done_end[fwd_key], fwd_done_gpu[fwd_key])]
                if stage_idx > 0:
                    bwd_key = (stage_idx - 1, microbatch_idx)
                    deps.append((bwd_done_end[bwd_key], bwd_done_gpu[bwd_key]))

                # Dependencies gate the backward itself; recompute is placed
                # right before it, so the GPU must be free that much earlier.
                bwd_ready, comm_stall, comm_hops = _resolve_start(
                    config.comm_time,
                    gpu_idx,
                    gpu_free_time[gpu_idx] + pre_bwd_duration,
                    deps,
                )
                comm_total += comm_hops
                comm_exposed += comm_stall
                pre_bwd_start = bwd_ready - pre_bwd_duration
                lmhead_fwd_end = pre_bwd_start + lmhead_fwd_duration
                recompute_start = lmhead_fwd_end
                recompute_end = recompute_start + recompute_duration
                bwd_start = recompute_end
                bwd_end = bwd_start + bwd_duration

                _append_recompute_bwd_layer_events_roundpipe(
                    config=config,
                    layer_events=layer_events,
                    stage_idx=stage_idx,
                    stage_layers=stage_layers,
                    has_lmhead_bwd=has_lmhead_bwd,
                    include_recompute=include_recompute,
                    gpu_idx=gpu_idx,
                    microbatch_idx=microbatch_idx,
                    recompute_start=pre_bwd_start,
                    expected_end=bwd_end,
                )
                if has_lmhead_bwd and lmhead_fwd_duration > 1e-8:
                    op_events.append(
                        OperationEvent(
                            gpu=gpu_idx,
                            virtual_stage=stage_idx,
                            microbatch=microbatch_idx,
                            kind="fwd",
                            start=pre_bwd_start,
                            end=lmhead_fwd_end,
                        )
                    )
                if recompute_end - recompute_start > 1e-8:
                    op_events.append(
                        OperationEvent(
                            gpu=gpu_idx,
                            virtual_stage=stage_idx,
                            microbatch=microbatch_idx,
                            kind="recompute",
                            start=recompute_start,
                            end=recompute_end,
                        )
                    )
                op_events.append(
                    OperationEvent(
                        gpu=gpu_idx,
                        virtual_stage=stage_idx,
                        microbatch=microbatch_idx,
                        kind="bwd",
                        start=bwd_start,
                        end=bwd_end,
                    )
                )

                bwd_done_end[(stage_idx, microbatch_idx)] = bwd_end
                bwd_done_gpu[(stage_idx, microbatch_idx)] = gpu_idx
                gpu_free_time[gpu_idx] = bwd_end

        round_start_gpu = min(
            range(config.num_gpus), key=lambda idx: (gpu_free_time[idx], idx)
        )

    expected_fwd = num_fwd_stages * config.num_microbatches
    expected_bwd = num_bwd_stages * config.num_microbatches
    if len(fwd_done_end) != expected_fwd or len(bwd_done_end) != expected_bwd:
        raise RuntimeError(
            "incomplete roundpipe schedule execution: "
            f"fwd={len(fwd_done_end)}, bwd={len(bwd_done_end)}, "
            f"expected_fwd={expected_fwd}, expected_bwd={expected_bwd}"
        )

    sorted_layer_events = sorted(
        layer_events,
        key=lambda event: (event.start, event.stage, event.lane, event.microbatch),
    )
    sorted_op_events = sorted(
        op_events,
        key=lambda event: (event.start, event.gpu, event.virtual_stage, event.microbatch, event.kind),
    )
    return (
        sorted_layer_events,
        sorted_op_events,
        CommStats(comm_time=config.comm_time, total=comm_total, exposed=comm_exposed),
    )


def _append_forward_layer_events(
    config: SimulationConfig,
    layer_events: List[LayerTimelineEvent],
    stage_idx: int,
    gpu_idx: int,
    microbatch_idx: int,
    start_time: float,
) -> None:
    cursor = start_time
    stage_layers = config.stage_partition[stage_idx]
    for layer_idx in range(stage_layers):
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=_layer_lane(stage_idx, layer_idx),
                microbatch=microbatch_idx,
                kind="fwd",
                start=cursor,
                end=cursor + config.fwd_time_per_layer,
                is_lmhead=False,
            )
        )
        cursor += config.fwd_time_per_layer

    if stage_idx == config.num_stages - 1 and config.lmhead_fwd_time > 0:
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=_lmhead_lane(config, stage_idx),
                microbatch=microbatch_idx,
                kind="fwd",
                start=cursor,
                end=cursor + config.lmhead_fwd_time,
                is_lmhead=True,
            )
        )


def _append_recompute_bwd_layer_events(
    config: SimulationConfig,
    layer_events: List[LayerTimelineEvent],
    stage_idx: int,
    gpu_idx: int,
    microbatch_idx: int,
    recompute_start: float,
    expected_end: float,
) -> None:
    cursor = recompute_start
    stage_layers = config.stage_partition[stage_idx]

    if stage_idx == config.num_stages - 1 and config.lmhead_bwd_time > 0:
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=_lmhead_lane(config, stage_idx),
                microbatch=microbatch_idx,
                kind="bwd",
                start=cursor,
                end=cursor + config.lmhead_bwd_time,
                is_lmhead=True,
            )
        )
        cursor += config.lmhead_bwd_time

    for layer_idx in range(stage_layers - 1, -1, -1):
        rec_start = cursor
        rec_end = rec_start + config.fwd_time_per_layer
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=_layer_lane(stage_idx, layer_idx),
                microbatch=microbatch_idx,
                kind="recompute",
                start=rec_start,
                end=rec_end,
                is_lmhead=False,
            )
        )

        bwd_start = rec_end
        bwd_end = bwd_start + config.bwd_time_per_layer
        layer_events.append(
            LayerTimelineEvent(
                stage=gpu_idx,
                virtual_stage=stage_idx,
                lane=_layer_lane(stage_idx, layer_idx),
                microbatch=microbatch_idx,
                kind="bwd",
                start=bwd_start,
                end=bwd_end,
                is_lmhead=False,
            )
        )
        cursor = bwd_end

    if abs(cursor - expected_end) > 1e-8:
        raise RuntimeError(
            "layer timeline generation mismatch: "
            f"stage={stage_idx}, microbatch={microbatch_idx}, "
            f"generated_end={cursor}, expected_end={expected_end}"
        )


def _simulate_planned_actions(
    config: SimulationConfig,
    rank_plans: Dict[int, List[Optional[_PlannedAction]]],
    schedule_name: str,
    enforce_stage_equals_rank: bool,
) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
    gpu_free_time = [0.0 for _ in range(config.num_gpus)]
    plan_cursor = [0 for _ in range(config.num_gpus)]

    fwd_done_end: Dict[Tuple[int, int], float] = {}
    bwd_done_end: Dict[Tuple[int, int], float] = {}
    fwd_done_gpu: Dict[Tuple[int, int], int] = {}
    bwd_done_gpu: Dict[Tuple[int, int], int] = {}

    layer_events: List[LayerTimelineEvent] = []
    op_events: List[OperationEvent] = []
    comm_total = 0.0
    comm_exposed = 0.0

    total_planned_ops = sum(
        1 for rank in rank_plans for action in rank_plans[rank] if action is not None
    )
    executed_ops = 0

    while executed_ops < total_planned_ops:
        candidates: List[Tuple[float, int, _PlannedAction, float, float]] = []

        for gpu_idx in range(config.num_gpus):
            rank_plan = rank_plans[gpu_idx]
            while plan_cursor[gpu_idx] < len(rank_plan) and rank_plan[plan_cursor[gpu_idx]] is None:
                plan_cursor[gpu_idx] += 1

            if plan_cursor[gpu_idx] >= len(rank_plan):
                continue

            action = rank_plan[plan_cursor[gpu_idx]]
            if action is None:
                continue

            if enforce_stage_equals_rank and action.stage_index != gpu_idx:
                raise RuntimeError(
                    "invalid 1f1b rank plan: "
                    f"rank={gpu_idx} got stage_index={action.stage_index}"
                )

            deps: List[Tuple[float, Optional[int]]] = []
            if action.kind == "fwd":
                if action.stage_index > 0:
                    key = (action.stage_index - 1, action.microbatch_index)
                    if key not in fwd_done_end:
                        continue
                    deps.append((fwd_done_end[key], fwd_done_gpu[key]))
            elif action.kind == "bwd":
                local_fwd_key = (action.stage_index, action.microbatch_index)
                if local_fwd_key not in fwd_done_end:
                    continue
                # Saved activations stay on the GPU that produced them.
                deps.append((fwd_done_end[local_fwd_key], None))

                if action.stage_index < config.num_stages - 1:
                    key = (action.stage_index + 1, action.microbatch_index)
                    if key not in bwd_done_end:
                        continue
                    deps.append((bwd_done_end[key], bwd_done_gpu[key]))
            else:
                raise RuntimeError(f"unsupported action kind {action.kind}")

            candidate_start, candidate_stall, candidate_hops = _resolve_start(
                config.comm_time, gpu_idx, gpu_free_time[gpu_idx], deps
            )
            candidates.append(
                (candidate_start, gpu_idx, action, candidate_stall, candidate_hops)
            )

        if not candidates:
            raise RuntimeError(f"{schedule_name} schedule deadlocked; no action is ready")

        candidates.sort(key=lambda item: (item[0], item[1]))
        start_time, gpu_idx, action, comm_stall, comm_hops = candidates[0]
        comm_total += comm_hops
        comm_exposed += comm_stall

        if action.kind == "fwd":
            end_time = start_time + config.fwd_duration(action.stage_index)
            _append_forward_layer_events(
                config=config,
                layer_events=layer_events,
                stage_idx=action.stage_index,
                gpu_idx=gpu_idx,
                microbatch_idx=action.microbatch_index,
                start_time=start_time,
            )
            op_events.append(
                OperationEvent(
                    gpu=gpu_idx,
                    virtual_stage=action.stage_index,
                    microbatch=action.microbatch_index,
                    kind="fwd",
                    start=start_time,
                    end=end_time,
                )
            )
            fwd_done_end[(action.stage_index, action.microbatch_index)] = end_time
            fwd_done_gpu[(action.stage_index, action.microbatch_index)] = gpu_idx
        else:
            recompute_end = start_time + config.recompute_duration(action.stage_index)
            bwd_end = recompute_end + config.bwd_duration(action.stage_index)
            _append_recompute_bwd_layer_events(
                config=config,
                layer_events=layer_events,
                stage_idx=action.stage_index,
                gpu_idx=gpu_idx,
                microbatch_idx=action.microbatch_index,
                recompute_start=start_time,
                expected_end=bwd_end,
            )
            op_events.append(
                OperationEvent(
                    gpu=gpu_idx,
                    virtual_stage=action.stage_index,
                    microbatch=action.microbatch_index,
                    kind="recompute",
                    start=start_time,
                    end=recompute_end,
                )
            )
            op_events.append(
                OperationEvent(
                    gpu=gpu_idx,
                    virtual_stage=action.stage_index,
                    microbatch=action.microbatch_index,
                    kind="bwd",
                    start=recompute_end,
                    end=bwd_end,
                )
            )
            end_time = bwd_end
            bwd_done_end[(action.stage_index, action.microbatch_index)] = bwd_end
            bwd_done_gpu[(action.stage_index, action.microbatch_index)] = gpu_idx

        gpu_free_time[gpu_idx] = end_time
        plan_cursor[gpu_idx] += 1
        executed_ops += 1

    expected_stage_mb = config.num_stages * config.num_microbatches
    if len(fwd_done_end) != expected_stage_mb or len(bwd_done_end) != expected_stage_mb:
        raise RuntimeError(
            f"incomplete {schedule_name} schedule execution: "
            f"fwd={len(fwd_done_end)}, bwd={len(bwd_done_end)}, expected={expected_stage_mb}"
        )

    sorted_layer_events = sorted(
        layer_events,
        key=lambda event: (event.start, event.stage, event.lane, event.microbatch),
    )
    sorted_op_events = sorted(
        op_events,
        key=lambda event: (event.start, event.gpu, event.virtual_stage, event.microbatch, event.kind),
    )
    return (
        sorted_layer_events,
        sorted_op_events,
        CommStats(comm_time=config.comm_time, total=comm_total, exposed=comm_exposed),
    )


@dataclass(frozen=True)
class _PlannedAction:
    stage_index: int
    kind: str
    microbatch_index: int


def _get_warmup_ops_standard(
    rank: int,
    n_local_stages: int,
    microbatches_per_round: int,
    pp_group_size: int,
    n_microbatches: int,
) -> int:
    warmups_ops_last_stage = (n_local_stages - 1) * microbatches_per_round
    warmup_ops = warmups_ops_last_stage + 2 * ((pp_group_size - 1) - rank)
    return min(warmup_ops, n_microbatches * n_local_stages)


def _get_1f1b_rank_ops_standard(
    n_local_stages: int,
    pp_group_size: int,
    warmup_ops: int,
    fwd_bwd_ops: int,
    cooldown_ops: int,
    rank: int,
    forward_stage_index,
    backward_stage_index,
) -> List[Optional[_PlannedAction]]:
    fwd_stage_mb_index: Dict[int, int] = defaultdict(int)
    bwd_stage_mb_index: Dict[int, int] = defaultdict(int)

    rank_ops: List[Optional[_PlannedAction]] = [None for _ in range(rank)]
    post_warmup_ops = (
        n_local_stages * pp_group_size + 2 * (pp_group_size - 1 - rank)
    ) - (warmup_ops + rank)

    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

    for op in range(total_ops):
        if op < warmup_ops:
            fwd_stage = forward_stage_index(op)
            fwd_mb = fwd_stage_mb_index[fwd_stage]
            fwd_stage_mb_index[fwd_stage] += 1
            rank_ops.append(_PlannedAction(fwd_stage, "fwd", fwd_mb))

            if op == warmup_ops - 1:
                rank_ops.extend([None] * post_warmup_ops)
        elif warmup_ops <= op < warmup_ops + fwd_bwd_ops:
            fwd_stage = forward_stage_index(op)
            fwd_mb = fwd_stage_mb_index[fwd_stage]
            fwd_stage_mb_index[fwd_stage] += 1
            rank_ops.append(_PlannedAction(fwd_stage, "fwd", fwd_mb))

            bwd_stage = backward_stage_index(op)
            bwd_mb = bwd_stage_mb_index[bwd_stage]
            bwd_stage_mb_index[bwd_stage] += 1
            rank_ops.append(_PlannedAction(bwd_stage, "bwd", bwd_mb))
        else:
            rank_ops.append(None)

            bwd_stage = backward_stage_index(op)
            bwd_mb = bwd_stage_mb_index[bwd_stage]
            bwd_stage_mb_index[bwd_stage] += 1
            rank_ops.append(_PlannedAction(bwd_stage, "bwd", bwd_mb))

    return rank_ops


def _build_interleaved_rank_plans_standard(
    config: SimulationConfig,
) -> Dict[int, List[Optional[_PlannedAction]]]:
    if config.num_stages % config.num_gpus != 0:
        raise ValueError(
            "interleaved-1f1b requires len(stage_partition) to be a multiple of num_gpus"
        )

    n_local_stages = config.num_stages // config.num_gpus
    number_of_rounds = max(1, config.num_microbatches // config.num_gpus)
    microbatches_per_round = config.num_microbatches // number_of_rounds
    if config.num_microbatches % number_of_rounds != 0:
        raise ValueError(
            "Interleaved 1F1B requires num_microbatches to be a multiple of number_of_rounds, "
            f"got {config.num_microbatches} and {number_of_rounds}"
        )

    rank_plans: Dict[int, List[Optional[_PlannedAction]]] = {}
    for rank in range(config.num_gpus):
        warmup_ops = _get_warmup_ops_standard(
            rank=rank,
            n_local_stages=n_local_stages,
            microbatches_per_round=microbatches_per_round,
            pp_group_size=config.num_gpus,
            n_microbatches=config.num_microbatches,
        )
        microbatch_ops = n_local_stages * config.num_microbatches
        fwd_bwd_ops = microbatch_ops - warmup_ops
        cooldown_ops = microbatch_ops - fwd_bwd_ops

        def forward_stage_index(step: int) -> int:
            local_index = (step // microbatches_per_round) % n_local_stages
            return (local_index * config.num_gpus) + rank

        def backward_stage_index(step: int) -> int:
            local_index = (
                n_local_stages
                - 1
                - ((step - warmup_ops) // microbatches_per_round) % n_local_stages
            )
            return (local_index * config.num_gpus) + rank

        rank_plans[rank] = _get_1f1b_rank_ops_standard(
            n_local_stages=n_local_stages,
            pp_group_size=config.num_gpus,
            warmup_ops=warmup_ops,
            fwd_bwd_ops=fwd_bwd_ops,
            cooldown_ops=cooldown_ops,
            rank=rank,
            forward_stage_index=forward_stage_index,
            backward_stage_index=backward_stage_index,
        )

    return rank_plans


@dataclass(frozen=True)
class _RuntimeAction:
    stage_index: int
    kind: str
    microbatch_index: int


def _infer_stage_to_rank_mapping(
    config: SimulationConfig,
    rank_plans: Dict[int, List[Optional[_PlannedAction]]],
) -> Dict[int, int]:
    stage_to_rank: Dict[int, int] = {}
    for rank, actions in rank_plans.items():
        for action in actions:
            if action is None:
                continue
            existing_rank = stage_to_rank.get(action.stage_index)
            if existing_rank is None:
                stage_to_rank[action.stage_index] = rank
            elif existing_rank != rank:
                raise RuntimeError(
                    "interleaved-1f1b stage mapping conflict: "
                    f"stage={action.stage_index}, rank={rank}, existing_rank={existing_rank}"
                )

    if len(stage_to_rank) != config.num_stages:
        raise RuntimeError(
            "interleaved-1f1b stage mapping incomplete: "
            f"mapped={len(stage_to_rank)}, expected={config.num_stages}"
        )

    return stage_to_rank


def _ready_to_schedule_compute_action_with_comms(
    action: _RuntimeAction,
    prev_actions: set[_RuntimeAction],
    num_stages: int,
) -> bool:
    stage_idx = action.stage_index
    mb_idx = action.microbatch_index

    if action.kind == "fwd" and stage_idx != 0:
        if _RuntimeAction(stage_idx, "recv_f", mb_idx) in prev_actions:
            return True
        if _RuntimeAction(stage_idx - 1, "fwd", mb_idx) in prev_actions:
            return True
        return False

    if action.kind == "bwd" and stage_idx != num_stages - 1:
        if _RuntimeAction(stage_idx, "recv_b", mb_idx) in prev_actions:
            return True
        if _RuntimeAction(stage_idx + 1, "bwd", mb_idx) in prev_actions:
            return True
        return False

    return True


def _has_comms_action(
    action: _RuntimeAction,
    stage_to_rank: Dict[int, int],
    num_stages: int,
) -> bool:
    if action.kind == "fwd":
        return (
            action.stage_index != num_stages - 1
            and stage_to_rank[action.stage_index + 1] != stage_to_rank[action.stage_index]
        )
    if action.kind == "bwd":
        return (
            action.stage_index != 0
            and stage_to_rank[action.stage_index - 1] != stage_to_rank[action.stage_index]
        )
    return False


def _get_send_recv_actions(action: _RuntimeAction) -> Tuple[_RuntimeAction, _RuntimeAction]:
    stage_idx = action.stage_index
    mb_idx = action.microbatch_index
    if action.kind == "fwd":
        return (
            _RuntimeAction(stage_idx, "send_f", mb_idx),
            _RuntimeAction(stage_idx + 1, "recv_f", mb_idx),
        )
    if action.kind == "bwd":
        return (
            _RuntimeAction(stage_idx, "send_b", mb_idx),
            _RuntimeAction(stage_idx - 1, "recv_b", mb_idx),
        )
    raise RuntimeError(f"invalid compute action kind for comm lowering: {action.kind}")


def _build_interleaved_schedule_with_comms(
    config: SimulationConfig,
    rank_plans: Dict[int, List[Optional[_PlannedAction]]],
) -> Tuple[Dict[int, List[_RuntimeAction]], Dict[int, int]]:
    stage_to_rank = _infer_stage_to_rank_mapping(config, rank_plans)

    compute_actions: Dict[int, List[_RuntimeAction]] = {
        rank: [
            _RuntimeAction(action.stage_index, action.kind, action.microbatch_index)
            for action in rank_plans[rank]
            if action is not None
        ]
        for rank in sorted(rank_plans)
    }

    comm_actions: Dict[int, List[_RuntimeAction]] = {
        rank: [] for rank in sorted(rank_plans)
    }
    prev_actions: Dict[int, set[_RuntimeAction]] = {
        rank: set() for rank in sorted(rank_plans)
    }

    while compute_actions:
        progress = False
        for rank in sorted(list(compute_actions.keys())):
            if len(compute_actions[rank]) == 0:
                continue

            action = compute_actions[rank][0]
            if not _ready_to_schedule_compute_action_with_comms(
                action,
                prev_actions[rank],
                config.num_stages,
            ):
                continue

            comm_actions[rank].append(action)
            prev_actions[rank].add(action)

            if _has_comms_action(action, stage_to_rank, config.num_stages):
                send_action, recv_action = _get_send_recv_actions(action)
                comm_actions[rank].append(send_action)
                prev_actions[rank].add(send_action)

                recv_rank = stage_to_rank[recv_action.stage_index]
                comm_actions[recv_rank].append(recv_action)
                prev_actions[recv_rank].add(recv_action)

            compute_actions[rank].pop(0)
            if len(compute_actions[rank]) == 0:
                del compute_actions[rank]
            progress = True

        if not progress:
            raise RuntimeError("malformed interleaved compute schedule; cannot lower comms")

    return comm_actions, stage_to_rank


def _ready_to_schedule_runtime_action(
    action: _RuntimeAction,
    prev_ops_rank: set[_RuntimeAction],
    prev_ops_by_rank: Dict[int, set[_RuntimeAction]],
    stage_to_rank: Dict[int, int],
    num_stages: int,
) -> bool:
    stage_idx = action.stage_index
    mb_idx = action.microbatch_index

    if action.kind == "fwd":
        if stage_idx == 0:
            return True
        if _RuntimeAction(stage_idx, "recv_f", mb_idx) in prev_ops_rank:
            return True
        prev_stage_rank = stage_to_rank[stage_idx - 1]
        return _RuntimeAction(stage_idx - 1, "fwd", mb_idx) in prev_ops_by_rank[prev_stage_rank]

    if action.kind == "bwd":
        if stage_idx == num_stages - 1:
            return True
        if _RuntimeAction(stage_idx, "recv_b", mb_idx) in prev_ops_rank:
            return True
        next_stage_rank = stage_to_rank[stage_idx + 1]
        return _RuntimeAction(stage_idx + 1, "bwd", mb_idx) in prev_ops_by_rank[next_stage_rank]

    if action.kind == "send_f":
        return _RuntimeAction(stage_idx, "fwd", mb_idx) in prev_ops_rank

    if action.kind == "recv_f":
        peer_stage = stage_idx - 1
        peer_rank = stage_to_rank[peer_stage]
        return _RuntimeAction(peer_stage, "send_f", mb_idx) in prev_ops_by_rank[peer_rank]

    if action.kind == "send_b":
        return _RuntimeAction(stage_idx, "bwd", mb_idx) in prev_ops_rank

    if action.kind == "recv_b":
        peer_stage = stage_idx + 1
        peer_rank = stage_to_rank[peer_stage]
        return _RuntimeAction(peer_stage, "send_b", mb_idx) in prev_ops_by_rank[peer_rank]

    raise RuntimeError(f"unsupported runtime action kind {action.kind}")


def _simulate_interleaved_runtime_steps_with_comms(
    comm_actions: Dict[int, List[_RuntimeAction]],
    stage_to_rank: Dict[int, int],
    num_stages: int,
) -> Dict[int, List[Optional[_RuntimeAction]]]:
    pending: Dict[int, List[_RuntimeAction]] = {
        rank: list(actions) for rank, actions in sorted(comm_actions.items())
    }
    scheduled: Dict[int, List[Optional[_RuntimeAction]]] = {
        rank: [] for rank in sorted(comm_actions)
    }
    prev_ops_by_rank: Dict[int, set[_RuntimeAction]] = {
        rank: set() for rank in sorted(comm_actions)
    }

    while pending:
        progress = False

        for rank in sorted(list(pending.keys())):
            if len(pending[rank]) == 0:
                continue

            action = pending[rank][0]
            if _ready_to_schedule_runtime_action(
                action,
                prev_ops_by_rank[rank],
                prev_ops_by_rank,
                stage_to_rank,
                num_stages,
            ):
                scheduled[rank].append(action)
                prev_ops_by_rank[rank].add(action)
                pending[rank].pop(0)
                progress = True
            else:
                scheduled[rank].append(None)

        for rank in sorted(list(pending.keys()), reverse=True):
            if len(pending[rank]) == 0:
                del pending[rank]

        for rank in sorted(list(pending.keys())):
            if len(pending[rank]) == 0:
                continue
            if len(scheduled[rank]) == 0 or scheduled[rank][-1] is not None:
                continue

            action = pending[rank][0]
            if _ready_to_schedule_runtime_action(
                action,
                prev_ops_by_rank[rank],
                prev_ops_by_rank,
                stage_to_rank,
                num_stages,
            ):
                scheduled[rank][-1] = action
                prev_ops_by_rank[rank].add(action)
                pending[rank].pop(0)

        for rank in sorted(list(pending.keys()), reverse=True):
            if len(pending[rank]) == 0:
                del pending[rank]

        if not progress:
            raise RuntimeError("interleaved-1f1b runtime schedule is not progressing")

    return scheduled


def _simulate_interleaved_actions_pytorch_style(
    config: SimulationConfig,
    rank_plans: Dict[int, List[Optional[_PlannedAction]]],
) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
    comm_actions, stage_to_rank = _build_interleaved_schedule_with_comms(config, rank_plans)
    runtime_schedule = _simulate_interleaved_runtime_steps_with_comms(
        comm_actions=comm_actions,
        stage_to_rank=stage_to_rank,
        num_stages=config.num_stages,
    )

    gpu_free_time = [0.0 for _ in range(config.num_gpus)]
    fwd_done_end: Dict[Tuple[int, int], float] = {}
    bwd_done_end: Dict[Tuple[int, int], float] = {}
    fwd_done_gpu: Dict[Tuple[int, int], int] = {}
    bwd_done_gpu: Dict[Tuple[int, int], int] = {}

    layer_events: List[LayerTimelineEvent] = []
    op_events: List[OperationEvent] = []
    comm_total = 0.0
    comm_exposed = 0.0

    max_steps = max((len(actions) for actions in runtime_schedule.values()), default=0)

    for step_idx in range(max_steps):
        pending_compute: List[Tuple[int, _RuntimeAction]] = []
        for rank in sorted(runtime_schedule):
            if step_idx >= len(runtime_schedule[rank]):
                continue
            action = runtime_schedule[rank][step_idx]
            if action is not None and action.kind in ("fwd", "bwd"):
                pending_compute.append((rank, action))

        while pending_compute:
            progressed = False
            for idx in range(len(pending_compute) - 1, -1, -1):
                gpu_idx, action = pending_compute[idx]

                deps: List[Tuple[float, Optional[int]]] = []
                if action.kind == "fwd":
                    if action.stage_index > 0:
                        key = (action.stage_index - 1, action.microbatch_index)
                        if key not in fwd_done_end:
                            continue
                        deps.append((fwd_done_end[key], fwd_done_gpu[key]))
                else:
                    local_fwd_key = (action.stage_index, action.microbatch_index)
                    if local_fwd_key not in fwd_done_end:
                        continue
                    # Saved activations stay on the GPU that produced them.
                    deps.append((fwd_done_end[local_fwd_key], None))

                    if action.stage_index < config.num_stages - 1:
                        key = (action.stage_index + 1, action.microbatch_index)
                        if key not in bwd_done_end:
                            continue
                        deps.append((bwd_done_end[key], bwd_done_gpu[key]))

                start_time, comm_stall, comm_hops = _resolve_start(
                    config.comm_time, gpu_idx, gpu_free_time[gpu_idx], deps
                )
                comm_total += comm_hops
                comm_exposed += comm_stall

                if action.kind == "fwd":
                    end_time = start_time + config.fwd_duration(action.stage_index)
                    _append_forward_layer_events(
                        config=config,
                        layer_events=layer_events,
                        stage_idx=action.stage_index,
                        gpu_idx=gpu_idx,
                        microbatch_idx=action.microbatch_index,
                        start_time=start_time,
                    )
                    op_events.append(
                        OperationEvent(
                            gpu=gpu_idx,
                            virtual_stage=action.stage_index,
                            microbatch=action.microbatch_index,
                            kind="fwd",
                            start=start_time,
                            end=end_time,
                        )
                    )
                    fwd_done_end[(action.stage_index, action.microbatch_index)] = end_time
                    fwd_done_gpu[(action.stage_index, action.microbatch_index)] = gpu_idx
                else:
                    recompute_end = start_time + config.recompute_duration(action.stage_index)
                    bwd_end = recompute_end + config.bwd_duration(action.stage_index)
                    _append_recompute_bwd_layer_events(
                        config=config,
                        layer_events=layer_events,
                        stage_idx=action.stage_index,
                        gpu_idx=gpu_idx,
                        microbatch_idx=action.microbatch_index,
                        recompute_start=start_time,
                        expected_end=bwd_end,
                    )
                    op_events.append(
                        OperationEvent(
                            gpu=gpu_idx,
                            virtual_stage=action.stage_index,
                            microbatch=action.microbatch_index,
                            kind="recompute",
                            start=start_time,
                            end=recompute_end,
                        )
                    )
                    op_events.append(
                        OperationEvent(
                            gpu=gpu_idx,
                            virtual_stage=action.stage_index,
                            microbatch=action.microbatch_index,
                            kind="bwd",
                            start=recompute_end,
                            end=bwd_end,
                        )
                    )
                    end_time = bwd_end
                    bwd_done_end[(action.stage_index, action.microbatch_index)] = bwd_end
                    bwd_done_gpu[(action.stage_index, action.microbatch_index)] = gpu_idx

                gpu_free_time[gpu_idx] = end_time
                pending_compute.pop(idx)
                progressed = True

            if not progressed:
                raise RuntimeError(
                    "interleaved-1f1b timing deadlock inside one runtime step: "
                    f"step={step_idx}, pending={len(pending_compute)}"
                )

    expected_stage_mb = config.num_stages * config.num_microbatches
    if len(fwd_done_end) != expected_stage_mb or len(bwd_done_end) != expected_stage_mb:
        raise RuntimeError(
            "incomplete interleaved-1f1b schedule execution: "
            f"fwd={len(fwd_done_end)}, bwd={len(bwd_done_end)}, expected={expected_stage_mb}"
        )

    sorted_layer_events = sorted(
        layer_events,
        key=lambda event: (event.start, event.stage, event.lane, event.microbatch),
    )
    sorted_op_events = sorted(
        op_events,
        key=lambda event: (event.start, event.gpu, event.virtual_stage, event.microbatch, event.kind),
    )
    return (
        sorted_layer_events,
        sorted_op_events,
        CommStats(comm_time=config.comm_time, total=comm_total, exposed=comm_exposed),
    )


def _simulate_interleaved_1f1b_standard(
    config: SimulationConfig,
) -> Tuple[List[LayerTimelineEvent], List[OperationEvent], CommStats]:
    config.validate()

    rank_plans = _build_interleaved_rank_plans_standard(config)
    return _simulate_interleaved_actions_pytorch_style(config, rank_plans)


SCHEDULER_REGISTRY: Dict[str, PipelineScheduler] = {
    OneFOneBScheduler.name: OneFOneBScheduler(),
    GPipeScheduler.name: GPipeScheduler(),
    InterleavedOneFOneBScheduler.name: InterleavedOneFOneBScheduler(),
    LoopedBFSScheduler.name: LoopedBFSScheduler(),
    RoundPipeScheduler.name: RoundPipeScheduler(),
}


def parse_stage_partitions(raw: str) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
    def parse_one(part_raw: str, name: str) -> Tuple[int, ...]:
        values = [chunk.strip() for chunk in part_raw.split(",") if chunk.strip()]
        if not values:
            raise ValueError(f"{name} is empty")
        return tuple(int(value) for value in values)

    if "|" in raw:
        chunks = [part.strip() for part in raw.split("|")]
        if len(chunks) != 2:
            raise ValueError(
                "dual stage partition must be exactly 'fwd_stage_partition|bwd_stage_partition'"
            )
        fwd = parse_one(chunks[0], "forward stage_partition")
        bwd = parse_one(chunks[1], "backward stage_partition")
        return fwd, bwd

    return parse_one(raw, "stage_partition"), None


def _layer_lane(stage_idx: int, layer_in_stage: int) -> int:
    return layer_in_stage


def _lmhead_lane(config: SimulationConfig, stage_idx: int) -> int:
    return config.stage_partition[stage_idx]


def _build_stage_intervals(
    events: Sequence[LayerTimelineEvent],
    num_gpus: int,
) -> Dict[int, List[Tuple[float, float]]]:
    intervals_by_stage: Dict[int, List[Tuple[float, float]]] = {
        stage_idx: [] for stage_idx in range(num_gpus)
    }
    for event in events:
        if event.end > event.start:
            intervals_by_stage[event.stage].append((event.start, event.end))
    return intervals_by_stage


def _compute_busy_time_and_active_span(intervals: List[Tuple[float, float]]) -> Tuple[float, float]:
    if not intervals:
        return 0.0, 0.0

    intervals.sort(key=lambda item: item[0])
    current_start, current_end = intervals[0]
    stage_span_start = current_start
    stage_busy_time = 0.0

    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            stage_busy_time += current_end - current_start
            current_start, current_end = start, end

    stage_busy_time += current_end - current_start
    stage_active_span = current_end - stage_span_start
    return stage_busy_time, stage_active_span


def compute_bubble_stats(
    events: Sequence[LayerTimelineEvent],
    num_gpus: int,
) -> Tuple[float, float, float]:
    if num_gpus <= 0:
        raise ValueError("num_gpus must be > 0")

    if not events:
        return 0.0, 0.0, 0.0

    makespan = max(event.end for event in events)
    if makespan <= 0:
        return makespan, 0.0, 0.0

    intervals_by_stage = _build_stage_intervals(events, num_gpus)

    total_busy_time = 0.0
    for stage_idx in range(num_gpus):
        stage_busy_time, _ = _compute_busy_time_and_active_span(intervals_by_stage[stage_idx])
        total_busy_time += stage_busy_time

    total_capacity = makespan * num_gpus
    bubble_time = max(total_capacity - total_busy_time, 0.0)
    bubble_ratio = bubble_time / total_capacity if total_capacity > 0 else 0.0
    return makespan, bubble_time, bubble_ratio


def _warmup_forward_end(
    gpu_events: Sequence[LayerTimelineEvent],
    warmup_count: int,
    first_bwd_start: float,
) -> Optional[float]:
    """End of the `warmup_count`-th forward operation on a GPU.

    Layer events are per-layer, so forwards are regrouped into operations by
    (microbatch, virtual_stage) first. Only forwards that complete before the
    first backward count; if there are fewer than `warmup_count` of them, the
    last one is used.
    """
    op_end: Dict[Tuple[int, int], float] = {}
    for event in gpu_events:
        if event.kind != "fwd":
            continue
        key = (event.microbatch, event.virtual_stage)
        op_end[key] = max(op_end.get(key, event.end), event.end)

    ends = sorted(end for end in op_end.values() if end <= first_bwd_start)
    if not ends:
        return None
    if warmup_count <= 0:
        # No warm-up phase: anchor on the last forward before the backward.
        return ends[-1]
    return ends[min(warmup_count, len(ends)) - 1]


def compute_bubble_breakdown(
    events: Sequence[LayerTimelineEvent],
    num_gpus: int,
    has_warmup_phase: bool = True,
) -> Dict[str, float]:
    """Split the pipeline bubble into fill / switch / drain / intra.

    fill   : per GPU, the idle time from t=0 until its first operation.
    switch : per GPU, the idle time between the end of its warm-up forwards
             (the `num_gpus - gpu_id`-th forward) and its first backward, i.e.
             the wait for the incoming gradient. Forwards pulled into that gap
             are busy time and do not count. Without a warm-up phase
             (RoundPipe) the window starts at the last forward before the
             backward instead, which leaves this at 0.
    drain  : per GPU, the idle time from its last backward until the makespan.
    intra  : whatever bubble is left inside the steady state (total bubble
             minus the three above).
    """
    empty = {
        "fill_time": 0.0,
        "switch_time": 0.0,
        "drain_time": 0.0,
        "intra_time": 0.0,
        "fill_ratio": 0.0,
        "switch_ratio": 0.0,
        "drain_ratio": 0.0,
        "intra_ratio": 0.0,
    }
    if num_gpus <= 0:
        raise ValueError("num_gpus must be > 0")
    if not events:
        return dict(empty)

    makespan = max(event.end for event in events)
    if makespan <= 0:
        return dict(empty)

    events_by_gpu: Dict[int, List[LayerTimelineEvent]] = {
        gpu: [] for gpu in range(num_gpus)
    }
    for event in events:
        if event.end > event.start:
            events_by_gpu[event.stage].append(event)

    fill_time = 0.0
    switch_time = 0.0
    drain_time = 0.0
    for gpu in range(num_gpus):
        gpu_events = sorted(events_by_gpu[gpu], key=lambda item: (item.start, item.end))
        if not gpu_events:
            continue

        fill_time += gpu_events[0].start

        first_bwd_start = min(
            (event.start for event in gpu_events if event.kind == "bwd"),
            default=None,
        )
        if first_bwd_start is not None:
            warmup_count = num_gpus - gpu if has_warmup_phase else 0
            warmup_end = _warmup_forward_end(gpu_events, warmup_count, first_bwd_start)
            if warmup_end is not None:
                # The window is not fully idle: recompute sits right before the
                # bwd, and forwards can be pulled into the gap (interleaved-1F1B),
                # so subtract whatever the GPU is actually busy with inside it.
                window = [
                    (max(event.start, warmup_end), min(event.end, first_bwd_start))
                    for event in gpu_events
                    if event.end > warmup_end and event.start < first_bwd_start
                ]
                busy, _ = _compute_busy_time_and_active_span(
                    [(start, end) for start, end in window if end > start]
                )
                switch_time += max(first_bwd_start - warmup_end - busy, 0.0)

        last_bwd_end = max(
            (event.end for event in gpu_events if event.kind == "bwd"),
            default=None,
        )
        if last_bwd_end is None:
            last_bwd_end = max(event.end for event in gpu_events)
        drain_time += makespan - last_bwd_end

    _, bubble_time, _ = compute_bubble_stats(events, num_gpus)
    intra_time = max(bubble_time - fill_time - switch_time - drain_time, 0.0)

    total_capacity = makespan * num_gpus
    return {
        "fill_time": fill_time,
        "switch_time": switch_time,
        "drain_time": drain_time,
        "intra_time": intra_time,
        "fill_ratio": fill_time / total_capacity,
        "switch_ratio": switch_time / total_capacity,
        "drain_ratio": drain_time / total_capacity,
        "intra_ratio": intra_time / total_capacity,
    }


def print_operation_schedule(
    op_events: Sequence[OperationEvent],
    limit: int,
) -> None:
    if not op_events:
        print("schedule: (empty)")
        return

    first_bwd_start = min((event.start for event in op_events if event.kind == "bwd"), default=float("inf"))
    injected_before_first_bwd = sorted(
        {
            event.microbatch
            for event in op_events
            if event.kind == "fwd" and event.virtual_stage == 0 and event.start < first_bwd_start
        }
    )

    print("schedule_summary:")
    if first_bwd_start != float("inf"):
        print(f"  first_bwd_start={first_bwd_start:.4f}")
        print(
            "  stage0_microbatches_before_first_bwd="
            f"{len(injected_before_first_bwd)} -> {injected_before_first_bwd}"
        )
    else:
        print("  first_bwd_start=none")

    total = len(op_events)
    max_rows = total if limit <= 0 else min(limit, total)
    print(f"schedule_events_shown={max_rows}/{total}")

    for idx, event in enumerate(op_events[:max_rows], start=1):
        print(
            f"[{idx:04d}] t=[{event.start:.4f},{event.end:.4f}] "
            f"gpu={event.gpu} vstage={event.virtual_stage} kind={event.kind} mb={event.microbatch}"
        )


def plot_timeline(
    events: Sequence[LayerTimelineEvent],
    config: SimulationConfig,
    output_path: Path,
    title: str,
    scheduler_name: str,
) -> None:
    base_color_map = {
        "fwd": "#4F81BD",
        "recompute": "#F79646",
        "bwd": "#C0504D",
    }

    def hsl_to_hex(h: float, s: float, l: float) -> str:
        red, green, blue = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"

    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def event_facecolor(event: LayerTimelineEvent) -> str:
        gradient_schedulers = {
            InterleavedOneFOneBScheduler.name,
            LoopedBFSScheduler.name,
        }

        if scheduler_name not in gradient_schedulers:
            return base_color_map[event.kind]

        kind_hsl = {
            "fwd": (210.0 / 360.0, 0.45, 0.52),
            "recompute": (30.0 / 360.0, 0.80, 0.56),
            "bwd": (0.0 / 360.0, 0.55, 0.52),
        }
        hue, saturation, lightness = kind_hsl[event.kind]

        local_stage_count = max(1, config.num_stages // config.num_gpus)
        local_stage_idx = event.virtual_stage // config.num_gpus if config.num_gpus > 0 else 0

        if local_stage_count > 1:
            stage_norm = local_stage_idx / (local_stage_count - 1)
            lightness += (stage_norm - 0.5) * 0.28

        if config.num_gpus > 1:
            gpu_norm = event.stage / (config.num_gpus - 1)
            lightness += (gpu_norm - 0.5) * 0.08

        lightness = clamp(lightness, 0.28, 0.78)
        saturation = clamp(saturation, 0.35, 0.90)
        return hsl_to_hex(hue, saturation, lightness)

    fig_height = max(4.0, config.num_gpus * 0.7)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for event in events:
        y_bottom = event.stage + 0.1
        height = 0.8
        width = max(event.end - event.start, 0.0)

        ax.broken_barh(
            [(event.start, width)],
            (y_bottom, height),
            facecolors=event_facecolor(event),
            edgecolors="black",
            linewidth=0.3,
        )

    max_end_time = max(event.end for event in events) if events else 0.0
    ax.set_xlim(0, max_end_time * 1.02 if max_end_time > 0 else 1.0)
    ax.set_ylim(config.num_gpus + 0.2, 0)
    ax.set_xlabel("Time")
    ax.set_ylabel("GPU / Stage")
    ax.set_yticks([idx + 0.5 for idx in range(config.num_gpus)])
    ax.set_yticklabels([f"GPU{idx}" for idx in range(config.num_gpus)])
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

    for stage_idx in range(config.num_gpus + 1):
        ax.axhline(y=stage_idx, color="black", linewidth=0.4, alpha=0.3)

    ax.set_title(title)

    legend_handles = [
        Patch(facecolor=base_color_map["fwd"], edgecolor="black", label="Forward"),
        Patch(facecolor=base_color_map["recompute"], edgecolor="black", label="Recompute"),
        Patch(facecolor=base_color_map["bwd"], edgecolor="black", label="Backward"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline parallel schedule simulator."
    )
    parser.add_argument("--num-gpus", type=int, required=True, help="Number of GPUs / stages.")
    parser.add_argument("--num-microbatches", type=int, required=True, help="Number of microbatches.")
    parser.add_argument(
        "--stage-partition",
        type=str,
        required=True,
        help=(
            "Comma-separated transformer layer counts per virtual stage. "
            "For scheduler=roundpipe, use 'fwd0,fwd1,...|bwd0,bwd1,...'. "
            "For scheduler=1f1b, length must equal num_gpus; "
            "for scheduler=interleaved-1f1b, length must be a multiple of num_gpus."
        ),
    )
    parser.add_argument("--num-layers", type=int, required=True, help="Total transformer layer count.")
    parser.add_argument(
        "--fwd-time-per-layer",
        type=float,
        required=True,
        help="Forward time for one transformer layer on one microbatch.",
    )
    parser.add_argument(
        "--bwd-time-per-layer",
        type=float,
        required=True,
        help="Backward time for one transformer layer on one microbatch.",
    )
    parser.add_argument(
        "--lmhead-fwd-time",
        type=float,
        required=True,
        help=(
            "LM head forward time per microbatch. "
            "For scheduler=roundpipe, this runs in the first backward stage before backward."
        ),
    )
    parser.add_argument(
        "--lmhead-bwd-time",
        type=float,
        required=True,
        help="LM head backward time per microbatch (applies on last stage).",
    )
    parser.add_argument(
        "--comm-time",
        type=float,
        default=0.0,
        help="Point-to-point communication time for one hand-off between consecutive "
        "stages, charged only when the two stages sit on different GPUs.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="1f1b",
        choices=sorted(SCHEDULER_REGISTRY.keys()),
        help="Pipeline scheduling algorithm: 1f1b, gpipe, interleaved-1f1b, looped-bfs, or roundpipe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pp_schedule.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pipeline Schedule",
        help="Title for generated plot.",
    )
    parser.add_argument(
        "--print-schedule",
        action="store_true",
        help="Print operation-level schedule events to stdout.",
    )
    parser.add_argument(
        "--print-schedule-limit",
        type=int,
        default=200,
        help="Max number of operation events to print when --print-schedule is enabled (<=0 means all).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    fwd_partition, bwd_partition = parse_stage_partitions(args.stage_partition)
    config = SimulationConfig(
        num_gpus=args.num_gpus,
        num_microbatches=args.num_microbatches,
        stage_partition=fwd_partition,
        backward_stage_partition=bwd_partition,
        num_layers=args.num_layers,
        fwd_time_per_layer=args.fwd_time_per_layer,
        bwd_time_per_layer=args.bwd_time_per_layer,
        lmhead_fwd_time=args.lmhead_fwd_time,
        lmhead_bwd_time=args.lmhead_bwd_time,
        comm_time=args.comm_time,
    )
    config.validate()

    scheduler = SCHEDULER_REGISTRY[args.scheduler]
    layer_events, op_events, comm_stats = scheduler.simulate(config)

    plot_timeline(
        events=layer_events,
        config=config,
        output_path=args.output,
        title=args.title,
        scheduler_name=scheduler.name,
    )

    makespan, bubble_time, bubble_ratio = compute_bubble_stats(layer_events, config.num_gpus)
    breakdown = compute_bubble_breakdown(
        layer_events,
        config.num_gpus,
        has_warmup_phase=scheduler.has_warmup_phase,
    )
    total_capacity = makespan * config.num_gpus
    comm_exposed_ratio = comm_stats.exposed / total_capacity if total_capacity > 0 else 0.0
    print(f"scheduler={scheduler.name}")
    print(f"events={len(layer_events)}")
    print(f"makespan={makespan:.4f}")
    print(f"bubble_time={bubble_time:.4f}")
    print(f"bubble_ratio={bubble_ratio:.4%}")
    print(f"fill_bubble_time={breakdown['fill_time']:.4f}")
    print(f"fill_bubble_ratio={breakdown['fill_ratio']:.4%}")
    print(f"switch_bubble_time={breakdown['switch_time']:.4f}")
    print(f"switch_bubble_ratio={breakdown['switch_ratio']:.4%}")
    print(f"drain_bubble_time={breakdown['drain_time']:.4f}")
    print(f"drain_bubble_ratio={breakdown['drain_ratio']:.4%}")
    print(f"intra_bubble_time={breakdown['intra_time']:.4f}")
    print(f"intra_bubble_ratio={breakdown['intra_ratio']:.4%}")
    # Exposed comm is a slice of the bubble above, not a fifth category.
    print(f"comm_time_per_hop={comm_stats.comm_time:.4f}")
    print(f"comm_total_time={comm_stats.total:.4f}")
    print(f"comm_exposed_time={comm_stats.exposed:.4f}")
    print(f"comm_exposed_ratio={comm_exposed_ratio:.4%}")
    print(f"output={args.output}")

    if args.print_schedule:
        print_operation_schedule(op_events=op_events, limit=args.print_schedule_limit)


if __name__ == "__main__":
    main()