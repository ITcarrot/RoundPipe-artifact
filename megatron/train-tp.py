#!/usr/bin/env python3
"""Megatron tensor-parallel throughput benchmark using NeMo.

Mirrors the FSDP benchmark in ../fsdp/train.py but uses NeMo's Megatron-Core
backend with tensor parallelism (TP=8), full activation recomputation, and
no CPU offloading.

Usage:
    torchrun --nproc_per_node=8 train-tp.py <model_name> <global_batch_size> <seq_length> <micro_batch_size>

    model_name can be a path like /model/Qwen3-1.7B or just the name Qwen3-1.7B.
    Only the basename is used to look up the NeMo model config.
    No actual model weights or tokenizer files are loaded (random init + mock data).
"""

import argparse
import os
import sys
import time

import lightning.pytorch as pl
import torch
torch.backends.cuda.matmul.allow_fp16_accumulation = True

import nemo.collections.llm as llm
import nemo.lightning as nl
from megatron.core.optimizer import OptimizerConfig

# ---- model registry ----------------------------------------------------------
# (config_class, model_class)
MODEL_REGISTRY = {
    "Qwen3-0.6B":        (llm.Qwen3Config600M, llm.Qwen3Model),
    "Qwen3-1.7B":        (llm.Qwen3Config1P7B, llm.Qwen3Model),
    "GPT-OSS-20B":       (llm.GPTOSSConfig20B, llm.GPTOSSModel),
    "Qwen3-32B":         (llm.Qwen3Config32B, llm.Qwen3Model),
    "Meta-Llama-3.1-8B": (llm.Llama31Config8B, llm.LlamaModel),
}

TP = 8  # tensor parallelism degree
WARMUP_STEPS = 5
MEASURE_STEPS = 10


# ---- throughput callback -----------------------------------------------------
class ThroughputCallback(pl.Callback):
    """Records wall-clock time over a measurement window and prints throughput."""

    def __init__(self, warmup_steps, measure_steps, seq_length, global_batch_size):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.seq_length = seq_length
        self.global_batch_size = global_batch_size
        self.start_time = None
        self.step_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_count += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        if rank == 0:
            if self.step_count <= self.warmup_steps:
                print(f"Warmup iteration {self.step_count - 1}")
            else:
                print(f"Measured iteration {self.step_count - self.warmup_steps - 1}")

        if self.step_count == self.warmup_steps:
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            self.start_time = time.perf_counter()
            if rank == 0:
                print(f"Warmup done ({self.warmup_steps} steps). Starting measurement...")

        if self.step_count == self.warmup_steps + self.measure_steps:
            torch.cuda.synchronize()
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            end_time = time.perf_counter()
            duration = end_time - self.start_time
            tokens = self.measure_steps * self.global_batch_size * self.seq_length
            if rank == 0:
                print(f"Time for {self.measure_steps} iterations: {duration:.2f}s")
                print(f"Throughput: {tokens / duration:.2f} tokens/s (global)")
            trainer.should_stop = True


# ---- main --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Megatron TP throughput benchmark")
    p.add_argument("model_name", help="Model name or path, e.g. /model/Qwen3-1.7B or Qwen3-1.7B")
    p.add_argument("global_batch_size", type=int, help="Global batch size (total across all GPUs)")
    p.add_argument("seq_length", type=int, help="Sequence length")
    p.add_argument("micro_batch_size", type=int, help="Micro batch size per GPU")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve model name (accept both paths and bare names)
    model_name = os.path.basename(args.model_name)
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}. Supported: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    config_cls, model_cls = MODEL_REGISTRY[model_name]
    model_kwargs = {}
    if model_name == "GPT-OSS-20B":
        model_kwargs["bias_activation_fusion"] = False
        model_kwargs["expert_model_parallel_size"] = TP
        model_kwargs["expert_tensor_parallel_size"] = 1
        model_kwargs["sequence_parallel"] = True

    # ---- model config --------------------------------------------------------
    config = config_cls(
        seq_length=args.seq_length,
        # Tensor parallelism (must match strategy)
        tensor_model_parallel_size=TP,
        # Full activation recomputation
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        # Precision
        fp16=True,
        params_dtype=torch.float16,
        # No offloading
        cpu_offloading=False,
        **model_kwargs,
    )
    model = model_cls(config=config)

    # ---- data (MockDataModule with default tokenizer) ------------------------
    # No tokenizer from model path needed: MockDataModule generates random data.
    # Its default GPT2 tokenizer vocab is smaller than all our models' embedding
    # tables, so random token IDs stay in-range.
    data = llm.MockDataModule(
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
    )

    # ---- callback ------------------------------------------------------------
    throughput_cb = ThroughputCallback(
        warmup_steps=WARMUP_STEPS,
        measure_steps=MEASURE_STEPS,
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
    )

    # ---- strategy ------------------------------------------------------------
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=TP,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=TP if model_name == "GPT-OSS-20B" else 1,
        sequence_parallel = True if model_name == "GPT-OSS-20B" else False,
        setup_optimizers=True,
    )

    # ---- precision -----------------------------------------------------------
    precision_plugin = nl.MegatronMixedPrecision(
        precision="fp16-mixed",
        params_dtype=torch.float16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )

    # ---- trainer -------------------------------------------------------------
    total_steps = WARMUP_STEPS + MEASURE_STEPS
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=TP,
        num_nodes=1,
        max_steps=total_steps,
        callbacks=[throughput_cb],
        strategy=strategy,
        plugins=precision_plugin,
        log_every_n_steps=1,
        enable_checkpointing=False,
        use_distributed_sampler=False,
        val_check_interval=total_steps + 1,  # disable validation
        limit_val_batches=0,
    )

    # ---- optimizer -----------------------------------------------------------
    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(lr=1e-4, use_distributed_optimizer=False),
    )

    # ---- train ---------------------------------------------------------------
    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        optim=optim,
    )


if __name__ == "__main__":
    main()
