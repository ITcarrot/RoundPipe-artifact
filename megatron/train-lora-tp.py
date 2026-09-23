#!/usr/bin/env python3
"""Megatron tensor-parallel LoRA throughput benchmark using NeMo.

Mirrors the FSDP benchmark in ../fsdp/train.py but uses NeMo's Megatron-Core
backend with tensor parallelism (TP=8), full activation recomputation, and
no CPU offloading.

Usage:
    torchrun --nproc_per_node=8 train-lora-tp.py <model_name> <global_batch_size> <seq_length> <micro_batch_size>

    model_name can be a path like /model/Qwen3-1.7B or just the name Qwen3-1.7B.
    Only the basename is used to look up the NeMo model config.
    No actual model weights or tokenizer files are loaded (random init + mock data).
"""

import argparse
import inspect
import os
import sys
import time
from typing import Dict

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
torch.backends.cuda.matmul.allow_fp16_accumulation = True

import nemo.collections.llm as llm
import nemo.lightning as nl
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from megatron.core.optimizer import OptimizerConfig

# ---- model registry ----------------------------------------------------------
# (config_class, model_class)
MODEL_REGISTRY = {
    "Qwen3-30B": (llm.Qwen3Config30B_A3B, llm.Qwen3Model),
    "Qwen3-235B": (llm.Qwen3Config235B_A22B, llm.Qwen3Model),
}

TP = 8
EP = 8
WARMUP_STEPS = 5
MEASURE_STEPS = 10
LORA_RANK = 32


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


class _RandomGPTDataset(Dataset):
    """Random GPT-style samples with Megatron expected tensor keys."""

    def __init__(self, vocab_size: int, num_samples: int, seq_length: int, seed: int = 42):
        self.vocab_size = int(vocab_size)
        self.length = int(num_samples)
        self.seq_length = int(seq_length)
        self.seed = int(seed)
        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        np_gen = np.random.default_rng(seed=self.seed + idx)
        tokens = torch.from_numpy(np_gen.integers(self.vocab_size, size=[self.seq_length + 1], dtype=np.int64))
        return {
            "tokens": tokens[:-1],
            "labels": tokens[1:],
            "loss_mask": self.loss_mask,
            "position_ids": self.position_ids,
        }


class OfflineMockDataModule(pl.LightningDataModule):
    """Offline random data module, no tokenizer/network dependency."""

    def __init__(
        self,
        seq_length: int,
        vocab_size: int,
        micro_batch_size: int,
        global_batch_size: int,
        rampup_batch_size=None,
        num_train_samples: int = 10_000_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        self._train_ds = _RandomGPTDataset(self.vocab_size, self.num_train_samples, self.seq_length, seed=42)
        self._validation_ds = _RandomGPTDataset(self.vocab_size, self.num_val_samples, self.seq_length, seed=4242)
        self._test_ds = _RandomGPTDataset(self.vocab_size, self.num_test_samples, self.seq_length, seed=2424)

    def _create_dataloader(self, dataset):
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=default_collate,
        )

    def train_dataloader(self):
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self):
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self):
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def reconfigure_limit_batches(self):
        from nemo.collections.llm.gpt.data.utils import _reconfigure_limit_batches

        self.trainer.limit_train_batches = _reconfigure_limit_batches(self.trainer.limit_train_batches, self._train_ds)
        self.trainer.limit_val_batches = _reconfigure_limit_batches(self.trainer.limit_val_batches, self._validation_ds)

        try:
            from megatron.core.num_microbatches_calculator import get_num_microbatches
        except (ImportError, ModuleNotFoundError):
            from apex.transformer.pipeline_parallel.utils import get_num_microbatches

        self.trainer.num_sanity_val_steps *= get_num_microbatches()


def _get_vocab_size(config):
    for attr in ("padded_vocab_size", "vocab_size"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise RuntimeError("Could not infer vocab size from config.")


def _build_lora_transform(rank, model_name):
    peft_ns = getattr(llm, "peft", None)
    if peft_ns is None:
        raise RuntimeError("This NeMo build does not expose llm.peft namespace.")

    lora_cls = getattr(peft_ns, "LoRA", None)
    if lora_cls is None:
        raise RuntimeError("llm.peft.LoRA is not available.")

    return lora_cls(
        target_modules=[
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
        ],
        dim=rank,
        alpha=rank,
        dropout=0.0,
    )


def _run_lora_finetune(model, data, trainer, optim, lora_transform):
    finetune_fn = getattr(llm, "finetune", None)
    if finetune_fn is None:
        raise RuntimeError("llm.finetune is not available in this NeMo version.")

    kwargs = {
        "model": model,
        "data": data,
        "trainer": trainer,
        "optim": optim,
    }

    sig = inspect.signature(finetune_fn)
    if "peft" in sig.parameters:
        kwargs["peft"] = lora_transform
    elif "model_transform" in sig.parameters:
        kwargs["model_transform"] = lora_transform
    else:
        raise RuntimeError("Unable to inject LoRA: finetune() has no peft/model_transform argument.")

    finetune_fn(**kwargs)


# ---- main --------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Megatron TP LoRA throughput benchmark")
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
    vocab_size = _get_vocab_size(config_cls())

    extra_model_kwargs = {}
    # Important for MoE LoRA coverage:
    # grouped GEMM can turn experts into grouped linear modules that share one adapter.
    # Disable it to expose per-expert linear modules for LoRA (closer to HF PEFT behavior).
    extra_model_kwargs["moe_grouped_gemm"] = False

    # ---- model config --------------------------------------------------------
    config = config_cls(
        seq_length=args.seq_length,
        # Tensor parallelism (must match strategy)
        tensor_model_parallel_size=TP,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
        # Full activation recomputation
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        # Precision
        fp16=True,
        params_dtype=torch.float16,
        # No offloading
        cpu_offloading=False,
        **extra_model_kwargs,
    )
    model = model_cls(config=config)

    # ---- data (MockDataModule with default tokenizer) ------------------------
    # No tokenizer from model path needed: MockDataModule generates random data.
    # Its default GPT2 tokenizer vocab is smaller than all our models' embedding
    # tables, so random token IDs stay in-range.
    data = OfflineMockDataModule(
        seq_length=args.seq_length,
        vocab_size=vocab_size,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        num_workers=0,
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
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
        sequence_parallel=True,
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
        devices=EP,
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

    lora_transform = _build_lora_transform(rank=LORA_RANK, model_name=model_name)

    # ---- finetune (LoRA) ----------------------------------------------------
    _run_lora_finetune(
        model=model,
        data=data,
        trainer=trainer,
        optim=optim,
        lora_transform=lora_transform,
    )


if __name__ == "__main__":
    main()
