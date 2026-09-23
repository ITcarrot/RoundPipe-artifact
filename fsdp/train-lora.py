import argparse
import os
import time
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.loss.loss_utils import ForCausalLMLoss
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from peft import LoraConfig, TaskType, LoraModel

torch.backends.cuda.matmul.allow_fp16_accumulation = True

def parse_args():
    parser = argparse.ArgumentParser(description="FSDP causal LM benchmark")
    parser.add_argument("model_path")
    parser.add_argument("batch_size", type=int)
    parser.add_argument("seq_length", type=int)
    parser.add_argument("accum_steps", type=int)
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    parser.add_argument("--zero", type=int, choices=[0, 2, 3], default=3, help="FSDP sharding stage (0,2,3)")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offload for params and optimizer states")
    parser.add_argument("--grad_chkpt", action="store_true", help="Enable gradient checkpointing")
    return parser.parse_args()

def setup_distributed(local_rank: int):
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)

def build_model(args, device: torch.device):
    config = AutoConfig.from_pretrained(args.model_path)
    config._attn_implementation = "flash_attention_2"
    config.use_cache = False
    config.dtype = torch.float16
    # Build the model on meta device to avoid materializing full weights on every rank.
    with torch.device("meta"):
        base_model = AutoModelForCausalLM.from_config(config)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        )
        base_model = LoraModel(base_model, lora_config, "default").model
        for p in base_model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float16)

    if args.grad_chkpt:
        check_fn = lambda submodule: isinstance(submodule, GradientCheckpointingLayer)
        apply_activation_checkpointing(base_model, check_fn=check_fn)

    def param_init_fn(module):
        # Move this module's direct parameters to the target CUDA device without materializing data first.
        module.to_empty(device=device, recurse=False)
        # Let layers initialize themselves.
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    if args.zero == 0:
        sharding_strategy = ShardingStrategy.NO_SHARD
    elif args.zero == 2:
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    elif args.zero == 3:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    else:
        raise ValueError(f"Unsupported zero stage: {args.zero}")

    mp_policy = MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16)
    cpu_offload = CPUOffload(offload_params=True) if args.offload else None
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=(GradientCheckpointingLayer,))
    fsdp_model = FSDP(
        base_model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        device_id=device,
        mixed_precision=mp_policy,
        sync_module_states=True,
        use_orig_params=True,
        param_init_fn=param_init_fn,
        cpu_offload=cpu_offload,
    )
    return fsdp_model

def build_data(tokenizer, batch_size: int, seq_length: int, accum_steps: int, device: torch.device):
    tokens = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_length), device=device)
    attention = torch.ones_like(tokens, device=device)
    micro_inputs = tokens.chunk(accum_steps)
    micro_masks = attention.chunk(accum_steps)
    return micro_inputs, micro_masks

def train_step(model, micro_inputs, micro_masks, optim, accum_steps: int):
    model.train()
    optim.zero_grad(set_to_none=True)
    for idx, (input_ids, attn_mask) in enumerate(zip(micro_inputs, micro_masks)):
        sync_needed = accum_steps == 1 or idx == accum_steps - 1
        sync_context = nullcontext() if sync_needed else model.no_sync()
        with sync_context:
            with torch.autocast('cuda', dtype=torch.float16):
                loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids).loss
            loss.backward()
    optim.step()

def main():
    args = parse_args()
    setup_distributed(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    world_size = dist.get_world_size()

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = build_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    micro_inputs, micro_masks = build_data(
        tokenizer,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        accum_steps=args.accum_steps,
        device=device,
    )

    warmup_iters = 5
    measure_iters = 10

    for i in range(warmup_iters):
        if dist.get_rank() == 0:
            print("Warmup iteration", i)
        train_step(model, micro_inputs, micro_masks, optimizer, args.accum_steps)

    dist.barrier()
    start = time.perf_counter()
    for i in range(measure_iters):
        if dist.get_rank() == 0:
            print("Measured iteration", i)
        train_step(model, micro_inputs, micro_masks, optimizer, args.accum_steps)
    dist.barrier()
    end = time.perf_counter()

    if dist.get_rank() == 0:
        tokens = measure_iters * args.batch_size * args.seq_length * world_size
        duration = end - start
        print(f"Time for {measure_iters} iterations: {duration:.2f}s")
        print(f"Throughput: {tokens / duration:.2f} tokens/s (global)")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
