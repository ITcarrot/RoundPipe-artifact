import argparse
import json
import os
import time
import contextlib

import deepspeed
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


torch.backends.cuda.matmul.allow_fp16_accumulation = True
transformers.modeling_utils._init_weights = False


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed causal LM microbenchmark")
    parser.add_argument("model_path", type=str, help="HF model name or path")
    parser.add_argument("seq_length", type=int, help="Token sequence length")
    parser.add_argument("config", type=str, help="DeepSpeed JSON config")
    parser.add_argument("bs_per_gpu", type=int, help="Batch size per GPU")
    parser.add_argument("accum_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed from distributed launcher")
    parser.add_argument("--grad_chkpt", action="store_true", help="Enable gradient checkpointing")
    return parser.parse_args()


def build_ds_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        ds_config = json.load(f)

    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    micro_batch = args.bs_per_gpu
    accum_steps = args.accum_steps

    train_batch_size = micro_batch * accum_steps * world_size
    # ds_config["train_batch_size"] = train_batch_size
    ds_config["gradient_accumulation_steps"] = accum_steps
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch

    use_fp16 = bool(ds_config.get("fp16", {}).get("enabled", False))
    use_bf16 = bool(ds_config.get("bf16", {}).get("enabled", False))
    if not use_fp16 and not use_bf16:
        ds_config.setdefault("fp16", {})["enabled"] = True
        use_fp16 = True
    if use_fp16 and use_bf16:
        raise ValueError("Config enables both fp16 and bf16; choose one")

    return ds_config


def main():
    args = parse_args()

    ds_config = build_ds_config(args)
    torch.distributed.init_process_group()

    micro_batch = args.bs_per_gpu
    accum_steps = args.accum_steps
    use_bf16 = bool(ds_config.get("bf16", {}).get("enabled", False))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    config._attn_implementation = "flash_attention_2"
    config.use_cache = False
    config.dtype = torch.bfloat16 if use_bf16 else torch.float16

    init_dtype = torch.bfloat16 if use_bf16 else torch.float16
    init_context = torch.device(f"cuda:{args.local_rank}") if ds_config['zero_optimization']['stage'] < 3 else deepspeed.zero.Init(config_dict_or_path=ds_config, dtype=init_dtype)
    with init_context:
        model = AutoModelForCausalLM.from_config(config)
        if args.grad_chkpt:
            print("Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules="all-linear",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, config=ds_config
    )

    device = model_engine.device
    total_batch = micro_batch * accum_steps
    all_input = torch.randint(0, tokenizer.vocab_size, (total_batch, args.seq_length), device=device)
    attention_masks = torch.ones_like(all_input)
    micro_inputs = all_input.chunk(accum_steps)
    micro_masks = attention_masks.chunk(accum_steps)

    def train_iter():
        for input_tensor, mask in zip(micro_inputs, micro_masks):
            outputs = model_engine(input_ids=input_tensor, attention_mask=mask, labels=input_tensor)
            loss = outputs.loss
            model_engine.backward(loss)
            model_engine.step()

    for i in range(5):
        if args.local_rank == 0:
                print("Warmup iteration", i)
        train_iter()

    start = time.perf_counter()
    for i in range(10):
        if args.local_rank == 0:
            print("Measured iteration", i)
        train_iter()
    end = time.perf_counter()

    if args.local_rank == 0:
        print(f"World Size: {os.environ.get('WORLD_SIZE', '1')}")
        print(f"Global Batch Size: {total_batch * int(os.environ.get('WORLD_SIZE', '1'))}")
        tokens = 10 * all_input.shape[0] * all_input.shape[1] * int(os.environ.get("WORLD_SIZE", "1"))
        print(f"Time taken for 10 iterations: {end - start} seconds")
        print(f"Throughput: {tokens / (end - start)} tokens/second")


if __name__ == "__main__":
    main()
