#!/bin/bash

# A800 80G Zero2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 2048 zero2.json 8 8 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 33792 zero2.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /data/llama-3.1-8B 2048 zero2.json 8 4 --grad_chkpt
deepspeed --num_gpus=8 train.py /data/llama-3.1-8B 26624 zero2.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /public/huggingface-models/openai/gpt-oss-20b 2048 zero2.json 1 1 --grad_chkpt #OOM

deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 2048 zero2.json 1 1 --grad_chkpt #OOM

# A800 80G Zero-Infinity

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 2048 zero-inf.json 8 8 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 34816 zero-inf.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /data/llama-3.1-8B 2048 zero-inf.json 8 4 --grad_chkpt
deepspeed --num_gpus=8 train.py /data/llama-3.1-8B 34816 zero-inf.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /public/huggingface-models/openai/gpt-oss-20b 2048 zero-inf.json 8 2 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/openai/gpt-oss-20b 26624 zero-inf.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 2048 zero-inf.json 4 4 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 27648 zero-inf.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train-lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 2048 zero-inf.json 8 1 --grad_chkpt
deepspeed --num_gpus=8 train-lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 21504 zero-inf.json 1 1 --grad_chkpt

# 4090 24G Zero2

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 2048 zero2.json 2 32 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 7168 zero2.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 2048 zero2.json 1 1 --grad_chkpt # OOM

# 4090 24G Zero-Infinity

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO

deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 2048 zero-inf.json 2 32 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 9216 zero-inf.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 2048 zero-inf.json 2 16 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 9216 zero-inf.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /public/huggingface-models/openai/gpt-oss-20b 2048 zero-inf.json 2 8 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/openai/gpt-oss-20b 6144 zero-inf.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 2048 zero-inf.json 1 16 --grad_chkpt
deepspeed --num_gpus=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 6144 zero-inf.json 1 1 --grad_chkpt

deepspeed --num_gpus=8 train-lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 2048 zero-inf.json 1 1 --grad_chkpt # OOM
