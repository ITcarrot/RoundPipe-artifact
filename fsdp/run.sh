#!/bin/bash

# A800 80GB

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 64 2048 4 --zero 3 --grad_chkpt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 1 38912 1 --zero 3 --grad_chkpt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /data/llama-3.1-8B 32 2048 4 --zero 3 --grad_chkpt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /data/llama-3.1-8B 1 29696 1 --zero 3 --grad_chkpt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /public/huggingface-models/openai/gpt-oss-20b 4 2048 1 --zero 3 --grad_chkpt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /public/huggingface-models/openai/gpt-oss-20b 1 18432 1 --zero 3 --grad_chkpt

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 16 2048 16 --zero 3 --grad_chkpt # OOM
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 2 2048 1 --zero 3 --grad_chkpt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 1 11264 1 --zero 3 --grad_chkpt

# Must apply PR before run: https://github.com/pytorch/pytorch/pull/168333
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 train-lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 1 2048 1 --zero 3 --grad_chkpt # OOM

# 4090 24GB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=7
torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 64 2048 32 --zero 3 --grad_chkpt
torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 1 10240 1 --zero 3 --grad_chkpt

torchrun --nproc_per_node=8 train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 2 2048 1 --zero 3 --grad_chkpt
torchrun --nproc_per_node=8 train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 1 5120 1 --zero 3 --grad_chkpt

# 4090 24GB offload
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=7
torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 64 2048 16 --zero 3 --grad_chkpt --offload
torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-1.7B 1 11264 1 --zero 3 --grad_chkpt --offload

torchrun --nproc_per_node=8 train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 4 2048 1 --zero 3 --grad_chkpt --offload
torchrun --nproc_per_node=8 train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 1 11264 1 --zero 3 --grad_chkpt --offload

torchrun --nproc_per_node=8 train.py /public/huggingface-models/Qwen/Qwen3-32B 1 10 1 --zero 3 --grad_chkpt --offload # OOM
