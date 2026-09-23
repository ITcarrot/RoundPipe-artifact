#!/bin/bash

# 4090 24GB PP

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=7

torchrun --nproc_per_node=8 train.py Qwen3-1.7B 512 2048 4
torchrun --nproc_per_node=8 train.py Qwen3-1.7B 8 9216 1

torchrun --nproc_per_node=8 train.py Meta-Llama-3.1-8B 256 2048 2
torchrun --nproc_per_node=8 train.py Meta-Llama-3.1-8B 8 5120 1

# 4090 24GB TP

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=7

torchrun --nproc_per_node=8 train-tp.py Qwen3-1.7B 512 2048 16
torchrun --nproc_per_node=8 train-tp.py Qwen3-1.7B 8 63488 1

torchrun --nproc_per_node=8 train-tp.py Meta-Llama-3.1-8B 256 2048 8
torchrun --nproc_per_node=8 train-tp.py Meta-Llama-3.1-8B 8 20480 1

# A800 80GB PP

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8

torchrun --nproc_per_node=8 train.py Qwen3-1.7B 512 2048 16
torchrun --nproc_per_node=8 train.py Qwen3-1.7B 8 46080 1

torchrun --nproc_per_node=8 train.py Meta-Llama-3.1-8B 256 2048 16
torchrun --nproc_per_node=8 train.py Meta-Llama-3.1-8B 8 46080 1

torchrun --nproc_per_node=8 train.py GPT-OSS-20B 128 2048 4
torchrun --nproc_per_node=8 train.py GPT-OSS-20B 8 11264 1

torchrun --nproc_per_node=8 train.py Qwen3-32B 8 2048 1 # OOM

torchrun --nproc_per_node=8 train-lora.py Qwen3-235B 64 2048 2
torchrun --nproc_per_node=8 train-lora.py Qwen3-235B 8 7168 1

# A800 80GB TP

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8

torchrun --nproc_per_node=8 train-tp.py Qwen3-1.7B 512 2048 64
torchrun --nproc_per_node=8 train-tp.py Qwen3-1.7B 8 247808 1

torchrun --nproc_per_node=8 train-tp.py Meta-Llama-3.1-8B 256 2048 32
torchrun --nproc_per_node=8 train-tp.py Meta-Llama-3.1-8B 8 188416 1

torchrun --nproc_per_node=8 train-tp.py GPT-OSS-20B 128 2048 16
torchrun --nproc_per_node=8 train-tp.py GPT-OSS-20B 8 110592 1

torchrun --nproc_per_node=8 train-tp.py Qwen3-32B 128 2048 2
torchrun --nproc_per_node=8 train-tp.py Qwen3-32B 8 7168 1

torchrun --nproc_per_node=8 train-lora-tp.py Qwen3-235B 8 2048 1
# ValueError: num_query_groups (4) must be a multiple of tensor_model_parallel_size (8).
