#!/bin/bash

# 4090
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py /public/huggingface-models/Qwen/Qwen3-1.7B 512 2048 10 4 stage
python train.py /public/huggingface-models/Qwen/Qwen3-1.7B 10 74752 10 1 layer

python train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 256 2048 10 2 stage
python train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 10 50176 10 1 layer

python train.py /public/huggingface-models/openai/gpt-oss-20b 128 2048 16 1 stage
python train.py /public/huggingface-models/openai/gpt-oss-20b 1 39936 1 1 layer

python train.py /public/huggingface-models/Qwen/Qwen3-32B 128 2048 10 2 stage
python train.py /public/huggingface-models/Qwen/Qwen3-32B 1 28672 1 1 stage

python train_lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 1 64 2048 10 1
python train_lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 0 4 31744 4 1

# 4090 sync
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train-sync.py /public/huggingface-models/Qwen/Qwen3-1.7B 512 2048 10 7 stage
python train-sync.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 256 2048 10 5 stage
python train-sync.py /public/huggingface-models/openai/gpt-oss-20b 128 2048 16 1 stage
python train-sync.py /public/huggingface-models/Qwen/Qwen3-32B 128 2048 10 2 stage
python train_lora-sync.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 1 64 2048 10 1

# 4090 vary sequence length
export HF_ENDPOINT="https://hf-mirror.com"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py Qwen/Qwen3-1.7B 512 512 10 1 stage
python train.py Qwen/Qwen3-1.7B 256 1024 10 1 stage
python train.py Qwen/Qwen3-1.7B 128 2048 10 1 stage
python train.py Qwen/Qwen3-1.7B 60 4096 10 1 stage
python train.py Qwen/Qwen3-1.7B 30 8192 10 1 stage
python train.py Qwen/Qwen3-1.7B 10 16384 10 1 stage
python train.py Qwen/Qwen3-1.7B 10 32768 10 1 layer
python train.py Qwen/Qwen3-1.7B 10 65536 10 1 layer

# A800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train.py /public/huggingface-models/Qwen/Qwen3-1.7B 512 2048 10 1 stage
python train.py /public/huggingface-models/Qwen/Qwen3-1.7B 10 294912 10 1 layer

python train.py /data/llama-3.1-8B 256 2048 10 1 stage
python train.py /data/llama-3.1-8B 8 226304 8 1 layer

python train.py /public/huggingface-models/openai/gpt-oss-20b 128 2048 10 1 stage
python train.py /public/huggingface-models/openai/gpt-oss-20b 1 196608 1 1 layer

python train.py /public/huggingface-models/Qwen/Qwen3-32B 128 2048 10 2 stage
python train.py /public/huggingface-models/Qwen/Qwen3-32B 1 129024 1 1 stage

python train_lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 1 64 2048 10 1
python train_lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 0 1 120832 1 1

# A800 sync
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train-sync.py /public/huggingface-models/Qwen/Qwen3-1.7B 512 2048 10 4 stage
python train-sync.py /data/llama-3.1-8B 256 2048 10 5 stage
python train-sync.py /public/huggingface-models/openai/gpt-oss-20b 128 2048 10 1 stage
python train-sync.py /public/huggingface-models/Qwen/Qwen3-32B 128 2048 10 2 stage
python train_lora-sync.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 1 64 2048 10 1

# W7800
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export HF_ENDPOINT="https://hf-mirror.com"
python train.py Qwen/Qwen3-1.7B 512 2048 10 2 stage
python train.py meta-llama/Llama-3.1-8B 256 2048 10 2 stage
python train.py Qwen/Qwen3-32B 128 2048 10 4 stage
python train_lora.py Qwen/Qwen3-235B-A22B 0 64 2048 10 1

# 910B
export HF_ENDPOINT="https://hf-mirror.com"
python train.py Qwen/Qwen3-1.7B 512 2048 10 2 stage
python train.py meta-llama/Llama-3.1-8B 256 2048 10 1 stage
python train.py Qwen/Qwen3-32B 128 2048 10 1 stage
python train_lora.py Qwen/Qwen3-235B-A22B 0 64 2048 10 1
