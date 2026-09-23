#!/bin/bash

python train.py /public/huggingface-models/Qwen/Qwen3-1.7B 480 2048 10 12 stage
python train.py /public/huggingface-models/meta-llama/Meta-Llama-3.1-8B 240 2048 10 6 stage
python train.py /public/huggingface-models/openai/gpt-oss-20b 120 2048 10 3 stage
python train.py /public/huggingface-models/Qwen/Qwen3-32B 120 2048 10 3 stage
python train_lora.py /public/huggingface-models/Qwen/Qwen3-235B-A22B 1 120 2048 10 3
