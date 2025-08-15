#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python ../../DISC-LawLLM-main/eval/src/main.py \
  --model_name qwen \
  --tasks mcq_sing \
  > disclaw.log 2>&1 &