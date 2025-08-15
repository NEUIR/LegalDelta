#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python ../../opencompass/run.py \
  ../../opencompass/configs/eval_disclaw.py \
  --debug \
  > lawbench.log 2>&1 &