#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python ../../src/opencompass/run.py \
  ../../src/opencompass/configs/eval_disclaw.py \
  --debug \
  > lawbench.log 2>&1 &