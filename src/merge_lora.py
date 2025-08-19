import argparse
import json
import os

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, 
                    default=None)
parser.add_argument("--save_path", type=str,
                    default=None)
parser.add_argument("--base_model_path", type=str,
                    default=None, help="覆盖配置中的基础模型路径")

args = parser.parse_args()
config = PeftConfig.from_pretrained(args.model_name_or_path)

# 如果提供了base_model_path参数，则覆盖config中的路径
if args.base_model_path:
    base_model_path = args.base_model_path
else:
    base_model_path = config.base_model_name_or_path

base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
model = PeftModel.from_pretrained(model, args.model_name_or_path)

model = model.merge_and_unload()
model.save_pretrained(args.save_path)
base_tokenizer.save_pretrained(args.save_path)