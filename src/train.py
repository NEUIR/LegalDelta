import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["UNSLOTH_COMPILE_OVERWRITE"] = "0"
os.environ["UNSLOTH_CACHE_DIR"] = "path/to/yours/Legal_Delta/scripts/unsloth_compiled_cache"
import torch._dynamo
import torch
import re
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import json
import swanlab
import numpy as np
from trl import GRPOConfig, GRPOTrainer
from unsloth import is_bfloat16_supported
from reward import enhanced_scoring_function_v2, xmlcount_reward_func
from peft import LoraConfig, get_peft_model 
import time
from datetime import datetime

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True


experiment_name = f"qwen14B_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", f"{experiment_name}.log")

def log_print(message):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

swanlab.init(
    project="GRPO_training", 
    name=experiment_name,
    config={
        "model": "qwen14B",
        "lora_rank": 32,
        "learning_rate": 5e-5,
        "max_steps": 1000,
        "batch_size": 2,
    },
    settings=swanlab.Settings(init_timeout=120)
)

per_device_batch_size = 2
gradient_accumulation_steps = 16
num_gpus = torch.cuda.device_count()  
max_grad_norm = 0.1
bf16 = is_bfloat16_supported()
fp16 = not is_bfloat16_supported()
max_seq_length = 32760 
lora_rank = 32  


start_time = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/data2/xubuqiang/qwen14sft",
    max_seq_length=max_seq_length,
    load_in_4bit=False,  
    fast_inference=True,  
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6, 
)


SYSTEM_PROMPT_base = """
用户与助手之间的对话。用户提出问题，助手解决它。答案包含在 <answer> </answer> 标签中。

请按照以下格式回答问题：
<answer>
在此提供简洁明确的最终答案。
</answer>
"""


SYSTEM_PROMPT = """
用户与助手之间的对话。用户提出问题，助手解决它。助手首先思考推理过程，然后提供用户答案。推理过程和答案分别包含在 <reasoning> </reasoning> 和 <answer> </answer> 标签中。

请按照以下格式回答问题：
<reasoning>
在此详细分析问题并展示完整的推理过程，包括思考步骤、相关知识和逻辑分析。
</reasoning>
<answer>
在此提供简洁明确的最终答案。
</answer>
"""

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  
    random_state=3407,
)

# ========== read dataset ==========
def get_questions(split="train") -> Dataset:
    if split == "train":
        file_path = "../data/GRPO_training.json"
    elif split == "valid":
        file_path = "../data/GRPO_dev.json"
    data = load_dataset("json", data_files=file_path) 
    data = data['train']  
    
    EXAMPLE_TEXT ="""
    你是一个法律专家，请按照以下要求回答：
     """
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f"[QUERY_ID:{x['id']}]\n" + EXAMPLE_TEXT + x['instruction'] + x['question']}
        ],
        'prompt_base': [
            {'role': 'system', 'content': SYSTEM_PROMPT_base},
            {'role': 'user', 'content': f"[QUERY_ID:{x['id']}]\n" + EXAMPLE_TEXT + x['instruction_base'] + x['question']},
            {'role': 'assistant', 'content': f"<answer>\n{x['answer']}\n</answer>"}
        ],
        'answer': x['answer'],
        'id': x['id'] 
    })

    return data


train_dataset = get_questions(split="train")
val_dataset = get_questions(split="valid")

# ========== training config ==========
training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_generations=6,  
    per_device_eval_batch_size=6,
    max_prompt_length=4096,
    max_completion_length=512,
    num_train_epochs=5, 
    eval_strategy="epoch",
    save_strategy="epoch",
    do_eval=True,
    logging_strategy="steps",
    log_level="info",  
    max_grad_norm=max_grad_norm,
    report_to="tensorboard",
    output_dir=os.path.join("outputs", experiment_name),
    local_rank=-1,
    deepspeed=None,
)

# ========== trainer ==========
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[xmlcount_reward_func, enhanced_scoring_function_v2],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# ========== training ==========
trainer.train()