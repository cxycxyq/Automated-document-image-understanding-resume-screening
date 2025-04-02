import os
import json
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr, pearsonr
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 路径配置
model_path = "/root/autodl-tmp/qwen2.5/qwen/Qwen2.5-3B-Instruct/"
data_path = "/root/autodl-tmp/qwen2.5/train_score323/qwen_sft_training_data_reformatted.jsonl"
output_dir = "/root/autodl-tmp/qwen2.5/qwen2.5funsd/"

# 加载数据
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]
train_data, val_data = train_test_split(raw_data, test_size=0.1, random_state=42)
train_dataset = Dataset.from_list(train_data)
val_data_for_eval = val_data[:100]  # 用于 spearman 等评估
val_dataset = Dataset.from_list(val_data)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# BitsAndBytesConfig + QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

# 修复关键：适配 Qwen 的 LoRA 插入模块
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ✅ 检查是否有可训练参数
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Trainable params: {trainable_params}")
assert trainable_params > 0, "❌ 没有任何参数参与训练，LoRA 插入失败！"

# 训练配置
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=20,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=1,                        # 每一步都记录训练日志
    eval_steps=1,                          # 每5步评估一次
    save_steps=30,                         # 每50步保存一次
    evaluation_strategy="steps",            # 按步评估
    save_strategy="steps",                  # 按步保存
    save_total_limit=1,
    bf16=True,
    logging_dir=f"{output_dir}/logs",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_args,
    dataset_text_field="output",
    max_seq_length=2048
)

# 启动训练
train_result = trainer.train()
trainer.save_model(f"{output_dir}/final_model")

# 日志
with open(f"{output_dir}/summary.txt", "w") as f:
    f.write(json.dumps(train_result.metrics, indent=2, ensure_ascii=False))

# loss 可视化
log_history = trainer.state.log_history
df = pd.DataFrame([x for x in log_history if "loss" in x or "eval_loss" in x])
df.to_csv(f"{output_dir}/metrics.csv", index=False)
plt.figure()
if "loss" in df.columns:
    df["loss"].plot(label="Train Loss")
if "eval_loss" in df.columns:
    df["eval_loss"].plot(label="Eval Loss")
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Step")
plt.grid()
plt.savefig(f"{output_dir}/loss_curve.png")
