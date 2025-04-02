import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
import csv

# === 配置路径 ===
MODEL_PATH = "/root/autodl-tmp/layoutlmv3_funsd/models/layoutlmv3_funsd/"
PROCESSED_DATA_PATH = "/root/autodl-tmp/LayoutXLM/train_en/data/dataloader1700/"
OUTPUT_PATH = "/root/autodl-tmp/LayoutXLM/layoutlmv3_finetuned1700/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

LABELS = [
    "PERSONAL_INFO", "JOB_TITLE", "PROFESSIONAL_SUMMARY", "SKILL", "CERTIFICATION",
    "LANGUAGE", "EDUCATION", "WORK_EXPERIENCE", "HOBBY",
    "ADDITIONAL_ACTIVITY", "VOLUNTEERING", "PROJECT", "OTHERS"
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# 加载arrow数据
dataset = load_from_disk(PROCESSED_DATA_PATH)

# 划分训练和验证集
dataset = dataset.train_test_split(test_size=0.1, seed=42)

processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# 评价函数定义
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [LABELS[label] for (label, pred) in zip(label_row, pred_row) if label != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]

    true_predictions = [
        [LABELS[pred] for (label, pred) in zip(label_row, pred_row) if label != -100]
        for label_row, pred_row in zip(labels, predictions)
    ]

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# Loss和Metrics保存Callback
class MetricsSaverCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "train_loss", "eval_loss", "precision", "recall", "f1", "accuracy"])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            with open(self.log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([state.epoch, state.global_step, logs['loss'], '', '', '', '', ''])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            with open(self.log_path, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    state.epoch, state.global_step, '',
                    metrics.get('eval_loss', ''),
                    metrics.get('eval_precision', ''),
                    metrics.get('eval_recall', ''),
                    metrics.get('eval_f1', ''),
                    metrics.get('eval_accuracy', '')
                ])

# 设置训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_dir=os.path.join(OUTPUT_PATH, "logs"),
    logging_steps=1,
    fp16=False,
    dataloader_num_workers=4,  # 多进程加速数据加载
    report_to="none"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[MetricsSaverCallback(os.path.join(OUTPUT_PATH, "metrics_log.csv"))]
)

# 开始训练
trainer.train()

# 保存最佳模型和processor
model.save_pretrained(os.path.join(OUTPUT_PATH, "best"))
processor.save_pretrained(os.path.join(OUTPUT_PATH, "best"))

print("✅ 训练完成，模型和指标数据已保存！")
