import pandas as pd
import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime

# === 参数设置 ===
data_path = "/root/autodl-tmp/bert/bert/train_crossencoder0/cross_encoder_train_data.csv"
model_path = "/root/autodl-tmp/bert/bert/ms-marco-MiniLM-L6-v2/"
output_dir = "/root/autodl-tmp/bert/bert/train_crossencoder/train1/"
os.makedirs(output_dir, exist_ok=True)
batch_size = 32
epochs = 20
warmup_steps = 100
evaluation_steps = 100
early_stopping_patience = 5

# === 加载数据 ===
df = pd.read_csv(data_path)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_samples = [InputExample(texts=[row['text_a'], row['text_b']], label=float(row['label'])) for _, row in train_df.iterrows()]
val_samples = [InputExample(texts=[row['text_a'], row['text_b']], label=float(row['label'])) for _, row in val_df.iterrows()]
test_df = df.sample(frac=0.1, random_state=1234)
test_samples = [InputExample(texts=[row['text_a'], row['text_b']], label=float(row['label'])) for _, row in test_df.iterrows()]

# === 初始化模型和 evaluator ===
model = CrossEncoder(model_path, num_labels=1)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size, collate_fn=model.smart_batching_collate)
val_evaluator = CECorrelationEvaluator.from_input_examples(val_samples, name="val")

# === 训练并自动保存最佳模型 ===
model.fit(
    train_dataloader=train_dataloader,
    evaluator=val_evaluator,
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path=output_dir,
    evaluation_steps=evaluation_steps,
    save_best_model=True
)

# === 验证并导出日志 + 图表 ===
history = pd.read_csv(os.path.join(output_dir, "val_evaluation_results.csv"), sep="\t")
history["epoch"] = history["epoch"] + 1
history.to_csv(os.path.join(output_dir, "metrics_history.csv"), index=False)

with open(os.path.join(output_dir, "summary.txt"), "w") as f:
    for _, row in history.iterrows():
        f.write(f"[Epoch {int(row['epoch'])}] Pearson: {row['pearson']:.4f}, Spearman: {row['spearman']:.4f}\n")

# === 绘制曲线 ===
plt.figure()
plt.plot(history["epoch"], history["spearman"], label="Spearman")
plt.plot(history["epoch"], history["pearson"], label="Pearson")
plt.xlabel("Epoch")
plt.ylabel("Correlation")
plt.title("Validation Metrics over Epochs")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "training_curve.png"))

# === 保存 config.json ===
config = {
    "model_path": model_path,
    "output_dir": output_dir,
    "batch_size": batch_size,
    "epochs": epochs,
    "warmup_steps": warmup_steps,
    "evaluation_steps": evaluation_steps,
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

# === 测试集评估与导出 ===
def evaluate_and_export(model, samples, out_csv):
    gold_scores = [sample.label for sample in samples]
    text_pairs = [sample.texts for sample in samples]
    pred_scores = model.predict(text_pairs)

    df_result = pd.DataFrame({
        "text_a": [t[0] for t in text_pairs],
        "text_b": [t[1] for t in text_pairs],
        "label": gold_scores,
        "predicted_score": pred_scores
    })
    df_result.to_csv(out_csv, index=False)

    mse = mean_squared_error(gold_scores, pred_scores)
    pearson, _ = pearsonr(gold_scores, pred_scores)
    spearman = spearmanr(gold_scores, pred_scores).correlation
    print(f"[Test Eval] MSE={mse:.4f}, Pearson={pearson:.4f}, Spearman={spearman:.4f}")
    return mse, pearson, spearman

print("\n🔍 Evaluating best model on test set...")
best_model = CrossEncoder(output_dir)
test_csv = os.path.join(output_dir, "test_predictions.csv")
evaluate_and_export(best_model, test_samples, test_csv)

# === 推理函数 ===
def predict(model_path, text1, text2):
    model = CrossEncoder(model_path)
    score = model.predict([(text1, text2)])
    print(f"[PREDICT] Score: {score[0]:.4f}")
    return score[0]

# 示例调用（可注释）：
# predict(output_dir, "我有五年Java开发经验", "招聘后台开发工程师，要求熟悉Java和微服务")

