import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from torch.utils.data import DataLoader

# === 参数设置 ===
data_path = "/root/autodl-tmp/cn-cross/train/data/CN_cross_encoder_train_data.csv"
model_path = "/root/autodl-tmp/cn-cross/Erlangshen-Roberta-110M-Similarity/"
output_dir = "/root/autodl-tmp/cn-cross/output/"
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
# === 初始化模型和 evaluator ===
model = CrossEncoder(
    model_path,
    num_labels=1,
    max_length=512,  # 加上 max_length 参数，确保每个输入最多 512 tokens
    default_activation_function=None,
    automodel_args={"ignore_mismatched_sizes": True}
)

# === 创建 DataLoader ===
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size, collate_fn=model.smart_batching_collate)

# === 初始化验证评估器 ===
val_evaluator = CECorrelationEvaluator.from_input_examples(val_samples, name="val")


# === 训练并保存验证分数 ===
val_loss_history = []

def custom_callback(score, epoch, steps):
    print(f"Epoch: {epoch}, Steps: {steps}, Score: {score}")
    val_loss_history.append(score)

model.fit(
    train_dataloader=train_dataloader,
    evaluator=val_evaluator,
    epochs=epochs,
    warmup_steps=warmup_steps,
    output_path=output_dir,
    evaluation_steps=evaluation_steps,
    save_best_model=True,
    callback=custom_callback
)

# === 绘图：验证相关性指标 ===
val_result_path = os.path.join(output_dir, "val_evaluation_results.csv")
if os.path.exists(val_result_path):
    history = pd.read_csv(val_result_path, sep="\t")
    history["epoch"] = history["epoch"] + 1
    plt.figure()
    plt.plot(history["epoch"], history["spearman"], label="Spearman")
    plt.plot(history["epoch"], history["pearson"], label="Pearson")
    plt.xlabel("Epoch")
    plt.ylabel("Correlation")
    plt.title("Validation Correlation Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "val_correlation_curve.png"))

# === 测试集评估 ===
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


