import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModel
import os

# === 配置路径 ===
data_path = "/root/autodl-tmp/bert/bert/MiniLM-train/cross_encoder_train_data1.csv"
model_name = "/root/autodl-tmp/bert/bert/all-MiniLM-L6-v2/"
model_save_path = "/root/autodl-tmp/bert/bert/all-MiniLM-L6-v2-funsd/"
summary_path = os.path.join(model_save_path, "summary.txt")
batch_size = 32
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 数据集类 ===
class BiEncoderDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.pairs = list(zip(df["text_a"], df["text_b"]))
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a, b = self.pairs[idx]
        label = self.labels[idx]
        return a, b, torch.tensor(label, dtype=torch.float)

# === 模型类 ===
class SentenceEmbeddingModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = lambda x: x.last_hidden_state.mean(dim=1)

    def forward(self, a_inputs, b_inputs):
        a_emb = self.pooling(self.encoder(**a_inputs))
        b_emb = self.pooling(self.encoder(**b_inputs))
        return a_emb, b_emb

# === 数据准备 ===
df = pd.read_csv(data_path)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def collate_fn(batch):
    texts_a, texts_b, labels = zip(*batch)
    a_inputs = tokenizer(list(texts_a), padding=True, truncation=True, return_tensors="pt")
    b_inputs = tokenizer(list(texts_b), padding=True, truncation=True, return_tensors="pt")
    return a_inputs, b_inputs, torch.tensor(labels)

train_loader = DataLoader(BiEncoderDataset(train_df, tokenizer), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(BiEncoderDataset(val_df, tokenizer), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# === 模型与优化 ===
model = SentenceEmbeddingModel(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
mse_loss_fn = nn.MSELoss()
cosine_sim = nn.CosineSimilarity(dim=1)

# === 日志文件 ===
with open(summary_path, "w") as f:
    f.write("Epoch\tTrain_MSE\tTrain_CosineLoss\tVal_Pearson\tVal_Spearman\n")

# === 训练循环 ===
for epoch in range(num_epochs):
    model.train()
    total_mse, total_cos, steps = 0, 0, 0

    for a_inputs, b_inputs, labels in train_loader:
        a_inputs = {k: v.to(device) for k, v in a_inputs.items()}
        b_inputs = {k: v.to(device) for k, v in b_inputs.items()}
        labels = labels.to(device)

        a_emb, b_emb = model(a_inputs, b_inputs)
        pred_scores = cosine_sim(a_emb, b_emb)

        mse_loss = mse_loss_fn(pred_scores, labels)
        cos_loss = 1 - cosine_sim(a_emb, b_emb).mean()

        loss = mse_loss + cos_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_mse += mse_loss.item()
        total_cos += cos_loss.item()
        steps += 1

    avg_mse = total_mse / steps
    avg_cos = total_cos / steps

    # === 验证评估 ===
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for a_inputs, b_inputs, labels in val_loader:
            a_inputs = {k: v.to(device) for k, v in a_inputs.items()}
            b_inputs = {k: v.to(device) for k, v in b_inputs.items()}
            a_emb, b_emb = model(a_inputs, b_inputs)
            preds = cosine_sim(a_emb, b_emb).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    pearson = pearsonr(all_preds, all_labels)[0]
    spearman = spearmanr(all_preds, all_labels)[0]

    print(f"Epoch {epoch+1} | MSE: {avg_mse:.4f} | CosLoss: {avg_cos:.4f} | Pearson: {pearson:.4f} | Spearman: {spearman:.4f}")
    with open(summary_path, "a") as f:
        f.write(f"{epoch+1}\t{avg_mse:.4f}\t{avg_cos:.4f}\t{pearson:.4f}\t{spearman:.4f}\n")

    torch.save(model.state_dict(), os.path.join(model_save_path, f"checkpoint-epoch{epoch+1}.pt"))