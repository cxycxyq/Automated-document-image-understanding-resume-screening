import json
import os
import torch
from PIL import Image
from datasets import Dataset
from transformers import LayoutLMv3Processor

MODEL_PATH = "/root/autodl-tmp/layoutlmv3_funsd/models/layoutlmv3_funsd/"
DATASET_PATH = "/root/autodl-tmp/LayoutXLM/train_en/data/final_labeled_output_split.json"
IMAGE_ROOT = "/root/autodl-tmp/dataset_linkedln/"
PROCESSED_DATA_PATH = "/root/autodl-tmp/LayoutXLM/train_en/data/dataloader1700/"
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)

LABELS = [
    "PERSONAL_INFO", "JOB_TITLE", "PROFESSIONAL_SUMMARY", "SKILL", "CERTIFICATION",
    "LANGUAGE", "EDUCATION", "WORK_EXPERIENCE", "HOBBY",
    "ADDITIONAL_ACTIVITY", "VOLUNTEERING", "PROJECT", "OTHERS"
]
label2id = {label: i for i, label in enumerate(LABELS)}

with open(DATASET_PATH, "r") as f:
    all_data = json.load(f)

processed_samples = []

for idx, sample in enumerate(all_data):
    image_path = os.path.join(IMAGE_ROOT, sample["image"])
    try:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        words = [x["text"] for x in sample["ocr_labeled"]]
        boxes = [list(map(int, x["bbox"])) for x in sample["ocr_labeled"]]
        word_labels = [label2id.get(x["label"], -1) for x in sample["ocr_labeled"]]

        normalized_boxes = [
            [max(0, min(1000, int(1000 * (coord / (width if i % 2 == 0 else height)))))
             for i, coord in enumerate(box)]
            for box in boxes
        ]

        encoding = processor(
            image,
            words,
            boxes=normalized_boxes,
            word_labels=word_labels,
            padding="max_length",
            truncation=True,
            max_length=512
        )

        processed_samples.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "bbox": encoding["bbox"],
            "labels": encoding["labels"]
        })

        if idx % 50 == 0:
            print(f"已处理 {idx}/{len(all_data)} 条数据")

    except Exception as e:
        print(f"❌ 跳过样本: {image_path} | 错误: {str(e)}")

# 确保至少有一个样本被成功处理
if len(processed_samples) == 0:
    raise ValueError("没有任何样本被成功处理！")

# 保存为.arrow文件（只做一次，之后训练直接读取）
dataset = Dataset.from_list(processed_samples)
dataset.save_to_disk(PROCESSED_DATA_PATH)
print("✅ 数据预处理完成，已保存！")