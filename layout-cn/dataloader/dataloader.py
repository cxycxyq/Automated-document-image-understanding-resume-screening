import json
import os
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm
from datasets import Dataset
from transformers import LayoutLMv3Processor

# === 配置路径 ===
json_path = "/root/autodl-tmp/Layoutv3-CN/data/cn-labeled_split.json"
image_root = "/root/autodl-tmp/LayoutXLM/resume_train_20200121/pdf/"
save_path = "/root/autodl-tmp/Layoutv3-CN/model/"
model_path = "/root/autodl-tmp/LayoutXLM/layoutlmv3_finetuned1700/best/"

# 加载 processor 和标签
processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
LABEL_LIST = [
    "PERSONAL_INFO", "Nationality", "gender", "PHONE", "EMAIL", "BIRTHDAY",
    "NATIVE_PLACE", "REGISTERED_CITY", "POLITICAL_STATUS", "EDUCATION",
    "WORK_EXPERIENCE", "PROJECT_EXPERIENCE", "SKILL", "CERTIFICATION",
    "LANGUAGE", "SELF_EVALUATION", "HOBBY", "OTHER"
]
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

# 读取标注 JSON 数据
with open(json_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

# 分组：file_name + page_num 作为 key
data_by_page = {}
for item in all_data:
    resume_id = os.path.splitext(item["file_name"])[0]
    image_key = f"{resume_id}_page0"  # 默认先用 page0（后续可扩展）
    if image_key not in data_by_page:
        data_by_page[image_key] = []
    data_by_page[image_key].append(item)

processed_samples = []

for image_key, items in tqdm(data_by_page.items()):
    resume_id = image_key.replace("_page0", "")
    pdf_path = os.path.join(image_root, f"{resume_id}.pdf")

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=300)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        width, height = image.size

        words = [x["text"] for x in items]
        boxes = [x["bbox"] for x in items]
        word_labels = [label2id.get(x["label"], label2id["OTHER"]) for x in items]

        # === bbox 坐标归一化（按图片尺寸转换成 0-1000 区间） ===
        normalized_boxes = []
        for box in boxes:
            flat_box = [coord for pt in box for coord in pt]  # [x1,y1,x2,y2,x3,y3,x4,y4]
            norm_box = [
                max(0, min(1000, int(1000 * (coord / (width if i % 2 == 0 else height)))))
                for i, coord in enumerate(flat_box)
            ]
            normalized_boxes.append(norm_box)

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

    except Exception as e:
        print(f"❌ 跳过 {pdf_path}: {e}")

# 保存 arrow 数据集
if len(processed_samples) == 0:
    raise ValueError("❗ 没有成功处理任何样本！")

dataset = Dataset.from_list(processed_samples)
os.makedirs(save_path, exist_ok=True)
dataset.save_to_disk(save_path)
print(f"✅ 数据集已保存到 {save_path}")

