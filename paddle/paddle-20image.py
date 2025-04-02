import os
import pandas as pd
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import Levenshtein
import numpy as np
import re

# ==== 初始化 OCR ====
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 英文模型

# ==== 路径配置 ====
csv_path = "/root/autodl-tmp/paddle328/data/csv/_20_____.csv"
pdf_dir = "/root/autodl-tmp/paddle328/data/image-en/"
output_dir = "/root/autodl-tmp/paddle328/outcome/paddle/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "ocr_eval_summary.txt")

# ==== 加载真实文本 ====
df = pd.read_csv(csv_path)
id_to_text = dict(zip(df['ID'].astype(str), df['Resume_str'].astype(str)))

# ==== 指标累计变量 ====
total_chars = total_words = total_char_errs = total_word_errs = 0
total_confidence = 0.0
matched_files = 0

# ==== 遍历 PDF ====
for filename in os.listdir(pdf_dir):
    if not filename.endswith(".pdf"):
        continue

    file_id_match = re.findall(r"\d+", filename)
    if not file_id_match:
        continue
    file_id = file_id_match[0]

    if file_id not in id_to_text:
        continue

    true_text = id_to_text[file_id]
    true_words = true_text.split()
    total_chars += len(true_text)
    total_words += len(true_words)

    pdf_path = os.path.join(pdf_dir, filename)
    pages = convert_from_path(pdf_path, dpi=300)

    ocr_text = ""
    confidences = []

    for page_num, img in enumerate(pages):
        try:
            img_np = np.array(img)
            result = ocr.ocr(img_np, cls=True)

            if result and result[0]:
                for line in result[0]:
                    text = line[1][0].strip()
                    conf = line[1][1]
                    ocr_text += text + " "
                    confidences.append(conf)
        except Exception as e:
            print(f"[⚠️ ERROR] 文件 {filename} 第 {page_num+1} 页识别失败，跳过。错误：{e}")

    ocr_text = ocr_text.strip()
    ocr_words = ocr_text.split()

    char_distance = Levenshtein.distance(ocr_text, true_text)
    word_distance = Levenshtein.distance(" ".join(ocr_words), " ".join(true_words))
    cer = char_distance / max(1, len(true_text))
    wer = word_distance / max(1, len(true_words))
    avg_conf = np.mean(confidences) if confidences else 0.0

    total_char_errs += char_distance
    total_word_errs += word_distance
    total_confidence += avg_conf
    matched_files += 1

    # 单文件输出
    print(f"\n📄 文件：{filename}")
    print(f"字符级准确率：{1 - cer:.2%}")
    print(f"单词级准确率：{1 - wer:.2%}")
    print(f"CER（字符错误率）：{cer:.2%}")
    print(f"WER（单词错误率）：{wer:.2%}")
    print(f"OCR 平均置信度：{avg_conf:.2%}")

# ==== 汇总输出 ====
avg_char_acc = 1 - total_char_errs / max(1, total_chars)
avg_word_acc = 1 - total_word_errs / max(1, total_words)
avg_cer = total_char_errs / max(1, total_chars)
avg_wer = total_word_errs / max(1, total_words)
avg_conf = total_confidence / max(1, matched_files)

summary = f"""
====== 📊 总体评估 ======
平均字符准确率：{avg_char_acc:.2%}
平均单词准确率：{avg_word_acc:.2%}
CER（平均字符错误率）：{avg_cer:.2%}
WER（平均单词错误率）：{avg_wer:.2%}
OCR 平均置信度：{avg_conf:.2%}
共处理文档：{matched_files}
"""

print(summary)

# ==== 保存文件 ====
with open(output_file, "w", encoding="utf-8") as f:
    f.write(summary)
