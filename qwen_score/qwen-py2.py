import subprocess

# 运行模型推理脚本，生成中间结果
subprocess.run(["python3", "/root/autodl-tmp/qwen2.5/test/test323-new_reason_only.py"], check=True)

import json
import re

# === 路径配置 ===
input_path = "/root/autodl-tmp/qwen2.5/test/outcome/eval_results_qwen.json"
output_json_path = "/root/autodl-tmp/qwen2.5/test/outcome/final_cleaned_deduplicated.json"
output_txt_path = "/root/autodl-tmp/qwen2.5/test/outcome/final_cleaned_sorted_readable.txt"

# === 步骤 1：读取原始 JSON 文件 ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 步骤 2：处理每条记录 ===
processed_data = []

for item in data:
    for field in ["true_score", "reference_output", "reason_only"]:
        item.pop(field, None)

    raw_text = item.get("generated_output", "")

    # 保留从第二个 "score" 开始往后的内容
    score_matches = list(re.finditer(r'"score"\s*:', raw_text))
    if len(score_matches) >= 2:
        start_pos = score_matches[1].start()
        sliced = raw_text[start_pos:]
    else:
        sliced = raw_text

    # 删除后续重复的 "score": 出现的内容
    dedup_matches = list(re.finditer(r'"score"\s*:', sliced))
    if len(dedup_matches) >= 2:
        sliced = sliced[:dedup_matches[1].start()]

    item["generated_output"] = sliced.strip()

    # 提取实际的 score 值
    try:
        match = re.search(r'"score"\s*:\s*([0-9.]+)', sliced)
        if match:
            item["score"] = float(match.group(1))
        else:
            item["score"] = -1  # 若提取失败可设置默认值
    except:
        item["score"] = -1

    processed_data.append(item)

# === 步骤 3：保存清洗后的 JSON 文件 ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(processed_data, f, indent=2, ensure_ascii=False)

# === 步骤 4：根据 "score" 字段排序 ===
sorted_data = processed_data  # 保持原顺序

lines = []
for item in sorted_data:
    lines.append(
        f"ID: {item['id']}\n"
        f"Score: {item['score']}\n"
        f"Generated Output:\n{item['generated_output']}\n"
        + "-" * 80
    )

with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(lines))

print("✅ 完成：已按 score 排序生成 readable.txt 文件")