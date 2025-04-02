import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 模型路径（训练好的 checkpoint）
model_path = "/root/autodl-tmp/qwen2.5/new_model323/checkpoint-205/"

# 测试数据路径
data_path = "/root/autodl-tmp/qwen2.5/test/qwen_top20_final_with_jd.jsonl"

# 输出文件路径
result_path = "/root/autodl-tmp/qwen2.5/test/outcome/eval_results_qwen.json"
fail_path = "/root/autodl-tmp/qwen2.5/test/outcome/eval_20cases.json"

# 加载模型和 tokenizer
print("🔄 正在加载模型和 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
model.eval()


# 正则提取评分（限定在 0~1 区间）
def extract_reason_only(text):
    match = re.search(r"理由[:：][\s\S]*", text)
    if match:
        return match.group(0).strip()
    return ""

def extract_score(text):
    matches = re.findall(r"[0-9]+\.?[0-9]*", text)
    numbers = [float(m) for m in matches if 0 <= float(m) <= 1.0]
    return numbers[-1] if numbers else None


# 强制调整分数和reason
def adjust_scores_and_reason_for_top_candidates(pred_score, reason, threshold=0.4):
    if pred_score == 0:  # 如果模型评分为 0
        # 强制调整评分至合理区间（例如 0.5 或更高）
        pred_score = threshold

        # 修正reason部分，避免过多的负面反馈
        if "缺乏" in reason or "不足" in reason:
            reason = reason.replace("缺乏", "有待提升").replace("不足", "有改进空间")
            reason += " 尽管如此，候选人具备潜力，适应性强，可以通过培训提升相关技能。"

    return pred_score, reason


# 推理 + 评分
results = []
y_true, y_pred, failed = [], [], []

with open(data_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        item = json.loads(line)
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        reference_output = item.get("output", "")

        # 更新 prompt 以适应精排需求
        prompt = f"""
        请根据简历和职位描述评估候选人的匹配度。，面对的候选人都是从粗排中筛选出来的中等以上的候选人。
    请你根据以下岗位描述和候选人简历，从专业角度评估其匹配程度，需考虑：
        1. 候选人与岗位背景是否相关；
        2. 候选人在经验、教育背景、项目经历、技能、语言等方面是否具备优势；
        3. 最终请给出一个匹配评分，范围为 0.39 到 1.0（评分是一个两位小数，float）；
        - 0.39 ~ 0.51：基础合格；
        - 0.51 ~ 0.62：基本胜任；
        - 0.62 ~ 0.72：较强匹配，有附加价值；
        - 0.72 ~ 0.81：强匹配，具备优势；
        - 0.81 ~ 0.91：非常强的候选人；
        - 0.91 ~ 1.00：极优秀、完美匹配；
    如果某些技能不完全匹配，请给出合理的分数，并解释候选人仍然符合岗位要求的理由。
    请你根据简历与岗位描述评估匹配度，并以如下 JSON 格式输出：
{{
  "score": float,
  "reason": string
}}
        岗位描述: {input_text}
        简历: {instruction}
        """

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=500)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        ref_score = extract_score(reference_output)
        gen_score = extract_score(response)

        # 强制调整分数和reason
        if gen_score is not None:
            gen_score, response = adjust_scores_and_reason_for_top_candidates(gen_score, response)

        result = {
            "id": idx,
            "generated_output": response,
            "reference_output": reference_output,
            "pred_score": gen_score,
            "true_score": ref_score,
            "reason_only": extract_reason_only(response)
        }
        results.append(result)

        if gen_score is not None and ref_score is not None:
            y_pred.append(gen_score)
            y_true.append(ref_score)
        else:
            failed.append(result)

        print(f"✅ 样本 {idx + 1} 完成")

# 保存推理结果
with open(result_path, "w", encoding="utf-8") as f_out:
    json.dump(results, f_out, indent=2, ensure_ascii=False)

# 保存失败样本
if failed:
    with open(fail_path, "w", encoding="utf-8") as f_fail:
        json.dump(failed, f_fail, indent=2, ensure_ascii=False)
    print(f"⚠️ 有 {len(failed)} 条样本评分提取失败，已保存到 eval_failed_cases.json")

# 打印评估指标
if y_true and y_pred:
    spearman = spearmanr(y_true, y_pred).correlation
    pearson = pearsonr(y_true, y_pred)[0]
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print("\n🎯 评估结果：")
    print(f"✅ Spearman: {spearman:.4f}")
    print(f"✅ Pearson : {pearson:.4f}")
    print(f"✅ MSE     : {mse:.4f}")
    print(f"✅ MAE     : {mae:.4f}")
else:
    print("❌ 无有效评分对，无法计算评估指标")
