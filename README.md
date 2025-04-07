# Intelligent Resume Screening System

This repository contains the implementation of an end-to-end **intelligent resume filtering pipeline**, designed for multilingual (Chinese-English) document image understanding and candidate-job semantic matching.

> 🧠 Built with enhanced OCR, layout-aware field extraction, deep semantic matching, and interpretable LLM-based scoring.

---

## 🚀 Project Overview

Traditional resume screening methods struggle with scanned images, multilingual content, and complex layouts. This project proposes a **modular deep learning pipeline** to improve accuracy and interpretability in automated resume parsing and scoring.

### 🎯 Main Objectives

- Enhance OCR performance for bilingual resume images
- Extract structured information using LayoutLMv3
- Score resume-job pairs using fine-tuned Cross-Encoders
- Generate human-readable explanations with Qwen2.5
- Support local, offline deployment for data privacy and speed

---

## 🛠 System Architecture

This project implements a fully customized and fine-tuned deep learning pipeline, where models is trained or adapted using domain-specific bilingual resume data**.

1. **Image Super-Resolution**
   - 🔧 **Model Used**: Real-ESRGAN
   - 🧩 Purpose: Enhance low-quality or scanned resume images for better OCR accuracy.

2. **OCR & Preprocessing**
   - 🔧 **Model Used**: PaddleOCR
   - 🧩 Function: Extracts bounding boxes, recognized text, and structure annotations. Outputs reformatted for LayoutLM input.

3. **Field-Level Information Extraction**
   - 🔧 **Model Used**: LayoutLMv3 (fine-tuned with annotated resume fields)
   - 🧩 Trained to extract structured fields such as NAME, EDUCATION, WORK_EXPERIENCE, etc. on both English and Chinese resumes.

4. **Semantic Matching & Scoring**
   - 🔧 **Models Used**: Cross-Encoders (e.g., ms-marco-MiniLM, Erlangshen-Roberta-110M) — **both Trained to score Resume–Job data**
   - 🧩 Input: Resume–Job description pairs  
   - 🧠 Output: Relevance scores in the range [0.0, 1.0], aligned with human annotations.

5. **Interpretable Scoring via LLM**
   - 🔧 **Model Used**: Qwen2.5-3B-Instruct + LoRA (instruction fine-tuned)
   - 🧩 Purpose: Generates interpretable JSON output: `{"score": float, "reason": string}` for each resume-job match.

> 🔬 Models above are trained or fine-tuned using curated, bilingual, partially synthetic resume datasets to optimize performance for multilingual, real-world recruitment scenarios.


---

## 📊 Dataset Overview

| Source | Description |
|--------|-------------|
| Kaggle | 1,500 English resume images (multi-profession) |
| Alibaba Cloud Competition | 1,000+ manually constructed Chinese resumes (synthetic, anonymized) |
| Custom Annotations | 500+ resume–job relevance pairs with scores & reasoning |

All data used is either publicly available or anonymized, with no real personal information retained.

---

## 📈 Results

| Module | Metric | Result |
|--------|--------|--------|
| PaddleOCR + Real-ESRGAN | Chinese OCR Confidence | 98.84% |
| LayoutLMv3 | Field Extraction Accuracy | 0.778 |
| Cross-Encoder (EN) | Spearman | 0.956 |
| Cross-Encoder (CN) | Spearman | 0.907 |
| Qwen2.5 | Score + Reason Output | JSON format with detailed rationale |

---

## 🧩 Model Downloads (To be filled)

- [Download PaddleOCR config](#)
- [Download LayoutLMv3 weights](#)
- [Download Cross-Encoder (English)](#)
- [Download Cross-Encoder (Chinese)](#)
- [Download Qwen2.5 LoRA adapter](#)

---

## 🖥️ Environment & Deployment

- Python 3.8
- PyTorch 2.0
- Ubuntu 20.04
- vGPU 32GB or above (e.g., RTX 4090 supported)
- HuggingFace Transformers
- Local & offline inference supported

---

## 🧠 Citation

If you find this work useful, please cite:

```bibtex
@mastersthesis{chen2025resume,
  title={Automated Resume Filtering via Enhanced OCR and Deep Learning Pipeline},
  author={Chen, Xingyu},
  year={2025},
  school={Nanyang Technological University}
}
