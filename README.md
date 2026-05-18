# Visual Product Search Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-HNSW-green?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> Visual Recognition Course Project — DeepFashion In-Shop Clothes Retrieval

A query-by-image fashion search engine. The user uploads a photograph of a clothing item; the system returns the most visually and semantically similar products from the catalog — no text keywords required.

---

## Overview

The system operates in two phases:

**Offline Indexing** — Runs once over the catalog. Each gallery image is localized with YOLOv8, described with BLIP-2, encoded with fine-tuned CLIP, and indexed in a FAISS HNSW vector store.

**Online Retrieval** — Per query. The user's image is detected with YOLOS-Fashionpedia, a specific garment is selected, encoded with CLIP, searched against the FAISS index, and results are re-ranked with BLIP-2 ITM scoring.

---

## Repository Structure

```
VR-FINAL-PROJECT/
│
├── Offline Pipeline/
│   ├── 3c-fp1-p1-yolo.ipynb          # YOLOv8 training on DeepFashion bboxes
│   ├── 3c-blip.ipynb                 # BLIP-2 caption generation for gallery
│   ├── 3c-clip.ipynb                 # CLIP fine-tuning (5 seeds)
│   ├── 3c-faiss-indexing-fixed.py    # FAISS HNSW index construction
│   └── 3c-ablation-study.ipynb       # Ablation A / B / C evaluation
│
├── Online Pipeline/
│   └── streamlit-demo.ipynb          # Launches app_fixed.py on Kaggle
│
├── Batch Evaluation/
|   └── batch_eval.py                 # End-to-end metric computation script
```

---

## Models Used

| Role | Model | Fine-tuned |
|------|-------|-----------|
| Offline detection / cropping | YOLOv8 (custom-trained) | Yes — 3-class on DeepFashion |
| Online detection / cropping | `valentinafeve/yolos-fashionpedia` | No (off-the-shelf) |
| Semantic captioning | BLIP-2 | No (frozen) |
| Cross-modal embedding | `openai/clip-vit-base-patch32` | Yes — contrastive, 5 seeds |
| ANN indexing | FAISS HNSW | — |

---

## Results

Ablation study across three configurations, evaluated at K ∈ {5, 10, 15}. Config C results are reported as mean ± std over 5 random seeds.

| Config | Description | Recall@5 | NDCG@5 | mAP@5 |
|--------|-------------|----------|--------|-------|
| A | Vision-only CLIP (α=1.0) | 0.3856 | 0.1728 | 0.1250 |
| B | Frozen CLIP + BLIP-2 (α=0.7) | 0.4039 | 0.1831 | 0.1333 |
| **C** | **Fine-tuned CLIP + BLIP-2 (α=0.7)** | **0.8146 ± 0.001** | **0.5334 ± 0.002** | **0.4523 ± 0.002** |

---

## Running the Demo

The demo is designed to run on **Kaggle** with a T4 GPU. Open `Online Pipeline/streamlit-demo.ipynb` and run all cells. It will install dependencies, write `app_fixed.py`, and launch the Streamlit app via `localtunnel`.

---

## Batch Evaluation

```bash
# Mode 1 — DeepFashion query split (default)
python batch_eval.py

# Mode 2 — External ground truth CSV
python batch_eval.py --gt_csv ground_truth.csv --query_dir /path/to/images/
```

The CSV for Mode 2 must contain two columns: `query_image` and `item_id`.

Outputs `batch_eval_results.csv` with Recall@K, NDCG@K, and mAP@K.

---

## Team

| Name | Roll Number |
|------|------------|
| Sawant Hrushikesh Rahul | IMT2023619 |
| Satyam Ambi | IMT2023623 |
| Akshat Mittal | IMT2023606 |
| Ramkushal B | IMT2023601 |
