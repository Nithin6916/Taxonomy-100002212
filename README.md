# Taxonomy-2025
# Semantic Taxonomy Recovery (Icecat) — Unsupervised Clustering (L3)

This project discovers product taxonomy structure from raw Icecat product text using **fully unsupervised clustering**.  
**Category labels are never used for training/clustering** — they are used only at the end for evaluation.

---

## What we did (end-to-end)
1) **Load & preprocess** the Icecat JSON dataset (~1.2GB, ~490k products)  
2) **Build a document per product** from text fields (brand, name, bullet points, descriptions)  
   - includes **spec tokens** extracted from text (e.g., `gb:16`, `hz:144`, `res:1920x1080`) to improve separation of similar products  
3) **Embed documents** using SentenceTransformers (**BAAI/bge-m3**)  
4) **Dimensionality reduction (PCA)** to make large-scale clustering tractable  
5) **Scalable clustering**
   - BIRCH microclustering
   - Leiden community detection on a centroid similarity graph  
6) **Balanced consolidation** (optional merge) to reduce fragmentation and align closer to L3 granularity  
7) **Evaluate** with Purity + NMI/ARI/V-measure/AMI (labels used only here)

---

## Why PCA (why 256 dimensions)
PCA is an **unsupervised, label-free** transformation used for **efficiency and stability** at 490k scale:
- reduces compute/memory for kNN graph building + clustering
- removes noise in high-dimensional embedding space
- keeps semantic neighborhood structure largely intact

We used **PCA=256** as a practical “sweet spot”: high enough to retain semantic structure while making the full pipeline feasible and stable on large data. Labels are not used to fit PCA or tune it; labels are used only for final reporting.

---
## Dataset

The project uses the **Icecat ~1.2GB JSON dataset**.

- **Dataset access:** The Icecat JSON dataset is not included in this repository due to size and licensing restrictions.  
    For academic reproduction, the dataset is available via a private Google Drive link (shared with the instructor/TA) or can be obtained from the official Icecat source.
  
- **Size:** ~1.2 GB (raw), ~489,902 products (full dataset run)
- **Input Features (used for clustering):**
  - Product title / name
  - Brand
  - Bullet points
  - Short/long descriptions (cleaned)
  - Extracted **spec tokens** from text (e.g., `gb:16`, `hz:144`, `res:1920x1080`) — still label-free
- **Target (used only for evaluation/testing):**
  - Icecat taxonomy **Level-3 (L3)** category derived from the dataset taxonomy path  
  - **Never used for training, embedding, clustering, or merging decisions**


## Final Results (two variants)
We report two clusterings to show the well-known trade-off that **purity alone can be inflated by over-segmentation**.

| Variant | Purity | NMI | ARI | AMI | #Clusters |
|---|---:|---:|---:|---:|---:|
| purity_optimized | 0.8901 | 0.6669 | 0.0964 | 0.6621 | 366 |
| balanced | 0.8667 | 0.7270 | 0.3453 | 0.7242 | 238 |

Interpretation:
- **purity_optimized**: higher purity but fragmented partition (low ARI)
- **balanced**: slightly lower purity but much stronger agreement with the original taxonomy structure (higher ARI/NMI/AMI) and cluster count closer to L3

---


## Web App Demo (Streamlit)

This repo includes a lightweight web demo that assigns uploaded products (100–300 rows) into the learned taxonomy clusters.

### 1) Export inference artifacts (one-time)
```bat
python -m src.steps.11_export_inference_artifacts

## Repository Structure

- **src/**: Modular Python code (preprocessing, embeddings, clustering, evaluation, utilities).
  - **src/steps/**: Step-by-step pipeline scripts (01…09) to reproduce each stage.
  - **src/utils/**: Helper modules (text building, metrics, etc.).
  - **src/run_pipeline.py**: Optional single-command runner to execute the full pipeline end-to-end (reuses existing artifacts when available).

- **configs/**: Configuration files (e.g., `config.yaml`) for paths and algorithm parameters.

- **data/**:
  - **data/raw/**: Raw dataset input (e.g., `icecat_data_train.json`).
  - **data/interim/**: Intermediate artifacts (docs, labels parquet).
  - **data/processed/**: Large computed artifacts (embeddings `.mmap`, PCA outputs, cluster label `.npy`).

- **outputs/**: Generated reports and visualization artifacts.
  - **outputs/final/**: Final deliverables for submission (metrics CSV/JSON, plots PNG, final cluster label files).

- **requirements.txt**: Python dependencies (Torch installed separately for CUDA compatibility).
- **README.md**: Project overview + reproducible run instructions.

---
## Acknowledgements

**Project Supervisor:** Dr. Binh Vu (@binhvd)
- **For guidance on applying unsupervised learning and clustering techniques to e-commerce taxonomy discovery.

**Author:** Nithin Kiran
- **Created as part of the Taxonomy Unsupervised Learning project at **SRH University of Applied Sciences Heidelberg**.