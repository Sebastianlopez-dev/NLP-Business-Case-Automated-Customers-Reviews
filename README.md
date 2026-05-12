# NLP Business Case — Automated Customer Reviews

> **Ironhack Data Science Bootcamp** · Sentiment Analysis + Category Clustering + Review Summarization  
> Dataset: Amazon Reviews 2023 (571M reviews, 33 categories) · Pipeline: 5 Jupyter notebooks

---

## Pipeline Overview

```
Amazon Reviews 2023 (UCSD)
         │
         ▼
   ┌─ N01 ──────────────────────────────────────┐
   │  Stream 33 categories × 30K from .jsonl.gz │
   │  Clean → Balance → Stratified 70/15/15     │
   │  Output: 213,945 reviews (Arrow)            │
   └──────────────────┬─────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   ┌─ N02 ──────────┐     ┌─ N03 ──────────┐
   │  DistilBERT     │     │  RoBERTa        │
   │  66M params     │     │  125M params    │
   │  F1 = 0.77 ✅   │     │  F1 = 0.69      │
   └────────┬────────┘     └────────┬────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
                 ┌─ N04 ───────────────────────────────┐
                 │  nomic-embed (768-dim)               │
                 │  MiniBatchKMeans (k=6)                │
                 │  6 semantic clusters                 │
                 │  Output: clusters.csv + profiles.json │
                 └──────────────────┬───────────────────┘
                                    ▼
                           ┌─ N05 ───────────────────────┐
                           │  Extractive (Python)         │
                           │  Abstractive (Gemini Flash)  │
                           │  6 markdown articles          │
                           └──────────────────────────────┘
```

---

## Quick Results

| Component | Model | Key Metric |
|-----------|-------|------------|
| **Sentiment** | DistilBERT (66M) | Accuracy **77.3%** · Weighted F1 **0.77** |
| **Sentiment** (comparison) | RoBERTa (125M) | Accuracy 69.3% · LR bottleneck identified |
| **Clustering** | nomic-embed + MiniBatchKMeans | k=6 · Silhouette 0.031 · 36× faster than K-Means |
| **Summarization** | Gemini 1.5 Flash | 6 articles · Extractive-Abstractive pipeline |
| **Web App** | HTML/Tailwind/Chart.js | Interactive dashboard with live HF Inference |

---

## N01 — Data Preparation & EDA

**Strategy**: Stream all 33 product categories directly from UCSD McAuley Lab — zero disk cache.

| Decision | Rationale |
|----------|-----------|
| 33 categories × 30K reviews | Maximize lexical diversity → prevent domain overfitting |
| `requests.get(stream=True)` + `gzip.GzipFile` | Avoids ~120 GB HuggingFace cache; reads `.jsonl.gz` line by line |
| Undersampling to minority class | Neutro (71,315) → final balanced dataset: 213,945 reviews |
| Stratified 70/15/15 split | Index-based leakage verification → zero false positives |
| Filtro `len(text) < 10` chars | Eliminates vacuous 5★ spam ("Great" ×1,960) while keeping "Not great." |

**Key findings**: 67% of raw corpus is 5★ · Neutral reviews are the *longest* (405 chars avg) · 8% of reviews are exact duplicates · Negative reviews cluster lexically — only 31 distinctive 1★ words vs 14,735 for 5★.

---

## N02 — DistilBERT Sentiment Classification

### Model Decision

```
                 ┌──────────────────────────────────┐
                 │  distilbert-base-uncased (66M)   │
                 │  ─────────────────────────────── │
                 │  Dropout: 0.3 (↑ from 0.1)       │
                 │  Max tokens: 256 (covers 98%)    │
                 │  LR: 2e-5 · Batch: 32 · Epochs: 5│
                 │  Training: ~80 min on T4 GPU     │
                 └──────────────────────────────────┘
```

### Training Evolution

```
Epoch 1  ████████████░░░░░░░░░░░░  75.0% acc · F1 0.747
Epoch 2  ███████████████░░░░░░░░░  76.7% acc · F1 0.767
Epoch 3  ████████████████░░░░░░░░  77.1% acc · F1 0.771  ← plateau starts
Epoch 4  █████████████████░░░░░░░  77.3% acc · F1 0.771
Epoch 5  █████████████████░░░░░░░  77.5% acc · F1 0.775
```

### Per-Class Performance

```
           Precision  Recall  F1     Support
Negative   ████████    0.754   0.781   0.767   10,697
Neutral    ██████      0.685   0.657   0.671   10,697  ← hardest class
Positive   █████████   0.875   0.880   0.877   10,698
──────────────────────────────────────────────────
Weighted               0.771   0.773   0.772   32,092
```

**Why DistilBERT over RoBERTa?** DistilBERT achieved 77.3% accuracy with 66M params vs RoBERTa's 69.3% with 125M. The 8-point gap traces to a learning rate bottleneck — RoBERTa's LR=1e-5 was half of what 125M params need. When both run at optimal settings, RoBERTa typically outperforms, but for this bootcamp timeline DistilBERT delivered faster convergence and better results with simpler tuning.

---

## N03 — RoBERTa (Comparison Model)

| Factor | DistilBERT | RoBERTa | Impact |
|--------|-----------|---------|--------|
| Parameters | 66M | 125M | 1.9× larger |
| Learning Rate | **2e-5** | **1e-5** | 🔴 Too low for model size |
| Batch Size | 32 | 16 | 🟠 Noisier gradients |
| Dropout | 0.3 | 0.1 | 🟡 Less regularization |
| Test Accuracy | **77.3%** | 69.3% | −8.0 pp |
| Neutral F1 | **0.671** | 0.587 | −8.4 pp |

**Root cause**: LR=1e-5 combined with batch_size=16 gives ~4× less effective update signal than DistilBERT's (2e-5 × 32/16). RoBERTa stagnated at epoch 3 (val loss ~0.711) — EarlyStopping never triggered because deltas were below threshold. Fix identified: bump LR to 2e-5 (deferred — requires Colab GPU time).

---

## N04 — Product Category Clustering

### Algorithm Decision

```
                     219,998 reviews
                           │
                    nomic-embed (768-dim)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       K-Means (Lloyd)          MiniBatchKMeans
       k=2…10 sweep             k=2…10 sweep
       n_init=10, max_iter=300  batch=1024, 10 epochs
              │                         │
              └────────────┬────────────┘
                           ▼
                     k=6 selected
              (elbow + domain interpretability)
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
       Final centroids           Label assignment
       (6, 768) float32          219,998 reviews
       SERVED WEIGHTS            PRODUCTION CHOICE
```

**Why two algorithms?** K-Means provides mathematically cleaner centroids (the weights you download). MiniBatchKMeans labels all 219,998 reviews at 36× speed (19.6s vs 56.9s) while retaining 102% of K-Means quality (Silhouette 0.0311 vs 0.0304 — within noise).

### K-Sweep Comparison

```
k=2   ██          Sil 0.037
k=3   ██          Sil 0.029
k=4   ██          Sil 0.030
k=5   ██          Sil 0.029
k=6   ██          Sil 0.030  ← selected: domain coherence + brief spec (4–6)
k=7   ██          Sil 0.030
k=8   ██          Sil 0.031
k=9   ██          Sil 0.035
k=10  ██          Sil 0.034
```

### Cluster Profiles

| # | Label | Size | ★ | Sentiment (Neg/Neu/Pos) | Top Terms |
|---|-------|------|---|------------------------|-----------|
| 0 | Salud & Belleza | 13.4% | 2.69 | 38% / **46%** / 17% | like, use, hair, smell, skin |
| 1 | Genérico Positivo | 5.6% | 4.74 | 0% / 7% / **93%** | good product, great, works |
| 2 | Libros & Entretenimiento | 17.6% | 2.98 | 31% / **43%** / 27% | book, story, read, magazine |
| 3 | Moda & Vestimenta | 16.3% | 2.96 | 26% / **53%** / 21% | fit, size, small, described |
| 4 | **Electrónica & Hogar** | **24.6%** | **1.94** | **66%** / 30% / 4% | work, money, cheap, quality |
| 5 | Juguetes & Regalos | 22.4% | 4.68 | 2% / 9% / **90%** | great, love, gift, easy, nice |

**Key insight**: Two cluster types emerged — **semantic** (0, 2, 3, 4: clear domain boundaries via TF-IDF + original categories) and **sentiment-driven** (1, 5: fragmented categories, >89% positive, avg rating >4.6★). Cluster 4 is the richest for insights: 66% negative, dominated by electronics complaints with a highly distinctive failure vocabulary ("doesn't work", "waste of money").

---

## N05 — Review Summarization

**Pipeline**: Extractive (Python facts) → Abstractive (Gemini 1.5 Flash styling)

| Phase | Tool | Role |
|-------|------|------|
| Extractive | pandas + NumPy | Hard facts: top products by `parent_asin`, sentiment stats, representative reviews |
| Abstractive | Gemini 1.5 Flash (Google API) | Polished blog-style article from structured facts |

**Output**: 6 markdown articles in `data/summaries/` — one per cluster. Each article includes top 3 products, key differences, top complaints, and the worst product to avoid.

**Anti-hallucination design**: Every claim in the final article is traceable to data. The LLM's job is *styling*, not *discovery* — product names, ratings, and statistics come from the extractive phase.

---

## Web Dashboard

Single-page interactive dashboard (`web/raw-web-vs3.html`):
- **Emotion Engine** tab: Live sentiment inference via HF Inference API (`distilbert-amazon-reviews-sentiment`)
- **Category Intelligence** tab: Cluster profiles from `hybridKMeans-category-clustering`
- **Training Evolution** chart: DistilBERT vs RoBERTa loss curves (Chart.js)
- Pipeline selector: "Full Pipeline" vs "Sentiment Only"

---

## How to Run

### Prerequisites
- Google Colab (recommended) or local Python 3.10+
- Google Drive mounted at `/content/drive/MyDrive/nlp-project/business-case-01/`
- Gemini API key (for N05 only)

### Setup
```bash
pip install -r requirements.txt
```

### Notebook Order
1. `notebook_01_data_prep_eda.ipynb` → produces `data/dataset/` (Arrow)
2. `notebook_02_distilbert_sentiment.ipynb` → fine-tunes sentiment model
3. `notebook_03_roberta_sentiment.ipynb` → comparison model (optional)
4. `notebook_04_category_clustering.ipynb` → clusters reviews into 6 categories
5. `notebook_05_review_summarisation.ipynb` → generates markdown articles

N02–N05 all consume the Arrow dataset from N01. Run sequentially — each notebook validates upstream artifacts before starting.

---

## Repository Structure

```
├── README.md                                   ← You are here
├── requirements.txt                            ← Python dependencies
├── concepts.md                                 ← Research behind every decision
├── analysisN01.md                              ← EDA findings & preprocessing decisions
├── analysisN02.md                              ← DistilBERT results & training analysis
├── analysisN03.md                              ← RoBERTa comparison & LR bottleneck
├── analysisN04.md                              ← Clustering pipeline & cluster profiles
├── notebook_01_data_prep_eda.ipynb             ← Data loading, cleaning, balancing
├── notebook_02_distilbert_sentiment.ipynb      ← DistilBERT fine-tuning
├── notebook_03_roberta_sentiment.ipynb         ← RoBERTa fine-tuning
├── notebook_04_category_clustering.ipynb       ← Embedding + clustering
├── notebook_05_review_summarisation.ipynb      ← Extractive-abstractive summarization
└── web/
    ├── raw-web-vs3.html                        ← Interactive dashboard
    └── train_history.js                        ← Training loss data for charts
```

---

## Key Decisions

| Decision | Why |
|----------|-----|
| Stream from UCSD, not HuggingFace Hub | Avoids 120 GB disk cache; `datasets>=2.19` API instability |
| Undersampling (not oversampling) | No synthetic data — transparent and reproducible |
| DistilBERT over RoBERTa for production | 77.3% accuracy in 66M params vs 69.3% in 125M (LR bottleneck) |
| nomic-embed (137M) over larger models | Fits CPU inference; 768-dim embeddings downloaded in seconds |
| MiniBatchKMeans over K-Means | 36× faster, 102% quality retention, scales to full 571M dataset |
| Extractive-Abstractive for N05 | Prevents LLM hallucination of products/numbers |
| Gemini 1.5 Flash over Mistral | Free tier sustainability; comparable quality for synthesis |
| Stratified split on indices (not text) | Zero leakage false positives vs text-based overlap checks |

---

## Evaluation Criteria Mapping

| Criterion | Points | Covered by |
|-----------|--------|------------|
| Data Preprocessing | 15 | N01 — streaming, cleaning, balancing, 16 EDA plots |
| Review Classification | 20 | N02 — DistilBERT F1=0.77 + N03 — RoBERTa comparison |
| Clustering Model | 20 | N04 — nomic-embed + MiniBatchKMeans k=6, k-sweep analysis |
| Summarization Model | 20 | N05 — extractive-abstractive pipeline, 6 articles |
| Deployment | 10 | Web dashboard with live HF Inference API |
| PDF Report | 5 | analysisN01–N04.md + this README |
| PPT Presentation | 10 | To be created from analysis docs |

**Total: up to 100 pts (+10 bonus for public hosting)**
