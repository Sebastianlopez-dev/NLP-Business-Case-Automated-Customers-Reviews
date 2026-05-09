# Analysis — Notebook 02: DistilBERT Sentiment Classification

> **Generated**: 2026-05-09 | **Audited**: 2026-05-09  
> **Notebook**: `notebook_02_distilbert_sentiment.ipynb`  
> **Status**: ✅ Complete — 56 cells (10 markdown, 46 code)

---

## Executive Summary

Notebook 02 fine-tunes `distilbert-base-uncased` for 3-class sentiment classification
on the preprocessed Amazon Reviews dataset (N01). The model achieves **77.26% test accuracy**
with a weighted F1 of 0.7718 [Cell 40/41 output]. The dataset is **perfectly balanced** across
all three classes (~33.3% each in every split) [Cell 13 output].

---

## Notebook Structure

| Section | Cells | Purpose |
|---------|-------|---------|
| Setup (0.x) | 0-8 | Env detection, imports, seeds, paths, GPU check |
| Input Validation | 6-7 | Verify N01 Arrow artifacts exist |
| §1 — Load Dataset | 9-16 | Load splits, class distribution (balanced at ~33.3%), class weights (all 1.0) |
| §2 — Tokenization | 17-22 | DistilBertTokenizerFast, max_length=256, batch map, tensor format |
| §3 — Model Setup | 23-27 | DistilBertForSequenceClassification, dropout=0.3, 67.0M params |
| §4 — Training | 28-36 | WeightedTrainer, 5 epochs, batch=32, lr=2e-5, warmup=10% |
| §5 — Evaluation | 37-43 | Test metrics, confusion matrix, classification report |
| §6 — Inference | 44-49 | Predictions CSV (text, true_label, predicted_label, confidence), model save |
| §7 — Results | 50-55 | Summary table, 5 correct/incorrect examples, loss curve, file checklist |

---

## Technical Decisions

### 1. Model Selection: DistilBERT

| Aspect | Choice | Reason |
|--------|--------|--------|
| Base model | `distilbert-base-uncased` | 40% smaller than BERT, 60% faster, retains ~97% performance |
| Parameters | 66,955,779 | Fits T4 GPU (15.6 GB total) [Cell 26 output] |
| Max length | 256 tokens | Covers ~95%+ of reviews; balances coverage vs. memory |
| Dropout | 0.3 (attention, hidden, classifier) | Increased from default 0.1 to reduce overfitting [Cell 25 output] |

### 2. Dataset: Perfectly Balanced

The N01 preprocessing produced a **fully balanced** dataset — every split has ~33.3% per class [Cell 13]:

| Split | Negative | Neutral | Positive | Total |
|-------|----------|---------|----------|-------|
| Train | 49,920 (33.3%) | 49,921 (33.3%) | 49,920 (33.3%) | 149,761 |
| Validation | 10,698 (33.3%) | 10,697 (33.3%) | 10,697 (33.3%) | 32,092 |
| Test | 10,697 (33.3%) | 10,697 (33.3%) | 10,698 (33.3%) | 32,092 |

Computed class weights are all 1.0000 (no imbalance to correct) [Cell 15 output].
The `WeightedTrainer` is implemented in the code but produces identical behavior
to the standard `Trainer` on this balanced dataset.

### 3. Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 32 | Standard for T4 fine-tuning |
| Epochs | 5 | 23,400 total steps (4,680/epoch) [Cell 31 output] |
| Learning rate | 2e-5 | AdamW default for BERT-family |
| Weight decay | 0.01 | L2 regularization |
| Warmup steps | 2,340 (10%) | Linear LR increase for first 10% of steps |
| Eval strategy | Per epoch | Saves checkpoint each epoch; no early stopping callback |

---

## Results

### Overall Performance [Cell 38/40 output]

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **77.26%** |
| Test Loss | 0.5478 |
| Weighted F1 | 0.7718 |
| Test samples | 32,092 |

### Per-Class Metrics (sklearn) [Cell 40/41 output]

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative (0) | 0.7542 | 0.7807 | 0.7672 | 10,697 |
| Neutral (1) | 0.6852 | 0.6573 | 0.6710 | 10,697 |
| Positive (2) | 0.8749 | 0.8798 | 0.8773 | 10,698 |
| **Weighted Avg** | **0.7714** | **0.7726** | **0.7718** | **32,092** |

**Key insight**: Positive class is easiest (F1=0.8773), Neutral is hardest (F1=0.6710).
This is expected — 3-class sentiment where the middle ground is inherently ambiguous.

### Epoch-Level Training Progress [Cell 36 output]

| Epoch | Eval Loss | Accuracy | F1 |
|-------|-----------|----------|-----|
| 1 | 0.6009 | 0.7504 | 0.7466 |
| 2 | 0.5510 | 0.7674 | 0.7668 |
| 3 | 0.5464 | 0.7710 | 0.7705 |
| 4 | 0.5514 | 0.7732 | 0.7714 |
| 5 | 0.5497 | 0.7751 | 0.7748 |

Training plateaued after epoch 3 — minimal gains in epochs 4-5 (F1: 0.7705→0.7714→0.7748).
Final training loss: 0.5366. Total runtime: 4,786.3 sec (**~80 min**) on T4 [Cell 35 output].

---

## Output Artifacts [Cell 54 verified]

| File | Contents |
|------|----------|
| `data/predictions_distilbert.csv` | 32,092 rows, columns: text, true_label, predicted_label, confidence (11.39 MB) |
| `data/metrics_distilbert.json` | Full classification report + hyperparameters + training history |
| `data/models/distilbert_sentiment/` | model.safetensors (267.84 MB) + config.json + tokenizer |
| `data/plots/nb02_class_distribution.png` | Bar chart confirming balanced splits |
| `data/plots/nb02_confusion_matrix.png` | Row-normalized confusion matrix (3×3) |
| `data/plots/nb02_training_loss_curve.png` | Train loss curve + eval loss points |

---

## Code Quality

### ✅ Strengths

1. **Input validation**: Catches missing N01 artifacts before computation [Cell 7]
2. **Reproducibility**: RANDOM_SEED=42, deterministic cuDNN, warning filters [Cells 3-4]
3. **Checkpoint infrastructure**: Per-epoch saves to Google Drive (resilience against Colab disconnects) [Cell 35]
4. **Balanced dataset produced by N01**: No class rebalancing needed — all weights are 1.0 [Cell 15]
5. **Comprehensive outputs**: Metrics JSON, labeled CSV, loss plots, classification report [Cell 54]

### ⚠️ Observations

1. **No early stopping**: Trains full 5 epochs (gains minimal after epoch 3)
2. **No LR scheduling**: Flat 2e-5 with only warmup; ReduceLROnPlateau could help
3. **Hardcoded Colab paths**: `/content/drive/MyDrive/nlp-project/business-case-01/`
4. **Model file: 267.84 MB** — large for deployment; consider ONNX export or quantization
5. **WeightedTrainer has no effect**: Data is balanced; weights=1.0 produces identical loss

---

## Key Learnings

1. **Balanced data simplifies training**: N01's stratified split eliminates the need for class weights or oversampling
2. **Neutral class is inherently hard**: F1=0.6710 — 3-class sentiment always struggles with ambiguous middle ground
3. **DistilBERT converges fast**: Most learning happens in epochs 1-2 (accuracy: 75.04%→76.74%); epochs 3-5 add only ~1%
4. **Inference pipeline works**: Model reload + smoke test produces sensible predictions (confidence 0.9945-0.9970) [Cell 49]

---

## Recommendations

1. **Add early stopping**: `EarlyStoppingCallback(early_stopping_patience=2)` — saves ~32 min (2 epochs × ~16 min/epoch)
2. **Reduce epochs to 3**: Epoch metrics plateau after epoch 3; training beyond adds minimal gain
3. **Compare with RoBERTa**: If GPU allows — notebook 03 should benchmark against this baseline
4. **Threshold tuning**: Explore per-class decision thresholds (especially for Neutral class)
5. **ONNX export**: Reduce 267.84 MB model to a deployable size

---

## Pipeline Integration

```
N01 (Preprocessing) → [Arrow dataset] → N02 (Sentiment) → [predictions.csv] → N05 (Summarisation)
```

**Dependency**: `data/dataset/` from N01  
**Deliverable**: `data/predictions_distilbert.csv` for N05

---

*Analysis audited against notebook output cells — 2026-05-09*
