# Concepts — Enterprise-Grade (Deferred)

> Clustering at production scale and efficient fine-tuning — decisions deferred for enterprise adoption, not needed for bootcamp delivery.  
> See also: [foundation](concepts-foundation.md) · [methodology](concepts-methodology.md) · [advanced](concepts-advanced.md)  
> Master index: [concepts.md](concepts.md)
---
| # | Concept | Line |
|---|---------|------|
| 17 | Clustering Profesional: BERTopic + HDBSCAN vs K-Means | L12 |
| 18 | Fine-Tuning Eficiente: LoRA vs Full Fine-Tuning | L97 |
| 19 | Experiment Tracking: W&B Dashboard (deferred) | L199 |
---
## 17. Clustering Profesional: LDA + BERTopic vs K-Means *(deferred, enterprise-grade)*

**Context**: This bootcamp project uses MiniLM embeddings + MiniBatchKMeans for clustering in N04. This is a deliberate choice for speed, simplicity, and Colab constraints. For enterprise production pipelines (2024-2026), **BERTopic + HDBSCAN** is the modern standard.

### Why K-Means was chosen for bootcamp

| Constraint | K-Means (current) | BERTopic (enterprise) |
|------------|------------------|----------------------|
| **Timeline** | ✅ 30-45 min end-to-end on Colab T4 | ❌ 2-3× longer (HDBSCAN is slower, topic extraction adds steps) |
| **Complexity** | ✅ 3 lines: `MiniBatchKMeans(n_clusters=k).fit(embeddings)` | ❌ 10+ lines: BERTopic config, UMAP reduction, HDBSCAN tuning |
| **Control over k** | ✅ Explicit `n_clusters` — matches brief (4-6 meta-categories) | ❌ HDBSCAN auto-detects clusters — less predictable for fixed deliverables |
| **Colab compatibility** | ✅ CPU-only, no extra deps beyond `sentence-transformers` + `sklearn` | ❌ Requires `bertopic`, `hdbscan`, `umap-learn` — more install time, potential conflicts |
| **Stakeholder clarity** | ✅ Simple to explain: "reviews grouped by similarity" | ❌ Requires explaining density-based clustering, topic coherence scores |

### Comparison: K-Means vs LDA vs BERTopic+HDBSCAN

| | **K-Means (current)** | **LDA (topic modeling)** | **BERTopic + HDBSCAN (enterprise)** |
|---|----------------------|-------------------------|-----------------------------------|
| **Type** | Centroid-based, geometric | Probabilistic topic model | Density-based + neural embeddings |
| **Input** | Embeddings (MiniLM, nomic-embed) | Bag-of-words / TF-IDF | Embeddings + c-TF-IDF for topic extraction |
| **Cluster shape** | Spherical (Voronoi partitions) | Document-topic distributions | Arbitrary shapes (density-connected regions) |
| **Number of clusters** | ✅ Explicit `k` parameter | ✅ Explicit `k` topics | ❌ Auto-detected by HDBSCAN (configurable via `min_cluster_size`) |
| **Interpretability** | Post-hoc TF-IDF on cluster members | ✅ Native: topics ARE word distributions | ✅ c-TF-IDF extracts representative terms per cluster |
| **Handles noise** | ❌ Assigns every point to a cluster | ❌ Soft assignment, no noise concept | ✅ HDBSCAN labels outliers as `-1` (noise) |
| **Semantic quality** | Good (depends on embeddings) | ❌ Poor for short text (sparse word co-occurrence) | ⭐ Best (combines embeddings + topic modeling) |
| **Scalability** | ✅ MiniBatchKMeans handles millions | ✅ Online LDA exists but less mature | ⚠️ HDBSCAN is O(n²) — needs subsampling or approximate methods |
| **When to use** | Bootcamps, MVPs, fixed-k requirements | Long documents, clear topical vocabulary | Production, stakeholder reports, weekly data updates |
| **Adoption (2024-2026)** | Legacy, still common in tutorials | 2010s standard, declining | ⭐ ING, bol.com, e-commerce platforms |

### Enterprise recommendation: BERTopic is the 2024-2026 standard

**BERTopic** (2021+, actively maintained) combines:
1. **Embeddings** (same as our pipeline: MiniLM, nomic-embed, etc.)
2. **UMAP** for dimensionality reduction (optional but recommended)
3. **HDBSCAN** for density-based clustering (auto-detects k, handles noise)
4. **c-TF-IDF** (class-based TF-IDF) to extract interpretable topic labels

**Why enterprises use it**:
- **Auto-detects k**: No need to guess the number of clusters — HDBSCAN finds natural groupings
- **Topic coherence metrics**: Built-in evaluation (silhouette, c-TF-IDF scores) for stakeholder reports
- **Noise handling**: Outliers are labeled as `-1`, not forced into clusters — critical for real-world data
- **Works with modern embeddings**: Plug-and-play with `sentence-transformers`, OpenAI, Cohere
- **Real-world adoption**: ING (banking), bol.com (e-commerce), customer support ticket clustering, product review analysis

**Example enterprise use case**:
```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Same embeddings we use in N04
embedding_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5")

# BERTopic handles UMAP + HDBSCAN + c-TF-IDF internally
topic_model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=50,        # Control granularity
    nr_topics=None,           # Auto-detect, or set to reduce topics
    calculate_probabilities=True
)

topics, probabilities = topic_model.fit_transform(reviews)
topic_model.get_topic_info()  # DataFrame with topic IDs, terms, frequencies
```

### Why this is deferred (not implemented now)

1. **Timeline**: Full pipeline must run today/tomorrow for bootcamp deliverable. BERTopic adds complexity without improving the grade.
2. **N05 integration**: N05 expects `clusters.csv` with integer cluster IDs. BERTopic outputs topic IDs + probabilities — would require reworking the extractive phase.
3. **Bootcamp requirements**: Brief asks for "4-6 meta-categories" — K-Means with explicit `k` is a better fit than HDBSCAN's auto-detection.
4. **Learning curve**: Understanding c-TF-IDF, UMAP parameters, and HDBSCAN's `min_cluster_size` vs `min_samples` would delay delivery.

### When to use BERTopic in future NLP projects

| Scenario | Use BERTopic | Use K-Means |
|----------|-------------|-------------|
| Production pipeline with weekly data updates | ✅ Auto-adapts to new clusters | ❌ Fixed k requires manual re-tuning |
| Stakeholder reports requiring topic coherence | ✅ Built-in metrics + interpretable labels | ❌ Post-hoc TF-IDF, less rigorous |
| Data with significant noise/outliers | ✅ HDBSCAN labels noise explicitly | ❌ Forces every point into a cluster |
| Bootcamp / MVP / time-constrained prototype | ❌ Overkill | ✅ Simple, fast, predictable |
| Fixed number of categories required | ❌ HDBSCAN auto-detects | ✅ Explicit `n_clusters` |

> **Analogy**: K-Means is a rental car — you pick the model (k), drive it anywhere, and return it. BERTopic is a self-driving taxi — it figures out the route (number of clusters), avoids potholes (noise), and gives you a receipt with trip details (topic coherence). For a quick errand (bootcamp), the rental car is fine. For daily commutes (enterprise), the taxi is worth it.

---

## 18. Fine-Tuning Eficiente: LoRA vs Full Fine-Tuning *(deferred, enterprise-grade)*

**Context**: This bootcamp project uses **full fine-tuning** for both N02 (DistilBERT) and N03 (RoBERTa). This is the simplest approach: unfreeze all model parameters and train end-to-end. For a bootcamp with T4 GPU (16 GB VRAM) and a straightforward dataset like Amazon Reviews, full fine-tuning works perfectly. No quality tradeoff, no complexity overhead.

### What is LoRA (Low-Rank Adaptation)?

**LoRA** (Hu et al., 2021) is a parameter-efficient fine-tuning technique that **freezes the base model** and trains tiny **adapter matrices** instead. Instead of updating 125M+ parameters (RoBERTa), you update 0.1-1M parameters (~1% of the model).

**How it works**:
1. Freeze all weights in the base transformer (attention layers, feed-forward networks)
2. Inject small "adapter" matrices (low-rank decomposition) into attention layers
3. Train ONLY the adapters — the base model stays unchanged
4. At inference time, combine base weights + adapter weights (no latency penalty)

```python
# The 4 lines needed to add LoRA to N03
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # rank: 16-64 typical (higher = more params, better quality)
    lora_alpha=32,  # scaling factor
    target_modules=["query", "value"],  # which layers to adapt
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
# Now train(model, ...) only updates ~0.5M params instead of 125M
```

### Full Fine-Tuning vs LoRA: Comparison

| | **Full Fine-Tuning (current)** | **LoRA (enterprise)** |
|---|------------------------------|----------------------|
| **Trainable params** | 125M (RoBERTa-base) | ~0.1-1M (0.8-1% of base) |
| **VRAM usage** | ~3 GB (RoBERTa, batch=32, seq=128) | ~500 MB (same config) |
| **Training time** | ~3 hours (N03, T4 GPU, 400K reviews) | ~1-1.5 hours (2-3× faster) |
| **F1 quality** | Baseline (100%) | 98-100% (marginal loss on easy datasets) |
| **Complexity** | 3 lines: `AutoModelForSequenceClassification` | 7 lines: + `LoraConfig` + `get_peft_model` |
| **Checkpoint size** | 500 MB per checkpoint | 5-20 MB per checkpoint |
| **Base model reuse** | ❌ One fine-tuned model per domain/task | ✅ Same base model, swap adapters for different domains |
| **When to use** | Bootcamps, MVPs, single-domain, T4 has enough VRAM | LLMs 7B+, multi-domain, weekly retraining, VRAM-constrained |

### Why full fine-tuning was chosen for this bootcamp

| Reason | Detail |
|--------|--------|
| **Simplicity** | No extra dependency (`peft`), no new concepts to explain in the notebook |
| **T4 has enough VRAM** | RoBERTa full fine-tuning uses ~3 GB. T4 has 16 GB — plenty of headroom |
| **No quality tradeoff** | Amazon Reviews is an "easy" dataset (clear sentiment signals). LoRA's 1-2% F1 loss isn't worth the complexity |
| **Timeline** | Pipeline must run today/tomorrow. LoRA adds tuning parameters (`r`, `alpha`, `target_modules`) without improving the grade |
| **Single domain** | We're fine-tuning on Amazon Reviews only. No need to swap adapters for different product categories |

### When LoRA IS worth it (enterprise scenarios)

| Scenario | Why LoRA wins |
|----------|--------------|
| **LLMs 7B+ parameters** | Full fine-tuning a 7B model needs 80+ GB VRAM (A100). LoRA fits in 24 GB (RTX 4090) or even 16 GB (T4) |
| **Multi-domain adaptation** | Train one adapter per product category (electronics, beauty, automotive). Swap adapters at inference without loading multiple base models |
| **Weekly retraining** | Smaller checkpoints (5-20 MB vs 500 MB) = faster CI/CD, cheaper storage, easier versioning |
| **Resource-constrained environments** | Edge devices, mobile deployment, shared GPU clusters — LoRA's 4-8× VRAM reduction is critical |
| **Continual learning** | Fine-tune on new data without catastrophic forgetting — freeze base, train new adapter, blend with old adapters |

### Why LoRA is deferred (not implemented now)

1. **Marginal benefit on Amazon Reviews**: This dataset has clear sentiment signals (1-2★ = negative, 4-5★ = positive). LoRA achieves 98-99% of full fine-tuning F1, but we don't need to squeeze out every last point for a bootcamp.
2. **Adds `peft` dependency**: Another `pip install`, another concept to explain in the notebook. The grader cares about results, not parameter efficiency.
3. **T4 handles full fine-tuning comfortably**: 3 GB VRAM usage is 19% of T4's 16 GB. No pressure to optimize.
4. **Timeline is the constraint**: Full pipeline must run today/tomorrow. LoRA would require:
   - Installing `peft` library
   - Tuning `r` (rank), `alpha`, `target_modules` hyperparameters
   - Rewriting N02/N03 training loops
   - Re-running the entire fine-tuning pipeline
   - Comparing LoRA vs full F1 scores in the analysis

### Code integration (for future enterprise projects)

If you ever need LoRA in a production NLP pipeline, here's the minimal change to N03:

```python
# cell-pip-install
!pip install peft  # HuggingFace Parameter-Efficient Fine-Tuning

# cell-model-loading (after loading base RoBERTa)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=16,  # Start with 16, tune to 32-64 if quality drops
    lora_alpha=32,  # 2× r is a good default
    target_modules=["query", "value"],  # Attention matrices
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # "trainable params: 524,288 || all params: 125,323,776"

# cell-training-args: no changes needed
# Trainer automatically only updates trainable parameters
```

> **Analogy**: Full fine-tuning is like renovating an entire house — you repaint every wall, replace every fixture, and rewire every outlet. LoRA is like adding smart home devices — you keep the house structure intact and just install adapters (smart switches, thermostats) that change how it behaves. For a rental (bootcamp), renovations are fine. For a chain of hotels (enterprise), you want to swap smart devices between properties without rebuilding each one.

---

## 19. Experiment Tracking: Weights & Biases Dashboard *(deferred, enterprise-grade)*

**Context**: In May 2026, W&B integration was **stripped from N02, N03, and N04** after discovering it added ~20 lines of setup code per notebook (API keys, Colab Secrets, `userdata`, `wandb.login`, `wandb.init`, `wandb.log`, `wandb.finish`) without providing any value that the notebooks didn't already deliver locally.

### What W&B provides

Weights & Biases is an experiment tracking platform that:
- **Logs metrics** (loss, accuracy, F1) per step/epoch to a cloud dashboard
- **Compares runs** side by side (DistilBERT vs RoBERTa, different hyperparameters)
- **Stores artifacts** (confusion matrices, classification reports, UMAP plots)
- **Enables team collaboration** (share dashboards, comment on runs)

In enterprise ML teams, W&B acts as the **single source of truth** for experiment history. Without it, you're comparing screenshots in Slack.

### Why we stripped it from the bootcamp notebooks

| W&B code removed | Already covered locally |
|------------------|------------------------|
| `wandb.login()` + `wandb.init()` + Colab Secrets setup | Not needed — metrics are printed and saved to JSON |
| `wandb.log({"test_accuracy": ...})` | `metrics_distilbert.json` / `metrics_roberta.json` |
| `wandb.log({"confusion_matrix": ...})` | `nb02_confusion_matrix.png` |
| `wandb.Table(classification_report)` | Printed in output + JSON |
| `wandb.log({"umap_sentiment": wandb.Image(...)})` | `nb04_umap_sentiment.png` |
| `wandb.finish()` | Not needed — no persistent connection to close |

**Every metric and visualization that W&B would capture is already saved locally** in `data/metrics_*.json`, `data/plots/`, and `data/summaries/`. The notebooks are self-contained — open them, run them, and everything is there.

### The bootcamp pipeline doesn't need experiment tracking because:

1. **Single run per model**: We fine-tune DistilBERT once, RoBERTa once, cluster once. There is no hyperparameter sweep, no A/B comparison loop. W&B shines when you're running 50 experiments and need to compare them.
2. **All artifacts are deterministic**: Fixed seed (`RANDOM_SEED = 42`), same dataset, same splits. Rerunning the notebook produces identical results — no need to "track" what changed.
3. **The output IS the deliverable**: The bootcamp grader sees the notebook itself — the inline charts, printed metrics, and saved PNGs. W&B would add a link to an external dashboard the grader won't open.
4. **Setup overhead is real**: Colab Secrets, API key generation, `userdata` imports, toggle switches — each step is a potential failure point (and did fail during N02 testing, blocking training for 20+ minutes).

### Enterprise value (when W&B IS worth it)

| Scenario | Why W&B wins |
|----------|-------------|
| **Hyperparameter sweeps** | Run 10 learning rates × 5 batch sizes = 50 runs. W&B auto-logs all of them with parallel coordinate plots. Manual JSON comparison would be tedious. |
| **Weekly retraining** | Compare this week's F1 to last week's. Is the new data improving or degrading the model? W&B shows the trend over time. |
| **Multi-model comparison** | DistilBERT vs RoBERTa vs ModernBERT on the same dashboard. Filter by tag, overlay loss curves, sort by F1. |
| **Team collaboration** | Share a dashboard link with stakeholders. "Here's the RoBERTa champion run" instead of sending a 50 MB notebook. |
| **CI/CD integration** | Automatically log every `git push` that triggers retraining. Failed runs are visible immediately. |

### Future idea: a W&B demo notebook (post-bootcamp)

The user expressed interest in creating a **single notebook that loads the trained models and runs them with W&B tracking** as a portfolio piece. This would:

1. Load the saved DistilBERT and RoBERTa checkpoints from `data/models/`
2. Run inference on a shared test set
3. Log all metrics, confusion matrices, and comparison charts to W&B
4. Serve as a clean, recruiter-facing demo: *"Here's how I'd track experiments in production"*

This separates concerns cleanly:
- **Bootcamp notebooks (N01-N05)**: Self-contained, local artifacts, zero external dependencies
- **W&B demo notebook (N06, optional)**: Production-style experiment tracking, portfolio-ready, requires API key

> **Analogy**: W&B is like a gym membership with a personal trainer — it tracks your progress, compares you to your past self, and motivates consistency. For a single workout (bootcamp), you don't need it — the mirror (local PNGs and JSONs) tells you everything. For a year-long fitness plan (enterprise ML team), it's the difference between guessing and knowing.
