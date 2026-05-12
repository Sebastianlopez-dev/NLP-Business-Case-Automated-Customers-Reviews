# Concepts — Advanced Topics

> Cross-notebook audit patterns, clustering alternatives, and embedding model upgrades (N02, N03, N04).  
> See also: [foundation](concepts-foundation.md) · [methodology](concepts-methodology.md) · [enterprise](concepts-enterprise.md)  
> Master index: [concepts.md](concepts.md)

---

| # | Concept | Line |
|---|---------|------|
| 14 | N02/N03 Audit: Cross-Notebook Bug Patterns | L17 |
| 15 | Clustering Alternatives: LDA, K-Means vs MiniBatchKMeans | L81 |
| 16 | Embedding Model: nomic-embed vs MiniLM | L138 |

---

## 14. N02/N03 Audit: Cross-Notebook Bug Patterns *(N02, N03, fixed 2026-05-07)*

**Problem**: N01 underwent a major refactor (streaming from UCSD, 5th column, 18 fixes in Colab), but N02 and N03 were never tested against the new N01 output. A deep audit found 15 bugs — three of them silent (no error, wrong data).

### Bug Categories Discovered

| # | Category | Example | Impact | N02 | N03 |
|---|----------|---------|--------|-----|-----|
| 1 | Cross-environment paths | `DATASET_DIR` assigned inside `else` (local-only) — undefined in Colab | 🔴 NameError. Notebook designed for Colab can't run in Colab | — | ✅ |
| 2 | Variable naming drift | `seed=SEED` but variable is `RANDOM_SEED` — NameError at training time | 🔴 Training crashes before first epoch | ✅ | ✅ |
| 3 | Split key inconsistency | `tokenized_dataset["val"]` but N01 saves as `"validation"` — KeyError | 🔴 Trainer can't instantiate | ✅ | — |
| 4 | Cross-notebook file paths | N03 loads `metrics_distilbert.json` from `DATASET_DIR` but N02 saves to `OUTPUT_DIR` — file never found | 🔴 Silent: comparison section uses hardcoded placeholders instead of real metrics | — | ✅ |
| 5 | Sklearn labels omission | `precision_score(labels, preds, average=None)` WITHOUT explicit `labels=[0,1,2]` — if a class is never predicted, sklearn returns 2-element array → IndexError | 🔴 Crashes evaluation cells on imbalanced predictions | ✅ | — |
| 6 | Hardcoded class counts | `num_labels=3`, `range(3)`, `labels=[0,1,2]`, `LABEL_NAMES = {0: …, 1: …, 2: …}` — all assume exactly 3 classes | 🟡 Works today, breaks if label strategy changes | ✅ | ✅ |
| 7 | Stale comments | Split expected as `"val"` but code uses `"validation"`; split ratio documented as 80/10/10 but project convention is 70/15/15 | 🟡 Misleads future maintainers | ✅ | ✅ |
| 8 | Imports outside cell-imports | `import torch.nn as nn`, `from transformers import pipeline`, `import subprocess` in processing cells | 🟡 Violates project convention; hidden imports confuse readers | ✅ | — |

### Why These Patterns Emerged

Three root causes explain all 15 bugs:

**1. Environment asymmetry**: N02 and N03 were developed locally (where `else` branch runs) but designed for Colab (where `IN_COLAB = True`). Any variable defined only in the local branch silently works during development and crashes in production. The `DATASET_DIR` bug is the canonical example.

**2. Naming drift over time**: The project evolved its conventions (`"val"` → `"validation"`, `SEED` → `RANDOM_SEED`) but notebooks were created at different points in that evolution. N01 uses the latest conventions; N02/N03 used intermediate ones that were never back-ported.

**3. Default-to-hardcode culture**: Early notebooks used hardcoded values (`num_labels=3`, `labels=[0,1,2]`) because "the label scheme is fixed." Later auditing (AGENTS.md §8, traps #10-11) established the principle of deriving everything dynamically — but N02/N03 were written before that principle existed.

### Fix Strategy

All 15 bugs were fixed in the local `.ipynb` files (2026-05-07). The fixes follow a consistent principle:

| Was | Now | Pattern |
|-----|-----|---------|
| `seed=SEED` | `seed=RANDOM_SEED` | Match the variable that actually exists |
| `tokenized_dataset["val"]` | `tokenized_dataset["validation"]` | Match N01's actual DatasetDict keys |
| `DATASET_DIR` in else-only | `DATASET_DIR` outside if/else | All path variables apply to both environments |
| `os.path.join(DATASET_DIR, …)` | `os.path.join(OUTPUT_DIR, …)` | Read from where the producer actually writes |
| `num_labels=3` | `num_labels=len(id2label)` | Derive from the data, not a constant |
| `labels=[0, 1, 2]` | `labels=list(range(NUM_LABELS))` | Same principle |
| `range(3)` | `sorted(id2label.keys())` | Same principle |
| No guard clause | `if 'var' not in dir(): …` | Survive out-of-order cell execution |

### Verification Protocol

The audit established a protocol that future cross-notebook checks must follow:

```
1. Read N01 → note every output (columns, keys, paths, directories)
2. Read consumer → verify every assumption against N01's actual output
3. For each assumption mismatch, classify:
   - CRITICAL = crashes consumer (NameError, KeyError, IndexError)
   - WARNING = works but wrong data (silent fallback, stale comments)
   - SUGGESTION = works correctly but fragile (hardcoded, no guard)
4. Fix CRITICAL and WARNING; defer SUGGESTIONs to itwouldenhance.md
```

> **Analogy**: N01 is the factory that produces parts. N02 and N03 are assembly lines that consume those parts. If the factory changes the shape of a connector, the assembly line jams — but it might jam silently, producing broken products that look fine until someone tests them.

### What Was Deferred

5 SUGGESTION-level items were deferred to [itwouldenhance.md](itwouldenhance.md): duplicate label mappings, padding strategy, warmup estimation, buried imports, and hardcoded DistilBERT parameter count in display strings. None affect execution correctness.

---

## 15. Clustering Alternatives — LDA, K-Means vs MiniBatchKMeans *(N04)*

**Problem**: We chose MiniBatchKMeans for clustering in N04. But is it the best algorithm for discovering product categories from review text? Two alternatives were considered: LDA (topic modeling) and standard K-Means (exact Lloyd's algorithm).

### LDA (Latent Dirichlet Allocation) — why not?

LDA is a probabilistic topic model that discovers latent topics directly from word frequencies — no embeddings needed. Each topic is a distribution over words, making interpretation **native** (no post-hoc TF-IDF step).

| | MiniLM + MiniBatchKMeans (current) | LDA |
|---|---|---|
| **How it represents text** | 384-dimensional dense vectors capturing semantic meaning | Bag-of-words — word counts only |
| **Clustering method** | Euclidean distance in embedding space | Probabilistic topic assignment (each review is a mixture of topics) |
| **Interpretability** | Requires post-hoc TF-IDF to find key terms | Topics ARE word lists — inherently interpretable |
| **Handles negation?** | ✅ "Not good" ≠ "good" in embedding space | ❌ "Not good" and "good" share words — LDA sees them as similar |
| **Handles synonyms?** | ✅ "Amazing" ≈ "excellent" via semantic similarity | ❌ "Amazing" and "excellent" are unrelated words |
| **Multiple cluster membership** | ❌ Each review assigned to exactly 1 cluster | ✅ Each review is a mixture of topics (soft assignment) |
| **Scalability** | ✅ MiniBatchKMeans handles millions | ✅ Online LDA exists but less mature in sklearn |
| **Works best with** | Short, diverse text where context matters | Long documents with clear topical vocabulary |

**Why we did NOT use LDA**:
1. **Amazon reviews are short** (median ~52 tokens). LDA struggles with short texts because word co-occurrence statistics are sparse — there simply aren't enough words per document to reliably estimate topic distributions.
2. **Context matters for reviews** — "This battery is terrible" and "The battery life is amazing" share the word "battery" but have opposite sentiment. LDA would conflate them into the same topic; embeddings separate them.
3. **MiniLM embeddings capture more signal** — a 384-dim vector encodes semantic relationships (synonyms, analogies, negation) that bag-of-words discards. For product review clustering, this semantic depth matters more than LDA's native interpretability.

> **Analogy**: LDA is like sorting books by which words they contain — "battery" appears in both a 5-star and a 1-star review, so they end up on the same shelf. MiniLM + K-Means is like sorting books by what they *mean* — the glowing review and the angry complaint go to different shelves even though they use similar vocabulary.

### K-Means vs MiniBatchKMeans — empirical comparison

The project brief requires 4–6 meta-categories, so k must be controllable. Both K-Means and MiniBatchKMeans satisfy this constraint, but they differ in convergence quality:

| Algorithm | Memory | Convergence | Quality | Best for |
|-----------|--------|-------------|---------|----------|
| **K-Means (Lloyd's)** | O(n·k·d) — all data in RAM | Iterates until no points change cluster | Exact Lloyd's optimum | Datasets that fit in memory |
| **MiniBatchKMeans** | O(batch·k·d) — one batch at a time | Approximate — random batches | ~98–99% of K-Means quality | Datasets too large for RAM |

**Our approach**: Instead of choosing one blindly, Notebook 04 runs **both algorithms** on the same MiniLM embeddings and compares them empirically:

```
K_VALUES = range(2, 11)  →  for each k:
    ├── K-Means (Lloyd's)    → inertia_km, silhouette_km, time_km
    └── MiniBatchKMeans      → inertia_mb, silhouette_mb, time_mb
```

The results are displayed as:
- **Overlaid elbow and silhouette plots** — K-Means (solid cyan) vs MiniBatchKMeans (dashed orange) on the same axes
- **Comparison table** — per-k metrics for both algorithms side by side
- **Quality retention score** — what percentage of K-Means silhouette does MiniBatchKMeans retain?

**Why this matters for a bootcamp**:
1. It demonstrates **empirical thinking** — we don't just claim MiniBatchKMeans is "almost as good," we measure it.
2. It shows understanding of the **scale-quality tradeoff** — MiniBatchKMeans exists because K-Means doesn't scale; both have their place.
3. The final model is MiniBatchKMeans (scalable to 571M reviews), but K-Means serves as the **gold-standard baseline** that validates our choice.

> **Analogy**: K-Means is the architect who measures every wall before placing furniture. MiniBatchKMeans is the mover who eyeballs it — slightly less precise, but finishes before the truck leaves. We hired the mover (MiniBatchKMeans) because the house has 571 million rooms, but we measured one room with the architect (K-Means) just to confirm the mover's eye is good enough.

---

## 16. Embedding Model: nomic-embed vs MiniLM *(N04, upgraded 2026-05-08)*

**Problem**: N04 originally used `all-MiniLM-L6-v2` (2021) for clustering. The MTEB benchmark shows several 2024 models outperform it significantly for clustering tasks. Is the upgrade worth it?

| Model | Year | Params | Dims | MTEB Clustering | Speed | Best for |
|-------|------|--------|------|-----------------|-------|----------|
| all-MiniLM-L6-v2 | 2021 | 22M | 384 | 39.2 (v1) | 🚀 Fastest | Original choice — compact, fast |
| nomic-embed-text-v1.5 | 2024 | 137M | 768 | 49.7 (v2) | Medium | ⭐ Best clustering quality |
| ModernBERT-base | 2024 | 149M | 768 | 51.2 (v2) | Slow | Max quality, heaviest |

**Why nomic-embed over MiniLM**:
1. **MTEB leap**: nomic-embed scores ~10 points higher on clustering tasks. For our use case (product category discovery from reviews), this means clusters are more semantically coherent — "wireless earbuds" and "Bluetooth headphones" end up together, not split across clusters.
2. **Matryoshka support**: nomic-embed supports dimensionality truncation without retraining. We use full 768-dim for quality, but could truncate to 384 for speed in production.
3. **Still sentence-transformers compatible**: same `SentenceTransformer()` API, same `.encode()` method — one-line change.

**Why NOT ModernBERT**:
ModernBERT edges out nomic-embed on MTEB but is ~10× slower to encode and requires a GPU. For 500K reviews on Colab Pro (T4, 16 GB VRAM), nomic-embed fits comfortably while ModernBERT would take hours. The 0.5-point MTEB gap isn't worth the encoding time.

**What changed in N04 (2026-05-08)**:
- `EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"` (+ `trust_remote_code=True`)
- Embedding dimensions: 384 → 768
- Encoding time: ~1.5× longer than MiniLM (~45 min vs 30 min for 500K reviews)
- W&B config updated to track the new embedder name and dims
- Downstream (K-Means, silhouette, UMAP, N05) unchanged — same interfaces

**Tradeoff**: Slightly slower encoding (137M params vs 22M) in exchange for measurably better cluster coherence. The W&B baseline from the MiniLM run lets us quantify the improvement with silhouette score.

> **Analogy**: MiniLM is ordering a pizza by listing ingredients ("cheese, tomato, pepperoni"). nomic-embed is describing the *experience* of the pizza ("New York style, wood-fired, spicy pepperoni that curls at the edges"). The second description captures what actually matters — and clustering algorithms work better with richer descriptions.
