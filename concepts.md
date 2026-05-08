# Concepts — NLP Business Case

> Research behind the decisions. Each concept follows: **Problem → Options → Tradeoff → Decision**.  
> Explained simply, like to a 5-year-old, but technically precise.

---

> | # | Concept | Line |
> |---|---------|------|
> | 1 | Data Loading: Streaming vs Full Download | L29 |
> | 2 | Storage Format: Arrow vs CSV | L68 |
> | 3 | Data Accumulation: Python Dicts vs Stream→Arrow | L84 |
> | 4 | Dataset Balancing: Cap vs No Cap | L99 |
> | 5 | Sentiment Models: DistilBERT vs RoBERTa | L116 |
> | 6 | Embedding Model: MiniLM vs Alternatives | L145 |
> | 7 | Clustering Algorithm: MiniBatchKMeans vs KMeans | L161 |
> | 8 | Summarization Model: Mistral Medium 3.5 | L178 |
> | 9 | Summarization Method: Extractive-Abstractive | L198 |
> | 10 | EDA Voice: Research Tone vs Assertive Tone | L213 |
> | 11 | Stopwords: NLTK + Domain-Specific Hybrid | L232 |
> | — | Quick Reference: Models at a Glance | — |
> | 12 | Pipeline Architecture: Why CSVs? | L267 |
> | 13 | `load_dataset` API: `split` + `streaming` Deprecation | L306 |
> | 14 | N02/N03 Audit: Cross-Notebook Bug Patterns | L346 |
> | 15 | Clustering — LDA, K-Means vs MiniBatchKMeans | L410 |

---

## 1. Data Loading: Streaming vs Full Download *(N01)*

**Problem**: The Amazon Reviews 2023 dataset is 750 GB compressed. Downloading it all would fill any hard drive. Loading it all into RAM is impossible.

| Approach | How it works | RAM used | Disk cache | When to use |
|----------|-------------|----------|------------|-------------|
| Full download | Download all JSONL.gz files → unzip → load into memory | 750 GB+ (impossible) | 750 GB+ | Never for this dataset |
| HF `load_dataset()` | Downloads .jsonl.gz → indexes ALL rows into Arrow → samples 30K | ~90 MB per category | **~2× per category** (raw .gz + Arrow index = ~120 GB total) | Small datasets only — fills Colab disk before 33 categories |
| Raw JSONL streaming ✅ | `requests.get(stream=True)` + `gzip.GzipFile` from UCSD. Reads line by line, stops at 30K | ~90 MB per category (the "straw") | **0 GB** — never writes to disk | ✅ **Always for datasets > RAM** (2026-05-05 onwards) |

### The HuggingFace cache problem (discovered 2026-05-05)

The HuggingFace approach — even without explicit `streaming=True` — downloads the `.jsonl.gz` file AND converts it to an Arrow index in `~/.cache/huggingface/`. This **doubles** the disk footprint per category:

```
HuggingFace per category:   .jsonl.gz (downloaded) + Arrow index + metadata
                             ~3-4 GB per category × 33 = ~100-120 GB
Raw JSONL per category:      0 bytes on disk — only 30K dicts in RAM
```

At 8 categories, Colab was already at 80 GB / 225 GB — the pipeline would have crashed before completing all 33.

### Why raw JSONL streaming from UCSD

| | HF `load_dataset()` | Raw JSONL from UCSD |
|---|---|---|
| URL source | HuggingFace Hub (redirects to UCSD anyway) | [mcauleylab.ucsd.edu](https://amazon-reviews-2023.github.io) — the original host |
| Disk cache | ~120 GB across 33 categories | **0 GB** |
| "Generating full split" | Indexes ALL rows (19.9M for Automotive) | ❌ Never happens |
| Time per large category | 3-5 min (indexing) + 30s (sampling) | 10-30s total |
| API stability | Broke in `datasets>=2.19` (`split` + `streaming` deprecated) | Pure Python stdlib + `requests` — no version risk |
| Code complexity | 1 line of `load_dataset()` | ~15 lines of HTTP + gzip + JSON |

**Why we chose raw JSONL streaming**: It's what §1 promised from day one — "streaming from a 750 GB ocean." With the HuggingFace approach, we were downloading and indexing the ocean just to take 30K sips. Raw streaming actually delivers: zero disk, zero indexing, true lazy reading. The URL pattern is verified from the official UCSD site — no risk of wrong links.

> **Analogy**: HuggingFace was like a librarian who unpacks, catalogs, and shelves all 571M books before letting you check out 30K. Raw streaming is like walking into the warehouse, grabbing the first 30K books off the first shelf, and leaving.

---

## 2. Storage Format: Arrow vs CSV *(N01)*

**Problem**: After preprocessing, we need to save data so N02 and N03 can load it fast. CSV is the default, but it's slow and ambiguous.

| | CSV | Arrow |
|---|---|---|
| How it stores | Row by row, plain text. A 1 GB CSV takes 1 GB to load | Column by column, binary. Memory-mapped — you only load what you use |
| Loading speed | `pd.read_csv('file.csv')` parses every comma and quote. Slow | `load_from_disk('dir/')` is instant — no parsing |
| Data types | Everything becomes string. `True` → `"True"` → must convert back | Preserves `int`, `float`, `str` natively |
| Disk size | Bigger (text is verbose) | Smaller (binary, compressed) |

**Why we chose Arrow**: HuggingFace `datasets` uses Arrow internally. `save_to_disk()` / `load_from_disk()` is the native format. Models tokenise directly from Arrow without pandas overhead. It's like saving a photo as PNG vs describing every pixel in a text file.

---

## 3. Data Accumulation: Python Dicts vs Stream→Arrow *(N01)*

**Problem**: When streaming 33 categories × 30K reviews, we need to hold the data somewhere before preprocessing. Two ways to do it.

| | Python dicts (current) | Stream → Arrow per category |
|---|---|---|
| How | Stream → collect in a list → convert to DataFrame at the end | Stream → preprocess → save Arrow → free RAM → repeat for each category → concatenate at the end |
| RAM | All records live simultaneously as Python objects. 1 dict ≈ 2-3 KB. 990K ≈ **1.4 GB** | Only 1 category in RAM at a time. 30K dicts ≈ **90 MB**. Arrow files use memory-mapping |
| Complexity | Simple. One list, one DataFrame | Medium. Must manage per-category files, concatenation logic |
| Ceiling | Breaks above ~100K/category (~3 GB of dicts) | Scales indefinitely — RAM is constant regardless of total |

**Why we chose Python dicts**: With Colab Pro (25 GB RAM), 1.4 GB is 6% of available memory. The simplicity isn't worth sacrificing for 90 MB savings. If we ever go to 100K+ per category, we switch to Stream→Arrow.

---

## 4. Dataset Balancing: Cap vs No Cap *(N01)*

**Problem**: Amazon reviews are 70% positive. If we train a model on raw data, it learns "always say positive" and gets 70% accuracy — useless. We need equal amounts of Positive, Neutral, and Negative. But how many?

| | With cap (MAX_PER_CLASS) | Without cap (natural) |
|---|---|---|
| Who decides the size | A fixed number you choose (e.g. 10,000) | The minority class — whatever naturally exists |
| How it works | `min(minority_count, 10_000)` → if minority has 150K, you still cap at 10K | `minority_count` → if minority has 150K, you use all 150K per class |
| Result | 3 × 10K = 30K. Fast training, less data | 3 × 150K = 450K. Better model, slower training |
| What you lose | Data you already streamed, cleaned, and labeled — thrown away | Nothing |

> **Analogy**: You cooked dinner for 20 people. With a cap, you serve 4 plates and throw away 16. Without a cap, you serve all 20.

**Why we chose no cap**: By the time we reach balancing, we already paid the cost of streaming from HuggingFace (10-15 min), cleaning text, and labeling sentiment. Discarding data at that point wastes work. Training 400K examples takes ~1.5 h on T4 GPU — totally reasonable.

---

## 5. Sentiment Models: DistilBERT vs RoBERTa *(N02, N03)*

**Problem**: We need to classify reviews as Positive / Neutral / Negative. BERT is the standard, but which BERT is best?

| Model | Params | Layers | Speed | Quality | Best for |
|-------|--------|--------|-------|---------|----------|
| DistilBERT | 66M | 6 | 🚀 60% faster | ~97% of BERT | Baseline, fast iteration, limited GPU |
| BERT-base | 110M | 12 | 1× (baseline) | Baseline | The original — but superseded |
| RoBERTa | 125M | 12 | Similar to BERT | ⭐ Better than BERT in most benchmarks | Best quality for nuanced text |

**Papers**:
- **BERT** — Devlin et al., 2018. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805) — the foundational transformer encoder.
- **DistilBERT** — Sanh et al., 2019. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) — knowledge distillation: 40% smaller, 60% faster, 97% of BERT's performance.
- **RoBERTa** — Liu et al., 2019. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692) — BERT retrained with more data, dynamic masking, no NSP task. Official model card: [huggingface.co/FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base).

**Why two models?** DistilBERT is the *baseline* — fast to train, proves the pipeline works. RoBERTa is the *champion* — trained on 160 GB+ of text, better at sarcasm, nuanced opinions, and long reviews. Having both lets us compare: "Did the extra 59M parameters actually help on our data?"

> **Token limit — ultra-verified (2026-05-02):** Both models use exactly **512 tokens** as maximum input length. The [official RoBERTa model card](https://huggingface.co/FacebookAI/roberta-base) states: *"The inputs of the model take pieces of 512 contiguous tokens"* and confirms training at *"a sequence length of 512."* The [official DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased) states: *"The only constrain is that the result has a combined length of less than 512 tokens."* DistilBERT, as a distilled BERT, inherits the same positional embedding architecture. The word count analysis in N01 estimates ~394 English words ≈ 512 tokens, applicable to both models.
>
> **Training truncation decision — `max_length=128` (2026-05-05):** Both N02 and N03 tokenize at **128 tokens**, not the full 512. This is deliberate:
> - Amazon reviews are typically short (N01 EDA: median ~40 words ≈ 52 tokens). 128 tokens cover ~95% of reviews without truncation.
> - Self-attention complexity is O(n²): reducing sequence length from 512 to 128 divides training time by ~4× (512²/128² = 16× fewer attention pairs per review).
> - 128 is a well-documented sweet spot in the fine-tuning literature for short-text classification (GLUE SST-2, IMDb, Amazon reviews).
> - **Tradeoff**: ~5% of long reviews lose their tail content. The time savings on T4 GPU outweigh this loss for a bootcamp project. For production, consider 256 or dynamic batching.
>
> **Colab disconnect protection — `resume_from_checkpoint` (2026-05-08):**
> Colab runtimes disconnect after ~90 minutes of inactivity, or randomly under heavy load. During fine-tuning (10–60 min for DistilBERT/RoBERTa), a mid-training disconnect means losing all progress since the last checkpoint.
>
> Both N02 and N03 now detect the latest checkpoint before calling `trainer.train()` by scanning `CHECKPOINT_DIR` / `CHECKPOINTS_DIR` for directories starting with `"checkpoint"`. Because `save_strategy="epoch"` writes to Google Drive every epoch, checkpoints survive runtime resets. If found, `trainer.train(resume_from_checkpoint=last_checkpoint)` resumes the optimizer state, learning rate schedule, and epoch counter from exactly where it stopped.
>
> | Scenario | Without resume | With resume (current) |
> |----------|---------------|----------------------|
> | Disconnect at epoch 1/3 | Restart from epoch 1 → 60 min total | Resume from epoch 2 → ~40 min remaining |
> | Disconnect at epoch 3/3 (min 55) | Restart from epoch 1 → 115 min total | Resume from epoch 3 → ~5 min remaining |
> | No disconnect | Same (fresh start) | Same (no checkpoint found → starts fresh) |
>
> This is a zero-cost protection: the checkpoint scanning overhead is negligible (<1 second), and `trainer.train(resume_from_checkpoint=None)` behaves identically to `trainer.train()` on first run.

> **DistilBERT** = knowledge distilled from BERT (teacher → student). Like a summary of a textbook — lighter but retains the key ideas.  
> **RoBERTa** = BERT retrained with more data, bigger batches, and no "next sentence prediction" task that BERT had. Like the textbook's second edition — same structure, better content.

---

## 6. Embedding Model: MiniLM vs Alternatives *(N04)*

**Problem**: To cluster reviews, we first convert text to numbers (embeddings). The model choice is a tradeoff: bigger models produce better embeddings, but slower and heavier.

| Model | Size | Dims | Speed | Quality | Best for |
|-------|------|------|-------|---------|----------|
| all-MiniLM-L6-v2 | 22 MB | 384 | 🚀 Fastest | Good | Large datasets, fast iteration |
| all-MiniLM-L12-v2 | 120 MB | 384 | Medium | Better | Middle ground |
| all-mpnet-base-v2 | 420 MB | 768 | Slow | ⭐ Best | Small datasets, max quality |

**Why MiniLM-L6**: We need to embed potentially millions of reviews. mpnet-base would take hours and might OOM. MiniLM-L6 is 22 MB — smaller than a JPEG — yet produces embeddings that cluster well. For product categories (broad topics, not fine sentiment), 384 dimensions is enough.

> **Analogy**: MiniLM is a sketch artist — captures the essence in seconds. mpnet is a portrait painter — more detail, but takes longer.

---

## 7. Clustering Algorithm: MiniBatchKMeans vs KMeans *(N04)*

**Problem**: KMeans is the standard clustering algorithm, but it loads ALL data into memory. With 571M reviews, that's impossible.

| Algorithm | Memory | How it works | Ceiling |
|-----------|--------|-------------|---------|
| KMeans | O(n × d) — all data, all dimensions | Iterates over every point every iteration | Breaks above ~100K points on Colab |
| MiniBatchKMeans | O(batch × d) — only one batch at a time | Random batches of 1,024 points. Converges faster, slightly less precise | ✅ Handles millions of points |

**Why MiniBatchKMeans**: It was designed for datasets too large for memory. Each iteration sees a random mini-batch — like reading random pages of a book instead of memorising the whole thing. The clusters are ~99% as good as full KMeans, but it runs in minutes instead of hours (or crashing).

**Evaluation**: Since clustering is unsupervised (no "correct answer"), we use **Silhouette Score** (how well-separated are the clusters?) and **UMAP** (visual 2D projection). Accuracy/F1 don't apply here.

---

## 8. Summarization Model: Mistral Medium 3.5 vs Alternatives *(N05)*

**Problem**: We need to generate blog-style articles from review insights. The model must write coherent, persuasive English. Options range from tiny local models to massive cloud APIs.

| Model | Size | Where | Quality | Cost | Limitations |
|-------|------|-------|---------|------|-------------|
| flan-t5-large | 780 MB | Local Colab | Basic. Repetitive, formulaic | Free | Fits in Colab RAM but text quality is low. Good for MVP, not final |
| Mistral 7B | 14 GB | Local (needs A100) | Good | Free (if you have the GPU) | Doesn't fit in T4 GPU (16 GB VRAM but 14 GB model + overhead = OOM) |
| GPT-4o / Claude | Cloud API | API call | Excellent | ~$0.01-0.03 per article | Expensive at scale, API key needed |
| **Mistral Medium 3.5 (128B)** | Cloud API (NVIDIA) | NVIDIA API | ⭐ Excellent | Free (existing credits) | Requires NVIDIA credits, internet connection |

**Why Mistral Medium 3.5 via NVIDIA**: 
1. **Quality**: 128B parameters produce natural, persuasive text — far beyond what local models can do.
2. **Cost**: You already have NVIDIA API credits. No additional expense.
3. **Convenience**: No GPU required. One API call per article. 128K context window fits the full prompt with all extracted insights.
4. **Why not local?** flan-t5 produces robotic text. Mistral 7B needs an A100 GPU to run. Medium 3.5 via API is the sweet spot.

> **Why NVIDIA and not HuggingFace Inference?** HuggingFace inference endpoints charge per hour (even when idle). NVIDIA charges per token — you only pay for generation. Plus, NVIDIA hosts Mistral Medium 3.5 specifically; HuggingFace doesn't.

---

## 9. Summarization Method: Extractive-Abstractive *(N05)*

**Problem**: How do we turn thousands of reviews into a single article? Pure LLM (dump all reviews into the prompt) would hit token limits and cost a fortune.

| Method | How it works | Pros | Cons |
|--------|-------------|------|------|
| Pure Abstractive | Dump 1,000 reviews into the LLM prompt → "write an article" | Simple | Token limit (even 128K can't fit 1,000 long reviews). Expensive. LLM might hallucinate products that don't exist |
| Extractive-Abstractive | Step 1 (Python): Extract insights (top 3 products, common complaints, worst product). Step 2 (LLM): Feed structured insights → generate article | ✅ Controllable. LLM works with facts, not raw noise. Cheaper (shorter prompt). No hallucinations | Two steps instead of one |

**Why Extractive-Abstractive**: Step 1 uses Python (pandas, n-grams) to extract hard facts from data. Step 2 gives the LLM a clean, structured prompt with precise numbers. The LLM's job is *styling* the facts into a blog article — not discovering them. This is cheaper (smaller prompt), more reliable (no hallucinations), and more transparent (you can trace every claim back to data).

> **Analogy**: Extractive = you read all the reviews and take notes. Abstractive = you hand those notes to a professional writer who crafts the article.

---

## 10. EDA Voice: Research Tone vs Assertive Tone *(N01)*

**Problem**: In a scientific notebook, the language shapes how the reader interprets findings. Assertive language ("reviews are longer when negative") sounds like a proven fact. Research language ("I test whether negative reviews are longer") signals that we are investigating, not declaring.

| | Assertive (old) | Research (new) |
|---|---|---|
| Example | "Amazon data is famously positivity-biased" | "I check whether the dataset shows the positivity bias commonly reported for Amazon reviews" |
| Example | "what the model will learn to distinguish" | "what the model might learn to distinguish across classes" |
| Example | "HTML tags are artefacts" | "HTML tags appear to be artefacts" |
| Reader's impression | The author already knows the answer — the analysis is a formality | The author is genuinely exploring — the analysis might surprise them |
| When appropriate | In a textbook or lecture (you're teaching established knowledge) | In research EDA (you're generating insights from data) |

**Why we chose research tone**: This is a bootcamp project evaluated on critical thinking, not on reciting facts. Every analysis becomes a question to answer, not an answer to illustrate. Even well-documented patterns (like negativity bias in reviews) are presented as "I test whether this holds in our data" — because data can always surprise you.

> **Analogy**: A detective who says "the butler did it — let me show you the evidence" has already closed the case. A detective who says "let's follow the evidence wherever it leads" might find the butler… or might find something unexpected. EDA is detective work, not prosecution.

**What changed (2026-05-02)**: 6 tone fixes applied across N01. Every markdown cell and code comment reviewed. Assertions softened to hypotheses. Certainty replaced with tentativeness. The goal: the notebook reads like an investigation, not a manual.

---

## 11. Stopwords: NLTK + Domain-Specific Hybrid *(N01)*

**Problem**: To analyse word frequencies by sentiment class, we must remove stopwords — words that appear everywhere but carry no meaning ("the", "and", "is"). But standard stopword lists miss review-domain noise terms that dominate Amazon data.

| Approach | Coverage | Domain awareness | Citable? |
|----------|----------|-----------------|-----------|
| Manual list | ~130 words, hand-picked | ✅ Includes "product", "bought", "star" | ❌ Looks like a personal preference |
| NLTK only | 179 words, academically standard | ❌ Leaves "product" as the #1 word in every class — useless insight | ✅ Standard, published resource |
| **NLTK + review-specific (hybrid)** | 179 + 23 = 202 words | ✅ Academic base + curated domain terms | ✅ NLTK is the foundation; additions are documented and justified |

**Why hybrid**: 
1. **Academic credibility**: NLTK's `stopwords.words('english')` is a published, peer-reviewed resource. Citing it shows we know the NLP ecosystem.
2. **Domain precision**: Amazon review text has noise terms NLTK doesn't cover — "bought", "product", "purchased", "star" appear in virtually every review but carry zero sentiment. Without removing them, the frequency analysis shows "product" as the top word for Negative, Neutral, AND Positive — three identical charts, zero insight.
3. **Transparency**: The domain-specific additions are listed explicitly in the notebook with justification. No black box.

> **Installation strategy**: NLTK is installed *only* in the cell that needs it (`cell-top-words`), not in the global `cell-pip-install`. This keeps the notebook's dependency footprint minimal — N02, N03, N04, and N05 never touch NLTK and don't need it.

**Why not spaCy?** spaCy would also work (`spacy.load('en_core_web_sm')` includes stopwords), but it requires a 12 MB model download (`python -m spacy download en_core_web_sm`). For a single frequency analysis in one notebook cell, NLTK's 1 MB download is lighter and faster. spaCy is the better choice when you need full pipeline features (POS tagging, NER, dependency parsing) — overkill for stopword removal.

> **Analogy**: NLTK is a good dictionary. The domain-specific list is like adding local slang that the dictionary doesn't cover. Together they filter noise without losing signal.

---

## Quick Reference: Models at a Glance

| Notebook | Model | Size | Why this one | Paper |
|----------|-------|------|-------------|-------|
| N02 | DistilBERT | 66M params / 260 MB | Fast baseline. Proves the pipeline works | [1910.01108](https://arxiv.org/abs/1910.01108) |
| N03 | RoBERTa | 125M params / 500 MB | Best quality for nuanced sentiment | [1907.11692](https://arxiv.org/abs/1907.11692) |
| N04 (embed) | MiniLM-L6 | 22 MB | Fastest embedding for large datasets | — |
| N04 (cluster) | MiniBatchKMeans | N/A (algorithm) | Scales to millions of points without OOM | — |
| N05 | Mistral Medium 3.5 | 128B params (API) | Best text quality; free via NVIDIA credits | — |

---

## 12. Pipeline Architecture: Why CSVs? *(all notebooks)*

**Problem**: Each notebook produces CSVs and JSONs alongside the Arrow dataset. Why not just read the Arrow dataset directly in every notebook? Isn't this redundant?

The Arrow dataset from N01 is the **foundation** — it contains `text`, `label`, `rating`, `category`, and `parent_asin`. CSVs and JSONs are **augmentations**: each notebook adds columns computed by expensive operations, then passes only the new columns forward. This is ETL caching — you never re-run a GPU training job just to change a clustering parameter.

```
N01: Arrow (text, label, rating, category, parent_asin)
       │
       ├── N02: GPU fine-tuning (2 hrs) → CSV: predicted_label, confidence
       │
       ├── N03: GPU fine-tuning (3 hrs) → CSV: predicted_label, confidence, is_correct
       │
       └── N04: Arrow + CSV → CPU clustering (30 min) → CSV: cluster
              │
              └── N05: Arrow + CSV + JSON → API summarisation (15 min) → MD articles
```

| Stage | Reads | Computes (expensive) | Writes (cheap to re-read) |
|-------|-------|---------------------|--------------------------|
| N01 | HF Hub stream | Sampling 1.4 GB → Arrow | `data/dataset/` (Arrow) |
| N02 | Arrow | DistilBERT fine-tuning (66M params, T4 GPU, ~2 hrs) | `predictions_distilbert.csv` + `metrics_distilbert.json` |
| N03 | Arrow | RoBERTa fine-tuning (125M params, T4 GPU, ~3 hrs) | `predictions_roberta.csv` + `metrics_roberta.json` |
| N04 | Arrow + CSV | MiniLM embeddings + MiniBatchKMeans (CPU, ~30 min) | `clusters.csv` + `cluster_profiles.json` |
| N05 | Arrow + CSV + JSON | Mistral API calls (128B params, cloud, ~15 min) | `category_*_summary.md` |

**Why this works:**

| Principle | Detail |
|-----------|--------|
| **Narrow CSVs** | Every CSV carries only the NEW columns, not the full dataset (5 MB vs 1.4 GB). `predictions_distilbert.csv` has 4 columns; the Arrow has 5. |
| **Runtime join** | N04 and N05 `pd.merge()` Arrow + CSVs at runtime. The original text and metadata always come from Arrow — CSVs only add computed columns. |
| **Caching** | If you change the number of clusters in N04, you re-run from N04 forward. N02 and N03 (GPU training) stay cached. This turns a 5-hour re-run into 30 minutes. |
| **Separation** | `data/dataset/` = Arrow ONLY (immutable foundation). `data/` = intermediate artifacts (regenerable). `data/models/` = trained weights. `data/summaries/` = final deliverable. |

> **Analogy**: The Arrow dataset is a library of raw reviews. Each notebook is a researcher who reads the library, writes a report (CSV), and leaves it on the desk. The next researcher reads the library AND the reports — they don't need to re-do the previous researchers' work.

---

## 13. `load_dataset` API: `split` + `streaming=True` Deprecation *(N01, fixed 2026-05-04)*

**Problem**: The N01 notebook used `load_dataset(id, config, split='full', streaming=True, trust_remote_code=True)` to load Amazon Reviews 2023 categories. This broke on Colab because the `datasets` library version ≥2.19 deprecated the `split` parameter when combined with `streaming=True` for datasets with custom loading scripts. The official Colab environment ships a recent `datasets` version, while local environments with older versions silently accepted the deprecated syntax.

| Pattern | `split` + `streaming` (old) | No `split`, no `streaming` (correct) |
|---------|---------------------------|--------------------------------------|
| API call | `load_dataset(id, config, split='full', streaming=True, ...)` | `load_dataset(id, config, trust_remote_code=True)` + `dataset["full"]` |
| Works in `datasets<2.19` | ✅ Yes (but deprecated) | ✅ Yes |
| Works in `datasets>=2.19` | ❌ **BREAKS** | ✅ Yes |
| Memory behaviour | Explicit lazy loading via `streaming=True` flag | Implicit lazy loading — the dataset's custom script reads JSONL.gz files lazily |
| Official docs source | Derived from generic `datasets` streaming tutorial | [HuggingFace dataset card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023#quick-start) + [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/data_loading/huggingface.html) |

**Why the old pattern broke**: The `datasets` library tightened its validation: when a dataset has a custom loading script (like Amazon Reviews 2023), passing both `split` and `streaming=True` creates ambiguity — the library doesn't know whether to trust the script's split handling or its own streaming split logic. From version 2.19 onward, this combination raises an error.

**Why we pinned `datasets==2.19.0`**: 

| | Pin to `2.19.0` | Always use latest |
|---|---|---|
| Reproducibility | ✅ Exact version, same behaviour forever | ❌ Colab updates silently — your code may break between sessions |
| Bug fixes | ❌ Stuck on March 2024 fixes | ✅ Gets security patches |
| API stability | ✅ Guaranteed | ❌ No guarantee |
| For a bootcamp project | ✅ Nobody will rerun this notebook in production | N/A |

**Decision**: Pin `datasets==2.19.0` and use the no-split-no-streaming pattern. This is a bootcamp deliverable — it needs to run reliably for grading, not for years of production use. The pin guarantees the grader's Colab environment will match what we tested against.

**Official sources confirming the correct pattern** (two independent sources, same result):

| # | Source | URL | Command+F | What it shows |
|---|--------|-----|-----------|---------------|
| 1 | HuggingFace dataset card | `https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023` | `Load User Reviews` | `load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)` — no `split`, no `streaming` |
| 2 | UCSD official site (subpage) | `https://amazon-reviews-2023.github.io/data_loading/huggingface.html` | `Load Review Samples` | Same code as source #1 — `load_dataset(...)` + `dataset["full"]` for reviews |

> **Important**: The root page `https://amazon-reviews-2023.github.io/` uses **native JSON loading**, not HuggingFace `datasets`. It explicitly redirects to the subpage: *"Check data loading examples and Huggingface datasets APIs in **Common Data Loading** section."* When citing, use sources #1 or #2 above — not the root page.

**How we discovered this**: The Colab AI agent flagged `split='full'` + `streaming=True` as an obsolete pattern. Cross-referencing both official sources above confirmed that the correct pattern never included `split` or `streaming` at all — the loading script handles lazy reading internally.

> **Analogy**: You don't need to tell a librarian "give me book #42, but only one page at a time." If the library already sends pages on demand, your extra instruction just confuses the system. The Amazon loading script is that librarian — it already reads lazily from the compressed files.

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
