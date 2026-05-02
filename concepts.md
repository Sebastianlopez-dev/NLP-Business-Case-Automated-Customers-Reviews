# Concepts — NLP Business Case

> Research behind the decisions. Each concept follows: **Problem → Options → Tradeoff → Decision**.  
> Explained simply, like to a 5-year-old, but technically precise.

---

## 1. Data Loading: Streaming vs Full Download *(N01)*

**Problem**: The Amazon Reviews 2023 dataset is 750 GB compressed. Downloading it all would fill any hard drive. Loading it all into RAM is impossible.

| Approach | How it works | RAM used | When to use |
|----------|-------------|----------|-------------|
| Full download | Download all JSONL.gz files → unzip → load into memory | 750 GB+ (impossible) | Never for this dataset |
| HF Streaming | `load_dataset(streaming=True)` reads one record at a time from HuggingFace Hub. Like a straw sipping from a 750 GB ocean — you only hold one sip at a time | ~90 MB per category (the "straw") | ✅ **Always for datasets > RAM** |

**Why we chose streaming**: It's the only way to touch 571M reviews without a datacenter. The data is never downloaded — it flows through the pipeline and only the 30K samples per category stay in RAM.

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

> **Token limit — ultra-verified (2026-05-02):** Both models use exactly **512 tokens** as maximum input length. The [official RoBERTa model card](https://huggingface.co/FacebookAI/roberta-base) states: *"The inputs of the model take pieces of 512 contiguous tokens"* and confirms training at *"a sequence length of 512."* DistilBERT, as a distilled BERT, inherits the same positional embedding architecture. The word count analysis in N01 estimates ~394 English words ≈ 512 tokens, applicable to both models.

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
