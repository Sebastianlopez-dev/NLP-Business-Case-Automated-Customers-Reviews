# Concepts — Methodology & Pipeline

> Summarization strategy, voice conventions, stopword design, and data pipeline architecture (N01, N05).  
> See also: [foundation](concepts-foundation.md) · [advanced](concepts-advanced.md) · [enterprise](concepts-enterprise.md)  
> Master index: [concepts.md](concepts.md)

---

| # | Concept | Line |
|---|---------|------|
| 8 | Summarization Model: Gemini 1.5 Flash vs Alternatives | L21 |
| 9 | Summarization Method: Extractive-Abstractive | L42 |
| 10 | EDA Voice: Research Tone vs Assertive Tone | L57 |
| 11 | Stopwords: NLTK + Domain-Specific Hybrid | L77 |
| — | Quick Reference: Models at a Glance | L100 |
| 12 | Pipeline Architecture: Why CSVs? | L112 |
| 13 | `load_dataset` API: `split` + `streaming=True` Deprecation | L151 |

---

## 8. Summarization Model: Gemini 1.5 Flash vs Alternatives *(N05)*

**Problem**: We need to generate blog-style articles from review insights. The model must write coherent, persuasive English. Options range from tiny local models to massive cloud APIs.

| Model | Size | Where | Quality | Cost | Limitations |
|-------|------|-------|---------|------|-------------|
| flan-t5-large | 780 MB | Local Colab | Basic. Repetitive, formulaic | Free | Fits in Colab RAM but text quality is low. Good for MVP, not final |
| Mistral 7B | 14 GB | Local (needs A100) | Good | Free (if you have the GPU) | Doesn't fit in T4 GPU (16 GB VRAM but 14 GB model + overhead = OOM) |
| GPT-4o / Claude | Cloud API | API call | Excellent | ~$0.01-0.03 per article | Expensive at scale, API key needed |
| **Gemini 1.5 Flash** | Cloud API (Google) | Google AI Studio | ⭐ Excellent | Free (generous tier) | Rate limits on free tier, requires internet |

**Why Gemini 1.5 Flash (Migration 2026-05-09)**: 
1. **Quality**: Its ability to handle long-context and follow complex persona instructions makes it ideal for synthesizing business articles from structured facts.
2. **Cost**: The Free Tier from Google AI Studio is currently the most sustainable option for a student bootcamp project (unlike Mistral/NVIDIA credits which are finite).
3. **Speed**: "Flash" is optimized for low latency, making the article generation phase of N05 very fast.
4. **Context**: 1M+ context window is overkill for this task but guarantees we never truncate insights.

> **Why Gemini and not Mistral?** While Mistral Medium 3.5 is an excellent model, the availability of free credits through NVIDIA's platform reached its limit. Gemini 1.5 Flash provides a reliable, long-term free alternative with comparable quality for abstractive synthesis in the product review domain.

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
| N04 (embed) | nomic-embed-text-v1.5 | 137M params / MTEB top-3 | Best clustering quality for review text | — |
| N04 (cluster) | MiniBatchKMeans | N/A (algorithm) | Scales to millions of points without OOM | — |
| N05 | Review Summarization | Gemini 1.5 Flash (Google API) | Best quality for abstractive synthesis; sustainable free tier | — |

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
