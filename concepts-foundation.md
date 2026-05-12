# Concepts — Foundation Decisions

> Core architecture and model decisions (N01–N04). Problem → Options → Tradeoff → Decision.  
> See also: [methodology](concepts-methodology.md) · [advanced](concepts-advanced.md) · [enterprise](concepts-enterprise.md)  
> Master index: [concepts.md](concepts.md)

---

| # | Concept | Line |
|---|---------|------|
| 1 | Data Loading: Streaming vs Full Download | L21 |
| 2 | Storage Format: Arrow vs CSV | L60 |
| 3 | Data Accumulation: Python Dicts vs Stream→Arrow | L75 |
| 4 | Dataset Balancing: Cap vs No Cap | L90 |
| 5 | Sentiment Models: DistilBERT vs RoBERTa | L107 |
| 6 | Embedding Model: MiniLM vs Alternatives | L150 |
| 7 | Clustering Algorithm: MiniBatchKMeans vs KMeans | L166 |

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

**Why we chose no cap**: By the time we reach balancing, we already paid the cost of streaming from UCSD (10-15 min), cleaning text, and labeling sentiment. Discarding data at that point wastes work. Training ~214K examples takes ~1 h on T4 GPU — totally reasonable.

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

> **⚠️ Unequal playing field — N01 ASCII-only cleaning (2026-05-09):** N01's `clean_text()` applies `text.encode('ascii', errors='ignore')`, which strips all non-ASCII characters (accents, emoji, rating symbols like ★, currency signs). Both models receive the same cleaned text, so the comparison is *technically fair*, BUT: RoBERTa's BPE tokenizer operates at the byte level and can encode ANY Unicode character without producing `[UNK]`, while DistilBERT's WordPiece has no such advantage. The ASCII filter nullifies one of RoBERTa's architectural strengths — its universal Unicode coverage. Examples of lost signal: `Pokémon` → `Pokmon`, `★☆☆☆☆` → (empty), `café` → `caf`, `😊` → (empty).
>
> This is a **known limitation, not a bug**. The impact on English Amazon reviews is small (~1-2% accuracy at most; >95% of reviews are ASCII-only), but removing the ASCII filter and replacing it with `unidecode()` (which transliterates rather than deletes: `café` → `cafe`) would give RoBERTa a fairer chance to show its full capability. Deferred — would require regenerating the Arrow dataset from N01.

> **Token limit — ultra-verified (2026-05-02):** Both models use exactly **512 tokens** as maximum input length. The [official RoBERTa model card](https://huggingface.co/FacebookAI/roberta-base) states: *"The inputs of the model take pieces of 512 contiguous tokens"* and confirms training at *"a sequence length of 512."* The [official DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased) states: *"The only constrain is that the result has a combined length of less than 512 tokens."* DistilBERT, as a distilled BERT, inherits the same positional embedding architecture. The word count analysis in N01 estimates ~394 English words ≈ 512 tokens, applicable to both models.
>
> **Training truncation decision — `max_length=256` (2026-05-09):** Both N02 and N03 tokenize at **256 tokens**, not the full 512. Originally set to 128 (2026-05-05), but raised to 256 after N02 execution showed DistilBERT benefited from more context. RoBERTa's BPE tokenization produces ~30% more subwords than DistilBERT's WordPiece, so 256 provides equivalent effective coverage. This is deliberate:
> - Amazon reviews are typically short (N01 EDA: median ~40 words ≈ 52 tokens). 256 tokens cover ~98%+ of reviews without truncation.
> - Self-attention complexity is O(n²): reducing sequence length from 512 to 256 divides training time by ~4× (512²/256² = 4× fewer attention pairs per review).
> - 256 is a well-documented sweet spot in the fine-tuning literature for short-text classification (GLUE SST-2, IMDb, Amazon reviews) when GPU memory permits.
> - **Tradeoff**: ~2% of very long reviews lose their tail content. The quality gain over 128 tokens outweighs the modest time increase on T4 GPU.
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
