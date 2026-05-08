# Concepts — NLP Business Case

> Research behind every decision. Each concept follows: **Problem → Options → Tradeoff → Decision**.  
> Split into 4 sub-files (<200 lines each) for agent readability.

---

## Index

| § | File | Concepts | Lines |
|---|------|----------|-------|
| 1–7 | **[concepts-foundation.md](concepts-foundation.md)** | Data loading, storage, accumulation, balancing, sentiment models, embedding models, clustering algorithm | 177 |
| 8–13 | **[concepts-methodology.md](concepts-methodology.md)** | Summarization model & method, EDA voice, stopwords, quick reference, pipeline architecture, `load_dataset` API | 187 |
| 14–16 | **[concepts-advanced.md](concepts-advanced.md)** | Cross-notebook audit patterns, clustering alternatives (LDA vs K-Means), nomic-embed upgrade | 165 |
| 17–18 | **[concepts-enterprise.md](concepts-enterprise.md)** | BERTopic + HDBSCAN (enterprise clustering), LoRA vs Full Fine-Tuning (both deferred) | 198 |

---

## Quick Lookup

| § | Concept | File |
|---|---------|------|
| 1 | Data Loading: Streaming vs Full Download | [foundation](concepts-foundation.md#L21) |
| 2 | Storage Format: Arrow vs CSV | [foundation](concepts-foundation.md#L60) |
| 3 | Data Accumulation: Python Dicts vs Stream→Arrow | [foundation](concepts-foundation.md#L75) |
| 4 | Dataset Balancing: Cap vs No Cap | [foundation](concepts-foundation.md#L90) |
| 5 | Sentiment Models: DistilBERT vs RoBERTa | [foundation](concepts-foundation.md#L107) |
| 6 | Embedding Model: MiniLM vs Alternatives | [foundation](concepts-foundation.md#L150) |
| 7 | Clustering Algorithm: MiniBatchKMeans vs KMeans | [foundation](concepts-foundation.md#L166) |
| 8 | Summarization Model: Mistral Medium 3.5 | [methodology](concepts-methodology.md#L21) |
| 9 | Summarization Method: Extractive-Abstractive | [methodology](concepts-methodology.md#L42) |
| 10 | EDA Voice: Research Tone vs Assertive Tone | [methodology](concepts-methodology.md#L57) |
| 11 | Stopwords: NLTK + Domain-Specific Hybrid | [methodology](concepts-methodology.md#L77) |
| — | Quick Reference: Models at a Glance | [methodology](concepts-methodology.md#L100) |
| 12 | Pipeline Architecture: Why CSVs? | [methodology](concepts-methodology.md#L112) |
| 13 | `load_dataset` API Deprecation | [methodology](concepts-methodology.md#L151) |
| 14 | N02/N03 Audit: Cross-Notebook Bug Patterns | [advanced](concepts-advanced.md#L17) |
| 15 | Clustering Alternatives: LDA, K-Means vs MiniBatchKMeans | [advanced](concepts-advanced.md#L81) |
| 16 | Embedding Model: nomic-embed vs MiniLM | [advanced](concepts-advanced.md#L138) |
| 17 | Clustering Profesional: BERTopic + HDBSCAN | [enterprise](concepts-enterprise.md#L12) |
| 18 | Fine-Tuning Eficiente: LoRA vs Full Fine-Tuning | [enterprise](concepts-enterprise.md#L97) |
