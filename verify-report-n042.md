## Verification Report — Notebook 04.2 Path Update + Full Code Review

### Task A — Path Update Status
**Status**: ✅ **Successfully updated**

- **Notebook** (`notebook_04_2_hybrid_clustering.ipynb`):
  - Added `N042_OUTPUT_DIR`, `N042_MODELS_DIR`, `N042_PLOTS_DIR` to `cell-paths` with `os.makedirs(..., exist_ok=True)`.
  - Replaced **25** path references throughout the notebook.
  - `DATASET_DIR` left untouched (still points to `data/dataset` from N01).
  - `PRED_PATH` left untouched (reads upstream `predictions_distilbert.csv` from `OUTPUT_DIR`).
  - Verified: **no remaining bare `OUTPUT_DIR` / `MODELS_DIR` / `PLOTS_DIR` references** for N04.2 outputs.

- **Dashboard** (`web/raw-web-vs3.html`):
  - Updated all 3 `fetch()` URLs:
    - `data/n042_cluster_assignments_bge.json` → `data/n042_outputs/n042_cluster_assignments_bge.json`
    - `data/n042_sanity_check.json` → `data/n042_outputs/n042_sanity_check.json`
    - `data/n042_cluster_profiles_bge.json` → `data/n042_outputs/n042_cluster_profiles_bge.json`

---

### Task B — Issues Found

#### CRITICAL (3)

| # | File | Cell | What's Wrong | Fix Needed |
|---|------|------|--------------|------------|
| C1 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-row-filter` | Scattered import: `from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS` inside a data-processing cell. Violates AGENTS.md §5 / trap 7. | Move import to `cell-imports`. |
| C2 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-umap-clusters` | Scattered import: `from umap import UMAP` inside a plotting cell. Violates AGENTS.md §5 / trap 7. | Move import to `cell-imports`. |
| C3 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-sanity-check-code` | Scattered imports: `from sklearn.preprocessing import OneHotEncoder` and `from sklearn.cluster import MiniBatchKMeans` inside a validation cell. Violates AGENTS.md §5 / trap 7. | Move imports to `cell-imports`. |

#### WARNING (5)

| # | File | Cell | What's Wrong | Fix Needed |
|---|------|------|--------------|------------|
| W1 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-merge-data` | Hardcoded label names: `label_names = {lbl: name for lbl, name in zip(unique_labels, ["Negative", "Neutral", "Positive"])}` assumes exactly 3 labels in fixed order. Violates AGENTS.md trap 11. | Build display names dynamically from `df_merged['label'].value_counts().index` using a `label_map` dict. |
| W2 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-tfidf-terms`, `cell-ctfidf-terms`, `cell-entropy`, `cell-metrics-table`, `cell-sentiment-heatmap`, `cell-category-heatmap`, `cell-export-assignments`, `cell-export-profiles`, `cell-export-history`, `cell-checklist` | Missing cross-cell dependency guards. These cells use `df_merged`, `cluster`, `embeddings`, `BEST_K`, `fit_steps`, `epoch_steps` without checking they exist. Violates AGENTS.md trap 6. | Add guards like `if 'df_merged' not in dir() or 'cluster' not in df_merged.columns: raise NameError("Run Section X first.")`. |
| W3 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-export-history` | Metrics JSON structure deviates from N04's `metrics_clustering.json`. Missing `training_history.kmeans`, `k_sweep.kmeans`, and `kmeans_vs_minibatch` keys. Downstream consumers expecting the N04 schema will fail. | Add a comment documenting the schema change, or add a compatibility wrapper if downstream relies on the old keys. |
| W4 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-select-k` | No guard against empty `target_k_values`. If `K_VALUES` is ever changed to exclude 4-6 (e.g. `range(2,4)`), `np.argmax([])` raises `ValueError`. | Add `assert len(target_k_values) > 0, "No k values in target range 4-6"`. |
| W5 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-hdbscan` | HDBSCAN retry ladder has only **one rung** (1000 → 2000). If noise is still >15% after retry, the code accepts it with no further fallback. | Add a second fallback (e.g., `min_cluster_size=3000`) or a hard cap that logs a validation warning. |

#### SUGGESTION (3)

| # | File | Cell | What's Wrong | Fix Needed |
|---|------|------|--------------|------------|
| S1 | `notebook_04_2_hybrid_clustering.ipynb` | Multiple | Magic numbers without 1-line justification comments. Violates AGENTS.md trap 9. Examples: `MIN_TEXT_WORDS = 20`, `MAX_SAMPLES = 220_000`, `MB_N_EPOCHS = 10`, `MB_N_INIT = 3`, `MB_BATCH_FIT = 1024`, `EVAL_SAMPLE_SIZE = min(10_000, ...)`, `SIL_SAMPLE_SIZE = min(10_000, ...)`, `SAMPLE_SIZE = 5000`, `TOP_N_TERMS = 10`. | Add a brief comment above each explaining the threshold (e.g. `# 20 words: filters out "i like it thanks" generic reviews`). |
| S2 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-validate-upstream` | No Arrow schema validation. Only checks that split directories exist. If N01 changes column names (`text`, `label`, `rating`, `category`), downstream cells will fail with cryptic `KeyError`. | Add a schema guard: `assert set(['text','label','rating','category']).issubset(df_all.columns), "Unexpected schema"`. |
| S3 | `notebook_04_2_hybrid_clustering.ipynb` | `cell-export-history` | Missing `batches_per_epoch` in `training_history.minibatch_kmeans`. N04 includes it; N04.2 omits it. | Add `"batches_per_epoch": n_batches_per_epoch` to the JSON export for parity with N04. |

---

### Cross-Reference Checklist (N04 vs N04.1 vs N04.2)

| Check | Status | Notes |
|-------|--------|-------|
| N04 `cell-k-sweep`: logic lost in N04.2? | ✅ Intentionally dropped | N04.2 runs **only** MiniBatchKMeans (no K-Means gold-standard comparison). Documented in markdown. |
| N04 `cell-fit-minibatch`: per-step tracking intact? | ✅ Yes | `fit_steps` and `epoch_steps` structures are identical (step, init, epoch, batch, inertia, time_ms / eval_inertia, train_inertia). Variable names changed from `mb_fit_steps` → `fit_steps`. |
| N04.1 `cell-emotional-filter`: Tier 2 words NOT in N04.2? | ✅ Verified absent | N04.2 explicitly uses **Tier 1 only (40 words)** and adds a row-level length filter. No `STOPWORDS_TIER2` defined. |
| N04.1 `cell-load-bge`: model loading identical? | ✅ Yes | Same model (`BAAI/bge-large-en-v1.5`), same `max_seq_length=256`, same `normalize_embeddings=True`. |
| N04 `cell-save-metrics`: metrics JSON structure maintained? | ⚠️ Partially | Per-step tracking structure is maintained, but `kmeans` keys and `kmeans_vs_minibatch` comparison are missing (expected since N04.2 doesn't run K-Means). See **W3**. |
| N04.1 BERTopic retry ladder NOT copied? | ✅ Confirmed absent | N04.2 uses MiniBatchKMeans + post-hoc HDBSCAN noise tagging. No BERTopic, no 300/200/100/800/1000 retry ladder. |

---

### Edge Cases

| Case | Handled? | Notes |
|------|----------|-------|
| `BEST_K` not in range 4-6? | ⚠️ Partially | `target_k_values` filters to 4-6, but no guard against empty list. See **W4**. |
| Noise tagging >15%? | ⚠️ Minimal ladder | Retries once at `min_cluster_size=2000`. No further fallback. See **W5**. |
| Arrow dataset has unexpected schema? | ❌ No guard | Only directory existence checked. See **S2**. |
| Predictions CSV missing? | ✅ Guarded | `assert os.path.exists(PRED_PATH)` in `cell-validate-upstream`. |

---

### Verdict

**PASS WITH WARNINGS**

- **Paths**: All N04.2 outputs now route to `data/n042_outputs/` (JSON/CSV), `data/n042_outputs/models/` (NPZ/NPY), and `data/n042_outputs/plots/` (PNG). Dashboard fetch URLs updated. No regressions on upstream paths (`DATASET_DIR`, `PRED_PATH`).
- **Code quality**: 3 CRITICAL scattered imports must be moved to `cell-imports` before the notebook is considered compliant with AGENTS.md §5. 5 WARNINGS should be addressed before merge. 3 SUGGESTIONS are polish items.

**Total issues count: 11** (3 CRITICAL, 5 WARNING, 3 SUGGESTION)
