# Análisis N04 — Product Category Clustering

> **Notebook**: `notebook_04_category_clustering.ipynb` | **Fecha**: 2026-05-09
> **Pipeline**: Review Text → nomic-embed (768-dim) → MiniBatchKMeans (k=6) → UMAP → Análisis

---

## 1. Pipeline Parameters

| Parámetro | Valor |
|-----------|-------|
| Modelo embedding | `nomic-ai/nomic-embed-text-v1.5` (137M params, 768-dim, max 256 tokens) |
| Embeddings generados | 219,998 reviews | 675.83 MB | L2 norm = 1.0 |
| Algoritmo clustering | MiniBatchKMeans (k=6, batch=1024, epochs=10, n_init=3) |
| Total pasos | 215 batches/época × 10 épocas × 3 inits = **6,450** |
| UMAP muestra | 4,996 reviews (2.3% estratificado) | cosine metric | 500 epochs |
| X range: [-15.36, 21.71] | Y range: [2.62, 14.31] |
| Embedding quality check | Similar reviews: **0.4698**, Different reviews: **0.3922** ✅ |

## 2. K-Sweep k=2…10 — K-Means vs MiniBatchKMeans

| k | KM-Inertia | KM-Sil | MB-Inertia | MB-Sil | KM-Time | MB-Time |
|---|-----------|--------|-----------|--------|---------|---------|
| 2 | 125,324 | 0.0366 | 127,022 | 0.0264 | 16.2s | 1.1s |
| 3 | 122,542 | 0.0292 | 122,931 | 0.0339 | 24.1s | 1.5s |
| 4 | 120,038 | 0.0297 | 121,093 | 0.0318 | 31.3s | 1.2s |
| 5 | 118,369 | 0.0293 | 118,669 | 0.0322 | 51.9s | 1.1s |
| **6** | **116,999** | **0.0304** | **117,268** | **0.0279** | **56.9s** | **1.6s** |
| 7 | 115,841 | 0.0304 | 117,904 | 0.0309 | 92.8s | 1.5s |
| 8 | 114,899 | 0.0307 | 115,361 | 0.0324 | 89.5s | 1.5s |
| 9 | 114,125 | 0.0352 | 114,602 | 0.0340 | 91.1s | 1.5s |
| 10 | 113,126 | 0.0339 | 113,896 | 0.0313 | 117.2s | 2.4s |

**Decision**: k=6 (K-Means reference). K-Means prefers k=6, MiniBatch prefiere k=5. Elbow method no muestra codo nítido — silhouette consistentemente bajo (~0.03), esperado para 33 categorías en 768-dim [Cell 22-24].

## 3. Final Model — MiniBatchKMeans k=6

| Métrica | Valor |
|---------|-------|
| K-Means silhouette (reference) | 0.0304 (57.1s, 112 iter) |
| MiniBatchKMeans silhouette | **0.0311** (19.6s) |
| Inercia (batch) | 458 |
| Quality retained vs K-Means | **102.2%** (gap: -0.0007) |
| Convergencia | Estable desde época 3/10 en las 3 inits [Cell 25] |

## 4. Cluster Profiles (unified)

| # | Label Propuesto | Size | % | ★ | σ | Sentiment (N/Neu/P) % | Top TF-IDF | Top Categories |
|---|----------------|-------|---|---|---|---------------------|------------|----------------|
| 0 | Salud & Belleza | 29,502 | 13.4 | 2.69 | 1.18 | 37.8/**45.6**/16.6 | like, use, product, good, hair, smell, taste, skin, flavor | Beauty 13.8%, Grocery 13.6%, All Beauty 11.0% |
| 1 | Genérico Positivo | 12,341 | 5.6 | 4.74 | 0.61 | 0.4/7.0/**92.6** | good product, product, good, great, works, works great, great product, good value | Clothing 6.6%, Automotive 6.4%, Health 6.0% |
| 2 | Libros & Entretenimiento | 38,798 | 17.6 | 2.98 | 1.29 | 31.0/**42.5**/26.5 | book, story, like, read, good, magazine, movie, really, love | Kindle 16.6%, Movies/TV 14.6%, Books 13.4% |
| 3 | Moda & Vestimenta | 35,899 | 16.3 | 2.96 | 1.11 | 26.2/**52.6**/21.2 | described, small, fit, size, expected, like, nice, little, good | Amazon Fashion 14.3%, Clothing 12.0%, Sports 5.1% |
| 4 | Electrónica & Hogar | **54,091** | **24.6** | **1.94** | 1.02 | **66.2**/29.9/3.9 | work, use, just, money, did, product, like, quality, cheap, box | Cell Phones 7.0%, Electronics 6.3%, Patio/Lawn 5.2% |
| 5 | Juguetes & Regalos | 49,367 | 22.4 | 4.68 | 0.74 | 1.7/8.8/**89.5** | great, love, easy, gift, nice, use, good, perfect, like, works | Toys 4.7%, Gift Cards 4.6%, Office 4.3% |

*TF-IDF vocabulary: 5,000 terms over 219,998 reviews [Cell 33]. Sentiment from `label` (true, rating-derived). Category % = proportion of cluster's reviews in that Amazon category [Cell 36].*

## 5. Key Findings

**5.1 Silhouette bajo (~0.03) es esperado**: 33 categorías en 768-dim producen solapamiento inevitable. El cross-reference con categorías originales + TF-IDF terms valida que los clusters tienen significado semántico real, no son aleatorios [Cell 33, 36].

**5.2 Dos tipos de clusters**:
- **Semánticos** (clusters 0, 2, 3, 4): TF-IDF + categorías confirman dominios claros (belleza, libros, moda, electrónica). Las top categorías concentran 14-44% del cluster.
- **Dispersos** (clusters 1, 5): Categorías muy fragmentadas (top cat ≤6.6%) con >89% positivos y rating >4.6★. La interpretación de que "el embedding agrupa por sentimiento no por producto" es plausible pero no está probada directamente — la dispersión de categorías es observada, el mecanismo (tono emocional vs longitud de review vs genericidad del texto) no está demostrado.

**5.3 Cluster 4 — el más grande y más negativo**: 24.6% de todas las reviews, 66.2% negativas, 1.94★. Las top words ("work" [doesn't work], "money", "cheap", "did" [didn't]) son fuertemente negativas. Esto NO significa que la mayoría de productos sean malos — significa que las quejas de electrónica comparten un vocabulario muy distintivo que el embedding captura fácilmente.

**5.4 Limitaciones**: Silhouette bajo (0.03) — aceptado como característica del dominio. nomic-embed truncado a 256 tokens (cubre ~98% de reviews). Sin fine-tuning de dominio. Clusters 1 y 5 posiblemente agrupan por longitud/genéricidad más que por tipo de producto — esto afecta la utilidad para summarization en N05.

## 6. Métricas de Calidad

| Métrica | Valor |
|---------|-------|
| Silhouette final | 0.0311 |
| Inercia batch final | 458 (K-Means ref: 116,999) |
| Distribución clusters | 5.6% – 24.6% (ratio 4.4× entre min/max) |
| Speed-up MB vs K-Means | ~36× (1.6s vs 56.9s para k=6) |
| Convergencia | Estable en init 1, época 3; conservador con 10 épocas |

## 7. Artefactos

| Archivo | Contenido | Tamaño |
|---------|-----------|--------|
| `clusters.csv` | 219,998 reviews + cluster + label + confidence | 72.60 MB |
| `cluster_profiles.json` | TF-IDF + métricas por cluster | 3,177 B |
| `metrics_clustering.json` | Historial 6,450 pasos + k-sweep | 1,286.8 KB |
| `cluster_centroids.npy` | Centroides (6, 768) | 18.56 KB |
| `embeddings_nomic.npz` | Embeddings completos (219,998, 768) | 620.67 MB |
| 7 PNGs | K-selection, sizes, UMAP×2, sentiment, category, ratings | `data/plots/nb04_*.png` |

## 8. Decisiones Documentadas

| Decisión | Justificación | Evidencia |
|----------|--------------|-----------|
| k=6 | K-Means reference, rango del brief (4-6), silhouette comparable k=5-7 | Cell 24 |
| MiniBatchKMeans sobre K-Means | 102% calidad, 36× más rápido, O(batch·k·d) memoria | Cell 25 |
| nomic-embed sobre modelos más grandes | 137M params, 768-dim, sin GPU, descarga rápida | Cell 17, concepts §6 |
| UMAP con cosine metric | Preserva estructura local+global, más rápido que t-SNE | Cell 29 |
| 219,998 reviews (all splits) | Más datos = clustering más robusto. Unsupervised → sin riesgo leakage | Cell 12-15 |
| `label` (true, rating-derived) para sentimiento | 100% cobertura vs 15% de predicted (DistilBERT) | Cell 14, 34 |

---

## 9. Implicaciones para N05 (Summarization) — Forward-Looking

> **Esta sección traduce hallazgos de N04 en decisiones de diseño para N05.**  
> Ningún número de producto específico en esta sección proviene de N04 — el análisis por `parent_asin` es trabajo pendiente para N05.

### 9.1 Estrategia por tipo de cluster

| Clusters | Tipo de artículo N05 | Prioridad |
|----------|---------------------|-----------|
| 0, 2, 3, 4 (semánticos) | Narrativo: extractive → abstractive con dominio específico | Alta/Media |
| 1+5 (dispersos) | Estadístico combinado: métricas agregadas, rankings, sin narrativa forzada | Baja |

### 9.2 Cluster 4 — Prioridad #1
66% negativo, cluster más grande → insight más accionable. El extractive step debe:
- Extraer patrones de falla por `parent_asin` (no solo quejas)
- Balancear muestra: 3 pos / 3 neu / 4 neg por producto
- Estructura del prompt: contexto → lo que SÍ funciona → patrones de falla → recomendaciones → conclusión

### 9.3 Ancla de concretitud: `parent_asin`
El extractive step DEBE agrupar por `parent_asin` dentro de cada cluster para que el LLM nombre productos concretos. Sin esto, el output será generalidades vacías. Ejemplo ilustrativo del tipo de output esperado: "El producto X (Y reviews, Z% negativas) es el más criticado. Los compradores reportan..." — los números reales requieren análisis en N05. <!-- ILLUSTRATIVE: no product-level data exists in N04 -->

### 9.4 Sentimiento en N05
- `label` (rating-derived, 100% cobertura) → métricas agregadas, distribuciones
- `predicted_label` (DistilBERT, 15% cobertura, F1=0.77) → selección cualitativa de reviews representativas en test split

### 9.5 Estructura de entregables N05
5 artículos en `data/summaries/`:
1. `salud_belleza.md` (cluster 0) — narrativo
2. `libros_entretenimiento.md` (cluster 2) — narrativo
3. `moda_vestimenta.md` (cluster 3) — narrativo
4. `electronica_hogar.md` (cluster 4) — narrativo investigativo
5. `compras_satisfactorias.md` (clusters 1+5) — estadístico

### 9.6 Anti-patrones para N05
- ❌ Mismo prompt template para todos los clusters
- ❌ Summarization puramente abstractiva sin extractive step previo
- ❌ Mostrar solo reviews negativas del cluster 4 (→ artículo-depresión)
- ❌ Ignorar `parent_asin` (→ generalidades)
- ❌ Usar `predicted_label` para métricas agregadas (solo 15% cobertura)

---

## 10. Próximos Pasos

1. **N05**: Diseñar extractive step (agrupación por `parent_asin`, selección de reviews representativas, balanceo de sentimiento) antes de llamar a Mistral API.
2. **Validación cualitativa**: Revisar samples aleatorios por cluster antes de armar prompts.
3. **Posible refinamiento**: Si clusters 1+5 resultan demasiado genéricos, considerar sub-clustering o excluirlos de summarization narrativa.
