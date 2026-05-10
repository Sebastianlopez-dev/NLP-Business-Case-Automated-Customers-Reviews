# Análisis N04.1 — Category Clustering (BGE + BERTopic)

> **Notebook**: `notebook_04_1_category_clustering.ipynb` | **Fecha**: 2026-05-10  
> **Pipeline**: Review Text → Emotional Lexicon Filter → BGE-large (1024-dim) → BERTopic (HDBSCAN) → UMAP → Validación  
> **Comparación contra**: N04 (nomic-embed 768-dim + MiniBatchKMeans k=6)

---

## 🎯 Objetivo declarado

N04.1 intentó resolver tres limitaciones de N04:

| Limitación N04 | Solución propuesta en N04.1 |
|----------------|---------------------------|
| Clusters 1 y 5 son sentiment-driven (89-93% positivos, top TF-IDF = palabras evaluativas como "great", "love") | Emotional Lexicon Filter: remover vocabulario evaluativo ANTES de embeddear |
| nomic-embed (768-dim, 256 tokens max) puede no capturar suficiente señal semántica | BGE-large-en-v1.5: 1024-dim, 335M params, 512 tokens max |
| K-Means fuerza TODAS las reviews en un cluster, sin detección de outliers | HDBSCAN: density-based, permite noise (-1) para reviews no clusterizables |

**Hipótesis implícita**: Si removemos el vocabulario emocional, BGE debería agrupar por **tipo de producto**, produciendo clusters con mayor pureza de categoría.

---

## 1. Pipeline Comparison — Side by Side

| Dimensión | N04 (referencia) | N04.1 (experimento) |
|-----------|-----------------|---------------------|
| **Embedding model** | `nomic-ai/nomic-embed-text-v1.5` (137M, 768-dim, 256 tok) | `BAAI/bge-large-en-v1.5` (335M, 1024-dim, 512 tok) |
| **Preprocesamiento** | Raw text (solo limpieza base de N01) | Emotional Lexicon Filter: 63 stopwords (Tier 1 + 2) |
| **Algoritmo clustering** | MiniBatchKMeans k=6 (seleccionado de sweep 2-10) | BERTopic → HDBSCAN con retry ladder (500→800→1000) |
| **Reviews procesadas** | 219,998 | 219,998 |
| **Noise/outliers** | 0 (K-Means asigna todo) | 6,769 (3.08%) |
| **Clusters efectivos** | 6 | 5 |
| **Tiempo total fit** | ~19.6s (MiniBatchKMeans final) | ~2h 09min (3 iteraciones BERTopic: 500, 800, 1000) |
| **GPU** | CPU (nomic-embed no requiere GPU) | T4 GPU (BGE requiere GPU para tiempo razonable) |

---

## 2. The Retry Ladder — De 11 a 5 clusters

El HDBSCAN inicial (min_cluster_size=500) encontró **11 clusters** + noise. Demasiados para el brief (4-6). Se activó la retry ladder:

```
min_cluster_size=500  →  11 clusters  (demasiados, >6 → retry)
min_cluster_size=800  →   7 clusters  (aún >6 → retry)
min_cluster_size=1000 →   5 clusters  ✅ en rango 4-6
```

| Intento | min_cluster_size | n_clusters | n_noise (%) | Tiempo |
|---------|-----------------|------------|-------------|--------|
| #1 | 500 | 11 | 4,101 (1.86%) | ~68 min |
| #2 | 800 | 7 | — | ~60 min |
| #3 | **1000** | **5** | 6,769 (3.08%) | ~60 min |

**Lo que pasó realmente**: A medida que se subió min_cluster_size, HDBSCAN fue tragando los clusters pequeños y fusionándolos en uno gigante. No es que encontró "mejores" clusters — simplemente absorbió la diversidad del embedding en un solo cluster ruidoso.

---

## 3. Resultados Finales — N04.1

### 3.1 Validation Table (Cell 33 output)

| Cluster | Size | % | Purity% | Entropy | Top Terms (c-TF-IDF) | Top Categories | Avg ★ |
|---------|------|---|---------|---------|----------------------|----------------|-------|
| 0 | **204,877** | **93.7%** | 11.6 | 1.585 | like, just, use, good, really | Software(4.1%), Fashion(3.8%), Beauty(3.7%) | 3.06 |
| 1 | 4,758 | 2.2% | 27.4 | 0.563 | works, product, gift, price, good | Clothing(12.0%), Home(8.5%), Health(6.8%) | 4.65 |
| 2 | 1,300 | 0.6% | 22.2 | 0.875 | expected, replenishment, described, expect | Clothing(8.5%), Home(7.5%), GiftCards(6.2%) | 4.42 |
| 3 | 1,278 | 0.6% | 23.9 | 0.345 | product, value, works, didnt, work | Automotive(9.9%), Health(7.3%), PersonalCare(6.7%) | 4.78 |
| 4 | 1,016 | 0.5% | 32.1 | 0.868 | described, listing, stated, thank | GiftCards(15.1%), Automotive(10.8%), ArtsCrafts(6.2%) | 4.32 |

### 3.2 Aggregate Metrics

| Métrica | N04 | N04.1 | Delta |
|---------|-----|-------|-------|
| Clusters con purity >40% | 3/6 | **0/5** | 🔴 Colapso total |
| Mean sentiment entropy | >1.0 (estimado) | **0.847** | 🔴 Más entropy baja = clusters MÁS sentiment-driven |
| Clusters con entropy <1.0 | 2 | **4/5** | 🔴 Peor separación sentimiento/tópico |
| Max category concentration | 44% (Cluster 2: Kindle+Movies+Books) | **15.1%** (Cluster 4: GiftCards) | 🔴 Señal de categoría DILUIDA |
| Ratio max/min cluster size | 4.4× (54k/12k) | **201×** (205k/1k) | 🔴 Desbalance catastrófico |
| Top terms signal | "book", "hair", "smell", "fit", "size" | "like", "just", "use", "good", "product" | 🔴 Genéricos inútiles |

---

## 4. Lo que SÍ funcionó — Aspectos rescatables

### 4.1 ✅ Noise Detection (6,769 reviews, 3.08%)

HDBSCAN identificó correctamente ~7K reviews que no encajan en ningún cluster. Esto es valioso: son reviews atípicas o de nichos demasiado pequeños. K-Means las fuerza a un cluster, HDBSCAN las declara outliers. **Este concepto DEBE preservarse en N04.2.**

### 4.2 ✅ c-TF-IDF topic representation

BERTopic usa class-based TF-IDF que pondera términos por cluster vs corpus, no solo frecuencia cruda. En teoría es superior al TF-IDF plano de N04. El problema no fue el algoritmo de representación — fue que los clusters no tenían suficiente señal para que c-TF-IDF funcionara.

### 4.3 ✅ BGE-large embeddings (1024-dim)

El embedding de BGE es objetivamente mejor que nomic-embed: más dimensiones, más contexto (512 vs 256 tokens), mejor separación tópico/tono en benchmarks. El problema NO fue el embedding — fue lo que se le dio de input (texto filtrado) y cómo se clusterizó (HDBSCAN).

### 4.4 ✅ Emotional Lexicon Filter — concepto correcto

La intención de remover vocabulario evaluativo es CORRECTA. N04 tuvo 2 clusters sentiment-driven (1 y 5). El problema fue la **magnitud** del filtro: 63 stopwords que incluyen palabras con doble función semántica y evaluativa (ej: "good" describe calidad del producto, "like" puede ser verbo descriptivo).

---

## 5. Lo que FALLÓ — Root Cause Analysis

### 5.1 🔴 CLUSTER 0: La catástrofe del 93.7%

204,877 reviews en UN SOLO cluster con 11.6% purity. Esto es peor que random (con 33 categorías, random = ~3% purity, así que 11.6% es marginalmente mejor que random — pero para clustering es un desastre).

**Causa raíz**: El emotional lexicon filter REMOVIÓ la señal que hace distinguibles a las reviews. Al sacar palabras como "good", "great", "love", "like", "nice", "best", "perfect", el texto filtrado perdió ~15-20% de su vocabulario distintivo. Peor aún: estas palabras aparecen en contextos DIFERENTES según la categoría ("good book" vs "good battery" vs "good fit"), y al remover "good", el embedding pierde la capacidad de distinguir esos contextos.

**Evidencia**: Las top terms de N04.1 Cluster 0 son "like, just, use, good, really" — son las palabras que SOBREVIVIERON al filtro (o no estaban en la lista). Son genéricas precisamente porque todo el vocabulario específico de dominio fue obscurecido por el filtrado.

### 5.2 🔴 HDBSCAN con texto filtrado = un solo blob

HDBSCAN es density-based: busca regiones densas en el espacio de embedding. Con el texto sin filtrar (N04 implícito), hay regiones densas claras: libros, belleza, electrónica, moda, etc.

Con el texto filtrado, **todas las reviews colapsan a un espacio semántico más homogéneo**. Las diferencias sutiles que sobreviven (palabras como "hair", "book", "fit") ya no son suficientes para formar regiones densas separadas — forman UN solo blob con 4 "satélites" minúsculos.

**Analogía**: Es como si quisieras agrupar personas por profesión, pero les quitas las palabras "médico", "abogado", "ingeniero", "profesor". Lo que queda es "trabajo", "oficina", "persona" — imposible clusterizar.

### 5.3 🔴 Purity collapse: de 44% a 15.1%

La máxima concentración de categoría en N04 fue 44% (Cluster 2: 17% Kindle + 15% Movies + 13% Books = 45% en 3 categorías). En N04.1, la máxima es 15.1% (Cluster 4: GiftCards). **La pureza colapsó 3×.**

Esto significa que incluso los clusters "buenos" de N04.1 (los 4 pequeños) no logran concentrar categorías tan bien como cualquier cluster de N04 — excepto por la concentración por sentimiento.

### 5.4 🔴 Los clusters pequeños son sentiment-driven, no category-driven

Cluster 3 (1,278 reviews): entropy **0.345** — ESTO es un cluster puramente de sentimiento. Avg rating 4.78★. Es el mismo problema que N04 clusters 1 y 5, pero PEOR (entropy más baja = más puro el sentimiento).

**El emotional lexicon filter NO eliminó la señal de sentimiento** — solo la redujo lo suficiente para que el embedding colapsara, pero no lo suficiente para que HDBSCAN dejara de encontrar islas de sentimiento puro.

---

## 6. ¿Qué pasó realmente con el filtro? — Análisis de Tier 2

El filtro tiene dos tiers:

**Tier 1 (40 palabras)** — Pura emoción: "amazing", "terrible", "wonderful", "horrible". Estas son correctas para filtrar. Raramente son descriptivas.

**Tier 2 (23 frases)** — Juicios de valor: "good product", "waste of money", "highly recommend", "good quality". El problema es que frases como "good quality" o "good value" SÍ contienen señal de categoría en contexto:
- "good quality [leather]" → moda
- "good value [for a phone]" → electrónica  
- "good quality [paper]" → libros

Al remover estas frases, se pierde no solo el sentimiento sino el contexto. Y como el filtro aplica regex con word boundaries, remover "good product" también afecta variaciones como "good product for the price" → "for the price" (pierde el sujeto).

---

## 7. N04 vs N04.1 — Veredicto Final

| Dimensión | Ganador | Margen |
|-----------|---------|--------|
| Balance de clusters | N04 (ratio 4.4×) | N04.1 201× peor |
| Category purity | N04 (max 44%) | N04.1 15.1% (3× peor) |
| Sentiment separation | N04 (2/6 sentiment-driven) | N04.1 (1/5 sentiment-driven, pero peor) |
| Interpretabilidad | N04 ("Salud & Belleza", "Libros") | N04.1 (nombres imposibles de asignar) |
| Utilidad para N05 | N04 (6 artículos accionables) | N04.1 (1 artículo inútil + 4 microscópicos) |
| Noise detection | N04.1 ✅ | Feature que N04 no tiene |
| Embedding quality | N04.1 ✅ | BGE > nomic-embed (en teoría) |
| Costo computacional | N04 (19.6s) | N04.1 (~2h GPU) |

**Veredicto: N04.1 es una regresión en clustering efectivo**. N04 con MiniBatchKMeans + nomic-embed produce clusters utilizables. N04.1 con BERTopic + BGE + filtro produce 1 cluster gigante inútil y 4 micro-clusters.

---

## 8. Recomendaciones para N04.2

### 8.1 Combinar lo mejor de ambos

```
BGE-large (N04.1) + MiniBatchKMeans (N04) + Noise detection (N04.1)
```

| Componente | Origen | Por qué |
|-----------|--------|---------|
| Embedding | N04.1 — BGE-large-en-v1.5 | 1024-dim, mejor separación semántica que nomic-embed |
| Clustering | N04 — MiniBatchKMeans k=6 | Distribución balanceada, escalable, interpretable |
| Preprocesamiento | **Tier 1 ONLY** del filtro N04.1 | Remover emoción pura, preservar vocabulario descriptivo |
| Noise detection | N04.1 — HDBSCAN post-hoc | Identificar outliers después del clustering principal |
| Topic representation | N04.1 — c-TF-IDF | Class-based TF-IDF es superior a TF-IDF plano |

### 8.2 Pipeline propuesto

1. **Tier 1 filter**: Solo las 40 palabras de emoción pura (NO Tier 2)
2. **BGE-large embed**: 1024-dim sobre texto con filtro reducido
3. **MiniBatchKMeans k=6**: Mismo pipeline que N04, mejores embeddings
4. **Post-hoc noise tagging**: HDBSCAN sobre los embeddings para marcar outliers (-1) sin forzar reasignación
5. **c-TF-IDF**: Para topic representation de cada cluster

### 8.3 Validación esperada

Con BGE + filtro reducido + MiniBatchKMeans:
- Category purity debería **mejorar** respecto a N04 (BGE captura más señal semántica)
- Sentiment-driven clusters deberían **reducirse** (filtro Tier 1 ayuda)
- Silhouette debería ser comparable (~0.03-0.04 en este dominio)
- Clusters deberían mantener balance (ratio <10×)

### 8.4 Lo que NO debe repetirse

- ❌ Tier 2 del filtro (frases de juicio de valor) — destruye señal de categoría
- ❌ HDBSCAN como clusterizador principal — en este dominio, la estructura es de "blobs", no de "islas densas"
- ❌ Retry ladder forzando k artificialmente — si HDBSCAN necesita min_cluster_size=1000 para dar 5 clusters, es que no hay 5 regiones densas naturales
- ❌ BERTopic end-to-end sin entender qué hace cada componente

---

## 9. Implicaciones para N05 (Summarization)

N04.1 **NO es usable para N05** tal como está:

- **Cluster 0 (93.7%)**: Imposible de resumir — cubre todas las categorías, todos los sentimientos, sin foco
- **Clusters 1-4 (0.5-2.2%)**: Muy pequeños para artículos de summarization con significancia estadística
- **Noise (3.08%)**: No clusterizables por definición

**Recomendación**: Continuar N05 con los clusters de **N04** (k=6, MiniBatchKMeans, nomic-embed). Si N04.2 produce mejores resultados con BGE + MiniBatchKMeans, migrar N05 a esos clusters. No usar N04.1 para producción.

---

## 10. Artefactos

| Archivo | Contenido | Estado |
|---------|-----------|--------|
| `reviews_filtered.csv` | 219,998 reviews con columna text_filtered | Guardado (no en git, en Drive) |
| `embeddings_bge.npz` | Embeddings BGE (219,998, 1024) | Guardado |
| `cluster_assignments.json` | Asignaciones de cluster N04.1 | Guardado |
| `cluster_profiles_bge.json` | Perfiles c-TF-IDF + métricas | Guardado |
| `umap_coordinates.json` | Coordenadas UMAP 2D para visualización | Guardado |

---

## 11. Key Learnings

1. **Remover vocabulario NO es neutral**: El filtro emocional removió ~15-20% del vocabulario distintivo, colapsando la capacidad del embedding para separar categorías. Lo que funciona en papel (separar tono de tópico) no siempre funciona en la práctica cuando el vocabulario "emocional" también es "descriptivo".

2. **HDBSCAN asume estructura de islas densas**: Amazon reviews en 1024-dim NO forman islas densas — forman un continuo con regiones de mayor densidad relativa. K-Means (que particiona el espacio) es más apropiado para este tipo de datos que HDBSCAN (que busca islas).

3. **El mejor embedding no salva un mal pipeline**: BGE-large es superior a nomic-embed en benchmarks. Pero con input degradado (filtro agresivo) y algoritmo inadecuado (HDBSCAN), el resultado es peor que nomic-embed + MiniBatchKMeans con texto crudo.

4. **La retry ladder es un antipatrón cuando fuerza k**: Subir min_cluster_size hasta que HDBSCAN dé 5 clusters es equivalente a forzar K-Means con k=5 — pero con el efecto secundario de crear UN cluster gigante en vez de varios balanceados.

5. **Noise detection es el único feature rescatable**: La capacidad de HDBSCAN para marcar outliers (-1) es genuinamente útil. Debe integrarse como paso post-clustering, no como algoritmo principal.

---

## 12. Próximos Pasos

1. **N04.2**: Implementar pipeline híbrido (BGE + Tier 1 filter + MiniBatchKMeans + HDBSCAN noise tagging)
2. **Validación cruzada**: Comparar pureza de categoría y entropy de sentimiento contra N04 baseline
3. **N05**: Proceder con clusters de N04 mientras se evalúa N04.2
4. **Revisar Tier 2**: Analizar qué frases del Tier 2 son puramente evaluativas vs. descriptivas — podar la lista
