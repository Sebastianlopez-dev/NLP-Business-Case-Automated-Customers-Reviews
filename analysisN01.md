# Análisis — Notebook 01: Data Preparation & EDA

**Dataset**: Amazon Reviews 2023 (UCSD / McAuley Lab)
**Notebook**: `notebook_01_data_prep_eda.ipynb` (49 celdas, 5 secciones)
**Output final**: `213,945` reseñas balanceadas en formato HuggingFace Arrow, splits 70/15/15

---

## 1. Estrategia de carga

**Decisión clave**: bajar TODAS las 33 categorías del dataset, no una sola.

- `30,000` reseñas por categoría → ~990,000 objetivo, **976,216 reales** (Subscription_Boxes solo tiene 16K totales)
- Streaming directo desde `mcauleylab.ucsd.edu` con `gzip.GzipFile` sobre `requests.raw`
- Se evita el cache de HuggingFace (ahorra ~2× disco) y la API inestable de `datasets>=2.19`
- `parent_asin`: 633,101 productos únicos · `user_id`: 189,161 usuarios únicos

**Por qué importa**: maximizar diversidad léxica entre dominios (Books vs Automotive vs Beauty) reduce overfitting y prepara el modelo para reseñas “in-the-wild” en N02/N03.

---

## 2. Calidad del dato

| Check | Resultado |
|---|---|
| Missing en columnas críticas (`text`, `rating`) | **0** |
| Texto vacío / whitespace | 203 (0.02%) |
| Duplicados exactos | **77,707 (7.96%)** |
| Casi-duplicados (lower + strip) | **81,901 (8.39%)** |
| Mismo `user_id` + mismo `text` | 34,682 |

**Hallazgo crítico**: los duplicados son reseñas-comodín (`"Great"` ×1960, `"Good"` ×1682, `"Love it"` ×1176). Son 5★ vacíos sin valor semántico y van a ser eliminados aguas abajo por el filtro de longitud mínima (`< 10 chars`).

---

## 3. Distribución de ratings — el sesgo positivo

| Rating | Conteo | % |
|---|---|---|
| 1★ | 63,617 | 6.5% |
| 2★ | 39,967 | 4.1% |
| 3★ | 73,326 | 7.5% |
| 4★ | 143,832 | 14.7% |
| 5★ | **655,474** | **67.1%** |

- Media: **4.32** · Mediana: **5.0**
- Confirma el sesgo positivo reportado para Amazon: **2/3 del corpus es 5★**
- Tras mapear a etiquetas (1-2★→Neg, 3★→Neu, 4-5★→Pos): **81.6% Positivo / 7.6% Neutro / 10.8% Negativo** → desbalance brutal que justifica el undersampling de la Sección 4

---

## 4. Longitudes de texto

### Caracteres (raw)
- Min: 0 · P10: 20 · Mediana: **144** · Media: **313** · P90: 742 · P99: 2,542 · Max: 33,276

### Palabras
- Mediana: 27 · Media: 58 · P99: 452 · Max: 6,040
- **Solo 1.5%** (14,186 reseñas) excederían 512 tokens → el límite de DistilBERT/RoBERTa NO es un problema real para este corpus

### Por clase de sentimiento (chars)
| Clase | Media | Mediana |
|---|---|---|
| Negativo (1-2★) | 295 | 159 |
| **Neutro (3★)** | **405** | **204** |
| Positivo (4-5★) | 307 | 137 |

**Hallazgo no obvio**: las reseñas neutras son las MÁS LARGAS. Tiene sentido — para justificar un 3★ hay que matizar pros y contras; los extremos son más concisos (`"Love it"`, `"Trash"`). Esto importa para el modelo: la longitud puede actuar como feature implícita para la clase neutra.

---

## 5. Señales de metadata

### Verified purchase (76.9% verificadas, 23.1% no)

| Rating | Verified % | Unverified % |
|---|---|---|
| 1★ | 7.2 | 4.3 |
| 4★ | 12.3 | **22.9** |
| 5★ | **69.0** | 60.9 |

**Lectura**: las **no-verificadas se concentran en 4★** (casi 2× la tasa de las verificadas). Sospechoso de incentivized reviews / promos. Las verificadas, en cambio, están más polarizadas (más 1★ y más 5★).

### Helpful votes
- Solo 29.4% reciben ≥1 voto útil
- **Negatividad = utilidad**: 1★ promedia 3.3 votos útiles vs 1.3 en 5★. Los compradores valoran más las quejas detalladas que los elogios.

---

## 6. Sentimiento por categoría

Top categorías más negativas (% Negativo):
- **Subscription_Boxes**: 25% Neg / 11% Neu / 64% Pos
- **Software**: 21% / 11% / 68%
- Amazon_Fashion, All_Beauty, Health_and_Personal_Care: ~14-16% Neg

La mayoría de categorías están en torno al 80% positivo. **Subscription_Boxes** es outlier — probable insatisfacción recurrente con renovaciones automáticas.

---

## 7. Vocabulario distintivo (1★ vs 5★)

Comparación por ratio de frecuencia (≥2× entre clases):
- **31 palabras distintivas de 1★** (vocabulario pobre y repetitivo: queja directa)
- **14,735 palabras distintivas de 5★** (vocabulario amplio y específico al producto)

**Implicación para el modelo**: las reseñas negativas tienen un “sello léxico” compacto y reconocible. Las positivas son léxicamente más diversas — lo que paradójicamente puede dificultar la clasificación de Neutro vs Positivo (más solapamiento de vocabulario).

---

## 8. Otros chequeos

- **Title vs body**: el cuerpo es 14.4× más largo que el título (media). Title medio: 24 chars · Body medio: 313 chars → conviene entrenar sobre el `text`, el `title` es ruido para sentiment.
- **Rango temporal**: Sep 1997 → Ago 2023 (26 años, 310 meses únicos). Útil si después se quiere ver drift temporal.

---

## 9. Pipeline de preprocesamiento

`clean_text()` aplica en orden:
1. `lower()`
2. Decodificar entidades HTML (`html.unescape`)
3. Quitar tags HTML (`<br/>`, etc.)
4. Quitar URLs (`http(s)://...`)
5. Quitar no-ASCII (limitación: rompe acentos — `café` → `caf`. Trade-off aceptado para inglés)
6. Quitar caracteres especiales preservando puntuación básica
7. Colapsar whitespace múltiple

### Drop-rate por etapa

| Etapa | Reseñas | Eliminadas |
|---|---|---|
| Raw cargado | 976,216 | — |
| Drop missing/empty `text`/`rating` | 976,013 | 203 (0.02%) |
| Filtro `len(text) < 10` | **942,346** | 33,667 (3.4%) |
| Balanceado por undersampling | **213,945** | 728,401 descartados |

**Filtro de 10 chars**: deja pasar `"Not great."` (10) pero elimina `"Bad."` (4) y los duplicados `"Ok"`. Conservador pero limpia ruido.

---

## 10. Balanceo y splits

### Balanceo
- Estrategia: **undersampling** a la clase minoritaria (Neutro: 71,315)
- Se descartan: **30,520 negativas** y **697,881 positivas** (sí, 87% del corpus positivo)
- Justificación: simplicidad y transparencia. No se generan datos sintéticos. La pérdida es asumida.

### Splits estratificados (70/15/15)
| Split | Tamaño | %Neg / %Neu / %Pos |
|---|---|---|
| Train | 149,761 | 33.3 / 33.3 / 33.3 |
| Val | 32,092 | 33.3 / 33.3 / 33.3 |
| Test | 32,092 | 33.3 / 33.3 / 33.3 |

### Verificación de leakage
Se compara los **sets de índices originales** entre splits (`train_ids & val_ids`, etc.). Las tres intersecciones dan **0**. Es el chequeo correcto: detecta leakage sin falsos positivos por reseñas duplicadas y sin O(n) memoria de strings.

---

## 11. Salidas escritas a disco

```
OUTPUT_DIR/
├── dataset/                  # HuggingFace Arrow (DatasetDict: train/val/test)
├── dataset_summary.csv       # Metadata por split y label
└── plots/                    # 14+ PNGs (rating dist, text length, top words, etc.)
```

`dataset/` es el contrato hacia N02 (DistilBERT) y N03 (RoBERTa) — ambos hacen `load_from_disk(DATASET_DIR)` y consumen `train` / `validation` / `test`.

---

## 12. Lecturas para los notebooks siguientes

1. **DistilBERT/RoBERTa van a ver clases perfectamente balanceadas** — F1 macro y accuracy serán comparables. Sin balanceo el accuracy estaría inflado por la clase mayoritaria.
2. **El 99% de las reseñas cabe en 512 tokens** → no es necesario truncar agresivamente; `max_length=512` con `truncation=True` cubre todo el corpus útil.
3. **La clase Neutro es la “difícil”**: textos más largos, vocabulario solapado con Positivo, menor separabilidad léxica. Esperar el F1 más bajo en Neutro en N02/N03.
4. **Verified purchase y helpful_vote no se usan como features** — solo como señales de EDA. Pero el dataset Arrow conserva las columnas, así que están disponibles si N04 (clustering) o N05 (summarization) las quiere.
5. **Hay duplicados pre-cleaning**: aunque el filtro `<10 chars` elimina los `"Great"` masivos, conviene recordar que el corpus original tiene 8% de duplicación — relevante si se cambia el threshold a futuro.

---

## 13. Decisiones arquitectónicas a recordar

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| Stream gzip directo desde UCSD | HF `datasets` con `streaming=True` | API inestable en `datasets>=2.19` + cache duplicado en disco |
| Las 33 categorías | Solo 1-2 categorías | Diversidad léxica para evitar overfitting de dominio |
| Undersampling | Oversampling / SMOTE / pesos de clase | Transparencia: ningún dato sintético |
| Filtro `len < 10` | Sin filtro / `len < 20` | Compromiso entre limpiar ruido y no descartar reseñas válidas cortas |
| Verificación de leakage por índices | Comparación de strings | O(n) memoria + sin falsos positivos por duplicados |
| Arrow (HF Datasets) | CSV / Parquet | Streaming nativo en `Trainer` + tipos preservados |
