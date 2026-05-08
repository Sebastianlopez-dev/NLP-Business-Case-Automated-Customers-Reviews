# web-to-enhance.md — raw-web-vs3.html Audit

> Audit ejecutado: 2026-05-08 | Archivo: `web/raw-web-vs3.html` (961 líneas)
> Enfoque: bugs funcionales, accesibilidad, performance, calidad de código, diseño responsivo.

---

## 🔴 CRÍTICOS (bugs funcionales)

### 1. Chart.js se inicializa en tabs ocultos → dimensiones 0

**Líneas:** 879–901 (`clusterChart` en `initCharts()`), 421–461 (`tab-clusters` con `display:none`)

**Problema:** `document.addEventListener('DOMContentLoaded', ...)` llama a `initCharts()` que crea **ambos** charts de una sola vez. Pero `clusterChart` vive dentro de `#tab-clusters`, que arranca con `display:none` (solo `tab-eda` tiene `active`). Chart.js intenta renderizar sobre un canvas con dimensiones 0×0 → el chart queda invisible o deformado incluso después de cambiar de tab.

**Fix:**
```js
// Opción A: Lazy init al cambiar de tab
switchTab: function(id) {
  // ...
  if (id === 'tab-clusters' && !this.charts.cluster) {
    this.initClusterChart();
  }
}

// Opción B: Resize al cambiar de tab
if (id === 'tab-clusters') {
  this.charts.cluster.resize();
}
```

**Severidad:** Alta — el cluster chart es uno de los 2 charts del dashboard y nunca se ve correctamente.

---

### 2. Sidebar fijo de 240px sin comportamiento responsivo

**Líneas:** 194 (`w-[240px] flex-shrink-0`), 191 (`flex h-screen w-full`)

**Problema:** El layout usa flexbox con sidebar fijo de 240px + main flexible. En viewports < 768px el sidebar ocupa ~60% del ancho disponible, aplastando el contenido. No hay media queries, hamburger menu, ni `transform` para ocultarlo.

**Fix mínimo:**
```css
@media (max-width: 768px) {
  aside { display: none; }
  aside.open { display: flex; position: fixed; z-index: 50; height: 100vh; }
}
```
Agregar un botón hamburger en el header que togglee `aside.open`.

**Severidad:** Alta — la app es inusable en mobile.

---

## 🟠 ALTOS (accesibilidad y UX)

### 3. Navegación sin ARIA labels ni roles explícitos

**Líneas:** 194 (`<aside>`), 204 (`<nav>`), 593 (botón Evaluate)

**Problema:**
- El `<aside>` no tiene `aria-label` (lector de pantalla anuncia "complementary" genérico).
- Los `<button>` de navegación línea 206–236 no tienen `aria-label` ni `aria-current="page"` (solo clase `.active` visual).
- El botón "Evaluate" línea 593 cambia su texto dinámicamente pero nunca anuncia el cambio a screen readers.
- El avatar placeholder línea 256 es un `<div>` vacío sin `aria-hidden="true"` — el lector puede intentar leerlo.

**Fix:**
```html
<aside aria-label="Main navigation" ...>
<button aria-current="page" aria-label="Data and EDA section" ...>
<div aria-hidden="true" class="w-8 h-8 rounded-full ..."></div>
```

**Severidad:** Alta — inaccesible para usuarios con lectores de pantalla.

---

### 4. `runInference()` es regex simulation sin indicarlo claramente al usuario

**Líneas:** 904–951

**Problema:** El Playground promete "Live simulation of combined inference: Transformer Evaluator → LLM Synthesis" pero implementa un `string.match()` con 12 keywords hardcodeadas y un `Math.random()` para el confidence. El usuario no tiene forma de saber que es una simulación estática. Los resultados están pre-escritos (líneas 927–938) y no dependen del modelo seleccionado.

**Fix:** Agregar un badge visible "SIMULATION" en el tab de Playground. O al menos un `console.warn` / tooltip que aclare que no hay modelo real corriendo.

**Severidad:** Media — misleading para quien abra el dashboard por primera vez.

---

### 5. Tailwind CSS via CDN en producción (300KB+ runtime)

**Línea:** 8 (`<script src="https://cdn.tailwindcss.com"></script>`)

**Problema:** El CDN de Tailwind es un **compilador JIT que corre en el browser**. Pesa ~300KB, parsea todo el DOM buscando clases, y genera CSS en runtime. Esto causa:
- Flash of unstyled content (FOUC)
- 300KB de JS bloqueante
- CPU time perdido en cada carga de página

**Fix:** Generar un build estático con solo las clases usadas (~10KB en vez de 300KB):
```bash
npx tailwindcss -i input.css -o output.css --minify
```
O mantener el CDN solo para desarrollo con un comment `<!-- DEV ONLY -->`.

**Severidad:** Alta — performance degradation significativa.

---

## 🟡 MEDIOS (calidad de código)

### 6. CSS custom scrollbar solo para WebKit (sin soporte Firefox)

**Líneas:** 153–169

**Problema:** Solo se usa `::-webkit-scrollbar`. Firefox usa `scrollbar-width` y `scrollbar-color`. En Firefox los scrollbars se ven con el estilo default del OS (rompiendo el diseño "app-like").

**Fix:**
```css
* {
  scrollbar-width: thin;
  scrollbar-color: #E2E8F0 transparent;
}
```

---

### 7. Variables JS globales (`app` en `window`)

**Líneas:** 739–952

**Problema:** `const app = { ... }` en el scope global contamina `window.app`. Cualquier script externo o extensión del browser puede leer/modificar el objeto. Además, `runInference`, `switchTab`, `toggleCode` son accesibles desde la consola sin restricción.

**Fix:** Envolver en IIFE:
```js
(function() {
  const app = { ... };
  document.addEventListener('DOMContentLoaded', () => app.initCharts());
})();
```
Exponer solo lo necesario si otros scripts lo requieren.

---

### 8. Inline style debería ser una clase CSS

**Línea:** 505 (`style="height: auto; min-height: 280px;"`)

**Problema:** Tiene un inline style que sobreescribe `.code-viewport { height: 280px; }`. Mezcla dos approaches. Si el height fijo es el default, el `height: auto` debería ser una clase utilitaria como `.code-viewport-auto`.

**Fix:**
```css
.code-viewport-auto { height: auto; min-height: 280px; }
```
```html
<div class="code-viewport code-viewport-auto" id="llm-code-display">
```

---

### 9. Sin `<meta name="description">` ni favicon

**Líneas:** 4–189 (`<head>` completo)

**Problema:** El `<head>` no tiene:
- `<meta name="description">` — afecta SEO y link previews (Open Graph, Slack, WhatsApp).
- `<link rel="icon">` — la tab del browser se ve genérica.

**Fix:**
```html
<meta name="description" content="NLP Business Case — Automated Customer Reviews Dashboard. Sentiment analysis, clustering, and summarization of Amazon Reviews 2023.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📊</text></svg>">
```

---

### 10. Google Fonts sin `display=swap` → FOIT (Flash of Invisible Text)

**Línea:** 15

**Problema:** La URL de Google Fonts ya incluye `&display=swap` (✅ correcto), pero no tiene `font-display: swap` como fallback en el CSS. Si Google Fonts falla o tarda, los textos en Inter/JetBrains Mono quedan invisibles hasta que cargue.

**Fix:** Ya está bien en este caso — la URL tiene `display=swap`. Pero conviene agregar la regla en CSS también como defensa:
```css
body { font-display: swap; }
```

**Severidad:** Baja (ya está corregido en la URL).

---

### 11. Heading hierarchy rota — `<h1>` en sidebar, no en main

**Líneas:** 198 (`<h1>` en sidebar), 264–735 (tabs arrancan con `<p>` o `<h2>`)

**Problema:** El `<h1>` "NLP Business" está en el sidebar. El contenido principal (`<main>`) no tiene `<h1>`, solo `<p>` (línea 266) o `<h2>` (línea 328). Un lector de pantalla que navegue por headings encontrará `h1 → h2 → h2` pero el `h1` está fuera del flujo de contenido principal.

**Fix:** Mover el `<h1>` al header principal, o usar `<span>` en el sidebar y `<h1>` en cada tab:
```html
<h1 class="text-2xl font-bold ...">Data & EDA</h1>
```
El sidebar puede usar `<span class="text-sm font-bold">` en vez de `<h1>`.

---

## 🟢 BAJOS (nice-to-haves)

### 12. Sin manejo de errores para CDNs externas

**Líneas:** 8 (Tailwind), 9 (Chart.js), 19 (Phosphor Icons)

**Problema:** Si `cdn.tailwindcss.com` o `cdn.jsdelivr.net` están caídos, la página pierde todo el estilo y los charts. No hay fallback ni mensaje de error.

**Fix:**
```html
<script src="https://cdn.tailwindcss.com"></script>
<script>if (!window.tailwind) document.body.innerHTML = '<p style="padding:2rem">Failed to load styles. Check your connection.</p>';</script>
```
O mejor: empaquetar los assets localmente.

---

### 13. `overflow: hidden` en body bloquea scroll nativo

**Línea:** 37

**Problema:** `body { overflow: hidden; }` con comentario "App-like feel, scroll handled in main". Esto es un anti-patrón de SPA: deshabilita el scroll nativo del browser (rueda del mouse, swipe en touchpad, Page Up/Down). Si el `<main>` no captura el foco, el usuario no puede scrollear.

**Fix:** Eliminar `overflow: hidden` del body. Mantener solo `h-screen` y el scroll en `<main>`. El "app-like feel" se logra con `height: 100vh` + `overflow-y: auto` en main, sin tocar body.

---

## 📊 Resumen

| Prioridad | Cantidad | Issues |
|-----------|----------|--------|
| 🔴 Crítico | 2 | #1 (Chart.js hidden tabs), #2 (no responsive) |
| 🟠 Alto | 3 | #3 (ARIA), #4 (simulación misleading), #5 (Tailwind CDN) |
| 🟡 Medio | 4 | #6 (Firefox scrollbar), #7 (global scope), #8 (inline style), #9 (meta+favicon) |
| 🟢 Bajo | 2 | #10 (FOIT - ya OK), #11 (heading hierarchy), #12 (CDN fallback), #13 (overflow hidden) |

**Recomendación:** Arrancar por #1 y #2 (bugs reales), después tacklear #3 y #5 (accesibilidad y performance). Los demás se pueden ir resolviendo por lote.

---

## 📊 MEJORAS DE CONTENIDO — Basadas en N01 (Data Prep & EDA)

> Audit ejecutado: 2026-05-08 | Fuente: `notebook_01_data_prep_eda.ipynb` + `analysisN01.md`
> Enfoque: qué datos descubrió N01 que la web no está mostrando, y cómo integrarlos.

---

### Sección 1 — Data & EDA (tab-eda)

#### 14. El "571.54M → 213.9K" no explica la ruta de reducción

**Líneas:** 291-298 (hero corpus volume), 317-320 (Natural Imbalance card)

**Problema:** La web muestra dos números enormes — 571.54M (corpus total) y 213.9K (corpus final balanceado) — pero **nunca explica cómo se pasó de uno al otro**. El usuario no sabe que hubo un muestreo, una limpieza, y un balanceo. N01 documenta cada etapa con precisión:

| Etapa | Reseñas | Qué pasó |
|---|---|---|
| Amazon Reviews 2023 completo | 571.54M | Dataset UCSD completo |
| Muestreo (33 cats × 30K) | 976,216 | Streaming desde UCSD sin tocar disco |
| Drop missing/empty | 976,013 | Solo 203 rows (0.02%) — dataset muy limpio |
| Filtro `len(text) < 10` | 942,346 | Elimina `"Great."`, `"Ok"`, duplicados vacíos |
| Balanceo (undersampling) | **213,945** | Minoría (Neutral: 71,315) × 3 clases |

**Fix:** Agregar un **mini pipeline visual** de 4 pasos en el hero de la sección Data & EDA: un row de 4 cards pequeñas con el conteo en cada etapa y una flecha entre ellas. Algo como:

```
[571.54M] → [976K muestreado] → [942K limpio] → [214K balanceado]
  Total       33 cats × 30K       -3.4% ruido       ×3 clases
```

Esto cuenta la historia real del pipeline y le da contexto al número final.

**Severidad:** Alta — sin esta narrativa, los números parecen arbitrarios.

**Fuente N01:** `analysisN01.md` §9 (drop-rate por etapa), §1 (estrategia de carga)

---

#### 15. La class distribution solo muestra 3 clases colapsadas — falta la distribución por estrellas (1★–5★)

**Líneas:** 303-312 (EDA chart con solo Neg/Neu/Pos)

**Problema:** El chart `edaChart` muestra 3 barras (10.8% Neg, 7.6% Neu, 81.6% Pos). Pero N01 tiene la distribución **por estrella individual** que es mucho más informativa:

| Rating | Conteo | % |
|---|---|---|
| 1★ | 63,617 | 6.5% |
| 2★ | 39,967 | 4.1% |
| 3★ | 73,326 | 7.5% |
| 4★ | 143,832 | 14.7% |
| 5★ | **655,474** | **67.1%** |

El dato de que **2/3 del corpus es 5★** y que la mediana es 5.0 es mucho más impactante que el 81.6% colapsado.

**Fix:** Agregar un **segundo chart** (o un toggle en el existente) que muestre la distribución 1★–5★ con barras individuales. La barra de 5★ al 67.1% es el verdadero "wow moment" del EDA.

**Severidad:** Alta — la distribución por estrella es el hallazgo #1 del EDA y no está.

**Fuente N01:** `analysisN01.md` §3, cell `cell-rating-dist`

---

#### 16. Las reseñas NEUTRAS son las más largas — hallazgo no obvio que no se muestra

**Líneas:** 317-320 (Natural Imbalance card solo habla de skew positivo)

**Problema:** N01 descubrió que las reseñas neutras (3★) son las **más largas** en promedio: 405 chars vs 295 (Neg) y 307 (Pos). Es contraintuitivo — uno esperaría que las quejas sean las más detalladas. Pero no: para justificar un 3★ el usuario matiza pros y contras, mientras que los extremos son más concisos (`"Love it"`, `"Trash"`).

Este hallazgo es:
- Un **insight memorable** para cualquier presentación
- Una **señal para el modelo**: la longitud puede actuar como feature implícita para detectar la clase Neutra
- Un **puente narrativo** hacia la Sección 2 (Architecture), donde se decide `max_length=128`

**Fix:** Reemplazar o expandir la card "Natural Imbalance" para que incluya una tabla de 3 filas:

```
Text Length by Sentiment
─────────────────────────
Negative  295 chars avg  (median 159)
Neutral   405 chars avg  (median 204)  ← más largas
Positive  307 chars avg  (median 137)
```

Y una línea de insight: *"Neutral reviews are the longest — users justify ambivalence with detail."*

**Severidad:** Alta — es el hallazgo más contraintuitivo del EDA y no está en la web.

**Fuente N01:** `analysisN01.md` §4, cell `cell-text-length-by-sentiment`

---

#### 17. Solo 1.5% de las reseñas exceden 512 tokens — el `max_length=128` está justificado pero no se explica

**Líneas:** 380-392 (Spatial Optimization card en tab-benchmark)

**Problema:** La Sección 2 (Architecture) dice *"Truncating from 512 to 128 speeds up training by 4× due to quadratic attention reduction, safely retaining context for ~95% of reviews (median ~52 tokens)."* Pero este claim no está respaldado con datos visibles en la Sección 1 (EDA).

N01 calculó:
- Mediana de palabras: **27**
- Media de palabras: **58**
- Reseñas que excederían 512 tokens (~394 palabras): **14,186 (1.5%)**

**Fix:** Agregar una card en la Sección 1 (Data & EDA) con la distribución de word count, mostrando que el 98.5% del corpus cabe holgadamente en 128 tokens. Esto crea un **puente narrativo explícito** entre la decisión de arquitectura y el EDA que la justifica.

La card en la Sección 2 puede entonces referenciar este dato: *"As shown in EDA (§1), 98.5% of reviews fit within 128 tokens."*

**Severidad:** Media — mejora la cohesión narrativa entre secciones.

**Fuente N01:** `analysisN01.md` §4, cell `cell-word-count`

---

#### 18. El 8.4% del corpus raw son duplicados — dato de calidad ausente

**Líneas:** 282-345 (toda la sección Data & EDA)

**Problema:** N01 encontró **81,901 reseñas duplicadas (8.39%)** en el corpus raw. Son mayormente reseñas-comodín: `"Great"` ×1960, `"Good"` ×1682, `"Love it"` ×1176. La web nunca menciona data quality issues.

Este dato:
- Demuestra rigor analítico (no solo se cargaron datos, se **auditaron**)
- Explica por qué el filtro `len(text) < 10` es necesario (elimina estos duplicados vacíos)
- Es un "gotcha" que cualquier científico de datos aprecia

**Fix:** Agregar una mini-card de "Data Quality" en la sección EDA:

```
Data Quality
────────────
Duplicate reviews   81,901 (8.4%)
  Top: "Great" ×1,960 | "Good" ×1,682 | "Love it" ×1,176
Missing text/rating  203 (0.02%)
```

**Severidad:** Baja — es un nice-to-have que muestra rigor.

**Fuente N01:** `analysisN01.md` §2 (calidad del dato), cell `cell-duplicate-detection`

---

#### 19. Las categorías NO son iguales en sentimiento — la web solo muestra el agregado

**Líneas:** 303-312 (EDA chart agregado), 475-478 (Category Anomaly en tab-clusters pero sin datos)

**Problema:** La web menciona *"Subscription_Boxes emerges as a negative outlier (25% Neg)"* recién en la Sección 3 (Clustering), pero este es un hallazgo del **EDA**, no del clustering. N01 tiene la tabla completa de 33 categorías con su % Neg/Neu/Pos.

Las 5 categorías más negativas según N01:
| Categoría | % Neg | % Neu | % Pos |
|---|---|---|---|
| Subscription_Boxes | 25% | 11% | 64% |
| Software | 21% | 11% | 68% |
| Amazon_Fashion | 14% | 11% | 75% |
| All_Beauty | 16% | 9% | 75% |
| Magazine_Subscriptions | 15% | 7% | 77% |

**Fix:** Agregar una **tabla "Top 5 Most Critical Categories"** en la sección Data & EDA, con las categorías más negativas. Esto:
- Muestra que el análisis no es solo agregado
- Justifica por qué se incluyeron las 33 categorías (diversidad de sentimiento entre dominios)
- Le da contexto real al outlier de Subscription_Boxes que después aparece en clustering

**Severidad:** Media — enriquece significativamente la narrativa del EDA.

**Fuente N01:** `analysisN01.md` §6, cell `cell-sentiment-by-category`

---

#### 20. Las reseñas negativas reciben 2.5× más "helpful votes" que las positivas

**Líneas:** No existe en la web.

**Problema:** N01 analizó la columna `helpful_vote`: solo 29.4% de reseñas reciben ≥1 voto útil, pero **1★ promedia 3.3 votos vs 1.3 en 5★**. Los compradores valoran más las quejas detalladas que los elogios.

Este es un hallazgo de **sociología del consumo** que le da profundidad al dashboard — no es solo "clasificamos sentimiento", es "entendemos qué reseñas impactan más a otros compradores".

**Fix:** Agregar una card en Data & EDA:

```
Helpful Votes by Rating
───────────────────────
1★ reviews avg 3.3 helpful votes
5★ reviews avg 1.3 helpful votes

→ Negative reviews are 2.5× more "useful" to other shoppers.
```

**Severidad:** Baja — insight de color, no esencial para el pipeline. Pero memorable.

**Fuente N01:** `analysisN01.md` §5, cell `cell-helpful-votes`

---

#### 21. "Verified purchase" analysis — las no-verificadas se concentran en 4★

**Líneas:** No existe en la web.

**Problema:** N01 encontró que el 76.9% de reseñas son "verified purchase", pero las **no-verificadas** están sobrerrepresentadas en 4★ (22.9% de no-verificadas vs 12.3% de verificadas). Esto es sospechoso de *incentivized reviews* / promos.

**Fix:** Agregar una mini-card o un bullet en la sección de Data Quality:

```
Verified vs Unverified
──────────────────────
76.9% verified  |  23.1% unverified
⚠ Unverified reviews cluster at 4★ (potential incentivized reviews)
```

**Severidad:** Baja — detalle técnico que muestra profundidad de análisis.

**Fuente N01:** `analysisN01.md` §5, cell `cell-verified-purchase`

---

#### 22. Vocabulario distintivo: 1★ usa solo 31 palabras únicas dominantes vs 14,735 en 5★

**Líneas:** No existe en la web.

**Problema:** N01 comparó el vocabulario distintivo entre ratings extremos (1★ vs 5★) por ratio de frecuencia. Resultado: las reseñas de 1★ tienen un **sello léxico compacto** (31 palabras dominantes como "waste", "return", "broke"), mientras que las de 5★ despliegan **14,735 palabras distintivas** — vocabulario mucho más rico y específico al producto.

Esto tiene una implicación directa para el modelo que la web podría mencionar: *"Negative reviews have a compact lexical fingerprint — easier to classify. Positive/Neutral share more vocabulary, making the Neutral class the hardest to separate."*

**Fix:** Agregar una card o un callout en la sección Data & EDA con estos números y la implicación para el modelo. Conecta el EDA con la decisión de arquitectura (por qué RoBERTa > DistilBERT para la clase Neutra).

**Severidad:** Baja — insight avanzado, pero le da profundidad académica al dashboard.

**Fuente N01:** `analysisN01.md` §7, cell `cell-extreme-words`

---

### Sección 2 — Architecture & Benchmark (tab-benchmark)

#### 23. La arquitectura de streaming zero-disk está invisibilizada

**Líneas:** 297-298 (solo dice "Disk Cost: 0 GB (Raw JSONL Streaming)")

**Problema:** Una de las decisiones arquitectónicas más impresionantes de N01 — streamear 750GB de `.jsonl.gz` desde UCSD sin tocar disco, usando `requests.get(stream=True)` + `gzip.GzipFile` — está reducida a **un texto de 10 palabras** en el footer del hero number. Esto es un flex técnico que merece su propio espacio visual.

**Fix:** Agregar una card en la Sección 2 (Architecture) que explique el streaming pipeline:

```
Zero-Disk Streaming Architecture
────────────────────────────────
① requests.get(stream=True)  →  HTTP chunked download
② gzip.GzipFile              →  Decompress on-the-fly
③ json.loads(line)           →  Parse one review at a time
④ Accumulate in RAM          →  ~90 MB per category

Result: 750 GB of raw data processed without a single byte
touching disk. Colab's 225 GB drive stays free for training.
```

Esto reemplaza el texto genérico actual y le da peso arquitectónico a la decisión.

**Severidad:** Media — es el diferenciador técnico más fuerte del proyecto.

**Fuente N01:** `analysisN01.md` §1, §13 (decisiones arquitectónicas), cell `cell-streaming-load`

---

#### 24. El umbral `max_length=128` está justificado en la Sección 2 pero no conectado al EDA de la Sección 1

**Líneas:** 380-392 (Spatial Optimization card)

**Problema:** La card dice *"safely retaining context for ~95% of reviews (median ~52 tokens)"* pero este dato viene del EDA (Sección 1) y no hay un link visual ni textual entre ambas secciones.

**Fix:** En la card "Spatial Optimization", cambiar el texto para que referencie explícitamente el hallazgo del EDA:

```
Truncating from 512 to 128 tokens speeds up training by 4×.
EDA (§1) confirmed: 98.5% of reviews fit within 128 tokens
(median 27 words → ~52 tokens). The <1.5% truncated are
outlier essays — their sentiment signal is in the first 128
tokens.
```

Y en la Sección 1, asegurarse de que el word count distribution chart sea visible (ver #17).

**Severidad:** Baja — mejora de cohesión narrativa.

**Fuente N01:** `analysisN01.md` §4, §12.2

---

### Sección 6 — Playground (tab-play)

#### 25. La simulación del Playground no refleja los hallazgos léxicos de N01

**Líneas:** 979-989 (keywords hardcodeadas: `good|great|love|excellent|perfect|happy|recommend`)

**Problema:** La simulación de `runInference()` usa 12 keywords naïve que no reflejan lo que N01 realmente encontró sobre el vocabulario distintivo por clase. N01 identificó:
- **Palabras típicas de 1★**: `waste`, `broke`, `return`, `refund`, `disappointed`, `junk`, `trash`
- **Palabras típicas de 5★**: mucho más diversas y específicas al producto

El set actual de keywords positivas (`good`, `great`, `love`, `excellent`, `perfect`, `happy`, `recommend`) es genérico. Es peor aún: `recommend` no es inherentemente positivo — podés decir *"I don't recommend"*.

**Fix:** Enriquecer las keywords con los hallazgos reales de N01:

```js
negWords: /(waste|broke|return|refund|junk|trash|terrible|horrible|disappoint|useless|defective|damaged)/
posWords: /(perfect|excellent|amazing|love|worth|favorite|impressed|recommend|best|glad|pleased)/
```

Y agregar un caso especial: si el texto tiene palabras de AMBAS listas (ej: *"Love the product but arrived broken"*), clasificar como Neutral — esto simula el comportamiento real de RoBERTa con ironía/contraste.

**Severidad:** Baja — la simulación es temporal hasta HF deployment, pero mientras tanto debería ser fiel a los datos.

**Fuente N01:** `analysisN01.md` §7, cell `cell-extreme-words`

---

### Sección 7 — Documentation (tab-docs)

#### 26. Las cards de Documentation son genéricas — faltan las decisiones arquitectónicas reales de N01

**Líneas:** 671-757 (4 cards: Foundation, Methodology, Advanced, Enterprise)

**Problema:** Las cards de la Sección 7 son vagas. Por ejemplo, "Data Loading: Raw JSONL streaming from UCSD to achieve 0 GB disk cache" es correcto pero no explica **por qué** se eligió sobre la alternativa (HuggingFace `datasets`).

N01 documenta 6 decisiones arquitectónicas con tabla de tradeoffs (`analysisN01.md` §13):

| Decisión | Alternativa descartada | Razón |
|---|---|---|
| Stream gzip desde UCSD | HF `datasets` streaming | API inestable + cache duplicado |
| 33 categorías | 1-2 categorías | Diversidad léxica |
| Undersampling | SMOTE / pesos | Transparencia, sin datos sintéticos |
| Filtro `len < 10` | Sin filtro / `len < 20` | Compromiso limpieza vs pérdida |
| Verificación leakage por índices | Comparación strings | Sin falsos positivos |
| Arrow (HF Datasets) | CSV / Parquet | Streaming nativo en Trainer |

**Fix:** Expandir la card "Foundation Decisions" para que incluya una mini-tabla con las 3-4 decisiones más importantes y su justificación. Esto le da sustancia a la sección de Documentation en vez de ser un placeholder.

**Severidad:** Media — la sección Documentation debería ser el "source of truth" del proyecto, no un resumen vago.

**Fuente N01:** `analysisN01.md` §13

---

### General / Multi-sección

#### 27. Faltan los números absolutos en todo el dashboard — solo se muestran porcentajes

**Líneas:** 303-312 (chart %), 317-320 (Natural Imbalance card)

**Problema:** La web muestra `10.8%`, `7.6%`, `81.6%` y `213.9K`. Pero N01 tiene los números absolutos que cuentan una historia más impactante:
- Se descartaron **697,881 reseñas positivas** en el balanceo (87% del corpus positivo)
- Solo **71,315 reseñas neutras** dictaron el tamaño final
- 1★ son **63,617** reseñas vs 5★ son **655,474** (10× diferencia)

Los números absolutos son más memorables que los porcentajes para una presentación.

**Fix:** Agregar tooltips o anotaciones en el chart con los conteos absolutos. Ej: `81.6% (655,474 reviews)`. Y en la card "Natural Imbalance", cambiar el texto para incluir: *"We discarded 697,881 positive reviews to match the 71,315 neutral minority."*

**Severidad:** Baja — pero mejora el impacto narrativo.

**Fuente N01:** `analysisN01.md` §10

---

#### 28. El título del dashboard dice "v4.0 Production" pero los datos son de un prototipo educativo

**Líneas:** 209 (`v4.0 Production`), 266 (`Project v4.0`)

**Problema:** El badge dice "Production" y "v4.0". Esto es misleading — el proyecto es un bootcamp (Ironhack), los datos son un sample de 976K sobre 571M, y la inferencia es una simulación con regex. Para una presentación de clase, esto puede generar preguntas incómodas.

**Fix:** Cambiar el badge a algo más honesto y que igual suene profesional:

```html
<span>Project v1.0</span>  <!-- o "Research Pipeline v1.0" -->
```

O si se quiere mantener el espíritu aspiracional: `"v1.0 — Research Pipeline"`.

**Severidad:** Baja — pero evita un momento awkward en la defensa.

---

## 📊 Resumen de mejoras N01

| # | Sección web | Mejora | Impacto |
|---|---|---|---|
| 14 | Data & EDA | Pipeline visual 571M→976K→942K→214K | 🔴 Alto |
| 15 | Data & EDA | Distribución 1★–5★ (no solo 3 clases) | 🔴 Alto |
| 16 | Data & EDA | Reseñas neutras son las más largas (405 chars) | 🔴 Alto |
| 17 | Data & EDA | Word count: 98.5% cabe en 128 tokens | 🟠 Medio |
| 18 | Data & EDA | 8.4% duplicados — data quality card | 🟢 Bajo |
| 19 | Data & EDA | Top 5 categorías más negativas (no solo agregado) | 🟠 Medio |
| 20 | Data & EDA | Helpful votes: 1★ → 3.3 votes, 5★ → 1.3 | 🟢 Bajo |
| 21 | Data & EDA | Verified purchase: no-verificadas cluster en 4★ | 🟢 Bajo |
| 22 | Data & EDA | Vocabulario distintivo 1★ (31 words) vs 5★ (14,735) | 🟢 Bajo |
| 23 | Architecture | Arquitectura zero-disk streaming como card propia | 🟠 Medio |
| 24 | Architecture | Conectar max_length=128 con el hallazgo EDA de word count | 🟢 Bajo |
| 25 | Playground | Keywords de simulación basadas en vocabulario real de N01 | 🟢 Bajo |
| 26 | Documentation | Decisions table con tradeoffs reales de N01 §13 | 🟠 Medio |
| 27 | General | Números absolutos + tooltips (no solo %) | 🟢 Bajo |
| 28 | General | Badge "Production v4.0" → "Research Pipeline v1.0" | 🟢 Bajo |

**Total: 14 mejoras de contenido** (más 1 de honestidad de branding).

**Recomendación de orden de ataque:**
1. **#14 y #15** — el pipeline visual y la distribución 1★–5★ transforman la sección Data & EDA de "un chart y dos cards" a una narrativa de datos real.
2. **#16** — el hallazgo de que las reseñas neutras son las más largas es el insight más memorable del EDA y hoy está ausente.
3. **#19 y #23** — la tabla de categorías más negativas y la arquitectura zero-disk le dan profundidad técnica.
4. El resto (#17, #18, #20–22, #24–28) son polish que se pueden aplicar en batch.
