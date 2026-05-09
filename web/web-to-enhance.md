# Dashboard Enhancements — NLP Business Case

> Estado actualizado: 2026-05-09. Archivo: `web/raw-web-vs3.html`

## 🔲 PENDIENTES (To Do)

**Sección 1: Data & EDA**
- [x] **#18 Data Quality:** Mostrar % de duplicados (8.4%) y vacíos.
- [x] **#20 Helpful Votes:** Indicar que las reseñas 1★ reciben 2.5× más votos útiles que las 5★.
- [x] **#21 Verified Purchase:** Señalar anomalía de reseñas no verificadas agrupadas en 4★.
- [x] **#22 Vocabulario Extremo:** Destacar que 1★ usa 31 palabras distintivas vs 14K en 5★.

**Sección 2: Architecture**
- [x] **#24 Conexión EDA-Arq:** Vincular explícitamente el `max_length=128` con el hallazgo de word count del EDA.

**Sección 7: Documentation**
- [x] **#26 Matriz de Decisiones:** Reemplazar cards genéricas por la tabla real de tradeoffs de N01 (Streaming vs Disco, DistilBERT vs RoBERTa, etc.).

---

## ✅ COMPLETADOS (Done)

**UX & UI**
- [x] #1: Fix bug de Chart.js en tabs ocultos.
- [x] #3: Añadir ARIA labels y accesibilidad básica.
- [x] #6, #8, #9, #10, #11, #13: Ajustes menores de UI (scrollbar, meta, headings).
- [x] #27: Mostrar números absolutos en tooltips.
- [x] #28: Branding "Research Pipeline v1.0".
- [x] UX: Animación interactiva en `eda-band` y foco narrativo.

**Contenido N01 Integrado**
- [x] #14: Pipeline visual de reducción (571M → 214K).
- [x] #15: Gráfico de distribución de 5 estrellas.
- [x] #16: Insight "Paradoxa Neutral" (reseñas neutras son más largas).
- [x] #17: Carrusel validando `max_length=128`.
- [x] #19: Foco en top 5 categorías negativas.
- [x] #4, #25: Simulación Playground con Lexical Fingerprints reales y lógica del Neutral Paradox.

---

## 🚫 POSTERGADOS / DESCARTADOS (Deferred/Skipped)

- [ ] #2: Responsive / Mobile-First (Descartado, prioridad presentación).
- [ ] #5, #12: Generación estática Tailwind y fallbacks CDN.
- [ ] #7: Encapsulación de variables globales JS.
- [ ] #23: Card dedicada a Zero-Disk Streaming.
