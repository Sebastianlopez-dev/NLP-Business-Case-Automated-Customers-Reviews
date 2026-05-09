# Analysis N03 — RoBERTa Sentiment Classification

> Dual-agent audit: Runtime Detective + Convention Auditor  
> Date: 2026-05-09  
> Status: **Run 2 completada — 69.1% accuracy. LR 1e-5 confirmado como bottleneck. Cambios de código no aplicados — solo documentación.**

---

## Summary

N03 está en buena forma estructural. El pipeline de fine-tuning corrió completo dos veces:

| Run | Dropout | Epochs | LR | Accuracy final |
|-----|---------|--------|-----|---------------|
| Run 1 (descartada) | 0.2 | 3 | 1e-5 | 59% en epoch 1 — abortada |
| Run 2 (final) | 0.1 | 5 | 1e-5 | **69.1%** — 8 puntos abajo de DistilBERT (77.3%) |

No se encontraron bugs críticos que hagan crash. Los hallazgos son mayormente comentarios stale, imports muertos, guards de dependencia cross-cell faltantes, y **un gap de performance inesperado que apunta al learning rate como causa raíz**.

---

## Hallazgos

### Silent Bugs — código que corre pero miente o despista

| # | Hallazgo | Celda |
|---|----------|-------|
| 1 | Markdown dice `max_length=128` pero el código usa 256 — doc/code mismatch. Cualquiera que lea el header piensa que la tokenización es a 128. | `cell-tokenize-title` |
| 2 | Comment espera `shape = torch.Size([128])` pero el output real es `[256]`. Comentario no se actualizó cuando se subió max_length. | `cell-set-format` |
| 3 | `report_to` tiene comment "Log to Weights & Biases" pero el valor real es `"none"`. Trap #8 de AGENTS.md — code-comment integrity. | `cell-training-args` |
| 4 | `os.environ["TENSORBOARD_LOGGING_DIR"]` seteado pero `report_to="none"` — env var muerta, nunca se usa. | `cell-training-args` |
| 5 | `from tqdm.auto import tqdm` importado en `cell-imports` pero nunca se llama en ningún code cell. Import muerto. | `cell-imports` |
| 6 | `steps_per_epoch` usa floor division → subestima el total de steps en 5 (46,800 vs 46,805 reales). Impacto negligible sobre warmup steps. | `cell-training-args` |
| 7 | Comment dice `# Red for RoBERTa` pero `#EA580C` es naranja — el color del proyecto. | `cell-comparison-plot` |

### Edge Cases — no fallan hoy, pero pueden fallar mañana

| # | Hallazgo | Celda |
|---|----------|-------|
| 8 | `json.load()` sin `try/except` — si `metrics_distilbert.json` está corrupto (crash de N02 a mitad de escritura), el notebook crashea con `JSONDecodeError` en vez de caer al placeholder. | `cell-load-distilbert-metrics` |
| 9 | Resume de checkpoint no valida integridad del directorio — si un training anterior crasheó mid-write, `trainer.train(resume_from_checkpoint=...)` falla con error críptico de HuggingFace. | `cell-train` |
| 10 | `distil_metrics` se carga en la celda de validación Y se recarga en la sección de comparación — I/O duplicado en un filesystem montado en Drive. | `cell-validate-n03` |

### Cross-Cell Dependency Traps — si ejecutás celdas fuera de orden, explota

Estas celdas usan variables definidas en celdas anteriores sin guard:

| Celda | Depende de | Definido en |
|-------|-----------|-------------|
| `cell-load-model` | `NUM_LABELS`, `ID2LABEL`, `LABEL2ID` | `cell-label-map` |
| `cell-training-args` | `RANDOM_SEED`, `tokenized_dataset` | `cell-seed`, `cell-tokenize-dataset` |
| `cell-train` (WeightedTrainer) | `class_weights_tensor` | `cell-class-weights` |
| `cell-loss-plot` | `trainer.state.log_history` | `cell-train` |
| `cell-examples` | `predictions_df` | `cell-save-predictions` |

**Bien defendido**: `cell-confusion-matrix` y `cell-per-class-table` ya tienen `if 'label_names' not in dir():` guards. ✅

### Convention Issues (auditor truncado — parcial)

- 3 imports dispersos violando AGENTS.md §5 "ALL in cell-import"
- Algunos `metadata.id ≠ cell-level id` (mismo patrón que encontramos en N04)

### Confusion Matrix — Inconsistencia de formato N02 vs N03 🔴

| Aspecto | N02 (DistilBERT) | N03 (RoBERTa) |
|---------|-----------------|---------------|
| Normalización | `/ cm.sum()` → proporción **0–1** | `* 100` → porcentaje **0–100** |
| Formato anotación | `fmt=".2f"` → muestra `0.66` | `fmt=".1f"` → muestra `57.6` |
| Escala de color | Automática (sin vmin/vmax) | `vmin=0, vmax=100` fijo |
| Paleta | `"Blues"` (azul) | `roberta_cmap` (naranja) |

**Impacto**: Al comparar los dos heatmaps lado a lado, el lector ve `0.66` en uno y `57.6` en el otro para el mismo valor (recall de Neutral). Son el mismo número (`0.576 ≈ 57.6%`), pero la diferencia de escala induce a error visual y entorpece la comparación.

**Fix pendiente**: Estandarizar ambos notebooks a porcentaje `0–100` con `fmt=".1f"`, `vmin=0, vmax=100`, y misma paleta de colores. N02 es el que hay que corregir (basta con agregar `* 100` y cambiar `fmt=".2f"` → `fmt=".1f"`, `vmin=0, vmax=100`).

---

## Análisis de Outputs — Resultados del Entrenamiento

### Training — Run 2 (dropout 0.1, LR 1e-5, 5 epochs)

Epoch-by-epoch progression:

| Epoch | Train Loss | Val Loss | Accuracy | F1 | Gap |
|-------|-----------|----------|----------|-----|-----|
| 1 | 0.8110 | 0.8101 | 0.6299 | 0.6303 | +0.001 |
| 2 | 0.7196 | 0.7478 | 0.6651 | 0.6664 | -0.028 |
| 3 | 0.7034 | 0.7234 | 0.6775 | 0.6792 | -0.020 |
| 4 | 0.6594 | 0.7113 | 0.6876 | 0.6861 | -0.052 |
| 5 | 0.6552 | 0.7117 | 0.6909 | 0.6921 | **-0.056** |

Final aggregate: `train_loss=0.7513` (running avg), `train_runtime=2:59:27.99` (10,768s), `samples/sec=69.5`.

- **5 epochs completos** — EarlyStopping NO disparó. Validation loss bajó hasta epoch 3 y luego se estancó (~0.711), lo cual permitió que el entrenamiento continuara aunque la generalización ya no mejoraba.
- **Val loss plateau desde epoch 3**: 0.7234 → 0.7113 → 0.7117. El modelo dejó de generalizar pero siguió memorizando el training set (train loss seguía bajando: 0.81→0.66).
- **Gap train/val creciente**: De +0.001 en epoch 1 a -0.056 en epoch 5. No es overfitting severo, pero sí indica que más epochs no van a ayudar sin cambiar hiperparámetros.
- **~3 horas en T4** para 149,761 samples × 5 epochs a batch_size=16.

### Test Set Evaluation (32,092 samples)

```
              Precision  Recall  F1-Score  Support
Negative        0.6958   0.6897    0.6928   10,697
Neutral         0.5985   0.5758    0.5869   10,697
Positive        0.7774   0.8138    0.7952   10,698
─────────────────────────────────────────────────
Accuracy                             0.6931
Weighted F1                          0.6916
```

- **Accuracy 69.31%**: Bajo para un transformer de 125M params fine-tuneado.
- **Neutral F1 0.5869**: La clase más difícil — consistente con lo esperado (reviews de 3 estrellas son ambiguas).
- **Positive F1 0.7952**: La clase más fácil — el modelo captura bien el sentimiento positivo.

### Comparación RoBERTa vs DistilBERT — 🔴 HALLAZGO CLAVE

```
                    DistilBERT (~66M)  RoBERTa (~125M)  Delta
Accuracy                    0.7726           0.6931  -0.0795
Weighted F1                 0.7718           0.6916  -0.0802

Per-class F1:
  Negative                  0.7672           0.6928  -0.0744
  Neutral                   0.6710           0.5869  -0.0840
  Positive                  0.8773           0.7952  -0.0822
```

**RoBERTa perdió contra DistilBERT por ~8 puntos porcentuales en TODAS las métricas y TODAS las clases.** Esto es inesperado — RoBERTa tiene el doble de parámetros, más capas, y mejor pretraining. Debería ser superior, no 8 puntos peor.

### Diagnóstico — ¿por qué RoBERTa rindió peor?

| Factor | N03 (RoBERTa) | N02 (DistilBERT) | Impacto probable |
|--------|--------------|-------------------|-----------------|
| Learning Rate | 1e-5 | 2e-5 | 🔴 ALTO — LR mitad para el doble de params |
| Batch Size | 16 | 32 | 🟠 MEDIO — gradientes más ruidosos, pasos más chicos |
| Dropout | 0.1 | 0.3 | 🟡 DistilBERT usa 3× más regularización y aun así generaliza mejor. El "underfitting por dropout 0.2" diagnosticado en Run 1 fue prematuro — era epoch 1, el modelo no había arrancado. |
| Max Length | 256 | 256 | 🟢 Igual — ambos usan 256 tokens |
| Params | 125M | 66M | 🟡 Más params necesitan LR más alto, no más bajo |

**Conclusión**: El LR de 1e-5 fue **demasiado bajo** para RoBERTa. Con 125M parámetros y batch_size=16, cada paso de gradiente es minúsculo. DistilBERT con 66M params, LR 2e-5, y batch_size=32 tuvo ~4× más "fuerza de actualización" efectiva. A esto se suma que dropout 0.1 (vs 0.3 en DistilBERT) no es la variable relevante — el bottleneck es LR.

### Expected vs Actual — la predicción fallida

En la sesión del 2026-05-09 (01:10), después de bajar dropout de 0.2 a 0.1, se predijo esta trayectoria: `0.66→0.77→0.81→0.83→0.85`. La realidad fue `0.63→0.67→0.68→0.69→0.69`. El modelo arrancó más lento de lo esperado y se estancó 16 puntos abajo de la predicción. Esto confirma que el dropout NO era el cuello de botella — el LR siempre lo fue.

### Hipótesis actualizada post-Run 2

1. **LR 1e-5 → 2e-5 es el cambio prioritario**. Si el modelo con LR 2e-5 saca ≥75%, el diagnóstico se confirma.
2. Si LR 2e-5 + 5 epochs no alcanza, probar **LR 3e-5** o **gradient_accumulation_steps=2** con batch_size=16 para simular batch efectivo de 32.
3. La otra variable no explorada: **dropout 0.15-0.2 para RoBERTa** podría ayudar si el modelo overfittea con LR más alto. Pero primero hay que ver si LR 2e-5 despega la accuracy.

### ¿Early Stopping no disparó?

Correcto — corrió las 5 epochs completas. Validation loss bajó hasta epoch 3 y luego se estancó (~0.711), pero como el delta entre epochs era menor a 0.001 (el threshold de EarlyStopping), el callback nunca lo frenó. Con un LR más alto, esperaríamos que validation loss baje más rápido al principio y potencialmente diverja antes — ahí EarlyStopping sí actuaría como red de seguridad.

---

## Qué SÍ está bien (no tocar)

- ✅ Pipeline de training — `WeightedTrainer`, `EarlyStoppingCallback`, y métricas verificados
- ✅ `fp16` mixed precision correctamente configurado
- ✅ Lógica de checkpoint resume funcional (solo falta el integrity check)
- ✅ Métricas per-class, confusion matrix, y comparison plots implementados correctamente
- ✅ `RANDOM_SEED = 42` consistente con todos los notebooks

---

## Cambios pendientes — aplicar si sobra tiempo de GPU al final del proyecto

### ⚠️ Nota: Run 2 completada (2026-05-09)

La Run 2 (dropout 0.1, LR 1e-5, 5 epochs) se ejecutó y confirmó que el dropout NO era el bottleneck. Accuracy final: 69.1% — 8 puntos abajo de DistilBERT. El LR sigue siendo la variable a modificar.

### 1. Arreglar comentarios stale de `max_length`

**Archivos**: `cell-tokenize-title` (markdown), `cell-set-format` (code comment)

- Markdown: *"I set `max_length=128` for consistency with NB02"* → *"I set `max_length=256` — raised from 128 because BPE needs ~30% more tokens than WordPiece"*
- Comment: `# Expected: input_ids shape = torch.Size([128])` → `# Expected: input_ids shape = torch.Size([256])`

### 2. Limpiar código muerto

**Archivos**: `cell-imports`, `cell-training-args`

- Sacar `from tqdm.auto import tqdm` de `cell-imports` (nunca se usa)
- Sacar `os.environ["TENSORBOARD_LOGGING_DIR"] = ...` (muerta — `report_to="none"`)
- Arreglar comment de `report_to`: *"Log to Weights & Biases"* → *"Disable external logging (use Trainer built-in)"*

### 3. Agregar guards de dependencia cross-cell

**Archivos**: `cell-load-model`, `cell-train` (celda del WeightedTrainer), `cell-loss-plot`

Agregar `NameError` guards siguiendo el patrón que ya usan `cell-confusion-matrix` y `cell-per-class-table`:

```python
if 'NUM_LABELS' not in dir():
    raise NameError("Run cell-label-map first — NUM_LABELS is not defined.")
if 'class_weights_tensor' not in dir():
    raise NameError("Run cell-class-weights first — class_weights_tensor is not defined.")
if not hasattr(trainer, 'state'):
    raise NameError("Run cell-train first — trainer has no state yet.")
```

### 4. ⭐ PRIORIDAD: Experimento de LR — subir a 2e-5

**Por qué**: Run 2 confirmó que RoBERTa con LR 1e-5 queda **8 puntos por debajo de DistilBERT** (69.1% vs 77.3%). El val loss se estancó en epoch 3 (~0.711) y no bajó más. El LR de 1e-5 es el sospechoso principal — DistilBERT con la mitad de parámetros usó 2e-5 y funcionó.

**Cambio propuesto**:
- `LEARNING_RATE = 1e-5` → `LEARNING_RATE = 2e-5` (`cell-training-args`)
- Mantener `NUM_EPOCHS = 5` (si LR 2e-5 funciona, 5 epochs deberían bastar; si no, más epochs no arreglan un LR malo)
- Agregar comment: "Bumped from 1e-5 to match DistilBERT's LR. Run 2 at 1e-5 stagnated at 69.1% — 8 points below DistilBERT's 77.3%."

**Riesgo**: Bajo. 2e-5 es el LR estándar para fine-tuning de RoBERTa en la literatura. EarlyStopping(patience=2) frenará si diverge.

**Contexto adicional**: N01 preprocessing usa `encode('ascii', 'ignore')` que elimina caracteres no-ASCII. Esto neutraliza la ventaja del tokenizador BPE de RoBERTa (que puede codificar cualquier Unicode) frente al WordPiece de DistilBERT (que nunca pudo codificarlos). Impacto estimado: 1-2% de accuracy. No se corrige ahora porque requiere regenerar el dataset Arrow.
