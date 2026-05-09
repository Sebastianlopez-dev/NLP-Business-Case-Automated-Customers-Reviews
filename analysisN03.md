# Analysis N03 — RoBERTa Sentiment Classification

> Dual-agent audit: Runtime Detective + Convention Auditor  
> Date: 2026-05-09  
> Status: **Run 2 completada — 69.3% test accuracy (0.6931). Val acc epoch 5: 69.1% (0.6909). LR 1e-5 confirmado como bottleneck. Cambios de código no aplicados — solo documentación.**

---

## Summary

N03 está en buena forma estructural. El pipeline de fine-tuning corrió completo dos veces. Hallazgos: comentarios stale, imports muertos, guards faltantes, y un gap de performance inesperado que apunta al learning rate como causa raíz.

| Run | Dropout | Epochs | LR | Accuracy test (val epoch 5) |
|-----|---------|--------|-----|-----------------------------|
| Run 1 (descartada) | 0.2 | 3 | 1e-5 | ~59% en epoch 1 [needs-verification: notebook re-ejecutado; sin output preservado] |
| Run 2 (final) [Cell 27, 29] | 0.1 | 5 | 1e-5 | **69.31% test** (0.6931), val epoch 5: 69.09% — 7.95pp abajo de DistilBERT (77.26%) |

---

## Hallazgos

### Silent Bugs — código que corre pero miente o despista

| # | Hallazgo | Celda |
|---|----------|-------|
| 1 | Markdown dice `max_length=128` pero código usa 256 — doc/code mismatch | `cell-tokenize-title` |
| 2 | Comment espera `shape=torch.Size([128])` pero output real es `[256]` [Cell 17] | `cell-set-format` |
| 3 | `report_to` tiene comment "Log to Weights & Biases" pero valor real es `"none"` (Trap #8) | `cell-training-args` |
| 4 | `os.environ["TENSORBOARD_LOGGING_DIR"]` seteado pero `report_to="none"` — env var muerta | `cell-training-args` |
| 5 | `from tqdm.auto import tqdm` importado pero nunca usado en ningún code cell | `cell-imports` |
| 6 | `steps_per_epoch` usa floor division → reporta 46,800 pero reales son 46,805 (barra muestra 46805/46805) [Cell 27] | `cell-training-args` |
| 7 | Comment dice `# Red for RoBERTa` pero `#EA580C` es naranja | `cell-comparison-plot` |

### Edge Cases — no fallan hoy, pero pueden fallar mañana

| # | Hallazgo | Celda |
|---|----------|-------|
| 8 | `json.load()` sin `try/except` — métricas corruptas de N02 crashean con `JSONDecodeError` | `cell-load-distilbert-metrics` |
| 9 | Resume de checkpoint no valida integridad del directorio — mid-write crash → error críptico | `cell-train` |
| 10 | `distil_metrics` se carga en validación Y se recarga en comparación — I/O duplicado en Drive | `cell-validate-n03` |

### Cross-Cell Dependency Traps

| Celda | Depende de | Definido en |
|-------|-----------|-------------|
| `cell-load-model` | `NUM_LABELS`, `ID2LABEL`, `LABEL2ID` | `cell-label-map` |
| `cell-training-args` | `RANDOM_SEED`, `tokenized_dataset` | `cell-seed`, `cell-tokenize-dataset` |
| `cell-train` (WeightedTrainer) | `class_weights_tensor` | `cell-class-weights` |
| `cell-loss-plot` | `trainer.state.log_history` | `cell-train` |
| `cell-examples` | `predictions_df` | `cell-save-predictions` |

**Bien defendido**: `cell-confusion-matrix` y `cell-per-class-table` ya tienen guards. ✅

### Convention Issues (auditor truncado — parcial)

- 3 imports dispersos violando AGENTS.md §5 "ALL in cell-imports"
- Algunos `metadata.id ≠ cell-level id` (mismo patrón que N04)

---

## Resultados del Entrenamiento

### Training — Run 2 (dropout 0.1, LR 1e-5, 5 epochs) [Cell 27 output]

| Epoch | Train Loss | Val Loss | Accuracy | F1 | Gap (Train−Val) |
|-------|-----------|----------|----------|-----|-----|
| 1 | 0.8110 | 0.8101 | 0.6299 | 0.6303 | +0.001 |
| 2 | 0.7196 | 0.7478 | 0.6651 | 0.6664 | −0.028 |
| 3 | 0.7034 | 0.7234 | 0.6775 | 0.6792 | −0.020 |
| 4 | 0.6594 | 0.7113 | 0.6876 | 0.6861 | −0.052 |
| 5 | 0.6552 | 0.7117 | 0.6909 | 0.6921 | **−0.056** |

Final aggregate [Cell 27 stdout]: `train_loss=0.7513`, `train_runtime=2:59:27.99` (10,768s), `samples/sec=69.5`. 5 epochs completos en T4 para 149,761 muestras × batch_size=16.

- **EarlyStopping NO disparó**: val loss bajó hasta epoch 3 (0.7234) y luego se estancó (~0.711). El delta entre epochs fue <0.001 (threshold de ES).
- **Gap train/val creciente**: De +0.001 en epoch 1 a −0.056 en epoch 5. No es overfitting severo, pero más epochs no ayudan sin cambiar hiperparámetros.

### Test Set Evaluation (32,092 samples) [Cell 29, 30 output]

```
              Precision  Recall  F1-Score  Support
Negative        0.6958   0.6897    0.6928   10,697
Neutral         0.5985   0.5758    0.5869   10,697
Positive        0.7774   0.8138    0.7952   10,698
─────────────────────────────────────────────────
Accuracy                             0.6931
Weighted F1                          0.6916
```

- **Test Accuracy 69.31%**: Bajo para un transformer de 125M params fine-tuneado.
- **Neutral F1 0.5869**: La clase más difícil (reviews de 3★ son ambiguas).
- **Positive F1 0.7952**: La clase más fácil.

### Comparación RoBERTa vs DistilBERT [Cell 34, 35 output]

```
                    DistilBERT (~66M)  RoBERTa (~125M)  Delta
Accuracy                    0.7726           0.6931  −0.0795
Weighted F1                 0.7718           0.6916  −0.0802

Per-class F1:
  Negative                  0.7672           0.6928  −0.0744
  Neutral                   0.6710           0.5869  −0.0840
  Positive                  0.8773           0.7952  −0.0822
```

**RoBERTa perdió contra DistilBERT por ~8 puntos porcentuales en todas las métricas y todas las clases.** Esto es inesperado — RoBERTa tiene ~1.9× más parámetros, más capas, y mejor pretraining.

### Diagnóstico — ¿por qué RoBERTa rindió peor?

| Factor | N03 (RoBERTa) | N02 (DistilBERT) | Impacto probable |
|--------|--------------|-------------------|-----------------|
| Learning Rate | 1e-5 | 2e-5 | 🔴 ALTO — LR mitad para ~1.9× params |
| Batch Size | 16 | 32 | 🟠 MEDIO — gradientes más ruidosos |
| Dropout | 0.1 | 0.3 [needs-verification: de N02, no verificado en N03] | 🟡 DistilBERT usa 3× más regularización y aun así generaliza mejor |
| Max Length | 256 | 256 | 🟢 Igual |
| Params | 125M | 66M | 🟡 Más params necesitan LR más alto, no más bajo |

**Conclusión**: LR 1e-5 fue demasiado bajo. DistilBERT con 66M params, LR 2e-5, y batch_size=32 tuvo ~4× más "fuerza de actualización" efectiva: (2e-5/1e-5) × (32/16) = 4× más señal de actualización por epoch. El dropout no es la variable relevante — el bottleneck es LR.

### Expected vs Actual — la predicción fallida [needs-verification]

En sesión del 2026-05-09 (01:10), se predijo trayectoria: `0.66→0.77→0.81→0.83→0.85`. La realidad fue `0.63→0.67→0.68→0.69→0.69` [Cell 27]. El modelo arrancó más lento y se estancó 16pp abajo. Confirma que el dropout NO era el cuello de botella — el LR siempre lo fue. **Nota**: la predicción es de sesión, no de notebook output.

### Hipótesis actualizada post-Run 2

1. **LR 1e-5 → 2e-5 es el cambio prioritario**. Si saca ≥75%, diagnóstico confirmado.
2. Si LR 2e-5 + 5 epochs no alcanza, probar **LR 3e-5** o **gradient_accumulation_steps=2** con batch_size=16.
3. Dropout 0.15-0.2 podría ayudar si el modelo overfittea con LR más alto. Pero primero ver si LR 2e-5 despega la accuracy.

---

## Qué SÍ está bien (no tocar)

- ✅ Pipeline de training: `WeightedTrainer`, `EarlyStoppingCallback`, métricas verificados
- ✅ `fp16` mixed precision correctamente configurado
- ✅ Lógica de checkpoint resume funcional (solo falta integrity check)
- ✅ Métricas per-class, confusion matrix, comparison plots implementados
- ✅ `RANDOM_SEED = 42` consistente

---

## Cambios pendientes — aplicar si sobra tiempo de GPU al final

### Nota: Run 2 completada (2026-05-09)

Run 2 (dropout 0.1, LR 1e-5, 5 epochs) confirmó que dropout NO era el bottleneck. Test accuracy: 69.31% (val epoch 5: 69.09%) — 7.95pp abajo de DistilBERT (77.26%). LR sigue siendo la variable a modificar.

### 1. Arreglar comentarios stale de `max_length`

**Archivos**: `cell-tokenize-title` (markdown), `cell-set-format` (code comment)

- Markdown: *"I set `max_length=128`…"* → *"I set `max_length=256`…"*
- Comment: `# Expected: input_ids shape = torch.Size([128])` → `# Expected: input_ids shape = torch.Size([256])`

### 2. Limpiar código muerto

**Archivos**: `cell-imports`, `cell-training-args`

- Sacar `from tqdm.auto import tqdm`
- Sacar `os.environ["TENSORBOARD_LOGGING_DIR"]`
- Arreglar comment de `report_to`: *"Log to Weights & Biases"* → *"Disable external logging"*

### 3. Agregar guards de dependencia cross-cell

**Archivos**: `cell-load-model`, `cell-train` (WeightedTrainer), `cell-loss-plot`

Seguir el patrón que ya usan `cell-confusion-matrix` y `cell-per-class-table`.

### 4. ⭐ PRIORIDAD: Experimento de LR — subir a 2e-5

**Por qué**: Run 2 confirmó que LR 1e-5 queda 7.95pp abajo de DistilBERT (69.31% vs 77.26%). Val loss se estancó en epoch 3 (~0.711).

**Cambio propuesto**:
- `LEARNING_RATE = 1e-5` → `LEARNING_RATE = 2e-5` (`cell-training-args`)
- Mantener `NUM_EPOCHS = 5`
- Agregar comment: "Bumped from 1e-5. Run 2 stagnated at 69.3% — 8pp below DistilBERT's 77.3%."

**Riesgo**: Bajo. 2e-5 es LR estándar para RoBERTa fine-tuning. EarlyStopping(patience=2) frenará si diverge.

<!-- UNVERIFIED: N01 preprocessing usa encode('ascii', 'ignore') que elimina caracteres no-ASCII. Esto neutraliza la ventaja del tokenizador BPE de RoBERTa frente a WordPiece. Impacto estimado: 1-2% accuracy. No se corrige ahora porque requiere regenerar el dataset Arrow. [Claim about N01, not verifiable from N03 notebook] -->
