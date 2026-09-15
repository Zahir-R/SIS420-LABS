# Cheat Sheet — Parcial IA1: Redes Neuronales

Resumen de todo lo visto hasta **optimización y regularización**, para repasar el lunes antes del parcial. Acompañar con los cuadernillos de `LABCLASS`.

---

## 1. Arquitectura (Perceptrón → MLP)

- **Perceptrón**: `y = σ(w·x + b)`. Sólo separa datos lineales.
- **MLP** = perceptrones apilados en capas. Las capas ocultas usan activación **no-lineal** (`ReLU`).
- ⚡ ¿Por qué no-lineal? Componer funciones lineales da otra función lineal → el MLP valdría lo mismo que un único perceptrón.
- Salida según el problema (ver tabla abajo). Backprop = descenso por el gradiente aplicado a todas las capas vía regla de la cadena.

## 2. ¿Qué salida / pérdida / métrica según el problema?

| Problema | Salida del modelo | Función de pérdida | Métrica |
|---|---|---|---|
| Regresión | 1 neurona, **sin** activación | **MSE** | R², RMSE, MAE |
| Clasificación binaria | 1 neurona + **Sigmoid** (`D_out=1`) | **BCE** (`BCELoss`) con `y_pred>0.5` | accuracy, precision, recall, F1 |
| Clasificación multiclase | `D_out = num_clases`, **sin softmax** | **CrossEntropy** (`CrossEntropyLoss`) | accuracy |

> `CrossEntropyLoss` ya aplica softmax internamente. Para obtener probabilidades/etiquetas en multiclase: `argmax(softmax(logits), axis=1)`.

## 3. El target `y`

- El problema lo define **la variable objetivo**, no el dataset: "¿qué quiero predecir?" → `y`, el resto → `X`.
- **Regresión**: `y` numérica y continua (precio, temperatura, carga…) — muchos valores distintos.
- **Clasificación**: `y` discreta / etiqueta — `df[col].nunique()` chico (2 = binaria, N = multiclase).
- `y` siempre se descarta de `X`.

## 4. Receta de entrenamiento (`07 receta_entrenamiento`)

1. **EDA**: shape, dtype, mean/std, distribución de clases (balance).
2. **Normalizar `X`**: `X/255` (imágenes) ó z-score `(X-μ)/σ` (μ, σ sólo del train).
3. **Split train/val/test**; en clasificación `stratify=y`.
4. Clase `Dataset` (`__init__`, `__len__`, `__getitem__`) + `DataLoader(batch, shuffle=True)`.
5. **Modelo**: `nn.Sequential(Linear(D_in,H), ReLU, …, Linear(H,D_out))` `.to(device)`.
6. **Valida antes de entrenar**:
   - dimensiones (`randn((64, D_in))` → `(64, D_out)`),
   - **fit de 1 muestra** (debe bajar el loss),
   - **fit de 1 batch**.
   Si no aprende 1 muestra, hay un bug.
7. **Tune rápido en un subconjunto** (probar lr y batch) antes de entrenar con todo.
8. **Entrenar final** con scheduler + early stopping, guardando el mejor `state_dict`.
9. Evaluar en **test** (sólo al final).

## 5. Bucle de entrenamiento (know it by heart)

```python
model.train()
for x_b, y_b in dataloader['train']:
    loss = criterion(model(x_b), y_b)      # forward
    optimizer.zero_grad()                  # limpiar gradientes
    loss.backward()                        # backprop automático
    optimizer.step()                       # actualizar pesos
# validación sin gradientes:
model.eval()
with torch.no_grad():
    ...  # métricas sobre dataloader['val']
```

- `torch.save(model.state_dict(), 'ckpt.pt')` y `model.load_state_dict(torch.load('ckpt.pt', weights_only=True))`.

## 6. Overfitting (sobreajuste)

- **Señal en las curvas**: loss/acc de *train* sigue mejorando mientras **val** se estanca o **empeora** y se aleja.
- Reducir overfitting ≠ mejorar métricas: es reducir la brecha train–val.

## 7. Regularización

| Técnica | Idea | En PyTorch |
|---|---|---|
| **L2 / weight decay** | penaliza pesos grandes | `optimizer(..., weight_decay=0.01)` (típico 0.001–0.01) |
| **Dropout** | apaga neuronas al azar en train | `nn.Dropout(p=0.2)` |
| **Early stopping** | guardar mejor modelo por val y parar si no mejora | a mano en el `fit` (patience) |
| **BatchNorm** | normaliza activaciones (acelera) | `nn.BatchNorm1d(H)` |
| **Más datos / data augmentation** | el mejor antídoto; transformaciones aleatorias | `albumentations` |
| **Batch size / menor capacidad** | menos parámetros → menos memorización | reducir `H` |

> ⚡ `model.train()` activa Dropout/BatchNorm; `model.eval()` los desactiva. **Olvidarlo es el error clásico del examen.**

## 8. Optimizadores

- **SGD**: `w = w - lr·grad`. Simple, lento.
- **SGD + momentum**: acumula "inercia" de gradientes previos → más rápido.
- **RMSProp**: paso adaptativo por parámetro (divide por la media de gradientes²).
- **Adam**: momentum + RMSProp juntos. **Elección por defecto: `Adam(lr=0.001)`**.
- **Batch size** típico: 16 / 32 / 64 / 128 (lo más grande que quepa, ajustando lr).

## 9. Learning Rate Scheduling

- Variar el `lr` durante el entrenamiento.
- **StepLR**: `StepLR(optimizer, step_size=10, gamma=0.1)` → multiplica lr por 0.1 cada 10 epochs.
- **CyclicLR**: `CyclicLR(optimizer, base_lr=..., max_lr=..., step_size_up=..., step_size_down=...)`.

## 10. Configuración inicial recomendada (valores que casi siempre funcionan)

- Optimizador: **Adam `lr=0.001`**
- Batch size: **32–64**
- Capas ocultas: ReLU; **1–2 capas** de ~64–128 para datos tabulares, 100–200 para imágenes aplanadas
- Regularización: **early stopping siempre** + dropout/weight decay si hay overfitting
- Normalización: `X/255` o z-score

## 11. Lo que hay que saber "de memoria" antes del examen

1. Identificar si un dataset es regresión o clasificación y cuál columna es `y`.
2. Armar un `nn.Sequential` correcto (D_in, H, D_out y activación final según el problema).
3. Escribir el bucle forward/loss/backward/step y la validación con `no_grad`.
4. Escribir una clase `Dataset` + usar `DataLoader`.
5. Detectar overfitting en las curvas y proponer una técnica de regularización.
6. Explicar diferencias entre SGD, momentum, RMSProp y Adam; por qué usar Adam.
7. Diferenciar `train()` de `eval()`, y qué es un scheduler.
8. Diagnosticar si un modelo "no aprendió": accuracy vs baseline, AUC, matriz de confusión con filas/columnas en 0.

## 12. Diagnóstico: skew, outliers, clases y soluciones triviales

> Caso real (proyecto NEO): una red daba **accuracy 0.868 = baseline** y **AUC 0.26** (< 0.5 = peor que azar). No era falta de datos: era skew extremo (`diam` skew 34) + desbalance (13% positivos) → la red "ganaba" prediciendo siempre la clase mayoritaria.

### A. ANTES de entrenar — mirar la materia prima

```python
df.describe().T                        # rangos por columna: si max/min es 10^7, escala demoledora
df.skew(numeric_only=True).round(2)    # skew > 2 → transformar con log1p ANTES del z-score
df.hist(bins=50, figsize=(12, 8))      # colas largas → gorditas a la derecha
df[df.columns].boxplot(figsize=(12, 4))# puntos sueltos más allá del bigote = outliers
df['y'].value_counts(normalize=True)   # desbalance de clases
```

- ⚡ **El outlier puede ser la señal**: en NEO los diámetros enormes SON la clase peligrosa → se **transforman** (`np.log1p`), no se borran.
- **Borrar/dropear** solo si es un **error de medición** (valor imposible), no un dato extremo real.

### B. DESPUÉS de entrenar — diagnosticar el resultado

```python
p = model(Xtest).detach().cpu().numpy().ravel()
pred = (p > 0.5).astype(int)

print("baseline (predecir la clase mayoritaria):", round(1 - y_test.mean(), 4))
print("accuracy:", accuracy_score(y_test, pred))
print("AUC:     ", roc_auc_score(y_test, p))
print(confusion_matrix(y_test, pred))
pd.Series(p).hist(bins=30)   # las probabilidades se esparcen o todas < 0.5?
```

| Alarma | Qué ves | Diagnóstico |
|---|---|---|
| accuracy ≈ baseline | 0.868 ≈ 0.868 | Solución trivial: no aprendió, ignora la clase chica |
| AUC < 0.5 | 0.256 | Ranking invertido / roto |
| Matriz con fila/columna en 0 | nunca predice "1" | Colapso de clase |
| `pred.mean()` ≈ 0 | histograma todo < 0.5 | Nunca se compromete con la clase positiva |
| val loss plano | curva estancada | No aprende |
| val sube + train baja | gráfica divergente | Overfitting |

- ⚡ **Con clases desbalanceadas, accuracy sola miente**: predecir siempre "mayoría" la da gratis. Siempre comparar contra baseline + mirar AUC y matriz.

### C. La cura para este diagnóstico

```python
# 1) Skew → log1p antes del z-score
df['miss_km'] = np.log1p(df['miss_km'])
df['diam_min_m'] = np.log1p(df['diam_min_m'])

# 2) Desbalance → sobremuestrear la clase minoritaria (o pos_weight en el loss)
weights = torch.where(targets == 1, torch.tensor(6.0), torch.tensor(1.0))
sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)
dl = torch.utils.data.DataLoader(list(zip(Xtr, targets)), batch_size=64, sampler=sampler)
```

- alternativa al sampler: `nn.BCEWithLogitsLoss(pos_weight=peso)` (peso ≈ #neg/#pos).
- Resultado en el caso NEO tras la cura: **AUC 0.26 → 0.93**, recall de peligrosos de 0 → 0.87.