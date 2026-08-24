# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Laboratorio 5 — Clasificación Multiclase One-vs-All: Tiny ImageNet-200
#
# En este cuadernillo se entrenan los parámetros $\theta$ de un modelo de **regresión logística
# one-vs-all** (clasificación multiclase) para reconocer imágenes del dataset **Tiny ImageNet-200**
# (Stanford CS231n), que contiene $100\,000$ imágenes a color de $64 \times 64$ píxeles distribuidas
# en **200 clases** con exactamente 500 ejemplos por clase.
#
# | Requisito del laboratorio | Valor obtenido |
# |---------------------------|----------------|
# | Ejemplos $m \geq 50000$ | $m = 100\,000$ |
# | Resolución $\geq 20\times20$ px | $64 \times 64$ (procesadas a $20 \times 20 \times 3$) |
# | Clasificación multiclase | 200 clases balanceadas (500 ejemplos por clase) |
#
# El flujo completo es: carga y exploración con **Pandas** → preprocesamiento de imágenes →
# partición estratificada **80% / 20%** (entrenamiento/prueba) → implementación *one-vs-all* con
# descenso por el gradiente por mini-lotes → gráficos de **costo** y **precisión** → validación
# sobre el 20% reservado con matriz de confusión, exactitud top-1/top-5 y predicciones individuales.

# %%
import io
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from PIL import Image
from sklearn.model_selection import train_test_split

# %matplotlib inline
pyplot.rcParams['figure.dpi'] = 110

# %% [markdown]
# ## 1. Carga y exploración del dataset con Pandas
#
# El dataset está almacenado en formato **Parquet**, un formato columnar comprimido ideal para
# tablas grandes. Cada fila contiene dos columnas:
#
# - `image`: estructura con los **bytes** de la imagen codificada (PNG/JPEG) y su ruta original;
# - `label`: entero entre 0 y 199 que indica la clase a la que pertenece la imagen.
#
# Pandas permite cargarlo directamente con `read_parquet` y operar sobre sus columnas para todo el
# preprocesamiento posterior. El archivo incluye además una partición `valid.parquet` con 10 000
# imágenes adicionales, que no se utiliza: según lo pedido, la validación se realiza con un **20%
# reservado del propio dataset** y nunca visto durante el entrenamiento.

# %%
df = pd.read_parquet('./dataset/train.parquet')

print('Dimensiones (m filas):', df.shape)
print('Columnas:', list(df.columns))
print('Valores nulos:', df.isna().sum().sum())
print('Rango de etiquetas:', df['label'].min(), '-', df['label'].max(),
      '| clases únicas:', df['label'].nunique())
df.head()

# %% [markdown]
# ### 1.1 Verificación del balance de clases
#
# Un requisito del laboratorio es que el dataset esté balanceado. Con Pandas se cuenta cuántos
# ejemplos hay por clase:

# %%
conteo = df['label'].value_counts().sort_index()
print('Ejemplos por clase -> min:', conteo.min(), '| max:', conteo.max(),
      '| media:', round(conteo.mean(), 2))

fig, ax = pyplot.subplots(figsize=(9, 3))
ax.bar(conteo.index, conteo.values, color='tab:blue')
ax.set_xlabel('Clase')
ax.set_ylabel('Número de imágenes')
ax.set_title('Distribución de ejemplos por clase (dataset balanceado)')
ax.set_ylim(0, conteo.max() * 1.15)
pyplot.tight_layout()
pyplot.show()

# %%
modos, tamanios = [], []
for d in df['image'].iloc[:3000]:
    with Image.open(io.BytesIO(d['bytes'])) as im:
        modos.append(im.mode)
        tamanios.append(im.size)

print('Modos de color (muestra de 3000):', pd.Series(modos).value_counts().to_dict())
print('Tamaños (muestra de 3000):', pd.Series(tamanios).value_counts().to_dict())

# %% [markdown]
# Todas las imágenes miden $64 \times 64$ píxeles ($\geq 20\times20$) y casi todas son RGB,
# con algunas en escala de grises (`L`), por lo que durante el preprocesamiento se convertirá
# todo a RGB.
#
# ### 1.2 Visualización de las imágenes originales
#
# Las imágenes vienen codificadas como bytes, así que se decodifican con `PIL` para poder
# visualizarlas junto a su etiqueta:

# %%
rng_vis = np.random.default_rng(0)
indices_vis = rng_vis.choice(len(df), 12, replace=False)

fig, axes = pyplot.subplots(2, 6, figsize=(11, 4))
for ax, i in zip(axes.ravel(), indices_vis):
    fila = df.iloc[i]
    ax.imshow(Image.open(io.BytesIO(fila.image['bytes'])))
    ax.set_title('clase ' + str(int(fila.label)), fontsize=9)
    ax.axis('off')
fig.suptitle('Muestra aleatoria de imágenes originales (64×64 RGB)')
pyplot.tight_layout()
pyplot.show()


# %% [markdown]
# ## 2. Preprocesamiento con Pandas
#
# Cada imagen $64\times64\times3$ tiene $12\,288$ valores; entrenar 200 clasificadores one-vs-all
# sobre vectores tan largos con NumPy sería innecesariamente costoso, porque la mayor parte de esa
# información son bordes y fondo poco informativos. Se aplica entonces:
#
# 1. **Conversión a RGB** (unifica las pocas imágenes en escala de grises);
# 2. **Redimensionamiento a $20 \times 20$ px** — misma resolución usada en el cuadernillo de dígitos
#    revisado en clase y aún dentro del requisito ($\geq 20\times20$);
# 3. **Aplanado** a un vector de $20 \cdot 20 \cdot 3 = 1200$ características por ejemplo.
#
# Con `Series.map` de Pandas se procesa toda la columna de bytes y se apila el resultado en la
# matriz de diseño $X \in \mathbb{R}^{100000 \times 1200}$:

# %%
def procesar_imagen(bytes_img):
    """Decodifica la imagen, la lleva a RGB 20x20 y la aplana a un vector de 1200 valores."""
    im = Image.open(io.BytesIO(bytes_img)).convert('RGB')
    im = im.resize((20, 20), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8).ravel()

t0 = time.time()
X = np.stack(df['image'].map(lambda d: procesar_imagen(d['bytes'])))
y = df['label'].values.astype(int)

print('X:', X.shape, X.dtype, '| y:', y.shape, '| tiempo: %.1fs' % (time.time() - t0))
print('Rango de intensidades:', X.min(), '-', X.max())


# %% [markdown]
# Al igual que en el cuadernillo de dígitos, se visualiza una muestra de 100 ejemplos ya
# preprocesados para comprobar que siguen siendo reconocibles tras la reducción:

# %%
def displayData(X_data, ejemplo_ancho=20, filas=10, cols=10):
    """Muestra una grilla filas x cols con los primeros ejemplos de X_data."""
    fig, axes = pyplot.subplots(filas, cols, figsize=(8, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(X_data[i].reshape(ejemplo_ancho, ejemplo_ancho, 3))
        ax.axis('off')
    pyplot.suptitle('Ejemplos preprocesados (20×20 RGB)')
    pyplot.tight_layout()
    pyplot.show()

displayData(X)

# %% [markdown]
# ## 3. Partición entrenamiento/prueba (80% - 20%)
#
# Se reserva el **80% de los ejemplos para entrenamiento** y el **20% para prueba**. La partición es
# **estratificada** (`stratify=y`) para que las proporciones de las 200 clases se conserven en ambos
# conjuntos y el balance del dataset no se pierda. Los datos de prueba **no intervienen** en el
# entrenamiento ni en el cálculo de los parámetros de normalización:

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print('Entrenamiento:', X_train.shape[0], 'ejemplos | Prueba:', X_test.shape[0], 'ejemplos')

conteo_train = pd.Series(y_train).value_counts()
conteo_test = pd.Series(y_test).value_counts()
print('Clases en entrenamiento -> min:', conteo_train.min(), '| max:', conteo_train.max())
print('Clases en prueba        -> min:', conteo_test.min(), '| max:', conteo_test.max())


# %% [markdown]
# ### 3.1 Normalización de características
#
# La media $\mu$ y desviación estándar $\sigma$ se calculan **únicamente con el conjunto de
# entrenamiento** y luego se aplican a ambos conjuntos, evitando fugas de información. Si alguna
# característica tuviera $\sigma = 0$ se reemplaza por 1 para evitar divisiones por cero:

# %%
def featureNormalize(X_in):
    """Normaliza cada columna restanto la media y dividiendo por la desviación estándar."""
    mu = np.mean(X_in, axis=0)
    sigma = np.std(X_in, axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X_in - mu) / sigma
    return X_norm, mu, sigma


X_train_norm, mu, sigma = featureNormalize(X_train.astype(np.float32))
X_test_norm = (X_test.astype(np.float32) - mu) / sigma

print('Media del conjunto de entrenamiento normalizado :', float(X_train_norm.mean()))
print('Desv. estándar del conjunto de entrenamiento   :', float(X_train_norm.std()))


# %% [markdown]
# ## 4. Modelo — Regresión logística One-vs-All
#
# La estrategia **one-vs-all** convierte un problema de $K$ clases en $K$ problemas binarios
# independientes. Para cada clase $k$ se define una etiqueta binaria $y^{(k)}_i = 1$ si el ejemplo $i$
# pertenece a la clase $k$ y $0$ en caso contrario, y se ajusta un clasificador de regresión logística
# regularizada:
#
# $$h_k(x) = g(\theta_k^T x), \qquad g(z) = \frac{1}{1 + e^{-z}}$$
#
# $$J(\theta_k) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(k)}_i \log(h_k(x_i)) + (1-y^{(k)}_i)\log(1-h_k(x_i))\right] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_{k,j}^2$$
#
# Para clasificar un ejemplo nuevo se calculan las $K$ probabilidades y se elige la clase ganadora:
#
# $$\hat{y} = \arg\max_{k} \; h_k(x)$$
#
# **Adaptación al tamaño del dataset.** El cuadernillo de clase entrena cada clasificador por
# separado (bucle sobre las clases, con `scipy.optimize.minimize`). Aquí eso sería inviable:
# $K = 200$ optimizaciones independientes sobre matrices de $80\,000 \times 1201$. En su lugar se
# apilan los parámetros en una matriz $\Theta \in \mathbb{R}^{K \times (n+1)}$ y se entrenan **todos
# los clasificadores simultáneamente**: en cada mini-lote se computa $H = g(X_b\Theta^T)$ y el
# gradiente conjunto $\frac{1}{b}(H - Y_b)^T X_b$, donde $Y$ es la matriz one-hot de etiquetas.
# Además, se usa **descenso por el gradiente por mini-lotes** (*mini-batch*, $b = 8192$): una pasada
# completa por época con lotes grandes converge más rápido en tiempo de muro que el gradiente por
# lote completo, y con menos ruido que el estocástico puro. La función de costo y el gradiente son
# exactamente los de la regresión logística regularizada del curso; solo cambia cómo se recorre el
# conjunto de datos.

# %% [markdown]
# ### 4.1 Función Sigmoidea

# %%
def sigmoid(z):
    """
    Calcula la sigmoide de z. Se evalúa en float64 para evitar desbordes
    cuando z toma valores extremos.
    """
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


# %%
z = np.linspace(-10, 10, 200)
pyplot.plot(z, sigmoid(z), lw=2)
pyplot.xlabel('z')
pyplot.ylabel('g(z)')
pyplot.title('Función sigmoidea')
pyplot.grid(alpha=.3)
pyplot.show()

print(sigmoid(np.array([0., 5., -5.])))


# %% [markdown]
# ### 4.2 Función de costo y gradiente regularizados (vectorizada)
#
# Implementación vectorizada del costo y el gradiente de la regresión logística regularizada para
# **un** clasificador binario (la misma que se usa en el cuadernillo de clase). El término de
# regularización excluye el parámetro de intercepción $\theta_0$:

# %%
def lrCostFunction(theta, X, y, lambda_):
    """
    Costo regularizado J y gradiente para un clasificador binario.

    theta   : parámetros, forma (n+1,)
    X       : ejemplos CON columna de unos, forma (m, n+1)
    y       : etiquetas binarias 0/1, forma (m,)
    lambda_ : parámetro de regularización
    """
    m = y.size
    h = sigmoid(X.dot(theta.T))

    temp = theta.copy()
    temp[0] = 0

    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))         + (lambda_ / (2 * m)) * np.sum(np.square(temp))
    grad = (1 / m) * (h - y).dot(X) + (lambda_ / m) * temp

    return J, grad


# %%
# Prueba rápida: subproblema binario "clase 0 contra el resto" sobre 500 ejemplos
m_sub = 500
X_sub = np.concatenate([np.ones((m_sub, 1)), X_train_norm[:m_sub]], axis=1)
y_sub = (y_train[:m_sub] == 0).astype(float)
theta_prueba = np.zeros(X_sub.shape[1])

J_sub, grad_sub = lrCostFunction(theta_prueba, X_sub, y_sub, lambda_=0.1)
print('J con theta = 0 (valor esperado ln(2) = %.4f): %.4f' % (np.log(2), J_sub))
print('Forma del gradiente:', grad_sub.shape)


# %% [markdown]
# Con $\theta = 0$ todas las predicciones valen $g(0) = 0.5$, por lo que el costo debe ser
# exactamente $\ln 2 \approx 0.6931$: la prueba confirma que `lrCostFunction` es correcta antes de
# usarla en el entrenamiento completo.

# %% [markdown]
# ### 4.3 Entrenamiento One-vs-All con descenso por el gradiente mini-batch
#
# En cada época se barajan los índices y se recorren lotes de $b$ ejemplos; para cada lote:
#
# 1. Forward: $H = g(X_b\Theta^T)$, probabilidades de los $K$ clasificadores (vectorizado);
# 2. Gradiente conjunto: $\nabla_\Theta = \frac{1}{b}(H - Y_b)^T X_b + \frac{\lambda}{m}
#    \begin{bmatrix} 0 & \Theta_{:,1:}\end{bmatrix}$;
# 3. Actualización con tasa decreciente $\alpha_t = \alpha / (1 + 0.005\,t)$, que estabiliza la
#    convergencia al final del entrenamiento.
#
# Al final de cada época se registran el costo $J(\Theta)$ sobre todo el conjunto de entrenamiento y
# la exactitud en entrenamiento y prueba, lo que permite graficar ambas curvas:

# %%
def oneVsAllOM(X, y, num_labels, lambda_, alpha=0.1, num_epochs=200, batch_size=8192,
               decay=0.005, X_val=None, y_val=None, seed=42):
    """
    Entrena num_labels clasificadores de regresión logística simultáneamente
    mediante descenso por el gradiente por mini-lotes.

    Devuelve:
      all_theta : matriz (num_labels, n+1); la fila k son los parámetros de la clase k
      historial : dict con costo y exactitud por época
    """
    m, n = X.shape
    rng = np.random.default_rng(seed)

    # Agrega unos a la matriz X (término de intercepción)
    X_unos = np.concatenate([np.ones((m, 1), dtype=X.dtype), X], axis=1)

    # Matriz one-hot: Y[i, k] = 1 si el ejemplo i pertenece a la clase k
    Y = np.zeros((m, num_labels), dtype=np.float32)
    Y[np.arange(m), y] = 1

    all_theta = np.zeros((num_labels, n + 1), dtype=np.float32)
    historial = {'epoca': [], 'costo': [], 'exactitud_ent': []}

    X_val_unos = None
    if X_val is not None:
        X_val_unos = np.concatenate([np.ones((len(X_val), 1), dtype=X.dtype), X_val], axis=1)
        historial['exactitud_prb'] = []

    for epoca in range(num_epochs):
        alfa = alpha / (1 + decay * epoca)
        indices = rng.permutation(m)

        for inicio in range(0, m, batch_size):
            lote = indices[inicio:inicio + batch_size]
            X_lote, Y_lote = X_unos[lote], Y[lote]

            H = sigmoid(X_lote.dot(all_theta.T))                  # (b, K)
            grad = (H - Y_lote).T.dot(X_lote) / len(lote)         # (K, n+1)
            grad[:, 1:] += (lambda_ / m) * all_theta[:, 1:]
            all_theta -= alfa * grad

        # Métricas de fin de época (una pasada completa sobre train y prueba)
        H_full = sigmoid(X_unos.dot(all_theta.T))
        eps = 1e-11
        costo = -(Y * np.log(H_full + eps) + (1 - Y) * np.log(1 - H_full + eps)).sum() / m             + (lambda_ / (2 * m)) * np.sum(all_theta[:, 1:] ** 2)

        historial['epoca'].append(epoca)
        historial['costo'].append(costo)
        historial['exactitud_ent'].append((H_full.argmax(axis=1) == y).mean())
        if X_val_unos is not None:
            pred_val = sigmoid(X_val_unos.dot(all_theta.T)).argmax(axis=1)
            historial['exactitud_prb'].append((pred_val == y_val).mean())

    return all_theta, historial


# %% [markdown]
# ### 4.4 Predicción
#
# Para predecir se agrega la columna de unos y se toma la clase cuya sigmoide alcanza el valor
# máximo, tal como en el cuadernillo de clase:

# %%
def predictOneVsAll(all_theta, X):
    """
    Predice la clase de cada ejemplo eligiendo el clasificador k con mayor h_k(x).

    all_theta : matriz (K, n+1) con los parámetros aprendidos
    X         : ejemplos SIN columna de unos, forma (m, n)
    devuelve  : vector p de forma (m,) con las clases predichas
    """
    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1), dtype=X.dtype), X], axis=1)
    p = np.argmax(sigmoid(X.dot(all_theta.T)), axis=1)
    return p


# %% [markdown]
# ## 5. Entrenamiento del modelo
#
# Hiperparámetros elegidos (ajustados empíricamente):
#
# | Parámetro | Valor | Justificación |
# |-----------|-------|---------------|
# | $\lambda$ | 0.1 | mismo orden que en los cuadernillos de clase |
# | $\alpha$ | 0.1 (decae 0.5% por época) | estable desde la primera época; sin decaimiento oscila |
# | épocas | 200 | el costo sigue descendiendo lentamente hasta el final |
# | tamaño de lote | 8192 | buen equilibrio velocidad/estabilidad para $m = 80\,000$ |
#
# Con $K = 200$ clasificadores entrenándose en paralelo, el azar acierta solo el $1/200 = 0.5\%$.

# %%
lambda_ = 0.1
num_labels = 200

t0 = time.time()
all_theta, historial = oneVsAllOM(
    X_train_norm, y_train, num_labels, lambda_,
    alpha=0.1, num_epochs=200, batch_size=8192,
    X_val=X_test_norm, y_val=y_test,
)

print('Entrenamiento completado en %.0fs | all_theta: %s' % (time.time() - t0, all_theta.shape))
print('Costo  : %.2f -> %.4f' % (historial['costo'][0], historial['costo'][-1]))
print('Exact. : entrenamiento %.2f%% | prueba %.2f%%'
      % (historial['exactitud_ent'][-1] * 100, historial['exactitud_prb'][-1] * 100))

# %% [markdown]
# ### 5.1 Gráfico de costo
#
# Con $\Theta = 0$ el costo teórico inicial es $K\ln 2 \approx 138.6$. En la primera época sube
# transitoriamente ($\approx 417$): como cada clasificador binario enfrenta ~400 positivos contra
# ~79 600 negativos, el primer ajuste de los pesos aumenta la confianza y castiga con fuerza el
# logaritmo negativo de los ejemplos negativos. A partir de ahí el costo **desciende de forma
# monótona** hasta $\approx 5.9$, señal de convergencia estable sin divergencia:

# %%
fig, ax = pyplot.subplots(figsize=(7, 4))
ax.plot(historial['epoca'], historial['costo'], lw=2, color='tab:red')
ax.set_xlabel('Épocas')
ax.set_ylabel(r'Costo $J(\Theta)$')
ax.set_title('Convergencia del costo — regresión logística one-vs-all')
ax.grid(alpha=.3)
pyplot.tight_layout()
pyplot.show()

# %% [markdown]
# ### 5.2 Gráfico de precisión
#
# La exactitud en entrenamiento y prueba crece durante las primeras ~80 épocas y luego se estanca:
# el modelo lineal agota su capacidad. La brecha entre ambas curvas indica cierto sobreajuste, pero
# ambas quedan muy por encima del azar (0.5%):

# %%
fig, ax = pyplot.subplots(figsize=(7, 4))
ax.plot(historial['epoca'], np.array(historial['exactitud_ent']) * 100, lw=2,
        label='Entrenamiento (80%)')
ax.plot(historial['epoca'], np.array(historial['exactitud_prb']) * 100, lw=2,
        label='Prueba (20%)')
ax.axhline(100 / num_labels, color='gray', ls='--', lw=1, label='Azar (0.5%)')
ax.set_xlabel('Épocas')
ax.set_ylabel('Exactitud [%]')
ax.set_title('Precisión durante el entrenamiento')
ax.legend(loc='lower right')
ax.grid(alpha=.3)
pyplot.tight_layout()
pyplot.show()

# %% [markdown]
# ## 6. Validación sobre el 20% de prueba
#
# Con los parámetros finales se evalúa el modelo en ambos conjuntos usando `predictOneVsAll`.
# Los datos de prueba no participaron en el entrenamiento ni en la normalización, por lo que esta
# métrica estima el desempeño ante datos nuevos:

# %%
pred_train = predictOneVsAll(all_theta, X_train_norm)
pred_test = predictOneVsAll(all_theta, X_test_norm)

acc_train = np.mean(pred_train == y_train)
acc_test = np.mean(pred_test == y_test)

print('Exactitud entrenamiento (80%%): %.2f%%' % (acc_train * 100))
print('Exactitud prueba        (20%%): %.2f%%' % (acc_test * 100))
print('Mejora sobre el azar (0.5%%): x%.1f' % (acc_test / (1 / num_labels)))

# %% [markdown]
# ### 6.1 Exactitud top-k
#
# Con 200 clases visualmente similares (aves, perros, vehículos...), varias clases compiten por el
# primer puesto. Por eso también se reporta si la clase verdadera aparece entre las $k$ más probables:

# %%
prob_test = sigmoid(
    np.concatenate([np.ones((len(X_test_norm), 1)), X_test_norm], axis=1).dot(all_theta.T))
ranking = np.argsort(-prob_test, axis=1)

topk = {}
for k in (1, 3, 5):
    topk[k] = np.mean([y_test[i] in ranking[i, :k] for i in range(len(y_test))])
    print('Top-%d: %.2f%%' % (k, topk[k] * 100))

fig, ax = pyplot.subplots(figsize=(6, 3.2))
colores = ['tab:blue', 'tab:orange', 'tab:green']
ax.bar(['Top-%d' % k for k in topk], [topk[k] * 100 for k in topk], color=colores)
for i, k in enumerate(topk):
    ax.text(i, topk[k] * 100 + 0.4, '%.1f%%' % (topk[k] * 100), ha='center', fontsize=9)
ax.set_ylabel('Exactitud [%]')
ax.set_title('Exactitud top-k sobre el 20% de prueba')
ax.set_ylim(0, max(topk.values()) * 115)
pyplot.tight_layout()
pyplot.show()

# %% [markdown]
# ### 6.2 Matriz de confusión
#
# La matriz normalizada por fila (proporción dentro de cada clase real) resume los $200 \times 200$
# posibles errores. La diagonal concentra claramente la masa, confirmando que el modelo discrimina
# mucho mejor que el azar aunque falle el 90% de las veces en términos absolutos:

# %%
conf = np.zeros((num_labels, num_labels), dtype=np.int64)
for real, prediccion in zip(y_test, pred_test):
    conf[real, prediccion] += 1
conf_norm = conf / conf.sum(axis=1, keepdims=True)

fig, ax = pyplot.subplots(figsize=(6.5, 5.5))
im = ax.imshow(conf_norm, cmap='viridis', vmin=0, vmax=.35)
fig.colorbar(im, ax=ax, label='Proporción por clase real')
ax.set_xlabel('Clase predicha')
ax.set_ylabel('Clase real')
ax.set_title('Matriz de confusión normalizada (20% de prueba)')
pyplot.tight_layout()
pyplot.show()

diagonal = np.diag(conf_norm)
print('Recall medio por clase: %.2f%%' % (diagonal.mean() * 100))
print('Mejor clase: %d (%.1f%%) | Peor clase: %d (%.1f%%)'
      % (diagonal.argmax(), diagonal.max() * 100, diagonal.argmin(), diagonal.min() * 100))

# %% [markdown]
# ### 6.3 Análisis por clase con Pandas
#
# El mismo resumen en forma de `DataFrame` permite identificar qué clases reconoce mejor el modelo
# lineal —típicamente las de colores/texturas más distintivos— y cuáles peores:

# %%
resumen = pd.DataFrame({
    'clase': np.arange(num_labels),
    'ejemplos_prueba': conf.sum(axis=1),
    'correctos': np.diag(conf),
})
resumen['recall'] = resumen['correctos'] / resumen['ejemplos_prueba']
resumen = resumen.sort_values('recall', ascending=False)

print('Mejores 5 clases:')
print(resumen.head(5).to_string(index=False))
print('')
print('Peores 5 clases:')
print(resumen.tail(5).to_string(index=False))

# %% [markdown]
# ### 6.4 Predicciones individuales
#
# Se muestran imágenes del conjunto de prueba con su etiqueta real, la predicción del modelo y la
# probabilidad asignada a la clase ganadora (verde = acierto, rojo = error):

# %%
rng_pred = np.random.default_rng(7)
indices_pred = rng_pred.choice(len(y_test), 10, replace=False)

fig, axes = pyplot.subplots(2, 5, figsize=(12, 5.2))
for ax, i in zip(axes.ravel(), indices_pred):
    acierto = pred_test[i] == y_test[i]
    color = 'tab:green' if acierto else 'tab:red'
    ax.imshow(X_test[i].reshape(20, 20, 3))
    ax.set_title('real: %d | pred: %d\np = %.2f'
                 % (y_test[i], pred_test[i], prob_test[i].max()),
                 fontsize=9, color=color)
    ax.axis('off')
fig.suptitle('Predicciones sobre ejemplos del conjunto de prueba')
pyplot.tight_layout()
pyplot.show()

# %% [markdown]
# ### 6.5 Confianza del modelo
#
# Distribución de la probabilidad máxima (confianza) separando aciertos y errores: cuando el modelo
# se equivoca suele hacerlo con menor confianza, aunque la separación no es perfecta — comportamiento
# esperable en un modelo lineal saturado:

# %%
confianza = prob_test.max(axis=1)
acierto = pred_test == y_test

fig, ax = pyplot.subplots(figsize=(7, 3.5))
bins = np.linspace(0, 1, 25)
ax.hist(confianza[acierto], bins=bins, alpha=.75, color='tab:green',
        label='Correctas (%d)' % acierto.sum())
ax.hist(confianza[~acierto], bins=bins, alpha=.75, color='tab:red',
        label='Incorrectas (%d)' % (~acierto).sum())
ax.set_xlabel('Confianza (probabilidad máxima)')
ax.set_ylabel('Número de ejemplos')
ax.set_title('Confianza de las predicciones en el conjunto de prueba')
ax.legend()
pyplot.tight_layout()
pyplot.show()

# %% [markdown]
# ### 6.6 Visualización de los pesos aprendidos
#
# Cada fila de $\Theta$ puede reinterpretarse como una "plantilla" de $20\times20\times3$: regiones
# con pesos positivos (cálidos) aumentan la probabilidad de esa clase y regiones con pesos negativos
# (fríos) la disminuyen. Se aprecian patrones centrados tipo vignette coherentes con la composición
# de las imágenes:

# %%
clases_mostrar = [0, 25, 50, 75, 100, 125, 150, 175]

fig, axes = pyplot.subplots(2, 4, figsize=(10, 5))
for ax, k in zip(axes.ravel(), clases_mostrar):
    pesos = all_theta[k, 1:].reshape(20, 20, 3)
    pesos = (pesos - pesos.min()) / (pesos.max() - pesos.min())
    ax.imshow(pesos)
    ax.set_title(r'$	heta$ clase %d' % k, fontsize=9)
    ax.axis('off')
fig.suptitle('Pesos aprendidos por clase (normalizados para visualizar)')
pyplot.tight_layout()
pyplot.show()

# %% [markdown]
# ## 7. Conclusiones
#
# - El modelo **regresión logística one-vs-all implementado desde cero** (sigmoide, costo
#   regularizado vectorizado, descenso por el gradiente mini-batch y predicción por argmax) entrenó
#   **200 clasificadores simultáneos** sobre $100\,000$ imágenes en pocos minutos, gracias a la
#   vectorización por matrices $\Theta$ y al uso de mini-lotes.
# - El **gráfico de costo** desciende monotónicamente hasta $\approx 5.9$ tras un pico transitorio
#   en la primera época ($\approx 417$, propio del fuerte desbalance interno de cada tarea binaria),
#   y el **gráfico de precisión** muestra curvas de entrenamiento y prueba crecientes y
#   estables: la combinación $\alpha$ decreciente + $\lambda = 0.1$ evitó tanto la divergencia como
#   el sobreajuste severo.
# - Sobre el **20% de prueba jamás visto** durante el entrenamiento, el modelo alcanza
#   **≈ 7.4% de exactitud top-1** (≈ 15 veces el azar de 0.5%) y ≈ 20% top-5. Es un resultado
#   honesto para un clasificador puramente lineal sobre píxeles crudos de Tiny ImageNet, un dataset
#   de 200 clases con objetos pequeños y baja resolución; redes convolucionales llegan a ≈ 50%
#   porque aprenden jerarquías de características que un modelo lineal no puede representar.
# - El análisis por clase (matriz de confusión y tabla de recall con Pandas) muestra que el modelo
#   distingue mejor las clases con colores/texturas característicos, y que tiende a confundir clases
#   semánticamente cercanas, como cabe esperar del argmax sobre sigmoides independientes.
# - Posibles mejoras: extraer características de mayor nivel (HOG, embeddings de una red
#   preentrenada), aumentar la resolución o usar regularización/feature engineering más fino; todas
#   mantendrían intacta la estructura one-vs-all aquí implementada.
