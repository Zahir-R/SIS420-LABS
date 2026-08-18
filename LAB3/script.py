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
# # Laboratorio 3 — Regresión Lineal Multivariable, Regresión Polinómica y Ecuación de la Normal
#
# En este cuadernillo se estiman los parámetros $\theta$ de un modelo de regresión sobre el dataset
# `energydata_complete.csv` de tres maneras distintas:
#
# 1. **Regresión lineal multivariable** mediante el descenso por el gradiente.
# 2. **Regresión polinómica** de grado 2 (descenso por el gradiente y solución cerrada con scikit-learn).
# 3. **Ecuación de la normal** (solución directa en forma cerrada).
#
# Para cada modelo se incluye su **gráfico de costo**, su entrenamiento y su **validación** con
# predicciones sobre datos no vistos (más de 100 predicciones), junto con las métricas MSE, RMSE y R².

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# ## 1. Descripción del dataset
#
# Se emplea el dataset `energydata_complete.csv`, que registra cada 10 minutos el consumo de energía
# de los electrodomésticos de una vivienda (columna objetivo `Appliances`, en Wh) junto con
# 27 variables de contexto: temperaturas y humedades en distintas zonas, temperatura/humedad exterior,
# presión atmosférica, velocidad del viento, visibilidad y otras.
#
# El dataset cumple los requisitos del enunciado: **n = 27 características (≥ 10)** y
# **m = 19,735 ejemplos (≥ 2,000)**, y ya fue registrado ante la estudiante responsable
# (Crismar Salazar Ramos). La columna `date` se elimina por ser temporal y no aportar información
# a la regresión, y no existen valores nulos.

# %%
df = pd.read_csv('./energydata_complete.csv')

print('Dimensiones del dataset (m filas, n+1 columnas):', df.shape)
print('Valores nulos:', df.isna().sum().sum())
print('\nPrimeras 5 filas:')
print(df.head())

# %%
print('Estadísticos descriptivos:')
print(df.describe().T)

# %%
y = df['Appliances'].values
X_df = df.drop(columns=['date', 'Appliances'])
X = X_df.values

m = y.size
n = X.shape[1]

print(f'Número de instancias (m): {m}')
print(f'Número de características (n): {n}')

print('\nMuestra de los primeros 10 registros:')
print('{:>12s}{:>12s}{:>14s}'.format('X[:, 0]', 'X[:, 1]', 'y (Appliances)'))
print('-' * 40)
for i in range(10):
    print('{:12.2f}{:12.2f}{:14.0f}'.format(X[i, 0], X[i, 1], y[i]))

# %% [markdown]
# ## 2. Análisis de correlación
#
# Antes de entrenar se inspecciona la relación de cada característica con la variable objetivo.
# Las correlaciones absolutas son bajas (la más alta corresponde a `lights`, ≈ 0.20), por lo que el
# desempeño esperado de los modelos será moderado, pero suficiente para demostrar el procedimiento
# completo de entrenamiento y validación. Esta baja relación individual también motiva el uso de un
# modelo polinómico con interacciones, que pueda capturar relaciones no lineales entre variables.
#
# **Nota sobre colinealidad:** las columnas `rv1` y `rv2` son idénticas entre sí (variables
# aleatorias), lo que produce una matriz $X^TX$ singular. Por ello se emplean métodos numéricamente
# robustos en todo el cuadernillo: la pseudoinversa (`pinv`) en la ecuación de la normal y la
# descomposición SVD que usa scikit-learn.

# %%
corr = df.corr(numeric_only=True)['Appliances'].drop('Appliances')

fig, ax = plt.subplots(figsize=(9, 7))
corr_abs = corr.abs().sort_values(ascending=True)
ax.barh(corr_abs.index, corr_abs.values, color='tab:orange')
ax.set_xlabel('|Correlación con Appliances|')
ax.set_title('Correlación de cada característica con la variable objetivo')
ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Partición entrenamiento/validación
#
# Para que la validación sea honesta, el dataset se divide en **80% de entrenamiento**
# (15,788 ejemplos) y **20% de validación** (3,947 ejemplos). Las predicciones finales
# (más de 100) se realizan sobre el conjunto de validación, que el modelo no vio al entrenar.

# %%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

m_train = y_train.size
m_val = y_val.size

print(f'Tamaño del conjunto de entrenamiento: {m_train}')
print(f'Tamaño del conjunto de validación: {m_val}')
print(f'Forma de X_train: {X_train.shape}')
print(f'Forma de X_val: {X_val.shape}')

# %% [markdown]
# ## 4. Modelo 1 — Regresión Lineal Multivariable (Descenso por el Gradiente)
#
# Se usan las 27 características de forma lineal. Como las variables tienen escalas muy distintas,
# primero se **normalizan** (media 0 y desviación estándar 1) calculando $\mu$ y $\sigma$ únicamente
# con el conjunto de entrenamiento y aplicándolas después también a la validación. Luego se agrega el
# término de intersección $x_0 = 1$.
#
# La función de costo es
# $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 $$
# y la regla de actualización del descenso por el gradiente por lotes es
# $$ \theta := \theta - \frac{\alpha}{m} X^T (X\theta - y) . $$
#
# Se entrena con $\alpha = 0.01$ y 10,000 iteraciones; el gráfico de costo muestra la convergencia.

# %%
def featureNormalize(X):
    X_norm = X.copy()
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    m = y.shape[0]
    J = (1 / (2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history

# %%
X_train_norm, mu, sigma = featureNormalize(X_train)

X_train_b = np.concatenate([np.ones((m_train, 1)), X_train_norm], axis=1)
X_val_norm = (X_val - mu) / sigma
X_val_b = np.concatenate([np.ones((m_val, 1)), X_val_norm], axis=1)

alpha = 0.01
num_iters = 10000

theta_dg = np.zeros(X_train_b.shape[1])
theta_dg, J_history_dg = gradientDescentMulti(X_train_b, y_train, theta_dg, alpha, num_iters)

print('theta calculado por el descenso por el gradiente (primeros 5 valores):', theta_dg[:5])
print('Costo inicial J:', J_history_dg[0])
print('Costo final J:', J_history_dg[-1])

# %%
plt.figure(figsize=(9, 4))
plt.plot(np.arange(1, len(J_history_dg) + 1), J_history_dg, color='tab:blue', lw=2)
plt.title('Gráfico de costo — Regresión Lineal Multivariable (DG)')
plt.xlabel('Número de iteraciones')
plt.ylabel('Costo J')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# %%
y_pred_dg = np.dot(X_val_b, theta_dg)

mse_dg = mean_squared_error(y_val, y_pred_dg)
rmse_dg = np.sqrt(mse_dg)
r2_dg = r2_score(y_val, y_pred_dg)

print('--- Validación: Regresión Lineal Multivariable (DG) ---')
print(f'Predicciones realizadas sobre validación: {y_pred_dg.size}')
print(f'MSE: {mse_dg:.2f}')
print(f'RMSE: {rmse_dg:.2f}')
print(f'R²: {r2_dg:.4f}')

# %% [markdown]
# ## 5. Modelo 2 — Regresión Polinómica de Grado 2
#
# Para capturar relaciones no lineales se amplía la matriz de características con
# `PolynomialFeatures(degree=2)`, que genera los términos lineales, los cuadráticos y todas las
# interacciones entre pares de variables: las 27 características originales se convierten en
# **405 características polinómicas** (más el término de intersección $x_0$).
#
# Este modelo se resuelve de dos formas:
#
# **a) Descenso por el gradiente** sobre el polinomio normalizado, igual que el Modelo 1 pero con
# 405 parámetros. Debido a la fuerte colinealidad entre tantas características, la convergencia con
# un único ritmo de aprendizaje es más lenta; el gráfico de costo lo evidencia.
#
# **b) Solución cerrada con scikit-learn** (`LinearRegression` sobre el polinomio), que resuelve los
# mínimos cuadrados por SVD y sirve como referencia exacta del óptimo.

# %%
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

print('Características polinómicas (grado 2), incluye x0:', X_train_poly.shape[1])
print('Forma de X_train_poly:', X_train_poly.shape)
print('Forma de X_val_poly:', X_val_poly.shape)

# %%
X_train_poly_norm, mu_poly, sigma_poly = featureNormalize(X_train_poly[:, 1:])
X_train_poly_b = np.concatenate([np.ones((m_train, 1)), X_train_poly_norm], axis=1)

X_val_poly_norm = (X_val_poly[:, 1:] - mu_poly) / sigma_poly
X_val_poly_b = np.concatenate([np.ones((m_val, 1)), X_val_poly_norm], axis=1)

# %%
alpha_poly = 0.01
num_iters_poly = 10000

theta_poly = np.zeros(X_train_poly_b.shape[1])
theta_poly, J_history_poly = gradientDescentMulti(X_train_poly_b, y_train, theta_poly, alpha_poly, num_iters_poly)

print('theta polinómico por DG (primeros 5 valores):', theta_poly[:5])
print('Costo inicial J:', J_history_poly[0])
print('Costo final J:', J_history_poly[-1])

# %%
plt.figure(figsize=(9, 4))
plt.plot(np.arange(1, len(J_history_poly) + 1), J_history_poly, color='tab:green', lw=2)
plt.title('Gráfico de costo — Regresión Polinómica Grado 2 (DG)')
plt.xlabel('Número de iteraciones')
plt.ylabel('Costo J')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# %%
y_pred_poly_dg = np.dot(X_val_poly_b, theta_poly)

mse_poly_dg = mean_squared_error(y_val, y_pred_poly_dg)
rmse_poly_dg = np.sqrt(mse_poly_dg)
r2_poly_dg = r2_score(y_val, y_pred_poly_dg)

print('--- Validación: Regresión Polinómica (DG) ---')
print(f'MSE: {mse_poly_dg:.2f}')
print(f'RMSE: {rmse_poly_dg:.2f}')
print(f'R²: {r2_poly_dg:.4f}')

# %%
reg_poly = LinearRegression()
reg_poly.fit(X_train_poly, y_train)

y_pred_poly = reg_poly.predict(X_val_poly)

mse_poly = mean_squared_error(y_val, y_pred_poly)
rmse_poly = np.sqrt(mse_poly)
r2_poly = r2_score(y_val, y_pred_poly)

print('theta polinómico por scikit-learn (primeros 5 coeficientes):', reg_poly.coef_[:5])
print('término independiente b:', reg_poly.intercept_)

# %%
print('--- Validación: Regresión Polinómica (scikit-learn) ---')
print(f'Predicciones realizadas sobre validación: {y_pred_poly.size}')
print(f'MSE: {mse_poly:.2f}')
print(f'RMSE: {rmse_poly:.2f}')
print(f'R²: {r2_poly:.4f}')

# %% [markdown]
# ## 6. Modelo 3 — Ecuación de la Normal
#
# La ecuación de la normal obtiene directamente los parámetros óptimos en forma cerrada:
# $$ \theta = (X^T X)^{-1} X^T y $$
# y a diferencia del descenso por el gradiente **no requiere normalización** de las características.
#
# Como $X^TX$ es singular por la colinealidad existente (`rv1` = `rv2`), se utiliza la
# **pseudoinversa** (`pinv`) para obtener una solución numéricamente estable. Este modelo se aplica
# sobre las 27 características lineales (agregando $x_0 = 1$), y su costo final se compara con el de
# los otros dos modelos en la sección siguiente.

# %%
def normalEqn(X, y):
    theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta

# %%
X_train_ne = np.concatenate([np.ones((m_train, 1)), X_train], axis=1)
X_val_ne = np.concatenate([np.ones((m_val, 1)), X_val], axis=1)

theta_ne = normalEqn(X_train_ne, y_train)

print('theta calculado por la ecuación de la normal (primeros 5 valores):', theta_ne[:5])

J_ne = computeCostMulti(X_train_ne, y_train, theta_ne)
print(f'Costo final J (entrenamiento): {J_ne:.2f}')

# %%
y_pred_ne = np.dot(X_val_ne, theta_ne)

mse_ne = mean_squared_error(y_val, y_pred_ne)
rmse_ne = np.sqrt(mse_ne)
r2_ne = r2_score(y_val, y_pred_ne)

print('--- Validación: Ecuación de la Normal ---')
print(f'Predicciones realizadas sobre validación: {y_pred_ne.size}')
print(f'MSE: {mse_ne:.2f}')
print(f'RMSE: {rmse_ne:.2f}')
print(f'R²: {r2_ne:.4f}')

# %% [markdown]
# ## 7. Comparación de modelos y predicciones
#
# Se comparan los tres modelos sobre el conjunto de validación (3,947 muestras, superando ampliamente
# las 100 predicciones exigidas). Se muestran: **(a)** el costo final de entrenamiento de cada caso,
# **(b)** las primeras 150 predicciones de cada modelo contra los valores reales y **(c)** una tabla
# resumen de las métricas MSE, RMSE y R² calculadas sobre validación.

# %%
J_dg = computeCostMulti(X_train_b, y_train, theta_dg)
J_poly_dg = computeCostMulti(X_train_poly_b, y_train, theta_poly)

print('Costo final J (entrenamiento) por modelo:')
print(f'  Regresión lineal multivariable (DG):  {J_dg:.2f}')
print(f'  Regresión polinómica grado 2 (DG):    {J_poly_dg:.2f}')
print(f'  Ecuación de la normal:                {J_ne:.2f}')

# %%
plt.figure(figsize=(8, 5))
modelos_costo = ['Lineal (DG)', 'Polinómica (DG)', 'Ecuación de la Normal']
costos = [J_dg, J_poly_dg, J_ne]
plt.bar(modelos_costo, costos, color=['tab:blue', 'tab:green', 'tab:red'])
plt.ylabel('Costo J (entrenamiento)')
plt.title('Costo final J para cada modelo')
for i, c in enumerate(costos):
    plt.text(i, c + 20, f'{c:.0f}', ha='center', va='bottom')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.show()

# %%
n_pred = 150
idx = np.arange(n_pred)

plt.figure(figsize=(14, 5))
plt.plot(idx, y_val[:n_pred], 'k-', linewidth=1.5, label='Valor real (validación)')
plt.plot(idx, y_pred_dg[:n_pred], 'r--', label='Lineal multivariable (DG)')
plt.plot(idx, y_pred_poly[:n_pred], 'b:', linewidth=2, label='Polinómica grado 2 (sklearn)')
plt.plot(idx, y_pred_ne[:n_pred], 'g-.', label='Ecuación de la Normal')
plt.xlabel('Índice de muestra')
plt.ylabel('Consumo de energía (Wh)')
plt.title('Comparación de predicciones vs valores reales (primeras 150 muestras de validación)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

print(f'Total de predicciones realizadas sobre el conjunto de validación: {m_val} (requisito: ≥ 100)')
print(f'Predicciones graficadas: {n_pred}')

# %%
resumen = pd.DataFrame({
    'Modelo': [
        'Lineal multivariable (DG)',
        'Polinómica grado 2 (DG)',
        'Polinómica grado 2 (scikit-learn)',
        'Ecuación de la Normal'
    ],
    'MSE': [mse_dg, mse_poly_dg, mse_poly, mse_ne],
    'RMSE': [rmse_dg, rmse_poly_dg, rmse_poly, rmse_ne],
    'R²': [r2_dg, r2_poly_dg, r2_poly, r2_ne]
})

print(resumen.to_string(index=False))

# %% [markdown]
# ## 8. Conclusiones
#
# - El **modelo polinómico de grado 2 con scikit-learn** obtuvo el mejor desempeño en validación
#   (R² ≈ 0.28, RMSE ≈ 85 Wh), superando al modelo lineal multivariable (R² ≈ 0.17). Esto confirma
#   que los términos cuadráticos y las interacciones entre variables aportan información útil que la
#   regresión lineal no captura.
# - El **descenso por el gradiente sobre las 405 características polinómicas** converge lentamente a
#   causa de la alta colinealidad entre variables (temperaturas/humedades correlacionadas y las
#   columnas idénticas `rv1`/`rv2`). Este comportamiento evidencia la ventaja de la solución en forma
#   cerrada (ecuación de la normal / mínimos cuadrados con SVD) frente a un gradiente con un único
#   ritmo de aprendizaje en problemas con muchas características correlacionadas.
# - La **ecuación de la normal** y el descenso por el gradiente sobre el modelo lineal alcanzan un
#   costo y métricas casi idénticos, como era de esperar al ser el mismo problema de mínimos cuadrados.
# - En todos los casos las predicciones siguen la tendencia general del consumo real; la dispersión
#   muestral es alta porque el consumo de electrodomésticos depende de variables que no están
#   presentes en el dataset (uso de los aparatos), lo que limita el R² alcanzable.
