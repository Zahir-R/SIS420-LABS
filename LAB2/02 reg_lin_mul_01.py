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

# %% [markdown] id="tfwOfkjclHQx"
# # Ejercicio de programación Regresión Lineal Multiple

# %% executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1786467219790, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="f0WGMMljlHQz"
# utilizado para manejos de directorios y rutas
import os

# Computacion vectorial y cientifica para python
import numpy as np
import pandas as pd
# Librerias para graficación (trazado de gráficos)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # Necesario para graficar superficies 3D

# llama a matplotlib a embeber graficas dentro de los cuadernillos
# %matplotlib inline

import seaborn as sns

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 3319, "status": "ok", "timestamp": 1786467246502, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="6BP5grkxlHQ0" outputId="55347f71-73ab-4806-f15d-78afdc3c4153"
# Cargar datos
df = pd.read_csv('./energydata_complete.csv')
y = df['Appliances'].values
X_df = df.drop(columns=['date', 'Appliances'])
X = X_df.values
m = y.size
print(X.shape[1])
print(m)
# imprimir algunos puntos de datos
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))


# %% executionInfo={"elapsed": 17, "status": "ok", "timestamp": 1786467246520, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="7xFD8H3WlHQ1"
def  featureNormalize(X):
    X_norm = X.copy()

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1786467246531, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="ipL_QsTZlHQ2" outputId="51cfd809-27fe-4dff-99df-cefcbc13a20f"
X_norm, mu, sigma = featureNormalize(X)

print(X)
print('Media calculada:', mu)
print('Desviación estandar calculada:', sigma)
print(X_norm)

# %% executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1786467246565, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="nNaGVxgolHQ2"
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1786467246578, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="nbUVohnhlHQ3" outputId="78ebbb71-6e28-4df6-8995-be61c94e256d"
print(X)


# %% executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1786467246597, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="EQCMN7KqlHQ3"
def computeCostMulti(X, y, theta):
    # Inicializa algunos valores utiles
    m = y.shape[0] # numero de ejemplos de entrenamiento
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J


# %% executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1786467246607, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="Vdy_aQUklHQ4"
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # Inicializa algunos valores
    m = y.shape[0] # numero de ejemplos de entrenamiento
    theta = theta.copy()

    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


# %% colab={"base_uri": "https://localhost:8080/", "height": 500} executionInfo={"elapsed": 336, "status": "ok", "timestamp": 1786467246942, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="sId0DiH9lHQ4" outputId="ee8ec106-7d11-4df4-ae91-3206fad9e14d"
alpha = 0.001 # alpha = 0.003
num_iters = 10000

# inicializa theta y ejecuta el descenso por el gradiente
theta = np.zeros(X.shape[1])
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Grafica la convergencia del costo
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Numero de iteraciones')
pyplot.ylabel('Costo J')

# Muestra los resultados del descenso por el gradiente
print('theta calculado por el descenso por el gradiente: {:s}'.format(str(theta)))

ejemplo = X_df.iloc[0].values
ejemplo_dg = (ejemplo - mu) / sigma
X_array = np.insert(ejemplo_dg, 0, 1)
consumo_pred = np.dot(X_array, theta)

print(f'Valor de Appliances: {y[0]}')
print(f'Consumo de energía predicho por descenso por el gradiente: {consumo_pred:.2f}')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 10, "status": "ok", "timestamp": 1786467246971, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="C6j77JNmlHQ5" outputId="8012b7f4-c1a0-464e-a939-696d29726182"
# Cargar datos
df = pd.read_csv('./energydata_complete.csv')
X_df = df.drop(columns=['date', 'Appliances'])
X = X_df.values
y = df['Appliances'].values
m = y.size
print(m)
print(X.shape[1])
X = np.concatenate([np.ones((m, 1)), X], axis=1)


# %% executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1786467246986, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="gVZdjjk9lHQ5"
def normalEqn(X, y):
    #theta = np.zeros(X.shape[1])

    #theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    theta = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 26, "status": "ok", "timestamp": 1786467247013, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="Ybyw-FfolHQ5" outputId="b436aeaf-b33e-47cd-e03c-badd0e09ab8a"
# Calcula los parametros con la ecuación de la normal
theta = normalEqn(X, y);

# Muestra los resultados optenidos a partir de la aplicación de la ecuación de la normal
print('Theta calculado a partir de la ecuación de la normal: {:s}'.format(str(theta)));

ejemplo_n = np.insert(X_df.iloc[0].values, 0, 1)
consumo_pred_n = np.dot(ejemplo_n, theta)

print(f'Consumo predicho por Ecuación de la Normal: {consumo_pred_n:.2f}')

# %%
y_pred_dg = np.dot(X, theta)

pyplot.plot(y[:100], color='black', lw=2)
pyplot.plot(y_pred_dg[:100], lw=2)
pyplot.xlabel('Índice de muestra')
pyplot.ylabel('Consumo de Energía (Wh)')

# %%
