# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="qx6hgOEy05B9"
# # Ejercicio de programación Regresión Polinomial

# %% id="sBcv-SUP05B_" executionInfo={"status": "ok", "timestamp": 1724338143227, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# utilizado para manejos de directorios y rutas
import os

# Computacion vectorial y cientifica para python
import numpy as np

# Librerias para graficación (trazado de gráficos)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # Necesario para graficar superficies 3D

# llama a matplotlib a embeber graficas dentro de los cuadernillos
# %matplotlib inline

# %% colab={"base_uri": "https://localhost:8080/"} id="3m3VdQdvoV_R" executionInfo={"status": "ok", "timestamp": 1724338173085, "user_tz": 240, "elapsed": 29861, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="bdefe772-5346-4b0a-b84a-5b1b9c71825a"
from google.colab import drive
drive.mount('/content/drive')

# %% id="JtKE0YbT05CA" executionInfo={"status": "ok", "timestamp": 1724338462590, "user_tz": 240, "elapsed": 485, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# Cargar datos
#data = np.loadtxt(os.path.join('Datasets', 'capacidad_adquisitiva.csv'), delimiter=';')
#from numpy import genfromtxt
#data = genfromtxt(os.path.join('Datasets', 'capacidad_adquisitiva.csv'), delimiter=';')
data = np.loadtxt('/content/drive/MyDrive/Colab Notebooks/machine learning/datasets/capacidad_adquisitiva.csv', delimiter=";",skiprows=1)

#print(data)
X = data[:, :1]
y = data[:, 1]
m = y.size
#print(X)
#print(y)
# imprimir algunos puntos de datos
#print('{:>8s}{:>10s}'.format(X, y))
#print('-'*26)
#for i in range(10):
#    print('{:8.0f}{:10.0f}'.format(X[i, 0], y[i]))

# %% colab={"base_uri": "https://localhost:8080/"} id="j-ukDJHd05CA" executionInfo={"status": "ok", "timestamp": 1724338468370, "user_tz": 240, "elapsed": 464, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="0f16ccd0-2723-42f8-be41-92c221be297f"
# Importamos la clase de Regresión Lineal de scikit-learn
from sklearn.linear_model import LinearRegression

# para generar características polinómicas
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree = 5)    # usaremos polinomios de grado 3
print(X.shape)
X = pf.fit_transform(X.reshape(-1,1))  # transformamos la entrada en polinómica
print(X.shape)
regresion_lineal = LinearRegression() # creamos una instancia de LinearRegression

# instruimos a la regresión lineal que aprenda de los datos (ahora polinómicos) (X,y)
regresion_lineal.fit(X, y)

# vemos los parámetros que ha estimado la regresión lineal
print('theta = ' + str(regresion_lineal.coef_) + ', b = ' + str(regresion_lineal.intercept_))

# resultado: w = [0 -4.54 4.95 0.1], b = -57.52

# %% colab={"base_uri": "https://localhost:8080/"} id="Th3KmhJW05CB" executionInfo={"status": "ok", "timestamp": 1724338474336, "user_tz": 240, "elapsed": 396, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="ee3c20be-ce98-4270-cb3e-1f0681f57313"
from sklearn.metrics import mean_squared_error # importamos el cálculo del error cuadrático medio (MSE)
# Predecimos los valores y para los datos usados en el entrenamiento
prediccion_entrenamiento = regresion_lineal.predict(X)
# Calculamos el Error Cuadrático Medio (MSE = Mean Squared Error)
mse = mean_squared_error(y_true = y, y_pred = prediccion_entrenamiento)
# La raíz cuadrada del MSE es el RMSE
rmse = np.sqrt(mse)
print('Error Cuadrático Medio (MSE) = ' + str(mse))
print('Raíz del Error Cuadrático Medio (RMSE) = ' + str(rmse))
# calculamos el coeficiente de determinación R2
r2 = regresion_lineal.score(X, y)
print('Coeficiente de Determinación R2 = ' + str(r2))


# %% id="F-G4y_tJ05CB" executionInfo={"status": "ok", "timestamp": 1724338476918, "user_tz": 240, "elapsed": 407, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def plotData(x, y):
    #Grafica los puntos x e y en una figura nueva.

    fig = pyplot.figure()  # abre una nueva figura

    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Edad personas')
    pyplot.xlabel('Capacidad adquisitiva 1,000s')



# %% colab={"base_uri": "https://localhost:8080/", "height": 466} id="4oZA6OpS05CB" executionInfo={"status": "ok", "timestamp": 1724338482700, "user_tz": 240, "elapsed": 636, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="89594c53-d989-4bae-d784-b3b7662e4846"
plotData(X[:,1], y)
pyplot.plot(X[:, 1], np.dot(X, regresion_lineal.coef_), '-')

# %% colab={"base_uri": "https://localhost:8080/"} id="DHtgLiyu05CC" executionInfo={"status": "ok", "timestamp": 1724338500211, "user_tz": 240, "elapsed": 455, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="dce7bafb-cb31-4a48-c0a0-d1875621f38d"
X_array = np.array([25])
pf = PolynomialFeatures(degree = 5)    # usaremos polinomios de grado 3
X_array = pf.fit_transform(X_array.reshape(-1,1))
prediccion_test = regresion_lineal.predict(X_array)
print(prediccion_test)
