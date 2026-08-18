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

# %% [markdown] id="tfwOfkjclHQx"
# # Ejercicio de programación Regresión Lineal Multiple

# %% id="f0WGMMljlHQz" executionInfo={"status": "ok", "timestamp": 1786467219790, "user_tz": 240, "elapsed": 10, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# utilizado para manejos de directorios y rutas
import os

# Computacion vectorial y cientifica para python
import numpy as np

# Librerias para graficación (trazado de gráficos)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # Necesario para graficar superficies 3D

# llama a matplotlib a embeber graficas dentro de los cuadernillos
import IPython
ip = IPython.get_ipython()
if ip is not None:
    ip.run_line_magic("matplotlib", "inline")

# %% id="LhERtghFlVGr" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786467243194, "user_tz": 240, "elapsed": 23403, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="cbf9a374-c299-4be6-f6a1-fb3de3558b7d"
# from google.colab import drive
# drive.mount("/content/gdrive")

# %% [markdown] id="G6-ykOwelHQ0"
# ## 2 Regresión lineal con multiples variables
#
# Se implementa la regresion lineal multivariable para predecir el precio de las casas. El archivo `Datasets/ex1data2.txt` contiene un conjunto de entrenamiento de precios de casas en Portland, Oregon. La primera columna es el tamaño de la casa en metros cuadrados, la segunda columna es el numero de cuartos, y la tercera columna es el precio de la casa.
#
# <a id="section4"></a>
# ### 2.1 Normalización de caracteristicas
#
# Al visualizar los datos se puede observar que las caracteristicas tienen diferentes magnitudes, por lo cual se debe transformar cada valor en una escala de valores similares, esto con el fin de que el descenso por el gradiente pueda converger mas rapidamente.

# %% id="6BP5grkxlHQ0" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786467246502, "user_tz": 240, "elapsed": 3319, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="55347f71-73ab-4806-f15d-78afdc3c4153"
# Cargar datos
# data = np.loadtxt(os.path.join('Datasets', 'ex1data2.txt'), delimiter=',')
data = np.loadtxt(os.path.join('Datasets', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
print(m)
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))


# %% [markdown] id="7iU_3mwZlHQ1"
# La desviación estándar es una forma de medir cuánta variación hay en el rango de valores de una característica en particular (la mayoría de los puntos caeran en un rango de ± 2 en relación a la desviaciones estándar de la media); esta es una alternativa a tomar el rango de valores (max-min). En `numpy`, se puede usar la función `std` para calcular la desviacion estandar.
#
# Por ejemplo, la caracteristica`X[:, 0]` contiene todos los valores de $x_1$ (tamaño de las casas) en el conjunto de entrenamiento, entonces `np.std(X[:, 0])` calcula la desviacion estandar de los tamaños de las casas.
# En el momento en que se llama a la función `featureNormalize`, la columna adicional de unos correspondiente a $ x_0 = 1 $ aún no se ha agregado a $ X $.
#
# <div class="alert alert-block alert-warning">
# **Notas para la implementación:** Cuando se normalize una caracteristica, es importante almacenar los valores usados para la normalización - el valor de la media y el valor de la desviación estandar usado para los calculos. Despues de aprender los parametros del modelo, se deseara predecir los precios de casas que no se han visto antes. Dado un nuevo valor de x (area del living room y el numero de dormitorios), primero se debe normalizar x usando la media y la desviacion estandar que se empleo anteriormente en el conjunto de entrenamiento para entrenar el modelo.
# </div>
# <a id="featureNormalize"></a>

# %% id="7xFD8H3WlHQ1" executionInfo={"status": "ok", "timestamp": 1786467246520, "user_tz": 240, "elapsed": 17, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


# %% id="ipL_QsTZlHQ2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786467246531, "user_tz": 240, "elapsed": 9, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="51cfd809-27fe-4dff-99df-cefcbc13a20f"
# llama featureNormalize con los datos cargados
X_norm, mu, sigma = featureNormalize(X)
print(X)
print('Media calculada:', mu)
print('Desviación estandar calculada:', sigma)
print(X_norm)

# %% id="Tw-uAOTXQVoV" executionInfo={"status": "ok", "timestamp": 1786467246542, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}

# %% [markdown] id="it9bMYuLlHQ2"
# Despues de `featureNormalize` la funcion es provada, se añade el temino de interseccion a `X_norm`:

# %% id="nNaGVxgolHQ2" executionInfo={"status": "ok", "timestamp": 1786467246565, "user_tz": 240, "elapsed": 22, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# Añade el termino de interseccion a X
# (Columna de unos para X0)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# %% id="nbUVohnhlHQ3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786467246578, "user_tz": 240, "elapsed": 11, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="78ebbb71-6e28-4df6-8995-be61c94e256d"
print(X)


# %% [markdown] id="xzLiIE6NlHQ3"
# <a id="section5"></a>
# ### 2.2 Descenso por el gradiente
#
# En el ejemplo anterior se implemento el descenso por el gradiente para un problema de regresion univariable. La unica diferencia es que ahora existe una caracteristica adicional en la matriz $X$. La función de hipótesis y la regla de actualización del descenso del gradiente por lotes permanecen sin cambios.
#
# La implementacion de las funciones `computeCostMulti` y `gradientDescentMulti` son similares a la funcion de costo y función de descenso por el gradiente de la regresión lineal multiple es similar al de la regresion lineal multivariable. Es importante garantizar que el codigo soporte cualquier numero de caracteristicas y esten bien vectorizadas.
#
# Se puede utilizar `shape`, propiedad de los arrays `numpy`, para identificar cuantas caracteristicas estan consideradas en el dataset.
#
# <div class="alert alert-block alert-warning">
# **Nota de implementación:** En el caso de multivariables, la función de costo puede se escrita considerando la forma vectorizada de la siguiente manera:
#
# $$ J(\theta) = \frac{1}{2m}(X\theta - \vec{y})^T(X\theta - \vec{y}) $$
#
# donde:
#
# $$ X = \begin{pmatrix}
# - (x^{(1)})^T - \\
# - (x^{(2)})^T - \\
# \vdots \\
# - (x^{(m)})^T - \\ \\
# \end{pmatrix} \qquad \mathbf{y} = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ \vdots \\ y^{(m)} \\\end{bmatrix}$$
#
# La version vectorizada es eficiente cuando se trabaja con herramientas de calculo numericos computacional como `numpy`.
# </div>
#
# <a id="computeCostMulti"></a>

# %% id="EQCMN7KqlHQ3" executionInfo={"status": "ok", "timestamp": 1786467246597, "user_tz": 240, "elapsed": 13, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def computeCostMulti(X, y, theta):
    m = y.shape[0] # numero de ejemplos de entrenamiento
    J = 0
    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))
    return J



# %% id="Vdy_aQUklHQ4" executionInfo={"status": "ok", "timestamp": 1786467246607, "user_tz": 240, "elapsed": 9, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0] # numero de ejemplos de entrenamiento
    theta = theta.copy()
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history


# %% [markdown] id="a8b_GwZslHQ4"
# #### 3.2.1 Seleccionando coheficientes de aprendizaje
#

# %% id="sId0DiH9lHQ4" colab={"base_uri": "https://localhost:8080/", "height": 500} executionInfo={"status": "ok", "timestamp": 1786467246942, "user_tz": 240, "elapsed": 336, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="ee8ec106-7d11-4df4-ae91-3206fad9e14d"
# Elegir algun valor para alpha (probar varias alternativas)
alpha = 0.001 # alpha = 0.003
num_iters = 10000

# inicializa theta y ejecuta el descenso por el gradiente
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Grafica la convergencia del costo
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Numero de iteraciones')
pyplot.ylabel('Costo J')

# Muestra los resultados del descenso por el gradiente
print('theta calculado por el descenso por el gradiente: {:s}'.format(str(theta)))

# Estimar el precio para una casa de 1650 sq-ft, con 3 dormitorios
X_array = [1, 1650, 3]
X_array[1:3] = (X_array[1:3] - mu) / sigma
price = np.dot(X_array, theta)   # Se debe cambiar esto

print('El precio predecido para una casa de 1650 sq-ft y 3 dormitorios (usando el descenso por el gradiente): ${:.0f}'.format(price))

# %% id="y0aZhWEqlHQ4" executionInfo={"status": "ok", "timestamp": 1786467246954, "user_tz": 240, "elapsed": 11, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
X_array = [1, 1222, 3]
# X_array[1:3] = (X_array[1:3] - mu) / sigma

# %% id="KZ4G9CbJlHQ4" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786467246961, "user_tz": 240, "elapsed": 17, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="b0bc5742-599a-4ae0-b278-8dd845c311a6"
price = np.dot(X_array, theta)   # Se debe cambiar esto

print('El precio predecido para una casa de 1222 sq-ft y 3 dormitorios (usando el descenso por el gradiente): ${:.0f}'.format(price))

# %% [markdown] id="-nMzqD8elHQ4"
# <a id="section7"></a>
# ### 2.3 Ecuacion de la Normal
#
# Una manera de calcular rapidamente el modelo de una regresion lineal es:
#
# $$ \theta = \left( X^T X\right)^{-1} X^T\vec{y}$$
#
# Utilizando esta formula no requiere que se escale ninguna caracteristica, y se obtendra una solucion exacta con un solo calculo: no hay “bucles de convergencia” como en el descenso por el gradiente.
#
# Primero se recargan los datos para garantizar que las variables no esten modificadas. Recordar que no es necesario escalar las caracteristicas, se debe agregar la columna de unos a la matriz $X$ para tener el termino de intersección($\theta_0$).

# %% id="C6j77JNmlHQ5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786467246971, "user_tz": 240, "elapsed": 10, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="8012b7f4-c1a0-464e-a939-696d29726182"
# Cargar datos
data = np.loadtxt(os.path.join('Datasets', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
print(m)
X = np.concatenate([np.ones((m, 1)), X], axis=1)


# %% id="gVZdjjk9lHQ5" executionInfo={"status": "ok", "timestamp": 1786467246986, "user_tz": 240, "elapsed": 14, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def normalEqn(X, y):
    theta = np.zeros(X.shape[1])
    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    return theta


# %% id="Ybyw-FfolHQ5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786467247013, "user_tz": 240, "elapsed": 26, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="b436aeaf-b33e-47cd-e03c-badd0e09ab8a"
# Calcula los parametros con la ecuación de la normal
theta = normalEqn(X, y);

# Muestra los resultados optenidos a partir de la aplicación de la ecuación de la normal
print('Theta calculado a partir de la ecuación de la normal: {:s}'.format(str(theta)));

# Estimar el precio para una casa de superficie de 1650 sq-ft y tres dormitorios
# %% 
X_array = [1, 500, 5]
price = np.dot(X_array, theta)
print(theta)
print('Precio predecido para una cada de superficie de 1650 sq-ft y 3 dormitorios (usando la ecuación de la normal): ${:.0f}'.format(price))
