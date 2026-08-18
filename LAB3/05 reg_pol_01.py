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

# %% [markdown] id="v14L-L099B4-"
# # Ejercicio de programación Regresión Polinomial

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 30136, "status": "ok", "timestamp": 1786630422237, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="LhERtghFlVGr" outputId="5004cc9f-142c-4f4a-8032-15d86a5e4a32"
from google.colab import drive
drive.mount("/content/gdrive")

# %% id="3FvMrI1J9B5A" executionInfo={"status": "ok", "timestamp": 1786630422239, "user_tz": 240, "elapsed": 5, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# utilizado para manejos de directorios y rutas
import os

# Computacion vectorial y cientifica para python
import numpy as np

# Librerias para graficación (trazado de gráficos)
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # Necesario para graficar superficies 3D

# llama a matplotlib a embeber graficas dentro de los cuadernillos
# %matplotlib inline

# %% [markdown] id="LFPSpSF29B5B"
# ## 2 Regresión polinomica
#
# Se implementa la regresion polinomial para predecir el la capacidad adquisitiva de una persona. El archivo `Datasets/capacidad_adquisitiva.csv` contiene un dataset para entrenamiento de capacidad adquisitiva de las personas considerando la edad de las personas de la ciudad de Sucre. La primera columna es la edad y la segunda columna es la capacidad adquisitiva en bolivianos por mes.
#
# <a id="section4"></a>
# ### 2.1 Normalización de caracteristicas
#
# Al visualizar los datos se puede observar que las caracteristicas tienen diferentes magnitudes, por lo cual se debe transformar cada valor en una escala de valores similares, esto con el fin de que el descenso por el gradiente pueda converger mas rapidamente.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1359, "status": "ok", "timestamp": 1786630423599, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="AxW2gEpi9B5B" outputId="63981137-c2e4-470e-b22c-8d18876a9bdc"
# Cargar datos
#data = np.loadtxt(os.path.join('Datasets', 'capacidad_adquisitiva.csv'), delimiter=';')
#from numpy import genfromtxt
#data = genfromtxt(os.path.join('Datasets', 'capacidad_adquisitiva.csv'), delimiter=';')
data = np.loadtxt('/content/gdrive/MyDrive/Colab Notebooks/machine learning/datasets/capacidad_adquisitiva.csv', delimiter=";",skiprows=1)
#print(data)
X = data[:, :1]
y = data[:, 1]
m = y.size
#print(X)
#print(y)
# imprimir algunos puntos de datos
#print('{:>8s}{:>10s}'.format(X, y))
#print('-'*26)
for i in range(20):
    print('{:8.0f}{:10.0f}'.format(X[i, 0], y[i]))


# %% [markdown] id="nebQ6gDk9B5C"
# La desviación estándar es una forma de medir cuánta variación hay en el rango de valores de una característica en particular (la mayoría de los puntos caeran en un rango de ± 2 en relación a la desviaciones estándar de la media); esta es una alternativa a tomar el rango de valores (max-min). En `numpy`, se puede usar la función `std` para calcular la desviacion estandar.
#
# Por ejemplo, la caracteristica`X[:, 0]` contiene todos los valores de $x_1$ (edades) en el conjunto de entrenamiento, entonces `np.std(X[:, 0])` calcula la desviacion estandar de las edades.
# En el momento en que se llama a la función `featureNormalize`, la columna adicional de unos correspondiente a $ x_0 = 1 $ aún no se ha agregado a $ X $.
#
# <div class="alert alert-block alert-warning">
# **Notas para la implementación:** Cuando se normalize una caracteristica, es importante almacenar los valores usados para la normalización - el valor de la media y el valor de la desviación estandar usado para los calculos. Despues de aprender los parametros del modelo, se deseara predecir la capacidad adquisitiva que no se han visto antes. Dado un nuevo valor de x (edad), primero se debe normalizar x usando la media y la desviacion estandar que se empleo anteriormente en el conjunto de entrenamiento para entrenar el modelo.
# </div>
# <a id="featureNormalize"></a>

# %% id="tppUto7Y9B5C" executionInfo={"status": "ok", "timestamp": 1786630423613, "user_tz": 240, "elapsed": 12, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def plotData(x, y):
    #Grafica los puntos x e y en una figura nueva.

    fig = pyplot.figure()  # abre una nueva figura

    pyplot.plot(x, y, 'ro', ms=10, mec='k')
    pyplot.ylabel('Edad personas')
    pyplot.xlabel('Capacidad adquisitiva 1,000s')



# %% colab={"base_uri": "https://localhost:8080/", "height": 449} executionInfo={"elapsed": 517, "status": "ok", "timestamp": 1786630424132, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="k51x70dt9B5D" outputId="f2d480b1-0f83-452d-a855-2dcdd2bab06a"
plotData(X, y)

# %% id="C4EFyCGs9B5D" executionInfo={"status": "ok", "timestamp": 1786630424145, "user_tz": 240, "elapsed": 6, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
X = np.concatenate([X, X * X], axis=1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 14, "status": "ok", "timestamp": 1786630424160, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="-ywnkafP9B5E" outputId="7e3c7848-1a5d-41a4-b226-23452a3d0441"
print(X)


# %% id="IYJ2E1T59B5E" executionInfo={"status": "ok", "timestamp": 1786630424163, "user_tz": 240, "elapsed": 2, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def  featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.mean(X, axis = 0)
    sigma = np.std(X, axis = 0)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma


# %% id="3GmXYXlp9B5E" executionInfo={"status": "ok", "timestamp": 1786630424186, "user_tz": 240, "elapsed": 22, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# llama featureNormalize con los datos cargados
X_norm, mu, sigma = featureNormalize(X)

#print(X)
#print('Media calculada:', mu)
#print('Desviación estandar calculada:', sigma)
#print(X_norm)

# %% colab={"base_uri": "https://localhost:8080/"} id="jwCqU-Qwm1r7" executionInfo={"status": "ok", "timestamp": 1786630424199, "user_tz": 240, "elapsed": 12, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="f4cc27fe-1a96-4f5d-dfb0-b2d2d5e2ef51"
print(X_norm)

# %% [markdown] id="mtjZN37n9B5F"
# Despues de `featureNormalize` la funcion es provada, se añade el temino de interseccion a `X_norm`:

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} executionInfo={"elapsed": 224, "status": "ok", "timestamp": 1786630424423, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="wyZf2SUV9B5F" outputId="c3d763c9-392d-45d6-af70-f812949918ea"
plotData(X_norm[:,1], y)

# %% id="nbHQvsVZ9B5F" executionInfo={"status": "ok", "timestamp": 1786630424425, "user_tz": 240, "elapsed": 1, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# Añade el termino de interseccion a X
# (Columna de unos para X0)
#X_norm = np.concatenate([X_norm, X_norm * X_norm], axis=1)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 41, "status": "ok", "timestamp": 1786630424467, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="65pwOHdb9B5F" outputId="c15e7007-5f67-4639-878f-422eaa90d2b0"
print(X)


# %% [markdown] id="ejIWPYMZ9B5F"
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

# %% id="AXrQGcgf9B5G" executionInfo={"status": "ok", "timestamp": 1786630424489, "user_tz": 240, "elapsed": 2, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def computeCostMulti(X, y, theta):
    # Inicializa algunos valores utiles
    m = y.shape[0] # numero de ejemplos de entrenamiento

    J = 0

    h = np.dot(X, theta)

    J = (1/(2 * m)) * np.sum(np.square(np.dot(X, theta) - y))

    return J



# %% id="PDQ_Sr7Z9B5G" executionInfo={"status": "ok", "timestamp": 1786630424491, "user_tz": 240, "elapsed": 1, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def gradientDescentMulti(X, y, theta, alpha, num_iters):

    # Inicializa algunos valores
    m = y.shape[0] # numero de ejemplos de entrenamiento

    # realiza una copia de theta, el cual será acutalizada por el descenso por el gradiente
    theta = theta.copy()

    J_history = []

    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


# %% [markdown] id="YgeZf5Pc9B5G"
# #### 3.2.1 Seleccionando coheficientes de aprendizaje
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 500} executionInfo={"elapsed": 3401, "status": "ok", "timestamp": 1786630427930, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="rIUXxlk69B5G" outputId="ea13f603-f33b-469b-868d-f8abd8d70025"
# Elegir algun valor para alpha (probar varias alternativas)
alpha = 0.001
num_iters = 100000

# inicializa theta y ejecuta el descenso por el gradiente
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Grafica la convergencia del costo
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Numero de iteraciones')
pyplot.ylabel('Costo J')

# Muestra los resultados del descenso por el gradiente
print('theta calculado por el descenso por el gradiente: {:s}'.format(str(theta)))

# La capacidad adquisitiva de una persona de 33 años
X_array = [1, 34, 1156]
X_array[1:3] = (X_array[1:3] - mu) / sigma
price = np.dot(X_array, theta)   # Se debe cambiar esto

print('La capacidad adquisitiva para una persona de 33 (usando el descenso por el gradiente): ${:.0f}'.format(price))

# %% colab={"base_uri": "https://localhost:8080/", "height": 466} executionInfo={"elapsed": 320, "status": "ok", "timestamp": 1786630428253, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="AcRsRvFQ9B5G" outputId="aa202ce1-9577-4a33-cc69-dab8064ea2b9"
plotData(X[:, 1], y)
pyplot.plot(X[:, 1], np.dot(X, theta), '-')

# %% id="SlY5GiBK9B5G" executionInfo={"status": "ok", "timestamp": 1786630428260, "user_tz": 240, "elapsed": 3, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
X_array = [1, 23, 529]
X_array[1:3] = (X_array[1:3] - mu) / sigma

# %% id="tJ3hCOwl9B5G" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786630428277, "user_tz": 240, "elapsed": 14, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="68a94eae-6403-4c3d-bc71-144ef9e93cca"
X_array[1:3]
print(np.dot(X_array, theta))

# %% [markdown] id="2Cbyh5UW9B5H"
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

# %% id="AMlouZTP9B5H" executionInfo={"status": "ok", "timestamp": 1786630428289, "user_tz": 240, "elapsed": 10, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# Cargar datos
data = np.loadtxt('/content/gdrive/MyDrive/Colab Notebooks/machine learning/datasets/capacidad_adquisitiva.csv', delimiter=";",skiprows=1)
X = data[:, :1]
y = data[:, 1]
m = y.size
X_original = X.copy()
X = np.concatenate([X, X * X], axis=1)
X = np.concatenate([np.ones((m, 1)), X], axis=1)


# %% id="POZIn2K39B5H" executionInfo={"status": "ok", "timestamp": 1786630428295, "user_tz": 240, "elapsed": 2, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def normalEqn(X, y):

    theta = np.zeros(X.shape[1])

    theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)

    return theta


# %% id="NNDsWCJ39B5H" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1786630428314, "user_tz": 240, "elapsed": 14, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="fcd41680-3c33-44f8-f532-797c6b26b90b"
# Calcula los parametros con la ecuación de la normal
theta = normalEqn(X, y);

# Muestra los resultados optenidos a partir de la aplicación de la ecuación de la normal
print('Theta calculado a partir de la ecuación de la normal: {:s}'.format(str(theta)));

# Estimar el precio para una casa de superficie de 1650 sq-ft y tres dormitorios

X_array = [1, 20, 400]
price = np.dot(X_array, theta)

print('Capacidad adquisitica para una edad de 20 años(usando la ecuación de la normal): ${:.0f}'.format(price))

# %% id="oDPCkGzf9B5H" colab={"base_uri": "https://localhost:8080/", "height": 466} executionInfo={"status": "ok", "timestamp": 1786630428595, "user_tz": 240, "elapsed": 278, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}} outputId="0c8d99c4-d1d4-46bd-8896-289acfdb6859"
plotData(X[:, 1], y)
#X = np.concatenate([np.ones((m, 1)), X], axis=1)
#X = np.concatenate([X, X * X], axis=1)

pyplot.plot(X[:, 1], np.dot(X, theta), '-')
