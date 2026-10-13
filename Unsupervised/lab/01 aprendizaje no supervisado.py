# %% [markdown]
# ## En 2D
# - Generar aleatoriamente un numero N de centroides 3 < N < 20 (Esto es K?)
# - Generar aleatoriamente las posiciones iniciales de los N centroides
# - Generar aleatoriamente las distancias de los puntos generados al N_i centroide
# - Generar aleatoriamente M: el numero de puntos por centroide 10 < M < 100
# - Aplicar K-Means (AI)
# - Mostrar los 5 puntos mas cercanos al centroide N_i (AI)
# ## En 3D
# - Lo mismo pero en 3D (AI)
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

# %% [markdown] id="UfVg51ES7Aqp"
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juansensio/blog/blob/master/096_ml_unsupervised/096_ml_unsupervised.ipynb)

# %% [markdown] id="aiY4e73I7Aqq"
# # ML - Aprendizaje No Supervisado

# %% [markdown] id="hjh47f5l7Aqr"
# Si bien todos los ejemplos de *Machine Learning* que hemos visto hasta ahora en esta serie se han basado en aprendizaje supervisado (nuestros datos van acompañados de las etiquetas correspondientes, ejemplos de la tarea que queremos llevar a cabo) la mayoría de los datos no están etiquetados. Antes esta problemática tenemos dos alternativas: etiquetar datos manualmente (lo cual require de tiempo, esfuerzo y dinero en el caso de querer desarrollar sistemas reales) o bien usar técnicas de aprendizaje no supervisado, o *Unsupervised Learning* en inglés.

# %% [markdown] id="M7GuSQ1j7Aqr"
# ## Clustering

# %% [markdown] id="vBiIgAYj7Aqr"
# De entre las diferentes técnicas de aprendizaje no supervisado, el *Clustering* es una de las más usadas. Este técnica consiste en identificar aquellas muestras similares del conjunto de datos y asignarlas a un *cluster*, o grupo. Esto permite aplicaciones tales como: segmentación de clientes, análisis de datos, reducción de la dimensionalidad, detección de anomalías, aprendizaje semi-supervisado, motores de búsqueda e incluso la semgentación de imágenes.

# %% [markdown] id="A5t7WNyc7Aqr"
# ### K-Means

# %% [markdown] id="ISahHdYb7Aqr"
# El algoritmo de *K-Means* es uno de los más usados a la hora de aplicar *Clustering*, ya que es un método rápido y eficiente. Vamos a generar un conjunto de datos sintético para aprender a usar esta técnica.

# %% executionInfo={"elapsed": 1805, "status": "ok", "timestamp": 1790699883633, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="jmBuLrUM7Aqs"
from sklearn.datasets import make_blobs
import numpy as np

blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.2,  1.8],
     [-2.5,  2.8],
     [-2.8,  1.3]])
#blob_std = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
#blob_std = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
blob_std = np.array([0.9, 0.9, 0.9, 0.9, 0.9])

X, y = make_blobs(n_samples=1000, centers=blob_centers,
                  cluster_std=blob_std, random_state=42)

# %% executionInfo={"elapsed": 20, "status": "ok", "timestamp": 1790699883651, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="tLCDbRD-7Aqs"
import matplotlib.pyplot as plt

def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)



# %% colab={"base_uri": "https://localhost:8080/", "height": 393} executionInfo={"elapsed": 498, "status": "ok", "timestamp": 1790699884162, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="pLtwvQ0l7Aqs" outputId="0d7b6fe7-74f1-4211-8999-7ac532e7b53c"
plt.figure(figsize=(8, 4))
plot_clusters(X)
plt.show()

# %% [markdown] id="WrZdFmB97Aqt"
# En *Scikit-Learn* puedes usar el objeto [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), indicando el número de clusters deseado.  

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 9, "status": "ok", "timestamp": 1790700022732, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="mm9Ad-3z7Aqt" outputId="3049e832-082d-4217-90f6-d433dc6eb420"
from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)
y_pred

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 4, "status": "ok", "timestamp": 1790699884802, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="zDMvUF0Y7Aqt" outputId="f916f40e-739a-42eb-bc8c-b71fd478cffe"
kmeans.cluster_centers_

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 49, "status": "ok", "timestamp": 1790700023183, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="-B1ssBXq7Aqt" outputId="c6b7008f-7bc5-473e-df02-da9802d5ca24"
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, -2.5]])
kmeans.predict(X_new)


# %% executionInfo={"elapsed": 19, "status": "ok", "timestamp": 1790700036660, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="8mA6t9T17Aqt"
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=10, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=10,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)


# %% colab={"base_uri": "https://localhost:8080/", "height": 393} executionInfo={"elapsed": 2043, "status": "ok", "timestamp": 1790699886929, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="uEKPpw6A7Aqt" outputId="61341b62-7c7b-4536-e4ff-68dc74fcd68d"
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
plt.show()

# %% [markdown] id="cms3bL2L7Aqt"
#
# En lugar de asignar un grupo a cada muestra (*hard clustering*) se le puede asignar una puntuación (*soft clustering*) en función de su distancia al centroide.

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 23, "status": "ok", "timestamp": 1790699886954, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}, "user_tz": 240} id="HrnAICFu7Aqu" outputId="fa33f19e-13ea-45f0-9402-ddc74829293b"
kmeans.transform(X_new)

