# %% [markdown]
# # Examen de Laboratorio SIS420 - Inteligencia Artificial I
# ## Ejercicio 2
# ### Estudiante: Rivero Teco Zahir Bari

# %%
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %% 
X = np.array([
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.2],
    [8.0, 8.0], [8.5, 9.0], [8.2, 8.5],
    [1.1, 0.8], [0.9, 1.1], [9.0, 8.2]
])
m = X.shape[0]

# %%
k = 2

kmeans = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=42)

y_pred = kmeans.fit_predict(X)

print("Coordenadas de los centroides:")
print(kmeans.cluster_centers_)

# %%
X_new = np.array([[1.3, 1.9], [8.1, 8.8]])
predicciones_nuevas = kmeans.predict(X_new)
transformacion = kmeans.transform(X_new)

# %% 
print("Predicción para X_new:", predicciones_nuevas)
print("Transformación (distancias) para X_new:\n", transformacion)
