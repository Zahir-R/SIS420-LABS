# %% [markdown]
# # Examen de Laboratorio SIS420 - Inteligencia Artificial I
# ## Ejercicio 1
# ### Estudiante: Rivero Teco Zahir Bari

# %%
import numpy as np

# %%
X = np.array([
    [1.5, 2.2, -0.8],
    [-0.5, 1.4, 0.9],
    [0.2, -1.5, -0.2],
    [2.1, 1.8, 1.5],
    [-1.1, -0.9, -1.0],
    [0.8, 0.3, 0.4]
])
y = np.array([0, 1, 0, 1, 0, 1])
m = y.size

# %% 
def featureNormalize(X):
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# %%
X_norm, mu, sigma = featureNormalize(X)
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

# %%
def sigmoid(z):
    z = np.array(z)
    return 1.0 / (1.0 + np.exp(-z))

# %%
def lrCostFunction(theta, X, y, lambda_):
    m = y.size
    if y.dtype == bool:
        y = y.astype(int)
    
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X.dot(theta.T))
    
    temp = theta.copy()
    temp[0] = 0
    
    J = (1 / m) * np.sum(-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h))) + (lambda_ / (2 * m)) * np.sum(np.square(temp))
    grad = (1 / m) * (h - y).dot(X) + (lambda_ / m) * temp
    
    return J, grad

# %%
def descensoGradienteReg(theta, X, y, alpha, num_iters, lambda_):
    theta = theta.copy()
    for i in range(num_iters):
        _, grad = lrCostFunction(theta, X, y, lambda_)
        theta = theta - alpha * grad
    return theta

# %%
theta_init = np.zeros(X.shape[1])
theta_opt, _ = descensoGradienteReg(theta_init, X, y, alpha=0.1, num_iters=1000, lambda_=0.05)
print(theta_opt)

