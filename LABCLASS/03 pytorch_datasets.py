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
#     name: python3
# ---

# %% [markdown] id="7VI4fFKqqhIs"
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sensioai/blog/blob/master/029_pytorch_datasets/pytorch_datasets.ipynb)

# %% [markdown] id="FjFFQJmWqhIw"
# # Pytorch - Datasets

# %% [markdown] id="Qx1GRqiSqhIx"
# En los posts anteriores hemos introducido los conceptos fundamentales de la librería de `Deep Learning` `Pytorch` y también hemos visto la funcionalidad que nos ofrece a la hora de diseñar y entrenar `redes neuronales`. En este post nos enfocamos en la herramientas que la librería nos da a la hora definir nuestros *datasets*.

# %% id="nAHiD268qhIx" executionInfo={"status": "ok", "timestamp": 1788280215444, "user_tz": 240, "elapsed": 5905, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
import torch

# %% [markdown] id="fGNLgMR7qhIz"
# ## Iterando tensores

# %% [markdown] id="kZ9DssmiqhIz"
# En los posts anteriores hemos utilizado el dataset MNIST para ilustrar los diferentes ejemplos que hemos visto. Vamos a seguir con este caso. A continuación tenemos una implementación en la que iteramos por los datos de manera explícita para entrenar nuestra red.

# %% id="Kf3B49IyqhI0" executionInfo={"status": "ok", "timestamp": 1788280234705, "user_tz": 240, "elapsed": 19259, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
from sklearn.datasets import fetch_openml
import numpy as np
# descarga datos

mnist = fetch_openml('mnist_784', version=1)
X, Y = mnist["data"], mnist["target"]

# %% id="PXR1_xNoUyg7" executionInfo={"status": "ok", "timestamp": 1788280235449, "user_tz": 240, "elapsed": 737, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# X_train, X_test, y_train, y_test = X[:60000] / 255., X[60000:] / 255., Y[:60000].astype(np.int), Y[60000:].astype(np.int)
x_2=np.array(X)
y_2=np.array(Y)

# normalización y split

X_train =x_2[:60000] / 255.
X_test =x_2[60000:] / 255.
y_train = y_2[:60000].astype(np.int32)
y_test = y_2[60000:].astype(np.int32)


X_t = torch.from_numpy(X_train).float().cuda()
Y_t = torch.from_numpy(y_train).long().cuda()

# %% id="DSH7GEmUqhI1" executionInfo={"status": "ok", "timestamp": 1788280235565, "user_tz": 240, "elapsed": 109, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
from sklearn.metrics import accuracy_score

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1,keepdims=True)

def evaluate(x):
    model.eval()
    y_pred = model(x)
    y_probas = softmax(y_pred)
    return torch.argmax(y_probas, axis=1)


# %% code_folding=[15] id="CmWA6s9hqhI1" outputId="aca52568-2783-4091-c1fc-c290873cff5c" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1788280241604, "user_tz": 240, "elapsed": 6037, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
D_in, H, D_out = 784, 100, 10

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

epochs = 100
log_each = 10
l = []
model.train()
for e in range(1, epochs+1):

    # forward
    y_pred = model(X_t)

    # loss
    loss = criterion(y_pred, Y_t)
    l.append(loss.item())

    # ponemos a cero los gradientes
    optimizer.zero_grad()

    # Backprop (calculamos todos los gradientes automáticamente)
    loss.backward()

    # update de los pesos
    optimizer.step()

    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")

y_pred = evaluate(torch.from_numpy(X_test).float().cuda())
accuracy_score(y_test, y_pred.cpu().numpy())

# %% [markdown] id="p9cLYR_jqhI3"
# ## Iterando por *Batches*

# %% [markdown] id="--FStgZhqhI3"
# En la implementación anterior estamos optimizando nuestro modelo con el algoritmo de `batch gradient descent`, en el que utilizamos todos nuestros datos en cada paso de optimización. Sin embargo, un algoritmo que puede converger más rápido (y única opción si nuestro dataset es tan grande que no cabe en memoria) es el de `mini-batch gradient descent` (el cual hemos ya utilizado en posts anteriores).

# %% code_folding=[17] id="6IqRUzSEqhI4" outputId="8b7ff5d1-beb7-4841-8760-9b5fbd6c971e" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1788280247861, "user_tz": 240, "elapsed": 6249, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
D_in, H, D_out = 784, 100, 10

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

epochs = 10
batch_size = 100
log_each = 1
l = []
model.train()
batches = len(X_t) // batch_size
for e in range(1, epochs+1):

    _l = []
    # iteramos por batches
    for b in range(batches):
        x_b = X_t[b*batch_size:(b+1)*batch_size]
        y_b = Y_t[b*batch_size:(b+1)*batch_size]

        # forward
        y_pred = model(x_b)

        # loss
        loss = criterion(y_pred, y_b)
        _l.append(loss.item())

        # ponemos a cero los gradientes
        optimizer.zero_grad()

        # Backprop (calculamos todos los gradientes automáticamente)
        loss.backward()

        # update de los pesos
        optimizer.step()

    l.append(np.mean(_l))
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")

y_pred = evaluate(torch.from_numpy(X_test).float().cuda())
accuracy_score(y_test, y_pred.cpu().numpy())


# %% [markdown] id="bfbouE7WqhI5"
# Si bien esta implementación es correcta y funcional, dependiendo de nuestros datos puede llegar a complicarse mucho (por ejemplo, si necesitamos cargar muchas imágenes a las cuales queremos aplicar transformaciones, juntar en batches, etc...). Además, es común reutilizar la lógica para cargar nuestros datos no sólo para entrenar la red, si no para generar predicciones. Este hecho motiva el uso de las clases especiales que `Pytorch` nos ofrece para ello.

# %% [markdown] id="MSGOAhTWqhI5"
# ## La clase *Dataset*

# %% [markdown] id="Mb6WxFXWqhI5"
# La primera clase que tenemos que conocer es la clase `Dataset`. Esta clase hereda de la clase madre `torch.utils.data.Dataset` y tenemos que definir, como mínimo, tres funciones:
#
# - `__init__`: el constructor
# - `__len__`: devuelve el número de muestras en el dataset
# - `__getitem__`: devuelve una muestra en concreto del dataset
#
# Una vez definida la clase, ésta puede usarse como si de cualquier iterador se tratase.

# %% id="YIXmp0UGqhI5" executionInfo={"status": "ok", "timestamp": 1788280247877, "user_tz": 240, "elapsed": 14, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# clase Dataset, hereda de la clase `torch.utils.data.Dataset`
# class DSPerros(torch.utils.data.Dataset):

class DatasetPersonalizado(torch.utils.data.Dataset):
    # constructor
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float().cuda()
        self.Y = torch.from_numpy(Y).long().cuda()
    # devolvemos el número de datos en el dataset
    def __len__(self):
        return len(self.X)
        # return len(self.Y)
    # devolvemos el elemento `ix` del dataset
    def __getitem__(self, ix):
        return self.X[ix], self.Y[ix]


# %% [markdown] id="7kDtghi2qhI6"
# Una vez definida la clase, podemos instanciar un objeto que podemos usar para iterar por nuestros datos.

# %% id="82ljNdg0qhI6" outputId="86bb9c22-578b-4b08-cc8d-774fbb0b12a3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1788280248028, "user_tz": 240, "elapsed": 149, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
dataset = DatasetPersonalizado(X_train, y_train)

len(dataset)

# %% code_folding=[2, 17] id="szluP5ZQqhI6" outputId="9e0b4bc4-e5ce-4312-8aa0-fb9993cde345" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1788280253844, "user_tz": 240, "elapsed": 5812, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
D_in, H, D_out = 784, 100, 10

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

epochs = 10
batch_size = 100
log_each = 1
l = []
model.train()
batches = len(dataset) // batch_size
for e in range(1, epochs+1):

    _l = []
    # iteramos por batches en el dataset
    for b in range(batches):
        x_b, y_b = dataset[b*batch_size:(b+1)*batch_size]

        # forward
        y_pred = model(x_b)

        # loss
        loss = criterion(y_pred, y_b)
        _l.append(loss.item())

        # ponemos a cero los gradientes
        optimizer.zero_grad()

        # Backprop (calculamos todos los gradientes automáticamente)
        loss.backward()

        # update de los pesos
        optimizer.step()

    l.append(np.mean(_l))
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")

y_pred = evaluate(torch.from_numpy(X_test).float().cuda())
accuracy_score(y_test, y_pred.cpu().numpy())

# %% [markdown] id="XV2nVro3qhI7"
# Podemos iterar directamente sobre el objeto `dataset` de la misma manera que hacíamos anteriormente, sin embargo `Pytorch` no ofrece otro objeto que nos facilita las cosas a la hora de iterar por batches.

# %% [markdown] id="Z1Wo3qx6qhI7"
# ## La clase *DataLoader*

# %% [markdown] id="iWsjciqnqhI7"
# La clase `DataLoader` recibe un `Dataset` e implementa la lógica para iterar nuestros datos en batches.

# %% id="kilkig6LqhI7" executionInfo={"status": "ok", "timestamp": 1788280253850, "user_tz": 240, "elapsed": 4, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
dataloader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=True)

# %% id="U3Q3311HqhI8" outputId="e72420cc-ecb1-4798-d2c7-ccc9d94854a2" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1788280253873, "user_tz": 240, "elapsed": 20, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
x, y = next(iter(dataloader))

x.shape, y.shape
# x[0], y[0]

# x, y = next(iter(dataloader))
# x.shape, y.shape
# x[0], y[0]

# %% [markdown] id="u7BlxvXtqhI8"
# También permite mezclar los datos al principio de cada epoch con el parámetro `shuffle`, de manera automática carga nuestros datos de manera optimizada utilizando varios *cores* de nuestra CPU si es posible, etc.

# %% code_folding=[15, 19] id="OaOwPTKxqhI8" outputId="c4f80c2a-1ea2-4b58-a572-0bc7bfb7b36b" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1788280261840, "user_tz": 240, "elapsed": 7960, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
D_in, H, D_out = 784, 100, 10

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

epochs = 10
log_each = 1
l = []
model.train()
for e in range(1, epochs+1):

    _l = []
    # iteramos por batches en el dataloader
    for x_b, y_b in dataloader:

        # forward
        y_pred = model(x_b)

        # loss
        loss = criterion(y_pred, y_b)
        _l.append(loss.item())

        # ponemos a cero los gradientes
        optimizer.zero_grad()

        # Backprop (calculamos todos los gradientes automáticamente)
        loss.backward()

        # update de los pesos
        optimizer.step()

    l.append(np.mean(_l))
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")

y_pred = evaluate(torch.from_numpy(X_test).float().cuda())
accuracy_score(y_test, y_pred.cpu().numpy())


# %% [markdown] id="C9i9OmmQqhI8"
# También permite definir nuestra propia lógica para crear los batches, algo que puede ser útil en ciertas ocasiones.

# %% id="AgBLI1sEqhI9" executionInfo={"status": "ok", "timestamp": 1788280261848, "user_tz": 240, "elapsed": 5, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
def collate_fn(batch):
    return torch.stack([x for x, y in batch]), torch.stack([y for x, y in batch])
    # , torch.stack([2.*x for x, y in batch])


# %% id="waqR8RfSqhI9" executionInfo={"status": "ok", "timestamp": 1788280261865, "user_tz": 240, "elapsed": 5, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=collate_fn)

# %% code_folding=[] id="BtchRSLiqhI9" outputId="495cc936-16ad-4a1e-fbf1-3fb3f61d2577" colab={"base_uri": "https://localhost:8080/", "height": 365} executionInfo={"status": "error", "timestamp": 1788280261946, "user_tz": 240, "elapsed": 77, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
D_in, H, D_out = 784, 100, 10

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

epochs = 10
log_each = 1
l = []
PATH = f"./checkpoint_9.pt"
model.load_state_dict(torch.load(PATH))
model.train()
for e in range(1, epochs+1):

    _l = []
    # iteramos por batches en el dataloader
    # no usamos x2_b, sólo es para ver un ejemplo
    for x_b, y_b in dataloader:

        # forward
        y_pred = model(x_b)

        # loss
        loss = criterion(y_pred, y_b)
        _l.append(loss.item())

        # ponemos a cero los gradientes
        optimizer.zero_grad()

        # Backprop (calculamos todos los gradientes automáticamente)
        loss.backward()

        # update de los pesos
        optimizer.step()

    l.append(np.mean(_l))
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")
        PATH = f"./checkpoint_{e}.pt"
        torch.save(model.state_dict(), PATH)

y_pred = evaluate(torch.from_numpy(X_test).float().cuda())
accuracy_score(y_test, y_pred.cpu().numpy())


# %% id="AgE727Y28kb2" executionInfo={"status": "aborted", "timestamp": 1788280261964, "user_tz": 240, "elapsed": 52833, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# guardar modelo

PATH = './checkpoint.pt'
torch.save(model.state_dict(), PATH)

# %% id="N0h5rb4Bk648" executionInfo={"status": "aborted", "timestamp": 1788280261971, "user_tz": 240, "elapsed": 52840, "user": {"displayName": "Carlos Walter Pacheco Lora", "userId": "05889892519883337793"}}
# guardar modelo

PATH = './checkpoint.pt'
torch.save(model.state_dict(), PATH)

# %% [markdown] id="O9uM6JWZqhI9"
# ## Resumen

# %% [markdown] id="Cg4Vj2LJqhI9"
# En este post hemos visto diferentes maneras en las que podemos iterar por nuestros datos para entrenar un modelo en `Pytorch`. Si nuestro dataset es sencillo y podemos representarlo como un simple `array` de `NumPy` podemos iterar directamente el `array`, transformándolo previamente en un `tensor`. Sin embargo, cuando nuestro dataset sea más grande y no quepa en memoria o necesite cierto pre-proceso o transformaciones, es muy conveniente utilizar las clases que `Pytorch` nos ofrece para ello. Estas clases son, principalmente, el `Dataset` y el `DataLoader`, las cuales nos van a permitir iterar por nuestros datos de manera eficiente y generar *batches* de forma sencilla (además de otras funcionalidades como mezclar los datos al principio de cada epoch, cargar datos en paralelo, etc).
