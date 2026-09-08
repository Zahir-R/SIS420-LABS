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

# %% [markdown]
# # PyTorch - Datasets (Adaptado para Tiny ImageNet-200)

# %%
import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %% [markdown]
# ## 1. Carga y Preprocesamiento de Tiny ImageNet-200

# %%
# Carga del dataset desde el archivo Parquet
df = pd.read_parquet('./dataset/train.parquet')

def procesar_imagen(bytes_img):
    """Decodifica la imagen, la lleva a RGB 20x20 y la aplana a un vector de 1200 valores."""
    im = Image.open(io.BytesIO(bytes_img)).convert('RGB')
    im = im.resize((20, 20), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).ravel()

X = np.stack(df['image'].map(lambda d: procesar_imagen(d['bytes'])))
y = df['label'].values.astype(np.int64)

# División Estratificada 80% / 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalización basada exclusivamente en entrenamiento
mu = np.mean(X_train, axis=0)
sigma = np.std(X_train, axis=0)
sigma[sigma == 0] = 1

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

print(f"Entrenamiento: {X_train.shape} | Prueba: {X_test.shape}")

# %%
X_t = torch.from_numpy(X_train).float().cuda()
Y_t = torch.from_numpy(y_train).long().cuda()

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1, keepdims=True)

def evaluate(model, x):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_probas = softmax(y_pred)
        return torch.argmax(y_probas, axis=1)


D_in, H, D_out = 1200, 256, 200

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 100
log_each = 20
l = []
model.train()
for e in range(1, epochs+1):

    # forward
    y_pred = model(X_t)

    # loss
    loss = criterion(y_pred, Y_t)
    l.append(loss.item())

    # reinicio de gradientes
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # actualización de pesos
    optimizer.step()

    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")

y_pred = evaluate(model, torch.from_numpy(X_test).float().cuda())
print(f"Exactitud (Batch Gradient Descent): {accuracy_score(y_test, y_pred.cpu().numpy()):.4f}")


# %%
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 10
batch_size = 256
log_each = 2
l = []
model.train()
batches = len(X_t) // batch_size

for e in range(1, epochs+1):
    _l = []
    for b in range(batches):
        x_b = X_t[b*batch_size:(b+1)*batch_size]
        y_b = Y_t[b*batch_size:(b+1)*batch_size]

        y_pred = model(x_b)
        loss = criterion(y_pred, y_b)
        _l.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    l.append(np.mean(_l))
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")

y_pred = evaluate(model, torch.from_numpy(X_test).float().cuda())
print(f"Exactitud (Mini-batch Manual): {accuracy_score(y_test, y_pred.cpu().numpy()):.4f}")


# %%
class DatasetPersonalizado(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        # Mantenemos los tensores en CPU para la carga en memoria y se envían dinámicamente
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix], self.Y[ix]

dataset = DatasetPersonalizado(X_train, y_train)


# %%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 10
log_each = 2
l = []
model.train()

for e in range(1, epochs+1):
    _l = []
    for x_b, y_b in dataloader:
        x_b, y_b = x_b.cuda(), y_b.cuda()

        y_pred = model(x_b)
        loss = criterion(y_pred, y_b)
        _l.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    l.append(np.mean(_l))
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")

y_pred = evaluate(model, torch.from_numpy(X_test).float().cuda())
print(f"Exactitud (DataLoader): {accuracy_score(y_test, y_pred.cpu().numpy()):.4f}")

# %% [markdown]
# ## 6. Collate Function y Guardado de Checkpoints

# %%
def collate_fn(batch):
    return torch.stack([x for x, y in batch]), torch.stack([y for x, y in batch])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to("cuda")

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 10
log_each = 5
l = []
model.train()

for e in range(1, epochs+1):
    _l = []
    for x_b, y_b in dataloader:
        x_b, y_b = x_b.cuda(), y_b.cuda()

        y_pred = model(x_b)
        loss = criterion(y_pred, y_b)
        _l.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    l.append(np.mean(_l))
    if not e % log_each:
        print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")
        PATH = f"./checkpoint_{e}.pt"
        torch.save(model.state_dict(), PATH)

