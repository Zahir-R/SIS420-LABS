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
# # Laboratorio 6 — Reconstrucción LAB5: Clasificación Multiclase con Red Neuronal
#
# Reconstrucción del **LAB5** (clasificación multiclase one-vs-all sobre Tiny ImageNet-200)
# usando **PyTorch**. Dataset: 100 000 imágenes a color, 200 clases balanceadas (500 por clase),
# procesadas a 20x20 RGB (1200 features).

# %%
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib import pyplot

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# %% [markdown]
# ## 1. Carga y preprocesamiento del dataset

# %%
def procesar_imagen(bytes_img):
    im = Image.open(io.BytesIO(bytes_img)).convert('RGB')
    im = im.resize((20, 20), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).ravel()

df = pd.read_parquet('../../LAB5/dataset/train.parquet')
print(f"Registros: {df.shape[0]}, Clases: {df['label'].nunique()}")

X = np.stack(df['image'].map(lambda d: procesar_imagen(d['bytes'])))
y = df['label'].values.astype(np.int64)

mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma == 0] = 1
X = (X - mu) / sigma

print(f"X shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Entrenamiento: {X_train.shape[0]} | Prueba: {X_test.shape[0]}")

# %% [markdown]
# ## 2. Objeto Dataset

# %%
class TinyImageNetDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix], self.Y[ix]

# %%
train_set = TinyImageNetDataset(X_train, y_train)
test_set = TinyImageNetDataset(X_test, y_test)

batch_size = 256
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# %% [markdown]
# ## 3. Modelo (red neuronal multicapa)

# %%
D_in, H, D_out = X.shape[1], 256, df['label'].nunique()

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to(device)

print(f"Entradas: {D_in} -> Ocultas: {H} -> Clases: {D_out}")

# %% [markdown]
# ## 4. Objetos de costo y optimización

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% [markdown]
# ## 5. Entrenamiento

# %%
num_epochs = 15
log_each = 3
J_history = []

model.train()
for epoch in range(1, num_epochs + 1):
    epoch_loss = []

    for x_b, y_b in train_loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)

        pred = model(x_b)
        loss = criterion(pred, y_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

    J_history.append(sum(epoch_loss) / len(epoch_loss))

    if not epoch % log_each:
        print(f"Epoca {epoch}/{num_epochs} Costo J = {J_history[-1]:.4f}")

# %% [markdown]
# ## 6. Gráfico de costo

# %%
pyplot.figure(figsize=(9, 6))
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Numero de epocas')
pyplot.ylabel('Costo J')
pyplot.title('Grafico de costo del entrenamiento (LAB5)')
pyplot.grid(True, linestyle='--', alpha=0.5)
pyplot.savefig('costo.png')
pyplot.show()

# %% [markdown]
# ## 7. Precisión y métricas

# %%
def evaluar(loader, model, nombre):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            preds = model(x_b.to(device))
            preds = torch.argmax(preds, axis=1)
            y_true.extend(y_b.numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    print(f"[{nombre}] Accuracy = {acc:.4f}")
    return y_true, y_pred

y_true_train, y_pred_train = evaluar(train_loader, model, "Entrenamiento")
y_true_test, y_pred_test = evaluar(test_loader, model, "Prueba")

# %% [markdown]
# ## 8. Guardar los pesos del modelo

# %%
PATH = './checkpoint.pt'
torch.save(model.state_dict(), PATH)
print("Pesos guardados en:", PATH)

# %% [markdown]
# ## 9. Cargar el modelo y predecir

# %%
model_loaded = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
).to(device)
model_loaded.load_state_dict(torch.load(PATH))
model_loaded.eval()

# Predicción de las primeras 5 muestras de prueba
x_sample = torch.from_numpy(X_test[:50]).to(device)
with torch.no_grad():
    preds = torch.argmax(model_loaded(x_sample), axis=1).cpu().numpy()

print("Predicciones sobre las primeras 5 muestras de prueba:")
for i in range(50):
    print(f"  Real: {y_test[i]} | Predicho: {preds[i]}")

# %%
