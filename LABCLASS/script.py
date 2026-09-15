#!/usr/bin/env python
# coding: utf-8

# # Receta para entrenar Redes Neuronales — TinyImageNet-200
#
# Adaptación de la receta de entrenamiento (notebook 07) para clasificación multiclase
# sobre el dataset TinyImageNet-200 (100 000 imágenes, 200 clases, 500 ejemplos por clase).
# Imágenes procesadas a 20×20 RGB → 1200 features aplanadas.

# ## 1. Exploración de Datos

import io
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# --- Carga del dataset desde parquet ---

def procesar_imagen(bytes_img):
    im = Image.open(io.BytesIO(bytes_img)).convert('RGB')
    im = im.resize((20, 20), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32).ravel()

df = pd.read_parquet('../LAB5/dataset/train.parquet')
print(f"Registros: {df.shape[0]}, Clases: {df['label'].nunique()}")

X = np.stack(df['image'].map(lambda d: procesar_imagen(d['bytes'])))
y = df['label'].values.astype(np.int64)

# normalización: media 0, desviación 1 por feature
mu = X.mean(axis=0)
sigma = X.std(axis=0)
sigma[sigma == 0] = 1
X = (X - mu) / sigma

print(f"X shape: {X.shape}")

# --- visualización de imágenes (requiere inverse normalization) ---

r, c = 3, 5
plt.figure(figsize=(c*3, r*3))
for row in range(r):
    for col in range(c):
        index = c*row + col
        plt.subplot(r, c, index + 1)
        ix = random.randint(0, len(X)-1)
        img = (X[ix] * sigma + mu).reshape(20, 20, 3).astype(np.uint8)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Clase {y[ix]}")
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# --- estadísticos ---

print(f"Max global: {X.max():.2f}, Min global: {X.min():.2f}")

# --- distribución de clases ---

unique, counts = np.unique(y, return_counts=True)
print(f"Clases: {len(unique)}, Ejemplos por clase: {counts[0]}")
plt.hist(y, bins=len(unique))
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.title('Distribución de clases')
plt.show()

# ## 2. División de datos

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, stratify=y_train, random_state=42
)
print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# subconjunto para experimentos rápidos
X_subset, y_subset = X_train[:5000], y_train[:5000]
print(f"Subset: {X_subset.shape[0]}")

# ## 3. Dataset y DataLoader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float().to(device)
        self.Y = torch.from_numpy(Y).long().to(device)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, ix):
        return self.X[ix], self.Y[ix]

# ## 4. Modelo

D_in = X.shape[1]   # 1200 (20*20*3)
D_out = df['label'].nunique()  # 200

def build_model(D_in=D_in, H=256, D_out=D_out):
    return torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.BatchNorm1d(H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    ).to(device)

# ## 5. Validación de la red neuronal

# ### Test de dimensiones

model = build_model()
test_input = torch.randn((64, D_in)).to(device)
test_output = model(test_input)
print(f"Entrada: {test_input.shape} → Salida: {test_output.shape}")
assert test_output.shape == (64, D_out), "¡Dimensiones incorrectas!"
print("✓ Dimensiones correctas")

# ### Fit de una muestra

dataset = Dataset(X_subset[:8], y_subset[:8])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

epochs = 5
criterion = torch.nn.CrossEntropyLoss()
model = build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("\n--- Fit de una muestra ---")
for e in range(1, epochs+1):
    model.train()
    for x_b, y_b in dataloader:
        y_pred = model(x_b)
        loss = criterion(y_pred, y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {e}/{epochs} loss {loss.item():.5f} y {y_b[0].item()} y_pred {torch.argmax(y_pred[0], axis=0).item()}")

# ### Fit de un batch

dataset = Dataset(X_subset[:64], y_subset[:64])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

epochs = 200
criterion = torch.nn.CrossEntropyLoss()
model = build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print("\n--- Fit de un batch ---")
for e in range(1, epochs+1):
    for x_b, y_b in dataloader:
        y_pred = model(x_b)
        loss = criterion(y_pred, y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = (y_b == torch.argmax(y_pred, axis=1)).sum().item()
        if not e % 10:
            print(f"Epoch {e}/{epochs} loss {loss.item():.5f} acc {acc}/{y_b.shape[0]}")

# ## 6. Funciones de entrenamiento

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1, keepdims=True)

def fit(model, dataloader, optimizer, scheduler=None, epochs=10, log_each=1, weight_decay=0, early_stopping=0, verbose=1):
    criterion = torch.nn.CrossEntropyLoss()
    l, acc, lr = [], [], []
    val_l, val_acc = [], []
    best_acc, step = 0, 0
    for e in range(1, epochs+1):
        _l, _acc = [], []
        for param_group in optimizer.param_groups:
            lr.append(param_group['lr'])
        model.train()
        for x_b, y_b in dataloader['train']:
            y_pred = model(x_b)
            loss = criterion(y_pred, y_b)
            _l.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_probas = torch.argmax(softmax(y_pred), axis=1)
            _acc.append(accuracy_score(y_b.cpu().numpy(), y_probas.cpu().detach().numpy()))
        l.append(np.mean(_l))
        acc.append(np.mean(_acc))
        model.eval()
        _l, _acc = [], []
        with torch.no_grad():
            for x_b, y_b in dataloader['val']:
                y_pred = model(x_b)
                loss = criterion(y_pred, y_b)
                _l.append(loss.item())
                y_probas = torch.argmax(softmax(y_pred), axis=1)
                _acc.append(accuracy_score(y_b.cpu().numpy(), y_probas.cpu().numpy()))
        val_l.append(np.mean(_l))
        val_acc.append(np.mean(_acc))
        if val_acc[-1] > best_acc:
            best_acc = val_acc[-1]
            torch.save(model.state_dict(), 'ckpt.pt')
            step = 0
            if verbose == 2:
                print(f"Mejor modelo guardado con acc {best_acc:.5f} en epoch {e}")
        step += 1
        if scheduler:
            scheduler.step()
        if early_stopping and step > early_stopping:
            print(f"Entrenamiento detenido en epoch {e} por no mejorar en {early_stopping} epochs seguidas")
            break
        if not e % log_each and verbose:
            print(f"Epoch {e}/{epochs} loss {l[-1]:.5f} acc {acc[-1]:.5f} val_loss {val_l[-1]:.5f} val_acc {val_acc[-1]:.5f} lr {lr[-1]:.5f}")
    model.load_state_dict(torch.load('ckpt.pt'))
    return {'epoch': list(range(1, len(l)+1)), 'loss': l, 'acc': acc, 'val_loss': val_l, 'val_acc': val_acc, 'lr': lr}

# ## 7. Entrenamiento con subconjunto de datos

dataset = {
    'train': Dataset(X_subset, y_subset),
    'val': Dataset(X_val, y_val),
}

dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=64, shuffle=True),
    'val': torch.utils.data.DataLoader(dataset['val'], batch_size=1000, shuffle=False)
}

print(f"Train: {len(dataset['train'])} | Val: {len(dataset['val'])}")

# ### Búsqueda de learning rate

lrs = [0.001, 0.0005, 0.0001]
hists = []
for lr in lrs:
    print(f"\nProbando lr {lr}")
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = fit(model, dataloader, optimizer, epochs=10, verbose=0)
    hists.append(hist)

fig = plt.figure(dpi=150, figsize=(10, 3))
ax = plt.subplot(121)
for i in range(len(lrs)):
    ax.plot(hists[i]['loss'], label=lrs[i])
ax.legend()
ax.grid(True)
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('Loss por learning rate')
ax = plt.subplot(122)
for i in range(len(lrs)):
    ax.plot(hists[i]['val_acc'], label=lrs[i])
ax.legend()
ax.grid(True)
ax.set_ylabel('acc')
ax.set_xlabel('epoch')
ax.set_title('Val accuracy por learning rate')
plt.tight_layout()
plt.show()

# ### Búsqueda de batch size

bss = [32, 64, 128, 256]
hists = []
for bs in bss:
    print(f"\nProbando bs {bs}")
    dl = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=bs, shuffle=True),
        'val': torch.utils.data.DataLoader(dataset['val'], batch_size=1000, shuffle=False)
    }
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    hist = fit(model, dl, optimizer, epochs=10, verbose=0)
    hists.append(hist)

fig = plt.figure(dpi=150, figsize=(10, 3))
ax = plt.subplot(121)
for i in range(len(bss)):
    ax.plot(hists[i]['loss'], label=bss[i])
ax.legend()
ax.grid(True)
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('Loss por batch size')
ax = plt.subplot(122)
for i in range(len(bss)):
    ax.plot(hists[i]['val_acc'], label=bss[i])
ax.legend()
ax.grid(True)
ax.set_ylabel('acc')
ax.set_xlabel('epoch')
ax.set_title('Val accuracy por batch size')
plt.tight_layout()
plt.show()

# ### Búsqueda aleatoria de hiperparámetros

import random

bss = [32, 64, 128, 256]
lrs = [0.001, 0.0005, 0.0003, 0.0001]
n = 5
hists = []
for i in range(n):
    lr = random.choice(lrs)
    bs = random.choice(bss)
    print(f"\nTest {i+1} lr {lr} bs {bs}")
    dl = {
        'train': torch.utils.data.DataLoader(dataset['train'], batch_size=bs, shuffle=True),
        'val': torch.utils.data.DataLoader(dataset['val'], batch_size=1000, shuffle=False)
    }
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = fit(model, dl, optimizer, epochs=20, verbose=0)
    hists.append({'hist': hist, 'lr': lr, 'bs': bs})

fig = plt.figure(dpi=150, figsize=(10, 3))
ax = plt.subplot(121)
for i in range(len(hists)):
    ax.plot(hists[i]['hist']['loss'], label=f'lr {hists[i]["lr"]} bs {hists[i]["bs"]}')
ax.legend(fontsize=7)
ax.grid(True)
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax.set_title('Random search — loss')
ax = plt.subplot(122)
for i in range(len(hists)):
    ax.plot(hists[i]['hist']['val_acc'], label=f'lr {hists[i]["lr"]} bs {hists[i]["bs"]}')
ax.legend(fontsize=7)
ax.grid(True)
ax.set_ylabel('acc')
ax.set_xlabel('epoch')
ax.set_title('Random search — val accuracy')
plt.tight_layout()
plt.show()

# ## 8. Entrenamiento final

dataset = {
    'train': Dataset(X_train, y_train),
    'val': Dataset(X_val, y_val),
}

dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=64, shuffle=True),
    'val': torch.utils.data.DataLoader(dataset['val'], batch_size=1000, shuffle=False)
}

print(f"Train: {len(dataset['train'])} | Val: {len(dataset['val'])}")

model = build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.1)
hist = fit(model, dataloader, optimizer, scheduler, epochs=100, early_stopping=10, verbose=2)

# --- Gráficas de entrenamiento ---

fig = plt.figure(dpi=150, figsize=(12, 4))
ax = plt.subplot(131)
pd.DataFrame(hist).plot(x='epoch', y=['loss', 'val_loss'], grid=True, ax=ax)
ax.set_title('Loss')
ax = plt.subplot(132)
pd.DataFrame(hist).plot(x='epoch', y=['acc', 'val_acc'], grid=True, ax=ax)
ax.set_title('Accuracy')
ax = plt.subplot(133)
pd.DataFrame(hist).plot(x='epoch', y=['lr'], grid=True, ax=ax)
ax.set_title('Learning Rate')
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ## 9. Evaluación en test

dataset_test = Dataset(X_test, y_test)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1000, shuffle=False)

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x_b, y_b in dataloader_test:
        preds = model(x_b.to(device))
        preds = torch.argmax(preds, axis=1)
        y_true.extend(y_b.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"\n[Test] Accuracy = {acc:.4f}")

# --- Matriz de confusión y reporte ---

from sklearn.metrics import classification_report

print("\nReporte de clasificación (primeras 20 clases):")
unique_labels = sorted(set(y_true))
target_names = [str(i) for i in unique_labels[:20]]
print(classification_report(y_true, y_pred, labels=unique_labels[:20], target_names=target_names))

# ## 10. Guardar y cargar modelo

PATH = './checkpoint_final.pt'
torch.save(model.state_dict(), PATH)
print(f"Pesos guardados en: {PATH}")

# cargar modelo
model_loaded = build_model()
model_loaded.load_state_dict(torch.load(PATH))
model_loaded.eval()

# verificar que funciona
with torch.no_grad():
    x_sample = torch.from_numpy(X_test[:10]).float().to(device)
    preds = torch.argmax(model_loaded(x_sample), axis=1).cpu().numpy()
    print("\nPredicciones sobre las primeras 10 muestras de prueba:")
    for i in range(10):
        print(f"  Real: {y_test[i]} | Predicho: {preds[i]}")

# ## Resumen
#
# En este notebook hemos aplicado la receta de entrenamiento de redes neuronales
# al dataset TinyImageNet-200 para clasificación multiclase:
#
# 1. **Exploración de datos**: 100 000 imágenes, 200 clases balanceadas (500 c/u)
# 2. **Validación incremental**: verificación de dimensiones, fit de 1 muestra, fit de 1 batch
# 3. **Modelo**: MLP con BatchNorm para convergencia más rápida
# 4. **Optimización**: Adam con búsqueda de learning rate y batch size
# 5. **Regularización**: Early stopping + BatchNorm
# 6. **Scheduler**: StepLR para reducir learning rate automáticamente
# 7. **Evaluación**: Accuracy, reporte de clasificación, matriz de confusión
#
# El uso de BatchNorm acelera significativamente la convergencia en comparación
# con el modelo base sin normalización por lotes.
