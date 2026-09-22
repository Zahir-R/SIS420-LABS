# %% [markdown]
# # Clasificación de sonidos bioacústicos (aves) con red neuronal
#
# Dataset: BioDCASE 2026 Task 3 (TinyML) — 11 clases (10 especies europeas + ruido de fondo),
# clips WAV de 3 s a 24 kHz, 200 muestras por clase en `Train/` y 50 en `Validation/`.

# %% [markdown]
# ## 0. Configuración

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.io import wavfile
from torchaudio.transforms import MFCC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from tqdm import tqdm
import time

SEED = 67
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando", device)

TRAIN_DIR = "Train"
TEST_DIR = "Validation"
RES = Path("resultados")
RES.mkdir(exist_ok=True)

SR = 24000
N_MFCC = 26
EPOCHS = 80
LR = 1e-3
WD = 1e-4
PATIENCE = 10
BATCH = 32

# %%
filas = []
for split, carpeta in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
    for ruta in sorted(Path(carpeta).rglob("*.wav")):
        filas.append({"path": str(ruta), "split": split, "clase": ruta.parent.name})
df = pd.DataFrame(filas)
clases = sorted(df["clase"].unique())
df["codigo"] = df["clase"].map({c: i for i, c in enumerate(clases)})
print("Muestras:", len(df), ", Clases:", len(clases))
df.head()

# %%
print(df.groupby(["split", "clase"]).size().to_string())
print()
sr0, w0 = wavfile.read(df["path"].iloc[0])
print(f"Tasa de muestreo: {sr0} Hz, Duración del primer clip: {len(w0) / sr0:.2f} s")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
df[df["split"] == "train"]["clase"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], title="Muestras por clase (Train)")
axes[0].tick_params(axis="x", rotation=45)
t = np.arange(len(w0)) / sr0
axes[1].plot(t, w0, lw=0.4)
axes[1].set_title("Onda del primer clip (3 s)")
axes[1].set_xlabel("Tiempo (s)")
plt.tight_layout()
plt.savefig(RES / "eda.png", bbox_inches="tight")
plt.show()

# %%
mfcc_transform = MFCC(sample_rate=SR, n_mfcc=N_MFCC,
                      melkwargs={"n_mels": 64, "n_fft": 1024})

def get_features(ruta):
    _, wav = wavfile.read(ruta)
    wave = torch.from_numpy(wav.astype(np.float32) / 32768.0)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    mfcc = mfcc_transform(wave)[0]
    stats = []
    for coef in mfcc:
        stats.append([coef.mean(), coef.std(), coef.min(), coef.max()])
    return np.asarray(stats, dtype=np.float32).ravel()

X = np.stack([get_features(p) for p in df["path"]])
y = df["codigo"].to_numpy(dtype=np.int64)
print("X:", X.shape, "| y:", y.shape)
print(X[0].reshape(26,4))

# %%
idx_train = np.where(df["split"] == "train")[0]
idx_test = np.where(df["split"] == "test")[0]
X_tr, X_te = X[idx_train], X[idx_test]
y_tr, y_te = y[idx_train], y[idx_test]

media = X_tr.mean(axis=0)
desv = X_tr.std(axis=0)
X_tr = (X_tr - media) / desv
X_te = (X_te - media) / desv

Xtrain, Xval, ytrain, yval = train_test_split(
    X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
print(f"train: {Xtrain.shape}, val: {Xval.shape}, test: {X_te.shape}")

# %%
class DatasetAudio(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def creat_dataloaders(batch_size=BATCH):
    train_dl = DataLoader(DatasetAudio(Xtrain, ytrain), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(DatasetAudio(Xval, yval), batch_size=batch_size)
    test_dl = DataLoader(DatasetAudio(X_te, y_te), batch_size=batch_size)
    return train_dl, val_dl, test_dl


# %%
def build_model(d_in, h1, h2, d_out, dropout):
    return nn.Sequential(
        nn.Linear(d_in, h1), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(h2, d_out),
    )

model = build_model(X.shape[1], 128, 64, len(clases), 0.3).to(device)
out = model(torch.randn((64, X.shape[1])).to(device))
print("Entrada: (64,", X.shape[1], ") -> Salida:", tuple(out.shape))

# %%
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
train_dl, val_dl, test_dl = creat_dataloaders()
xb, yb = next(iter(train_dl))
xb, yb = xb.to(device), yb.to(device)

opt.zero_grad()
loss_uno = criterion(model(xb[:1]), yb[:1])
loss_uno.backward()
opt.step()

opt.zero_grad()
loss_batch = criterion(model(xb), yb)
loss_batch.backward()
opt.step()

print("Costo 1 muestra:", round(loss_uno.item(), 4))
print("Costo 1 batch:", round(loss_batch.item(), 4))

# %%
def fit(model, tl, vl, epochs=EPOCHS, lr=LR, wd=WD, patience=PATIENCE):
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    hist = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    best = {"acc": -1.0, "epoch": 0, "state": None}
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        s_loss, s_hit, n = 0.0, 0, 0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            s_loss += loss.item() * len(xb)
            s_hit += (model(xb).argmax(dim=1) == yb).sum().item()
            n += len(xb)
        hist["loss"].append(s_loss / n)
        hist["acc"].append(s_hit / n)
        model.eval()
        with torch.no_grad():
            v_loss, v_hit, vn = 0.0, 0, 0
            for xb, yb in vl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                v_loss += criterion(logits, yb).item() * len(xb)
                v_hit += (logits.argmax(dim=1) == yb).sum().item()
                vn += len(xb)
        hist["val_loss"].append(v_loss / vn)
        hist["val_acc"].append(v_hit / vn)
        scheduler.step(hist["val_loss"][-1])
        if hist["val_acc"][-1] > best["acc"]:
            best = {"acc": hist["val_acc"][-1], "epoch": epoch,
                     "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if epoch - best["epoch"] >= patience:
            print(f"ES {epoch} , best {best['epoch']}")
            break
    model.load_state_dict(best["state"])
    return model, hist, best, time.time() - t0

def evaluar(model, dl):
    model.eval()
    y_true, y_pred, proba = [], [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            prob = torch.softmax(logits, dim=-1)
            y_true += yb.tolist()
            y_pred += logits.argmax(dim=1).tolist()
            proba += prob.cpu().tolist()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    proba = np.asarray(proba)
    met = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
    }
    return met, y_true, y_pred, proba

def plotear(hist, titulo, ax, metric="loss"):
    ax.plot(hist[metric], label="train")
    ax.plot(hist["val_" + metric], label="val")
    ax.set_title(titulo)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(alpha=0.3)

# %%
grid = []
for (h1, h2) in [(128, 64), (256, 128), (512, 256)]:
    for dropout in [0.0, 0.3]:
        for lr in [1e-3, 5e-4]:
            grid.append({"h1": h1, "h2": h2, "dropout": dropout, "lr": lr})

filas = []
fig, axes = plt.subplots(4, 3, figsize=(16, 15))
for i, cfg in enumerate(grid):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tl, vl, tst = creat_dataloaders()
    m = build_model(X.shape[1], cfg["h1"], cfg["h2"], len(clases), cfg["dropout"]).to(device)
    m, hist, best, tiempo = fit(m, tl, vl, lr=cfg["lr"])
    met_val, *_ = evaluar(m, vl)
    met_te, *_ = evaluar(m, tst)
    filas.append({
        "h1": cfg["h1"], "h2": cfg["h2"], "dropout": cfg["dropout"], "lr": cfg["lr"],
        "best_epoch": best["epoch"], "tiempo_s": round(tiempo, 1),
        "val_acc": round(met_val["accuracy"], 4), "val_f1": round(met_val["f1_macro"], 4),
        "test_acc": round(met_te["accuracy"], 4), "test_f1": round(met_te["f1_macro"], 4),
        "test_auc_ovr": round(met_te["roc_auc_macro"], 4),
    })
    ax = axes[i // 3, i % 3]
    plotear(hist, f"{cfg['h1']}-{cfg['h2']} d{cfg['dropout']} lr{cfg['lr']} | acc val={met_val['accuracy']:.3f}", ax, "loss")

plt.tight_layout()
plt.savefig(RES / "curvas_costo_sweep.png", bbox_inches="tight")
plt.show()

tabla = pd.DataFrame(filas).sort_values("test_acc", ascending=False)
tabla.to_csv(RES / "metricas_sweep.csv", index=False)
print(tabla)

# %%
bst_cfg = tabla.iloc[0]
print("Mejor:\n", bst_cfg.to_string())

torch.manual_seed(SEED)
np.random.seed(SEED)
tl_f, vl_f, tst_f = creat_dataloaders(batch_size=64)
m_final = build_model(X.shape[1], int(bst_cfg["h1"]), int(bst_cfg["h2"]),
                      len(clases), float(bst_cfg["dropout"])).to(device)
m_final, hist_final, bst_final, time_f = fit(
    m_final, tl_f, vl_f, epochs=100, lr=float(bst_cfg["lr"]), patience=12)
torch.save(m_final.state_dict(), RES / "mejor_checkpoint.pt")
print(f"Train: {time_f:.1f} s, mejor epoch {bst_final['epoch']} , val_acc {bst_final['acc']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
plotear(hist_final, "Costo", axes[0], "loss")
plotear(hist_final, "Precision (acc)", axes[1], "acc")
plt.tight_layout()
plt.savefig(RES / "curvas_final.png", bbox_inches="tight")
plt.show()

# %%
met_final, y_true, y_pred, proba = evaluar(m_final, tst_f)
print("Métricas")
for k, v in met_final.items():
    print(f"  {k:15s} {v:.4f}")
print()
print(classification_report(y_true, y_pred, target_names=clases))

# %%
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clases, yticklabels=clases)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Real")
plt.xlabel("Pred")
plt.title("CM")
plt.tight_layout()
plt.savefig(RES / "matriz_confusion.png", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(8, 7))
for i, cl in enumerate(clases):
    binario = (y_true == i).astype(int)
    fpr, tpr, _ = roc_curve(binario, proba[:, i])
    plt.plot(fpr, tpr, lw=1.4,
             label=f"{cl} (AUC {roc_auc_score(binario, proba[:, i]):.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC one-vs-rest")
plt.legend(fontsize=7, ncol=2, loc="lower right")
plt.tight_layout()
plt.savefig(RES / "roc_auc.png", bbox_inches="tight")
plt.show()

# %%
pred_df = pd.DataFrame({
    "archivo": df.loc[idx_test, "path"].values,
    "real": [clases[c] for c in y_true],
    "pred": [clases[c] for c in y_pred],
    "acierto": y_true == y_pred,
})
pred_df.to_csv(RES / "predicciones.csv", index=False)
print(f"Predicciones {pred_df.shape[0]} filas, aciertos: {pred_df['acierto'].mean():.2%}")
print(pred_df.iloc[4*50:4*50 + 50])

# %%
