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

# %%
import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import SelfTrainingClassifier

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)

RAIZ = Path.cwd()
DIR_FIGS = RAIZ / "figs"
DIR_FEATURES = RAIZ / "features"
for d in (DIR_FIGS, DIR_FEATURES):
    d.mkdir(exist_ok=True)

SEMILLA = 42
DATASET_ROOT = Path.home() / ".cache" / "speech_commands"
NUM_WORKERS = min(12, os.cpu_count() or 4)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.2})


def guardar_fig(nombre):
    ruta = DIR_FIGS / f"{nombre}.png"
    plt.savefig(ruta, bbox_inches="tight")
    print(f"Figura guardada en: {ruta}")
    return ruta


RES = []


def reg(bloque, parametro, valor, metrica, resultado, extra=None):
    RES.append({"bloque": bloque, "parametro": parametro, "valor": valor,
                "metrica": metrica, "resultado": float(resultado),
                "extra": extra})

# %%
def cargar_wav(ruta):
    sr, data = wavfile.read(str(ruta))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128.0) / 128.0
    else:
        data = data.astype(np.float32)
    return torch.from_numpy(np.atleast_2d(data)), sr


def obtener_rutas(subset):
    ds = SPEECHCOMMANDS(root=str(DATASET_ROOT), download=True, subset=subset)
    return list(ds._walker)


def n_estadisticos(nombre):
    return {"media_std": 2, "completo": 4}[nombre]


class FeaturesAudio(Dataset):
    def __init__(self, rutas, n_mfcc, n_mels, usar_deltas, estadisticos):
        self.rutas = rutas
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.usar_deltas = usar_deltas
        self.estadisticos = estadisticos
        self._mfcc = None

    def __len__(self):
        return len(self.rutas)

    def _mfcc_transform(self, sr):
        if self._mfcc is None:
            melkwargs = {"n_fft": 1024, "hop_length": 256, "n_mels": self.n_mels, "power": 2.0}
            self._mfcc = torchaudio.transforms.MFCC(
                sample_rate=sr, n_mfcc=self.n_mfcc, melkwargs=melkwargs)
        return self._mfcc

    def _estadisticos_vector(self, t):
        piezas = [t.mean(dim=-1), t.std(dim=-1)]
        if self.estadisticos == "completo":
            q = torch.tensor([0.1, 0.9], device=t.device)
            cuant = torch.quantile(t, q, dim=-1)
            piezas.extend([cuant[0], cuant[1]])
        return torch.cat(piezas, dim=-1).reshape(-1)

    def __getitem__(self, idx):
        ruta = self.rutas[idx]
        wav, sr = cargar_wav(ruta)
        label = Path(ruta).parent.name
        wav = wav.mean(dim=0, keepdim=True)
        objetivo = 16000
        if wav.shape[-1] > objetivo:
            wav = wav[..., :objetivo]
        elif wav.shape[-1] < objetivo:
            wav = F.pad(wav, (0, objetivo - wav.shape[-1]))
        wav = wav / (wav.abs().max() + 1e-9)

        mfcc = self._mfcc_transform(sr)(wav)
        bloques = [self._estadisticos_vector(mfcc)]
        if self.usar_deltas:
            d1 = torchaudio.functional.compute_deltas(mfcc)
            d2 = torchaudio.functional.compute_deltas(d1)
            bloques.append(self._estadisticos_vector(d1))
            bloques.append(self._estadisticos_vector(d2))
        return torch.cat(bloques, dim=0).float(), label, idx


def clave_config(n_mfcc, n_mels, usar_deltas, estadisticos):
    return f"mfcc{n_mfcc}_mels{n_mels}_d{int(usar_deltas)}_{estadisticos}"


def extraer(subset, n_mfcc, n_mels, usar_deltas, estadisticos, batch_size=128):
    if (n_mfcc == 40 and n_mels == 64 and usar_deltas
            and estadisticos == "media_std"):
        base = DIR_FEATURES / f"{subset}_40.npz"
        if base.exists():
            d = np.load(base, allow_pickle=True)
            return d["X"], d["y"], d["idx"]
    t0 = time.time()
    rutas = obtener_rutas(subset)
    ds = FeaturesAudio(rutas, n_mfcc, n_mels, usar_deltas, estadisticos)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    Xp, yp, ip = [], [], []
    for feats, labels, idxs in loader:
        Xp.append(feats.numpy().astype(np.float32))
        yp.extend(list(labels))
        ip.append(idxs.numpy())
    X = np.concatenate(Xp, axis=0)
    y = np.array(yp)
    idx = np.concatenate(ip, axis=0)
    print(f"[extraido] {subset} {clave_config(n_mfcc, n_mels, usar_deltas, estadisticos)}: "
          f"X={X.shape} en {time.time() - t0:.1f}s")
    return X, y, idx


def cargar_config(n_mfcc=40, n_mels=64, usar_deltas=True, estadisticos="media_std"):
    Xtr, ytr, _ = extraer("training", n_mfcc, n_mels, usar_deltas, estadisticos)
    Xva, yva, _ = extraer("validation", n_mfcc, n_mels, usar_deltas, estadisticos)
    Xte, yte, _ = extraer("testing", n_mfcc, n_mels, usar_deltas, estadisticos)
    Xp = np.vstack([Xtr, Xva])
    yp = np.concatenate([ytr, yva])
    return Xp, yp, Xte, yte

# %%
def entrenar_logreg(X, y, C=1.0, max_iter=500, class_weight=None, semilla=SEMILLA):
    clf = LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight,
                             random_state=semilla)
    clf.fit(X, y)
    return clf


def techo(Xp, yp, Xt, yt, C=1.0, max_iter=500, class_weight=None, escalar=True):
    if escalar:
        sc = StandardScaler().fit(Xp)
        Xp, Xt = sc.transform(Xp), sc.transform(Xt)
    return entrenar_logreg(Xp, yp, C=C, max_iter=max_iter,
                           class_weight=class_weight).score(Xt, yt)


def accuracy_budget(Xp, yp, Xt, yt, n_etiquetas, semilla=100):
    sc = StandardScaler().fit(Xp)
    Xp, Xt = sc.transform(Xp), sc.transform(Xt)
    idx = np.random.default_rng(semilla).choice(len(Xp), size=n_etiquetas, replace=False)
    return entrenar_logreg(Xp[idx], yp[idx]).score(Xt, yt)


def score_predicho(clf, X):
    pred = clf.predict(X)
    return pred

# %%
print("=== E1: hiperparametros de las caracteristicas ===")
RES_E1 = []
for n_mfcc in [13, 20, 40, 60]:
    Xp, yp, Xt, yt = cargar_config(n_mfcc=n_mfcc)
    n = Xp.shape[1]
    acc_techo = techo(Xp, yp, Xt, yt, max_iter=500)
    base_fija = accuracy_budget(Xp, yp, Xt, yt, 1000)
    reg("E1_features", "n_mfcc", n_mfcc, "techo", acc_techo, n)
    reg("E1_features", "n_mfcc", n_mfcc, "acc_1000", base_fija, n)
    RES_E1.append((n_mfcc, n, acc_techo, base_fija))
    print(f"  n_mfcc={n_mfcc:<3} n={n:<4} techo={acc_techo:.4f} acc_1000={base_fija:.4f}")
    del Xp, yp, Xt, yt

RES_E1_MELS = []
for n_mels in [40, 64, 128]:
    Xp, yp, Xt, yt = cargar_config(n_mfcc=40, n_mels=n_mels)
    acc_techo = techo(Xp, yp, Xt, yt, max_iter=500)
    reg("E1_features", "n_mels", n_mels, "techo", acc_techo, Xp.shape[1])
    RES_E1_MELS.append((n_mels, acc_techo))
    print(f"  n_mels={n_mels:<4} techo={acc_techo:.4f}")
    del Xp, yp, Xt, yt

RES_E1_VAR = {}
for nombre, kw in [("sin_deltas", {"usar_deltas": False}),
                   ("con_deltas", {"usar_deltas": True}),
                   ("estad_completo", {"estadisticos": "completo"})]:
    Xp, yp, Xt, yt = cargar_config(n_mfcc=40, n_mels=64, **kw)
    acc_techo = techo(Xp, yp, Xt, yt, max_iter=500)
    reg("E1_features", nombre, nombre, "techo", acc_techo, Xp.shape[1])
    RES_E1_VAR[nombre] = (acc_techo, Xp.shape[1])
    print(f"  {nombre:<15} n={Xp.shape[1]:<4} techo={acc_techo:.4f}")
    del Xp, yp, Xt, yt

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ns = [r[1] for r in RES_E1]
t = [r[2] for r in RES_E1]
f = [r[3] for r in RES_E1]
axes[0].plot(ns, t, "o-", label="Techo (todas las etiquetas)")
axes[0].plot(ns, f, "s--", label="1000 etiquetas")
for x, y in zip(ns, t):
    axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
axes[0].set_xlabel("Dimension n")
axes[0].set_ylabel("Exactitud en prueba")
axes[0].set_title("Efecto de N_MFCC")
axes[0].legend()

me = [r[0] for r in RES_E1_MELS]
mt = [r[1] for r in RES_E1_MELS]
axes[1].bar([str(x) for x in me], mt, color="#4c72b0")
for i, y in enumerate(mt):
    axes[1].text(i, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)
axes[1].set_xlabel("n_mels")
axes[1].set_ylabel("Exactitud en prueba")
axes[1].set_title("Efecto de n_mels (techo)")
axes[1].set_ylim(0, max(mt) * 1.2)

vn = list(RES_E1_VAR.keys())
vv = [RES_E1_VAR[k][0] for k in vn]
axes[2].bar(vn, vv, color=["#dd8452", "#55a868", "#8172b3"])
for i, y in enumerate(vv):
    axes[2].text(i, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)
axes[2].set_ylabel("Exactitud en prueba")
axes[2].set_title("Deltas y estadisticos (techo)")
axes[2].set_ylim(0, max(vv) * 1.2)

plt.suptitle("Punto 3 - Hiperparametros de las caracteristicas", fontsize=14)
plt.tight_layout()
guardar_fig("p3_hp_01_features")
plt.show()

best_mfcc = max(RES_E1, key=lambda r: r[2])[0]
print(f"Mejor N_MFCC por techo: {best_mfcc}")

# %%
Xp, yp, Xt, yt = cargar_config(n_mfcc=best_mfcc)
sc = StandardScaler().fit(Xp)
Xp_esc, Xt_esc = sc.transform(Xp), sc.transform(Xt)
print("Config base para E2-E4:", Xp_esc.shape)

print("=== E2: hiperparametros del clasificador ===")
RES_C = []
for C in [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]:
    a = entrenar_logreg(Xp_esc, yp, C=C, max_iter=500).score(Xt_esc, yt)
    reg("E2_clasificador", "C", C, "techo", a)
    RES_C.append((C, a))
    print(f"  C={C:<7} techo={a:.4f}")

RES_MI = []
for mi in [200, 500, 1000]:
    a = entrenar_logreg(Xp_esc, yp, C=1.0, max_iter=mi).score(Xt_esc, yt)
    reg("E2_clasificador", "max_iter", mi, "techo", a)
    RES_MI.append((mi, a))
    print(f"  max_iter={mi:<5} techo={a:.4f}")

RES_ESC = {}
for nombre, kw in [("con_escalado", {"escalar": True}),
                   ("sin_escalado", {"escalar": False}),
                   ("balanced", {"escalar": True, "class_weight": "balanced"})]:
    a = techo(Xp, yp, Xt, yt, max_iter=500, **kw)
    reg("E2_clasificador", nombre, nombre, "techo", a)
    RES_ESC[nombre] = a
    print(f"  {nombre:<14} techo={a:.4f}")

best_C = max(RES_C, key=lambda r: r[1])[0]
print(f"Mejor C: {best_C}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
cs = [r[0] for r in RES_C]
ca = [r[1] for r in RES_C]
axes[0].semilogx(cs, ca, "o-", color="#c44e52")
for x, y in zip(cs, ca):
    axes[0].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
axes[0].set_xlabel("C (escala log)")
axes[0].set_ylabel("Exactitud en prueba")
axes[0].set_title("Regularizacion C (techo)")

mis = [str(r[0]) for r in RES_MI]
mia = [r[1] for r in RES_MI]
axes[1].bar(mis, mia, color="#4c72b0")
for i, y in enumerate(mia):
    axes[1].text(i, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)
axes[1].set_xlabel("max_iter")
axes[1].set_ylabel("Exactitud en prueba")
axes[1].set_title("Iteraciones del optimizador")
axes[1].set_ylim(0, max(mia) * 1.2)

en = list(RES_ESC.keys())
ev = [RES_ESC[k] for k in en]
axes[2].bar(en, ev, color=["#55a868", "#dd8452", "#8172b3"])
for i, y in enumerate(ev):
    axes[2].text(i, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)
axes[2].set_ylabel("Exactitud en prueba")
axes[2].set_title("Escalado y class_weight (techo)")
axes[2].set_ylim(0, max(ev) * 1.2)

plt.suptitle("Punto 3 - Hiperparametros del clasificador", fontsize=14)
plt.tight_layout()
guardar_fig("p3_hp_02_clasificador")
plt.show()

# %%
print("=== E3: aprendizaje semisupervisado ===")


def propagar(X, y, Xt_eval, k, semilla=SEMILLA):
    km = MiniBatchKMeans(n_clusters=k, init="k-means++", n_init=10,
                         batch_size=2048, random_state=semilla).fit(X)
    lab, dist = km.labels_, km.transform(X)
    yp_ = np.empty(len(X), dtype=object)
    reps = 0
    for c in range(k):
        m = np.where(lab == c)[0]
        if len(m) == 0:
            continue
        r = m[np.argmin(dist[m, c])]
        yp_[m] = y[r]
        reps += 1
    mask = np.array([isinstance(v, str) for v in yp_])
    clf = entrenar_logreg(X[mask], yp_[mask].astype(str), C=best_C)
    return clf.score(Xt_eval, yt), reps


RES_K = []
for k in [35, 100, 500, 1000, 2000]:
    t0 = time.time()
    a, reps = propagar(Xp_esc, yp, Xt_esc, k)
    reg("E3_semi", "kmeans_k", k, "propagacion", a, reps)
    RES_K.append((k, a, reps))
    print(f"  k={k:<5} reps={reps:<5} acc={a:.4f} ({time.time() - t0:.0f}s)")

# %%
print("  Autoentrenamiento: umbral y semilla")
RES_THR = []
for thr in [0.5, 0.7, 0.9, 0.99]:
    idx_seed = np.random.default_rng(2024).choice(len(Xp_esc), size=1000, replace=False)
    y_semi = np.full(len(Xp_esc), -1, dtype=object)
    y_semi[idx_seed] = yp[idx_seed]
    st = SelfTrainingClassifier(
        LogisticRegression(C=best_C, max_iter=500, random_state=SEMILLA),
        threshold=thr, max_iter=3)
    st.fit(Xp_esc, y_semi)
    pseudo = int(np.sum(st.labeled_iter_ > 0))
    a = st.score(Xt_esc, yt)
    reg("E3_semi", "self_threshold", thr, "acc", a, pseudo)
    RES_THR.append((thr, a, pseudo))
    print(f"  threshold={thr:<5} acc={a:.4f} pseudo={pseudo}")

RES_SEM = []
for nseed in [500, 1000, 2000, 5000]:
    idx_seed = np.random.default_rng(2024).choice(len(Xp_esc), size=nseed, replace=False)
    solo = entrenar_logreg(Xp_esc[idx_seed], yp[idx_seed], C=best_C).score(Xt_esc, yt)
    y_semi = np.full(len(Xp_esc), -1, dtype=object)
    y_semi[idx_seed] = yp[idx_seed]
    st = SelfTrainingClassifier(
        LogisticRegression(C=best_C, max_iter=500, random_state=SEMILLA),
        threshold=0.9, max_iter=3)
    st.fit(Xp_esc, y_semi)
    con = st.score(Xt_esc, yt)
    reg("E3_semi", "self_nseed", nseed, "solo_supervisado", solo)
    reg("E3_semi", "self_nseed", nseed, "autotraining", con)
    RES_SEM.append((nseed, solo, con))
    print(f"  nseed={nseed:<5} solo={solo:.4f} auto={con:.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ks = [r[0] for r in RES_K]
ka = [r[1] for r in RES_K]
kr = [r[2] for r in RES_K]
ax0 = axes[0]
ax0.plot(ks, ka, "o-", color="#c44e52", label="Exactitud propagacion")
ax0.set_xscale("log")
ax0.set_xlabel("k (numero de clusters, escala log)")
ax0.set_ylabel("Exactitud en prueba", color="#c44e52")
for x, y in zip(ks, ka):
    ax0.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -14), fontsize=8)
ax1 = ax0.twinx()
ax1.plot(ks, kr, "s--", color="#4c72b0", label="Etiquetas usadas")
ax1.set_ylabel("Etiquetas usadas (repres.)", color="#4c72b0")
ax0.set_title("Propagacion por clusters vs k")

ts = [r[0] for r in RES_THR]
ta = [r[1] for r in RES_THR]
axes[1].plot(ts, ta, "o-", color="#55a868")
for x, y in zip(ts, ta):
    axes[1].annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=8)
axes[1].set_xlabel("Umbral de confianza del autoentrenamiento")
axes[1].set_ylabel("Exactitud en prueba")
axes[1].set_title("Autoentrenamiento: umbral (1000 etiquetas)")

plt.suptitle("Punto 3 - Hiperparametros del aprendizaje semisupervisado", fontsize=14)
plt.tight_layout()
guardar_fig("p3_hp_03_semisupervisado")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 4.8))
sn = [r[0] for r in RES_SEM]
ss = [r[1] for r in RES_SEM]
sa = [r[2] for r in RES_SEM]
ax.plot(sn, ss, "s--", color="#4c72b0", label="Solo supervisado")
ax.plot(sn, sa, "o-", color="#c44e52", label="Autoentrenamiento")
ax.set_xscale("log")
ax.set_xlabel("Etiquetas semilla (escala log)")
ax.set_ylabel("Exactitud en prueba")
ax.set_title("Autoentrenamiento vs. supervisado segun presupuesto")
ax.legend()
guardar_fig("p3_hp_04_autotrain_semilla")
plt.show()

# %%
print("=== E4: aprendizaje activo ===")


def seleccionar(probas, lote, estrategia):
    if estrategia == "margin":
        dos = np.sort(probas, axis=1)[:, -2:]
        prioridad = -(dos[:, 1] - dos[:, 0])
    elif estrategia == "least_confidence":
        prioridad = -probas.max(axis=1)
    elif estrategia == "entropy":
        prioridad = -(probas * np.log(probas + 1e-12)).sum(axis=1)
    else:
        prioridad = np.zeros(len(probas))
    return np.argsort(prioridad)[::-1][:lote]


def activo(idx_inicial, lote, total, estrategia, semilla):
    rng = np.random.default_rng(semilla)
    etiq = list(idx_inicial)
    libres = np.setdiff1d(np.arange(len(Xp_esc)), np.array(etiq))
    while len(etiq) < total:
        clf = entrenar_logreg(Xp_esc[etiq], yp[etiq], C=best_C)
        if estrategia == "aleatorio":
            eleg = rng.choice(libres, size=lote, replace=False)
        else:
            probas = clf.predict_proba(Xp_esc[libres])
            eleg = libres[seleccionar(probas, lote, estrategia)]
        etiq.extend(np.asarray(eleg).tolist())
        libres = np.setdiff1d(libres, np.asarray(eleg))
    clf = entrenar_logreg(Xp_esc[etiq], yp[etiq], C=best_C)
    return clf.score(Xt_esc, yt)


BUDGET = 2500
INICIAL = 100
RES_LOTE = {}
for lote in [100, 500]:
    for nombre, modo in [("activo", "margin"), ("aleatorio", "aleatorio")]:
        vals = []
        for s in range(2):
            rng = np.random.default_rng(50 + s)
            idx_ini = rng.choice(len(Xp_esc), size=INICIAL, replace=False)
            vals.append(activo(idx_ini, lote, BUDGET, modo, 300 + s))
        prom = float(np.mean(vals))
        RES_LOTE[(lote, nombre)] = prom
        reg("E4_activo", "lote", lote, nombre, prom)
        print(f"  lote={lote:<4} {nombre:<9} acc={prom:.4f}")

RES_EST = {}
for est in ["margin", "least_confidence", "entropy"]:
    vals = []
    for s in range(2):
        rng = np.random.default_rng(50 + s)
        idx_ini = rng.choice(len(Xp_esc), size=INICIAL, replace=False)
        vals.append(activo(idx_ini, 250, BUDGET, est, 300 + s))
    prom = float(np.mean(vals))
    RES_EST[est] = prom
    reg("E4_activo", "estrategia", est, "acc_final", prom)
    print(f"  estrategia={est:<16} acc={prom:.4f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
lotes = [100, 500]
ax_act = [RES_LOTE[(l, "activo")] for l in lotes]
ax_ale = [RES_LOTE[(l, "aleatorio")] for l in lotes]
x = np.arange(len(lotes))
ancho = 0.35
axes[0].bar(x - ancho / 2, ax_act, ancho, label="Activo (margin)", color="#c44e52")
axes[0].bar(x + ancho / 2, ax_ale, ancho, label="Aleatorio", color="#4c72b0")
for i, (a, b) in enumerate(zip(ax_act, ax_ale)):
    axes[0].text(i - ancho / 2, a + 0.003, f"{a:.3f}", ha="center", fontsize=8)
    axes[0].text(i + ancho / 2, b + 0.003, f"{b:.3f}", ha="center", fontsize=8)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"lote={l}" for l in lotes])
axes[0].set_ylabel("Exactitud en prueba")
axes[0].set_title(f"Tamano de lote (presupuesto {BUDGET})")
axes[0].legend()

en = list(RES_EST.keys())
ev = [RES_EST[k] for k in en]
axes[1].bar(en, ev, color=["#c44e52", "#dd8452", "#8172b3"])
for i, y in enumerate(ev):
    axes[1].text(i, y + 0.003, f"{y:.3f}", ha="center", fontsize=9)
axes[1].set_ylabel("Exactitud en prueba")
axes[1].set_title(f"Estrategias de consulta (lote 250, {BUDGET})")
axes[1].set_ylim(0, max(ev) * 1.2)

plt.suptitle("Punto 3 - Hiperparametros del aprendizaje activo", fontsize=14)
plt.tight_layout()
guardar_fig("p3_hp_05_activo")
plt.show()

# %%
print("=== Resumen y guardado ===")
df = pd.DataFrame(RES)
df.to_csv(RAIZ / "hiper_resultados.csv", index=False)
print(df.to_string(index=False))

mejores_lote = {l: RES_LOTE[(l, "activo")] for l in [100, 500]}
best_lote = max(mejores_lote, key=mejores_lote.get)
mejores = {
    "best_n_mfcc": int(best_mfcc),
    "best_C": float(best_C),
    "best_self_threshold": float(max(RES_THR, key=lambda r: r[1])[0]),
    "best_lote": int(best_lote),
    "best_estrategia": max(RES_EST, key=RES_EST.get),
    "techo": float(techo(Xp, yp, Xt, yt, C=best_C, max_iter=500)),
    "acc_activa": float(mejores_lote[best_lote]),
}
with open(RAIZ / "hiper_resumen.json", "w") as fh:
    json.dump(mejores, fh, indent=2)
print("Mejores hiperparametros:", json.dumps(mejores, indent=2))
