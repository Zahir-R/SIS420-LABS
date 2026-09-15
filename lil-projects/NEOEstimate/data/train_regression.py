import json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    DIR = os.getcwd()
os.chdir(DIR)

torch.manual_seed(67)

TARGET = "magnitude"
EXPLICIT_DROP = ["name", "date", "hazardous"]
TEST_FRAC = 0.2
RANDOM_SEED = 67
HIDDEN = 24
EPOCHS = 60
BATCH = 256
LR = 3e-2
SKEW_THRESH = 1.0
MIN_FREQ = 10

approaches = pd.read_csv("neo.csv")
objects = pd.read_csv("neo_objects.csv")

for c in objects.columns:
    if objects[c].dtype == object and c not in ("name", "orbit_class_type"):
        coerced = pd.to_numeric(objects[c], errors="coerce")
        if coerced.notna().any():
            objects[c] = coerced

best = approaches.loc[approaches.groupby("name")["miss_km"].idxmin()]
df = objects.merge(best, on="name", suffixes=("", "_app"))
df = df.drop(columns=[c for c in df.columns if c.endswith("_app")])
print("objetos únicos:", len(df))

categorical_cols = [c for c in df.columns
                    if not pd.api.types.is_numeric_dtype(df[c])
                    and c not in EXPLICIT_DROP
                    and df[c].notna().any()]
binary_cols = [c for c in df.columns
               if df[c].dtype in (bool, "int64") and df[c].nunique() == 2
               and c not in EXPLICIT_DROP and c != TARGET]
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in EXPLICIT_DROP and c != TARGET and c not in binary_cols]
print("numéricas:", num_cols)
print("categóricas:", categorical_cols)
print("binarias:", binary_cols)

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

log1p = {c: bool(df[c].skew() > SKEW_THRESH) for c in num_cols}
print("log1p:", {c: log1p[c] for c in num_cols})

cats, freqs = {}, {}
for c in categorical_cols:
    vc = df[c].value_counts()
    top = vc[vc >= MIN_FREQ].index.tolist()
    cats[c] = top
    freqs[c] = {k: int(vc[k]) for k in top}
    freqs[c]["other"] = int((vc < MIN_FREQ).sum())

idx = np.arange(len(df))
itr, ite = train_test_split(idx, test_size=TEST_FRAC, random_state=RANDOM_SEED)
y = df[TARGET].to_numpy(dtype=np.float32)
Xlog = df[num_cols].apply(lambda s: np.log1p(s) if log1p[s.name] else s)

def build_X(sel_nums, sel_cats, sel_bins, idx_arr):
    rows = []
    for i in idx_arr:
        vec = [float(Xlog[c].iloc[i]) for c in sel_nums]
        for c in sel_cats:
            val = df[c].iloc[i]
            one = [0.0] * (len(cats[c]) + 1)
            if val in cats[c]:
                one[cats[c].index(val)] = 1.0
            else:
                one[-1] = 1.0
            vec.extend(one)
        for c in sel_bins:
            vec.append(float(bool(df[c].iloc[i])))
        rows.append(vec)
    return np.array(rows, dtype=np.float32)

def train_model(Xtr_a, Xva_a):
    model = nn.Sequential(nn.Linear(Xtr_a.shape[1], HIDDEN),
                          nn.ReLU(),
                          nn.Linear(HIDDEN, 1))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    Xtr_t, Ytr_t = torch.tensor(Xtr_a), torch.tensor(y[itr])
    Xva_t = torch.tensor(Xva_a)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr_t, Ytr_t),
        batch_size=BATCH, shuffle=True)
    for ep in range(EPOCHS):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb).squeeze(1), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                p = model(Xva_t).numpy().ravel()
            print(f"  ep {ep+1:3d} | train MSE {tot/n:.4f} | val MAE {np.mean(np.abs(p - y[ite])):.3f}")
    model.eval()
    with torch.no_grad():
        pred = model(Xva_t).numpy().ravel()
    mse = mean_squared_error(y[ite], pred)
    met = {"mse": float(mse), "rmse": float(np.sqrt(mse)),
           "mae": float(mean_absolute_error(y[ite], pred)),
           "r2": float(r2_score(y[ite], pred)),
           "n_in": Xtr_a.shape[1]}
    return model, met, pred

approach_nums = [c for c in num_cols if c in set(approaches.columns)]
orbital_nums = [c for c in num_cols if c not in set(approaches.columns)]
groups = [
    ("G1 base",         approach_nums, [], []),
    ("G2 +orbital",     approach_nums + orbital_nums, [], []),
    ("G3 +categorias",  approach_nums + orbital_nums, categorical_cols, []),
    ("G4 +is_sentry",   approach_nums + orbital_nums, categorical_cols, binary_cols),
]

fits = {}
for gname, sn, sc, sb in groups:
    print("\n" + gname)
    Xtr_a, Xva_a = build_X(sn, sc, sb, itr), build_X(sn, sc, sb, ite)
    n_num = len(sn)
    mu = Xtr_a[:, :n_num].mean(0)
    sigma = Xtr_a[:, :n_num].std(0) + 1e-8
    Xtr_a[:, :n_num] = (Xtr_a[:, :n_num] - mu) / sigma
    Xva_a[:, :n_num] = (Xva_a[:, :n_num] - mu) / sigma
    model, met, pred = train_model(Xtr_a, Xva_a)
    fits[gname] = {"model": model, "met": met, "pred": pred,
                   "nums": sn, "cats": sc, "bins": sb,
                   "mu": mu, "sigma": sigma}
    print(gname, "->", {k: round(v, 4) for k, v in met.items() if isinstance(v, float)})

winner = min(fits, key=lambda g: fits[g]["met"]["mae"])
best = fits[winner]
print(f"\nGANADOR: {winner} | MAE {best['met']['mae']:.3f} | R2 {best['met']['r2']:.3f}")

cfg = {
    "target": {"col": TARGET, "min": float(df[TARGET].min()), "max": float(df[TARGET].max())},
    "numeric": [],
    "categorical": [],
    "binary": best["bins"],
    "n_in": best["met"]["n_in"],
    "hidden": HIDDEN,
    "winner": winner,
    "groups": {g: fits[g]["met"] for g in fits},
}
for i, c in enumerate(best["nums"]):
    cfg["numeric"].append({
        "col": c, "log1p": log1p[c],
        "mu": float(best["mu"][i]), "sigma": float(best["sigma"][i]),
        "median": float(df[c].median()),
        "min": float(df[c].min()), "max": float(df[c].max()),
    })
for c in best["cats"]:
    cfg["categorical"].append({"col": c, "cats": cats[c], "freq": freqs[c]})

torch.save(best["model"].state_dict(), "mag_model.pt")
with open("mag_stats.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("guardado: mag_model.pt + mag_stats.json")

pool_cols = list(best["nums"])
if "orbiting" not in pool_cols and "orbiting" in df.columns:
    pool_cols.append("orbiting")
pool_cols = pool_cols + [c for c in best["cats"] if c not in pool_cols]
pool_cols = pool_cols + [c for c in best["bins"] if c not in pool_cols]
pool_cols.append(TARGET)
pool = df.loc[ite, pool_cols].reset_index(drop=True)
pool.to_csv("pool.csv", index=False)
print(f"pool.csv: {pool.shape[0]} filas de test x {pool.shape[1]} columnas")

import matplotlib.pyplot as plt
yva = y[ite]
pred = best["pred"]
plt.figure(figsize=(5, 5))
plt.scatter(yva, pred, s=4, alpha=0.4)
lim = [min(yva.min(), pred.min()), max(yva.max(), pred.max())]
plt.plot(lim, lim, "r--")
plt.xlabel("magnitude real"); plt.ylabel("magnitude predicha")
plt.title(f"Ganador {winner} | MAE {best['met']['mae']:.3f} | R2 {best['met']['r2']:.3f}")
plt.tight_layout()
plt.show()
