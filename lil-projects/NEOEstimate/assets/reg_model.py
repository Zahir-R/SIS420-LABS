import json, os, numpy as np, pandas as pd, torch, torch.nn as nn

DIR = os.path.dirname(os.path.abspath(__file__))

class RegModel:
    def __init__(self, stats_path=None, weights_path=None):
        stats_path = stats_path or os.path.join(DIR, "mag_stats.json")
        weights_path = weights_path or os.path.join(DIR, "mag_model.pt")
        with open(stats_path) as f:
            self.cfg = json.load(f)
        self.model = nn.Sequential(
            nn.Linear(self.cfg["n_in"], self.cfg["hidden"]),
            nn.ReLU(),
            nn.Linear(self.cfg["hidden"], 1),
        )
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        self.num = {d["col"]: d for d in self.cfg["numeric"]}
        self.cat = {d["col"]: d for d in self.cfg["categorical"]}

    def vector(self, q):
        vec = []
        for d in self.cfg["numeric"]:
            v = q.get(d["col"])
            v = d["median"] if pd.isna(v) else v
            v = np.log1p(v) if d["log1p"] else v
            vec.append((v - d["mu"]) / d["sigma"])
        for d in self.cfg["categorical"]:
            cats = d["cats"]
            one = [0.0] * (len(cats) + 1)
            val = q.get(d["col"])
            one[cats.index(val) if val in cats else len(cats)] = 1.0
            vec.extend(one)
        for c in self.cfg["binary"]:
            v = q.get(c, 0)
            vec.append(0.0 if pd.isna(v) or not v else 1.0)
        return np.array(vec, dtype=np.float32)

    def predict(self, q):
        with torch.no_grad():
            return float(self.model(torch.tensor(np.array([self.vector(q)], dtype=np.float32))).item())

    def predict_many(self, qs):
        X = np.array([self.vector(q) for q in qs], dtype=np.float32)
        with torch.no_grad():
            return self.model(torch.tensor(X)).numpy().ravel().tolist()
