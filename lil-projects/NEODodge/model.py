# %%
import json, os, torch
import torch.nn as nn

# %%
try:
    DIR = os.path.dirname(os.path.abspath(__file__)) 
except NameError:
    DIR = os.getcwd()


# %%
class HazardNet(nn.Module):
    def __init__(self, d_in=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )


    @staticmethod
    def legacy(d_in):
        net = HazardNet(d_in=d_in)
        net.net = nn.Sequential(
            nn.Linear(d_in, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        return net


    def forward(self, x):
        return self.net(x)


# %%
class HazardModel:
    def __init__(self, weights="hazard_model_full.pt", stats="stats_full.json"):
        with open(os.path.join(DIR, stats)) as f:
            s = json.load(f)
        self.mu, self.sigma = torch.tensor(s["mu"]), torch.tensor(s["sigma"])
        self.features = s.get("features")
        state = torch.load(os.path.join(DIR, weights), map_location="cpu")
        d_in = state["net.0.weight"].shape[1]
        if "net.3.weight" in state:
            self.model = HazardNet(d_in=d_in)
            self.log_idx = ([self.features.index(c)
                             for c in s.get("log_features", [])]
                            if self.features else None)
        else:
            self.model = HazardNet.legacy(d_in)
            self.log_idx = None
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, features):
        x = torch.as_tensor(features, dtype=torch.float32).clone()
        with torch.no_grad():
            if self.log_idx is None:
                x[:, 1:] = torch.log1p(x[:, 1:]) # Escala logaritmica pq los outliers alteran mucho el resultado
            else:
                x[:, self.log_idx] = torch.log1p(x[:, self.log_idx])
            x = (x - self.mu) / self.sigma
            p = self.model(x.float())
        return p.flatten()

# %%
