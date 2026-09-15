import csv
from typing import Protocol
import pandas as pd
from game.domain import Round, RoundView

class RoundSource(Protocol):
    def next(self) -> Round:
        ...

    def reset(self) -> None:
        ...

class PoolRoundSource:
    def __init__(self, csv_path: str) -> None:
        self._pool = pd.read_csv(csv_path)
        self._i = 0

    def next(self) -> Round:
        r = self._pool.iloc[self._i]
        self._i = (self._i + 1) % len(self._pool)
        features = {k: (v if isinstance(v, str) else float(v))
                    for k, v in r.items() if k != "magnitude"}
        view = RoundView(float(r["diam_min_m"]), float(r["diam_max_m"]),
                         str(r["orbiting"],), str(r["orbit_class_type"]))
        return Round(features=features, view=view, target=float(r["magnitude"]))

    def reset(self) -> None:
        self._i = 0
