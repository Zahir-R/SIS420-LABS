from typing import Protocol
import numpy as np
from game.domain import Round
class Predictor(Protocol):
    def predict(self, query: dict[str, float | str]) -> float:
        ...


class ModelPredictor(Predictor):
    def __init__(self) -> None:
        from assets.reg_model import RegModel
        self._rm = RegModel()

    def predict(self, query):
        return self._rm.predict(query)

class Guesser(Protocol):
    def guess(self, rnd: Round) -> float:
        ...

class CpuGuesser(Guesser):
    def __init__(self, predictor: Predictor, sigma: float=0.0) -> None:
        self._p, self._sigma = predictor, sigma

    def guess(self, rnd: Round) -> float:
        return self._p.predict(rnd.features) + np.random.normal(0.0, self._sigma)


DIFFICULTIES = {"GAMBLER": 0.0, "HARD": 0.4, "NORMAL": 0.9, "EASY": 1.6}

