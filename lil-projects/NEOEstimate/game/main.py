import pygame
import os
from assets.reg_model import RegModel
from game.data_src import PoolRoundSource
from game.api import CpuGuesser, ModelPredictor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = PoolRoundSource(os.path.join(ROOT, "assets", "pool.csv"))
cpu = CpuGuesser(ModelPredictor())

NROUND = 0
SCORE_J = 0
SCORE_C = 0
results = []

def run(n: int = 7) -> None:
    for _ in range(n):
        r = src.next()
        print(f"diam {r.view.diam_min_m:.1f} - {r.view.diam_max_m:.1f} m"
            f"CPU {cpu.guess(r):.2f} vs real {r.target:.2f}")

def reveal(query: dict):
    ...


def check_winner(player_guess: float, cpu_guess: float):
    ...


def update():
    ...


def draw():
    ...


def loop():
    ...


if __name__ == "__main__":
    run()
