from dataclasses import dataclass

TIE_EPS = 0.05

@dataclass(frozen=True)
class RoundView:
    diam_min_m: float
    diam_max_m: float
    orbiting: str
    orbit_class_type: str

@dataclass(frozen=True)
class Round:
    features: dict[str, float | str]
    view: RoundView
    target: float

@dataclass(frozen=True)
class Outcome:
    winner: str
    err_player: float
    err_cpu: float


def decide_outcome(player_guess: float, cpu_guess: float, target: float) -> Outcome:
    err_player, err_cpu = abs(player_guess - target), abs(cpu_guess - target)
    if abs(err_player - err_cpu) <= TIE_EPS:
        return Outcome("tie", err_player, err_cpu)
    return Outcome("player" if err_player < err_cpu else "cpu", err_player, err_cpu)


class Bank:
    def __init__(self, start: int=1000) -> None:
        self.balance = start

    def can_afford(self, amount: int) -> bool:
        return 0 < amount <= self.balance

    def place(self, amount: int) -> None:
        if not self.can_afford(amount):
            raise ValueError("Apuesta invalida")
        self.balance -= amount

    def add(self, amount: int) -> None:
        self.balance += amount


class Pot:
    def __init__(self) -> None:
        self.amount = 0

    def win_double(self) -> None:
        self.amount *= 2

    def cash_out(self) -> int:
        v = self.amount
        self.amount = 0
        return v

    def reset(self) -> None:
        self.amount = 0

