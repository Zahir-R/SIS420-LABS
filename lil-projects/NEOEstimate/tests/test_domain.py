import unittest
from game.domain import Bank, Pot, decide_outcome

class BankTest(unittest.TestCase):
    def test_place_bankrupt(self):
        b = Bank(100)
        with self.assertRaises(ValueError):
            b.place(200)
        self.assertEqual(b.balance, 100)

    def test_place_refuse_0(self):
        with self.assertRaises(ValueError):
            Bank(10).place(0)

    def test_never_negative(self):
        b = Bank(10)
        b.place(7)
        b.place(3)
        self.assertEqual(b.balance, 0)
        with self.assertRaises(ValueError):
            b.place(1)

class PotTest(unittest.TestCase):
    def test_double(self):
        p = Pot()
        p.amount = 100
        p.win_double()
        self.assertEqual(p.amount, 200)

    def test_cashout_reset(self):
        p = Pot()
        p.amount = 100
        self.assertEqual(p.cash_out(), 100)
        self.assertEqual(p.amount, 0)

class OutcomeTest(unittest.TestCase):
    def test_tie(self):
        self.assertEqual(decide_outcome(10.0, 10.02, 10.0).winner, "tie")
    def test_player(self):
        self.assertEqual(decide_outcome(10.0, 11.0, 10.0).winner, "player")
    def test_cpu(self):
        self.assertEqual(decide_outcome(11.0, 10.0, 10.0).winner, "cpu")


if __name__ == "__main__":
    unittest.main()
