import unittest
import torch
from memory import Entry, ReplayMemory, replay_loss


def entry(sid, task):
    return Entry(sid, task, task, torch.ones(4, 1024), torch.tensor([0, 4, 2]),
                 torch.arange(5), torch.zeros(2, 5))


class MemoryTest(unittest.TestCase):
    def test_capacity_and_no_resurrection(self):
        memory = ReplayMemory(capacity=1)
        memory.update(0, [entry('a', 0), entry('b', 0)])
        self.assertEqual(len(memory.entries), 1)
        memory.update(1, [entry('c', 1)])
        with self.assertRaises(ValueError):
            memory.update(2, [entry('a', 2)])
        with self.assertRaises(ValueError):
            memory.update(2, [entry('d', 0)])
        self.assertEqual(memory.last_task, 1)
        self.assertGreater(memory.tensor_bytes, 4*1024*4)

    def test_masked_future_columns_do_not_poison_mse(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(2, 6))
            def forward(self, x, seq, task):
                return self.values.masked_fill(torch.arange(6)[None] == 5, -torch.inf)[None]
        model = Toy()
        loss = replay_loss(model, entry('a', 0), 'cpu')
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.values.grad).all())


if __name__ == '__main__':
    unittest.main()
