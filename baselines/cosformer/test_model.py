import unittest
import torch
from model import COSFormer


class ModelTest(unittest.TestCase):
    def test_expansion_replay_and_causal_prediction(self):
        torch.manual_seed(4)
        # Synthetic vectors only for structural testing, never formal training.
        m = COSFormer(torch.randn(12, 32), input_dim=8)
        m.add_task([4, 5])
        x = torch.randn(40, 8)
        ids = torch.tensor([[0, 4, 5]])
        old = m.projection.experts[0].weight.detach().clone()
        m.add_task([6, 7])
        logits = m(x, ids, task=0)
        loss = torch.nn.functional.cross_entropy(logits[0], torch.tensor([4, 5, 2]))
        loss.backward()
        self.assertIsNone(m.projection.experts[0].weight.grad)
        self.assertIsNotNone(m.projection.experts[1].weight.grad)
        torch.testing.assert_close(old, m.projection.experts[0].weight)
        self.assertTrue(torch.isneginf(logits[..., 8:]).all())
        m.eval()
        first = m(x, torch.tensor([[0, 4, 5]]), task=None)
        other = m(x, torch.tensor([[0, 6, 7]]), task=None)
        torch.testing.assert_close(first[:, 0, :8], other[:, 0, :8])
        scores = m.class_scores(x, {0: [0, 4, 5, 2], 1: [0, 6, 7, 2]})
        self.assertEqual(set(scores), {0, 1})
        self.assertTrue(all(torch.isfinite(v) for v in scores.values()))


if __name__ == '__main__':
    unittest.main()
