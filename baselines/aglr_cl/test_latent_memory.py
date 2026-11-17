import tempfile
import unittest
from pathlib import Path
import numpy as np
from latent_memory import FitConfig, LatentMemory, select_attention


class MemoryTest(unittest.TestCase):
    def bags(self, label=0):
        rng = np.random.default_rng(9)
        return [(label, rng.normal(size=(n, 3)), rng.random(n)) for n in (10, 12, 14, 16, 18, 20)]

    def config(self, **changes):
        return FitConfig(feature_components=(1, 2), count_components=(1, 2), max_iter=300, **changes)

    def test_stable_top_attention(self):
        x = np.arange(15).reshape(5, 3)
        np.testing.assert_array_equal(select_attention(x, [1, 1, 1, 1, 0]), x[:4])

    def test_replay_roundtrip_and_no_exemplars(self):
        memory = LatentMemory(self.config())
        memory.add_episode(0, self.bags(3))
        memory.add_episode(1, self.bags(8))
        self.assertEqual(memory.feature_dim, 3)
        self.assertEqual(len(memory.families), 2)
        self.assertGreater(memory.parameter_bytes, 0)
        self.assertEqual(set(vars(memory)), {"config", "families", "feature_dim", "rng"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory"
            self.assertGreater(memory.save(path), memory.parameter_bytes)
            restored = LatentMemory.load(path)
            for _ in range(5):
                x, label, episode = memory.sample()
                y, other_label, other_episode = restored.sample()
                np.testing.assert_array_equal(x, y)
                self.assertEqual((label, episode), (other_label, other_episode))
                self.assertEqual(x.shape[1], 3)
                self.assertGreater(len(x), 0)
            with self.assertRaises(FileExistsError):
                memory.save(path)

    def test_duplicate_episode_and_dimension_rejected(self):
        memory = LatentMemory(self.config())
        memory.add_episode(0, self.bags())
        with self.assertRaises(ValueError):
            memory.add_episode(0, self.bags())
        with self.assertRaises(ValueError):
            memory.add_episode(1, [(1, np.ones((10, 4)), np.ones(10))])
        self.assertEqual(len(memory.families), 1)

    def test_insufficient_samples_do_not_change_grid_or_state(self):
        memory = LatentMemory(FitConfig())
        with self.assertRaises(ValueError):
            memory.add_episode(0, [(0, np.ones((4, 3)), np.ones(4))])
        self.assertEqual(memory.families, [])

    def test_nonfinite_and_safety_ceiling(self):
        with self.assertRaises(ValueError):
            select_attention([[float("nan")]], [1])
        memory = LatentMemory(self.config(max_sample_patches=1))
        memory.add_episode(0, self.bags())
        with self.assertRaises(RuntimeError):
            memory.sample()


if __name__ == "__main__":
    unittest.main()
