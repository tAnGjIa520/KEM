import unittest
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from gpu_gmm import CudaGaussianMixture
from latent_memory import sample_gmm


@unittest.skipUnless(torch.cuda.is_available(), 'Requires CUDA')
class KernelParityTest(unittest.TestCase):
    def test_sampler_parity(self):
        state = {'weights': np.array([.3, .7]), 'means': np.arange(6.).reshape(2, 3),
                 'covariances': np.stack([np.eye(3), np.eye(3)*2])}
        cpu = sample_gmm(state, 100, np.random.default_rng(2))
        gpu = sample_gmm(state, 100, np.random.default_rng(2), 'cuda')
        np.testing.assert_allclose(cpu, gpu, atol=1e-12, rtol=1e-12)

    def test_full_uni_dimension_statistics(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=(64, 1024))
        r = rng.dirichlet([1, 1], len(x))
        from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
        expected = _estimate_gaussian_parameters(x, r, 1e-6, 'full')
        model = CudaGaussianMixture(n_components=2)
        model.chunk_size = 19
        actual = model._statistics(x, r)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(a, b, atol=1e-11, rtol=1e-10)

    def test_em_bic_and_chunk_parity(self):
        rng = np.random.default_rng(4)
        x = np.concatenate([rng.normal(-3, 1, (90, 7)), rng.normal(3, .7, (110, 7))])
        kwargs = dict(n_components=2, random_state=5, covariance_type='full', tol=1e-5)
        cpu = GaussianMixture(**kwargs).fit(x)
        gpu = CudaGaussianMixture(**kwargs)
        gpu.chunk_size = 31
        gpu.fit(x)
        for name in ('weights_', 'means_', 'covariances_', 'precisions_cholesky_'):
            np.testing.assert_allclose(getattr(cpu, name), getattr(gpu, name), rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(cpu.score_samples(x), gpu.score_samples(x), rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(cpu.bic(x), gpu.bic(x), places=7)
        self.assertEqual(cpu.n_iter_, gpu.n_iter_)


if __name__ == '__main__':
    unittest.main()
