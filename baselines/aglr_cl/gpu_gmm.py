"""CUDA float64 kernels for sklearn's full-covariance GaussianMixture.

Keep sklearn initialization, convergence, BIC and parameter serialization.
Only sufficient statistics and Gaussian log-density kernels move to CUDA.
All observations contribute; chunking is not subsampling.
"""
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


class CudaGaussianMixture(GaussianMixture):
    chunk_size = 8192

    def _statistics(self, X, resp):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA GMM requires CUDA')
        k, d = resp.shape[1], X.shape[1]
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        nt = torch.as_tensor(nk, device='cuda', dtype=torch.float64)
        sums = torch.zeros((k, d), device='cuda', dtype=torch.float64)
        for start in range(0, len(X), self.chunk_size):
            x = torch.as_tensor(X[start:start+self.chunk_size], device='cuda', dtype=torch.float64)
            r = torch.as_tensor(resp[start:start+self.chunk_size], device='cuda', dtype=torch.float64)
            sums.add_(r.T @ x)
        means = sums / nt[:, None]
        covariance = torch.zeros((k, d, d), device='cuda', dtype=torch.float64)
        for start in range(0, len(X), self.chunk_size):
            x = torch.as_tensor(X[start:start+self.chunk_size], device='cuda', dtype=torch.float64)
            r = torch.as_tensor(resp[start:start+self.chunk_size], device='cuda', dtype=torch.float64)
            for j in range(k):
                delta = x - means[j]
                covariance[j].add_((delta * r[:, j:j+1]).T @ delta)
        covariance /= nt[:, None, None]
        covariance.diagonal(dim1=-2, dim2=-1).add_(self.reg_covar)
        return nk, means.cpu().numpy(), covariance.cpu().numpy()

    def _initialize(self, X, resp):
        if self.covariance_type != 'full':
            raise ValueError('Only full covariance is supported')
        if any(v is not None for v in (self.weights_init, self.means_init, self.precisions_init)):
            raise ValueError('Explicit parameter initialization is unsupported')
        weights, self.means_, self.covariances_ = self._statistics(X, resp)
        self.weights_ = weights / len(X)
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'full')

    def _m_step(self, X, log_resp):
        self.weights_, self.means_, self.covariances_ = self._statistics(X, np.exp(log_resp))
        self.weights_ /= self.weights_.sum()
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_, 'full')

    def _estimate_log_prob(self, X):
        means = torch.as_tensor(self.means_, device='cuda', dtype=torch.float64)
        precision = torch.as_tensor(self.precisions_cholesky_, device='cuda', dtype=torch.float64)
        logdet = precision.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        out = np.empty((len(X), len(means)), dtype=np.float64)
        constant = X.shape[1] * np.log(2 * np.pi)
        for start in range(0, len(X), self.chunk_size):
            x = torch.as_tensor(X[start:start+self.chunk_size], device='cuda', dtype=torch.float64)
            values = torch.empty((len(x), len(means)), device='cuda', dtype=torch.float64)
            for j in range(len(means)):
                z = (x - means[j]) @ precision[j]
                values[:, j] = -.5 * (constant + z.square().sum(-1)) + logdet[j]
            out[start:start+len(x)] = values.cpu().numpy()
        return out
