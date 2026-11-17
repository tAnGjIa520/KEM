"""Parameter-only generative memory; see PROTOCOL.md for adaptation choices."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import warnings
import time

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class FitConfig:
    attention_fraction: float = 0.8
    feature_components: tuple = (8, 16, 24)
    count_components: tuple = (1, 2, 3, 4, 5)
    reg_covar: float = 1e-6
    max_iter: int = 100
    tol: float = 1e-3
    n_init: int = 1
    seed: int = 1
    max_sample_patches: int | None = None
    fit_backend: str = 'sklearn'


def select_attention(features, attention, fraction=0.8):
    x = np.asarray(features, dtype=np.float64)
    a = np.asarray(attention, dtype=np.float64)
    if x.ndim != 2 or not len(x) or x.shape[1] == 0 or a.shape != (len(x),):
        raise ValueError("Expected nonempty [patch, feature] and [patch] attention")
    if not np.isfinite(x).all() or not np.isfinite(a).all():
        raise ValueError("Nonfinite current training features/attention")
    if not 0 < fraction <= 1:
        raise ValueError("Invalid attention fraction")
    indices = np.argsort(-a, kind="stable")[:int(np.ceil(fraction * len(x)))]
    return x[indices]


def fit_bic(values, candidates, config, seed):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite matrix")
    candidates = tuple(candidates)
    if not candidates or len(set(candidates)) != len(candidates) or min(candidates) < 1:
        raise ValueError("Invalid component grid")
    if len(x) < max(2, max(candidates)):
        raise ValueError("Insufficient samples for the specified BIC grid; do not silently reduce it")
    best, report = None, []
    for components in sorted(candidates):
        model_class = GaussianMixture
        if config.fit_backend == 'cuda' and x.shape[1] > 1:
            from gpu_gmm import CudaGaussianMixture
            model_class = CudaGaussianMixture
        elif config.fit_backend not in ('sklearn', 'cuda'):
            raise ValueError('Unknown GMM backend')
        started = time.time()
        print(json.dumps({'phase': 'gmm_candidate_start', 'samples': len(x),
                          'dimension': x.shape[1], 'components': components,
                          'backend': model_class.__name__, 'time': started}), flush=True)
        model = model_class(n_components=components, covariance_type="full",
            reg_covar=config.reg_covar, max_iter=config.max_iter, tol=config.tol,
            n_init=config.n_init, init_params="kmeans", random_state=(seed + components) % (2**32))
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x)
        bic = float(model.bic(x))
        report.append({"components": components, "bic": bic, "converged": bool(model.converged_),
                       "iterations": int(model.n_iter_), "warnings": [str(w.message) for w in messages]})
        print(json.dumps({'phase': 'gmm_candidate_done', **report[-1],
                          'seconds': time.time() - started}), flush=True)
        if not model.converged_ or not np.isfinite(bic):
            raise RuntimeError(f"BIC search not accepted: {report}")
        if best is None or bic < best[0]:
            best = (bic, model)
    model = best[1]
    state = {"weights": model.weights_.copy(), "means": model.means_.copy(),
             "covariances": model.covariances_.copy()}
    return state, report


def sample_gmm(state, count, rng, backend='sklearn'):
    weights, means, cov = (state[k] for k in ("weights", "means", "covariances"))
    components = rng.choice(len(weights), size=count, p=weights)
    out = np.empty((count, means.shape[1]), dtype=np.float64)
    for component in np.unique(components):
        where = np.flatnonzero(components == component)
        noise = rng.normal(size=(len(where), means.shape[1]))
        if backend == 'cuda' and means.shape[1] > 1:
            import torch
            chol = torch.linalg.cholesky(torch.as_tensor(cov[component], device='cuda'))
            center = torch.as_tensor(means[component], device='cuda')
            for start in range(0, len(where), 8192):
                z = torch.as_tensor(noise[start:start+8192], device='cuda')
                out[where[start:start+8192]] = (z @ chol.T + center).cpu().numpy()
        else:
            chol = np.linalg.cholesky(cov[component])
            out[where] = noise @ chol.T + means[component]
    if not np.isfinite(out).all():
        raise RuntimeError("Nonfinite synthetic samples")
    return out


class LatentMemory:
    def __init__(self, config=FitConfig()):
        self.config = config
        self.families = []
        self.feature_dim = None
        self.rng = np.random.default_rng(config.seed)

    def add_episode(self, episode, bags):
        """bags yields (global_label, frozen_features, current_model_attention).

        Caller must provide ONLY the current episode's training partition.
        Data are local temporaries; fitting failure does not partially mutate
        the persisted memory or consume the replay RNG.
        """
        if any(f["episode"] == int(episode) for f in self.families):
            raise ValueError("Episode already fitted")
        features, counts = {}, {}
        dimension = self.feature_dim
        for label, x, attention in bags:
            selected = select_attention(x, attention, self.config.attention_fraction)
            if dimension is None:
                dimension = selected.shape[1]
            if selected.shape[1] != dimension:
                raise ValueError("Frozen feature dimension changed")
            label = int(label)
            features.setdefault(label, []).append(selected)
            counts.setdefault(label, []).append(len(x))
        if not features:
            raise ValueError("Empty episode")
        fitted = []
        for label in sorted(features):
            seed = int(np.random.SeedSequence([self.config.seed, int(episode), label]).generate_state(1)[0])
            feature_state, feature_report = fit_bic(np.concatenate(features[label]),
                self.config.feature_components, self.config, seed)
            count_state, count_report = fit_bic(np.asarray(counts[label])[:, None],
                self.config.count_components, self.config, seed + 100)
            fitted.append({"episode": int(episode), "label": label, "slides": len(counts[label]),
                           "selected_patches": sum(len(x) for x in features[label]),
                           "feature": feature_state, "count": count_state,
                           "feature_bic": feature_report, "count_bic": count_report})
        self.families.extend(fitted)
        self.feature_dim = dimension

    def sample(self):
        if not self.families:
            raise ValueError("No past distributions")
        weights = np.asarray([f["slides"] for f in self.families], dtype=float)
        family = self.families[int(self.rng.choice(len(weights), p=weights / weights.sum()))]
        for _ in range(128):
            raw = float(sample_gmm(family["count"], 1, self.rng)[0, 0])
            count = int(np.floor(raw + 0.5))
            if count > 0:
                break
        else:
            raise RuntimeError("Count model failed to produce a positive length")
        limit = self.config.max_sample_patches
        if limit is not None and count > limit:
            raise RuntimeError(f"Synthetic length {count} exceeds declared safety ceiling {limit}; not clipped")
        x = sample_gmm(family["feature"], count, self.rng, self.config.fit_backend).astype(np.float32)
        if not np.isfinite(x).all():
            raise RuntimeError("Synthetic features overflow float32")
        return x, family["label"], family["episode"]

    @property
    def parameter_bytes(self):
        return sum(array.nbytes for family in self.families for kind in ("feature", "count")
                   for array in family[kind].values())

    def save(self, destination):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        meta = {"schema": "aglr-paper-reimplementation-v1", "config": asdict(self.config),
                "feature_dim": self.feature_dim, "rng": self.rng.bit_generator.state, "families": []}
        arrays = {}
        for i, family in enumerate(self.families):
            meta["families"].append({k: v for k, v in family.items() if k not in ("feature", "count")})
            for kind in ("feature", "count"):
                for key, value in family[kind].items():
                    arrays[f"f{i}_{kind}_{key}"] = value
        np.savez(destination / "parameters.npz", **arrays)
        (destination / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        return sum(p.stat().st_size for p in destination.iterdir())

    @classmethod
    def load(cls, source):
        source = Path(source)
        meta = json.loads((source / "metadata.json").read_text())
        if meta["schema"] != "aglr-paper-reimplementation-v1":
            raise ValueError("Unknown memory schema")
        cfg = meta["config"]
        for key in ("feature_components", "count_components"):
            cfg[key] = tuple(cfg[key])
        memory = cls(FitConfig(**cfg))
        memory.feature_dim = meta["feature_dim"]
        memory.rng.bit_generator.state = meta["rng"]
        with np.load(source / "parameters.npz", allow_pickle=False) as arrays:
            for i, row in enumerate(meta["families"]):
                family = dict(row)
                for kind in ("feature", "count"):
                    family[kind] = {key: arrays[f"f{i}_{kind}_{key}"].copy()
                                    for key in ("weights", "means", "covariances")}
                memory.families.append(family)
        return memory
