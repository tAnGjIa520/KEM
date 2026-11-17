"""Bilevel replay patch selection with scalable WSI candidate generation."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch

from .adapters import ModelOutput, ReplayModelAdapter
from .candidates import build_candidate_pool
from .solvers import solve_weights


@dataclass(frozen=True)
class BilevelConfig:
    solver: str = "unrolled"
    proxy_scope: str = "aggregator_head"
    prefilter: str = "maxrand"
    coarse_ratio: float = 2.0
    candidate_max: int = 2048
    num_groups: int = 256
    cluster_dim: int = 64
    cluster_steps: int = 5
    outer_steps: int = 3
    inner_steps: int = 1
    proxy_lr: float = 1e-3
    weight_lr: float = 0.1
    distill_lambda: float = 0.1
    feature_lambda: float = 0.1
    temperature: float = 0.1
    neumann_steps: int = 3
    damping: float = 0.1
    nystrom_rank: int = 5
    bome_eta: float = 0.5
    seed: int = 1

    def __post_init__(self):
        if self.solver not in {"unrolled", "neumann", "bome", "nystrom"}:
            raise ValueError(f"Unsupported bilevel_solver={self.solver!r}")
        if self.prefilter not in {"maxrand", "maxminrand", "none", "hierarchical"}:
            raise ValueError(f"Unsupported bilevel_prefilter={self.prefilter!r}")
        if self.coarse_ratio < 1 or self.candidate_max < 1 or self.num_groups < 1:
            raise ValueError("Invalid bilevel candidate configuration")
        if self.outer_steps < 1 or self.inner_steps < 1 or self.neumann_steps < 0:
            raise ValueError("Bilevel iteration counts must be positive")
        if self.proxy_lr <= 0 or self.weight_lr <= 0 or self.temperature <= 0:
            raise ValueError("Bilevel learning rates and temperature must be positive")
        if self.damping <= 0 or self.nystrom_rank < 1:
            raise ValueError("Bilevel damping and Nyström rank must be positive")


class BilevelReplaySelector:
    """Select replay patches while keeping the trained MIL model immutable."""

    def __init__(self, model_name: str, config: BilevelConfig):
        self.adapter = ReplayModelAdapter(model_name)
        self.config = config
        self.proxy_model = None

    def select(self, model, features, label, seen_classes, budget, scores=None):
        started = time.perf_counter()
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        features = features.detach().to(device)
        label = label.detach().to(device).long().view(-1)
        budget = min(max(int(budget), 1), features.shape[0])

        teacher_started = time.perf_counter()
        with torch.no_grad():
            teacher_raw = self.adapter.forward(model, features, seen_classes)
            teacher = ModelOutput(
                logits=teacher_raw.logits.detach(),
                features=teacher_raw.features.detach(),
            )
        teacher_seconds = time.perf_counter() - teacher_started

        candidate_started = time.perf_counter()
        pool = build_candidate_pool(features, scores, budget, self.config)
        candidate_seconds = time.perf_counter() - candidate_started
        if pool.source_indices.numel() <= budget and pool.mode == "patch":
            diagnostics = self._diagnostics(pool, budget, None, started, device)
            diagnostics.update({
                "bilevel/teacher_time_seconds": teacher_seconds,
                "bilevel/candidate_time_seconds": candidate_seconds,
            })
            return pool.source_indices[:budget].cpu(), diagnostics

        proxy = self._reset_proxy(model)
        solve_started = time.perf_counter()
        weights, solver_diagnostics = solve_weights(
            self.adapter, proxy, pool.features, label, seen_classes,
            pool.optimization_budget, teacher, self.config,
        )
        solve_seconds = time.perf_counter() - solve_started
        selected = pool.finalize(weights, budget)
        diagnostics = self._diagnostics(pool, budget, weights, started, device)
        diagnostics.update(solver_diagnostics)
        diagnostics.update({
            "bilevel/teacher_time_seconds": teacher_seconds,
            "bilevel/candidate_time_seconds": candidate_seconds,
            "bilevel/solver_time_seconds": solve_seconds,
        })
        return selected, diagnostics

    def _reset_proxy(self, model):
        if self.proxy_model is None:
            self.proxy_model = copy.deepcopy(model)
        else:
            self.proxy_model.load_state_dict(model.state_dict())
        self.proxy_model.to(next(model.parameters()).device)
        self.proxy_model.eval()
        self.adapter.configure_proxy_parameters(self.proxy_model, self.config.proxy_scope)
        return self.proxy_model

    def _diagnostics(self, pool, selected_count, weights, started, device):
        values = {
            "bilevel/original_patch_count": pool.original_patch_count,
            "bilevel/candidate_patch_count": pool.source_indices.numel(),
            "bilevel/optimization_variable_count": pool.features.shape[0],
            "bilevel/selected_patch_count": selected_count,
            "bilevel/group_mode": float(pool.mode == "group"),
            "bilevel/selection_time_seconds": time.perf_counter() - started,
        }
        if weights is not None:
            values["bilevel/weight_mean"] = float(weights.mean().cpu())
            values["bilevel/weight_std"] = float(weights.std(unbiased=False).cpu())
        if device.type == "cuda":
            values["bilevel/peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
        return values
