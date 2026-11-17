"""Hypergradient solvers for bilevel replay patch selection."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F


def solve_weights(adapter, proxy, features, label, seen_classes, budget, teacher, config):
    solver = config.solver.lower()
    if solver == "unrolled":
        return _solve_unrolled(adapter, proxy, features, label, seen_classes, budget, teacher, config)
    if solver == "neumann":
        return _solve_implicit(adapter, proxy, features, label, seen_classes, budget, teacher, config, "neumann")
    if solver == "nystrom":
        return _solve_implicit(adapter, proxy, features, label, seen_classes, budget, teacher, config, "nystrom")
    if solver == "bome":
        return _solve_bome(adapter, proxy, features, label, seen_classes, budget, teacher, config)
    raise ValueError(f"Unknown bilevel solver: {config.solver!r}")


def _base_params(proxy):
    return OrderedDict((name, value) for name, value in proxy.named_parameters() if value.requires_grad)


def _grads(loss, params, create_graph=False, retain_graph=None):
    values = tuple(params.values())
    raw = torch.autograd.grad(
        loss, values, create_graph=create_graph, retain_graph=retain_graph, allow_unused=True
    )
    return tuple(torch.zeros_like(value) if grad is None else grad for value, grad in zip(values, raw))


def _soft_exact_budget(weights, budget, temperature):
    with torch.no_grad():
        low = weights.min() - 20 * temperature
        high = weights.max() + 20 * temperature
        for _ in range(30):
            threshold = (low + high) / 2
            count = torch.sigmoid((weights - threshold) / temperature).sum()
            if count > budget:
                low = threshold
            else:
                high = threshold
        threshold = (low + high) / 2
    return torch.sigmoid((weights - threshold) / temperature)


def _ste_mask(weights, budget, temperature):
    soft = _soft_exact_budget(weights, budget, temperature)
    hard = torch.zeros_like(weights)
    hard[torch.topk(weights, min(budget, weights.numel())).indices] = 1.0
    return hard + soft - soft.detach()


def _inner_loss(adapter, proxy, params, features, label, seen_classes, weights, budget, config):
    output = adapter.forward(
        proxy, features, seen_classes, params=params,
        patch_mask=_ste_mask(weights, budget, config.temperature),
    )
    return F.cross_entropy(output.logits, label)


def _outer_loss(adapter, proxy, params, features, label, seen_classes, weights, budget, teacher, config):
    output = adapter.forward(
        proxy, features, seen_classes, params=params,
        patch_mask=_ste_mask(weights, budget, config.temperature),
    )
    classification = F.cross_entropy(output.logits, label)
    kd = F.kl_div(
        F.log_softmax(output.logits / 2.0, dim=-1),
        F.softmax(teacher.logits / 2.0, dim=-1),
        reduction="batchmean",
    ) * 4.0
    feature = F.mse_loss(output.features, teacher.features)
    return classification + config.distill_lambda * kd + config.feature_lambda * feature


def _solve_unrolled(adapter, proxy, features, label, seen_classes, budget, teacher, config):
    weights = torch.zeros(features.shape[0], device=features.device, requires_grad=True)
    diagnostics = {}
    for outer_step in range(config.outer_steps):
        params = _base_params(proxy)
        inner_loss = None
        for _ in range(config.inner_steps):
            inner_loss = _inner_loss(adapter, proxy, params, features, label, seen_classes,
                                     weights, budget, config)
            grads = _grads(inner_loss, params, create_graph=True)
            params = OrderedDict(
                (name, value - config.proxy_lr * grad)
                for (name, value), grad in zip(params.items(), grads)
            )
        outer_loss = _outer_loss(adapter, proxy, params, features, label, seen_classes,
                                 weights, budget, teacher, config)
        gradient = torch.autograd.grad(outer_loss, weights)[0]
        weights = (weights - config.weight_lr * gradient).detach().requires_grad_(True)
        diagnostics = _step_diagnostics(inner_loss, outer_loss, gradient, outer_step)
    return weights.detach(), diagnostics


def _warm_params(adapter, proxy, features, label, seen_classes, weights, budget, config):
    params = OrderedDict((name, value.detach().clone().requires_grad_(True))
                         for name, value in _base_params(proxy).items())
    for _ in range(config.inner_steps):
        loss = _inner_loss(adapter, proxy, params, features, label, seen_classes,
                           weights, budget, config)
        grads = _grads(loss, params)
        params = OrderedDict(
            (name, (value - config.proxy_lr * grad).detach().requires_grad_(True))
            for (name, value), grad in zip(params.items(), grads)
        )
    return params


def _solve_implicit(adapter, proxy, features, label, seen_classes, budget, teacher, config, method):
    weights = torch.zeros(features.shape[0], device=features.device, requires_grad=True)
    diagnostics = {}
    for outer_step in range(config.outer_steps):
        params = _warm_params(adapter, proxy, features, label, seen_classes,
                              weights, budget, config)
        inner = _inner_loss(adapter, proxy, params, features, label, seen_classes,
                            weights, budget, config)
        inner_grads = _grads(inner, params, create_graph=True, retain_graph=True)
        outer = _outer_loss(adapter, proxy, params, features, label, seen_classes,
                            weights, budget, teacher, config)
        outer_param_grads = _grads(outer, params, retain_graph=True)
        if method == "neumann":
            inverse_hvp = _neumann_inverse_hvp(inner_grads, params, outer_param_grads, config)
        else:
            inverse_hvp = _nystrom_inverse_hvp(inner_grads, params, outer_param_grads, config)
        direct = torch.autograd.grad(outer, weights, retain_graph=True)[0]
        mixed = torch.autograd.grad(
            inner_grads, weights, grad_outputs=inverse_hvp, allow_unused=True
        )[0]
        if mixed is None:
            mixed = torch.zeros_like(weights)
        gradient = direct - mixed
        weights = (weights - config.weight_lr * gradient).detach().requires_grad_(True)
        diagnostics = _step_diagnostics(inner, outer, gradient, outer_step)
    return weights.detach(), diagnostics


def _hvp(inner_grads, params, vectors, retain_graph=True):
    raw = torch.autograd.grad(
        inner_grads, tuple(params.values()), grad_outputs=vectors,
        retain_graph=retain_graph, allow_unused=True,
    )
    return tuple(torch.zeros_like(value) if item is None else item
                 for value, item in zip(params.values(), raw))


def _neumann_inverse_hvp(inner_grads, params, vector, config):
    alpha = config.proxy_lr
    current = tuple(item.detach() for item in vector)
    estimate = tuple(alpha * item for item in current)
    for _ in range(config.neumann_steps):
        product = _hvp(inner_grads, params, current)
        current = tuple(
            value - alpha * (hessian_value + config.damping * value)
            for value, hessian_value in zip(current, product)
        )
        estimate = tuple(total + alpha * value for total, value in zip(estimate, current))
    return estimate


def _flatten(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _unflatten(vector, references):
    result, offset = [], 0
    for reference in references:
        size = reference.numel()
        result.append(vector[offset:offset + size].view_as(reference))
        offset += size
    return tuple(result)


def _nystrom_inverse_hvp(inner_grads, params, vector, config):
    references = tuple(params.values())
    flat_vector = _flatten(vector)
    rank = min(config.nystrom_rank, flat_vector.numel())
    generator = torch.Generator(device=flat_vector.device).manual_seed(config.seed)
    omega = torch.randn(flat_vector.numel(), rank, device=flat_vector.device, generator=generator)
    omega = torch.linalg.qr(omega, mode="reduced").Q
    h_omega = []
    for column in range(rank):
        product = _hvp(inner_grads, params, _unflatten(omega[:, column], references))
        h_omega.append(_flatten(product))
    q = torch.linalg.qr(torch.stack(h_omega, dim=1), mode="reduced").Q
    h_q = []
    for column in range(rank):
        product = _hvp(inner_grads, params, _unflatten(q[:, column], references))
        h_q.append(_flatten(product))
    projected_hessian = q.T @ torch.stack(h_q, dim=1)
    projected_hessian = (projected_hessian + projected_hessian.T) / 2
    coefficients = torch.linalg.solve(
        projected_hessian + config.damping * torch.eye(rank, device=q.device),
        q.T @ flat_vector,
    )
    projected = q @ coefficients
    residual = flat_vector - q @ (q.T @ flat_vector)
    inverse = projected + residual / config.damping
    return _unflatten(inverse, references)


def _solve_bome(adapter, proxy, features, label, seen_classes, budget, teacher, config):
    params = OrderedDict((name, value.detach().clone().requires_grad_(True))
                         for name, value in _base_params(proxy).items())
    weights = torch.zeros(features.shape[0], device=features.device, requires_grad=True)
    diagnostics = {}
    for outer_step in range(config.outer_steps):
        reference = params
        for _ in range(config.inner_steps):
            loss = _inner_loss(adapter, proxy, reference, features, label, seen_classes,
                               weights.detach(), budget, config)
            grads = _grads(loss, reference)
            reference = OrderedDict(
                (name, (value - config.proxy_lr * grad).detach().requires_grad_(True))
                for (name, value), grad in zip(reference.items(), grads)
            )
        inner_current = _inner_loss(adapter, proxy, params, features, label, seen_classes,
                                    weights, budget, config)
        inner_reference = _inner_loss(adapter, proxy, reference, features, label, seen_classes,
                                      weights, budget, config)
        constraint = inner_current - inner_reference
        outer = _outer_loss(adapter, proxy, params, features, label, seen_classes,
                            weights, budget, teacher, config)
        variables = tuple(params.values()) + (weights,)
        outer_grads = torch.autograd.grad(outer, variables, retain_graph=True, allow_unused=True)
        constraint_grads = torch.autograd.grad(constraint, variables, allow_unused=True)
        outer_grads = tuple(torch.zeros_like(v) if g is None else g for v, g in zip(variables, outer_grads))
        constraint_grads = tuple(torch.zeros_like(v) if g is None else g for v, g in zip(variables, constraint_grads))
        norm_sq = sum((g.square().sum() for g in constraint_grads)).clamp_min(1e-12)
        dot = sum((a * b).sum() for a, b in zip(outer_grads, constraint_grads))
        multiplier = torch.clamp((config.bome_eta * norm_sq - dot) / norm_sq, min=0.0).detach()
        combined = tuple(a + multiplier * b for a, b in zip(outer_grads, constraint_grads))
        params = OrderedDict(
            (name, (value - config.proxy_lr * grad).detach().requires_grad_(True))
            for (name, value), grad in zip(params.items(), combined[:-1])
        )
        gradient = combined[-1]
        weights = (weights - config.weight_lr * gradient).detach().requires_grad_(True)
        diagnostics = _step_diagnostics(inner_current, outer, gradient, outer_step)
        diagnostics["bilevel/bome_constraint"] = float(constraint.detach().cpu())
        diagnostics["bilevel/bome_multiplier"] = float(multiplier.cpu())
    return weights.detach(), diagnostics


def _step_diagnostics(inner, outer, gradient, step):
    return {
        "bilevel/inner_loss": float(inner.detach().cpu()),
        "bilevel/outer_loss": float(outer.detach().cpu()),
        "bilevel/gradient_norm": float(gradient.detach().norm().cpu()),
        "bilevel/outer_step": step,
    }
