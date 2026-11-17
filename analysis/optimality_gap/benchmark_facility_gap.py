#!/usr/bin/env python3
"""Exact small-instance audit of the production facility allocator.

All methods share a fixed, class-conditioned, multiple-choice patch-cap
ground set taken from archived real-WSI options. No test labels/accuracy
are used. This is a static optimization audit, not a full-scale CL trial.
"""
import argparse
import csv
from dataclasses import replace
import hashlib
import heapq
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from replay.unified_gradient_coverage import UnifiedGradientCoverageBuffer, _view_similarity


def ground_set(candidates):
    n = len(candidates)
    labels = sorted({c.label for c in candidates})
    weights = np.array([1 / (len(labels) * sum(x.label == c.label for x in candidates))
                        for c in candidates])
    refs = torch.stack([c.reference_views["frozen_feature"] for c in candidates])
    similarity = np.zeros((n, 4, n), dtype=np.float64)
    costs = np.array([0, 64, 128, 256])
    for i, candidate in enumerate(candidates):
        for j, option in enumerate(candidate.options, 1):
            values = _view_similarity(refs, option.views["frozen_feature"], "frozen_feature").numpy()
            similarity[i, j] = values * np.array([c.label == candidate.label for c in candidates])
    return similarity, costs, weights


def objective(state, similarity, weights):
    return float(similarity[np.arange(len(state)), state].max(axis=0) @ weights)


def exact_states(similarity, costs, weights):
    states = np.array(list(itertools.product(range(4), repeat=len(similarity))), dtype=np.int8)
    totals = costs[states].sum(axis=1)
    values = np.empty(len(states))
    for start in range(0, len(states), 4096):
        part = states[start:start + 4096]
        coverage = similarity[np.arange(len(similarity))[None, :], part].max(axis=1)
        values[start:start + len(part)] = coverage @ weights
    return states, totals, values


def lazy_density(similarity, costs, weights, budget, initial=None):
    n = len(similarity)
    state = np.zeros(n, dtype=np.int8) if initial is None else initial.copy()
    used = int(costs[state].sum())
    current = objective(state, similarity, weights)
    coverage = similarity[np.arange(n), state].max(axis=0)
    heap = []
    for i in range(n):
        if state[i]:
            continue
        for j in range(1, 4):
            gain = float((np.maximum(coverage, similarity[i, j]) - coverage) @ weights)
            heapq.heappush(heap, (-gain / costs[j], i, j, -1))
    iteration = 0
    while heap:
        neg_bound, i, j, evaluated = heapq.heappop(heap)
        if state[i] or used + costs[j] > budget:
            continue
        gain = float((np.maximum(coverage, similarity[i, j]) - coverage) @ weights)
        density = gain / costs[j]
        if evaluated != iteration:
            heapq.heappush(heap, (-density, i, j, iteration))
            continue
        if gain <= 1e-12:
            break
        state[i] = j
        used += int(costs[j])
        coverage = np.maximum(coverage, similarity[i, j])
        current = float(coverage @ weights)
        iteration += 1
    return state, current


def singleton_enumeration(similarity, costs, weights, budget):
    best_state, best_value = lazy_density(similarity, costs, weights, budget)
    for i in range(len(similarity)):
        for j in range(1, 4):
            if costs[j] > budget:
                continue
            state = np.zeros(len(similarity), dtype=np.int8)
            state[i] = j
            result, value = lazy_density(similarity, costs, weights, budget, state)
            if value > best_value:
                best_state, best_value = result, value
    return best_state, best_value


def choose_cache(cache_root, fold):
    for path in sorted(cache_root.glob(f"fold{fold}_task0_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        fields = payload.get("cache_fields", {})
        expected = {"fold": fold, "seed": 1, "net": "clam_sb", "patch_budget": 5000,
                    "trajectory_mode": "a", "signature_mode": "mean_std",
                    "candidate_pool": "maxminrand", "fixed_depth64": False}
        if all(fields.get(k) == v for k, v in expected.items()):
            return path, payload
    raise RuntimeError(f"no matching real candidate archive for fold {fold}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = args.run_root / "reports/FACILITY_GAP"
    output.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    budgets = [256, 384, 512, 768]
    for fold in range(5):
        path, payload = choose_cache(args.cache_root, fold)
        sources.append({"fold": fold, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "cache_fields": payload["cache_fields"]})
        pool = []
        for candidate in payload["candidates"]:
            options = tuple(sorted([o for o in candidate.options
                                    if o.trajectory == "a" and o.patch_count in (64, 128, 256)
                                    and "exact_closure_only" not in o.views], key=lambda o: o.patch_count))
            if [o.patch_count for o in options] == [64, 128, 256]:
                pool.append(replace(candidate, options=options))
        labels = sorted({c.label for c in pool})
        if len(labels) != 2:
            raise RuntimeError("task-0 benchmark expects two classes")
        for subset_seed in range(3):
            rng = np.random.default_rng(20260912 + fold * 100 + subset_seed)
            selected = []
            for label in labels:
                group = sorted([c for c in pool if c.label == label], key=lambda c: c.slide_id)
                selected.extend(group[i] for i in rng.choice(len(group), 4, replace=False))
            similarity, costs, weights = ground_set(selected)
            start = time.perf_counter()
            states, totals, values = exact_states(similarity, costs, weights)
            exact_seconds = time.perf_counter() - start
            for budget in budgets:
                best_index = np.where(totals <= budget)[0][np.argmax(values[totals <= budget])]
                optimum = float(values[best_index])
                results = [("exact_enumeration", states[best_index], optimum, exact_seconds)]
                buffer = UnifiedGradientCoverageBuffer(budget, 1, fold, levels=(64, 128, 256),
                            n_classes=11, projection_dim=64, objective="frozen_feature_coverage",
                            trajectory_mode="a", class_conditioned=True)
                start = time.perf_counter()
                chosen, used, recorded_value, _, _ = buffer._greedy(selected, budget)
                state = np.array([next((j for j,o in enumerate(c.options,1)
                                  if c.slide_id in chosen and o.patch_count == chosen[c.slide_id].patch_count),0)
                                  for c in selected], dtype=np.int8)
                runtime = time.perf_counter() - start
                value = objective(state, similarity, weights)
                assert abs(value-recorded_value) < 2e-6
                results.append(("production_density_upgrade", state, value, runtime))
                for name, algorithm in [("lazy_density_partition", lazy_density),
                                        ("singleton_enumeration_lazy", singleton_enumeration)]:
                    start = time.perf_counter()
                    state, value = algorithm(similarity, costs, weights, budget)
                    results.append((name, state, value, time.perf_counter()-start))
                for name, state, value, runtime in results:
                    assert costs[state].sum() <= budget and value <= optimum + 2e-6
                    rows.append({"fold":fold,"subset_seed":subset_seed,"budget":budget,"solver":name,
                                 "objective":value,"exact_objective":optimum,
                                 "absolute_gap":max(0.,optimum-value),
                                 "relative_gap":max(0.,optimum-value)/optimum,
                                 "runtime_seconds":runtime,"used_patches":int(costs[state].sum()),
                                 "state":json.dumps(state.tolist()),
                                 "slides":json.dumps([c.slide_id for c in selected])})
            print("GAP_INSTANCE_COMPLETE",fold,subset_seed,flush=True)
    with (output / "per_instance.csv").open("w") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import pandas as pd
    frame=pd.DataFrame(rows)
    frame.groupby("solver").agg(instances=("objective","size"),mean_relative_gap=("relative_gap","mean"),
        max_relative_gap=("relative_gap","max"),median_runtime_seconds=("runtime_seconds","median")).to_csv(output/"summary.csv")
    (output/"manifest.json").write_text(json.dumps({"sources":sources,"budgets":budgets,"rows":len(rows),
        "n_slides":8,"options_per_slide":3,"subsets_per_fold":3,"backend":"CPU, 2 torch threads",
        "runtime_note":"Exact exhaustive enumeration time is shared by all four budget queries. Reference solvers use precomputed similarities; production allocator builds its internal similarity cache. Runtime is implementation-specific.",
        "scope":"Static restricted ground set, <=budget, class-balanced facility coverage; no exact-closure interpolation, no online historical discard, no AACC claim."},indent=2))
    (output/"COMPLETE").touch()


if __name__ == "__main__":
    main()
