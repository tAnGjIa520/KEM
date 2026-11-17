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
            by_id = {c.slide_id: c for c in pool}
            selected = [by_id[s] for s in fixed_subsets[fold, subset_seed]]
            assert len(selected) == 8 and all(sum(c.label == label for c in selected) == 4 for label in labels)
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
