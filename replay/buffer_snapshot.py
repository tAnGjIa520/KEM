"""Portable, atomic replay-buffer snapshots for cross-method analysis."""

from __future__ import annotations

import copy
import csv
import json
import os
from collections import Counter
from pathlib import Path

import torch


SCHEMA_VERSION = "kem-replay-buffer-v1"


def _cpu_copy(value, fp16_features=False, key=None):
    if torch.is_tensor(value):
        result = value.detach().cpu().clone()
        if fp16_features and key == "features" and result.is_floating_point():
            result = result.half()
        return result
    if isinstance(value, dict):
        return {name: _cpu_copy(item, fp16_features, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item, fp16_features) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item, fp16_features) for item in value)
    return copy.deepcopy(value)


def _buffer_payload(buffer, fp16_features=False):
    common = {
        "buffer_size": int(buffer.buffer_size),
        "n_patches_total": int(buffer.n_patches_total),
        "labels": {int(key): int(value) for key, value in buffer.labels.items()},
    }
    if hasattr(buffer, "last_plan"):
        common.update({
            "buffer_type": "global_patch_budget",
            "buffers": _cpu_copy(buffer.buffers, fp16_features),
            "buffer_size_per_class": None,
            "num_classes": int(buffer.num_classes),
            "n_seen_samples_per_class": dict(buffer.n_seen_samples_per_class),
            "ratio": float(buffer.ratio),
            "floor": int(buffer.floor),
            "tolerance": float(buffer.tolerance),
            "last_plan": copy.deepcopy(buffer.last_plan),
            "plan_history": copy.deepcopy(buffer.plan_history),
        })
    elif hasattr(buffer, "buffers"):
        common.update({
            "buffer_type": "class_incremental",
            "buffers": _cpu_copy(buffer.buffers, fp16_features),
            "buffer_size_per_class": buffer.buffer_size_per_class,
            "num_classes": int(buffer.num_classes),
            "n_seen_samples_per_class": dict(buffer.n_seen_samples_per_class),
        })
    else:
        common.update({
            "buffer_type": "simple",
            "buffer": _cpu_copy(buffer.buffer, fp16_features),
            "n_seen_samples": int(buffer.n_seen_samples),
        })
    return common


def _samples(payload):
    if payload["buffer_type"] in {"class_incremental", "global_patch_budget"}:
        for label, samples in payload["buffers"].items():
            for sample in samples:
                yield int(label), sample
    else:
        for sample in payload["buffer"]:
            yield int(sample["label"].item()), sample


def _scalar(sample, name, default=-1):
    value = sample.get(name, default)
    if torch.is_tensor(value):
        return value.reshape(-1)[0].item() if value.numel() else default
    return value


def _slide_id(sample, index):
    for name in ("slide_id", "slide", "case_id"):
        if name in sample:
            value = sample[name]
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            return str(value)
    return f"buffer_item_{index:04d}"


def _atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_text(text, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _feature_rank_device():
    """Use the job's visible CUDA device for the expensive exact SVD."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def feature_spectrum_metrics(features, device=None):
    """Compute exact raw/centered feature-rank metrics in float64.

    Formal jobs are CUDA-only, so their SVD runs on the worker's single visible
    GPU.  Returned spectra remain CPU tensors to keep snapshots portable.
    """
    device = _feature_rank_device() if device is None else torch.device(device)
    matrix = features.detach().to(device=device, dtype=torch.float64)
    result, spectra = {}, {}
    for name, value in (("raw", matrix), ("centered", matrix - matrix.mean(dim=0, keepdim=True))):
        singular = torch.linalg.svdvals(value)
        spectra[name] = singular.detach().cpu()
        maximum = float(singular[0]) if singular.numel() else 0.0
        tolerance = max(value.shape) * torch.finfo(torch.float64).eps * maximum
        rank = int((singular > tolerance).sum())
        squares = singular.square()
        energy = float(squares.sum())
        stable_rank = energy / maximum**2 if maximum > 0 else 0.0
        singular_sum = float(singular.sum())
        if singular_sum > 0:
            probabilities = singular / singular_sum
            effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-300).log()).sum()))
        else:
            effective_rank = 0.0
        fourth = float(squares.square().sum())
        participation = energy**2 / fourth if fourth > 0 else 0.0
        cumulative = torch.cumsum(squares, dim=0) / max(energy, 1e-300)
        def energy_at(k):
            return float(cumulative[min(k, cumulative.numel())-1]) if cumulative.numel() else 0.0
        def k_for(fraction):
            threshold = torch.tensor(
                fraction, dtype=cumulative.dtype, device=cumulative.device,
            )
            return int(torch.searchsorted(cumulative, threshold).item()+1) if cumulative.numel() else 0
        minimum = float(singular[-1]) if singular.numel() else 0.0
        condition = maximum / minimum if minimum > tolerance else float("inf")
        prefix = f"{name}_"
        result.update({
            prefix+"rank": rank, prefix+"rank_ratio": rank/max(min(value.shape), 1),
            prefix+"rank_tolerance": tolerance, prefix+"stable_rank": stable_rank,
            prefix+"effective_rank": effective_rank, prefix+"participation_ratio": participation,
            prefix+"energy_top1": energy_at(1), prefix+"energy_top5": energy_at(5),
            prefix+"energy_top10": energy_at(10), prefix+"k90": k_for(0.90), prefix+"k95": k_for(0.95),
            prefix+"condition_number": condition,
        })
    return result, spectra


def save_buffer_snapshot(buffer, root, exp_name, fold, task, config, fp16_features=False,
                         save_full_buffer=False):
    directory = Path(root) / exp_name / f"fold_{fold}" / f"task_{task}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": exp_name,
        "fold": int(fold),
        "task": int(task),
        "config": dict(config),
        "buffer": _buffer_payload(buffer, fp16_features),
    }
    snapshot_path = directory / "buffer.pt"
    if save_full_buffer:
        _atomic_torch_save(payload, snapshot_path)
    rows, class_counts, task_counts = [], Counter(), Counter()
    rank_device = _feature_rank_device()
    rank_rows, spectrum_records = [], []
    for index, (label, sample) in enumerate(_samples(payload["buffer"])):
        features = sample["features"]
        source_task = int(_scalar(sample, "source_task", -1))
        logits = sample.get("logits")
        selected = sample.get("selected_indices")
        pseudo_bag_ids = sample.get("pseudo_bag_ids")
        rows.append({
            "item_index": index, "slide_id": _slide_id(sample, index), "label": label,
            "source_task": source_task, "selection_method": str(sample.get("selection_method", "unknown")),
            "original_patch_count": int(_scalar(sample, "original_patch_count", features.shape[0])),
            "selected_patch_count": int(features.shape[0]), "feature_shape": "x".join(map(str, features.shape)),
            "feature_dtype": str(features.dtype), "logit_dim": int(logits.shape[-1]) if torch.is_tensor(logits) else 0,
            "stored_class_count": int(sample.get("stored_classes", torch.empty(0)).numel()) if torch.is_tensor(sample.get("stored_classes")) else 0,
            "selected_index_count": int(selected.numel()) if torch.is_tensor(selected) else 0,
            "pseudo_bag_count": int(torch.unique(pseudo_bag_ids).numel())
            if torch.is_tensor(pseudo_bag_ids) else 0,
            "requested_ratio": float(_scalar(sample, "ratio_requested", float("nan"))),
            "actual_ratio": float(_scalar(sample, "ratio_actual", features.shape[0] / max(int(_scalar(sample, "original_patch_count", features.shape[0])), 1))),
            "ratio_floor_hit": int(_scalar(sample, "ratio_floor_hit", 0)),
            "ratio_small_bag": int(_scalar(sample, "ratio_small_bag", 0)),
        })
        class_counts[label] += 1; task_counts[source_task] += 1
        rank_metrics, spectra = feature_spectrum_metrics(features, rank_device)
        rank_rows.append({
            "item_index": index, "slide_id": _slide_id(sample, index), "label": label,
            "source_task": source_task, "method": exp_name, "n_patches": int(features.shape[0]),
            "feature_dim": int(features.shape[1]), **rank_metrics,
        })
        spectrum_records.append({
            "item_index": index, "slide_id": _slide_id(sample, index), "label": label,
            "source_task": source_task,
            "selected_indices": _cpu_copy(sample.get("selected_indices", torch.empty(0, dtype=torch.long))),
            "raw_singular_values": spectra["raw"], "centered_singular_values": spectra["centered"],
        })
    manifest_path = directory / "manifest.csv"
    fields = list(rows[0]) if rows else ["item_index", "slide_id", "label", "source_task"]
    temporary = manifest_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, manifest_path)
    rank_path = directory / "per_bag_feature_rank.csv"
    rank_fields = list(rank_rows[0]) if rank_rows else ["item_index", "slide_id", "label", "source_task"]
    temporary = rank_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rank_fields); writer.writeheader(); writer.writerows(rank_rows)
    os.replace(temporary, rank_path)
    _atomic_torch_save({"schema_version": SCHEMA_VERSION, "records": spectrum_records}, directory / "spectra.pt")
    summary = {
        "schema_version": SCHEMA_VERSION, "experiment": exp_name, "fold": int(fold), "task": int(task),
        "total_slides": len(rows), "total_patches": sum(row["selected_patch_count"] for row in rows),
        "per_class_slides": dict(sorted(class_counts.items())), "per_source_task_slides": dict(sorted(task_counts.items())),
        "snapshot_bytes": snapshot_path.stat().st_size if snapshot_path.exists() else 0,
        "spectra_bytes": (directory / "spectra.pt").stat().st_size,
        "fp16_features": bool(fp16_features), "full_buffer_saved": bool(save_full_buffer),
        "rank_device": str(rank_device),
        "seed": config.get("seed"),
        "selection_method": config.get("selection_method", config.get("distill_method")),
        "budget_plan": copy.deepcopy(payload["buffer"].get("last_plan", {})),
    }
    _atomic_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", directory / "summary.json")
    return directory


def load_buffer_snapshot(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_buffer_snapshot(payload)
    return payload


def validate_buffer_snapshot(payload):
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported buffer snapshot schema")
    buffer = payload.get("buffer", {})
    if buffer.get("buffer_type") not in {"simple", "class_incremental", "global_patch_budget"}:
        raise ValueError("invalid buffer type")
    samples = list(_samples(buffer))
    if len(samples) != sum(buffer.get("labels", {}).values()):
        raise ValueError("buffer label counts do not match samples")
    patches = sum(int(sample["features"].shape[0]) for _, sample in samples)
    if patches != int(buffer.get("n_patches_total", -1)):
        raise ValueError("buffer patch count does not match samples")
    return True
