"""Region construction and frozen-teacher supervision for KEM.

The implementation deliberately keeps region pseudo labels outside the model:
they supervise only the differentiable replay selector's virtual update.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from .model_registry import get_replay_adapter
from .mambamil_adapter import canonicalize_region_order, region_ordering_hash


@dataclass(frozen=True)
class RegionConfig:
    enabled: bool = False
    partition: str = "spatial_balanced_100"
    target_patches: int = 100
    fixed_count: int = 10
    teacher_temperature: float = 2.0
    confidence_power: float = 2.0
    loss_lambda: float = 0.5
    min_size: int = 16
    cache_dir: str = "runs/kem_region_teacher_cache"
    seed: int = 1

    @classmethod
    def from_args(cls, args):
        return cls(
            enabled=bool(getattr(args, "region_supervision", False)),
            partition=str(getattr(args, "region_partition", "spatial_balanced_100")),
            target_patches=int(getattr(args, "region_target_patches", 100)),
            fixed_count=int(getattr(args, "region_fixed_count", 10)),
            teacher_temperature=float(getattr(args, "region_teacher_temperature", 2.0)),
            confidence_power=float(getattr(args, "region_confidence_power", 2.0)),
            loss_lambda=float(getattr(args, "region_loss_lambda", 0.5)),
            min_size=int(getattr(args, "region_min_size", 16)),
            cache_dir=str(getattr(args, "region_cache_dir", "runs/kem_region_teacher_cache")),
            seed=int(getattr(args, "seed", 1)),
        )

    def __post_init__(self):
        allowed = {"spatial_balanced_100", "spatial_fixed_10", "feature_balanced_100", "random_balanced_100"}
        if self.partition not in allowed:
            raise ValueError(f"unknown region partition: {self.partition}")
        if min(self.target_patches, self.fixed_count, self.min_size) < 1:
            raise ValueError("region size/count settings must be positive")
        if self.teacher_temperature <= 0 or self.confidence_power <= 0 or self.loss_lambda < 0:
            raise ValueError("invalid region supervision hyperparameters")


def _region_count(n: int, config: RegionConfig) -> int:
    if n <= config.min_size:
        return 1
    if config.partition == "spatial_fixed_10":
        wanted = min(config.fixed_count, n)
    else:
        wanted = max(1, math.ceil(n / config.target_patches))
    # Never create tiny tail regions.
    return max(1, min(wanted, n // config.min_size))


def _balanced_sizes(n: int, k: int) -> list[int]:
    return [n // k + (i < n % k) for i in range(k)]


def _spatial_partition(coords: torch.Tensor, k: int) -> list[torch.Tensor]:
    def split(indices, count):
        if count == 1:
            return [indices]
        xy = coords.index_select(0, indices).float()
        axis = int(torch.var(xy, dim=0, unbiased=False).argmax())
        order = torch.argsort(xy[:, axis], stable=True)
        left_count = count // 2
        left_size = round(indices.numel() * left_count / count)
        return split(indices[order[:left_size]], left_count) + split(
            indices[order[left_size:]], count - left_count
        )

    groups = split(torch.arange(coords.shape[0], dtype=torch.long), k)
    return sorted(groups, key=lambda x: int(x.min()))


def _random_partition(n: int, k: int, seed: int) -> list[torch.Tensor]:
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    sizes = _balanced_sizes(n, k)
    result, offset = [], 0
    for size in sizes:
        result.append(order[offset:offset + size].sort().values)
        offset += size
    return result


def _feature_partition(features: torch.Tensor, k: int, seed: int) -> list[torch.Tensor]:
    x = features.detach().cpu().float()
    if x.shape[1] > 64:
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(x.shape[1], 64, generator=generator) / 8.0
        x = x @ projection
    x = F.normalize(x, dim=1)
    # Deterministic farthest-point initialisation.
    centers = [0]
    nearest = 1 - x @ x[0]
    for _ in range(1, k):
        index = int(nearest.argmax())
        centers.append(index)
        nearest = torch.minimum(nearest, 1 - x @ x[index])
    prototypes = x[centers]
    capacities = _balanced_sizes(x.shape[0], k)
    assignments = torch.full((x.shape[0],), -1, dtype=torch.long)
    # Process confident points first and take their best cluster with capacity.
    # This avoids sorting the full N*K pair matrix for giga-pixel bags.
    similarity = x @ prototypes.T
    preferences = torch.argsort(similarity, dim=1, descending=True, stable=True)
    point_order = torch.argsort(similarity.max(dim=1).values, descending=True, stable=True)
    remaining = capacities[:]
    for point in point_order.tolist():
        for cluster in preferences[point].tolist():
            if remaining[cluster] > 0:
                assignments[point] = cluster
                remaining[cluster] -= 1
                break
    return [torch.nonzero(assignments == i, as_tuple=False).flatten() for i in range(k)]


def partition_regions(features: torch.Tensor, coords: torch.Tensor | None, config: RegionConfig) -> list[torch.Tensor]:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("region features must be a non-empty [N,D] matrix")
    n, k = features.shape[0], _region_count(features.shape[0], config)
    if config.partition.startswith("spatial_"):
        if coords is None:
            raise ValueError("spatial region partition requires patch coordinates")
        coords = torch.as_tensor(coords)
        if coords.ndim != 2 or coords.shape != (n, 2):
            raise ValueError(f"coords shape {tuple(coords.shape)} does not match features {(n, 2)}")
        regions = _spatial_partition(coords, k)
    elif config.partition == "feature_balanced_100":
        regions = _feature_partition(features, k, config.seed)
    else:
        regions = _random_partition(n, k, config.seed)
    joined = torch.cat(regions)
    if joined.numel() != n or torch.unique(joined).numel() != n or joined.min() != 0 or joined.max() != n - 1:
        raise RuntimeError("region partition lost or duplicated patches")
    if min(region.numel() for region in regions) < config.min_size and n >= config.min_size:
        raise RuntimeError("region partition produced a region below region_min_size")
    return regions


def pad_regions(features: torch.Tensor, regions: list[torch.Tensor]):
    maximum = max(region.numel() for region in regions)
    padded = features.new_zeros((len(regions), maximum, features.shape[1]))
    mask = torch.zeros(len(regions), maximum, dtype=torch.bool, device=features.device)
    for row, indices in enumerate(regions):
        indices = indices.to(features.device)
        padded[row, :indices.numel()] = features.index_select(0, indices)
        mask[row, :indices.numel()] = True
    return padded, mask


class RegionBatchCLAMAdapter:
    """Vectorized CLAM-SB inference over padded region bags."""

    @staticmethod
    def forward(model, padded, mask, seen_classes, params=None):
        if padded.ndim != 3 or mask.shape != padded.shape[:2]:
            raise ValueError("expected padded [R,M,D] and mask [R,M]")
        if model.__class__.__name__ != "CLAM_SB":
            raise ValueError("region supervision currently supports CLAM_SB only")
        r, m, d = padded.shape
        flat = padded.reshape(r * m, d)
        if params is None:
            attention, hidden = model.attention_net(flat)
        else:
            attention_params = {name[len("attention_net."):]: value for name, value in params.items() if name.startswith("attention_net.")}
            attention, hidden = functional_call(model.attention_net, attention_params, (flat,))
        attention = attention.reshape(r, m, -1).transpose(1, 2)
        hidden = hidden.reshape(r, m, -1)
        neg = torch.finfo(attention.dtype).min
        attention = attention.masked_fill(~mask[:, None, :], neg)
        pooled = torch.bmm(F.softmax(attention, dim=-1), hidden).squeeze(1)
        if params is None:
            logits = model.classifiers(pooled)
        else:
            classifier_params = {name[len("classifiers."):]: value for name, value in params.items() if name.startswith("classifiers.")}
            logits = functional_call(model.classifiers, classifier_params, (pooled,))
        seen = torch.as_tensor(seen_classes, device=logits.device, dtype=torch.long)
        unseen = torch.ones(logits.shape[-1], device=logits.device, dtype=torch.bool)
        unseen[seen] = False
        return logits.masked_fill(unseen.unsqueeze(0), -100.0)


class RegionBatchTransMILAdapter:
    """Exact padded-region adapter using length-grouped microbatches.

    TransMIL builds a different square positional grid for each sequence
    length.  Cropping each region before forward avoids padded tokens changing
    PPEG/Nystrom geometry; the loop is therefore the correctness baseline.
    """

    @staticmethod
    def forward(model, padded, mask, seen_classes, params=None):
        if padded.ndim != 3 or mask.shape != padded.shape[:2]:
            raise ValueError("expected padded [R,M,D] and mask [R,M]")
        if model.__class__.__name__ != "TransMIL":
            raise ValueError("TransMIL region adapter requires a TransMIL model")
        lengths = []
        for row in range(padded.shape[0]):
            length = int(mask[row].sum())
            if length < 1 or not bool(mask[row, :length].all()) or bool(mask[row, length:].any()):
                raise ValueError("region mask must be a non-empty contiguous prefix")
            lengths.append(length)
        logits_by_row = [None] * padded.shape[0]
        for length in sorted(set(lengths)):
            rows = [row for row, value in enumerate(lengths) if value == length]
            features = padded[rows, :length]
            output = functional_call(model, params, (features,)) if params is not None else model(features)
            for local, row in enumerate(rows):
                logits_by_row[row] = output["logits"][local:local + 1]
        logits = torch.cat(logits_by_row, dim=0)
        seen = torch.as_tensor(seen_classes, device=logits.device, dtype=torch.long)
        unseen = torch.ones(logits.shape[-1], device=logits.device, dtype=torch.bool)
        unseen[seen] = False
        return logits.masked_fill(unseen.unsqueeze(0), -100.0)


def model_fingerprint(model) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, value in model.state_dict().items():
            digest.update(name.encode())
            tensor = value.detach().cpu().reshape(-1)
            digest.update(str(tuple(value.shape)).encode())
            if tensor.numel():
                sample = tensor[torch.linspace(0, tensor.numel() - 1, min(32, tensor.numel())).long()]
                digest.update(sample.numpy().tobytes())
    return digest.hexdigest()[:20]


class RegionSupervisor:
    def __init__(self, config: RegionConfig, args):
        self.config = config
        self.args = args
        self.cache_root = Path(config.cache_dir)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _h5_path(self, slide_id: str) -> Path:
        roots = getattr(self.args, "feature_roots", None)
        base = Path(getattr(self.args, "feature_root_base", ""))
        candidates = []
        if roots:
            candidates = [(Path(path) if Path(path).is_absolute() else base / path) / "h5_files" / f"{slide_id}.h5" for path in roots.values()]
        else:
            candidates = [Path(self.args.data_root) / self.args.feat_dir / "h5_files" / f"{slide_id}.h5"]
        found = [path for path in candidates if path.exists()]
        if len(found) != 1:
            raise FileNotFoundError(f"expected exactly one h5 file for {slide_id}, found {found}")
        return found[0]

    def load_and_validate_coords(self, slide_id: str, features: torch.Tensor) -> torch.Tensor:
        path = self._h5_path(slide_id)
        with h5py.File(path, "r") as handle:
            if "features" not in handle or "coords" not in handle:
                raise ValueError(f"{path} must contain features and coords")
            h5_features = torch.from_numpy(handle["features"][:features.shape[0]])
            coords = torch.from_numpy(handle["coords"][:features.shape[0]])
        if h5_features.shape != features.cpu().shape or coords.shape != (features.shape[0], 2):
            raise ValueError(f"h5/pt alignment shape mismatch for {slide_id}")
        if not torch.allclose(h5_features.float(), features.detach().cpu().float(), rtol=1e-4, atol=1e-5):
            raise ValueError(f"h5 feature order does not match pt feature order for {slide_id}")
        return coords

    def prepare(self, model, features, coords, seen_classes, slide_id, task, model_name=None):
        started = time.perf_counter()
        regions = partition_regions(features.detach().cpu(), coords, self.config)
        is_mambamil = model.__class__.__name__ == "MambaMIL"
        if is_mambamil:
            regions = canonicalize_region_order(regions)
            ordering_policy = "original_patch_index"
        else:
            ordering_policy = "partition_native"
        ordering_hash = region_ordering_hash(regions) if is_mambamil else None
        padded, mask = pad_regions(features, regions)
        fingerprint = model_fingerprint(model)
        key_payload = {
            "checkpoint": fingerprint, "slide": slide_id, "task": int(task),
            "model": model.__class__.__name__,
            "model_name": model_name or getattr(model, "replay_model_name", None),
            "seen_classes": list(map(int, seen_classes)), "config": asdict(self.config),
            "ordering_policy": ordering_policy, "ordering_hash": ordering_hash,
        }
        key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_root / f"{key}.pt"
        cache_hit = path.exists()
        if cache_hit:
            payload = torch.load(path, map_location=features.device, weights_only=False)
            probs, logits, confidence = payload["probs"], payload["logits"], payload["confidence"]
        else:
            with torch.no_grad():
                if model_name is not None and get_replay_adapter(model_name).region_forward is not None:
                    logits = get_replay_adapter(model_name).region_forward(
                        model, padded, mask, seen_classes
                    )
                elif model.__class__.__name__ == "TransMIL":
                    logits = RegionBatchTransMILAdapter.forward(model, padded, mask, seen_classes)
                elif is_mambamil:
                    region_forward = get_replay_adapter("mambamil").region_forward
                    if region_forward is None:
                        raise RuntimeError("MambaMIL replay adapter has no region forward hook")
                    logits = region_forward(model, padded, mask, seen_classes)
                else:
                    logits = RegionBatchCLAMAdapter.forward(model, padded, mask, seen_classes)
                active = logits[:, list(seen_classes)] / self.config.teacher_temperature
                probs = F.softmax(active, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                if probs.shape[1] == 1:
                    confidence = torch.ones_like(entropy)
                else:
                    confidence = (1 - entropy / math.log(probs.shape[1])).clamp(0, 1).pow(self.config.confidence_power)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            torch.save({"logits": logits.detach().cpu(), "probs": probs.detach().cpu(), "confidence": confidence.detach().cpu(), "key": key_payload}, temporary)
            temporary.replace(path)
            probs, logits, confidence = probs.detach(), logits.detach(), confidence.detach()
        membership = torch.empty(features.shape[0], dtype=torch.long)
        for index, region in enumerate(regions):
            membership[region] = index
        return {
            "padded": padded, "mask": mask, "regions": regions, "membership": membership,
            "coords": None if coords is None else torch.as_tensor(coords).cpu(),
            "teacher_logits": logits.to(features.device).detach(),
            "teacher_probs": probs.to(features.device).detach(),
            "confidence": confidence.to(features.device).detach(),
            "cache_hit": cache_hit, "cache_path": path,
            "prepare_seconds": time.perf_counter() - started,
            "ordering_policy": ordering_policy, "ordering_hash": ordering_hash,
        }


def weighted_region_kl(
    student_logits, teacher_probs, confidence, seen_classes, temperature,
    student_temperature=None,
):
    # CLAM's virtual student uses the distillation temperature.  Gradient-only
    # selectors have no virtual update, so they may use the natural student
    # temperature (1.0) against softened teacher targets to obtain a useful
    # pseudo-label gradient instead of the identically-zero self-KL gradient.
    student_temperature = temperature if student_temperature is None else student_temperature
    if student_temperature <= 0:
        raise ValueError("student_temperature must be positive")
    active = student_logits[:, list(seen_classes)] / student_temperature
    per_region = F.kl_div(F.log_softmax(active, dim=-1), teacher_probs, reduction="none").sum(dim=-1)
    weights = confidence.clamp_min(0)
    return (per_region * weights).sum() / weights.sum().clamp_min(1e-8)
