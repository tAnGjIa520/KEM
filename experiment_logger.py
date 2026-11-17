"""Fault-tolerant local experiment logging for KEM.

TensorBoard and Weights & Biases are optional sinks.  W&B defaults to offline
and is never logged in or synchronised by this module.
"""

from __future__ import annotations

import atexit
import math
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


def _finite(value: Any) -> float | None:
    try:
        value = float(value.detach().cpu() if torch.is_tensor(value) else value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def gradient_norm(model: torch.nn.Module) -> float:
    squares = [p.grad.detach().float().norm(2).square() for p in model.parameters()
               if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()]
    return float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0


_CLASSIFICATION_HEAD_NAMES = {
    "classifier", "classifiers", "instance_classifiers", "slide_classifier",
    "fc", "_fc2",
}


def classification_head_parameter_norms(model: torch.nn.Module) -> dict[str, float]:
    """Return L2 norms for recognized classification-head parameters.

    KEM's model families use several names for their output heads (for
    example ``classifiers`` in CLAM, ``_fc2`` in TransMIL, and ``fc`` in
    WiKG). Models without a recognized classification head are skipped.
    """
    parameters = []
    result = {}
    for name, parameter in model.named_parameters():
        path = {part.lower() for part in name.split(".")}
        if not path.intersection(_CLASSIFICATION_HEAD_NAMES):
            continue
        value = parameter.detach().float()
        if not value.numel() or not torch.isfinite(value).all():
            continue
        parameters.append(value.reshape(-1))
        result[f"parameters/classification_head/{name}_l2_norm"] = float(value.norm(2).cpu())
    if parameters:
        result["parameters/classification_head/l2_norm"] = float(
            torch.cat(parameters).norm(2).cpu()
        )
    return result


def _nanmean(values: Sequence[Any]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)) if array.size and not np.all(np.isnan(array)) else float("nan")


def continual_metrics(rows: Sequence[Mapping[str, Any]], stage: int) -> dict[str, float]:
    """Compute CL summaries from rows containing ``<task>_{acc,auroc,f1}``."""
    current = rows[stage]
    seen = range(stage + 1)
    result = {
        "AACC": _nanmean([current.get(f"{j}_acc", np.nan) for j in seen]),
        "mAUROC": _nanmean([current.get(f"{j}_auroc", np.nan) for j in seen]),
        "mF1": _nanmean([current.get(f"{j}_f1", np.nan) for j in seen]),
    }
    forgetting = []
    backward = []
    for j in seen:
        now = float(current.get(f"{j}_acc", np.nan))
        learned = float(rows[j].get(f"{j}_acc", np.nan))
        history = [float(rows[k].get(f"{j}_acc", np.nan)) for k in range(j, stage + 1)]
        value = 0.0 if stage == 0 else float(np.nanmax(history) - now)
        result[f"forgetting/task_{j}"] = value
        forgetting.append(value)
        if j < stage:
            backward.append(now - learned)
    result["BWT"] = float(np.nanmean(backward)) if backward else 0.0
    result["average_forgetting"] = float(np.nanmean(forgetting)) if forgetting else 0.0
    return result


class ExperimentLogger:
    """One fold-level TensorBoard/W&B run with a task-spanning global step."""

    def __init__(self, args: Any, fold: int, root: str | Path = "logs"):
        self.args, self.fold, self.global_step = args, fold, 0
        self.exp_name = str(getattr(args, "exp_name", "kem"))
        self.root = Path(root) / self.exp_name
        self.tb_dir = self.root / "tensorboard" / f"fold_{fold}"
        self.wandb_dir = self.root / "wandb"
        self.writer = self.run = None
        self._closed = False
        atexit.register(self.close, "failed")
        if bool(getattr(args, "tensorboard", True)):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.tb_dir))
            except Exception as exc:  # logging must never stop training
                warnings.warn(f"TensorBoard disabled after initialization failure: {exc}")
        mode = str(getattr(args, "wandb_mode", "offline")).lower()
        if mode not in {"offline", "online", "disabled"}:
            warnings.warn(f"Unknown wandb_mode={mode!r}; forcing offline")
            mode = "offline"
        if mode != "disabled":
            try:
                self.wandb_dir.mkdir(parents=True, exist_ok=True)
                os.environ["WANDB_MODE"] = mode
                os.environ["WANDB_DIR"] = str(self.wandb_dir.resolve())
                import wandb
                config = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, tuple, type(None)))}
                self.run = wandb.init(project=getattr(args, "project", "KEM"),
                                      group=self.exp_name, name=f"{self.exp_name}-fold-{fold}",
                                      config=config, mode=mode, dir=str(self.wandb_dir), reinit=True)
            except Exception as exc:
                warnings.warn(f"W&B disabled after initialization failure: {exc}")
                self.run = None

    def _safe(self, operation, label: str) -> None:
        try:
            operation()
        except Exception as exc:
            warnings.warn(f"Observability {label} failed (training continues): {exc}")

    def log_scalars(self, values: Mapping[str, Any], step: int | None = None) -> None:
        clean = {str(k): v for k, raw in values.items() if (v := _finite(raw)) is not None}
        if not clean:
            return
        step = self.global_step if step is None else int(step)
        if self.writer:
            self._safe(lambda: [self.writer.add_scalar(k, v, step) for k, v in clean.items()], "TensorBoard scalar write")
        if self.run:
            self._safe(lambda: self.run.log(clean, step=step), "W&B scalar write")

    def log_train_step(self, metrics: Mapping[str, Any], model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer, task: int, epoch: int,
                       batch: int, replay: bool = False) -> None:
        prefix = "train/replay" if replay else "train/current"
        aliases = {"loss": "loss", "bag_loss": "bag_loss", "inst_loss": "instance_loss",
                   "token_loss": "token_loss", "slide_loss": "slide_loss",
                   "diversity_loss": "diversity_loss", "kd_loss": "kd_loss",
                   "logits_loss": "logits_loss", "feature_loss": "feature_loss",
                   "features_loss": "feature_loss", "em_loss": "feature_loss",
                   "attention_loss": "attention_loss", "attn_loss": "attention_loss"}
        payload = {f"{prefix}/{dst}": metrics[src] for src, dst in aliases.items() if src in metrics}
        if "loss" in metrics:
            payload["train/total_loss"] = metrics["loss"]
        payload.update({"train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/gradient_norm": gradient_norm(model),
                        "runtime/task": task, "runtime/epoch": epoch, "runtime/batch": batch})
        payload.update(classification_head_parameter_norms(model))
        if torch.cuda.is_available():
            payload["runtime/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
            payload["runtime/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        self.log_scalars(payload)
        self.global_step += 1

    def log_histograms(self, model: torch.nn.Module, task: int, epoch: int) -> None:
        if not bool(getattr(self.args, "log_histograms", True)):
            return
        interval = max(1, int(getattr(self.args, "histogram_every_n_epochs", 1)))
        if (epoch + 1) % interval:
            return
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter_data = parameter.detach().float().cpu()
            if parameter_data.numel() and torch.isfinite(parameter_data).all():
                self._histogram(f"parameters/{name}", parameter_data)
            if parameter.grad is not None:
                grad = parameter.grad.detach().float().cpu()
                if grad.numel() and torch.isfinite(grad).all():
                    self._histogram(f"gradients/{name}", grad)

    def _histogram(self, tag: str, tensor: torch.Tensor) -> None:
        if self.writer:
            self._safe(lambda: self.writer.add_histogram(tag, tensor, self.global_step), "TensorBoard histogram write")
        if self.run:
            self._safe(lambda: self.run.log({tag: __import__("wandb").Histogram(tensor.numpy())}, step=self.global_step), "W&B histogram write")

    def log_image(self, tag: str, figure: Any) -> None:
        if self.writer:
            self._safe(lambda: self.writer.add_figure(tag, figure, self.global_step, close=False), "TensorBoard image write")
        if self.run:
            self._safe(lambda: self.run.log({tag: __import__("wandb").Image(figure)}, step=self.global_step), "W&B image write")

    def log_continual(self, rows: Sequence[Mapping[str, Any]], stage: int,
                      confusion_matrices: Mapping[int, np.ndarray] | None = None) -> dict[str, float]:
        summary = continual_metrics(rows, stage)
        payload = {f"continual/{k}": v for k, v in summary.items()}
        for j in range(stage + 1):
            for metric in ("acc", "auroc", "f1"):
                payload[f"test/task_{j}/{metric}"] = rows[stage].get(f"{j}_{metric}", np.nan)
        self.log_scalars(payload)
        if bool(getattr(self.args, "log_cl_matrices", True)):
            self._plot_matrices(rows, stage)
        if confusion_matrices and bool(getattr(self.args, "log_confusion_matrices", True)):
            self._plot_confusions(confusion_matrices)
        return summary

    def log_buffer(self, buffer: Any) -> None:
        values = {"buffer/size": len(buffer), "buffer/patch_count": getattr(buffer, "n_patches_total", 0)}
        values.update({f"buffer/class_count_{label}": count for label, count in getattr(buffer, "labels", {}).items()})
        self.log_scalars(values)

    def _plot_matrices(self, rows: Sequence[Mapping[str, Any]], stage: int) -> None:
        import matplotlib.pyplot as plt
        for metric in ("acc", "auroc", "f1"):
            matrix = np.full((stage + 1, stage + 1), np.nan)
            for i in range(stage + 1):
                for j in range(i + 1):
                    matrix[i, j] = rows[i].get(f"{j}_{metric}", np.nan)
            fig, ax = plt.subplots(figsize=(4, 3)); image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
            ax.set(xlabel="Test task", ylabel="Learned task", title=metric.upper()); fig.colorbar(image, ax=ax); fig.tight_layout()
            self.log_image(f"continual/{metric}_matrix", fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5, 3))
        for j in range(stage + 1):
            xs, ys = [], []
            for i in range(j, stage + 1):
                best = max(rows[k].get(f"{j}_acc", np.nan) for k in range(j, i + 1))
                xs.append(i); ys.append(best - rows[i].get(f"{j}_acc", np.nan))
            ax.plot(xs, ys, marker="o", label=f"task {j}")
        ax.set(xlabel="Learned task", ylabel="Forgetting", title="Forgetting curves"); ax.legend(); fig.tight_layout()
        self.log_image("continual/forgetting_curves", fig); plt.close(fig)

    def _plot_confusions(self, matrices: Mapping[int, np.ndarray]) -> None:
        import matplotlib.pyplot as plt
        for task, matrix in matrices.items():
            fig, ax = plt.subplots(figsize=(3, 3)); image = ax.imshow(matrix, cmap="Blues")
            ax.set(xlabel="Predicted", ylabel="True", title=f"Task {task}"); fig.colorbar(image, ax=ax); fig.tight_layout()
            self.log_image(f"confusion/task_{task}", fig); plt.close(fig)

    def close(self, status: str = "success") -> None:
        if self._closed:
            return
        self._closed = True
        if self.writer:
            self._safe(lambda: (self.writer.flush(), self.writer.close()), "TensorBoard close")
        if self.run:
            self._safe(lambda: self.run.finish(exit_code=0 if status == "success" else 1), "W&B close")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close("failed" if exc_type else "success")
        return False
