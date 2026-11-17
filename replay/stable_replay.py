"""Stable replay targets and balanced classifier calibration."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .model_registry import get_training_loss


def _kd_loss(student, teacher, temperature=2.0):
    return F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    )


def stored_logit_loss(output_logits, batch, fallback_classes, weight=1.0):
    """Distill only classes that existed when the replay item was stored."""
    classes = batch.get("stored_classes")
    if classes is None:
        classes = torch.as_tensor(fallback_classes, device=output_logits.device)
    classes = classes.to(output_logits.device).long().reshape(-1)
    stored = batch["logits"].to(output_logits.device)
    return float(weight) * _kd_loss(
        output_logits.index_select(1, classes), stored.index_select(1, classes)
    )


def _buffer_samples_by_class(buffer):
    if hasattr(buffer, "buffers"):
        return {int(label): list(samples) for label, samples in buffer.buffers.items()}
    grouped = defaultdict(list)
    for sample in buffer.buffer:
        grouped[int(sample["label"].item())].append(sample)
    return dict(grouped)


def _classification_head(model, net):
    if net in ["clam_sb", "clam_mb"]:
        return model.classifiers
    if net == "transmil":
        return model._fc2
    if net in ["mambamil", "chief_cached"]:
        return model.classifier
    if net == "wikg":
        return model.fc
    if net == "acmil":
        return torch.nn.ModuleList([model.Slide_classifier, *list(model.classifier)])
    raise NotImplementedError(net)


def _head_weights(model, net):
    if net == "acmil":
        return [model.Slide_classifier.fc.weight] + [head.fc.weight for head in model.classifier]
    return [_classification_head(model, net).weight]


def calibrate_classification_head(model, buffer, args, fabric, seen_classes, task):
    """Class-balanced task-end calibration that updates only the classifier."""
    if not args.head_calibration or task == 0 or len(buffer) == 0:
        return {"head_calibration/steps": 0.0}
    head = _classification_head(model, args.net)
    original_requires_grad = {name: value.requires_grad for name, value in model.named_parameters()}
    for value in model.parameters():
        value.requires_grad = False
    for value in head.parameters():
        value.requires_grad = True
    optimizer = torch.optim.Adam(head.parameters(), lr=args.head_calibration_lr)
    grouped = _buffer_samples_by_class(buffer)
    rng = random.Random(args.seed + 1009 * task)
    losses = []
    model.train()
    for _ in range(args.head_calibration_epochs):
        max_count = max(len(samples) for samples in grouped.values())
        schedule = []
        for offset in range(max_count):
            labels = list(grouped)
            rng.shuffle(labels)
            for label in labels:
                samples = grouped[label]
                schedule.append(samples[offset % len(samples)])
        for sample in schedule:
            sample = fabric.to_device(sample)
            optimizer.zero_grad()
            if args.net in ["clam_sb", "clam_mb"]:
                output = model(
                    sample["features"], sample["label"], instance_eval=False,
                    seen_classes=seen_classes,
                )
                loss = F.cross_entropy(output["logits"], sample["label"])
            elif args.net == "acmil":
                output = model(sample["features"], seen_classes=seen_classes)
                loss, _ = get_training_loss("acmil").compute(output, sample["label"])
            else:
                output = model(sample["features"])
                loss = F.cross_entropy(output["logits"], sample["label"])
            fabric.backward(loss)
            optimizer.step()
            if args.head_calibration_weight_norm:
                with torch.no_grad():
                    for weight in _head_weights(model, args.net):
                        weight.copy_(F.normalize(weight, dim=1))
            losses.append(float(loss.detach()))
    for name, value in model.named_parameters():
        value.requires_grad = original_requires_grad[name]
    model.eval()
    return {
        "head_calibration/steps": float(len(losses)),
        "head_calibration/loss": float(np.mean(losses)) if losses else 0.0,
        "head_calibration/weight_norm": float(torch.stack([
            weight.detach().norm().square() for weight in _head_weights(model, args.net)
        ]).sum().sqrt()),
    }
