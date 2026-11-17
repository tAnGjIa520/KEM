"""Shared continual replay distillation losses.

Keeping the coefficients here prevents replay selection and replay training
from silently optimizing different objectives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_kl(student, teacher, temperature=2.0):
    return F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction="batchmean",
    )


def replay_distillation_loss(model_name, student, teacher):
    zero = student["logits"].new_zeros(())
    logits_loss = 10.0 * distillation_kl(student["logits"], teacher["logits"], 2.0)
    attention_loss = zero
    hidden_loss = zero
    if model_name.startswith("clam_"):
        if "A" in student and "A" in teacher:
            attention_loss = distillation_kl(student["A"], teacher["A"], 2.0)
    elif model_name == "transmil":
        for key in ("attn1", "attn2"):
            if key in student and key in teacher and student[key].shape == teacher[key].shape:
                attention_loss = attention_loss + 10.0 * distillation_kl(
                    student[key], teacher[key], 1.0
                )
        for key in ("h1", "h2"):
            if key in student and key in teacher:
                hidden_loss = hidden_loss + 10.0 * F.mse_loss(student[key], teacher[key])
    elif model_name == "acmil":
        if "A" in student and "A" in teacher:
            attention_loss = distillation_kl(student["A"], teacher["A"], 2.0)
        if "token_logits" in student and "token_logits" in teacher:
            hidden_loss = distillation_kl(
                student["token_logits"], teacher["token_logits"], 2.0
            )
    total = logits_loss + attention_loss + hidden_loss
    return total, {
        "logits_loss": logits_loss,
        "attention_loss": attention_loss,
        "hidden_loss": hidden_loss,
    }
